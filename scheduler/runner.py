# scheduler/runner.py

import json
import uuid
import os
import glob
from datetime import datetime
import time
from reddit.scraper import scrape_subreddits
from db.writer import insert_post, update_post_filter_scores, update_post_insight, mark_insight_processed, mark_posts_in_history
from db.reader import get_top_insights_from_today, get_posts_by_ids
from db.schema import create_tables
from gpt.filters import prepare_batch_payload as prepare_filter_batch, estimate_batch_cost as estimate_filter_cost
from gpt.insights import prepare_insight_batch, estimate_insight_cost
import openai
from gpt.batch_api import (
    generate_batch_payload, submit_batch_job, poll_batch_status,
    download_batch_results, download_batch_results_if_available,
    get_processed_custom_ids, add_estimated_batch_cost
)
from db.cleaner import clean_old_entries
from scheduler.cost_tracker import initialize_cost_tracking, can_process_batch
from config.config_loader import get_config
from utils.logger import setup_logger
from utils.helpers import ensure_directory_exists, sanitize_text

log = setup_logger()
config = get_config()

def submit_with_backoff(batch_items, model, generate_file_fn, label="filter") -> str | None:
    delay = 10
    max_retries = 20
    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"[Retry {attempt}/{max_retries}] Submitting {label} batch with {len(batch_items)} items...")
            file_path = generate_file_fn(batch_items, model)
            batch_id = submit_batch_job(file_path)
            batch_info = poll_batch_status(batch_id)
            status = batch_info["status"]

            if status == "completed":
                return batch_id
            elif status == "cancelled":
                log.warning(f"{label.capitalize()} batch {batch_id} was cancelled. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
                if delay > 3600:
                    delay = 3600  # cap delay to 1 hour
                continue
            elif status == "failed":
                log.warning(f"{label.capitalize()} batch failed. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
                if delay > 3600:
                    delay = 3600  # cap delay to 1 hour
                continue
        except Exception as e:
            log.error(f"Error in {label} batch retry #{attempt}: {str(e)}")
            time.sleep(delay)
            delay *= 2
            if delay > 3600:
                    delay = 3600  # cap delay to 1 hour

    # All retries failed
    log.error(f"❌ {label.capitalize()} batch failed after {max_retries} retries. Deferring.")
    save_failed_batch(batch_items, label)
    return None

def _estimate_batch_tokens(batch_items):
    """Sum the estimated tokens for a batch from item metadata."""
    return sum(item.get("meta", {}).get("estimated_tokens", 300) for item in batch_items)



def submit_batches_parallel(all_sub_batches, model, generate_file_fn, label,
                            max_retries=3, enqueued_token_limit=5_000_000):
    """Submit batches with token-aware scheduling and parallel polling.

    Flow:
    1. Estimate tokens for each sub-batch upfront.
    2. Submit all batches that fit under the enqueued token limit without blocking.
    3. Poll all in-flight batches in parallel until they complete or fail.
    4. When capacity frees up, submit more batches.
    5. Repeat until all batches are processed.
    """
    result_paths = []
    pending = {}  # batch_id -> {"items": [...], "retries": int, "tokens": int}
    enqueued_tokens = 0  # tokens currently in-flight at OpenAI

    # Queue holds (items, retries) tuples; tokens are estimated on the fly
    submit_queue = [(batch_items, 0) for batch_items in all_sub_batches]

    while submit_queue or pending:
        # --- Submit phase: fire off all batches that fit under the token limit ---
        while submit_queue:
            batch_items, retries = submit_queue[0]
            batch_tokens = _estimate_batch_tokens(batch_items)

            # Check if this batch fits under the enqueued token limit
            if enqueued_tokens > 0 and enqueued_tokens + batch_tokens > enqueued_token_limit:
                log.info(f"Token budget full ({enqueued_tokens:,}/{enqueued_token_limit:,}). "
                         f"Next batch needs {batch_tokens:,} tokens — waiting for capacity.")
                break

            # Submit the batch without waiting for validation
            try:
                add_estimated_batch_cost(batch_items, model)
                file_path = generate_file_fn(batch_items, model)
                batch_id = submit_batch_job(file_path)
                log.info(f"Submitted {label} batch {batch_id} with {len(batch_items)} items ({batch_tokens:,} tokens)")
            except Exception as e:
                log.error(f"Failed to submit {label} batch: {e}. Deferring {len(batch_items)} items.")
                save_failed_batch(batch_items, label)
                submit_queue.pop(0)
                continue

            # Track immediately — poll phase will handle failures
            enqueued_tokens += batch_tokens
            pending[batch_id] = {"items": batch_items, "retries": retries, "tokens": batch_tokens}
            submit_queue.pop(0)

        # --- Poll phase: check all in-flight batches ---
        for batch_id, info in list(pending.items()):
            try:
                batch = openai.batches.retrieve(batch_id)
            except Exception as e:
                log.error(f"Error retrieving batch {batch_id}: {e}")
                continue

            status = batch.status
            request_counts = batch.request_counts
            completed_count = getattr(request_counts, "completed", 0)
            total_count = getattr(request_counts, "total", 0)
            log.info(f"Batch {batch_id} status: {status} — {completed_count}/{total_count} completed")

            if status == "completed":
                path = f"data/batch_responses/{label}_result_{uuid.uuid4().hex}.jsonl"
                try:
                    download_batch_results(batch_id, path)
                    result_paths.append(path)
                except Exception as e:
                    log.error(f"Failed to download results for batch {batch_id}: {e}")
                enqueued_tokens -= info["tokens"]
                del pending[batch_id]

            elif status == "expired":
                path = f"data/batch_responses/{label}_result_{uuid.uuid4().hex}.jsonl"
                if download_batch_results_if_available(batch_id, path):
                    result_paths.append(path)
                    processed_ids = get_processed_custom_ids(path)
                    unprocessed = [i for i in info["items"] if i["id"] not in processed_ids]
                    if unprocessed and info["retries"] < max_retries:
                        log.info(f"Retrying {len(unprocessed)} unprocessed items from expired batch {batch_id}")
                        submit_queue.append((unprocessed, info["retries"] + 1))
                    elif unprocessed:
                        log.warning(f"Max retries reached for {len(unprocessed)} items from batch {batch_id}")
                        save_failed_batch(unprocessed, label)
                else:
                    if info["retries"] < max_retries:
                        log.info(f"No partial results for expired batch {batch_id}. Re-queuing entire batch.")
                        submit_queue.append((info["items"], info["retries"] + 1))
                    else:
                        log.warning(f"Max retries reached for batch {batch_id}. Deferring.")
                        save_failed_batch(info["items"], label)
                enqueued_tokens -= info["tokens"]
                del pending[batch_id]

            elif status in ("failed", "cancelled"):
                if info["retries"] < max_retries:
                    log.warning(f"{label.capitalize()} batch {batch_id} {status}. Re-queuing for retry ({info['retries'] + 1}/{max_retries})...")
                    submit_queue.append((info["items"], info["retries"] + 1))
                else:
                    log.error(f"{label.capitalize()} batch {batch_id} {status} after {max_retries} retries. Deferring.")
                    save_failed_batch(info["items"], label)
                enqueued_tokens -= info["tokens"]
                del pending[batch_id]

            # else: still in_progress/validating/finalizing — keep polling

        if submit_queue or pending:
            time.sleep(60)

    return result_paths


def save_failed_batch(batch_items, label, folder="data/deferred"):
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, f"failed_{label}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in batch_items:
            f.write(json.dumps(item) + "\n")
    log.warning(f"Deferred {len(batch_items)} {label} items to {out_path}")

def is_valid_post(post):
    """Ensure post has valid title and body after sanitization."""
    title = sanitize_text(post.get("title", ""))
    body = sanitize_text(post.get("body", ""))
    return bool(title and body)

def split_batch_by_token_limit(payload, model: str, token_limit: int = 200_000):
    batches = []
    current_batch = []
    current_tokens = 0

    for item in payload:
        tokens = item.get("meta", {}).get("estimated_tokens", 300)
        if current_tokens + tokens > token_limit:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(item)
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    return batches

def clean_old_batch_files(folder="data/batch_responses", days_old=None):
    """Delete .jsonl files older than `days_old`. Defaults to config value."""
    days_old = days_old or config.get("cleanup", {}).get("batch_response_retention_days", 3)
    cutoff = time.time() - (days_old * 86400)

    deleted = 0
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if fname.endswith(".jsonl") and os.path.isfile(path):
            if os.path.getmtime(path) < cutoff:
                try:
                    os.remove(path)
                    deleted += 1
                except Exception as e:
                    log.warning(f"Failed to delete old file {path}: {e}")
    log.info(f"Cleaned up {deleted} old batch response files older than {days_old} days.")

def get_high_potential_ids_from_filter_results(score_threshold=7.0):
    processed = 0
    high_ids = set()
    all_processed_ids = set()
    weights = config["scoring"]
    for path in glob.glob("data/batch_responses/filter_result_*.jsonl"):
        with open(path, "r") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    processed += 1
                    post_id = result["custom_id"]
                    content = result["response"]["body"]["choices"][0]["message"]["content"]
                    scores = json.loads(content)
                    weighted_score = (
                        scores["relevance_score"] * weights["relevance_weight"] +
                        scores["emotional_intensity"] * weights["emotion_weight"] +
                        scores["pain_point_clarity"] * weights["pain_point_weight"] +
                        scores.get("implementability_score", 5) * weights.get("implementability_weight", 0)
                    )
                    update_post_filter_scores(post_id, scores)
                    all_processed_ids.add(post_id)
                    if weighted_score >= score_threshold:
                        high_ids.add(post_id)
                except Exception as e:
                    log.error(f"Error parsing filter result line: {e}")
    log.info(f"Processed {processed} filter results, found {len(high_ids)} high-potential posts")
    return high_ids, all_processed_ids

def run_daily_pipeline():
    log.info("\U0001F680 Starting Reddit scraping and analysis pipeline")

    ensure_directory_exists("data/deferred")
    ensure_directory_exists("data")
    ensure_directory_exists("data/batch_responses")
    clean_old_batch_files()
    create_tables()
    initialize_cost_tracking()

    log.info("Step 1: Cleaning old database entries...")
    clean_old_entries()

    log.info("Step 2: Scraping Reddit posts...")
    scraped_posts = scrape_subreddits()
    if not scraped_posts:
        log.warning("No posts found to analyze. Exiting pipeline.")
        return

    log.info(f"Found {len(scraped_posts)} posts before filtering invalid entries...")
    scraped_posts = [p for p in scraped_posts if is_valid_post(p)]
    log.info(f"{len(scraped_posts)} posts remain after sanitization/validation.")

    if not scraped_posts:
        log.warning("No valid posts after sanitization. Exiting pipeline.")
        return

    log.info("Step 3: Preparing posts for filtering...")
    filter_batch = prepare_filter_batch(scraped_posts)
    filter_cost = estimate_filter_cost(scraped_posts)
    log.info(f"Estimated cost for filtering: ${filter_cost:.2f}")

    if not can_process_batch(filter_cost):
        log.error("Insufficient budget for filtering. Exiting pipeline.")
        return

    model_filter = config["openai"]["model_filter"]
    filter_batches = split_batch_by_token_limit(filter_batch, model_filter)

    submit_batches_parallel(filter_batches, model_filter, generate_batch_payload, "filter")

    log.info("Step 4: Selecting high-potential posts from filter results...")
    high_potential_ids, all_filtered_ids = get_high_potential_ids_from_filter_results()

    # Mark posts that were filtered but NOT high-potential into history
    # (they're fully done — scored but below threshold, no further processing needed)
    below_threshold_ids = all_filtered_ids - high_potential_ids
    if below_threshold_ids:
        mark_posts_in_history(list(below_threshold_ids))
        log.info(f"Marked {len(below_threshold_ids)} below-threshold posts in history.")

    if not high_potential_ids:
        log.info("No high-value posts found. Exiting pipeline.")
        return

    deep_posts = get_posts_by_ids(high_potential_ids, require_unprocessed=True)
    if not deep_posts:
        log.info("No new posts left for deep insight. Exiting pipeline.")
        return

    insight_batch = prepare_insight_batch(deep_posts)
    insight_cost = estimate_insight_cost(insight_batch)
    log.info(f"Estimated cost for insight analysis: ${insight_cost:.2f}")

    if not can_process_batch(insight_cost):
        log.error("Insufficient budget for insight analysis. Exiting pipeline.")
        return

    log.info(f"Submitting batch of {len(insight_batch)} posts for deep analysis...")
    log.info(f"Preparing {len(insight_batch)} posts for deep insight...")
    model_deep = config["openai"]["model_deep"]
    insight_batches = split_batch_by_token_limit(insight_batch, model_deep)
    all_insight_paths = submit_batches_parallel(
        insight_batches, model_deep, generate_batch_payload, "insight"
    )

    log.info("Step 5: Updating posts with deep insights...")
    insight_completed_ids = []
    try:
        for insight_path in all_insight_paths:
            with open(insight_path, "r", encoding="utf-8") as f:
                for line in f:
                    result = json.loads(line)
                    post_id = result["custom_id"]
                    content = result["response"]["body"]["choices"][0]["message"]["content"]
                    try:
                        insight = json.loads(content)
                        update_post_insight(post_id, insight)
                        mark_insight_processed(post_id)
                        insight_completed_ids.append(post_id)
                    except Exception as e:
                        log.error(f"Error parsing insight for post {post_id}: {str(e)}")
    except Exception as e:
        log.error(f"Error reading insight results: {str(e)}")

    # Mark posts that completed insight analysis into history
    if insight_completed_ids:
        mark_posts_in_history(insight_completed_ids)
        log.info(f"Marked {len(insight_completed_ids)} insight-completed posts in history.")

    output_limit = config["scoring"]["output_top_n"]
    top_posts = get_top_insights_from_today(limit=output_limit)
    log.info(f"✅ Pipeline finished. Found {len(top_posts)} qualified leads.")

    for i, post in enumerate(top_posts[:5], 1):
        log.info(f"{i}. [{post['subreddit']}] {post['title']} — ROI: {post['roi_weight']} | Tags: {post['tags']} - {post['url']}")


if __name__ == "__main__":
    run_daily_pipeline()
