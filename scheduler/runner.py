# scheduler/runner.py

import json
import uuid
import os
import glob
from datetime import datetime
import time
from reddit.scraper import scrape_all_configured_subreddits
from db.writer import insert_post, update_post_filter_scores, update_post_insight, mark_insight_processed
from db.reader import get_top_insights_from_today, get_posts_by_ids
from db.schema import create_tables
from gpt.filters import prepare_batch_payload as prepare_filter_batch, estimate_batch_cost as estimate_filter_cost
from gpt.insights import prepare_insight_batch, estimate_insight_cost
from gpt.batch_api import generate_batch_payload, submit_batch_job, poll_batch_status, download_batch_results, add_estimated_batch_cost
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
                        scores["pain_point_clarity"] * weights["pain_point_weight"]
                    )
                    if weighted_score >= score_threshold:
                        high_ids.add(post_id)
                        update_post_filter_scores(post_id, scores)
                except Exception as e:
                    log.error(f"Error parsing filter result line: {e}")
    log.info(f"Processed {processed} filter results, found {len(high_ids)} high-potential posts")
    return high_ids

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
    scraped_posts = scrape_all_configured_subreddits()
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

    for i, batch in enumerate(filter_batches):
        log.info(f"Submitting sub-batch {i + 1}/{len(filter_batches)} with {len(batch)} entries...")
        add_estimated_batch_cost(batch, model_filter)

        batch_id = submit_with_backoff(
            batch_items=batch,
            model=model_filter,
            generate_file_fn=generate_batch_payload,
            label="filter"
        )

        if not batch_id:
            continue  # move on to next batch

        result_path = f"data/batch_responses/filter_result_{uuid.uuid4().hex}.jsonl"
        download_batch_results(batch_id, result_path)

    log.info("Step 4: Selecting high-potential posts from filter results...")
    high_potential_ids = get_high_potential_ids_from_filter_results()
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
    all_insight_paths = []

    for i, batch in enumerate(insight_batches):
        log.info(f"Submitting insight sub-batch {i + 1}/{len(insight_batches)} with {len(batch)} entries...")
        add_estimated_batch_cost(batch, model_deep)

        batch_id = submit_with_backoff(
            batch_items=batch,
            model=model_deep,
            generate_file_fn=generate_batch_payload,
            label="insight"
        )

        if not batch_id:
            continue

        insight_path = f"data/batch_responses/insight_result_{uuid.uuid4().hex}.jsonl"
        download_batch_results(batch_id, insight_path)
        all_insight_paths.append(insight_path)

    log.info("Step 5: Updating posts with deep insights...")
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
                    except Exception as e:
                        log.error(f"Error parsing insight for post {post_id}: {str(e)}")
    except Exception as e:
        log.error(f"Error reading insight results: {str(e)}")

    output_limit = config["scoring"]["output_top_n"]
    top_posts = get_top_insights_from_today(limit=output_limit)
    log.info(f"✅ Pipeline finished. Found {len(top_posts)} qualified leads.")

    for i, post in enumerate(top_posts[:5], 1):
        log.info(f"{i}. [{post['subreddit']}] {post['title']} — ROI: {post['roi_weight']} | Tags: {post['tags']} - {post['url']}")


if __name__ == "__main__":
    run_daily_pipeline()
