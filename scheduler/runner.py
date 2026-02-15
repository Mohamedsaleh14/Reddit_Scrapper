# scheduler/runner.py

import json
import uuid
import os
import glob
from datetime import datetime
import time
from reddit.scraper import scrape_subreddits
from db.writer import insert_post, update_post_filter_scores, update_post_insight, mark_insight_processed, mark_posts_in_history
from db.reader import get_top_insights_from_today, get_posts_by_ids, get_post_parent_mapping
from db.schema import create_tables
from gpt.filters import prepare_batch_payload as prepare_filter_batch, estimate_batch_cost as estimate_filter_cost
from gpt.insights import prepare_insight_batch, estimate_insight_cost
from gpt.batch_provider import (
    generate_batch_payload, submit_batch_job, poll_batch_status,
    download_batch_results, download_batch_results_if_available,
    get_processed_custom_ids, add_estimated_batch_cost,
    get_active_enqueued_tokens, probe_enqueued_capacity,
    retrieve_batch, extract_content_from_result
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



_provider = config["ai"]["provider"]
_provider_config = config["ai"][_provider]
ENQUEUED_TOKEN_LIMITS = _provider_config.get("enqueued_token_limits", {})
DEFAULT_ENQUEUED_TOKEN_LIMIT = _provider_config.get("default_enqueued_token_limit", 5_000_000)


def _wait_for_batch_confirmation(batch_id, timeout=300, poll_interval=10):
    """Quick-poll a newly submitted batch until it leaves 'validating' state.

    Returns the confirmed status: 'in_progress', 'failed', 'cancelled', etc.
    If the batch is still validating after timeout, returns 'validating'.

    For Anthropic, batches go straight to 'in_progress' (processing_status),
    so this typically returns immediately.
    """
    provider = config["ai"]["provider"]
    start = time.time()
    while time.time() - start < timeout:
        try:
            batch = retrieve_batch(batch_id)
            if provider == "anthropic":
                status = batch.processing_status
                # Anthropic statuses: in_progress, ended
                if status == "ended":
                    return "completed"
                return "in_progress"
            else:
                status = batch.status
                if status != "validating":
                    return status
        except Exception as e:
            log.warning(f"Error checking batch {batch_id}: {e}")
        time.sleep(poll_interval)
    return "validating"


def submit_batches_parallel(all_sub_batches, model, generate_file_fn, label,
                            max_retries=5, enqueued_token_limit=None,
                            max_completion_tokens=1024):
    """Submit batches with confirmed-capacity scheduling.

    Strategy: submit one batch at a time, wait for OpenAI to confirm it's
    actually processing (in_progress) before submitting the next. This avoids
    the "spray and pray" problem where 25 batches are submitted but 23 fail
    because OpenAI's internal enqueued token limit was exceeded.

    Multiple batches run in parallel — we just gate new submissions on
    confirmed capacity rather than blindly firing everything at once.

    Flow:
    1. Submit a batch.
    2. Quick-poll (every 10s) until it moves to in_progress or fails.
    3. If confirmed (in_progress): check if there's capacity for another batch.
    4. If failed with 0/0: capacity issue — back off, then retry.
    5. Poll all confirmed in-flight batches for completion.
    6. As batches complete and free capacity, submit more.
    7. Repeat until all batches are processed.
    """
    if enqueued_token_limit is None:
        enqueued_token_limit = ENQUEUED_TOKEN_LIMITS.get(model, DEFAULT_ENQUEUED_TOKEN_LIMIT)

    # Probe: verify enqueued capacity is actually free before starting.
    # This catches the OpenAI ghost-token bug where failed batches' tokens
    # are never released from the quota.
    if not probe_enqueued_capacity(model):
        log.error(f"Enqueued capacity for {model} is blocked (ghost tokens). "
                  f"Cannot proceed with {label} batch processing.")
        save_failed_batch(
            [item for batch in all_sub_batches for item in batch], label
        )
        return []

    result_paths = []
    confirmed = {}  # batch_id -> {"items": [...], "retries": int, "tokens": int}
    confirmed_tokens = 0  # tokens in confirmed in_progress/finalizing batches

    # Queue holds (items, retries) tuples
    submit_queue = [(batch_items, 0) for batch_items in all_sub_batches]
    backoff_delay = 0  # seconds to wait before next submit attempt

    log.info(f"Starting {label} batch processing: {len(submit_queue)} batches queued, "
             f"token limit: {enqueued_token_limit:,}")

    while submit_queue or confirmed:
        # --- Submit phase: submit one batch at a time, confirm before next ---
        while submit_queue and backoff_delay == 0:
            batch_items, retries = submit_queue[0]
            batch_tokens = _estimate_batch_tokens(batch_items)

            # Check capacity using confirmed tokens from OpenAI
            enqueued_tokens = get_active_enqueued_tokens()
            if enqueued_tokens > 0 and enqueued_tokens + batch_tokens > enqueued_token_limit:
                log.info(f"Token budget full ({enqueued_tokens:,}/{enqueued_token_limit:,}). "
                         f"Next batch needs {batch_tokens:,} tokens — waiting for capacity.")
                break

            # Submit the batch
            try:
                add_estimated_batch_cost(batch_items, model)
                file_path = generate_file_fn(batch_items, model,
                                             max_completion_tokens=max_completion_tokens)
                batch_id = submit_batch_job(file_path, estimated_tokens=batch_tokens)
                log.info(f"Submitted {label} batch {batch_id} with {len(batch_items)} items "
                         f"({batch_tokens:,} tokens, max_output={max_completion_tokens}). "
                         f"Waiting for confirmation...")
            except Exception as e:
                log.error(f"Failed to submit {label} batch: {e}. Deferring {len(batch_items)} items.")
                save_failed_batch(batch_items, label)
                submit_queue.pop(0)
                continue

            submit_queue.pop(0)

            # Wait for OpenAI to confirm (validating -> in_progress or failed)
            status = _wait_for_batch_confirmation(batch_id)

            if status in ("in_progress", "finalizing", "completed"):
                log.info(f"Batch {batch_id} confirmed: {status}")
                confirmed[batch_id] = {"items": batch_items, "retries": retries, "tokens": batch_tokens}
                confirmed_tokens += batch_tokens
                backoff_delay = 0  # reset backoff on success

                # If completed already during confirmation, handle it immediately
                if status == "completed":
                    path = f"data/batch_responses/{label}_result_{uuid.uuid4().hex}.jsonl"
                    try:
                        download_batch_results(batch_id, path)
                        result_paths.append(path)
                        processed_ids = get_processed_custom_ids(path)
                        unprocessed = [i for i in batch_items if i["id"] not in processed_ids]
                        if unprocessed and retries < max_retries:
                            log.warning(f"Retrying {len(unprocessed)} non-succeeded items from completed batch {batch_id}")
                            submit_queue.append((unprocessed, retries + 1))
                        elif unprocessed:
                            log.warning(f"Max retries reached for {len(unprocessed)} items from batch {batch_id}")
                            save_failed_batch(unprocessed, label)
                    except Exception as e:
                        log.error(f"Failed to download results for batch {batch_id}: {e}")
                    confirmed_tokens -= batch_tokens
                    del confirmed[batch_id]
            elif status == "failed":
                # 0/0 failure = likely capacity/ghost-token issue.
                # Probe to wait until capacity is actually free before retrying.
                if retries < max_retries:
                    log.warning(f"Batch {batch_id} failed (likely capacity). "
                                f"Probing to wait for capacity before retry "
                                f"({retries + 1}/{max_retries})...")
                    if probe_enqueued_capacity(model, max_wait=3600):
                        submit_queue.insert(0, (batch_items, retries + 1))
                    else:
                        log.error(f"Capacity still blocked after probing. Deferring batch.")
                        save_failed_batch(batch_items, label)
                else:
                    log.error(f"Batch {batch_id} failed after {max_retries} retries. Deferring.")
                    save_failed_batch(batch_items, label)
                break  # exit submit loop to wait/re-check
            elif status == "validating":
                # Still validating after timeout — treat as tentatively confirmed
                log.warning(f"Batch {batch_id} still validating after timeout. "
                            f"Tracking it and continuing.")
                confirmed[batch_id] = {"items": batch_items, "retries": retries, "tokens": batch_tokens}
                confirmed_tokens += batch_tokens
            else:
                # cancelled or other unexpected status
                if retries < max_retries:
                    log.warning(f"Batch {batch_id} status: {status}. Re-queuing (retry {retries + 1}/{max_retries})...")
                    submit_queue.insert(0, (batch_items, retries + 1))
                else:
                    log.error(f"Batch {batch_id} {status} after {max_retries} retries. Deferring.")
                    save_failed_batch(batch_items, label)

        # --- Poll phase: check all confirmed in-flight batches ---
        for batch_id, info in list(confirmed.items()):
            try:
                batch = retrieve_batch(batch_id)
            except Exception as e:
                log.error(f"Error retrieving batch {batch_id}: {e}")
                continue

            provider = config["ai"]["provider"]
            if provider == "anthropic":
                counts = batch.request_counts
                completed_count = counts.succeeded + counts.errored
                total_count = completed_count + counts.processing + counts.canceled + counts.expired
                status = "completed" if batch.processing_status == "ended" else batch.processing_status
                if status == "in_progress":
                    status = "in_progress"  # keep as-is for the logic below
            else:
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
                    processed_ids = get_processed_custom_ids(path)
                    unprocessed = [i for i in info["items"] if i["id"] not in processed_ids]
                    if unprocessed and info["retries"] < max_retries:
                        log.warning(f"Retrying {len(unprocessed)} non-succeeded items from completed batch {batch_id}")
                        submit_queue.append((unprocessed, info["retries"] + 1))
                    elif unprocessed:
                        log.warning(f"Max retries reached for {len(unprocessed)} items from batch {batch_id}")
                        save_failed_batch(unprocessed, label)
                except Exception as e:
                    log.error(f"Failed to download results for batch {batch_id}: {e}")
                confirmed_tokens -= info["tokens"]
                del confirmed[batch_id]

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
                confirmed_tokens -= info["tokens"]
                del confirmed[batch_id]

            elif status in ("failed", "cancelled"):
                if info["retries"] < max_retries:
                    log.warning(f"{label.capitalize()} batch {batch_id} {status}. "
                                f"Re-queuing for retry ({info['retries'] + 1}/{max_retries})...")
                    submit_queue.append((info["items"], info["retries"] + 1))
                else:
                    log.error(f"{label.capitalize()} batch {batch_id} {status} after {max_retries} retries. Deferring.")
                    save_failed_batch(info["items"], label)
                confirmed_tokens -= info["tokens"]
                del confirmed[batch_id]

            # else: still in_progress/finalizing — keep polling

        # --- Wait phase ---
        if submit_queue or confirmed:
            if backoff_delay > 0:
                log.info(f"Backing off for {backoff_delay}s before next submit attempt. "
                         f"Queue: {len(submit_queue)} batches, In-flight: {len(confirmed)} batches")
                time.sleep(backoff_delay)
                backoff_delay = 0  # reset after waiting
            else:
                remaining = len(submit_queue)
                in_flight = len(confirmed)
                log.info(f"Waiting 60s. Queue: {remaining} batches, "
                         f"In-flight: {in_flight} batches, "
                         f"Confirmed tokens: {confirmed_tokens:,}/{enqueued_token_limit:,}")
                time.sleep(60)

    log.info(f"{label.capitalize()} batch processing complete. "
             f"Downloaded {len(result_paths)} result files.")
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
    high_candidates = {}  # post_id -> weighted_score (before dedup)
    all_processed_ids = set()
    weights = config["scoring"]
    for path in glob.glob("data/batch_responses/filter_result_*.jsonl"):
        with open(path, "r") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    if result.get("result_type") and result.get("result_type") != "succeeded":
                        continue
                    processed += 1
                    post_id, content = extract_content_from_result(result)
                    scores = json.loads(content)
                    technical_depth = scores.get("technical_depth_score", 5)
                    min_depth = weights.get("min_technical_depth", 4)

                    weighted_score = (
                        scores["relevance_score"] * weights["relevance_weight"] +
                        scores["emotional_intensity"] * weights["emotion_weight"] +
                        scores["pain_point_clarity"] * weights["pain_point_weight"] +
                        scores.get("implementability_score", 5) * weights.get("implementability_weight", 0) +
                        technical_depth * weights.get("technical_depth_weight", 0.2)
                    )
                    update_post_filter_scores(post_id, scores)
                    all_processed_ids.add(post_id)
                    # Hard filter: reject vibe-codeable ideas with low technical depth
                    if technical_depth < min_depth:
                        log.debug(f"Rejected post {post_id}: technical_depth_score {technical_depth} < {min_depth}")
                        continue
                    if weighted_score >= score_threshold:
                        high_candidates[post_id] = weighted_score
                except Exception as e:
                    log.error(f"Error parsing filter result line: {e}")

    # Deduplicate: keep only the highest-scoring entry per thread
    parent_mapping = get_post_parent_mapping(set(high_candidates.keys()))
    thread_best = {}  # thread_id -> (post_id, weighted_score)
    for post_id, score in high_candidates.items():
        thread_id = parent_mapping.get(post_id) or post_id  # comments group under parent, posts are their own thread
        if thread_id not in thread_best or score > thread_best[thread_id][1]:
            thread_best[thread_id] = (post_id, score)

    high_ids = {post_id for post_id, _ in thread_best.values()}
    deduped_count = len(high_candidates) - len(high_ids)
    if deduped_count > 0:
        log.info(f"Deduplicated {deduped_count} same-thread entries (kept best per thread)")

    log.info(f"Processed {processed} filter results, found {len(high_ids)} high-potential posts (from {len(high_candidates)} candidates)")
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

    model_filter = config["ai"][config["ai"]["provider"]]["model_filter"]
    filter_batches = split_batch_by_token_limit(filter_batch, model_filter)

    submit_batches_parallel(filter_batches, model_filter, generate_batch_payload, "filter",
                            max_completion_tokens=512)

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
    model_deep = config["ai"][config["ai"]["provider"]]["model_deep"]
    insight_batches = split_batch_by_token_limit(insight_batch, model_deep)
    all_insight_paths = submit_batches_parallel(
        insight_batches, model_deep, generate_batch_payload, "insight",
        max_completion_tokens=1024
    )

    log.info("Step 5: Updating posts with deep insights...")
    insight_completed_ids = []
    try:
        for insight_path in all_insight_paths:
            with open(insight_path, "r", encoding="utf-8") as f:
                for line in f:
                    result = json.loads(line)
                    post_id, content = extract_content_from_result(result)
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
