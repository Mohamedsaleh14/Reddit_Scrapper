import json
import uuid
import time
import os
import openai
import requests
from config.config_loader import get_config
from scheduler.cost_tracker import add_cost
from utils.logger import setup_logger

log = setup_logger()
config = get_config()

RESPONSE_DIR = "data/batch_responses"

def generate_batch_payload(requests: list[dict], model: str) -> str:
    """Create a JSONL file from prompts for OpenAI batch processing."""
    os.makedirs(RESPONSE_DIR, exist_ok=True)
    job_id = str(uuid.uuid4())
    path = f"{RESPONSE_DIR}/batch_{job_id}.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for prompt in requests:
            body = {
                    "model": model,
                    "messages": prompt["messages"],
                    "response_format": {"type": "json_object"},
                }
            # gpt-5+ models only support the default temperature (1)
            if not model.startswith("gpt-5"):
                body["temperature"] = 0
            f.write(json.dumps({
                "custom_id": prompt.get("id", str(uuid.uuid4())),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }) + "\n")

    log.info(f"Batch payload generated at: {path} with {len(requests)} entries")
    return path

def submit_batch_job(file_path: str, endpoint: str = "/v1/chat/completions") -> str:
    """Uploads file and submits a batch job to OpenAI."""
    uploaded_file = openai.files.create(file=open(file_path, "rb"), purpose="batch")
    log.info(f"Uploaded file for batch: {uploaded_file.id}")

    batch = openai.batches.create(
        input_file_id=uploaded_file.id,
        endpoint=endpoint,
        completion_window="24h",
    )
    log.info(f"Submitted batch job: {batch.id}")
    return batch.id

def poll_batch_status(batch_id: str, timeout_seconds: int = 10800) -> dict:
    """Polls batch status every 60s. Cancels if no progress in `timeout_seconds`. Waits for confirmed cancellation."""
    previous_completed = 0
    last_progress_time = time.time()

    while True:
        batch = openai.batches.retrieve(batch_id)
        status = batch.status
        request_counts = batch.request_counts
        completed = getattr(request_counts, "completed", 0)
        total = getattr(request_counts, "total", 0)

        log.info(f"Batch {batch_id} status: {status} — {completed}/{total} completed")

        if status in {"completed", "failed", "expired"}:
            return {"status": status, "batch": batch}

        # Reset timeout if there's progress
        if completed > previous_completed:
            last_progress_time = time.time()
            previous_completed = completed

        # Cancel if no progress for too long
        if time.time() - last_progress_time > timeout_seconds:
            log.warning(f"No progress in last {timeout_seconds // 60} mins. Cancelling batch {batch_id}...")
            try:
                openai.batches.cancel(batch_id)
            except Exception as e:
                log.error(f"Error cancelling batch {batch_id}: {e}")

            # Wait for cancellation confirmation
            while True:
                batch = openai.batches.retrieve(batch_id)
                log.info(f"Waiting for cancellation... Current status: {batch.status}")
                if batch.status in {"cancelled", "failed", "expired"}:
                    return {"status": "cancelled", "batch": batch}
                time.sleep(60)

        time.sleep(60)

def download_batch_results(batch_id: str, save_path: str):
    """Downloads and stores the results of a completed batch job."""
    batch = openai.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch_id} not completed.")

    output_file_id = batch.output_file_id  # ← updated attribute
    if not output_file_id:
        raise RuntimeError(f"No output file found for batch {batch_id}.")

    output_file = openai.files.retrieve(output_file_id)
    response = openai.files.content(output_file_id)

    with open(save_path, "wb") as f:
        f.write(response.read())

    log.info(f"Saved results to {save_path}")

def download_batch_results_if_available(batch_id: str, save_path: str) -> bool:
    """Downloads results from a batch if output_file_id exists, regardless of batch status.

    This handles expired batches that may have partial results which
    download_batch_results() cannot retrieve (it requires status == 'completed').
    """
    batch = openai.batches.retrieve(batch_id)
    output_file_id = batch.output_file_id

    if not output_file_id:
        log.warning(f"No output file available for batch {batch_id} (status: {batch.status})")
        return False

    response = openai.files.content(output_file_id)
    with open(save_path, "wb") as f:
        f.write(response.read())

    log.info(f"Saved available results from batch {batch_id} (status: {batch.status}) to {save_path}")
    return True


def get_processed_custom_ids(result_path: str) -> set:
    """Reads a results JSONL file and returns the set of custom_id values that were processed."""
    processed = set()
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                result = json.loads(line)
                custom_id = result.get("custom_id")
                if custom_id:
                    processed.add(custom_id)
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def add_estimated_batch_cost(requests: list[dict], model: str):
    """Estimate and record the cost of the batch job using accurate pricing."""
    # Per 1M token pricing in USD
    pricing = {
        "gpt-4.1": {"input": 0.0020, "output": 0.0080},
        "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    }

    discount_factor = 0.5  # 50% discount for batch jobs

    # Default fallback pricing
    model_pricing = pricing.get(model, {"input": 0.0010, "output": 0.0010})

    input_tokens = sum(req.get("meta", {}).get("estimated_tokens", 300) for req in requests)
    output_tokens = len(requests) * 300  # Assumes ~300 output tokens per completion

    input_cost = (input_tokens / 200_000) * model_pricing["input"] * discount_factor
    output_cost = (output_tokens / 500_000) * model_pricing["output"] * discount_factor
    estimated_cost = input_cost + output_cost

    log.info(f"Estimated cost for batch (input + output): ${estimated_cost:.4f}")
    add_cost(estimated_cost)
