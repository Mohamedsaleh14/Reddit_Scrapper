import json
import uuid
import time
import os
import anthropic
from config.config_loader import get_config
from scheduler.cost_tracker import add_cost
from utils.logger import setup_logger

log = setup_logger()
config = get_config()

RESPONSE_DIR = "data/batch_responses"


def _get_client():
    api_key = config["ai"]["anthropic"].get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=api_key)


def generate_batch_payload(requests: list[dict], model: str,
                           max_completion_tokens: int = 1024) -> str:
    """Create a JSONL file with Anthropic Message Batches request format.

    Each line: {"custom_id": "...", "params": {"model": "...", "max_tokens": ..., "messages": [...]}}
    """
    os.makedirs(RESPONSE_DIR, exist_ok=True)
    job_id = str(uuid.uuid4())
    path = f"{RESPONSE_DIR}/batch_{job_id}.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for prompt in requests:
            messages = prompt["messages"]

            # Anthropic separates system from messages — extract system if present
            system_content = None
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    user_messages.append(msg)

            # Add explicit JSON instruction for Anthropic (no response_format support)
            if system_content:
                system_content += "\n\nYou MUST respond with valid JSON only."
            else:
                system_content = "You MUST respond with valid JSON only."

            params = {
                "model": model,
                "max_tokens": max_completion_tokens,
                "messages": user_messages,
            }
            if system_content:
                params["system"] = system_content

            f.write(json.dumps({
                "custom_id": prompt.get("id", str(uuid.uuid4())),
                "params": params,
            }) + "\n")

    log.info(f"Batch payload generated at: {path} with {len(requests)} entries")
    return path


def submit_batch_job(file_path: str, estimated_tokens: int = 0) -> str:
    """Reads JSONL file and submits a batch job to Anthropic."""
    client = _get_client()

    # Read requests from JSONL
    requests = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            req = json.loads(line.strip())
            requests.append(req)

    log.info(f"Submitting Anthropic batch with {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    log.info(f"Submitted Anthropic batch job: {batch.id}")
    return batch.id


def poll_batch_status(batch_id: str, timeout_seconds: int = 10800) -> dict:
    """Polls Anthropic batch status every 60s until ended or timeout."""
    client = _get_client()
    start_time = time.time()

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status

        counts = batch.request_counts
        succeeded = counts.succeeded
        errored = counts.errored
        processing = counts.processing
        total = succeeded + errored + processing + counts.canceled + counts.expired

        log.info(f"Batch {batch_id} status: {status} — "
                 f"{succeeded}/{total} succeeded, {errored} errored, {processing} processing")

        if status == "ended":
            if errored > 0 and succeeded == 0:
                return {"status": "failed", "batch": batch}
            return {"status": "completed", "batch": batch}

        if time.time() - start_time > timeout_seconds:
            log.warning(f"Batch {batch_id} timed out after {timeout_seconds}s. "
                        f"Attempting to cancel...")
            try:
                client.messages.batches.cancel(batch_id)
            except Exception as e:
                log.error(f"Error cancelling batch {batch_id}: {e}")
            return {"status": "cancelled", "batch": batch}

        time.sleep(60)


def retrieve_batch(batch_id: str):
    """Retrieve batch object for status checks."""
    client = _get_client()
    return client.messages.batches.retrieve(batch_id)


def download_batch_results(batch_id: str, save_path: str):
    """Downloads and stores results from a completed Anthropic batch.

    Writes normalized JSONL where each line has:
    {"custom_id": "...", "content": "..."}
    """
    client = _get_client()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    count = 0
    with open(save_path, "w", encoding="utf-8") as f:
        for result in client.messages.batches.results(batch_id):
            entry = {
                "custom_id": result.custom_id,
            }
            if result.result.type == "succeeded":
                content_blocks = result.result.message.content
                text = content_blocks[0].text if content_blocks else ""
                entry["content"] = text
                entry["result_type"] = "succeeded"
            else:
                entry["content"] = ""
                entry["result_type"] = result.result.type
                error_obj = getattr(result.result, "error", None)
                if error_obj:
                    entry["error_type"] = getattr(error_obj, "type", None)
                    entry["error_message"] = getattr(error_obj, "message", None)
                log.warning(f"Non-succeeded result for {result.custom_id}: {result.result.type}")

            f.write(json.dumps(entry) + "\n")
            count += 1

    log.info(f"Saved {count} Anthropic results to {save_path}")


def download_batch_results_if_available(batch_id: str, save_path: str) -> bool:
    """Downloads results if the batch has any, regardless of overall status."""
    try:
        download_batch_results(batch_id, save_path)
        # Check if we actually got any results
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return True
        return False
    except Exception as e:
        log.warning(f"Could not download results for Anthropic batch {batch_id}: {e}")
        return False


def get_processed_custom_ids(result_path: str) -> set:
    """Return IDs with usable (succeeded) outputs.

    Anthropic result files include `result_type`. Only `succeeded` should be
    treated as processed so errored/cancelled/expired items can be retried.
    """
    processed = set()
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                result = json.loads(line)
                result_type = result.get("result_type")
                if result_type and result_type != "succeeded":
                    continue
                custom_id = result.get("custom_id")
                if custom_id:
                    processed.add(custom_id)
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def get_active_enqueued_tokens() -> int:
    """Estimate active enqueued tokens from Anthropic batches."""
    client = _get_client()
    total_tokens = 0
    avg_tokens_per_request = 350

    try:
        batches = client.messages.batches.list(limit=100)
        for batch in batches.data:
            if batch.processing_status != "in_progress":
                continue
            processing = batch.request_counts.processing
            total_tokens += processing * avg_tokens_per_request

        log.info(f"Active enqueued tokens at Anthropic: {total_tokens:,}")
    except Exception as e:
        log.warning(f"Failed to query active batches from Anthropic: {e}")

    return total_tokens


def probe_enqueued_capacity(model: str, max_wait=7200, poll_interval=300) -> bool:
    """Anthropic doesn't have the ghost token bug — always returns True."""
    log.info(f"[Probe] Anthropic provider — no ghost token bug. Capacity confirmed.")
    return True


def add_estimated_batch_cost(requests: list[dict], model: str):
    """Estimate and record the cost of an Anthropic batch job."""
    # Per 1M token pricing in USD (batch API gets 50% discount)
    pricing = {
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    }

    discount_factor = 0.5  # 50% discount for batch

    model_pricing = pricing.get(model, {"input": 3.00, "output": 15.00})

    input_tokens = sum(req.get("meta", {}).get("estimated_tokens", 300) for req in requests)
    output_tokens = len(requests) * 300

    input_cost = (input_tokens / 1_000_000) * model_pricing["input"] * discount_factor
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"] * discount_factor
    estimated_cost = input_cost + output_cost

    log.info(f"Estimated cost for Anthropic batch (input + output): ${estimated_cost:.4f}")
    add_cost(estimated_cost)
