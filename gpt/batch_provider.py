"""Provider routing layer for batch API operations.

Delegates all batch operations to either OpenAI (gpt/batch_api.py) or
Anthropic (gpt/anthropic_batch.py) based on config["ai"]["provider"].
"""

import json
import re
from config.config_loader import get_config

_config = get_config()


def _provider():
    return _config["ai"]["provider"]


def generate_batch_payload(requests, model):
    if _provider() == "anthropic":
        from gpt.anthropic_batch import generate_batch_payload as fn
    else:
        from gpt.batch_api import generate_batch_payload as fn
    return fn(requests, model)


def submit_batch_job(file_path, estimated_tokens=0):
    if _provider() == "anthropic":
        from gpt.anthropic_batch import submit_batch_job as fn
    else:
        from gpt.batch_api import submit_batch_job as fn
    return fn(file_path, estimated_tokens=estimated_tokens)


def poll_batch_status(batch_id, timeout_seconds=10800):
    if _provider() == "anthropic":
        from gpt.anthropic_batch import poll_batch_status as fn
    else:
        from gpt.batch_api import poll_batch_status as fn
    return fn(batch_id, timeout_seconds=timeout_seconds)


def download_batch_results(batch_id, save_path):
    if _provider() == "anthropic":
        from gpt.anthropic_batch import download_batch_results as fn
    else:
        from gpt.batch_api import download_batch_results as fn
    return fn(batch_id, save_path)


def download_batch_results_if_available(batch_id, save_path):
    if _provider() == "anthropic":
        from gpt.anthropic_batch import download_batch_results_if_available as fn
    else:
        from gpt.batch_api import download_batch_results_if_available as fn
    return fn(batch_id, save_path)


def get_processed_custom_ids(result_path):
    if _provider() == "anthropic":
        from gpt.anthropic_batch import get_processed_custom_ids as fn
    else:
        from gpt.batch_api import get_processed_custom_ids as fn
    return fn(result_path)


def get_active_enqueued_tokens():
    if _provider() == "anthropic":
        from gpt.anthropic_batch import get_active_enqueued_tokens as fn
    else:
        from gpt.batch_api import get_active_enqueued_tokens as fn
    return fn()


def probe_enqueued_capacity(model, max_wait=7200, poll_interval=300):
    if _provider() == "anthropic":
        from gpt.anthropic_batch import probe_enqueued_capacity as fn
    else:
        from gpt.batch_api import probe_enqueued_capacity as fn
    return fn(model, max_wait=max_wait, poll_interval=poll_interval)


def add_estimated_batch_cost(requests, model):
    if _provider() == "anthropic":
        from gpt.anthropic_batch import add_estimated_batch_cost as fn
    else:
        from gpt.batch_api import add_estimated_batch_cost as fn
    return fn(requests, model)


def clean_storage():
    """Clean provider's remote file storage before starting a run.

    For OpenAI, deletes old batch input/output files that accumulate and
    can block new submissions. For Anthropic, this is a no-op (no file storage).
    """
    if _provider() == "openai":
        from gpt.batch_api import clean_storage as fn
        fn()


def retrieve_batch(batch_id):
    """Retrieve a batch object for status checks."""
    if _provider() == "anthropic":
        from gpt.anthropic_batch import retrieve_batch as fn
        return fn(batch_id)
    else:
        import openai
        return openai.batches.retrieve(batch_id)


def _extract_json_from_text(text: str) -> str:
    """Extract JSON from text that may be wrapped in markdown code fences.

    Anthropic often returns ```json ... ``` followed by reasoning text.
    This extracts just the JSON block.
    """
    # Try to extract from markdown code fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no fence, try to find the first JSON object or array
    match = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def extract_content_from_result(result_line: dict) -> tuple[str, str]:
    """Extract (custom_id, content_text) from a result line regardless of provider.

    OpenAI format: {"custom_id": "...", "response": {"body": {"choices": [{"message": {"content": "..."}}]}}}
    Anthropic normalized format: {"custom_id": "...", "content": "..."}

    For Anthropic, the content may be wrapped in markdown code fences since
    Anthropic doesn't support response_format: json_object. This function
    extracts the raw JSON string from within the fences.
    """
    custom_id = result_line["custom_id"]

    if _provider() == "anthropic":
        content = result_line.get("content", "")
        content = _extract_json_from_text(content)
    else:
        # OpenAI nested format — content is already pure JSON
        content = result_line["response"]["body"]["choices"][0]["message"]["content"]

    return custom_id, content
