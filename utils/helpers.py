# utils/helpers.py

import os
import json
import re
import tiktoken
from datetime import datetime, timedelta, timezone

from config.config_loader import get_config
from utils.logger import setup_logger

# Setup logger and config
log = setup_logger()
config = get_config()

# Default tokenizer fallback
DEFAULT_ENCODING = "cl100k_base"
ENCODER = tiktoken.get_encoding(DEFAULT_ENCODING)

def _encoding_for_model(model: str):
    """Get best-available tokenizer encoding for a model string."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Newer OpenAI model families generally map better to o200k_base.
        if model.startswith(("gpt-4.1", "gpt-4o", "gpt-5", "o1", "o3", "o4")):
            return tiktoken.get_encoding("o200k_base")
        return tiktoken.get_encoding(DEFAULT_ENCODING)

def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Estimate token count for plain text."""
    enc = _encoding_for_model(model)
    return len(enc.encode(text or ""))

def estimate_chat_tokens(messages: list[dict], model: str = "gpt-4o-mini",
                         provider: str = "openai", response_format: dict | None = None,
                         extra_text: str = "") -> int:
    """Estimate tokens for a chat-completions style request.

    This is more accurate than counting only title/body text because it includes:
    - system/user wrapper message overhead
    - prompt template text
    - optional response_format payload
    - optional provider-specific extra instruction text
    """
    enc = _encoding_for_model(model)

    # OpenAI chat wrapper heuristic from tokenizer guidance.
    if provider == "openai":
        tokens_per_message = 3
        tokens_per_name = 1
        total = 3  # assistant priming

        for msg in messages or []:
            total += tokens_per_message
            for key, value in msg.items():
                if value is None:
                    continue
                if isinstance(value, str):
                    total += len(enc.encode(value))
                else:
                    total += len(enc.encode(json.dumps(value, ensure_ascii=False, separators=(",", ":"))))
                if key == "name":
                    total += tokens_per_name

        if response_format:
            total += len(enc.encode(json.dumps(response_format, ensure_ascii=False, separators=(",", ":"))))

        if extra_text:
            total += len(enc.encode(extra_text))

        return total

    # Generic fallback for non-OpenAI providers: serialize request payload.
    payload = {"model": model, "messages": messages or []}
    if response_format:
        payload["response_format"] = response_format
    if extra_text:
        payload["extra_text"] = extra_text
    return len(enc.encode(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))

def ensure_directory_exists(path: str):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def save_json(data: dict, filename: str):
    """Save dictionary as a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_json(filename: str) -> dict:
    """Load dictionary from JSON file."""
    if not os.path.exists(filename):
        return {}
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_datetime(dt: datetime = None) -> str:
    """Return timezone-aware datetime in ISO 8601 format (UTC)."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.isoformat()

def days_ago(days: int) -> datetime:
    """Return timezone-aware datetime for N days ago."""
    return datetime.now(timezone.utc) - timedelta(days=days)

def truncate(text: str, max_tokens: int = 1000) -> str:
    """Truncate a string to fit within a token limit."""
    tokens = ENCODER.encode(text or "")
    if len(tokens) <= max_tokens:
        return text
    return ENCODER.decode(tokens[:max_tokens])

def sanitize_text(text: str) -> str:
    #  TODO confirm *which* emojis/emoticons are removed. some are probably harmless, even meaningful and therefore desirable
    """Remove some emojis and non-printable unicode characters from the input."""
    if not isinstance(text, str):
        return ""
    emoji_pattern = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
    return emoji_pattern.sub("", text).strip()
