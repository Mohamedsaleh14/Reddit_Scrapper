# gpt/insights.py

import os
from typing import List, Dict, Any
from utils.helpers import estimate_tokens, sanitize_text
from utils.logger import setup_logger
from config.config_loader import get_config, PROMPT_INSIGHT

log = setup_logger()
config = get_config()

def build_insight_prompt(post: dict) -> List[Dict[str, str]]:
    """Constructs the GPT-4.1 prompt for extracting deeper insights using template."""
    return [
        {
            "role": "system",
            "content": "You are a product strategist evaluating Reddit posts for viable product opportunities."
        },
        {
            "role": "user",
            "content": f"{config['prompts'][PROMPT_INSIGHT]}\n\nPost title: {post['title']}\nPost body: {post['body']}"
        }
    ]


def prepare_insight_batch(posts: List[dict]) -> List[Dict[str, Any]]:
    """Prepares GPT-4.1 insight batch payload."""
    model = config["openai"].get("model_deep", "gpt-4.1")
    payload = []

    for post in posts:
        raw_title = post.get("title", "")
        raw_body = post.get("body", "")
        title = sanitize_text(raw_title)
        body = sanitize_text(raw_body)

        if not title or not body:
            continue  # skip malformed posts

        messages = build_insight_prompt({"title": title, "body": body})
        payload.append({
            "id": post["id"],
            "messages": messages,
            "meta": {
                "estimated_tokens": estimate_tokens(title + body, model)
            }
        })
    return payload


def estimate_insight_cost(batch: List[Dict]) -> float:
    """Estimate GPT-4.1 cost using real input token metadata."""
    cost_per_1k_input = 0.0020
    cost_per_1k_output = 0.0080
    discount = 0.05  # Batch API discount

    input_tokens = sum(item.get("meta", {}).get("estimated_tokens", 700) for item in batch)
    output_tokens = len(batch) * 300  # assume output tokens

    return (
        (input_tokens / 1_000_000 * cost_per_1k_input) +
        (output_tokens / 1_000_000 * cost_per_1k_output)
    ) * discount
