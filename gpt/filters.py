# gpt/filters.py

import os
from typing import List, Dict
from utils.helpers import estimate_chat_tokens, sanitize_text
from utils.logger import setup_logger
from config.config_loader import get_config, PROMPT_FILTER

log = setup_logger()
config = get_config()


def _format_post_content(post: dict) -> str:
    """Format post/comment content with appropriate context."""
    post_body = post.get("post_body", "")
    if post_body:
        # This is a comment — include parent post context
        return (
            f"Post title: {post['title']}\n"
            f"Post body: {post_body}\n"
            f"Comment: {post['body']}"
        )
    return f"Post title: {post['title']}\nPost body: {post['body']}"


def build_filter_prompt(post: dict) -> List[Dict]:
    """Constructs the GPT-4.1 Mini prompt for a Reddit post using template."""
    return [
        {
            "role": "system",
            "content": "You are a market research analyst identifying posts that describe genuine pain points with product potential."
        },
        {
            "role": "user",
            "content": f"{config['prompts'][PROMPT_FILTER]}\n\n{_format_post_content(post)}"
        }
    ]


def prepare_batch_payload(posts: List[dict]) -> List[Dict]:
    """Returns list of payloads for batch submission."""
    payload = []
    provider = config["ai"]["provider"]
    model = config["ai"][provider].get("model_filter", "gpt-4o-mini")
    extra_text = "\n\nYou MUST respond with valid JSON only." if provider == "anthropic" else ""

    for post in posts:
        raw_title = post.get("title", "")
        raw_body = post.get("body", "")
        title = sanitize_text(raw_title)
        body = sanitize_text(raw_body)

        if not title or not body:
            log.error("post is invalid, this should already have been asserted - skipping")
            continue  # skip malformed posts

        post_body = sanitize_text(post.get("post_body", ""))
        messages = build_filter_prompt({"title": title, "body": body, "post_body": post_body})
        payload.append({
            "id": post["id"],
            "messages": messages,
            "meta": {
                "estimated_tokens": estimate_chat_tokens(
                    messages,
                    model=model,
                    provider=provider,
                    response_format={"type": "json_object"} if provider == "openai" else None,
                    extra_text=extra_text,
                )
            }
        })
    return payload


def estimate_batch_cost(batch: List[dict], model: str = "gpt-4o-mini", avg_tokens: int = 300) -> float:
    """
    Estimate cost of a filtering batch using actual model pricing.
    """
    pricing = {
        "gpt-4.1": {"input": 0.0010},
        "gpt-4.1-mini": {"input": 0.0002},
        "gpt-4o-mini": {"input": 0.000075},
    }

    model_pricing = pricing.get(model, {"input": 0.0005})  # default fallback
    if batch and isinstance(batch[0], dict) and "meta" in batch[0]:
        total_tokens = sum(item.get("meta", {}).get("estimated_tokens", avg_tokens) for item in batch)
    else:
        total_tokens = len(batch) * avg_tokens
    input_cost_per_1k = model_pricing["input"]

    return (total_tokens / 1000) * input_cost_per_1k
