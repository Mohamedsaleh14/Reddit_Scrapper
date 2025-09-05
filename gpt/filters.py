# gpt/filters.py

import openai
import os
from typing import List, Dict
from utils.helpers import estimate_tokens, sanitize_text
from utils.logger import setup_logger
from config.config_loader import get_config

log = setup_logger()
config = get_config()
PROMPT_PATH = "gpt/prompts/filter_prompt.txt"


def load_filter_prompt_template():
    """Load the filter prompt template from file."""
    try:
        with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        log.error(f"Error loading filter prompt template: {str(e)}")
        return (
            "You are a marketing assistant scoring Reddit posts for product relevance.\n\n"
            "Analyze:\n"
            "- Is the post about automation, cron jobs, scheduling tasks, or running scripts?\n"
            "- Does the user express frustration, confusion, or a need for a solution?\n"
            "- How emotionally charged is the post?\n\n"
            "Respond with JSON including relevance_score, emotional_intensity, pain_point_clarity, and summary."
        )


FILTER_PROMPT_TEMPLATE = load_filter_prompt_template()


def build_filter_prompt(post: dict) -> List[Dict]:
    """Constructs the GPT-4.1 Mini prompt for a Reddit post using template."""
    return [
        {"role": "system", "content": "You are a marketing assistant scoring Reddit posts for product relevance."},
        {"role": "user", "content": f"Post title: {post['title']}\nPost body: {post['body']}\n\n{FILTER_PROMPT_TEMPLATE}"}
    ]


def prepare_batch_payload(posts: List[dict]) -> List[Dict]:
    """Returns list of payloads for batch submission."""
    payload = []
    for post in posts:
        raw_title = post.get("title", "")
        raw_body = post.get("body", "")
        title = sanitize_text(raw_title)
        body = sanitize_text(raw_body)

        if not title or not body:
            log.error("post is invalid, this should already have been asserted - skipping")
            continue  # skip malformed posts

        messages = build_filter_prompt({"title": title, "body": body})
        payload.append({
            "id": post["id"],
            "messages": messages,
            "meta": {
                "estimated_tokens": estimate_tokens(title + body, config["openai"].get("model_filter", "gpt-4o-mini"))
            }
        })
    return payload


def estimate_batch_cost(posts: List[dict], model: str = "gpt-4o-mini", avg_tokens: int = 300) -> float:
    """
    Estimate cost of a filtering batch using actual model pricing.
    """
    pricing = {
        "gpt-4.1": {"input": 0.0010},
        "gpt-4.1-mini": {"input": 0.0002},
        "gpt-4o-mini": {"input": 0.000075},
    }

    model_pricing = pricing.get(model, {"input": 0.0005})  # default fallback
    total_tokens = len(posts) * avg_tokens
    input_cost_per_1k = model_pricing["input"]

    return (total_tokens / 1000) * input_cost_per_1k