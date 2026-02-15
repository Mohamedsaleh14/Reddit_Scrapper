# reddit/discovery.py
import json
import os
import re
from config.config_loader import get_config
from utils.logger import setup_logger
from config.config_loader import PROMPT_COMMUNITY_DISCOVERY, PROMPT_COMMUNITY_DISCOVERY_SYSTEM

log = setup_logger()
config = get_config()


def _get_client():
    """Return the appropriate API client based on the configured provider."""
    provider = config["ai"]["provider"]
    if provider == "anthropic":
        import anthropic
        api_key = config["ai"]["anthropic"].get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        return anthropic.Anthropic(api_key=api_key), "anthropic"
    else:
        from openai import OpenAI
        return OpenAI(), "openai"


def _extract_json_block(text: str) -> str:
    """Extract a likely JSON block from model text with optional markdown fences."""
    if not text:
        return text

    # Handle fenced output with or without closing fence.
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]  # Drop opening fence line (e.g. ```json)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    # First try strict fenced extraction (for mixed text + fenced JSON).
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: capture first top-level array/object-like span.
    match = re.search(r"[\{\[].*[\}\]]", stripped, re.DOTALL)
    if match:
        return match.group(0).strip()

    return stripped


def build_discovery_prompt(top_post_summaries: list[str]) -> list:
    """Builds the prompt for discovering adjacent subreddits using template."""
    joined = "\n".join(f"- {summary}" for summary in top_post_summaries)

    return [
        {
            "role": "system",
            "content": config['prompts'][PROMPT_COMMUNITY_DISCOVERY_SYSTEM],
        },
        {
            "role": "user",
            "content": config['prompts'][PROMPT_COMMUNITY_DISCOVERY].replace("{SUMMARIES}", joined),
        }
    ]

def discover_adjacent_subreddits(summaries: list[str], model: str = None) -> list:
    """Uses the configured AI provider to suggest exploratory subreddits."""
    provider = config["ai"]["provider"]
    model = model or config["ai"][provider].get("model_deep", "gpt-4.1")
    log.info(f"Running discovery with {len(summaries)} post summaries using model: {model}")

    prompt = build_discovery_prompt(summaries)
    client, client_type = _get_client()

    try:
        if client_type == "anthropic":
            # Extract system message for Anthropic's separate system parameter
            system_content = None
            user_messages = []
            for msg in prompt:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    user_messages.append(msg)

            max_tokens = config["ai"].get("discovery_max_tokens", 4096)
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": user_messages,
            }
            if system_content:
                kwargs["system"] = system_content

            response = client.messages.create(**kwargs)
            content = response.content[0].text
        else:
            max_tokens = config["ai"].get("discovery_max_tokens", 4096)
            kwargs = {
                "model": model,
                "messages": prompt,
            }
            # gpt-5+ requires max_completion_tokens instead of max_tokens
            if model.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["temperature"] = 0.3
                kwargs["max_tokens"] = max_tokens
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

        log.debug(f"Discovery API response: {content[:200]}...")
        content = _extract_json_block(content)

        suggestions = json.loads(content)
        if not isinstance(suggestions, list):
            log.error("GPT response is not a list. Skipping.")
            return []

        valid_suggestions = [
            item for item in suggestions
            if isinstance(item, dict) and "subreddit" in item
        ]

        # Remove already-known subreddits
        primary_set = {sub.lower() for sub in config["subreddits"]["primary"]}
        filtered = [s for s in valid_suggestions if s["subreddit"].lower() not in primary_set]

        if not filtered:
            log.warning("No valid *new* subreddit suggestions found in GPT response.")
            return []

        return filtered[:config["subreddits"]["exploratory_limit"]]

    except json.JSONDecodeError as je:
        log.error(f"Failed to parse discovery response as JSON: {str(je)}")
        log.debug(f"Discovery response after JSON extraction: {content[:500]}")
        return []
    except Exception as e:
        log.error(f"Error in discovery process: {str(e)}")
        return []
