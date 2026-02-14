# scheduler/cost_tracker.py

import json
import os
from datetime import datetime, timezone
from config.config_loader import get_config
from utils.logger import setup_logger
from utils.helpers import save_json, load_json, ensure_directory_exists

log = setup_logger()
config = get_config()

# Constants for pricing (adjust if OpenAI changes their pricing)
GPT4_MINI_INPUT_COST = 0.15  # $ per 1000 tokens
GPT4_MINI_OUTPUT_COST = 0.60
GPT4_INPUT_COST = 0.30
GPT4_OUTPUT_COST = 0.60

COST_TRACKING_FILE = "data/cost_tracking.json"

def get_current_month() -> str:
    """Returns the current UTC month as YYYY-MM."""
    return datetime.now(timezone.utc).strftime("%Y-%m")

def initialize_cost_tracking():
    """Initialize the cost tracking file if it doesn't exist."""
    ensure_directory_exists("data")
    if not os.path.exists(COST_TRACKING_FILE):
        data = {
            "monthly_costs": {},
            "current_month": get_current_month(),
            "current_month_total": 0.0,
            "monthly_budget": config["openai"]["monthly_budget_usd"]
        }
        save_json(data, COST_TRACKING_FILE)
    return load_json(COST_TRACKING_FILE)

def track_api_usage(input_tokens: int, output_tokens: int, model: str) -> float:
    """Track API usage and calculate costs."""
    tracking_data = load_json(COST_TRACKING_FILE)
    current_month = get_current_month()

    if tracking_data["current_month"] != current_month:
        tracking_data["current_month"] = current_month
        tracking_data["current_month_total"] = 0.0

    if "mini" in model.lower():
        input_cost = (input_tokens / 1000) * GPT4_MINI_INPUT_COST
        output_cost = (output_tokens / 1000) * GPT4_MINI_OUTPUT_COST
    else:
        input_cost = (input_tokens / 1000) * GPT4_INPUT_COST
        output_cost = (output_tokens / 1000) * GPT4_OUTPUT_COST

    total_cost = input_cost + output_cost

    tracking_data["monthly_costs"].setdefault(current_month, 0.0)
    tracking_data["monthly_costs"][current_month] += total_cost
    tracking_data["current_month_total"] += total_cost

    save_json(tracking_data, COST_TRACKING_FILE)
    log.info(f"Tracked usage: {input_tokens} input + {output_tokens} output tokens. Cost: ${total_cost:.4f}")

    return total_cost

def remaining_budget() -> float:
    """Calculate remaining budget for the current month."""
    tracking_data = load_json(COST_TRACKING_FILE)
    current_month = get_current_month()

    if tracking_data["current_month"] != current_month:
        tracking_data["current_month"] = current_month
        tracking_data["current_month_total"] = 0.0
        save_json(tracking_data, COST_TRACKING_FILE)

    budget = tracking_data.get("monthly_budget", config["openai"]["monthly_budget_usd"])
    spent = tracking_data.get("current_month_total", 0.0)
    return max(0.0, budget - spent)

def can_process_batch(estimated_cost: float) -> bool:
    """Determine if a batch can be processed within budget."""
    remaining = remaining_budget()
    if estimated_cost > remaining:
        log.warning(f"Budget limit reached. Estimated cost ${estimated_cost:.2f}, remaining ${remaining:.2f}")
    return estimated_cost <= remaining

def add_cost(estimated_cost: float):
    """Add estimated cost to the current month's total."""
    tracking_data = load_json(COST_TRACKING_FILE)
    current_month = get_current_month()

    if tracking_data["current_month"] != current_month:
        tracking_data["current_month"] = current_month
        tracking_data["current_month_total"] = 0.0

    tracking_data["monthly_costs"].setdefault(current_month, 0.0)
    tracking_data["monthly_costs"][current_month] += estimated_cost
    tracking_data["current_month_total"] += estimated_cost

    save_json(tracking_data, COST_TRACKING_FILE)
    log.info(f"Added estimated cost: ${estimated_cost:.4f}")
