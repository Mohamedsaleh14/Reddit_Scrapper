import yaml
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file

CONFIG_PATH = "config/config.yaml"
PROMPT_PATH = "gpt/prompts"

# Prompt template constants
PROMPT_FILTER = "filter"
PROMPT_INSIGHT = "insight"
PROMPT_COMMUNITY_DISCOVERY = "community_discovery"
PROMPT_COMMUNITY_DISCOVERY_SYSTEM = "community_discovery_system"
PROMPTS_ALL = [
    PROMPT_FILTER, PROMPT_INSIGHT, PROMPT_COMMUNITY_DISCOVERY, PROMPT_COMMUNITY_DISCOVERY_SYSTEM
]

def get_config():
    with open(CONFIG_PATH, "r") as f:
        raw_config = yaml.safe_load(f)

    # Inject secrets from .env
    raw_config["reddit"] = {
        "client_id": os.getenv("REDDIT_CLIENT_ID"),
        "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
        "user_agent": os.getenv("REDDIT_USER_AGENT"),
        "username": os.getenv("REDDIT_USERNAME"),      # Added
        "password": os.getenv("REDDIT_PASSWORD"),      # Added
    }

    raw_config["ai"]["openai"]["api_key"] = os.getenv("OPENAI_API_KEY")
    raw_config["ai"]["anthropic"]["api_key"] = os.getenv("ANTHROPIC_API_KEY")

    # Backward compatibility: expose provider-specific config at top level
    # so existing code using config["openai"] still works during migration
    provider = raw_config["ai"]["provider"]
    provider_config = raw_config["ai"][provider]
    raw_config["openai"] = {**raw_config["ai"]["openai"], **{
        k: v for k, v in raw_config["ai"].items()
        if k not in ("provider", "openai", "anthropic")
    }}

    # Inject prompts from files
    raw_config["prompts"] = load_all_prompts()

    return raw_config


def get_provider(config=None):
    """Returns the active AI provider: 'openai' or 'anthropic'."""
    if config is None:
        config = get_config()
    return config["ai"]["provider"]


def get_model(stage, config=None):
    """Returns the model name for a given stage ('filter' or 'deep') based on active provider."""
    if config is None:
        config = get_config()
    provider = config["ai"]["provider"]
    key = f"model_{stage}"
    return config["ai"][provider][key]


def load_all_prompts() -> dict:
    """Load all prompt templates from the prompts directory and return as a dictionary."""
    prompts = {}

    for key in PROMPTS_ALL:
        filename = f"{key}.txt"
        path = os.path.join(PROMPT_PATH, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                prompts[key] = f.read().strip()
        except Exception as e:
            raise Exception(f"Error loading {key} prompt template from {path}: {str(e)}") from None
    return prompts
