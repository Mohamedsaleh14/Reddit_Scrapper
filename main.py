# main.py

import os
import sys
from scheduler.runner import run_daily_pipeline
from utils.logger import setup_logger
from db.schema import create_tables
from scheduler.cost_tracker import initialize_cost_tracking
from utils.helpers import ensure_directory_exists
from gpt.batch_api import RESPONSE_DIR

def setup_environment():
    """Initialize the environment for the application."""
    log = setup_logger()
    log.info("Setting up environment...")

    ensure_directory_exists("data")
    ensure_directory_exists(RESPONSE_DIR)
    ensure_directory_exists("logs")

    try:
        create_tables()
        log.info("Database initialized.")
    except Exception as e:
        log.error(f"Failed to initialize database: {str(e)}")
        return False

    try:
        initialize_cost_tracking()
        log.info("Cost tracking initialized.")
    except Exception as e:
        log.error(f"Failed to initialize cost tracking: {str(e)}")
        return False

    return True


if __name__ == "__main__":
    log = setup_logger()
    log.info("\U0001F680 Starting Cronlytic Reddit scraping pipeline...")

    if not setup_environment():
        log.error("Failed to set up environment. Exiting.")
        sys.exit(1)

    try:
        run_daily_pipeline()
        log.info("\u2705 Pipeline completed successfully.")
    except KeyboardInterrupt:
        log.info("Pipeline interrupted by user.")
    except Exception as e:
        log.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)
