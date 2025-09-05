# reddit/scraper.py

import praw
import datetime
import os
import socket
import time

from prawcore.exceptions import RequestException
from db.reader import is_already_processed
from db.writer import insert_post
from reddit.discovery import discover_adjacent_subreddits
from config.config_loader import get_config
from utils.logger import setup_logger
from utils.helpers import load_json, save_json, truncate
from reddit.rate_limiter import RedditRateLimiter

socket.setdefaulttimeout(10)  # Set global 10s timeout for HTTP

log = setup_logger()
config = get_config()

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=config["reddit"]["client_id"],
    client_secret=config["reddit"]["client_secret"],
    user_agent=config["reddit"]["user_agent"],
    username=config["reddit"]["username"],
    password=config["reddit"]["password"]
)

limiter = RedditRateLimiter(config["scraper"].get("rate_limit_per_minute", 60))
EXPLORATORY_FILE = "data/exploratory_subreddits.json"

def is_post_in_age_range(post, min_days, max_days) -> bool:
    post_date = datetime.datetime.fromtimestamp(post.created_utc)
    age_days = (datetime.datetime.utcnow() - post_date).days
    return min_days <= age_days <= max_days

def fetch_posts_from_subreddit(subreddit_name, limit=200) -> list:
    min_days = config["scraper"]["min_post_age_days"]
    max_days = config["scraper"]["max_post_age_days"]
    include_comments = config["scraper"].get("include_comments", False)
    results = []
    seen_ids = set()

    skipped_due_to_seen = 0
    skipped_due_to_age = 0
    skipped_due_to_duplicate = 0
    total_comments = 0
    comment_skipped_due_to_seen = 0
    comment_skipped_due_to_age = 0
    comment_skipped_due_to_duplicate = 0

    try:
        log.info(f"Fetching posts from r/{subreddit_name} using top, hot, and new...")
        subreddit = reddit.subreddit(subreddit_name)
        combined = []

        for fetch_name, fetch_method in [("top", subreddit.top(time_filter="month", limit=limit)),
                                         ("hot", subreddit.hot(limit=limit)),
                                         ("new", subreddit.new(limit=limit))]:
            limiter.wait()  # Apply rate limit per API fetch
            posts = safe_fetch(fetch_method, fetch_name)
            combined.extend(posts)

        log.info(f"Total fetched posts to process from r/{subreddit_name}: {len(combined)}")
        start_time = time.time()

        for i, post in enumerate(combined):
            if i % 10 == 0:
                log.info(f"Processing post #{i+1}/{len(combined)}")

            if post.id in seen_ids:
                skipped_due_to_seen += 1
                continue
            seen_ids.add(post.id)

            created_at = datetime.datetime.fromtimestamp(post.created_utc)
            log.debug(f"Post {post.id} at {created_at.isoformat()} — {post.title[:60]}")

            if not is_post_in_age_range(post, min_days, max_days):
                skipped_due_to_age += 1
                continue
            if is_already_processed(post.id):
                skipped_due_to_duplicate += 1
                continue

            results.append({
                "id": post.id,
                "title": post.title,
                "body": post.selftext,
                "created_utc": post.created_utc,
                "subreddit": subreddit_name,
                "url": f"https://www.reddit.com{post.permalink}",
                "type": "post"
            })

            if include_comments:
                try:
                    limiter.wait()  # One API call to fetch all comments
                    post.comments.replace_more(limit=0)
                    comments_list = post.comments.list()
                    total_comments += len(comments_list)
                    for comment in comments_list:
                        if comment.id in seen_ids:
                            comment_skipped_due_to_seen += 1
                            continue
                        seen_ids.add(comment.id)

                        if not is_post_in_age_range(comment, min_days, max_days):
                            comment_skipped_due_to_age += 1
                            continue
                        if is_already_processed(comment.id):
                            comment_skipped_due_to_duplicate += 1
                            continue

                        results.append({
                            "id": comment.id,
                            "title": post.title,
                            "body": comment.body,
                            "created_utc": comment.created_utc,
                            "subreddit": subreddit_name,
                            "url": f"https://www.reddit.com{comment.permalink}",
                            "type": "comment"
                        })
                except Exception as e:
                    log.warning(f"Failed to fetch comments for post {post.id}: {str(e)}")

        log.info(f"processed {len(combined)} posts and {total_comments} comments")
        log.info(
            f"r/{subreddit_name} — Found {len(results)} new items | "
            f"Skipped posts: {skipped_due_to_seen} (due to seen) {skipped_due_to_age} (due to age) | {skipped_due_to_duplicate} (duplicates)"
        )
        log.info(f"Skipped comments: {comment_skipped_due_to_seen} (due to seen) {comment_skipped_due_to_age} (due to age) | {comment_skipped_due_to_duplicate} (duplicates)")
    except Exception as e:
        log.error(f"Error fetching from r/{subreddit_name}: {str(e)}")

    log.info(f"Finished processing posts from r/{subreddit_name} in {time.time() - start_time:.2f} seconds")
    return results

def get_exploratory_subreddits():
    if not os.path.exists(EXPLORATORY_FILE):
        return []

    data = load_json(EXPLORATORY_FILE)
    last_updated = data.get("last_updated", "")
    refresh_days = config["subreddits"]["exploratory_refresh_days"]

    if not last_updated or (datetime.datetime.utcnow() - datetime.datetime.fromisoformat(last_updated)).days >= refresh_days:
        log.info("Exploratory subreddit list needs refresh")
        return []

    return data.get("subreddits", [])


def update_exploratory_subreddits(new_subreddits):
    data = {
        "last_updated": datetime.datetime.utcnow().isoformat(),
        "subreddits": new_subreddits
    }
    os.makedirs("data", exist_ok=True)
    save_json(data, EXPLORATORY_FILE)
    log.info(f"Updated exploratory subreddits: {', '.join(new_subreddits)}")

def scrape_all_configured_subreddits() -> list:
    primary_subreddits = config["subreddits"]["primary"]
    primary_pct = config["subreddits"]["primary_percentage"]
    exploratory_pct = config["subreddits"]["exploratory_percentage"]
    exploratory_limit = config["subreddits"]["exploratory_limit"]

    total_limit = config["scraper"]["max_items_per_day"]
    primary_limit = int((primary_pct / 100) * total_limit)
    exploratory_limit_posts = int((exploratory_pct / 100) * total_limit)

    log.info(f"Scraping {len(primary_subreddits)} primary subreddits...")
    per_primary_subreddit = max(1, primary_limit // len(primary_subreddits))
    primary_posts = []

    for sub in primary_subreddits:
        posts = fetch_posts_from_subreddit(sub, limit=per_primary_subreddit)
        for post in posts:
            insert_post(post, community_type="primary")
        primary_posts.extend(posts)

    exploratory_subreddits = get_exploratory_subreddits()

    if not exploratory_subreddits:
        if primary_posts:
            log.info("Discovering new exploratory subreddits...")
            summaries = [truncate(f"{p['title']} {p['body']}", 300) for p in primary_posts[:10]]
            suggestions = discover_adjacent_subreddits(summaries)
            exploratory_subreddits = [s["subreddit"] for s in suggestions][:exploratory_limit]
            update_exploratory_subreddits(exploratory_subreddits)
        else:
            log.warning("No primary posts found to discover exploratory subreddits")

    if exploratory_subreddits:
        log.info(f"Scraping {len(exploratory_subreddits)} exploratory subreddits...")
        per_exploratory = max(1, exploratory_limit_posts // len(exploratory_subreddits))

        for sub in exploratory_subreddits:
            posts = fetch_posts_from_subreddit(sub, limit=per_exploratory)
            for post in posts:
                insert_post(post, community_type="exploratory")
            primary_posts.extend(posts)

    log.info(f"Total items scraped: {len(primary_posts)}")
    return primary_posts

def safe_fetch(generator, name):
    try:
        log.info(f"→ Fetching {name}...")
        return list(generator)
    except (RequestException, socket.timeout) as e:
        log.error(f"Timeout/error while fetching {name}: {e}")
        return []
    except Exception as e:
        log.error(f"Unknown error while fetching {name}: {e}")
        return []
