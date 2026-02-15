[![Visit Cronlytic](https://img.shields.io/badge/Cronlytic.com-Explore%20Serverless%20Scheduling-blue)](https://www.cronlytic.com)

> Built to uncover hidden marketing signals on Reddit — and help power smarter growth for Cronlytic.com 🚀

[![Watch the Explainer Video](https://img.youtube.com/vi/UeMfjuDnE_0/0.jpg)](https://youtu.be/UeMfjuDnE_0)

> 📺 Click the thumbnail above to watch a full explainer — why I built this tool, how it works, and how you can use it to automate Reddit lead generation using GPT-4.

# Reddit Scraper
A Python application that scrapes Reddit for potential marketing leads, analyzes them with GPT models, and identifies high-value opportunities. Includes an interactive Streamlit dashboard for browsing and filtering results.

## 📑 Table of Contents
- [Overview](#-overview)
- [Setup](#-setup)
- [Configuration](#-configuration)
- [Running](#-running)
- [GUI Dashboard](#-gui-dashboard)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Cost Controls](#-cost-controls)
- [Contributors](#-contributors)
- [Why This Exists](#-why-this-exists)
- [License](#-license)
- [Third-Party Licenses](#-third-party-licenses)

## 📋 Overview

This tool uses a combination of Reddit's API and AI models (OpenAI or Anthropic) to:

1. Scrape relevant subreddits for discussions across diverse domains (tech, finance, parenting, fitness, business, and more)
2. Identify posts that express pain points with real product-building potential
3. Score and analyze posts using multi-dimensional metrics including technical depth, implementability, and emotional intensity
4. Store results in a local SQLite database for review
5. Browse and filter results through an interactive web dashboard

The application maintains a balance between focused and exploratory subreddits, intelligently refreshing the exploratory list based on discoveries. This exploration process happens automatically as part of the main workflow.

## 🚀 Setup

### Prerequisites

- Python 3.10+
- Reddit API credentials ([create an app here](https://www.reddit.com/prefs/apps))
- OpenAI API key **or** Anthropic API key (configurable via `config.yaml`)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Mohamedsaleh14/Reddit_Scrapper.git
   cd Reddit_Scrapper
   ```

2. Create a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables by copying `.env.template` to `.env`:
   ```
   cp .env.template .env
   ```

5. Edit `.env` and add your API credentials:
   ```
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=script:cronlytic-reddit-scraper:v1.0 (by /u/yourusername)
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## 🔧 Configuration

Configure the application by editing `config/config.yaml`. Key settings include:

- **AI provider**: Choose between `openai` or `anthropic` as the batch processing backend
- **Target subreddits**: Primary subreddits and exploratory subreddit settings
- **Post age range**: Only analyze posts within the configured age window
- **API rate limits**: Prevent hitting Reddit API limits
- **AI models**: Per-provider model configuration for filtering and deep analysis
- **Monthly budget**: Cap total API spending
- **Scoring weights**: How to weight different factors (relevance, pain point clarity, emotional intensity, implementability, technical depth) when scoring posts
- **Token limits**: Per-model enqueued token limits for batch API submissions

## 🏃‍♀️ Running

### One-time Run

To run the pipeline once:

```
python3 main.py
```

This will:
1. Scrape posts from configured primary subreddits
2. Automatically discover and scrape from exploratory subreddits
3. Analyze all posts with GPT models
4. Store results in the database

### Scheduled Operation

To run the pipeline daily at the configured time (TODO, Fix scheduler):

```
python3 scheduler/daily_scheduler.py
```

## 🖥️ GUI Dashboard

After running the pipeline at least once, you can explore the results using the interactive Streamlit dashboard:

```bash
./run_gui.sh
```

Or run it directly:

```bash
streamlit run gui/gui.py --server.port 8501 --server.address localhost
```

The dashboard provides:

- **Score filtering** — Adjust sliders for ROI, relevance, pain score, emotion score, implementability, and technical depth to focus on the posts that matter most
- **Subreddit filters** — Multi-select filters to narrow results by source subreddit
- **Sorting** — Sort by any score metric (including technical depth) or post date, ascending or descending
- **Pagination** — Browse through large result sets 10 posts at a time
- **Post cards** — Each post displays scores, pain point summary, product opportunity, technical depth, tags, and a link to the original Reddit thread
- **Expandable details** — Click into any post to read the body text, AI-generated justification, affected audience, business type, existing alternatives, build complexity, business model, and technical moat analysis
- **Summary statistics** — Sidebar shows total posts with average relevance, pain, emotion, and tech depth scores for the current filter

## 📊 Results

Results are stored in a SQLite database at `data/db.sqlite`. Besides the GUI, you can query it directly:

```sql
-- Today's top leads
SELECT * FROM posts
WHERE processed_at >= DATE('now')
ORDER BY roi_weight DESC, relevance_score DESC
LIMIT 10;

-- Posts with specific tag
SELECT * FROM posts
WHERE tags LIKE '%serverless%'
ORDER BY processed_at DESC;
```

## 📂 Project Structure

```
Reddit_Scrapper/
├── config/                  # Configuration files
│   ├── config.yaml          # Main configuration
│   └── config_loader.py     # Config + prompt loading
├── db/                      # Database interaction
│   ├── schema.py            # Table definitions
│   ├── reader.py            # Read queries
│   ├── writer.py            # Write operations
│   └── cleaner.py           # Old entry cleanup
├── gpt/                     # AI integration (OpenAI & Anthropic)
│   ├── batch_api.py         # OpenAI Batch API submission & polling
│   ├── anthropic_batch.py   # Anthropic Message Batches API integration
│   ├── batch_provider.py    # Provider routing layer (OpenAI/Anthropic)
│   ├── filters.py           # Pre-filtering prompt builder
│   ├── insights.py          # Deep insight prompt builder
│   └── prompts/             # Prompt templates
│       ├── filter.txt
│       ├── insight.txt
│       ├── community_discovery.txt
│       └── community_discovery_system.txt
├── gui/                     # Web dashboard
│   └── gui.py               # Streamlit application
├── reddit/                  # Reddit API interaction
│   ├── scraper.py           # Post & comment scraping
│   ├── discovery.py         # Exploratory subreddit discovery
│   └── rate_limiter.py      # API rate limiting
├── scheduler/               # Scheduling & cost tracking
│   ├── runner.py            # Main pipeline orchestration
│   └── cost_tracker.py      # Monthly budget tracking
├── utils/                   # Utility functions
│   ├── helpers.py           # Token estimation, sanitization
│   └── logger.py            # Logging setup
├── scripts/                 # Utility scripts
│   └── clean_openai_storage.py  # Clean accumulated OpenAI batch files
├── .env.template            # Template for environment variables
├── main.py                  # Application entry point
├── run_gui.sh               # GUI launcher script
└── requirements.txt         # Python dependencies
```

## 🔒 Cost Controls

The application includes several safeguards to control API costs:

- Monthly budget cap (configurable in `config.yaml`)
- Efficient batch processing using OpenAI's Batch API or Anthropic's Message Batches API
- Per-model enqueued token limits to avoid provider quota issues
- Automatic OpenAI storage cleanup (removes accumulated batch input/output files)
- Parallel batch submission with token-aware scheduling
- Partial result recovery from expired batches
- Pre-filtering with less expensive models before using more powerful models
- Cost tracking and logging

## Core Functionality
| Feature                                   | Status | Notes                                                 |
| ----------------------------------------- | ------ | ----------------------------------------------------- |
| **Reddit Scraping (Posts & Comments)**    | ✅ Done | Age-filtered, deduplicated, tracked via history table |
| **Primary & Exploratory Subreddit Logic** | ✅ Done | With refreshable `exploratory_subreddits.json`        |
| **GPT Filtering**                         | ✅ Done | Via batch API, scoring + threshold-based selection    |
| **GPT Insight Extraction**                | ✅ Done | With batch API, structured JSON, ROI + tags           |
| **SQLite Local DB Storage**               | ✅ Done | Full schema, type handling (`post`/`comment`)         |
| **Rate Limiting**                         | ✅ Done | Real limiter applied to avoid Reddit bans             |
| **Budget Control**                        | ✅ Done | Tracks monthly cost, blocks over-budget batches       |
| **Daily Runner Pipeline**                 | ✅ Done | Logs step-by-step, fail-safe batch handling           |
| **Anthropic Batch API Provider**          | ✅ Done | Full alternative to OpenAI with config-based switching |
| **Parallel Batch Processing**             | ✅ Done | Token-aware scheduling, partial result recovery       |
| **Technical Depth Scoring**               | ✅ Done | Measures engineering complexity and defensibility      |
| **Implementability Scoring**              | ✅ Done | Feasibility assessment with willingness-to-pay signals|
| **OpenAI Storage Cleanup**                | ✅ Done | Auto-cleans accumulated batch files before each run   |
| **Cached Summaries → GPT Discovery**      | ✅ Done | Based on post text, fallback if prompt fails          |
| **Comment scraping toggle**               | ✅ Done | Controlled via config key (`include_comments`)        |
| **Retry on GPT Batch Failures**           | ✅ Done | With exponential backoff and item-level retry         |
| **Streamlit GUI Dashboard**               | ✅ Done | Filter, sort, browse, and analyze results visually    |

## Future Improvements
| Feature                                   | Status                     | Suggestion                                   |
| ----------------------------------------- | -------------------------- | -------------------------------------------- |
| **Parallel subreddit fetching**           | 🟡 Manual (sequential)     | Consider async/threaded fetch in future      |
| **Tagged CSV Export / CLI**               | 🟡 Missing                 | Useful for non-technical review/debug        |
| **Multi-language / non-English handling** | 🟡 Not supported           | Detect & skip or flag for English-only use   |
| **Unit tests / mocks**                    | 🟡 Not present             | Add test coverage for scoring and DB logic   |

## 👥 Contributors

Thanks to the following people who have contributed to this project:

| Contributor | Contributions |
|-------------|--------------|
| [@Mohamedsaleh14](https://github.com/Mohamedsaleh14) | Creator & maintainer |
| [@Dieterbe](https://github.com/Dieterbe) | Bug fixes, prompt system refactoring, enhanced logging, GUI, batch optimization, and many quality-of-life improvements |
| [Claude Code](https://claude.ai/claude-code) | AI pair programmer — code implementation, issue triage, and PR integration |

## 🙏 Acknowledgements

- [PRAW (Python Reddit API Wrapper)](https://praw.readthedocs.io/)
- [OpenAI API](https://platform.openai.com/docs/api-reference)
- [Anthropic API](https://docs.anthropic.com/en/docs)
- [APScheduler](https://apscheduler.readthedocs.io/)
- [Streamlit](https://streamlit.io/)

## 🙋‍♂️ Why This Exists

This tool was created as part of the growth strategy for [**Cronlytic.com**](https://www.cronlytic.com) — a serverless cron job scheduler designed for developers, indie hackers, and SaaS teams.

If you're building something and want to:
- Run scheduled webhooks or background jobs
- Get reliable cron-like execution in the cloud
- Avoid over-engineering with full servers

👉 [**Check out Cronlytic**](https://www.cronlytic.com) — and let us know what you'd love to see.

## 📝 License

This project is open source for personal and non-commercial use only.
Commercial use (including hosting it as a backend or integrating into products) requires prior approval.

See the [LICENSE](LICENSE) file for full terms.

## 📄 Third-Party Licenses

This project uses open source libraries, which are governed by their own licenses:

- [PRAW](https://github.com/praw-dev/praw) — MIT License
- [APScheduler](https://github.com/agronholm/apscheduler) — MIT License
- [OpenAI Python SDK](https://github.com/openai/openai-python) — MIT License
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) — MIT License
- [Streamlit](https://github.com/streamlit/streamlit) — Apache License 2.0
- [Pandas](https://github.com/pandas-dev/pandas) — BSD 3-Clause License
- [Reddit API](https://www.reddit.com/dev/api/) — Subject to Reddit's [Terms of Service](https://www.redditinc.com/policies/data-api-terms)

Use of this project must also comply with these third-party licenses and terms.
