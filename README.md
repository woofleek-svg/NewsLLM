# NewsLLM

Automated AI news aggregator and alerting system for homelab deployment. Pulls articles from RSS feeds via Miniflux, summarizes and scores them using a local LLM (Qwen 3.5 via llama.cpp), and serves structured results through an MCP server for AI-assisted briefings and email delivery.

```
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Miniflux +    │      │  Python LLM      │      │  Output Postgres │
│   Postgres      │─────>│  Processor       │─────>│                  │
│   (Ingestion)   │ API  │                  │ SQL  │                  │
└─────────────────┘      └──────┬───────────┘      └────────┬─────────┘
                                │                           │
                                v                           v
                         ┌─────────────┐            ┌──────────────┐
                         │ Qwen 3.5    │            │  MCP Server  │
                         │ (llama.cpp) │            │  (HTTP)      │
                         └─────────────┘            └──────────────┘
```

## Features

- **RSS ingestion** via Miniflux with automatic polling, deduplication, and category support
- **LLM-powered analysis** — each article gets a 2-3 sentence summary, topic tags, named entity extraction, and a 3-tier urgency score (routine / notable / breaking)
- **MCP server** with progressive discovery tools — list sources, get briefings, search by keyword/tag, drill into articles
- **Email briefings** — styled HTML emails with article thumbnails, grouped by urgency, sent via Gmail SMTP
- **48-hour rolling window** — automatic purge keeps the database lightweight
- **Image extraction** — pulls article images from RSS enclosures or content HTML
- **Network isolation** — Miniflux DB is firewalled from everything except Miniflux itself

## Prerequisites

- Docker and Docker Compose
- A running [llama.cpp](https://github.com/ggerganov/llama.cpp) server with an OpenAI-compatible API (tested with Qwen 3.5 35B Q4)
- (Optional) Gmail account with an [app password](https://support.google.com/accounts/answer/185833) for email briefings

## Quick Start

```bash
# Clone the repo
git clone https://github.com/youruser/NewsLLM.git
cd NewsLLM

# Configure environment
cp .env.example .env
# Edit .env — set passwords, LLM endpoint, and optionally email credentials

# Start databases and Miniflux
docker compose up -d postgres-miniflux postgres-output miniflux

# Open Miniflux at http://localhost:8085
# - Log in with your MINIFLUX_ADMIN_USERNAME/PASSWORD
# - Add RSS feeds and organize into categories
# - Go to Settings > API Keys > generate a key
# - Paste the key into .env as MINIFLUX_API_KEY

# Start the processor and MCP server
docker compose up -d news-processor mcp-server
```

## Services

| Service | Port | Purpose |
|---|---|---|
| `miniflux` | 8085 | RSS reader UI and API |
| `postgres-miniflux` | — | Miniflux internal database |
| `postgres-output` | — | Processed articles database |
| `news-processor` | — | LLM processing pipeline |
| `mcp-server` | 8100 | MCP tool server (streamable-http) |

## MCP Tools

The MCP server exposes these tools for AI clients:

| Tool | Description |
|---|---|
| `list_sources()` | Discover available categories and feeds |
| `get_briefing(category, hours, limit)` | Recent articles sorted by urgency |
| `search_news(query, category, urgency_min)` | Keyword/tag search with relevance ranking |
| `get_article(article_id)` | Full details for a specific article |
| `get_breaking(hours)` | Urgency=3 alerts only |
| `get_stats()` | Pipeline health and article counts |
| `email_briefing(subject, category, hours, ...)` | Build and send an HTML briefing email |
| `send_email(subject, body)` | Send a short custom notification |

### Connecting an MCP Client

The server uses streamable-http transport at `http://<host>:8100/mcp`.

**Claude Desktop / Claude Code:**
```json
{
  "mcpServers": {
    "newsllm": {
      "url": "http://localhost:8100/mcp"
    }
  }
}
```

## Configuration

All configuration is via environment variables in `.env`. See [.env.example](.env.example) for the full list.

Key settings:

| Variable | Default | Description |
|---|---|---|
| `LLAMA_CPP_URL` | — | Your llama.cpp server's `/v1/chat/completions` endpoint |
| `LLAMA_MODEL` | `qwen3.5-35b` | Model identifier (for logging) |
| `POLL_INTERVAL` | `300` | Seconds between Miniflux polls |
| `MAX_CONTENT_LENGTH` | `64000` | Max article chars sent to the LLM |
| `SMTP_USER` / `SMTP_PASSWORD` | — | Gmail credentials for email briefings |
| `EMAIL_RECIPIENTS` | — | Comma-separated default email recipients |

## Urgency Scoring

The LLM scores each article on a 3-tier scale:

- **1 (Routine)** — Scheduled announcements, opinion pieces, feature stories, earnings reports
- **2 (Notable)** — Unexpected policy changes, major product launches, legal actions, leadership changes
- **3 (Breaking)** — Armed conflicts, natural disasters, government collapses, critical infrastructure failures

The prompt instructs the model to prefer lower scores when uncertain to avoid alert fatigue.

## Project Structure

```
NewsLLM/
├── docker-compose.yml        # All services with network isolation
├── .env.example              # Configuration template
├── init-output-db.sql        # Output database schema
├── processor/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py               # LLM processing pipeline
├── mcp-server/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── server.py             # MCP tool server + email
└── ai-news-aggregator-implementation-plan.md
```

## License

MIT
