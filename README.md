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
- **Themed email briefings** — styled HTML emails with article thumbnails, grouped by urgency, with customizable themes per category (edit `themes.json` — no code changes needed)
- **48-hour rolling window** — automatic purge keeps the database lightweight
- **Image extraction** — pulls article images from RSS enclosures or content HTML
- **Network isolation** — Miniflux DB is firewalled from everything except Miniflux itself

## Prerequisites

- Docker and Docker Compose
- A running LLM server with an OpenAI-compatible API (tested with Qwen 3.5 35B Q4 via llama.cpp; also supports Ollama, vLLM, or any compatible endpoint)
- (Optional) SMTP credentials for email briefings (Gmail with [app password](https://support.google.com/accounts/answer/185833), or any SMTP server)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/woofleek-svg/NewsLLM.git
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
| `list_themes()` | List available email themes |
| `email_briefing(subject, category, theme, ...)` | Build and send a themed HTML briefing email |
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
| `LLM_BACKEND` | `llama.cpp` | Backend type: `llama.cpp`, `ollama`, `vllm`, or `generic` |
| `LLM_URL` | — | Chat completions endpoint (see examples below) |
| `LLM_MODEL` | `qwen3.5-35b` | Model name passed to the API |
| `LLM_API_KEY` | — | Bearer token (required for vLLM with auth) |
| `POLL_INTERVAL` | `300` | Seconds between Miniflux polls |
| `MAX_CONTENT_LENGTH` | `64000` | Max article chars sent to the LLM |
| `SMTP_HOST` | `smtp.gmail.com` | SMTP server hostname |
| `SMTP_PORT` | `587` | SMTP server port |
| `SMTP_USER` / `SMTP_PASSWORD` | — | SMTP credentials for email briefings |
| `EMAIL_RECIPIENTS` | — | Comma-separated default email recipients |

### LLM Backend Examples

**llama.cpp** (default):
```env
LLM_BACKEND=llama.cpp
LLM_URL=http://192.168.1.100:8080/v1/chat/completions
LLM_MODEL=qwen3.5-35b
```

**Ollama:**
```env
LLM_BACKEND=ollama
LLM_URL=http://192.168.1.100:11434/v1/chat/completions
LLM_MODEL=qwen3:32b
```

**vLLM:**
```env
LLM_BACKEND=vllm
LLM_URL=http://192.168.1.100:8000/v1/chat/completions
LLM_MODEL=Qwen/Qwen3.5-32B-AWQ
LLM_API_KEY=your-api-key
```

**Generic** (any OpenAI-compatible API):
```env
LLM_BACKEND=generic
LLM_URL=https://api.example.com/v1/chat/completions
LLM_MODEL=your-model
LLM_API_KEY=your-key
```

## Urgency Scoring

The LLM scores each article on a 3-tier scale:

- **1 (Routine)** — Scheduled announcements, opinion pieces, feature stories, earnings reports
- **2 (Notable)** — Unexpected policy changes, major product launches, legal actions, leadership changes
- **3 (Breaking)** — Armed conflicts, natural disasters, government collapses, critical infrastructure failures

The prompt instructs the model to prefer lower scores when uncertain to avoid alert fatigue.

## Email Themes

Email briefings support visual themes that can be customized per category. Themes are defined in [`mcp-server/themes.json`](mcp-server/themes.json) and hot-reloaded on each send — no rebuild required.

Built-in themes: `default`, `cleveland`, `chicago`, `tech`, `national`

To add a custom theme, edit `themes.json` and add a new entry:

```json
{
  "my-city": {
    "name": "My City Briefing",
    "tagline": "Local News Digest",
    "font": "Georgia, serif",
    "bg_outer": "#1a1a2e",
    "header_from": "#16213e",
    "header_to": "#0f3460",
    "header_text": "#ffffff",
    "header_sub": "#a8b2d1",
    "badge_bg": "#e94560",
    "accent": "#e94560",
    "tag_bg": "#fce4ec",
    "tag_text": "#880e4f",
    "footer_bg": "#f5f5f5"
  }
}
```

Then use it: `email_briefing(subject="...", category="My City", theme="my-city")`

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
│   ├── server.py             # MCP tool server + email
│   └── themes.json           # Email theme definitions (hot-reloaded)
└── ai-news-aggregator-implementation-plan.md
```

## License

MIT
