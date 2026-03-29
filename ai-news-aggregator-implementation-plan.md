# AI News Aggregator — Implementation Plan

**Project:** Automated AI News Aggregator & Alerting Agent
**Date:** March 28, 2026
**Status:** Ready for Development

---

## 1. Architecture Overview

The system consists of four runtime components, all containerized via Docker Compose on the homelab:

```
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Miniflux +    │      │  Python LLM      │      │  Output Postgres │
│   Postgres #1   │─────▶│  Processing      │─────▶│  (MCP Resource)  │
│   (Ingestion)   │ API  │  Script          │ SQL  │  Postgres #2     │
└─────────────────┘      └──────┬───────────┘      └────────┬─────────┘
                                │                           │
                                ▼                           ▼
                         ┌─────────────┐            ┌──────────────┐
                         │ Qwen 3.5    │            │  MCP Server  │
                         │ (llama.cpp) │            │  (Phase 3)   │
                         └─────────────┘            └──────────────┘
```

**Key principle:** Miniflux and its Postgres instance are treated as a black box. The Python script reads from Miniflux via its REST API and writes structured output to a completely separate Postgres instance. The MCP server (Phase 3) only ever touches the output database.

---

## 2. Component 1 — Miniflux + Postgres (Ingestion)

### Purpose
Handle all RSS feed polling, HTML sanitization, deduplication, and raw article storage.

### Implementation Details

**Docker services:**
- `miniflux` — latest Miniflux image, exposed on a local port (e.g., `8085`)
- `postgres-miniflux` — Postgres 16, dedicated volume, internal network only

**Configuration decisions the team needs to make:**
- `POLLING_FREQUENCY` — How often Miniflux checks feeds. Recommend `30` (minutes) for a curated list of under 50 feeds. Can go as low as `10` for breaking news sources.
- `BATCH_SIZE` — Number of feeds refreshed per scheduler tick. Default `10` is fine for a small feed list.
- `CLEANUP_ARCHIVE_UNREAD_DAYS` / `CLEANUP_ARCHIVE_READ_DAYS` — Miniflux's own retention. Set these to `7` so Miniflux keeps articles slightly longer than our 48-hour output window. This provides a buffer if the LLM processing script falls behind.
- `FETCH_ODYSEE_WATCH_TIME` and `FETCH_YOUTUBE_WATCH_TIME` — Disable unless video feeds are in scope.

**Feed curation:**
- Maintain the feed list via Miniflux's web UI or OPML import.
- Organize feeds into Miniflux categories (e.g., `tech`, `geopolitics`, `markets`) — the processing script can read these categories and pass them as context to the LLM.
- Start with no more than 20–30 feeds. Each feed generates LLM inference load; keep it manageable during Phase 1–2.

**API access:**
- Generate an API key via Miniflux UI → Settings → API Keys.
- Store the key as an environment variable (`MINIFLUX_API_KEY`) available to the processing script.
- Base URL will be `http://miniflux:8080/v1` on the Docker internal network.

### Acceptance Criteria
- [ ] Miniflux + Postgres running in Docker Compose
- [ ] At least 5 test feeds added and polling successfully
- [ ] API key generated and confirmed working via `curl` (`GET /v1/entries?status=unread`)
- [ ] Categories assigned to feeds

---

## 3. Component 2 — Output Postgres (MCP Resource Database)

### Purpose
Store structured LLM output in a clean, purpose-built schema that the MCP server can query directly. This database is completely independent of Miniflux.

### Schema Definition

```sql
-- Core output table
CREATE TABLE processed_articles (
    id              SERIAL PRIMARY KEY,
    miniflux_id     BIGINT UNIQUE NOT NULL,       -- Miniflux entry ID (for dedup/tracking)
    source_feed     VARCHAR(255) NOT NULL,         -- Feed name or URL
    category        VARCHAR(100),                  -- Miniflux category (tech, markets, etc.)
    original_title  TEXT NOT NULL,
    original_url    TEXT NOT NULL,
    published_at    TIMESTAMPTZ NOT NULL,           -- Original publish time from feed
    processed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- LLM-generated fields
    summary         TEXT NOT NULL,                  -- 2-3 sentence summary
    tags            TEXT[] NOT NULL,                -- Array of topic tags
    entities        JSONB NOT NULL DEFAULT '[]',    -- Named entities [{name, type}]
    urgency_score   SMALLINT NOT NULL CHECK (urgency_score BETWEEN 1 AND 3),
    -- 1 = routine, 2 = notable, 3 = breaking

    -- Processing metadata
    model_used      VARCHAR(100) NOT NULL,          -- e.g., "qwen3.5-35b"
    processing_ms   INTEGER,                        -- Inference time in milliseconds
    raw_llm_output  JSONB                           -- Full LLM response for debugging
);

-- Index for the morning briefing query (recent articles, sorted by urgency)
CREATE INDEX idx_briefing ON processed_articles (processed_at DESC, urgency_score DESC);

-- Index for keyword/tag search
CREATE INDEX idx_tags ON processed_articles USING GIN (tags);

-- Index for entity search
CREATE INDEX idx_entities ON processed_articles USING GIN (entities);

-- Index for the 48-hour purge job
CREATE INDEX idx_purge ON processed_articles (processed_at);

-- Dead letter table for failed LLM processing
CREATE TABLE failed_articles (
    id              SERIAL PRIMARY KEY,
    miniflux_id     BIGINT UNIQUE NOT NULL,
    original_title  TEXT,
    original_url    TEXT,
    error_message   TEXT NOT NULL,
    raw_llm_output  TEXT,                           -- Whatever the LLM returned (may be malformed)
    failed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    retry_count     SMALLINT NOT NULL DEFAULT 0
);
```

### Design Decisions

**Urgency scoring uses a 3-tier scale, not 1–10.**
A 35B model will produce much more consistent results with `1 = routine`, `2 = notable`, `3 = breaking` than with a 10-point numeric scale. The two-tier routing from the scope doc maps cleanly: urgency 3 triggers alerts, urgency 1–2 goes into the passive briefing.

**Entities stored as JSONB, not a separate table.**
For a 48-hour rolling window, the data volume is small enough that JSONB with a GIN index is simpler and faster than a normalized entity table. If the system scales significantly, revisit this.

**Dead letter table is required.**
LLM output will occasionally be malformed. Rather than losing the article, log it with the error. A separate retry script can re-process these on a slower cadence.

### Acceptance Criteria
- [ ] Postgres instance running in Docker Compose, separate from Miniflux's Postgres
- [ ] Schema applied via migration script or init SQL
- [ ] Confirmed accessible from the processing script container
- [ ] Confirmed accessible from the future MCP server container (network-level)

---

## 4. Component 3 — Python LLM Processing Script

### Purpose
Bridge between Miniflux and the output database. Pulls unprocessed articles, sends them to Qwen 3.5 via llama.cpp, parses the structured JSON response, and writes to the output Postgres.

### Processing Flow

```
1. Query Miniflux API for unread entries (GET /v1/entries?status=unread&limit=50)
2. For each entry:
   a. Check if miniflux_id already exists in processed_articles (skip if so)
   b. Build the LLM prompt (see below)
   c. POST to llama.cpp /v1/chat/completions endpoint
   d. Parse JSON response
   e. If valid → INSERT into processed_articles
   f. If malformed → retry once with stricter prompt
   g. If still malformed → INSERT into failed_articles
   h. Mark entry as read in Miniflux (PUT /v1/entries)
3. Sleep for POLL_INTERVAL seconds, repeat
```

### LLM Prompt Specification

The system prompt sent to Qwen 3.5 must enforce structured output. Recommended prompt:

```
You are a news analysis assistant. You will receive a news article and must respond with ONLY a valid JSON object — no markdown, no explanation, no preamble.

The JSON must contain exactly these fields:

{
  "summary": "A 2-3 sentence summary of the article's key facts. No opinions.",
  "tags": ["tag1", "tag2", "tag3"],
  "entities": [
    {"name": "Entity Name", "type": "person|org|location|event|product"}
  ],
  "urgency_score": 1
}

Urgency scoring rules:
- 1 (routine): Scheduled announcements, opinion pieces, feature stories, product reviews, industry trends, earnings reports meeting expectations.
- 2 (notable): Unexpected policy changes, significant leadership changes, major product launches, surprising data releases, legal actions against major entities.
- 3 (breaking): Events with immediate widespread impact — armed conflicts, natural disasters, major government collapses, critical infrastructure failures, pandemic-level health emergencies, assassination or death of a head of state.

If in doubt between two levels, choose the LOWER one. Alert fatigue is worse than a missed notification.

Respond with ONLY the JSON object.
```

The user message should contain:

```
Category: {miniflux_category}
Title: {article_title}
Source: {feed_name}
Content: {article_content_truncated_to_4000_chars}
```

### Configuration (Environment Variables)

| Variable | Purpose | Suggested Default |
|---|---|---|
| `MINIFLUX_URL` | Miniflux API base URL | `http://miniflux:8080/v1` |
| `MINIFLUX_API_KEY` | API key for Miniflux | (generated in Miniflux UI) |
| `LLAMA_CPP_URL` | llama.cpp completions endpoint | `http://<homelab-ip>:8080/v1/chat/completions` |
| `LLAMA_MODEL` | Model identifier for logging | `qwen3.5-35b` |
| `OUTPUT_DB_URL` | Postgres connection string for output DB | `postgresql://user:pass@postgres-output:5432/news_output` |
| `POLL_INTERVAL` | Seconds between Miniflux API polls | `300` (5 minutes) |
| `MAX_CONTENT_LENGTH` | Truncate article content to N chars | `4000` |
| `MAX_RETRIES` | Retries on malformed LLM output | `1` |

### Dependencies

```
feedparser        # Not needed — Miniflux handles RSS parsing
requests          # HTTP calls to Miniflux API and llama.cpp
psycopg2-binary   # Postgres connection
```

This is intentionally minimal. No frameworks, no async libraries, no task queues. The script runs as a single long-lived process with a sleep loop. If throughput becomes a bottleneck, the team can add threading or switch to `asyncio` + `aiohttp` later.

### Error Handling

| Failure Mode | Handling |
|---|---|
| Miniflux API unreachable | Log warning, sleep, retry next cycle |
| llama.cpp server unreachable | Log error, skip article, retry next cycle |
| LLM returns malformed JSON | Retry once with stricter prompt. If still bad, write to `failed_articles` |
| LLM returns valid JSON with missing fields | Write to `failed_articles` with descriptive error |
| Output Postgres unreachable | Log critical error, halt processing (don't lose data silently) |
| Duplicate `miniflux_id` | Skip silently (already processed) |

### Acceptance Criteria
- [ ] Script connects to Miniflux API and retrieves unread entries
- [ ] Script sends articles to llama.cpp and receives structured JSON
- [ ] Valid responses written to `processed_articles` table
- [ ] Malformed responses written to `failed_articles` table
- [ ] Entries marked as read in Miniflux after processing
- [ ] Script runs continuously with configurable poll interval
- [ ] All config via environment variables

---

## 5. Component 4 — 48-Hour Purge Job

### Purpose
Delete all records from the output database older than 48 hours to maintain relevance and keep the database lightweight for MCP queries.

### Implementation

A cron job or pg_cron task that runs every hour:

```sql
DELETE FROM processed_articles WHERE processed_at < NOW() - INTERVAL '48 hours';
DELETE FROM failed_articles WHERE failed_at < NOW() - INTERVAL '48 hours';
```

### Implementation Options (pick one)

1. **pg_cron extension** — Install in the output Postgres container. Schedule the DELETE as a pg_cron job. Cleanest approach, no external dependency.
2. **Host cron + psql** — A cron entry on the host that runs `psql -c "..."` against the output database.
3. **In the Python script** — Run the purge at the start of each processing cycle. Simplest but couples concerns.

**Recommendation:** Option 1 (pg_cron) if the team is comfortable with Postgres extensions. Option 2 otherwise.

### Acceptance Criteria
- [ ] Records older than 48 hours are automatically deleted
- [ ] Purge runs at least once per hour
- [ ] Purge does not lock the table in a way that blocks MCP queries

---

## 6. Docker Compose Structure

```yaml
services:
  # --- Ingestion Layer ---
  postgres-miniflux:
    image: postgres:16
    volumes:
      - miniflux_data:/var/lib/postgresql/data
    environment:
      POSTGRES_USER: miniflux
      POSTGRES_PASSWORD: <generate>
      POSTGRES_DB: miniflux
    networks:
      - ingestion

  miniflux:
    image: miniflux/miniflux:latest
    depends_on:
      - postgres-miniflux
    environment:
      DATABASE_URL: postgres://miniflux:<password>@postgres-miniflux/miniflux?sslmode=disable
      RUN_MIGRATIONS: 1
      CREATE_ADMIN: 1
      ADMIN_USERNAME: admin
      ADMIN_PASSWORD: <generate>
      POLLING_FREQUENCY: 30
      CLEANUP_ARCHIVE_UNREAD_DAYS: 7
      CLEANUP_ARCHIVE_READ_DAYS: 7
    ports:
      - "8085:8080"
    networks:
      - ingestion
      - processing

  # --- Output Layer ---
  postgres-output:
    image: postgres:16
    volumes:
      - output_data:/var/lib/postgresql/data
      - ./init-output-db.sql:/docker-entrypoint-initdb.d/init.sql
    environment:
      POSTGRES_USER: newsagent
      POSTGRES_PASSWORD: <generate>
      POSTGRES_DB: news_output
    networks:
      - processing
      - mcp

  # --- Processing Layer ---
  news-processor:
    build: ./processor
    depends_on:
      - miniflux
      - postgres-output
    environment:
      MINIFLUX_URL: http://miniflux:8080/v1
      MINIFLUX_API_KEY: <generated-after-first-boot>
      LLAMA_CPP_URL: http://<homelab-ip>:8080/v1/chat/completions
      LLAMA_MODEL: qwen3.5-35b
      OUTPUT_DB_URL: postgresql://newsagent:<password>@postgres-output:5432/news_output
      POLL_INTERVAL: 300
    networks:
      - processing

volumes:
  miniflux_data:
  output_data:

networks:
  ingestion:    # Miniflux <-> its Postgres
  processing:   # Processor <-> Miniflux API + Output Postgres
  mcp:          # Output Postgres <-> MCP Server (Phase 3)
```

### Network Isolation Notes
- `postgres-miniflux` is only on the `ingestion` network. The processing script cannot access it directly — it must go through the Miniflux API.
- `postgres-output` is on both `processing` and `mcp` networks, so the future MCP server can reach it without touching the ingestion layer.
- The llama.cpp server is external to this Compose stack (already running on the homelab). The processor reaches it via the host network IP.

---

## 7. Startup Sequence

1. `docker compose up -d postgres-miniflux postgres-output` — Start both databases
2. `docker compose up -d miniflux` — Start Miniflux, run migrations
3. Log into Miniflux UI at `http://<homelab-ip>:8085`, add feeds, generate API key
4. Set `MINIFLUX_API_KEY` in the `.env` file or Docker Compose
5. `docker compose up -d news-processor` — Start the processing script
6. Verify articles are appearing in the `processed_articles` table

---

## 8. Phase 3 Preview — MCP Server Connection Point

The output Postgres is designed to be directly queryable by an MCP server. The three tools defined in the scope doc map to these queries:

| MCP Tool | SQL Query |
|---|---|
| `get_morning_briefing()` | `SELECT * FROM processed_articles WHERE processed_at > NOW() - INTERVAL '24 hours' ORDER BY urgency_score DESC, processed_at DESC LIMIT 20` |
| `search_recent_news(keyword)` | `SELECT * FROM processed_articles WHERE :keyword = ANY(tags) OR summary ILIKE '%' \|\| :keyword \|\| '%' ORDER BY processed_at DESC LIMIT 10` |
| `get_breaking_alerts()` | `SELECT * FROM processed_articles WHERE urgency_score = 3 AND processed_at > NOW() - INTERVAL '6 hours' ORDER BY processed_at DESC` |

The MCP server implementation is out of scope for this phase but the schema and indexes are designed to support these queries efficiently.

---

## 9. Open Questions for the Team

1. **llama.cpp network access:** Is the llama.cpp server on the same Docker host? If so, the processor can use `host.docker.internal` or `network_mode: host`. If it's on a different machine on the LAN, use the LAN IP. The team needs to confirm the endpoint.

2. **Notification transport (Phase 4):** The scope doc mentions push notifications for urgency 3 articles. What's the delivery channel? Options include Ntfy (self-hosted push), a webhook to Flowise, email via SMTP, or a Discord/Slack webhook. This doesn't affect the current phase but will influence how the MCP server exposes alerts.

3. **Feed list:** Who curates the initial feed list? Provide an OPML file or a list of RSS URLs for the team to import into Miniflux on first boot.

4. **Content truncation:** The plan truncates article content to 4,000 characters before sending to the LLM. The team should test whether Qwen 3.5 at 35B produces better summaries with more context (e.g., 8,000 chars) given the available VRAM and context window.
