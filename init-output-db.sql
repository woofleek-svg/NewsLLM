-- Output database schema for AI News Aggregator
-- Applied automatically on first boot via docker-entrypoint-initdb.d

-- Core output table
CREATE TABLE processed_articles (
    id              SERIAL PRIMARY KEY,
    miniflux_id     BIGINT UNIQUE NOT NULL,
    source_feed     VARCHAR(255) NOT NULL,
    category        VARCHAR(100),
    original_title  TEXT NOT NULL,
    original_url    TEXT NOT NULL,
    published_at    TIMESTAMPTZ NOT NULL,
    processed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- LLM-generated fields
    summary         TEXT NOT NULL,
    tags            TEXT[] NOT NULL,
    entities        JSONB NOT NULL DEFAULT '[]',
    urgency_score   SMALLINT NOT NULL CHECK (urgency_score BETWEEN 1 AND 3),

    -- Media
    image_url       TEXT,

    -- Processing metadata
    model_used      VARCHAR(100) NOT NULL,
    processing_ms   INTEGER,
    raw_llm_output  JSONB
);

-- Index for morning briefing query (recent articles sorted by urgency)
CREATE INDEX idx_briefing ON processed_articles (processed_at DESC, urgency_score DESC);

-- Index for keyword/tag search
CREATE INDEX idx_tags ON processed_articles USING GIN (tags);

-- Index for entity search
CREATE INDEX idx_entities ON processed_articles USING GIN (entities);

-- Index for 48-hour purge job
CREATE INDEX idx_purge ON processed_articles (processed_at);

-- Dead letter table for failed LLM processing
CREATE TABLE failed_articles (
    id              SERIAL PRIMARY KEY,
    miniflux_id     BIGINT UNIQUE NOT NULL,
    original_title  TEXT,
    original_url    TEXT,
    error_message   TEXT NOT NULL,
    raw_llm_output  TEXT,
    failed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    retry_count     SMALLINT NOT NULL DEFAULT 0
);
