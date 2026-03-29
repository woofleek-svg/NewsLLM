"""AI News Aggregator — LLM Processing Script.

Polls Miniflux for unread articles, sends them to Qwen 3.5 via llama.cpp,
and writes structured output to the output Postgres database.
"""

import json
import logging
import os
import re
import sys
import time
import urllib.parse

import psycopg2
import psycopg2.extras
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("news-processor")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MINIFLUX_URL = os.environ["MINIFLUX_URL"]
MINIFLUX_API_KEY = os.environ["MINIFLUX_API_KEY"]
LLM_URL = os.environ.get("LLM_URL") or os.environ.get("LLAMA_CPP_URL")  # LLAMA_CPP_URL kept for backwards compat
LLM_MODEL = os.environ.get("LLM_MODEL") or os.environ.get("LLAMA_MODEL", "qwen3.5-35b")
LLM_BACKEND = os.environ.get("LLM_BACKEND", "llama.cpp")  # llama.cpp | ollama | vllm | generic
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")  # Required for vLLM with auth, optional otherwise
OUTPUT_DB_URL = os.environ["OUTPUT_DB_URL"]
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "300"))
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", "64000"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "1"))

SYSTEM_PROMPT = """\
You are a news analysis assistant. You will receive a news article and must \
respond with ONLY a valid JSON object — no markdown, no explanation, no preamble.

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
- 1 (routine): Scheduled announcements, opinion pieces, feature stories, \
product reviews, industry trends, earnings reports meeting expectations.
- 2 (notable): Unexpected policy changes, significant leadership changes, \
major product launches, surprising data releases, legal actions against major entities.
- 3 (breaking): Events with immediate widespread impact — armed conflicts, \
natural disasters, major government collapses, critical infrastructure failures, \
pandemic-level health emergencies, assassination or death of a head of state.

If in doubt between two levels, choose the LOWER one. \
Alert fatigue is worse than a missed notification.

Respond with ONLY the JSON object."""

# ---------------------------------------------------------------------------
# Miniflux client
# ---------------------------------------------------------------------------

_miniflux_session = requests.Session()
_miniflux_session.headers["X-Auth-Token"] = MINIFLUX_API_KEY


def fetch_unread_entries(limit: int = 50) -> list[dict]:
    """Fetch unread entries from Miniflux."""
    resp = _miniflux_session.get(
        f"{MINIFLUX_URL}/entries",
        params={"status": "unread", "limit": limit},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("entries", [])


def mark_entry_read(entry_id: int) -> None:
    """Mark a single Miniflux entry as read."""
    try:
        resp = _miniflux_session.put(
            f"{MINIFLUX_URL}/entries",
            json={"entry_ids": [entry_id], "status": "read"},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Failed to mark entry %d as read: %s", entry_id, exc)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


def call_llm(category: str, title: str, feed_name: str, content: str) -> tuple[dict | None, str]:
    """Send article to the LLM and return (parsed_json, raw_text).

    Supports llama.cpp, Ollama, vLLM, and any OpenAI-compatible API.
    Returns (None, raw_text) if the response can't be parsed as JSON.
    Raises requests.RequestException if the server is unreachable.
    """
    user_message = (
        f"Category: {category}\n"
        f"Title: {title}\n"
        f"Source: {feed_name}\n"
        f"Content: {content[:MAX_CONTENT_LENGTH]}"
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.1,
    }

    # Backend-specific options
    if LLM_BACKEND == "llama.cpp":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif LLM_BACKEND == "vllm":
        payload["max_tokens"] = 1024
    elif LLM_BACKEND == "ollama":
        payload["options"] = {"num_predict": 1024}

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    resp = requests.post(LLM_URL, json=payload, headers=headers, timeout=300)
    resp.raise_for_status()

    raw_text = resp.json()["choices"][0]["message"]["content"]

    # Strip thinking tags if the model still emits them
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned), raw_text
    except (json.JSONDecodeError, KeyError):
        return None, raw_text


def validate_llm_output(data: dict) -> str | None:
    """Return an error message if the LLM output is missing required fields."""
    required = {"summary", "tags", "entities", "urgency_score"}
    missing = required - set(data.keys())
    if missing:
        return f"Missing fields: {missing}"

    if not isinstance(data["tags"], list):
        return "tags must be a list"

    if not isinstance(data["entities"], list):
        return "entities must be a list"

    if data["urgency_score"] not in (1, 2, 3):
        return f"urgency_score must be 1-3, got {data['urgency_score']}"

    return None


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------


def _optimize_image_url(url: str) -> str:
    """Rewrite image URLs to request a smaller version where the CDN supports it."""
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)

    # WordPress (wp.com / i0.wp.com) — use w= or resize= param
    if "wp.com" in parsed.hostname or "wordpress.com" in parsed.hostname:
        params["w"] = ["600"]
        params.pop("fit", None)
        params.pop("resize", None)
        new_query = urllib.parse.urlencode(params, doseq=True)
        return urllib.parse.urlunparse(parsed._replace(query=new_query))

    # NBC / Tegna media — use fit= param
    if "nbcnews.com" in parsed.hostname or "nbcchicago.com" in parsed.hostname or "tegna-media.com" in parsed.hostname:
        params["fit"] = ["600,400"]
        params["quality"] = ["75"]
        new_query = urllib.parse.urlencode(params, doseq=True)
        return urllib.parse.urlunparse(parsed._replace(query=new_query))

    # Atlantic CDN — thumbor URLs support path-based resizing
    if "cdn.theatlantic.com" in parsed.hostname and "/thumbor/" in parsed.path:
        # Replace existing size spec with a bounded one
        optimized_path = re.sub(r'/thumbor/[^/]+/', '/thumbor/600x0/', parsed.path)
        return urllib.parse.urlunparse(parsed._replace(path=optimized_path))

    return url


def extract_image_url(entry: dict) -> str | None:
    """Extract an image URL from a Miniflux entry using a fallback chain.

    1. Enclosure with image mime type
    2. First <img src="..."> in content HTML
    3. None

    URLs are optimized to request smaller versions where supported.
    """
    # Check enclosures first
    for enc in entry.get("enclosures") or []:
        mime = enc.get("mime_type", "")
        url = enc.get("url", "")
        if mime.startswith("image/") and url:
            try:
                parsed = urllib.parse.urlparse(url)
                if parsed.scheme in ("http", "https"):
                    return _optimize_image_url(url)
            except ValueError:
                pass

    # Fall back to first img tag in content
    content = entry.get("content", "")
    match = re.search(r'<img[^>]+src=["\']([^"\']+)', content)
    if match:
        url = match.group(1)
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme in ("http", "https"):
                return _optimize_image_url(url)
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------


def get_db_connection():
    """Create a new database connection."""
    return psycopg2.connect(OUTPUT_DB_URL)


def article_already_processed(cur, miniflux_id: int) -> bool:
    """Check if we've already processed this article."""
    cur.execute(
        "SELECT 1 FROM processed_articles WHERE miniflux_id = %s",
        (miniflux_id,),
    )
    return cur.fetchone() is not None


def insert_processed_article(cur, entry: dict, llm_output: dict, raw_text: str, processing_ms: int) -> None:
    """Insert a successfully processed article."""
    image_url = extract_image_url(entry)
    cur.execute(
        """
        INSERT INTO processed_articles
            (miniflux_id, source_feed, category, original_title, original_url,
             published_at, image_url, summary, tags, entities, urgency_score,
             model_used, processing_ms, raw_llm_output)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (miniflux_id) DO NOTHING
        """,
        (
            entry["id"],
            entry.get("feed", {}).get("title", "unknown"),
            entry.get("feed", {}).get("category", {}).get("title"),
            entry["title"],
            entry["url"],
            entry["published_at"],
            image_url,
            llm_output["summary"],
            llm_output["tags"],
            json.dumps(llm_output["entities"]),
            llm_output["urgency_score"],
            LLM_MODEL,
            processing_ms,
            json.dumps({"raw": raw_text}),
        ),
    )


def insert_failed_article(cur, entry: dict, error: str, raw_text: str | None = None) -> None:
    """Insert a failed article into the dead letter table."""
    cur.execute(
        """
        INSERT INTO failed_articles (miniflux_id, original_title, original_url, error_message, raw_llm_output)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (miniflux_id) DO NOTHING
        """,
        (entry["id"], entry.get("title"), entry.get("url"), error, raw_text),
    )


def purge_old_records(cur) -> int:
    """Delete records older than 48 hours. Returns total rows deleted."""
    cur.execute("DELETE FROM processed_articles WHERE processed_at < NOW() - INTERVAL '48 hours'")
    count = cur.rowcount
    cur.execute("DELETE FROM failed_articles WHERE failed_at < NOW() - INTERVAL '48 hours'")
    count += cur.rowcount
    return count


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------


class LLMUnavailableError(Exception):
    """Raised when the LLM server is unreachable or times out."""


def process_entry(cur, entry: dict) -> None:
    """Process a single Miniflux entry through the LLM pipeline.

    Raises LLMUnavailableError if the LLM server can't be reached,
    signaling the caller to abort the rest of the cycle.
    """
    miniflux_id = entry["id"]
    title = entry.get("title", "(no title)")

    if article_already_processed(cur, miniflux_id):
        log.debug("Skipping already-processed article %d: %s", miniflux_id, title)
        return

    content = entry.get("content", "")
    category = entry.get("feed", {}).get("category", {}).get("title", "uncategorized")
    feed_name = entry.get("feed", {}).get("title", "unknown")

    last_error = None
    raw_text = None

    for attempt in range(1 + MAX_RETRIES):
        try:
            start = time.monotonic()
            llm_output, raw_text = call_llm(category, title, feed_name, content)
            processing_ms = int((time.monotonic() - start) * 1000)
        except requests.RequestException as exc:
            log.error("LLM unreachable for article %d: %s — aborting cycle", miniflux_id, exc)
            mark_entry_read(miniflux_id)
            raise LLMUnavailableError(str(exc))

        if llm_output is None:
            last_error = "Failed to parse JSON from LLM response"
            log.warning("Attempt %d — parse error for %d (raw: %.200s)", attempt + 1, miniflux_id, raw_text)
            continue

        error = validate_llm_output(llm_output)
        if error:
            last_error = error
            log.warning("Attempt %d — validation failed for %d: %s", attempt + 1, miniflux_id, error)
            continue

        insert_processed_article(cur, entry, llm_output, raw_text, processing_ms)
        log.info("Processed article %d: %s (urgency=%d, %dms)", miniflux_id, title, llm_output["urgency_score"], processing_ms)
        mark_entry_read(miniflux_id)
        return

    # All retries exhausted (malformed output, not connectivity)
    insert_failed_article(cur, entry, last_error or "Unknown error", raw_text)
    log.warning("Article %d written to failed_articles: %s", miniflux_id, last_error)
    mark_entry_read(miniflux_id)


def run_cycle() -> None:
    """Run one processing cycle: fetch, process, purge."""
    try:
        entries = fetch_unread_entries()
    except requests.RequestException as exc:
        log.warning("Miniflux API unreachable: %s — will retry next cycle", exc)
        return

    if not entries:
        log.info("No unread entries")
        return

    log.info("Fetched %d unread entries", len(entries))

    try:
        conn = get_db_connection()
        conn.autocommit = False
    except psycopg2.Error as exc:
        log.critical("Output database unreachable: %s — halting cycle", exc)
        return

    try:
        with conn.cursor() as cur:
            for entry in entries:
                try:
                    process_entry(cur, entry)
                    conn.commit()
                except LLMUnavailableError:
                    conn.rollback()
                    log.warning("LLM server down — skipping remaining %d entries, will retry next cycle", len(entries))
                    break
                except Exception:
                    conn.rollback()
                    log.exception("Unexpected error processing entry %s", entry.get("id"))

            # Purge old records at end of cycle
            purged = purge_old_records(cur)
            conn.commit()
            if purged:
                log.info("Purged %d records older than 48 hours", purged)
    finally:
        conn.close()


def main() -> None:
    if not LLM_URL:
        log.critical("LLM_URL (or LLAMA_CPP_URL) environment variable is not set — cannot start")
        sys.exit(1)

    log.info("News processor starting (poll_interval=%ds, model=%s, backend=%s)", POLL_INTERVAL, LLM_MODEL, LLM_BACKEND)

    while True:
        run_cycle()
        log.info("Sleeping %d seconds until next cycle", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
