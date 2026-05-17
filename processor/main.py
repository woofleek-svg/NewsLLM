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
import threading
import typing
import urllib.parse
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import psycopg2

from shared.urls import is_safe_url, parse_safe_url
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

# Support multiple LLM endpoints for load balancing
_raw_llm_urls = (os.environ.get("LLM_URLS") or 
                 os.environ.get("LLM_URL") or 
                 os.environ.get("LLAMA_CPP_URL") or 
                 "http://10.0.0.5:8000/v1/chat/completions")
LLM_URLS = [u.strip() for u in _raw_llm_urls.split(",") if u.strip()]
LLM_URL_CYCLE = itertools.cycle(LLM_URLS)
LLM_URL_LOCK = threading.Lock()

def get_next_llm_url() -> str:
    """Thread-safe selection of the next LLM endpoint."""
    with LLM_URL_LOCK:
        return next(LLM_URL_CYCLE)

LLM_MODEL = os.environ.get("LLM_MODEL") or os.environ.get("LLAMA_MODEL", "qwen3.5-35b")
LLM_BACKEND = os.environ.get("LLM_BACKEND", "llama.cpp")  # llama.cpp | litellm | ollama | vllm | generic
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")  # Required for vLLM with auth, optional otherwise
OUTPUT_DB_URL = os.environ["OUTPUT_DB_URL"]
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "300"))
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", "64000"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "1"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "3"))
PURGE_INTERVAL_HOURS = int(os.environ.get("PURGE_INTERVAL_HOURS", "48"))
SYSTEM_PROMPT_FILE = os.environ.get("SYSTEM_PROMPT_FILE", "")
PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "1")
HEALTH_PORT = int(os.environ.get("HEALTH_PORT", "9090"))

if PURGE_INTERVAL_HOURS <= 0:
    raise ValueError("PURGE_INTERVAL_HOURS must be a positive integer.")

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

if SYSTEM_PROMPT_FILE:
    try:
        with open(SYSTEM_PROMPT_FILE) as f:
            SYSTEM_PROMPT = f.read().strip()
        log.info('Loaded system prompt from %s', SYSTEM_PROMPT_FILE)
    except (FileNotFoundError, PermissionError) as exc:
        log.error('Failed to load prompt file %s: %s — using default', SYSTEM_PROMPT_FILE, exc)

# ---------------------------------------------------------------------------
# Health Server
# ---------------------------------------------------------------------------

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        """Handle GET requests for health check."""
        if self.path == '/healthz':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            masked_db_url = OUTPUT_DB_URL
            try:
                parsed = urllib.parse.urlparse(OUTPUT_DB_URL)
                if parsed.password:
                    masked_db_url = OUTPUT_DB_URL.replace(":" + parsed.password + "@", ":***@")
            except Exception:
                masked_db_url = "***"

            resp = {
                "status": "ok",
                "llm_urls": LLM_URLS,
                "db_url": masked_db_url,
            }
            self.wfile.write(json.dumps(resp).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: typing.Any) -> None:
        """Log an arbitrary message."""
        # Suppress access logs for health checks
        pass

def start_health_server() -> None:
    """Start the background HTTP health server."""
    server = ThreadedHTTPServer(('0.0.0.0', HEALTH_PORT), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info("Started health server on port %d", HEALTH_PORT)

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
    if LLM_BACKEND == "litellm":
        payload["max_tokens"] = 1024
        payload["response_format"] = {"type": "json_object"}
    elif LLM_BACKEND == "llama.cpp":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["response_format"] = {"type": "json_object"}
    elif LLM_BACKEND == "vllm":
        payload["max_tokens"] = 1024
        payload["response_format"] = {"type": "json_object"}
    elif LLM_BACKEND == "ollama":
        payload["options"] = {"num_predict": 1024}
        payload["format"] = "json"

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    llm_url = get_next_llm_url()
    resp = requests.post(llm_url, json=payload, headers=headers, timeout=300)
    resp.raise_for_status()

    raw_text = resp.json()["choices"][0]["message"]["content"]

    # Strip thinking tags if the model still emits them
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    # Extract JSON from potential surrounding text or markdown fences
    json_match = re.search(r"({.*})", cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(1)
    else:
        # Fallback to existing fence stripping if regex fails to find braces
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


def _optimize_image_url(url: str, parsed: urllib.parse.ParseResult | None = None) -> str:
    """Rewrite image URLs to request a smaller version where the CDN supports it."""
    if parsed is None:
        parsed = urllib.parse.urlparse(url)
    if not parsed.hostname:
        return url

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
            parsed = parse_safe_url(url)
            if parsed:
                return _optimize_image_url(url, parsed)

    # Fall back to first valid img tag in content
    content = entry.get("content", "")
    for match in re.finditer(r'<img[^>]+src=["\']([^"\']+)', content):
        url = match.group(1)
        parsed = parse_safe_url(url)
        if parsed:
            return _optimize_image_url(url, parsed)

    return None


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------


def get_db_connection() -> psycopg2.extensions.connection:
    """Create a new database connection."""
    return psycopg2.connect(OUTPUT_DB_URL)


def get_already_processed_ids(cur: psycopg2.extensions.cursor, miniflux_ids: list[int]) -> set[int]:
    """Check which of the given Miniflux IDs have already been processed."""
    if not miniflux_ids:
        return set()
    cur.execute(
        "SELECT miniflux_id FROM processed_articles WHERE miniflux_id = ANY(%s)",
        (miniflux_ids,),
    )
    return {row[0] for row in cur.fetchall()}


def insert_processed_article(cur: psycopg2.extensions.cursor, entry: dict, llm_output: dict, raw_text: str, processing_ms: int) -> None:
    """Insert a successfully processed article."""
    image_url = extract_image_url(entry)
    original_url = entry.get("url")
    if not original_url or not is_safe_url(original_url):
        original_url = "#"

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
            original_url,
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


def insert_failed_article(cur: psycopg2.extensions.cursor, entry: dict, error: str, raw_text: str | None = None) -> None:
    """Insert a failed article into the dead letter table."""
    cur.execute(
        """
        INSERT INTO failed_articles (miniflux_id, original_title, original_url, error_message, raw_llm_output)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (miniflux_id) DO NOTHING
        """,
        (entry["id"], entry.get("title"), entry.get("url"), error, raw_text),
    )


def purge_old_records(cur: psycopg2.extensions.cursor) -> int:
    """Delete records older than the configured interval. Returns total rows deleted."""
    cur.execute("DELETE FROM processed_articles WHERE processed_at < NOW() - (%s || ' hours')::interval", (str(PURGE_INTERVAL_HOURS),))
    count = cur.rowcount
    cur.execute("DELETE FROM failed_articles WHERE failed_at < NOW() - (%s || ' hours')::interval", (str(PURGE_INTERVAL_HOURS),))
    count += cur.rowcount
    return count


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------


def process_article_task(entry: dict) -> dict:
    """Worker task to process a single article through the LLM.

    Returns a result dictionary to be handled by the main thread.
    """
    miniflux_id = entry["id"]
    title = entry.get("title", "(no title)")
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
            return {"entry": entry, "error": f"LLM unreachable: {exc}", "type": "critical"}

        if llm_output is None:
            last_error = "Failed to parse JSON from LLM response"
            log.warning("Attempt %d — parse error for %d (raw: %.200s)", attempt + 1, miniflux_id, raw_text)
            continue

        error = validate_llm_output(llm_output)
        if error:
            last_error = error
            log.warning("Attempt %d — validation failed for %d: %s", attempt + 1, miniflux_id, error)
            continue

        return {
            "type": "success",
            "entry": entry,
            "llm_output": llm_output,
            "raw_text": raw_text,
            "processing_ms": processing_ms
        }

    return {
        "type": "failed",
        "entry": entry,
        "error": last_error or "Unknown error",
        "raw_text": raw_text
    }


def run_cycle() -> None:
    """Run one processing cycle: fetch, parallel process, serial write, purge."""
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
            # Filter already processed
            processed_ids = get_already_processed_ids(cur, [e["id"] for e in entries])
            to_process = [e for e in entries if e["id"] not in processed_ids]

            if not to_process:
                log.info("All fetched entries were already processed")
            else:
                log.info("Processing %d new articles using %d workers", len(to_process), MAX_WORKERS)
                
                results = []
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    future_to_entry = {executor.submit(process_article_task, entry): entry for entry in to_process}
                    for future in as_completed(future_to_entry):
                        try:
                            results.append(future.result())
                        except Exception as exc:
                            entry = future_to_entry[future]
                            log.error("Unhandled exception in worker for article %d: %s", entry["id"], exc)

                # Serial database writes and Miniflux marking
                for res in results:
                    entry = res["entry"]
                    res_type = res["type"]

                    if res_type == "success":
                        try:
                            insert_processed_article(cur, entry, res["llm_output"], res["raw_text"], res["processing_ms"])
                            conn.commit()
                            mark_entry_read(entry["id"])
                            log.info("Processed article %d: %s (urgency=%d, %dms)", 
                                     entry["id"], entry.get("title"), res["llm_output"]["urgency_score"], res["processing_ms"])
                        except Exception as exc:
                            conn.rollback()
                            log.error("Failed to write success result for article %d: %s", entry["id"], exc)

                    elif res_type == "failed":
                        try:
                            insert_failed_article(cur, entry, res["error"], res.get("raw_text"))
                            conn.commit()
                            mark_entry_read(entry["id"])
                            log.warning("Article %d written to failed_articles: %s", entry["id"], res["error"])
                        except Exception as exc:
                            conn.rollback()
                            log.error("Failed to write failure result for article %d: %s", entry["id"], exc)

                    elif res_type == "critical":
                        log.error("Critical worker failure for article %d: %s — will retry next cycle", entry["id"], res["error"])
                        # We don't mark as read, so it will be retried

            # Purge old records at end of cycle
            try:
                purged = purge_old_records(cur)
                conn.commit()
                if purged:
                    log.info("Purged %d records older than %d hours", purged, PURGE_INTERVAL_HOURS)
            except Exception as exc:
                conn.rollback()
                log.error("Purge failed: %s", exc)

    finally:
        conn.close()


def main() -> None:
    """Run the news processor service."""
    if not LLM_URLS:
        log.critical("No LLM endpoints configured (set LLM_URLS or LLM_URL) — cannot start")
        sys.exit(1)

    start_health_server()
    log.info("News processor starting (poll_interval=%ds, model=%s, workers=%d, backends=%d)", 
             POLL_INTERVAL, LLM_MODEL, MAX_WORKERS, len(LLM_URLS))

    while True:
        run_cycle()
        log.info("Sleeping %d seconds until next cycle", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()