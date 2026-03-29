"""AI News Aggregator — MCP Server.

Exposes progressive discovery tools over the processed news database.
Designed for nested search: discover sources → browse briefings → drill into articles.
"""

import html
import json
import logging
import os
import smtplib
from contextlib import contextmanager
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import psycopg2
import psycopg2.extras
from mcp.server.fastmcp import FastMCP

log = logging.getLogger("mcp-server")

OUTPUT_DB_URL = os.environ["OUTPUT_DB_URL"]
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
EMAIL_RECIPIENTS = [e.strip() for e in os.environ.get("EMAIL_RECIPIENTS", "").split(",") if e.strip()]

mcp = FastMCP(
    "NewsLLM",
    host="0.0.0.0",
    port=8000,
    instructions=(
        "News aggregator with processed and summarized articles from RSS feeds. "
        "Start with list_sources() to discover available categories and feeds, "
        "then use get_briefing() or search_news() to find relevant articles. "
        "Use get_article() to read full details on a specific article. "
        "Use get_breaking() to check for urgent/breaking news alerts."
    ),
)


@contextmanager
def get_db():
    """Yield a database connection with RealDictCursor."""
    conn = psycopg2.connect(OUTPUT_DB_URL)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
    finally:
        conn.close()


def _format_article_summary(row: dict) -> dict:
    """Format an article row for summary listing (no raw LLM output)."""
    result = {
        "id": row["id"],
        "title": row["original_title"],
        "source": row["source_feed"],
        "category": row["category"],
        "url": row["original_url"],
        "published": row["published_at"].isoformat() if row["published_at"] else None,
        "summary": row["summary"],
        "tags": row["tags"],
        "urgency": row["urgency_score"],
    }
    if row.get("image_url"):
        result["image_url"] = row["image_url"]
    return result


def _format_article_full(row: dict) -> dict:
    """Format an article row with all details."""
    result = {
        "id": row["id"],
        "title": row["original_title"],
        "source": row["source_feed"],
        "category": row["category"],
        "url": row["original_url"],
        "image_url": row.get("image_url"),
        "published": row["published_at"].isoformat() if row["published_at"] else None,
        "processed": row["processed_at"].isoformat() if row["processed_at"] else None,
        "summary": row["summary"],
        "tags": row["tags"],
        "entities": row["entities"],
        "urgency": row["urgency_score"],
        "model": row["model_used"],
        "processing_ms": row["processing_ms"],
    }
    return result


# ---------------------------------------------------------------------------
# Tool 1: Discover what's available
# ---------------------------------------------------------------------------


@mcp.tool()
def list_sources() -> dict:
    """List all available news categories and feeds with article counts.

    Call this first to understand what sources are available before searching.
    Returns categories with their feeds and how many articles each has
    in the current 48-hour window.
    """
    with get_db() as cur:
        cur.execute("""
            SELECT
                coalesce(category, 'uncategorized') as category,
                source_feed,
                count(*) as article_count,
                max(published_at) as latest_article
            FROM processed_articles
            GROUP BY category, source_feed
            ORDER BY category, article_count DESC
        """)
        rows = cur.fetchall()

    sources = {}
    for row in rows:
        cat = row["category"]
        if cat not in sources:
            sources[cat] = {"feeds": [], "total_articles": 0}
        sources[cat]["feeds"].append({
            "name": row["source_feed"],
            "articles": row["article_count"],
            "latest": row["latest_article"].isoformat() if row["latest_article"] else None,
        })
        sources[cat]["total_articles"] += row["article_count"]

    return sources


# ---------------------------------------------------------------------------
# Tool 2: Briefing — recent notable articles
# ---------------------------------------------------------------------------


@mcp.tool()
def get_briefing(category: str = "", hours: int = 24, limit: int = 20) -> list[dict]:
    """Get a briefing of recent articles, prioritized by urgency.

    Args:
        category: Filter by category name (e.g. "Local", "News"). Empty for all.
        hours: How far back to look (default 24, max 48).
        limit: Max articles to return (default 20, max 50).

    Returns articles sorted by urgency (highest first), then recency.
    """
    hours = min(max(hours, 1), 48)
    limit = min(max(limit, 1), 50)

    with get_db() as cur:
        if category:
            cur.execute("""
                SELECT * FROM processed_articles
                WHERE processed_at > NOW() - make_interval(hours => %s)
                  AND category ILIKE %s
                ORDER BY urgency_score DESC, processed_at DESC
                LIMIT %s
            """, (hours, category, limit))
        else:
            cur.execute("""
                SELECT * FROM processed_articles
                WHERE processed_at > NOW() - make_interval(hours => %s)
                ORDER BY urgency_score DESC, processed_at DESC
                LIMIT %s
            """, (hours, limit))

        return [_format_article_summary(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Tool 3: Search — keyword and tag search
# ---------------------------------------------------------------------------


@mcp.tool()
def search_news(
    query: str,
    category: str = "",
    urgency_min: int = 1,
    limit: int = 10,
) -> list[dict]:
    """Search recent news by keyword or tag.

    Args:
        query: Search term — matched against tags, article titles, summaries, and entities.
        category: Filter by category name. Empty for all.
        urgency_min: Minimum urgency score (1=routine, 2=notable, 3=breaking).
        limit: Max results (default 10, max 30).

    Results are ranked by relevance: exact tag matches first, then title,
    then summary matches. Use get_article() to read full details on a result.
    """
    limit = min(max(limit, 1), 30)
    urgency_min = min(max(urgency_min, 1), 3)

    with get_db() as cur:
        params = [query, f"%{query}%", f"%{query}%", f"%{query}%", urgency_min]
        category_clause = ""
        if category:
            category_clause = "AND category ILIKE %s"
            params.append(category)
        params.append(limit)

        cur.execute(f"""
            SELECT *,
                CASE
                    WHEN %s = ANY(tags) THEN 3
                    WHEN original_title ILIKE %s THEN 2
                    WHEN summary ILIKE %s THEN 1
                    ELSE 0
                END as relevance
            FROM processed_articles
            WHERE (
                %s = ANY(tags)
                OR original_title ILIKE %s
                OR summary ILIKE %s
                OR entities::text ILIKE %s
            )
            AND urgency_score >= %s
            {category_clause}
            ORDER BY relevance DESC, urgency_score DESC, processed_at DESC
            LIMIT %s
        """, [query, f"%{query}%", f"%{query}%",
              query, f"%{query}%", f"%{query}%", f"%{query}%",
              urgency_min] + ([category] if category else []) + [limit])

        return [_format_article_summary(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Tool 4: Get full article details
# ---------------------------------------------------------------------------


@mcp.tool()
def get_article(article_id: int) -> dict:
    """Get full details for a specific article by its ID.

    Args:
        article_id: The article ID from a briefing or search result.

    Returns complete article data including entities, processing metadata,
    and the original article URL.
    """
    with get_db() as cur:
        cur.execute("SELECT * FROM processed_articles WHERE id = %s", (article_id,))
        row = cur.fetchone()

    if not row:
        return {"error": f"Article {article_id} not found"}

    return _format_article_full(row)


# ---------------------------------------------------------------------------
# Tool 5: Breaking alerts
# ---------------------------------------------------------------------------


@mcp.tool()
def get_breaking(hours: int = 6) -> list[dict]:
    """Check for breaking news alerts (urgency score 3).

    Args:
        hours: How far back to look (default 6, max 48).

    Returns only articles scored as breaking/critical urgency.
    Empty list means no breaking news — which is good.
    """
    hours = min(max(hours, 1), 48)

    with get_db() as cur:
        cur.execute("""
            SELECT * FROM processed_articles
            WHERE urgency_score = 3
              AND processed_at > NOW() - make_interval(hours => %s)
            ORDER BY processed_at DESC
        """, (hours,))

        return [_format_article_summary(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Tool 6: Stats overview
# ---------------------------------------------------------------------------


@mcp.tool()
def get_stats() -> dict:
    """Get a quick overview of the news database.

    Returns total article count, breakdown by urgency and category,
    and processing health stats. Useful for understanding the current
    state of the news pipeline.
    """
    with get_db() as cur:
        cur.execute("SELECT count(*) as total FROM processed_articles")
        total = cur.fetchone()["total"]

        cur.execute("""
            SELECT urgency_score, count(*) as count
            FROM processed_articles
            GROUP BY urgency_score ORDER BY urgency_score
        """)
        urgency = {row["urgency_score"]: row["count"] for row in cur.fetchall()}

        cur.execute("""
            SELECT coalesce(category, 'uncategorized') as category, count(*) as count
            FROM processed_articles
            GROUP BY category ORDER BY count DESC
        """)
        categories = {row["category"]: row["count"] for row in cur.fetchall()}

        cur.execute("SELECT count(*) as count FROM failed_articles")
        failed = cur.fetchone()["count"]

        cur.execute("""
            SELECT min(processed_at) as oldest, max(processed_at) as newest
            FROM processed_articles
        """)
        window = cur.fetchone()

    return {
        "total_articles": total,
        "failed_articles": failed,
        "urgency_breakdown": urgency,
        "categories": categories,
        "window": {
            "oldest": window["oldest"].isoformat() if window["oldest"] else None,
            "newest": window["newest"].isoformat() if window["newest"] else None,
        },
    }


# ---------------------------------------------------------------------------
# Email helpers
# ---------------------------------------------------------------------------

URGENCY_LABELS = {1: "Routine", 2: "Notable", 3: "Breaking"}
URGENCY_COLORS = {1: "#94a3b8", 2: "#f59e0b", 3: "#ef4444"}
URGENCY_BG = {1: "#f8fafc", 2: "#fffbeb", 3: "#fef2f2"}


def _build_briefing_html(articles: list[dict], intro: str = "") -> str:
    """Build a news-site styled HTML email from a list of article dicts."""
    # Separate by urgency for sectioned layout
    breaking = [a for a in articles if a.get("urgency_score", a.get("urgency", 1)) == 3]
    notable = [a for a in articles if a.get("urgency_score", a.get("urgency", 1)) == 2]
    routine = [a for a in articles if a.get("urgency_score", a.get("urgency", 1)) == 1]

    html = """\
<html>
<body style="margin: 0; padding: 0; background-color: #0f172a; font-family: -apple-system, 'Segoe UI', Roboto, Arial, sans-serif;">

<!-- Outer wrapper -->
<table width="100%" cellpadding="0" cellspacing="0" style="background-color: #0f172a; padding: 24px 0;">
<tr><td align="center">

<!-- Main card -->
<table width="640" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.3);">

<!-- Header -->
<tr>
<td style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 32px 32px 24px 32px;">
  <table width="100%" cellpadding="0" cellspacing="0">
  <tr>
    <td>
      <h1 style="margin: 0; font-size: 28px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">NewsLLM</h1>
      <p style="margin: 4px 0 0 0; font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px;">Daily Intelligence Briefing</p>
    </td>
    <td align="right" style="vertical-align: top;">
      <span style="display: inline-block; background: #2563eb; color: #ffffff; font-size: 11px; font-weight: 700; padding: 6px 12px; border-radius: 20px; letter-spacing: 0.5px;">""" + f"""{len(articles)} STORIES</span>
    </td>
  </tr>
  </table>
</td>
</tr>
"""

    if intro:
        html += f"""\
<tr>
<td style="padding: 20px 32px 0 32px;">
  <p style="margin: 0; color: #64748b; font-size: 14px; line-height: 1.5;">{intro}</p>
</td>
</tr>
"""

    def _render_section(section_articles, section_title, accent_color, bg_color, icon):
        if not section_articles:
            return ""
        s = f"""\
<tr>
<td style="padding: 24px 32px 8px 32px;">
  <table width="100%" cellpadding="0" cellspacing="0">
  <tr>
    <td style="border-bottom: 2px solid {accent_color}; padding-bottom: 8px;">
      <span style="font-size: 12px; font-weight: 800; color: {accent_color}; text-transform: uppercase; letter-spacing: 1.5px;">{icon} {section_title}</span>
    </td>
  </tr>
  </table>
</td>
</tr>
"""
        for i, a in enumerate(section_articles):
            title = html.escape(a.get("original_title", a.get("title", "Untitled")))
            url = html.escape(a.get("original_url", a.get("url", "#")), quote=True)
            summary = html.escape(a.get("summary", ""))
            source = html.escape(a.get("source_feed", a.get("source", "")))
            raw_image = a.get("image_url")
            image_url = html.escape(raw_image, quote=True) if raw_image else None
            tags = a.get("tags", [])
            tag_html = ""
            for t in tags[:4]:
                tag_html += f'<span style="display: inline-block; background: #e2e8f0; color: #475569; font-size: 10px; font-weight: 600; padding: 3px 8px; border-radius: 10px; margin-right: 4px; margin-bottom: 4px;">{html.escape(t)}</span>'

            border_bottom = f'border-bottom: 1px solid #e2e8f0;' if i < len(section_articles) - 1 else ''

            # Image block — thumbnail floated right if available
            if image_url:
                image_block = f"""\
    <table cellpadding="0" cellspacing="0" width="100%" style="margin-bottom: 10px;">
    <tr>
      <td style="vertical-align: top; padding-right: 16px;">
        <a href="{url}" style="text-decoration: none; color: #0f172a; font-size: 16px; font-weight: 700; line-height: 1.3; display: block; margin-bottom: 6px;">{title}</a>
        <p style="margin: 0; color: #475569; font-size: 13px; line-height: 1.55;">{summary}</p>
      </td>
      <td width="120" style="vertical-align: top;">
        <a href="{url}"><img src="{image_url}" width="120" height="90" alt="" style="border-radius: 6px; object-fit: cover; display: block;" /></a>
      </td>
    </tr>
    </table>"""
            else:
                image_block = f"""\
    <a href="{url}" style="text-decoration: none; color: #0f172a; font-size: 16px; font-weight: 700; line-height: 1.3; display: block; margin-bottom: 6px;">{title}</a>
    <p style="margin: 0 0 10px 0; color: #475569; font-size: 13px; line-height: 1.55;">{summary}</p>"""

            s += f"""\
<tr>
<td style="padding: 0 32px;">
  <div style="padding: 16px 0; {border_bottom}">
    {image_block}
    <table cellpadding="0" cellspacing="0" width="100%">
    <tr>
      <td style="font-size: 11px; color: #94a3b8; font-weight: 600;">{source}</td>
      <td align="right">{tag_html}</td>
    </tr>
    </table>
  </div>
</td>
</tr>
"""
        return s

    html += _render_section(breaking, "Breaking News", "#ef4444", "#fef2f2", "&#9888;")
    html += _render_section(notable, "Notable Stories", "#f59e0b", "#fffbeb", "&#9733;")
    html += _render_section(routine, "Latest News", "#64748b", "#f8fafc", "&#9679;")

    html += """\
<!-- Footer -->
<tr>
<td style="background: #f8fafc; padding: 20px 32px; border-top: 1px solid #e2e8f0;">
  <table width="100%" cellpadding="0" cellspacing="0">
  <tr>
    <td>
      <p style="margin: 0; font-size: 11px; color: #94a3b8;">Powered by <strong style="color: #64748b;">NewsLLM</strong> &mdash; AI News Aggregator</p>
    </td>
    <td align="right">
      <p style="margin: 0; font-size: 11px; color: #94a3b8;">Automated briefing &bull; Do not reply</p>
    </td>
  </tr>
  </table>
</td>
</tr>

</table>
<!-- /Main card -->

</td></tr>
</table>
<!-- /Outer wrapper -->

</body>
</html>"""
    return html


def _build_plain_text(articles: list[dict], intro: str = "") -> str:
    """Build a plain text fallback from a list of article dicts."""
    lines = ["NewsLLM Briefing", "=" * 40, ""]
    if intro:
        lines += [intro, ""]

    for a in articles:
        urgency = a.get("urgency_score", a.get("urgency", 1))
        label = URGENCY_LABELS.get(urgency, "Routine")
        title = a.get("original_title", a.get("title", "Untitled"))
        url = a.get("original_url", a.get("url", ""))
        summary = a.get("summary", "")
        source = a.get("source_feed", a.get("source", ""))

        lines += [
            f"[{label}] {title}",
            summary,
            f"Source: {source}",
            url,
            "",
        ]

    return "\n".join(lines)


def _send_smtp(subject: str, html: str, plain: str, to_addrs: list[str]) -> dict:
    """Send an email via Gmail SMTP. Returns status dict."""
    if not SMTP_USER or not SMTP_PASSWORD:
        return {"error": "SMTP credentials not configured"}
    if not to_addrs:
        return {"error": "No recipients configured or provided"}

    msg = MIMEMultipart("alternative")
    msg["From"] = f"NewsLLM <{SMTP_USER}>"
    msg["To"] = ", ".join(to_addrs)
    msg["Subject"] = subject
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, to_addrs, msg.as_string())
        log.info("Email sent to %s: %s", to_addrs, subject)
        return {"status": "sent", "recipients": to_addrs, "subject": subject}
    except smtplib.SMTPException as exc:
        log.error("Failed to send email: %s", exc)
        return {"error": f"SMTP error: {exc}"}


# ---------------------------------------------------------------------------
# Tool 7: Email a briefing
# ---------------------------------------------------------------------------


@mcp.tool()
def email_briefing(
    subject: str,
    category: str = "",
    hours: int = 24,
    limit: int = 20,
    intro: str = "",
    recipients: list[str] | None = None,
) -> dict:
    """Build and email a news briefing directly from the database.

    Queries recent articles, formats them into a styled HTML email, and
    sends to all configured recipients. This is the primary tool for
    scheduled briefing emails.

    Args:
        subject: Email subject line (e.g. "Morning News Briefing — March 29").
        category: Filter by category (e.g. "Local", "News"). Empty for all.
        hours: How far back to look (default 24, max 48).
        limit: Max articles to include (default 20, max 50).
        intro: Optional intro paragraph for the email (e.g. "Here's your morning update.").
        recipients: Optional override list of email addresses. Omit to use defaults.

    Returns send status with recipient list.
    """
    hours = min(max(hours, 1), 48)
    limit = min(max(limit, 1), 50)

    with get_db() as cur:
        if category:
            cur.execute("""
                SELECT * FROM processed_articles
                WHERE processed_at > NOW() - make_interval(hours => %s)
                  AND category ILIKE %s
                ORDER BY urgency_score DESC, processed_at DESC
                LIMIT %s
            """, (hours, category, limit))
        else:
            cur.execute("""
                SELECT * FROM processed_articles
                WHERE processed_at > NOW() - make_interval(hours => %s)
                ORDER BY urgency_score DESC, processed_at DESC
                LIMIT %s
            """, (hours, limit))

        articles = cur.fetchall()

    if not articles:
        return {"error": "No articles found for the given filters"}

    html = _build_briefing_html(articles, intro)
    plain = _build_plain_text(articles, intro)
    to_addrs = recipients if recipients else EMAIL_RECIPIENTS

    return _send_smtp(subject, html, plain, to_addrs)


# ---------------------------------------------------------------------------
# Tool 8: Send a custom email
# ---------------------------------------------------------------------------


@mcp.tool()
def send_email(subject: str, body: str, recipients: list[str] | None = None) -> dict:
    """Send a short custom email. For briefings, prefer email_briefing() instead.

    Use this for custom messages, alerts, or short notifications — not full
    briefings. Keep the body short (a few sentences).

    Args:
        subject: Email subject line.
        body: Plain text email body. Keep it brief.
        recipients: Optional override list. Omit to use defaults.
    """
    to_addrs = recipients if recipients else EMAIL_RECIPIENTS

    html_body = f"""\
<html><body style="font-family: -apple-system, Arial, sans-serif; max-width: 680px; margin: 0 auto;">
<h2 style="border-bottom: 2px solid #2563eb; padding-bottom: 8px;">NewsLLM</h2>
<p>{html.escape(body)}</p>
<hr style="border: none; border-top: 1px solid #e5e7eb;">
<p style="font-size: 11px; color: #9ca3af;">Sent by NewsLLM — AI News Aggregator</p>
</body></html>"""

    return _send_smtp(subject, html_body, body, to_addrs)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
