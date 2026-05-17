import os
import sys
from unittest.mock import MagicMock
from datetime import datetime, timezone
import pytest

# mock OUTPUT_DB_URL so server.py can be imported without crashing
os.environ["OUTPUT_DB_URL"] = "postgresql://mock"

sys.modules["psycopg2"] = MagicMock()
sys.modules["psycopg2.extras"] = MagicMock()
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.fastmcp"] = MagicMock()

from server import _format_article_summary, _format_article_full, _build_plain_text, URGENCY_LABELS

class TestFormatting:
    def setup_method(self, method):
        pass

    def test_format_article_summary_complete(self):
        dt = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
        row = {
            "id": 1,
            "original_title": "Test Title",
            "source_feed": "Test Source",
            "category": "Test Category",
            "original_url": "http://example.com",
            "published_at": dt,
            "summary": "Test summary.",
            "tags": ["tag1", "tag2"],
            "urgency_score": 2,
            "image_url": "http://example.com/image.png"
        }
        res = _format_article_summary(row)
        assert res["id"] == 1
        assert res["title"] == "Test Title"
        assert res["source"] == "Test Source"
        assert res["category"] == "Test Category"
        assert res["url"] == "http://example.com"
        assert res["published"] == dt.isoformat()
        assert res["summary"] == "Test summary."
        assert res["tags"] == ["tag1", "tag2"]
        assert res["urgency"] == 2
        assert res["image_url"] == "http://example.com/image.png"

    def test_format_article_summary_missing_optionals(self):
        row = {
            "id": 1,
            "original_title": "Test Title",
            "source_feed": "Test Source",
            "category": "Test Category",
            "original_url": "http://example.com",
            "published_at": None,
            "summary": "Test summary.",
            "tags": ["tag1", "tag2"],
            "urgency_score": 2,
        }
        res = _format_article_summary(row)
        assert res["published"] is None
        assert "image_url" not in res

    def test_format_article_full_complete(self):
        dt = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
        dt2 = datetime(2023, 1, 1, 12, 5, tzinfo=timezone.utc)
        row = {
            "id": 1,
            "original_title": "Test Title",
            "source_feed": "Test Source",
            "category": "Test Category",
            "original_url": "http://example.com",
            "image_url": "http://example.com/image.png",
            "published_at": dt,
            "processed_at": dt2,
            "summary": "Test summary.",
            "tags": ["tag1", "tag2"],
            "entities": {"orgs": ["Test Org"]},
            "urgency_score": 2,
            "model_used": "test-model",
            "processing_ms": 100,
        }
        res = _format_article_full(row)
        assert res["id"] == 1
        assert res["title"] == "Test Title"
        assert res["source"] == "Test Source"
        assert res["category"] == "Test Category"
        assert res["url"] == "http://example.com"
        assert res["image_url"] == "http://example.com/image.png"
        assert res["published"] == dt.isoformat()
        assert res["processed"] == dt2.isoformat()
        assert res["summary"] == "Test summary."
        assert res["tags"] == ["tag1", "tag2"]
        assert res["entities"] == {"orgs": ["Test Org"]}
        assert res["urgency"] == 2
        assert res["model"] == "test-model"
        assert res["processing_ms"] == 100

    def test_format_article_full_missing_optionals(self):
        row = {
            "id": 1,
            "original_title": "Test Title",
            "source_feed": "Test Source",
            "category": "Test Category",
            "original_url": "http://example.com",
            "image_url": None,
            "published_at": None,
            "processed_at": None,
            "summary": "Test summary.",
            "tags": ["tag1", "tag2"],
            "entities": {"orgs": ["Test Org"]},
            "urgency_score": 2,
            "model_used": "test-model",
            "processing_ms": 100,
        }
        res = _format_article_full(row)
        assert res["image_url"] is None
        assert res["published"] is None
        assert res["processed"] is None

    def test_build_plain_text_with_intro(self):
        articles = [
            {
                "original_title": "Test Title",
                "summary": "Test summary",
                "source_feed": "Test Source",
                "original_url": "http://example.com",
                "urgency_score": 2
            }
        ]
        res = _build_plain_text(articles, intro="Here is your news:")
        assert "NewsLLM Briefing" in res
        assert "========================================" in res
        assert "Here is your news:" in res
        assert "[Notable] Test Title" in res
        assert "Test summary" in res
        assert "Source: Test Source" in res
        assert "http://example.com" in res

    def test_build_plain_text_without_intro(self):
        articles = [
            {
                "original_title": "Test Title",
                "summary": "Test summary",
                "source_feed": "Test Source",
                "original_url": "http://example.com",
                "urgency_score": 3
            }
        ]
        res = _build_plain_text(articles)
        assert "NewsLLM Briefing" in res
        assert "========================================" in res
        assert "[Breaking] Test Title" in res
        assert "Test summary" in res
        assert "Source: Test Source" in res
        assert "http://example.com" in res

    def test_build_plain_text_fallbacks(self):
        articles = [
            {
                "title": "Fallback Title",
                "url": "http://fallback.com",
                "source": "Fallback Source",
                "urgency": 1,
                # missing original_title, original_url, source_feed, urgency_score
                # summary omitted entirely
            },
            {
                # Completely empty
            },
            {
                # Invalid urgency score
                "urgency_score": 99
            }
        ]
        res = _build_plain_text(articles)
        assert "[Routine] Fallback Title" in res
        assert "Source: Fallback Source" in res
        assert "http://fallback.com" in res

        # Test completely empty fallback
        assert "[Routine] Untitled" in res
        assert "Source: " in res

        # Test non-standard urgency
        assert res.count("[Routine] Untitled") == 2
