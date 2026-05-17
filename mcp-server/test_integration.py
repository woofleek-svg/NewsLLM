import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from datetime import datetime
import importlib

# Mock dependencies to allow import in restricted network environments
sys.modules['psycopg2'] = MagicMock()
sys.modules['psycopg2.extras'] = MagicMock()

class MockMCP:
    def __init__(self, *args, **kwargs):
        pass

    def tool(self):
        def decorator(func):
            return func
        return decorator

mock_mcp = MagicMock()
sys.modules['mcp'] = mock_mcp
sys.modules['mcp.server.fastmcp'] = MagicMock()
sys.modules['mcp.server.fastmcp'].FastMCP = MockMCP

os.environ['OUTPUT_DB_URL'] = 'postgres://fake'
os.environ['SMTP_USER'] = 'test@example.com'
os.environ['SMTP_PASSWORD'] = 'password'
os.environ['EMAIL_RECIPIENTS'] = 'user@example.com'

import server
importlib.reload(server)

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Reset mock for each test
        self.mock_conn = MagicMock()
        self.mock_cur = MagicMock()
        self.mock_conn.cursor.return_value.__enter__.return_value = self.mock_cur

        self.patcher = patch('server.psycopg2.connect', return_value=self.mock_conn)
        self.mock_connect = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_list_sources(self):
        self.mock_cur.fetchall.return_value = [
            {"category": "tech", "source_feed": "TechCrunch", "article_count": 5, "latest_article": datetime(2023, 1, 1)},
            {"category": "tech", "source_feed": "Wired", "article_count": 2, "latest_article": datetime(2023, 1, 2)},
            {"category": "news", "source_feed": "NYT", "article_count": 10, "latest_article": datetime(2023, 1, 3)},
        ]
        res = server.list_sources()
        self.assertIn("tech", res)
        self.assertIn("news", res)
        self.assertEqual(res["tech"]["total_articles"], 7)
        self.assertEqual(len(res["tech"]["feeds"]), 2)
        self.assertEqual(res["news"]["total_articles"], 10)

    def test_get_briefing(self):
        self.mock_cur.fetchall.return_value = [
            {
                "id": 1, "original_title": "T1", "source_feed": "S", "category": "C",
                "original_url": "U", "published_at": datetime(2023,1,1), "summary": "S",
                "tags": [], "urgency_score": 1
            }
        ]

        # test clamping
        server.get_briefing(hours=-1, limit=100)
        query, params = self.mock_cur.execute.call_args[0]
        self.assertEqual(params[0], 1) # clamped hours
        self.assertEqual(params[1], 50) # clamped limit

        # test with category
        server.get_briefing(category="tech", hours=24, limit=10)
        query, params = self.mock_cur.execute.call_args[0]
        self.assertEqual(params[0], 24)
        self.assertEqual(params[1], "tech")
        self.assertEqual(params[2], 10)

    def test_search_news(self):
        self.mock_cur.fetchall.return_value = []

        # Without category
        server.search_news(query="test", urgency_min=0, limit=100)
        query, params = self.mock_cur.execute.call_args[0]
        self.assertEqual(query.count("%s"), len(params))
        self.assertEqual(params[-2], 1) # urgency_min clamped
        self.assertEqual(params[-1], 30) # limit clamped

        # With category
        server.search_news(query="test", category="tech", urgency_min=4, limit=0)
        query, params = self.mock_cur.execute.call_args[0]
        self.assertEqual(query.count("%s"), len(params))
        self.assertEqual(params[-3], 3) # urgency_min clamped
        self.assertEqual(params[-2], "tech")
        self.assertEqual(params[-1], 1) # limit clamped

    def test_get_article(self):
        self.mock_cur.fetchone.return_value = {
            "id": 1, "original_title": "T", "source_feed": "S", "category": "C", "original_url": "U",
            "published_at": datetime(2023,1,1), "processed_at": datetime(2023,1,1), "summary": "S",
            "tags": [], "entities": [], "urgency_score": 1, "model_used": "M", "processing_ms": 100
        }
        res = server.get_article(1)
        self.assertEqual(res["id"], 1)

        self.mock_cur.fetchone.return_value = None
        res = server.get_article(999)
        self.assertIn("error", res)

    def test_get_breaking(self):
        self.mock_cur.fetchall.return_value = []
        server.get_breaking(hours=6)
        query, params = self.mock_cur.execute.call_args[0]
        self.assertIn("urgency_score = 3", query)
        self.assertEqual(params[0], 6)

    def test_get_stats(self):
        self.mock_cur.fetchone.side_effect = [
            {"total": 10},
            {"count": 2},
            {"oldest": datetime(2023,1,1), "newest": datetime(2023,1,2)}
        ]
        self.mock_cur.fetchall.side_effect = [
            [{"urgency_score": 1, "count": 5}],
            [{"category": "tech", "count": 5}]
        ]

        res = server.get_stats()
        self.assertEqual(res["total_articles"], 10)
        self.assertEqual(res["failed_articles"], 2)
        self.assertEqual(res["urgency_breakdown"][1], 5)
        self.assertEqual(res["categories"]["tech"], 5)
        self.assertEqual(res["window"]["oldest"], datetime(2023,1,1).isoformat())

    def test_build_briefing_html(self):
        articles = [
            {"id": 1, "title": "T1", "summary": "S1", "urgency": 1, "url": "http://a.com"},
            {"id": 2, "title": "T2", "summary": "S2", "urgency_score": 3, "url": "http://b.com"}
        ]
        html = server._build_briefing_html(articles, intro="<script>alert(1)</script>", theme_name="non_existent")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("T1", html)
        self.assertIn("T2", html)
        self.assertIn(server.DEFAULT_THEME["bg_outer"], html) # fallback theme works

        # Test theme merging (uses default fallback)
        with patch('server._load_themes', return_value={"custom": {"name": "Custom Theme"}}):
            html2 = server._build_briefing_html(articles, theme_name="custom")
            self.assertIn("Custom Theme", html2)
            self.assertIn(server.DEFAULT_THEME["bg_outer"], html2) # Merged with default missing keys

    def test_is_safe_url(self):
        self.assertTrue(server._is_safe_url("http://example.com"))
        self.assertTrue(server._is_safe_url("https://example.com"))
        self.assertFalse(server._is_safe_url("javascript:alert(1)"))
        self.assertFalse(server._is_safe_url("data:text/html,<html>"))
        self.assertFalse(server._is_safe_url("file:///etc/passwd"))
        self.assertFalse(server._is_safe_url(""))
        self.assertFalse(server._is_safe_url(None))

    def test_send_smtp(self):
        # Restore actual values after test
        orig_user = server.SMTP_USER
        orig_pwd = server.SMTP_PASSWORD

        with patch("server.smtplib.SMTP") as mock_smtp:
            # test with SMTP configured
            server.SMTP_USER = "user"
            server.SMTP_PASSWORD = "password"
            res = server._send_smtp("Subj", "<html>", "plain", ["a@b.com"])
            self.assertEqual(res["status"], "queued")
            # Note: mocking threading.Thread or waiting for worker would be cleaner for asserting sendmail

            # test without credentials
            server.SMTP_USER = ""
            res = server._send_smtp("Subj", "<html>", "plain", ["a@b.com"])
            self.assertFalse(res.get("success", True))
            self.assertIn("SMTP_USER", res["error"])

            # test without recipients
            server.SMTP_USER = "user"
            res = server._send_smtp("Subj", "<html>", "plain", [])
            self.assertFalse(res.get("success", True))
            self.assertIn("EMAIL_RECIPIENTS", res["error"])

        server.SMTP_USER = orig_user
        server.SMTP_PASSWORD = orig_pwd

    def test_email_briefing(self):
        self.mock_cur.fetchall.return_value = [
            {"id": 1, "title": "T1", "summary": "S1", "urgency": 1, "url": "http://a.com"}
        ]

        orig_user = server.SMTP_USER
        orig_pwd = server.SMTP_PASSWORD
        server.SMTP_USER = "user"
        server.SMTP_PASSWORD = "pwd"
        server.EMAIL_RECIPIENTS = ["a@b.com"]

        with patch("server._send_smtp") as mock_send:
            mock_send.return_value = {"status": "sent"}
            res = server.email_briefing("Test Subj")
            self.assertEqual(res.get("status"), "sent")
            mock_send.assert_called_once()
            self.assertIn("Test Subj", mock_send.call_args[0])

        server.SMTP_USER = orig_user
        server.SMTP_PASSWORD = orig_pwd

if __name__ == '__main__':
    unittest.main()
