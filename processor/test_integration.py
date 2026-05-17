import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock
import requests
import time

# Set up environment for module initialization
os.environ["MINIFLUX_URL"] = "http://miniflux"
os.environ["MINIFLUX_API_KEY"] = "test"
os.environ["OUTPUT_DB_URL"] = "postgresql://test:test@localhost/test"
os.environ["LLM_URL"] = "http://llm"
os.environ["POLL_INTERVAL"] = "60"
os.environ["PURGE_INTERVAL_HOURS"] = "48"

# Ensure we can import main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main

class TestRunCycle:
    """Integration tests for the main processing cycle."""
    @patch("main.fetch_unread_entries")
    @patch("main.get_db_connection")
    @patch("main.get_already_processed_ids")
    @patch("main.process_entry")
    @patch("main.purge_old_records")
    def test_run_cycle_success(self, mock_purge, mock_process, mock_get_processed, mock_get_db, mock_fetch):
        """Test a successful processing cycle with articles."""
        # Setup mocks
        entries = [{"id": 1, "title": "Article 1"}, {"id": 2, "title": "Article 2"}]
        mock_fetch.return_value = entries

        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        mock_get_processed.return_value = set()
        mock_purge.return_value = 5

        # Execute
        main.run_cycle()

        # Verify
        mock_fetch.assert_called_once()
        mock_get_db.assert_called_once()
        mock_get_processed.assert_called_once_with(mock_cur, [1, 2])

        assert mock_process.call_count == 2
        mock_process.assert_any_call(mock_cur, entries[0], set())
        mock_process.assert_any_call(mock_cur, entries[1], set())

        mock_purge.assert_called_once_with(mock_cur)
        assert mock_conn.commit.call_count == 3  # one for each article + one for purge
        mock_conn.close.assert_called_once()

    @patch("main.fetch_unread_entries")
    def test_run_cycle_no_entries(self, mock_fetch):
        """Test cycle when no new entries are available."""
        mock_fetch.return_value = []

        with patch("main.get_db_connection") as mock_get_db:
            main.run_cycle()
            mock_fetch.assert_called_once()
            mock_get_db.assert_not_called()

    @patch("main.fetch_unread_entries")
    def test_run_cycle_miniflux_unreachable(self, mock_fetch):
        """Test cycle when Miniflux API is unreachable."""
        mock_fetch.side_effect = requests.RequestException("Timeout")

        with patch("main.get_db_connection") as mock_get_db:
            main.run_cycle()
            mock_fetch.assert_called_once()
            mock_get_db.assert_not_called()

    @patch("main.fetch_unread_entries")
    @patch("main.get_db_connection")
    @patch("main.get_already_processed_ids")
    @patch("main.process_entry")
    def test_run_cycle_llm_unavailable_aborts(self, mock_process, mock_get_processed, mock_get_db, mock_fetch):
        """Test that cycle aborts if LLM server is down."""
        entries = [{"id": 1, "title": "A"}, {"id": 2, "title": "B"}]
        mock_fetch.return_value = entries

        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        mock_get_processed.return_value = set()

        # Make the first article raise LLMUnavailableError
        mock_process.side_effect = main.LLMUnavailableError("LLM down")

        main.run_cycle()

        mock_process.assert_called_once_with(mock_cur, entries[0], set())
        mock_conn.rollback.assert_called_once()

class TestProcessEntry:
    """Tests for processing a single entry."""
    def setup_method(self, method):
        self.entry = {
            "id": 123,
            "title": "Test Article",
            "content": "Content",
            "feed": {"title": "Test Feed", "category": {"title": "Tech"}}
        }
        self.mock_cur = MagicMock()
        self.processed_ids = set()

    @patch("main.call_llm")
    @patch("main.insert_processed_article")
    @patch("main.mark_entry_read")
    def test_process_entry_success(self, mock_mark_read, mock_insert, mock_call_llm):
        """Test successful processing of a single article."""
        valid_llm_output = {
            "summary": "Sum",
            "tags": ["tech"],
            "entities": [],
            "urgency_score": 1
        }
        mock_call_llm.return_value = (valid_llm_output, json.dumps(valid_llm_output))

        main.process_entry(self.mock_cur, self.entry, self.processed_ids)

        mock_call_llm.assert_called_once_with("Tech", "Test Article", "Test Feed", "Content")
        mock_insert.assert_called_once()
        mock_mark_read.assert_called_once_with(123)

    @patch("main.call_llm")
    @patch("main.insert_failed_article")
    @patch("main.mark_entry_read")
    def test_process_entry_parse_failure(self, mock_mark_read, mock_insert_failed, mock_call_llm):
        """Test article processing when LLM returns non-JSON output."""
        # call_llm returns (None, raw_text) on parse failure
        mock_call_llm.return_value = (None, "I am not JSON")

        main.process_entry(self.mock_cur, self.entry, self.processed_ids)

        # Should be called MAX_RETRIES + 1 times (MAX_RETRIES is 1 by default, so 2 times)
        assert mock_call_llm.call_count == main.MAX_RETRIES + 1
        mock_insert_failed.assert_called_once_with(self.mock_cur, self.entry, "Failed to parse JSON from LLM response", "I am not JSON")
        mock_mark_read.assert_called_once_with(123)

    @patch("main.call_llm")
    @patch("main.mark_entry_read")
    def test_process_entry_llm_unreachable(self, mock_mark_read, mock_call_llm):
        """Test article processing when LLM server is unreachable."""
        mock_call_llm.side_effect = requests.RequestException("Timeout")

        with pytest.raises(main.LLMUnavailableError):
            main.process_entry(self.mock_cur, self.entry, self.processed_ids)

        mock_mark_read.assert_not_called()

    @patch("main.call_llm")
    def test_process_entry_already_processed(self, mock_call_llm):
        """Test that already processed articles are skipped."""
        self.processed_ids.add(123)

        main.process_entry(self.mock_cur, self.entry, self.processed_ids)

        mock_call_llm.assert_not_called()

class TestPurgeOldRecords:
    """Tests for the database purge logic."""
    def test_purge_old_records(self):
        """Verify that purge calls use correct intervals and tables."""
        mock_cur = MagicMock()
        mock_cur.rowcount = 5

        # PURGE_INTERVAL_HOURS is 48 in our env setup
        total_deleted = main.purge_old_records(mock_cur)

        assert mock_cur.execute.call_count == 2

        first_call_args = mock_cur.execute.call_args_list[0][0][0]
        first_call_params = mock_cur.execute.call_args_list[0][0][1]
        assert "DELETE FROM processed_articles" in first_call_args
        assert "%s" in first_call_args
        assert first_call_params == ("48",)

        second_call_args = mock_cur.execute.call_args_list[1][0][0]
        second_call_params = mock_cur.execute.call_args_list[1][0][1]
        assert "DELETE FROM failed_articles" in second_call_args
        assert "%s" in second_call_args
        assert second_call_params == ("48",)

        assert total_deleted == 10  # 5 from processed + 5 from failed

class TestHealthServer:
    """Tests for the internal health check server."""
    @classmethod
    def setup_class(cls):
        # Start health server on a dynamic or known port
        main.HEALTH_PORT = 9091
        main.start_health_server()

        # Poll to ensure the server is up
        for _ in range(10):
            try:
                requests.get(f"http://localhost:{main.HEALTH_PORT}/healthz")
                break
            except requests.ConnectionError:
                time.sleep(0.1)

    def test_healthz_endpoint(self):
        """Verify that /healthz returns 200 and correct status info."""
        resp = requests.get(f"http://localhost:{main.HEALTH_PORT}/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["llm_url"] == main.LLM_URL
        assert data["db_url"] == "postgresql://test:***@localhost/test"

    def test_non_healthz_endpoint(self):
        """Verify that unknown endpoints return 404."""
        resp = requests.get(f"http://localhost:{main.HEALTH_PORT}/other")
        assert resp.status_code == 404
