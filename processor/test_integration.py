import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import json
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
    @patch("main.process_article_task")
    @patch("main.insert_processed_article")
    @patch("main.mark_entry_read")
    @patch("main.purge_old_records")
    def test_run_cycle_success(self, mock_purge, mock_mark_read, mock_insert, mock_process_task, mock_get_processed, mock_get_db, mock_fetch):
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

        # Mock process_article_task results
        mock_process_task.side_effect = [
            {"type": "success", "entry": entries[0], "llm_output": {"urgency_score": 1}, "raw_text": "...", "processing_ms": 100},
            {"type": "success", "entry": entries[1], "llm_output": {"urgency_score": 2}, "raw_text": "...", "processing_ms": 150}
        ]

        # Execute
        main.run_cycle()

        # Verify
        mock_fetch.assert_called_once()
        mock_get_db.assert_called_once()
        mock_get_processed.assert_called_once_with(mock_cur, [1, 2])

        assert mock_process_task.call_count == 2
        assert mock_insert.call_count == 2
        assert mock_mark_read.call_count == 2

        mock_purge.assert_called_once_with(mock_cur)
        assert mock_conn.commit.call_count == 3  # two articles + one purge
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

class TestProcessArticleTask:
    """Tests for the individual article processing task."""
    def setup_method(self, method):
        self.entry = {
            "id": 123,
            "title": "Test Article",
            "content": "Content",
            "feed": {"title": "Test Feed", "category": {"title": "Tech"}}
        }

    @patch("main.call_llm")
    def test_process_article_task_success(self, mock_call_llm):
        """Test successful task execution."""
        valid_llm_output = {
            "summary": "Sum",
            "tags": ["tech"],
            "entities": [],
            "urgency_score": 1
        }
        mock_call_llm.return_value = (valid_llm_output, json.dumps(valid_llm_output))

        result = main.process_article_task(self.entry)

        assert result["type"] == "success"
        assert result["entry"] == self.entry
        assert result["llm_output"] == valid_llm_output
        mock_call_llm.assert_called_once_with("Tech", "Test Article", "Test Feed", "Content")

    @patch("main.call_llm")
    def test_process_article_task_parse_failure(self, mock_call_llm):
        """Test task when LLM returns non-JSON output."""
        mock_call_llm.return_value = (None, "I am not JSON")

        result = main.process_article_task(self.entry)

        assert result["type"] == "failed"
        assert result["error"] == "Failed to parse JSON from LLM response"
        # call_count is MAX_RETRIES + 1 (default 2)
        assert mock_call_llm.call_count == main.MAX_RETRIES + 1

    @patch("main.call_llm")
    def test_process_article_task_unreachable(self, mock_call_llm):
        """Test task when LLM is unreachable."""
        mock_call_llm.side_effect = requests.RequestException("Connection error")

        result = main.process_article_task(self.entry)

        assert result["type"] == "critical"
        assert "LLM unreachable" in result["error"]

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
        assert data["llm_urls"] == main.LLM_URLS
        assert data["db_url"] == "postgresql://test:***@localhost/test"

    def test_non_healthz_endpoint(self):
        """Verify that unknown endpoints return 404."""
        resp = requests.get(f"http://localhost:{main.HEALTH_PORT}/other")
        assert resp.status_code == 404
class TestEndToEndCycle:
    """End-to-end tests verifying LLM output parsing and DB writes."""

    @patch("main.fetch_unread_entries")
    @patch("main.get_db_connection")
    @patch("main.mark_entry_read")
    @patch("requests.post")
    def test_run_cycle_with_thinking_tags(self, mock_post, mock_mark_read, mock_get_db, mock_fetch):
        # Setup mocks
        entries = [{"id": 101, "title": "Article 1", "published_at": "2023-10-10T00:00:00Z"}]
        mock_fetch.return_value = entries

        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        # mock_cur.fetchall should return empty for get_already_processed_ids
        mock_cur.fetchall.return_value = []

        # Mock requests.post for LLM
        mock_resp = MagicMock()
        valid_json = {
            "summary": "This is a summary.",
            "tags": ["test"],
            "entities": [],
            "urgency_score": 1
        }

        raw_llm_content = "<think>some reasoning</think>\n" + json.dumps(valid_json)
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": raw_llm_content}}]
        }
        mock_resp.text = raw_llm_content
        mock_post.return_value = mock_resp

        # Run cycle
        main.run_cycle()

        # Verify DB insert
        insert_calls = [
            call for call in mock_cur.execute.call_args_list
            if "INSERT INTO processed_articles" in call[0][0]
        ]
        assert len(insert_calls) == 1

        insert_params = insert_calls[0][0][1]

        # The 8th parameter is summary (index 7), 14th is raw_llm_output (index 13)
        assert insert_params[7] == "This is a summary."

        # The raw output should be stored unmodified
        raw_llm_output_json = json.loads(insert_params[13])
        assert raw_llm_output_json["raw"] == raw_llm_content

    @patch("main.fetch_unread_entries")
    @patch("main.get_db_connection")
    @patch("main.mark_entry_read")
    @patch("requests.post")
    def test_run_cycle_without_thinking_tags(self, mock_post, mock_mark_read, mock_get_db, mock_fetch):
        # Setup mocks
        entries = [{"id": 102, "title": "Article 2", "published_at": "2023-10-10T00:00:00Z"}]
        mock_fetch.return_value = entries

        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        mock_cur.fetchall.return_value = []

        # Mock requests.post for LLM
        mock_resp = MagicMock()
        valid_json = {
            "summary": "This is another summary.",
            "tags": ["test"],
            "entities": [],
            "urgency_score": 2
        }

        raw_llm_content = json.dumps(valid_json)
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": raw_llm_content}}]
        }
        mock_resp.text = raw_llm_content
        mock_post.return_value = mock_resp

        # Run cycle
        main.run_cycle()

        # Verify DB insert
        insert_calls = [
            call for call in mock_cur.execute.call_args_list
            if "INSERT INTO processed_articles" in call[0][0]
        ]
        assert len(insert_calls) == 1

        insert_params = insert_calls[0][0][1]

        assert insert_params[7] == "This is another summary."

        raw_llm_output_json = json.loads(insert_params[13])
        assert raw_llm_output_json["raw"] == raw_llm_content


class MockServerRequestHandler(BaseHTTPRequestHandler):
    payloads = []

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        MockServerRequestHandler.payloads.append(json.loads(post_data.decode('utf-8')))

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        valid_json = {"summary": "test", "tags": ["a"], "entities": [], "urgency_score": 1}
        self.wfile.write(json.dumps({
            "choices": [{"message": {"content": json.dumps(valid_json)}}]
        }).encode('utf-8'))

class TestMockServerIntegration:
    @classmethod
    def setup_class(cls):
        cls.server = HTTPServer(('localhost', 0), MockServerRequestHandler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()

    @classmethod
    def teardown_class(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setup_method(self, method):
        MockServerRequestHandler.payloads.clear()

    def test_mock_server_backends(self):
        # We will mock the load balancing to always return our mock server
        with patch("main.get_next_llm_url", return_value=f"http://localhost:{self.port}"):
            # Save original state
            orig_backend = main.LLM_BACKEND
            orig_extra = getattr(main, "LLM_EXTRA_PARAMS", {})

            try:
                # Test Litellm
                main.LLM_BACKEND = "litellm"
                main.LLM_EXTRA_PARAMS = {"temperature": 0.5, "custom_field": "litellm_test"}
                main.call_llm("Tech", "Title", "Feed", "Content")

                # Test llama.cpp
                main.LLM_BACKEND = "llama.cpp"
                main.LLM_EXTRA_PARAMS = {"temperature": 0.2, "custom_field": "llama_test"}
                main.call_llm("Tech", "Title", "Feed", "Content")

                # Test vllm
                main.LLM_BACKEND = "vllm"
                main.LLM_EXTRA_PARAMS = {"temperature": 0.3, "custom_field": "vllm_test"}
                main.call_llm("Tech", "Title", "Feed", "Content")

                # Test ollama
                main.LLM_BACKEND = "ollama"
                main.LLM_EXTRA_PARAMS = {"temperature": 0.4, "custom_field": "ollama_test"}
                main.call_llm("Tech", "Title", "Feed", "Content")
            finally:
                # Restore original state
                main.LLM_BACKEND = orig_backend
                main.LLM_EXTRA_PARAMS = orig_extra

        payloads = MockServerRequestHandler.payloads

        assert len(payloads) == 4

        # Verify litellm
        assert payloads[0]["max_tokens"] == 1024
        assert payloads[0]["temperature"] == 0.5
        assert payloads[0]["custom_field"] == "litellm_test"

        # Verify llama.cpp
        assert payloads[1]["chat_template_kwargs"]["enable_thinking"] is False
        assert payloads[1]["temperature"] == 0.2
        assert payloads[1]["custom_field"] == "llama_test"

        # Verify vllm
        assert payloads[2]["max_tokens"] == 1024
        assert payloads[2]["temperature"] == 0.3
        assert payloads[2]["custom_field"] == "vllm_test"

        # Verify ollama
        assert payloads[3]["options"]["num_predict"] == 1024
        assert payloads[3]["temperature"] == 0.4
        assert payloads[3]["custom_field"] == "ollama_test"
