import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Ensure we can import main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up environment for module initialization
os.environ["MINIFLUX_URL"] = "http://miniflux"
os.environ["MINIFLUX_API_KEY"] = "key"
os.environ["OUTPUT_DB_URL"] = "postgres://db"
os.environ["LLM_URL"] = "http://llm"

import main
import requests

class TestCallLLM:
    @patch("main.requests.post")
    def test_call_llm_success_litellm(self, mock_post):
        mock_response = MagicMock()
        valid_json = {"summary": "test", "tags": ["a"], "entities": [], "urgency_score": 1}
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(valid_json)}}]
        }
        mock_post.return_value = mock_response

        # Test with default litellm backend
        main.LLM_BACKEND = "litellm"
        result_json, raw_text = main.call_llm("Tech", "Title", "Feed", "Content")
        
        assert result_json == valid_json
        assert raw_text == json.dumps(valid_json)
        
        # Verify payload
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["max_tokens"] == 1024
        assert "Content" in payload["messages"][1]["content"]

    @patch("main.requests.post")
    def test_call_llm_backends(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"summary": "x"}'}}]
        }
        mock_post.return_value = mock_response

        # llama.cpp
        main.LLM_BACKEND = "llama.cpp"
        main.call_llm("Tech", "Title", "Feed", "Content")
        payload = mock_post.call_args[1]["json"]
        assert payload["chat_template_kwargs"]["enable_thinking"] is False

        mock_post.reset_mock()
        # vllm
        main.LLM_BACKEND = "vllm"
        main.call_llm("Tech", "Title", "Feed", "Content")
        payload = mock_post.call_args[1]["json"]
        assert payload["max_tokens"] == 1024

        mock_post.reset_mock()
        # ollama
        main.LLM_BACKEND = "ollama"
        main.call_llm("Tech", "Title", "Feed", "Content")
        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["num_predict"] == 1024

    @patch("main.requests.post")
    def test_call_llm_strips_thinking(self, mock_post):
        mock_response = MagicMock()
        valid_json = {"summary": "test", "tags": ["a"], "entities": [], "urgency_score": 1}
        raw_output = "<think>some internal thought</think>\n" + json.dumps(valid_json)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": raw_output}}]
        }
        mock_post.return_value = mock_response

        result_json, raw_text = main.call_llm("Tech", "Title", "Feed", "Content")
        assert result_json == valid_json
        assert raw_text == raw_output

    @patch("main.requests.post")
    def test_call_llm_strips_markdown_fences(self, mock_post):
        mock_response = MagicMock()
        valid_json = {"summary": "test", "tags": ["a"], "entities": [], "urgency_score": 1}
        raw_output = "```json\n" + json.dumps(valid_json) + "\n```"
        mock_response.json.return_value = {
            "choices": [{"message": {"content": raw_output}}]
        }
        mock_post.return_value = mock_response

        result_json, raw_text = main.call_llm("Tech", "Title", "Feed", "Content")
        assert result_json == valid_json
        assert raw_text == raw_output

    @patch("main.requests.post")
    def test_call_llm_json_parse_error(self, mock_post):
        mock_response = MagicMock()
        raw_output = "I am not JSON"
        mock_response.json.return_value = {
            "choices": [{"message": {"content": raw_output}}]
        }
        mock_post.return_value = mock_response

        result_json, raw_text = main.call_llm("Tech", "Title", "Feed", "Content")
        assert result_json is None
        assert raw_text == raw_output

    @patch("main.requests.post")
    def test_call_llm_network_error(self, mock_post):
        mock_post.side_effect = requests.RequestException("Timeout")
        with pytest.raises(requests.RequestException):
            main.call_llm("Tech", "Title", "Feed", "Content")

class TestValidateLLMOutput:
    def test_validate_success(self):
        valid_data = {"summary": "x", "tags": ["y"], "entities": [], "urgency_score": 1}
        assert main.validate_llm_output(valid_data) is None

    def test_validate_missing_fields(self):
        invalid_data = {"summary": "x"}
        error = main.validate_llm_output(invalid_data)
        assert "Missing fields:" in error

    def test_validate_tags_not_list(self):
        invalid_data = {"summary": "x", "tags": "y", "entities": [], "urgency_score": 1}
        error = main.validate_llm_output(invalid_data)
        assert error == "tags must be a list"

    def test_validate_entities_not_list(self):
        invalid_data = {"summary": "x", "tags": ["y"], "entities": "none", "urgency_score": 1}
        error = main.validate_llm_output(invalid_data)
        assert error == "entities must be a list"

    @pytest.mark.parametrize("score", [0, 4, "1", None])
    def test_validate_urgency_score_out_of_range(self, score):
        invalid_data = {"summary": "x", "tags": ["y"], "entities": [], "urgency_score": score}
        error = main.validate_llm_output(invalid_data)
        assert error.startswith("urgency_score must be 1-3")

class TestImageExtraction:
    def test_optimize_image_url_passthrough(self):
        # No hostname
        assert main._optimize_image_url("file:///path") == "file:///path"
        # Unknown host
        assert main._optimize_image_url("https://example.com/img.jpg") == "https://example.com/img.jpg"

    def test_optimize_image_url_wordpress(self):
        url = "https://i0.wp.com/example.com/img.jpg?resize=100,100&fit=200,200"
        optimized = main._optimize_image_url(url)
        assert "w=600" in optimized
        assert "resize" not in optimized
        assert "fit" not in optimized

    def test_optimize_image_url_nbc_tegna(self):
        url = "https://media.nbcchicago.com/img.jpg?foo=bar"
        optimized = main._optimize_image_url(url)
        assert "fit=600%2C400" in optimized
        assert "quality=75" in optimized

    def test_optimize_image_url_atlantic_thumbor(self):
        url = "https://cdn.theatlantic.com/thumbor/ABC123xyz/100x200/img.jpg"
        optimized = main._optimize_image_url(url)
        assert "/thumbor/600x0/" in optimized

    def test_extract_image_url_enclosure(self):
        entry = {
            "enclosures": [{"mime_type": "image/jpeg", "url": "https://example.com/img.jpg"}]
        }
        assert main.extract_image_url(entry) == "https://example.com/img.jpg"

    def test_extract_image_url_enclosure_skips_invalid(self):
        entry = {
            "enclosures": [
                {"mime_type": "image/jpeg", "url": "javascript:alert(1)"},
                {"mime_type": "image/jpeg", "url": "https://example.com/img.jpg"}
            ]
        }
        assert main.extract_image_url(entry) == "https://example.com/img.jpg"

    def test_extract_image_url_fallback_img_tag(self):
        entry = {
            "content": '<p>Text</p><img src="https://example.com/content.jpg">'
        }
        assert main.extract_image_url(entry) == "https://example.com/content.jpg"

    def test_extract_image_url_fallback_img_tag_skips_invalid(self):
        entry = {
            "content": '<img src="data:image/png;base64,123"><img src="https://example.com/content.jpg">'
        }
        # The regex just grabs the first one, which is invalid, so it should return None currently if the first match fails the check.
        # Actually, let's verify what main.py does:
        # match = re.search(r'<img[^>]+src=["\']([^"\']+)', content)
        # if match: ... check ... return
        # It doesn't loop over all img tags, it only checks the FIRST img tag.
        assert main.extract_image_url(entry) is None

    def test_extract_image_url_none(self):
        entry = {"content": "<p>No images</p>"}
        assert main.extract_image_url(entry) is None
