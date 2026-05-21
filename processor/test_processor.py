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
import importlib

class TestEnvParsing:
    @patch("main.log.warning")
    def test_invalid_json_extra_params(self, mock_warning):
        import unittest
        os.environ["LLM_EXTRA_PARAMS"] = "{invalid json"

        # Reload main to trigger top-level execution
        importlib.reload(main)

        assert main.LLM_EXTRA_PARAMS == {}
        mock_warning.assert_called_with(
            "Invalid JSON in LLM_EXTRA_PARAMS: %s, falling back to empty dict",
            unittest.mock.ANY
        )

        # Cleanup
        del os.environ["LLM_EXTRA_PARAMS"]
        importlib.reload(main)

    @patch("main.log.warning")
    def test_non_dict_extra_params(self, mock_warning):
        os.environ["LLM_EXTRA_PARAMS"] = '["a", "b"]'

        # Reload main to trigger top-level execution
        importlib.reload(main)

        assert main.LLM_EXTRA_PARAMS == {}
        mock_warning.assert_called_with(
            "LLM_EXTRA_PARAMS must be a JSON object, falling back to empty dict"
        )

        # Cleanup
        del os.environ["LLM_EXTRA_PARAMS"]
        importlib.reload(main)

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
    def test_call_llm_disable_thinking_true(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"summary": "x"}'}}]
        }
        mock_post.return_value = mock_response

        # Enable disable_thinking globally
        main.LLM_DISABLE_THINKING = True

        # litellm with deepseek-r1 (thinking model)
        main.LLM_BACKEND = "litellm"
        main.LLM_MODEL = "deepseek-r1:8b"
        main.call_llm("Tech", "Title", "Feed", "Content")
        payload = mock_post.call_args[1]["json"]
        assert payload["thinking"] == {"type": "disabled"}

        # litellm with non-thinking model (should not add thinking parameter)
        mock_post.reset_mock()
        main.LLM_MODEL = "qwen3.5-35b"
        main.call_llm("Tech", "Title", "Feed", "Content")
        payload = mock_post.call_args[1]["json"]
        assert "thinking" not in payload

        # llama.cpp
        mock_post.reset_mock()
        main.LLM_BACKEND = "llama.cpp"
        main.call_llm("Tech", "Title", "Feed", "Content")
        payload = mock_post.call_args[1]["json"]
        assert payload["chat_template_kwargs"]["enable_thinking"] is False

        # vllm
        mock_post.reset_mock()
        main.LLM_BACKEND = "vllm"
        main.call_llm("Tech", "Title", "Feed", "Content")
        payload = mock_post.call_args[1]["json"]
        assert payload["chat_template_kwargs"]["enable_thinking"] is False

        # ollama — "think" is not a standard Ollama API parameter, should not be present
        mock_post.reset_mock()
        main.LLM_BACKEND = "ollama"
        main.call_llm("Tech", "Title", "Feed", "Content")
        payload = mock_post.call_args[1]["json"]
        assert "think" not in payload
        assert "think" not in payload.get("options", {})

    @patch("main.requests.post")
    def test_call_llm_disable_thinking_false(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"summary": "x"}'}}]
        }
        mock_post.return_value = mock_response

        # Disable disable_thinking globally
        main.LLM_DISABLE_THINKING = False

        # litellm with deepseek-r1
        main.LLM_BACKEND = "litellm"
        main.LLM_MODEL = "deepseek-r1:8b"
        main.call_llm("Tech", "Title", "Feed", "Content")
        payload = mock_post.call_args[1]["json"]
        assert "thinking" not in payload

        # llama.cpp
        mock_post.reset_mock()
        main.LLM_BACKEND = "llama.cpp"
        main.call_llm("Tech", "Title", "Feed", "Content")
        payload = mock_post.call_args[1]["json"]
        assert "chat_template_kwargs" not in payload

        # vllm
        mock_post.reset_mock()
        main.LLM_BACKEND = "vllm"
        main.call_llm("Tech", "Title", "Feed", "Content")
        payload = mock_post.call_args[1]["json"]
        assert "chat_template_kwargs" not in payload

        # ollama
        mock_post.reset_mock()
        main.LLM_BACKEND = "ollama"
        main.call_llm("Tech", "Title", "Feed", "Content")
        payload = mock_post.call_args[1]["json"]
        assert "think" not in payload
        assert "think" not in payload["options"]

        # Restore default behavior
        main.LLM_DISABLE_THINKING = True

    @patch("main.requests.post")
    def test_call_llm_extra_params_merge(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"summary": "x"}'}}]
        }
        mock_post.return_value = mock_response

        # Set LLM_EXTRA_PARAMS
        main.LLM_EXTRA_PARAMS = {
            "temperature": 0.7,
            "options": {"num_predict": 512, "top_k": 40},
            "chat_template_kwargs": {"enable_thinking": True}
        }

        main.LLM_BACKEND = "ollama"
        main.call_llm("Tech", "Title", "Feed", "Content")
        payload = mock_post.call_args[1]["json"]

        assert payload["temperature"] == 0.7
        assert payload["options"]["num_predict"] == 512
        assert payload["options"]["top_k"] == 40
        assert payload["chat_template_kwargs"]["enable_thinking"] is True

        # Clean up
        main.LLM_EXTRA_PARAMS = {}

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
    def test_call_llm_with_extra_params(self, mock_post):
        mock_response = MagicMock()
        valid_json = {"summary": "test", "tags": ["a"], "entities": [], "urgency_score": 1}
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(valid_json)}}]
        }
        mock_post.return_value = mock_response

        # Store original and set new
        original_params = main.LLM_EXTRA_PARAMS
        main.LLM_EXTRA_PARAMS = {"top_p": 0.9, "presence_penalty": 0.5}

        try:
            main.call_llm("Tech", "Title", "Feed", "Content")

            mock_post.assert_called_once()
            payload = mock_post.call_args[1]["json"]
            assert payload["top_p"] == 0.9
            assert payload["presence_penalty"] == 0.5
            assert payload["temperature"] == 0.1 # Should remain unmodified
        finally:
            main.LLM_EXTRA_PARAMS = original_params

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
        # The logic loops over all img tags and returns the first valid one
        assert main.extract_image_url(entry) == "https://example.com/content.jpg"

    def test_extract_image_url_none(self):
        entry = {"content": "<p>No images</p>"}
        assert main.extract_image_url(entry) is None
