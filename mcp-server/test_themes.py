import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
import os
import sys

# Mocking external dependencies before importing server
sys.modules["psycopg2"] = MagicMock()
sys.modules["psycopg2.extras"] = MagicMock()
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.fastmcp"] = MagicMock()

# mock OUTPUT_DB_URL so server.py can be imported without crashing
os.environ["OUTPUT_DB_URL"] = "postgresql://mock"

from server import _load_themes, DEFAULT_THEME

class TestThemes(unittest.TestCase):
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_themes_file_not_found(self, mock_file):
        """Test that _load_themes falls back to default when file is missing."""
        result = _load_themes()
        self.assertEqual(result, {"default": DEFAULT_THEME})

    @patch("builtins.open", mock_open(read_data="invalid json"))
    @patch("json.load", side_effect=json.JSONDecodeError("Expecting value", "", 0))
    def test_load_themes_invalid_json(self, mock_json_load):
        """Test that _load_themes falls back to default when JSON is invalid."""
        result = _load_themes()
        self.assertEqual(result, {"default": DEFAULT_THEME})

    @patch("builtins.open", mock_open(read_data='{"custom": {"name": "Custom"}}'))
    @patch("json.load", return_value={"custom": {"name": "Custom"}})
    def test_load_themes_success(self, mock_json_load):
        """Test that _load_themes correctly loads themes from file."""
        # Note: when mocking json.load, we don't strictly need mock_open's read_data
        # but it's good practice.
        result = _load_themes()
        self.assertEqual(result, {"custom": {"name": "Custom"}})

if __name__ == "__main__":
    unittest.main()
