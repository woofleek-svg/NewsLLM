import os
import sys
from unittest.mock import MagicMock

# Mocking external dependencies before importing server
sys.modules["psycopg2"] = MagicMock()
sys.modules["psycopg2.extras"] = MagicMock()
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.fastmcp"] = MagicMock()

# mock OUTPUT_DB_URL so server.py can be imported without crashing
os.environ["OUTPUT_DB_URL"] = "postgresql://mock"

from server import _build_briefing_html

articles = []
intro_text = "<script>alert(1)</script>"

html_out = _build_briefing_html(articles, intro=intro_text)

assert "<script>alert(1)</script>" not in html_out, "Found unescaped intro string!"
assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out, "Did not find escaped intro string!"

print("Test passed: intro is properly escaped.")
