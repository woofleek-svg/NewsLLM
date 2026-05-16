import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock dependencies to allow import in restricted network environments
sys.modules['psycopg2'] = MagicMock()
sys.modules['psycopg2.extras'] = MagicMock()

# Instead of Mocking mcp completely, let's mock it to return the function unchanged
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

# Mock env vars
os.environ['OUTPUT_DB_URL'] = 'postgres://fake'

# Import the module to test
import server

class TestSearchNews(unittest.TestCase):
    @patch('server.get_db')
    def test_search_news_params_match(self, mock_get_db):
        mock_cur = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_cur
        mock_cur.fetchall.return_value = []
        
        # Test without category
        server.search_news(query="test", category="", urgency_min=1, limit=10)
        
        # Check the execute call
        mock_cur.execute.assert_called()
        call_args = mock_cur.execute.call_args[0]
        query = call_args[0]
        params = call_args[1]
        
        # Count placeholders in query
        placeholders_count = query.count('%s')
        self.assertEqual(placeholders_count, len(params), "Parameter count must match placeholders without category")
        
        # Ensure we're using params and not an inline list with mismatched parts if it was previously checked.
        # This will be verified because the function will be rewritten.
        
        # Test with category
        mock_cur.reset_mock()
        server.search_news(query="test", category="tech", urgency_min=1, limit=10)
        
        # Check the execute call
        mock_cur.execute.assert_called()
        call_args = mock_cur.execute.call_args[0]
        query = call_args[0]
        params = call_args[1]
        
        # Count placeholders in query
        placeholders_count = query.count('%s')
        self.assertEqual(placeholders_count, len(params), "Parameter count must match placeholders with category")

if __name__ == '__main__':
    unittest.main()
