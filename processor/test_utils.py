import sys
import os
import unittest
from unittest.mock import MagicMock

# Ensure we can import main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up environment for module initialization
os.environ["MINIFLUX_URL"] = "http://miniflux"
os.environ["MINIFLUX_API_KEY"] = "key"
os.environ["OUTPUT_DB_URL"] = "postgres://db"
os.environ["LLM_URL"] = "http://llm"

import main
from main import _optimize_image_url, get_already_processed_ids, extract_image_url, insert_failed_article

class TestOptimizeImageUrl(unittest.TestCase):
    def test_empty_url(self):
        self.assertEqual(_optimize_image_url(""), "")

    def test_no_hostname(self):
        self.assertEqual(_optimize_image_url("file:///path"), "file:///path")
        self.assertEqual(_optimize_image_url("just_a_string"), "just_a_string")
        self.assertEqual(_optimize_image_url("http://"), "http://")

    def test_unmatched_hostname(self):
        url = "https://example.com/image.jpg?w=100"
        self.assertEqual(_optimize_image_url(url), url)

    def test_wordpress(self):
        # Should add w=600, remove fit and resize
        url = "https://i0.wp.com/example.com/img.jpg?resize=100,100&fit=200,200&other=param"
        optimized = _optimize_image_url(url)
        self.assertIn("w=600", optimized)
        self.assertIn("other=param", optimized)
        self.assertNotIn("resize", optimized)
        self.assertNotIn("fit=", optimized)

        # Also wp.com
        url2 = "https://wp.com/example.com/img.jpg"
        optimized2 = _optimize_image_url(url2)
        self.assertIn("w=600", optimized2)

        # And wordpress.com
        url3 = "https://foo.wordpress.com/img.jpg"
        optimized3 = _optimize_image_url(url3)
        self.assertIn("w=600", optimized3)

    def test_nbc_tegna(self):
        # Should add fit=600,400 and quality=75
        for domain in ["nbcnews.com", "nbcchicago.com", "tegna-media.com"]:
            url = f"https://media.{domain}/img.jpg?foo=bar"
            optimized = _optimize_image_url(url)
            self.assertIn("fit=600%2C400", optimized) # urlencode encodes comma to %2C
            self.assertIn("quality=75", optimized)
            self.assertIn("foo=bar", optimized)

    def test_atlantic_thumbor(self):
        # Should replace the path component immediately following /thumbor/
        # e.g., /thumbor/ABCxyz/100x200/... -> /thumbor/600x0/100x200/...
        url = "https://cdn.theatlantic.com/thumbor/ABC123xyz/100x200/img.jpg"
        optimized = _optimize_image_url(url)
        self.assertIn("/thumbor/600x0/", optimized)
        self.assertNotIn("/thumbor/ABC123xyz/", optimized)

        # Should not modify if it doesn't match the pattern exactly
        url_no_thumbor = "https://cdn.theatlantic.com/images/100x200/img.jpg"
        optimized2 = _optimize_image_url(url_no_thumbor)
        self.assertEqual(optimized2, url_no_thumbor)

class TestGetAlreadyProcessedIds(unittest.TestCase):
    def test_get_already_processed_ids_empty(self):
        cur = MagicMock()
        result = get_already_processed_ids(cur, [])
        self.assertEqual(result, set())
        cur.execute.assert_not_called()

    def test_get_already_processed_ids_none(self):
        cur = MagicMock()
        result = get_already_processed_ids(cur, None)
        self.assertEqual(result, set())
        cur.execute.assert_not_called()

    def test_get_already_processed_ids_some(self):
        cur = MagicMock()
        cur.fetchall.return_value = [(1,), (3,)]

        result = get_already_processed_ids(cur, [1, 2, 3, 4])
        self.assertEqual(result, {1, 3})
        cur.execute.assert_called_once()

        # Verify query matches
        call_args = cur.execute.call_args[0]
        self.assertIn("= ANY(%s)", call_args[0])
        self.assertEqual(call_args[1], ([1, 2, 3, 4],))

    def test_get_already_processed_ids_none_found(self):
        cur = MagicMock()
        cur.fetchall.return_value = []

        result = get_already_processed_ids(cur, [1, 2])
        self.assertEqual(result, set())
        cur.execute.assert_called_once()

    def test_get_already_processed_ids_all_found(self):
        cur = MagicMock()
        cur.fetchall.return_value = [(1,), (2,)]

        result = get_already_processed_ids(cur, [1, 2])
        self.assertEqual(result, {1, 2})
        cur.execute.assert_called_once()






class TestExtractImageUrl(unittest.TestCase):
    def test_valid_enclosure(self):
        entry = {
            "enclosures": [
                {"mime_type": "image/jpeg", "url": "https://example.com/image.jpg"}
            ]
        }
        self.assertEqual(extract_image_url(entry), "https://example.com/image.jpg")

    def test_invalid_enclosure_scheme(self):
        entry = {
            "enclosures": [
                {"mime_type": "image/jpeg", "url": "javascript:alert(1)"}
            ]
        }
        self.assertIsNone(extract_image_url(entry))

        entry2 = {
            "enclosures": [
                {"mime_type": "image/png", "url": "file:///etc/passwd"}
            ]
        }
        self.assertIsNone(extract_image_url(entry2))

    def test_valid_img_tag(self):
        entry = {
            "content": '<p>Check out this image:</p><img src="http://example.org/pic.png" alt="pic">'
        }
        self.assertEqual(extract_image_url(entry), "http://example.org/pic.png")

    def test_invalid_img_tag_scheme(self):
        entry = {
            "content": '<img src="javascript:alert(1)">'
        }
        self.assertIsNone(extract_image_url(entry))

        entry2 = {
            "content": '<img src="data:image/png;base64,iVBORw0KGgo=">'
        }
        self.assertIsNone(extract_image_url(entry2))

        entry3 = {
            "content": '<img src="file:///etc/hosts">'
        }
        self.assertIsNone(extract_image_url(entry3))

    def test_enclosure_over_img_tag(self):
        entry = {
            "enclosures": [
                {"mime_type": "image/jpeg", "url": "https://example.com/enc.jpg"}
            ],
            "content": '<img src="https://example.com/content.jpg">'
        }
        self.assertEqual(extract_image_url(entry), "https://example.com/enc.jpg")

    def test_invalid_enclosure_falls_back_to_img_tag(self):
        entry = {
            "enclosures": [
                {"mime_type": "image/jpeg", "url": "javascript:alert(1)"}
            ],
            "content": '<img src="https://example.com/content.jpg">'
        }
        self.assertEqual(extract_image_url(entry), "https://example.com/content.jpg")

    def test_missing_fields(self):
        self.assertIsNone(extract_image_url({}))
        self.assertIsNone(extract_image_url({"enclosures": []}))
        self.assertIsNone(extract_image_url({"content": ""}))
        self.assertIsNone(extract_image_url({"content": "<p>No image here</p>"}))

    def test_optimizes_url(self):
        # The extract_image_url function calls _optimize_image_url
        entry = {
            "enclosures": [
                {"mime_type": "image/jpeg", "url": "https://i0.wp.com/example.com/img.jpg?resize=100,100"}
            ]
        }
        # Result should be optimized
        self.assertIn("w=600", extract_image_url(entry))




class TestInsertFailedArticle(unittest.TestCase):
    def test_insert_failed_article_with_raw_text(self):
        cur = MagicMock()
        entry = {
            "id": 123,
            "title": "Test Article",
            "url": "https://example.com/test",
        }
        error_msg = "Test Error"
        raw_text = "Raw LLM output"

        insert_failed_article(cur, entry, error_msg, raw_text)

        cur.execute.assert_called_once()

        call_args = cur.execute.call_args[0]
        query = call_args[0]
        params = call_args[1]

        self.assertIn("INSERT INTO failed_articles", query)
        self.assertIn("ON CONFLICT (miniflux_id) DO NOTHING", query)
        self.assertEqual(params, (123, "Test Article", "https://example.com/test", "Test Error", "Raw LLM output"))

    def test_insert_failed_article_without_raw_text(self):
        cur = MagicMock()
        entry = {
            "id": 456,
            # Missing title and url to test .get() defaults
        }
        error_msg = "Another Error"

        insert_failed_article(cur, entry, error_msg)

        cur.execute.assert_called_once()

        call_args = cur.execute.call_args[0]
        query = call_args[0]
        params = call_args[1]

        self.assertIn("INSERT INTO failed_articles", query)
        self.assertIn("ON CONFLICT (miniflux_id) DO NOTHING", query)
        self.assertEqual(params, (456, None, None, "Another Error", None))

if __name__ == '__main__':
    unittest.main()
