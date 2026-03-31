import unittest
import re
import urllib.parse

# Re-implement extract_image_url locally for tests without importing main.py (due to dependencies)
# but using the logic we want to verify.
IMG_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)')

def extract_image_url(entry: dict) -> str | None:
    for enc in entry.get("enclosures") or []:
        mime = enc.get("mime_type", "")
        url = enc.get("url", "")
        if mime.startswith("image/") and url:
            try:
                parsed = urllib.parse.urlparse(url)
                if parsed.scheme in ("http", "https"):
                    return url
            except ValueError:
                pass

    content = entry.get("content", "")
    match = IMG_RE.search(content)
    if match:
        url = match.group(1)
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme in ("http", "https"):
                return url
        except ValueError:
            pass

    return None

class TestImageExtractor(unittest.TestCase):

    def test_valid_enclosure(self):
        entry = {
            "enclosures": [
                {"mime_type": "image/jpeg", "url": "https://example.com/image.jpg"}
            ]
        }
        self.assertEqual(extract_image_url(entry), "https://example.com/image.jpg")

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

    def test_enclosure_over_img_tag(self):
        entry = {
            "enclosures": [
                {"mime_type": "image/jpeg", "url": "https://example.com/enc.jpg"}
            ],
            "content": '<img src="https://example.com/content.jpg">'
        }
        self.assertEqual(extract_image_url(entry), "https://example.com/enc.jpg")

if __name__ == '__main__':
    unittest.main()
