import unittest
from main import extract_image_url

class TestImageExtractor(unittest.TestCase):

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
        # We modified the enclosure block to not return the invalid URL.
        # However, the loop continues and then it falls back to the content img tag!
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

if __name__ == '__main__':
    unittest.main()
