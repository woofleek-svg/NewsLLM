import timeit
import re
import urllib.parse

# Re-implement the original unoptimized logic for comparison
def extract_image_url_original(entry: dict) -> str | None:
    # Check enclosures first
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

    # Fall back to first img tag in content
    content = entry.get("content", "")
    # This is what we're optimizing:
    match = re.search(r'<img[^>]+src=["\']([^"\']+)', content)
    if match:
        url = match.group(1)
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme in ("http", "https"):
                return url
        except ValueError:
            pass

    return None

# Optimized logic
IMG_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)')

def extract_image_url_optimized(entry: dict) -> str | None:
    # Check enclosures first
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

    # Fall back to first img tag in content
    content = entry.get("content", "")
    # Using the pre-compiled regex:
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

# Simulated data
content = '<p>Some text</p><img src="https://example.com/image.jpg" alt="test">' * 10
entry = {"content": content, "enclosures": []}

if __name__ == "__main__":
    n = 100000
    t_orig = timeit.timeit(lambda: extract_image_url_original(entry), number=n)
    t_opt = timeit.timeit(lambda: extract_image_url_optimized(entry), number=n)

    print(f"Original (re-compiling): {t_orig:.4f}s")
    print(f"Optimized (module-level): {t_opt:.4f}s")
    print(f"Improvement: {(t_orig - t_opt) / t_orig * 100:.2f}%")
