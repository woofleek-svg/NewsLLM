import urllib.parse
from typing import Optional

def is_safe_url(url: str) -> bool:
    """Check if a URL has a safe scheme (http or https)."""
    if not url:
        return False
    try:
        parsed = urllib.parse.urlparse(url)
        return parsed.scheme in ("http", "https")
    except ValueError:
        return False

def parse_safe_url(url: str) -> Optional[urllib.parse.ParseResult]:
    """Parse a URL and return the ParseResult if it has a safe scheme, otherwise None."""
    if not url:
        return None
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme in ("http", "https"):
            return parsed
        return None
    except ValueError:
        return None
