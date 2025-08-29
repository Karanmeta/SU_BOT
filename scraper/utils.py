# scraper/utils.py
import time, os
from urllib.parse import urlparse, urljoin, urldefrag

def rate_limit(delay: float):
    time.sleep(delay)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def normalize_url(base: str, href: str):
    if not href:
        return None
    href = urljoin(base, href)
    href, _ = urldefrag(href)
    return href

def same_domain(url: str, allowed_netloc: str) -> bool:
    try:
        return urlparse(url).netloc.endswith(allowed_netloc)
    except Exception:
        return False

def is_pdf(url: str) -> bool:
    if not url:
        return False
    return url.lower().split('?')[0].endswith('.pdf')
