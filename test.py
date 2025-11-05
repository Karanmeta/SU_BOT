import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

BASE_URL = "https://scet.ac.in"
DOMAIN = urlparse(BASE_URL).netloc

# Limit crawling
MAX_PAGES = 5000
CRAWL_DEPTH = 3

def is_valid_url(url):
    """Check if the link belongs to SCET domain and is valid."""
    return url.startswith(BASE_URL) and not any(x in url for x in ["#", "mailto", "tel", "javascript"])

def crawl_count(base_url=BASE_URL, max_pages=MAX_PAGES, max_depth=CRAWL_DEPTH):
    visited = set()
    queue = deque([(base_url, 0)])

    while queue and len(visited) < max_pages:
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        print(f"[{len(visited)}] Crawling: {url}")

        try:
            res = requests.get(url, timeout=10)
            if res.status_code != 200:
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"]).split("?")[0]
                if is_valid_url(link) and link not in visited:
                    queue.append((link, depth + 1))
        except Exception as e:
            print(f"⚠️ Error fetching {url}: {e}")
            continue

    return visited

if __name__ == "__main__":
    print("🚀 Counting SCET website pages...")
    pages = crawl_count()
    print("\n✅ Total unique internal pages found:", len(pages))

    # Optional: Save to a file
    with open("scet_pages_list.txt", "w", encoding="utf-8") as f:
        for p in sorted(pages):
            f.write(p + "\n")

    print("📄 Saved all page URLs to scet_pages_list.txt")
