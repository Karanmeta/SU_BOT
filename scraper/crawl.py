# scraper/crawl.py
import requests
from bs4 import BeautifulSoup
from collections import deque
from tqdm import tqdm
from requests.adapters import HTTPAdapter, Retry
from .utils import rate_limit, normalize_url, same_domain, is_pdf
from config import BASE_URL, REQUEST_DELAY, ALLOWED_NETLOC, MAX_PAGES
import concurrent.futures
from urllib.parse import urlparse

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def requests_session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update(HEADERS)
    return s

def process_url(url, session, seen, html_pages, pdf_links, q):
    """Process a single URL (thread-safe)"""
    if url in seen:
        return
    
    if not same_domain(url, ALLOWED_NETLOC):
        return
    
    try:
        rate_limit(REQUEST_DELAY)
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            seen.add(url)
            return
        
        ctype = r.headers.get("Content-Type", "") or ""
        if "text/html" not in ctype:
            if is_pdf(url):
                pdf_links.append(url)
            seen.add(url)
            return
        
        soup = BeautifulSoup(r.text, "lxml")
        html_pages.append(url)
        seen.add(url)

        for a in soup.find_all("a", href=True):
            href = normalize_url(url, a["href"])
            if not href:
                continue
            if is_pdf(href):
                if same_domain(href, ALLOWED_NETLOC):
                    pdf_links.append(href)
                continue
            if same_domain(href, ALLOWED_NETLOC) and href not in seen:
                q.append(href)
                
    except Exception:
        seen.add(url)

def crawl(start_url=BASE_URL):
    session = requests_session()
    seen = set()
    html_pages = []
    pdf_links = []
    q = deque([start_url])
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        with tqdm(total=MAX_PAGES, desc="Crawling", unit="page") as pbar:
            while q and len(seen) < MAX_PAGES:
                # Get batch of URLs to process
                batch_size = min(8, len(q), MAX_PAGES - len(seen))
                if batch_size <= 0:
                    break
                    
                current_batch = [q.popleft() for _ in range(batch_size)]
                
                # Process batch in parallel
                futures = []
                for url in current_batch:
                    futures.append(
                        executor.submit(
                            process_url, url, session, seen, 
                            html_pages, pdf_links, q
                        )
                    )
                
                # Wait for batch to complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        pass
                
                pbar.update(len(current_batch))

    pdf_links = sorted(list(set(pdf_links)))
    html_pages = sorted(list(set(html_pages)))
    return html_pages, pdf_links

if __name__ == "__main__":
    pages, pdfs = crawl()
    print("Found HTML pages:", len(pages))
    print("Found PDF links:", len(pdfs))