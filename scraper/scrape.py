# scraper/scrape.py
import json, os, re, requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from .utils import ensure_dir, rate_limit
from .crawl import crawl
from config import REQUEST_DELAY

DATA_DIR = "data"
RAW_HTML_DIR = os.path.join(DATA_DIR, "raw", "html")
PROCESSED = os.path.join(DATA_DIR, "processed", "docs.jsonl")

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def extract_text_from_html(html: str):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(["nav", "footer", "header", "form"]):
        tag.decompose()
    title = clean_text(soup.title.string if soup.title else "")
    pieces = []
    for el in soup.find_all(["h1","h2","h3","p","li","td"]):
        t = clean_text(el.get_text(separator=" ", strip=True))
        if t and len(t) > 30:
            pieces.append(t)
    text = "\n".join(pieces)
    return title, text

def run_scrape():
    ensure_dir(RAW_HTML_DIR)
    ensure_dir(os.path.join(DATA_DIR, "processed"))
    pages, pdfs = crawl()

    with open(PROCESSED, "w", encoding="utf-8") as out:
        for url in tqdm(pages, desc="Saving HTML"):
            try:
                rate_limit(REQUEST_DELAY)
                r = requests.get(url, timeout=15)
                if r.status_code != 200 or "text/html" not in (r.headers.get("Content-Type") or ""):
                    continue
                fname = url.replace("https://","").replace("http://","").replace("/","_")
                safe_fname = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', fname)
                with open(os.path.join(RAW_HTML_DIR, f"{safe_fname}.html"), "w", encoding="utf-8", errors="ignore") as f:
                    f.write(r.text)
                title, text = extract_text_from_html(r.text)
                if not text or len(text) < 60:
                    continue
                rec = {"id": safe_fname, "url": url, "title": title, "type": "html", "text": text}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                continue

        # write pdf placeholders (actual download/extraction in pdf_extract)
        for link in pdfs:
            rec = {"id": os.path.basename(link), "url": link, "title": "", "type": "pdf", "text": ""}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Saved processed docs to", PROCESSED)

if __name__ == "__main__":
    run_scrape()
