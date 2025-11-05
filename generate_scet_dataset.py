import os
import time
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

BASE_URL = "https://scet.ac.in"
OUTPUT_DIR = "data/scet_selenium"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_driver():
    options = Options()
    options.add_argument("--headless")  # Run Chrome in background
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("window-size=1920x1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def extract_page(driver, url):
    driver.get(url)
    time.sleep(2)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.text.strip() if soup.title else "Untitled"
    text_blocks = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = tag.get_text(" ", strip=True)
        if text and len(text) > 30:
            text_blocks.append(text)

    return {
        "url": url,
        "title": title,
        "content": "\n\n".join(text_blocks)
    }

def save_page(data):
    fname = os.path.join(OUTPUT_DIR, data["title"].replace("/", "_")[:80] + ".txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {data['title']}\nURL: {data['url']}\n\n{data['content']}")
    print(f"✅ Saved: {fname}")

def main():
    print("🚀 Launching Chrome to scrape SCET site...")
    driver = setup_driver()
    driver.get(BASE_URL)
    time.sleep(3)

    # Collect all visible links
    soup = BeautifulSoup(driver.page_source, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(BASE_URL) or href.startswith("/"):
            full = href if href.startswith("http") else BASE_URL + href
            if "mailto" not in full and "javascript" not in full:
                links.append(full)

    links = sorted(set(links))
    print(f"🔗 Found {len(links)} links to scrape")

    pages = []
    for link in links[:50]:  # limit to 50 pages
        try:
            data = extract_page(driver, link)
            if len(data["content"]) > 200:
                save_page(data)
                pages.append(data)
        except Exception as e:
            print(f"⚠️ Error scraping {link}: {e}")

    driver.quit()

    with open(os.path.join(OUTPUT_DIR, "_index.json"), "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2)

    print(f"🗂️ Saved {len(pages)} SCET pages in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
