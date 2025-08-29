# scraper/web_extractor.py
import json
import os
import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urlparse
from .utils import ensure_dir, rate_limit
from config import REQUEST_DELAY, SELENIUM_HEADLESS, SELENIUM_TIMEOUT, WEBDRIVER_PATH
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import concurrent.futures
import threading

DATA_DIR = "data"
RAW_HTML_DIR = os.path.join(DATA_DIR, "raw", "html")
PROCESSED = os.path.join(DATA_DIR, "processed", "docs.jsonl")

# Thread-local storage for WebDriver instances
thread_local = threading.local()

def get_driver():
    """Get a WebDriver instance for the current thread"""
    if not hasattr(thread_local, "driver"):
        chrome_options = Options()
        if SELENIUM_HEADLESS:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")
        
        try:
            thread_local.driver = webdriver.Chrome(
                executable_path=WEBDRIVER_PATH,
                options=chrome_options
            )
            thread_local.driver.set_page_load_timeout(SELENIUM_TIMEOUT)
        except WebDriverException as e:
            print(f"WebDriver initialization failed: {e}")
            # Fallback to requests
            thread_local.driver = None
    return thread_local.driver

def close_driver():
    """Close the WebDriver for the current thread"""
    if hasattr(thread_local, "driver") and thread_local.driver:
        try:
            thread_local.driver.quit()
        except:
            pass
        del thread_local.driver

class WebsiteExtractor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        })
    
    def clean_text(self, s: str) -> str:
        """Clean and normalize extracted text"""
        if not s:
            return ""
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    def extract_with_selenium(self, url: str) -> tuple:
        """Extract content using Selenium WebDriver"""
        driver = get_driver()
        if not driver:
            # Fallback to requests if WebDriver is not available
            try:
                response = self.session.get(url, timeout=15)
                soup = BeautifulSoup(response.text, "lxml")
                return self.extract_main_content(soup), response.text
            except Exception:
                return "", ""
        
        try:
            driver.get(url)
            # Wait for page to load
            WebDriverWait(driver, SELENIUM_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page source
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, "lxml")
            
            # Extract main content
            content = self.extract_main_content(soup)
            return content, page_source
            
        except TimeoutException:
            print(f"Timeout loading {url}")
            return "", driver.page_source if driver else ""
        except Exception as e:
            print(f"Selenium error for {url}: {e}")
            return "", driver.page_source if driver else ""
    
    def extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content using multiple strategies"""
        # Strategy 1: Try semantic tags first
        semantic_selectors = ['article', 'main', '[role="main"]', '.content', '.main-content']
        for selector in semantic_selectors:
            elements = soup.select(selector)
            if elements:
                text = ' '.join([elem.get_text(separator=' ', strip=True) for elem in elements])
                if len(text) > 200:  # Reasonable content length
                    return text
        
        # Strategy 2: Content density approach
        # Remove unwanted elements
        for unwanted in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
            unwanted.decompose()
        
        # Find elements with substantial text content
        elements = soup.find_all(['div', 'section'])
        best_element = None
        max_text_length = 0
        
        for element in elements:
            text = element.get_text(separator=' ', strip=True)
            if len(text) > max_text_length and len(text) > 100:
                max_text_length = len(text)
                best_element = element
        
        if best_element:
            return best_element.get_text(separator=' ', strip=True)
        
        # Strategy 3: Fallback to body text
        body = soup.find('body')
        if body:
            return body.get_text(separator=' ', strip=True)
        
        return ""
    
    def extract_metadata(self, soup: BeautifulSoup, url: str) -> dict:
        """Extract metadata from HTML"""
        metadata = {
            'title': '',
            'description': '',
            'url': url
        }
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = self.clean_text(title_tag.get_text())
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            metadata['description'] = self.clean_text(meta_desc['content'])
        
        # Open Graph title (for social media)
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content') and not metadata['title']:
            metadata['title'] = self.clean_text(og_title['content'])
        
        # If no title, try to extract from h1
        if not metadata['title']:
            h1 = soup.find('h1')
            if h1:
                metadata['title'] = self.clean_text(h1.get_text())
        
        return metadata
    
    def extract_from_url(self, url: str) -> dict:
        """Extract content from a single URL"""
        try:
            rate_limit(REQUEST_DELAY)
            
            # Use Selenium to extract content
            content, html_content = self.extract_with_selenium(url)
            
            if not content or len(content) < 60:
                return None
            
            soup = BeautifulSoup(html_content, "lxml") if html_content else None
            metadata = self.extract_metadata(soup, url) if soup else {"title": "", "description": "", "url": url}
            
            # Create safe filename
            fname = url.replace("https://", "").replace("http://", "").replace("/", "_")
            safe_fname = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', fname)
            
            # Save raw HTML
            ensure_dir(RAW_HTML_DIR)
            with open(os.path.join(RAW_HTML_DIR, f"{safe_fname}.html"), "w", encoding="utf-8", errors="ignore") as f:
                f.write(html_content if html_content else "")
            
            return {
                "id": safe_fname,
                "url": url,
                "title": metadata['title'],
                "description": metadata['description'],
                "type": "html",
                "text": content
            }
            
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None

def process_single_url(url, extractor):
    """Process a single URL (for parallel execution)"""
    return extractor.extract_from_url(url)

def run_web_extraction():
    """Run website extraction using the enhanced extractor"""
    from .crawl import crawl
    
    ensure_dir(RAW_HTML_DIR)
    ensure_dir(os.path.join(DATA_DIR, "processed"))
    
    # Get URLs to process
    pages, pdfs = crawl()
    
    extractor = WebsiteExtractor()
    processed_count = 0
    
    # Process URLs in parallel
    with open(PROCESSED, "w", encoding="utf-8") as out:
        # Process HTML pages in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(process_single_url, url, extractor): url 
                for url in pages
            }
            
            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(future_to_url), 
                total=len(pages), 
                desc="Extracting website content"
            ):
                result = future.result()
                if result:
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed_count += 1
        
        # Write PDF placeholders
        for link in pdfs:
            rec = {
                "id": os.path.basename(link),
                "url": link,
                "title": "",
                "type": "pdf",
                "text": ""
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    # Close all WebDriver instances
    close_driver()
    
    print(f"Extracted content from {processed_count} web pages. Saved to {PROCESSED}")

if __name__ == "__main__":
    run_web_extraction()