import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import time

# --- CONFIGURATION ---
# MODIFIED: Changed BASE_URL to the actual starting page of the website.
BASE_URL = "https://scet.ac.in/"
# MODIFIED: The domain is used to ensure the crawler stays on the target site.
DOMAIN = urlparse(BASE_URL).netloc
MAX_PAGES_TO_CRAWL = 500 # Increased limit to ensure full site crawl
PERSIST_DIRECTORY = "db_chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
# --- END CONFIGURATION ---

# --- Global set to keep track of visited URLs ---
visited_urls = set()

def is_valid_url(url):
    """Checks if a URL is a valid HTTP/HTTPS link and within the same domain."""
    parsed_url = urlparse(url)
    # MODIFIED: Correctly checks if the URL has a scheme and matches the target domain.
    return bool(parsed_url.scheme) and parsed_url.netloc == DOMAIN

def crawl_website(url, urls_to_visit):
    """
    Crawls a single page and adds new, valid links to the set of URLs to visit.
    """
    if url in visited_urls or len(visited_urls) >= MAX_PAGES_TO_CRAWL:
        return

    print(f"Crawling: {url}")
    visited_urls.add(url)
    
    try:
        response = requests.get(url, timeout=10, headers=HEADERS)
        response.raise_for_status()
        time.sleep(1) # Be polite to the server by waiting between requests

        soup = BeautifulSoup(response.content, 'html.parser')

        for link in soup.find_all('a', href=True):
            href = link['href']
            # MODIFIED: Robustly join URLs to handle absolute and relative paths correctly.
            full_url = urljoin(url, href).split('#')[0] # Use the current page URL for joining
            
            if is_valid_url(full_url) and full_url not in visited_urls and full_url not in urls_to_visit:
                urls_to_visit.add(full_url)

    except (requests.RequestException, Exception) as e:
        print(f"Error crawling {url}: {e}")

def load_documents_from_urls(urls):
    """Loads content from URLs, handling HTML and PDF files."""
    documents = []
    for url in tqdm(urls, desc="Loading documents"):
        try:
            # Check content type to be more reliable than file extension
            head_response = requests.head(url, timeout=10, headers=HEADERS)
            content_type = head_response.headers.get('Content-Type', '')

            if 'application/pdf' in content_type:
                response = requests.get(url, stream=True, timeout=30, headers=HEADERS)
                response.raise_for_status()
                
                # MODIFIED: Use a unique temporary file name to avoid conflicts
                temp_pdf_path = f"temp_{os.path.basename(urlparse(url).path)}.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(response.content)
                
                loader = PyPDFLoader(temp_pdf_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = url
                documents.extend(docs)
                os.remove(temp_pdf_path)
            elif 'text/html' in content_type:
                response = requests.get(url, timeout=10, headers=HEADERS)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # MODIFIED: More selective tag removal to preserve meaningful content
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()
                page_text = soup.get_text(separator='\n', strip=True)
                
                if page_text:
                    documents.append(Document(page_content=page_text, metadata={'source': url}))
        except Exception as e:
            print(f"Failed to load or process {url}: {e}")
    return documents

if __name__ == "__main__":
    print("--- Starting Step 1: Crawling Website ---")
    # MODIFIED: Use a queue-based approach for breadth-first crawling
    urls_to_visit = {BASE_URL}
    while urls_to_visit and len(visited_urls) < MAX_PAGES_TO_CRAWL:
        url = urls_to_visit.pop()
        crawl_website(url, urls_to_visit)

    print(f"\nCrawling complete. Found {len(visited_urls)} unique pages.")

    print("\n--- Starting Step 2: Loading Document Content ---")
    all_docs = load_documents_from_urls(list(visited_urls))
    print(f"Loading complete. Loaded {len(all_docs)} documents.")

    if not all_docs:
        print("No documents were loaded. Exiting.")
    else:
        print("\n--- Starting Step 3: Chunking Documents ---")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunked_docs = text_splitter.split_documents(all_docs)
        print(f"Chunking complete. Created {len(chunked_docs)} text chunks.")

        print("\n--- Starting Step 4: Creating and Saving Embeddings ---")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        db = Chroma.from_documents(
            chunked_docs, 
            embeddings, 
            persist_directory=PERSIST_DIRECTORY
        )
        print(f"\n--- Ingestion Complete! ---")
        print(f"Vector database created and saved in '{PERSIST_DIRECTORY}' directory.")
