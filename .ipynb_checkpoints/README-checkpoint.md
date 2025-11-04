su-chatbot/
├── app.py # Main FastAPI application
├── build_index.py # Builds FAISS vector index from documents
├── config.py # Configuration settings
├── requirements.txt # Python dependencies
├── readme.md # This file
├── scraper/ # Web scraping utilities
│ ├── init.py
│ ├── utils.py
│ ├── crawl.py # Web crawling functionality
│ ├── scrape.py # Main scraping orchestration
│ ├── pdf_extract.py # PDF text extraction
│ └── web_extractor.py # Web content extraction
├── retriever/ # Retrieval engine
│ ├── init.py
│ └── engine.py # Search and retrieval logic
├── data/ # Created at runtime - stores raw/scraped data
├── index/ # Created at runtime - stores FAISS index
├── processed/ # Created at runtime - processed documents

   
# 1. create virtualenv & activate
python -m venv venv
# (Linux/Mac)
source venv/bin/activate
# (Windows CMD)
venv\Scripts\activate.bat

# 2. save requirements.txt (below) and install
pip install -r requirements.txt

# 3. run full HTML crawl + save docs
python -m scraper.scrape

# 4. download PDFs + extract text
python -m scraper.pdf_extract

# 5. build chunks + embeddings + FAISS index
python -m build_index

# 6. start  app
python -m app


