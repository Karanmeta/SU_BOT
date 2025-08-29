# config.py
BASE_URL = "https://scet.ac.in/"
MAX_PAGES = 7000
REQUEST_DELAY = 0.8
ALLOWED_NETLOC = "https://scet.ac.in/"
MAX_WORKERS = 8  # For parallel processing
THREADS_PER_WORKER = 2  # For Selenium multithreading

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "distilbert-base-cased-distilled-squad"

# PDF extraction settings
PDF_EXTRACTION_METHOD = "auto"
SUPPRESS_PDF_WARNINGS = True
PDF_FALLBACK_ORDER = ["pdfplumber", "fitz", "pypdf2", "easyocr"]
MIN_TEXT_LENGTH = 100

# Selenium settings
SELENIUM_HEADLESS = True
SELENIUM_TIMEOUT = 30
WEBDRIVER_PATH = "chromedriver"  # Update this path if needed

# GPU acceleration settings
USE_GPU = True
CUDA_DEVICE = "cuda:0"  # Use first GPU
TORCH_DEVICE = "cuda" if USE_GPU else "cpu"

# EasyOCR settings
EASYOCR_LANGUAGES = ['en']
EASYOCR_GPU = USE_GPU
EASYOCR_MODEL_STORAGE_DIRECTORY = '~/.EasyOCR/model'