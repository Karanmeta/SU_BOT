# scraper/pdf_extract.py
import os
import json
import re
import pdfplumber
import requests
import warnings
from tqdm import tqdm
from typing import Optional
from urllib.parse import urlparse
from .utils import ensure_dir, rate_limit
import concurrent.futures

# Optional imports with availability checks
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Config with fallbacks
try:
    from config import REQUEST_DELAY, PDF_EXTRACTION_METHOD, SUPPRESS_PDF_WARNINGS, PDF_FALLBACK_ORDER, MIN_TEXT_LENGTH, EASYOCR_LANGUAGES, EASYOCR_GPU, EASYOCR_MODEL_STORAGE_DIRECTORY, USE_GPU
except ImportError:
    REQUEST_DELAY = 1.0
    PDF_EXTRACTION_METHOD = "auto"
    SUPPRESS_PDF_WARNINGS = True
    PDF_FALLBACK_ORDER = ["pdfplumber", "fitz", "pypdf2", "easyocr"]
    MIN_TEXT_LENGTH = 100
    EASYOCR_LANGUAGES = ['en']
    EASYOCR_GPU = True
    EASYOCR_MODEL_STORAGE_DIRECTORY = '~/.EasyOCR/model'
    USE_GPU = True

DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "raw", "pdfs")
DOCS_JSONL = os.path.join(DATA_DIR, "processed", "docs.jsonl")

# Global EasyOCR reader for GPU efficiency
_easyocr_reader = None

def get_easyocr_reader():
    """Get or create the EasyOCR reader (singleton pattern)"""
    global _easyocr_reader
    if _easyocr_reader is None and EASYOCR_AVAILABLE:
        try:
            _easyocr_reader = easyocr.Reader(
                EASYOCR_LANGUAGES,
                gpu=EASYOCR_GPU and USE_GPU,
                model_storage_directory=EASYOCR_MODEL_STORAGE_DIRECTORY
            )
        except Exception as e:
            print(f"EasyOCR initialization failed: {e}")
            _easyocr_reader = None
    return _easyocr_reader

def download_pdf(url: str, path: str):
    """Download PDF with better error handling"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/pdf, */*',
        'Accept-Encoding': 'gzip, deflate',
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=30, stream=True)
        r.raise_for_status()
        
        # Check if it's actually a PDF
        content_type = r.headers.get('content-type', '').lower()
        if 'pdf' not in content_type:
            print(f"Warning: {url} doesn't seem to be a PDF (Content-Type: {content_type})")
        
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download PDF {url}: {e}")
        return False

def clean_pdf_text(text: str) -> str:
    """Clean and normalize PDF text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove page numbers and headers/footers
    text = re.sub(r'\bPage\s*\d+\s*of\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text)  # Page X/Y
    
    # Remove common PDF artifacts
    artifacts = [
        r'http[s]?://\S+',  # URLs
        r'©.*\d{4}',        # Copyright
        r'Confidential|Proprietary',
    ]
    
    for pattern in artifacts:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text

def _extract_with_pdfplumber(path: str) -> str:
    """Extract text using pdfplumber"""
    texts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                t = clean_pdf_text(t)
                if t and len(t) > 10:
                    texts.append(t)
    except Exception as e:
        print(f"pdfplumber extraction failed: {e}")
        return ""
    return "\n".join(texts)

def _extract_with_pypdf2(path: str) -> str:
    """Extract text using PyPDF2"""
    if not PYPDF2_AVAILABLE:
        return ""
    
    texts = []
    try:
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                t = page.extract_text() or ""
                t = clean_pdf_text(t)
                if t and len(t) > 10:
                    texts.append(t)
    except Exception as e:
        print(f"PyPDF2 extraction failed: {e}")
        return ""
    return "\n".join(texts)

def _extract_with_fitz(path: str) -> str:
    """Extract text using PyMuPDF (fitz)"""
    if not FITZ_AVAILABLE:
        return ""
    
    texts = []
    try:
        doc = fitz.open(path)
        for page in doc:
            t = page.get_text() or ""
            t = clean_pdf_text(t)
            if t and len(t) > 10:
                texts.append(t)
        doc.close()
    except Exception as e:
        print(f"PyMuPDF extraction failed: {e}")
        return ""
    return "\n".join(texts)

def _extract_with_easyocr(path: str) -> str:
    """Extract text using EasyOCR"""
    reader = get_easyocr_reader()
    if not reader:
        return ""
    
    texts = []
    try:
        # Convert PDF to images
        images = convert_from_path(path, dpi=300)
        
        # Process each image with EasyOCR
        for image in images:
            results = reader.readtext(image, paragraph=True)
            page_text = " ".join([result[1] for result in results])
            page_text = clean_pdf_text(page_text)
            if page_text and len(page_text) > 50:
                texts.append(page_text)
                
    except Exception as e:
        print(f"EasyOCR extraction failed: {e}")
        return ""
    
    return "\n".join(texts)

def extract_pdf_text_standard(path: str, method: Optional[str] = None) -> str:
    """Extract text from PDF using specified method or auto-detect"""
    if method is None:
        method = PDF_EXTRACTION_METHOD
    
    if SUPPRESS_PDF_WARNINGS:
        warnings.filterwarnings("ignore", message="Cannot set gray non-stroke color")
    
    methods = {
        "pdfplumber": _extract_with_pdfplumber,
        "pypdf2": _extract_with_pypdf2,
        "fitz": _extract_with_fitz,
        "easyocr": _extract_with_easyocr
    }
    
    # Filter available methods
    available_methods = {}
    for name, func in methods.items():
        if (name == "pdfplumber" or 
            (name == "pypdf2" and PYPDF2_AVAILABLE) or 
            (name == "fitz" and FITZ_AVAILABLE) or
            (name == "easyocr" and EASYOCR_AVAILABLE)):
            available_methods[name] = func
    
    if method == "auto":
        for method_name in PDF_FALLBACK_ORDER:
            if method_name in available_methods:
                try:
                    text = available_methods[method_name](path)
                    if text and len(text.strip()) > MIN_TEXT_LENGTH:
                        print(f"Success with {method_name}: {len(text)} characters")
                        return text
                except Exception as e:
                    print(f"{method_name} failed: {e}")
                    continue
        return ""
    elif method in available_methods:
        return available_methods[method](path)
    else:
        return ""

def extract_pdf_text(path: str) -> str:
    """Main PDF text extraction function"""
    # Try standard methods first
    text = extract_pdf_text_standard(path, "auto")
    
    # Fallback to OCR if standard methods fail
    if not text or len(text) < MIN_TEXT_LENGTH:
        print(f"Standard extraction insufficient ({len(text)} chars), trying OCR...")
        ocr_text = _extract_with_easyocr(path)
        if ocr_text and len(ocr_text) > len(text):
            text = ocr_text
            print(f"OCR extracted {len(text)} characters")
    
    return text

def process_single_pdf(rec):
    """Process a single PDF record"""
    url = rec.get("url")
    parsed_url = urlparse(url)
    fname = os.path.basename(parsed_url.path) or "document.pdf"
    safe_fname = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', fname)
    local_path = os.path.join(PDF_DIR, safe_fname)
    
    try:
        rate_limit(REQUEST_DELAY)
        
        # Download if not exists
        if not os.path.exists(local_path):
            print(f"Downloading: {fname}")
            if not download_pdf(url, local_path):
                return None
        
        # Check file size
        if os.path.getsize(local_path) < 1024:  # Skip very small files
            print(f"Skipping small PDF: {fname}")
            return None
        
        # Extract text
        text = extract_pdf_text(local_path)
        
        if text and len(text) > MIN_TEXT_LENGTH:
            rec["text"] = text
            rec["title"] = rec.get("title") or safe_fname.replace('.pdf', '').replace('_', ' ')
            rec["length"] = len(text)
            print(f"✓ Processed: {fname} ({len(text)} chars)")
            return rec
        else:
            print(f"✗ Failed: {fname} (only {len(text) if text else 0} chars)")
            return None
            
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def run_pdf_pipeline():
    ensure_dir(PDF_DIR)
    records = []
    pdf_items = []
    
    if not os.path.exists(DOCS_JSONL):
        print("No processed docs file:", DOCS_JSONL)
        return

    # Read existing records
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "pdf" and rec.get("url"):
                pdf_items.append(rec)
            else:
                records.append(rec)

    # Process PDFs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single_pdf, rec) for rec in pdf_items]
        
        for future in tqdm(
            concurrent.futures.as_completed(futures), 
            total=len(futures), 
            desc="Processing PDFs"
        ):
            result = future.result()
            if result:
                records.append(result)

    # Write back updated records
    with open(DOCS_JSONL, "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"PDF extraction completed. Processed {len([r for r in records if r.get('type') == 'pdf' and r.get('text')])} PDFs")

if __name__ == "__main__":
    run_pdf_pipeline()