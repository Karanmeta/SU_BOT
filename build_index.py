# build_index.py
import os, json
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL, TORCH_DEVICE
import torch

DOCS_JSONL = "data/processed/docs.jsonl"
CHUNKS_JSONL = "data/processed/chunks.jsonl"
INDEX_DIR = "data/index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.jsonl")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        start = start + size - overlap
    return chunks

def build_chunks():
    ensure_dir(os.path.dirname(CHUNKS_JSONL))
    with open(DOCS_JSONL, "r", encoding="utf-8") as f, open(CHUNKS_JSONL, "w", encoding="utf-8") as out:
        for line in f:
            rec = json.loads(line)
            txt = rec.get("text","")
            if not txt:
                continue
            chunks = chunk_text(txt)
            for i,ch in enumerate(chunks):
                out.write(json.dumps({
                    "doc_id": rec.get("id"),
                    "url": rec.get("url"),
                    "title": rec.get("title",""),
                    "type": rec.get("type",""),
                    "chunk_id": i,
                    "text": ch
                }, ensure_ascii=False) + "\n")
    print("Chunks saved to", CHUNKS_JSONL)

def build_faiss():
    ensure_dir(INDEX_DIR)
    
    # Initialize model with GPU support
    model = SentenceTransformer(EMBED_MODEL, device=TORCH_DEVICE)
    
    # Set model to use GPU if available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        model = model.to(TORCH_DEVICE)
    else:
        print("Using CPU for embeddings")
    
    metas = []
    vectors = []
    batch_texts = []
    batch_metas = []
    
    # First, count total chunks for progress bar
    total_chunks = 0
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for _ in f:
            total_chunks += 1

    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing chunks", total=total_chunks):
            rec = json.loads(line)
            batch_texts.append(rec["text"])
            batch_metas.append(rec)
            
            # Process in larger batches for GPU efficiency
            if len(batch_texts) >= 256:
                vecs = model.encode(
                    batch_texts, 
                    convert_to_numpy=True, 
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=64  # Optimize for GPU
                )
                vectors.append(vecs)
                metas.extend(batch_metas)
                batch_texts, batch_metas = [], []
        
        # Process remaining texts
        if batch_texts:
            vecs = model.encode(
                batch_texts, 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64
            )
            vectors.append(vecs)
            metas.extend(batch_metas)

    if not vectors:
        raise SystemExit("No chunks to index. Run scraper first.")

    X = np.vstack(vectors).astype("float32")
    d = X.shape[1]
    
    # Use GPU for FAISS if available
    if torch.cuda.is_available():
        print("Using GPU for FAISS indexing")
        # Create a GPU index
        res = faiss.StandardGpuResources()
        cpu_index = faiss.IndexFlatIP(d)
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        print("Using CPU for FAISS indexing")
        index = faiss.IndexFlatIP(d)  # inner product on normalized vectors ~ cosine
    
    index.add(X)
    faiss.write_index(index, FAISS_PATH)
    
    with open(META_PATH, "w", encoding="utf-8") as m:
        for rec in metas:
            m.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print("FAISS index saved to", FAISS_PATH)
    print("Metadata saved to", META_PATH)
    print(f"Index contains {len(metas)} chunks with {d}-dimensional vectors")

if __name__ == "__main__":
    build_chunks()
    build_faiss()