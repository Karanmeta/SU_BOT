import os
import glob
import asyncio
from typing import List, Dict
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

INDEX_DIR = ".cache/scet_index"

def _ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

def _collect_docs(data_dir: str) -> List[Document]:
    docs = []
    for path in glob.glob(os.path.join(data_dir, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if not text:
                continue
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(path)}))
    return docs

def build_or_load_local_retriever(data_dir: str = "data/scet"):
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not files:
        print("⚠️ No SCET documents found.")
        return None

    _ensure_event_loop()

    # ✅ Local HuggingFace embeddings (no API or quota)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.isdir(INDEX_DIR) and glob.glob(os.path.join(INDEX_DIR, "*")):
        try:
            vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            return vs.as_retriever(search_kwargs={"k": 5})
        except Exception:
            pass

    docs = _collect_docs(data_dir)
    if not docs:
        return None

    vs = FAISS.from_documents(docs, embeddings)
    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)
    print(f"✅ Indexed {len(docs)} SCET documents locally.")
    return vs.as_retriever(search_kwargs={"k": 5})

def retrieve_local(query: str, retriever):
    """Retrieve relevant local docs."""
    if retriever is None:
        return []
    try:
        results = retriever.get_relevant_documents(query)
    except Exception as e:
        print(f"⚠️ Local retrieval failed: {e}")
        return []

    formatted = []
    for r in results:
        formatted.append({
            "title": r.metadata.get("source", "SCET Local"),
            "url": "",
            "content": r.page_content[:1200],
            "source": r.metadata.get("source", "SCET Local"),
            "kind": "local"
        })
    return formatted
