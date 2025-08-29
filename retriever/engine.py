# retriever/engine.py
import os, json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from typing import List, Dict, Any, Optional
import re
from config import EMBED_MODEL, QA_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TORCH_DEVICE
import torch

INDEX_DIR = "data/index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.jsonl")

class RetrieverQA:
    def __init__(self, snippets_only=False):
        # Initialize embedder with GPU support
        self.embedder = SentenceTransformer(EMBED_MODEL, device=TORCH_DEVICE)
        self.embedder.max_seq_length = 512  # Optimize for retrieval
        
        # Load index with GPU support if available
        if torch.cuda.is_available():
            print("Loading FAISS index with GPU support")
            self.index = faiss.index_cpu_to_all_gpus(faiss.read_index(FAISS_PATH))
        else:
            self.index = faiss.read_index(FAISS_PATH)
            
        self.metas = []
        with open(META_PATH, "r", encoding="utf-8") as f:
            for line in f:
                self.metas.append(json.loads(line))
        self.snippets_only = snippets_only
        self.qa = None
        
        if not snippets_only:
            try:
                # Set device for QA model (GPU if available, otherwise CPU)
                qa_device = 0 if torch.cuda.is_available() else -1
                tok = AutoTokenizer.from_pretrained(QA_MODEL)
                model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
                self.qa = pipeline(
                    "question-answering", 
                    model=model, 
                    tokenizer=tok,
                    device=qa_device,  # Use GPU if available
                    max_seq_length=384,
                    doc_stride=128
                )
            except Exception as e:
                print(f"QA model loading failed: {e}. Falling back to snippet mode.")
                self.snippets_only = True

    def preprocess_query(self, query: str) -> str:
        """Clean and enhance the query for better retrieval"""
        query = query.lower().strip()
        # Remove question words for better semantic matching
        question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'do', 'does', 'did', 'can', 'could']
        words = query.split()
        filtered_words = [word for word in words if word not in question_words]
        return ' '.join(filtered_words) if filtered_words else query

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced retrieval with better scoring and filtering"""
        # Preprocess query
        processed_query = self.preprocess_query(query)
        qv = self.embedder.encode(
            [processed_query], 
            convert_to_numpy=True, 
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")
        
        # Retrieve more candidates initially for better filtering
        scores, indices = self.index.search(qv, min(k * 3, len(self.metas)))
        
        hits = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= len(self.metas):  # Safety check
                continue
                
            rec = self.metas[idx]
            text = rec.get("text", "")
            
            # Filter out very low similarity scores
            if score < 0.1:  # Adjust threshold as needed
                continue
                
            # Calculate relevance score (combine semantic and content-based scoring)
            relevance_score = self.calculate_relevance_score(query, text, score)
            
            hits.append({
                "score": float(score),
                "relevance_score": float(relevance_score),
                "title": rec.get("title") or rec.get("url", ""),
                "url": rec.get("url", ""),
                "text": text,
                "type": rec.get("type", "unknown"),
                "length": len(text)
            })
        
        # Sort by relevance score and return top k
        hits.sort(key=lambda x: x["relevance_score"], reverse=True)
        return hits[:k]

    def calculate_relevance_score(self, query: str, text: str, semantic_score: float) -> float:
        """Calculate combined relevance score"""
        # Semantic similarity (primary)
        semantic_weight = 0.7
        
        # Keyword overlap (secondary)
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        keyword_overlap = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
        keyword_weight = 0.3
        
        # Text quality bonus (longer, cleaner text)
        quality_bonus = min(1.0, len(text) / 1000) * 0.1  # Bonus for longer texts
        
        return (semantic_score * semantic_weight + 
                keyword_overlap * keyword_weight + 
                quality_bonus)

    def extract_best_passage(self, question: str, text: str, window_size: int = 400) -> str:
        """Extract the most relevant passage from longer texts"""
        if len(text) <= window_size:
            return text
            
        # Split text into overlapping windows
        passages = []
        for i in range(0, len(text), window_size // 2):
            passage = text[i:i + window_size]
            passages.append(passage)
        
        # Find most relevant passage
        best_passage = ""
        best_score = -1
        
        for passage in passages:
            # Simple keyword matching for passage selection
            question_words = set(question.lower().split())
            passage_words = set(passage.lower().split())
            overlap = len(question_words.intersection(passage_words))
            
            if overlap > best_score:
                best_score = overlap
                best_passage = passage
        
        return best_passage if best_passage else text[:window_size]

    def answer(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Enhanced answer generation with better context handling"""
        hits = self.retrieve(question, k=k)
        
        if self.snippets_only or self.qa is None:
            return {
                "answer": None, 
                "snippets": hits,
                "confidence": 0.0
            }
        
        best_answer = None
        all_answers = []
        
        for h in hits:
            context = h.get("text", "")
            if not context.strip():
                continue
                
            try:
                # Use shorter, more focused context for QA
                focused_context = self.extract_best_passage(question, context)
                
                result = self.qa({
                    "question": question, 
                    "context": focused_context,
                    "max_answer_len": 100,  # Limit answer length
                    "handle_impossible_answer": True
                })
                
                if result["score"] > 0.01:  # Minimum confidence threshold
                    answer_data = {
                        "answer": result["answer"],
                        "score": float(result["score"]),
                        "source": h,
                        "context_snippet": focused_context[:200] + "..." if len(focused_context) > 200 else focused_context
                    }
                    all_answers.append(answer_data)
                    
                    if best_answer is None or result["score"] > best_answer["score"]:
                        best_answer = answer_data
                        
            except Exception as e:
                print(f"QA error for context: {e}")
                continue
        
        # If we have answers, return the best one with supporting evidence
        if best_answer:
            return {
                "answer": best_answer["answer"],
                "score": best_answer["score"],
                "snippets": hits,
                "source": best_answer["source"],
                "confidence": self.calculate_confidence(best_answer["score"], len(all_answers)),
                "alternative_answers": all_answers[:3]  # Top alternative answers
            }
        
        # Fallback: return most relevant snippet as answer
        if hits and hits[0]["text"]:
            return {
                "answer": hits[0]["text"][:150] + "..." if len(hits[0]["text"]) > 150 else hits[0]["text"],
                "score": hits[0]["relevance_score"],
                "snippets": hits,
                "confidence": hits[0]["relevance_score"] * 0.7,  # Lower confidence for snippet answers
                "is_snippet": True
            }
        
        return {
            "answer": "I couldn't find a specific answer to your question in the available documents.",
            "score": 0.0,
            "snippets": hits,
            "confidence": 0.0,
            "no_answer": True
        }

    def calculate_confidence(self, qa_score: float, num_answers: int) -> float:
        """Calculate overall confidence score"""
        base_confidence = qa_score
        
        # Boost confidence if multiple sources agree
        agreement_boost = min(0.2, num_answers * 0.05)
        
        return min(1.0, base_confidence + agreement_boost)

    def get_similar_documents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Get similar documents for exploratory search"""
        return self.retrieve(query, k=k)

    def batch_retrieve(self, queries: List[str], k: int = 3) -> List[List[Dict[str, Any]]]:
        """Batch processing for multiple queries"""
        processed_queries = [self.preprocess_query(q) for q in queries]
        query_vectors = self.embedder.encode(
            processed_queries, 
            convert_to_numpy=True, 
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")
        
        scores, indices = self.index.search(query_vectors, k)
        
        results = []
        for i, (query_scores, query_indices) in enumerate(zip(scores, indices)):
            query_hits = []
            for idx, score in zip(query_indices, query_scores):
                if idx < len(self.metas):
                    rec = self.metas[idx]
                    query_hits.append({
                        "score": float(score),
                        "title": rec.get("title") or rec.get("url", ""),
                        "url": rec.get("url", ""),
                        "text": rec.get("text", "")[:200] + "..." if len(rec.get("text", "")) > 200 else rec.get("text", "")
                    })
            results.append(query_hits)
        
        return results

# Utility function for initialization
def initialize_retriever(snippets_only=False):
    """Initialize the retriever with error handling"""
    try:
        if not os.path.exists(FAISS_PATH):
            raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}")
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(f"Metadata file not found at {META_PATH}")
            
        return RetrieverQA(snippets_only=snippets_only)
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        # Fallback: create a simple retriever that returns empty results
        class FallbackRetriever:
            def retrieve(self, query, k=3):
                return []
            def answer(self, question, k=3):
                return {"answer": "System initialization failed.", "snippets": []}
        return FallbackRetriever()