import os
import glob
import faiss
import numpy as np
from typing import List
from openai import OpenAI

from .config import settings


try:
    client = OpenAI(api_key=settings.openai_api_key)
except Exception:
    client = None

def _embed(texts: List[str]) -> np.ndarray:
    if not client or not settings.openai_api_key:
        return np.zeros((len(texts), 1536), dtype="float32") # Return dummy embeddings if no API key (for testing without crashing)
        
    try:
        print("using embedding model")
        resp = client.embeddings.create(
            model=settings.embedding_model,
            input=texts
        )
        vectors = [d.embedding for d in resp.data]
        return np.array(vectors).astype("float32")
    except Exception as e:
        print(f"Embedding error: {e}")
        return np.zeros((len(texts), 1536), dtype="float32")

class RAGRetriever:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.index = None
        self.chunks: List[str] = []

    def build_index(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            return

        files = glob.glob(os.path.join(self.data_dir, "*.md"))
        texts = []
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                content = fh.read()
            
            pieces = [content[i:i+800] for i in range(0, len(content), 800)] # naive chunking
            texts.extend(pieces)

        self.chunks = texts
        if not texts:
            return

        vecs = _embed(texts)
        if vecs.shape[0] > 0:
            dim = vecs.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(vecs)

    def retrieve(self, query: str, k: int = 4) -> str:
        if self.index is None or not self.chunks:
            self.build_index()
        if self.index is None or not self.chunks:
            return ""

        q_vec = _embed([query])
        _, I = self.index.search(q_vec, k)
        
        valid_indices = [i for i in I[0] if 0 <= i < len(self.chunks)] # Handle index out of bounds if k > n_samples
        retrieved = [self.chunks[i] for i in valid_indices]
        return "\n\n".join(retrieved)

rag_retriever = RAGRetriever()