"""vectorstore.py - embed chunks, save/load a FAISS index, and search it."""
import json
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
STORE_DIR = Path("vectorstore")
INDEX_FILE = STORE_DIR / "index.faiss"
CHUNKS_FILE = STORE_DIR / "chunks.json"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

import re

NAME_WORDS = ["ammar", "abdelghany", "mohamed"]
_NAME_RE = re.compile(r"\b(" + "|".join(NAME_WORDS) + r")\b", re.IGNORECASE)

def _strip_name(query):
    cleaned = _NAME_RE.sub("", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or query  

_model = None
_store = None  


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _embed(texts):
    vecs = _get_model().encode(
        texts, normalize_embeddings=True, show_progress_bar=len(texts) > 50
    )
    return np.asarray(vecs, dtype="float32")


def build_index(chunks):
    """Embed all chunks and save the index + chunk texts/metadata to disk."""
    global _store
    STORE_DIR.mkdir(exist_ok=True)

    vectors = _embed([c.page_content for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])  
    index.add(vectors)
    faiss.write_index(index, str(INDEX_FILE))

    records = [{"text": c.page_content, "metadata": c.metadata} for c in chunks]
    CHUNKS_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _store = None  # force a reload next time
    return index


def _load_store():
    global _store
    if _store is None:
        index = faiss.read_index(str(INDEX_FILE))
        records = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        _store = (index, records)
    return _store


def search(query, k=6):
    """Return the k most relevant chunks as (Document, similarity_score)."""
    index, records = _load_store()
    scores, ids = index.search(_embed([QUERY_PREFIX + _strip_name(query)]), k)

    results = []
    for score, i in zip(scores[0], ids[0]):
        if i == -1:  # FAISS returns -1 when it has fewer than k results
            continue
        item = records[i]
        doc = Document(page_content=item["text"], metadata=item["metadata"])
        results.append((doc, float(score)))
    return results


if __name__ == "__main__":
    from src.chunking import chunk_documents
    from src.loader import load_all

    chunks = chunk_documents(load_all())
    build_index(chunks)
    print(f"Indexed {len(chunks)} chunks -> {INDEX_FILE}")

    query = "Where did Ammar intern and what did he work on?"
    print(f"\nQuery: {query}")
    for doc, score in search(query):
        print(f"\n[{score:.2f}] {doc.metadata}")
        print(doc.page_content[:200].replace("\n", " "))