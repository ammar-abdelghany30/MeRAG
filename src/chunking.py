"""chunking.py - clean documents and split them into small, citable chunks."""
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)      # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)   
    return text.strip()


def chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  
    )

    cleaned = [
        Document(page_content=clean_text(d.page_content), metadata=dict(d.metadata))
        for d in docs
        if d.page_content.strip()
    ]

    chunks = splitter.split_documents(cleaned)  

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


if __name__ == "__main__":
    from src.loader import load_all

    chunks = chunk_documents(load_all())
    print(f"{len(chunks)} chunks")
    for c in chunks[:3]:
        print(c.metadata, "->", c.page_content[:150].replace("\n", " "))