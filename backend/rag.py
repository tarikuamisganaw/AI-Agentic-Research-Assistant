import os
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_WORDS, SIMILARITY_THRESHOLD, INDEX_PATH, META_PATH
from utils import clean_pdf_text

text_splitter = None
embedder = None
faiss_index = None
metadata = []

def initialize_models(embed_model: str):
    global text_splitter, embedder
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""], length_function=len, is_separator_regex=False
    )
    embedder = SentenceTransformer(embed_model)

def ingest_pdf(pdf_path: str) -> int:
    global faiss_index, metadata
    if not all([embedder, text_splitter]): raise RuntimeError("Models not initialized.")
    
    reader = PdfReader(pdf_path)
    pages_text = [{"page": i+1, "text": clean_pdf_text(p.extract_text())} for i, p in enumerate(reader.pages) if p.extract_text().strip()]
    chunks = [{"page": p["page"], "text": c, "word_count": len(c.split())} 
              for p in pages_text for c in text_splitter.split_text(p["text"]) 
              if len(c.split()) >= MIN_CHUNK_WORDS]
    if not chunks: raise ValueError("No readable text found.")
    
    vectors = embedder.encode([c["text"] for c in chunks], convert_to_numpy=True).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    faiss_index = faiss.IndexFlatIP(vectors.shape[1])
    faiss_index.add(vectors)
    faiss.write_index(faiss_index, INDEX_PATH)
    metadata = chunks
    with open(META_PATH, "w") as f: json.dump(metadata, f)
    return len(chunks)

def retrieve(query: str, is_summary: bool, k: int) -> list:
    global faiss_index, metadata, embedder
    if faiss_index is None or not metadata or embedder is None: return []
    
    fetch_k = 12 if is_summary else k * 2
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)
    distances, ids = faiss_index.search(query_vec, fetch_k)
    
    results, candidates = [], []
    for dist, idx in zip(distances[0], ids[0]):
        if idx == -1: continue
        txt = metadata[idx]["text"]
        if any(x in txt.lower() for x in ["references", "bibliography", "acknowledgments"]): continue
        if len(txt.split()) < MIN_CHUNK_WORDS: continue
        base_score = float(dist)
        page = metadata[idx]["page"]
        if page <= 3: base_score *= 1.15
        res = {"page": page, "text": txt, "score": round(base_score, 3)}
        candidates.append(res)
        if base_score >= SIMILARITY_THRESHOLD: results.append(res)
    if not results and candidates: results = sorted(candidates, key=lambda x: -x["score"])[:k]
    return sorted(results, key=lambda x: -x["score"])[:k]