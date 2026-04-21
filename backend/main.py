import os,logging, warnings
import re
import json
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai.errors import ClientError

# Load environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Hide download progress bars
os.environ["TRANSFORMERS_VERBOSITY"] = "error"    # Only show errors from transformers
warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers")  # Ignore sentence-transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)  # Reduce transformers log level
if not api_key or not api_key.startswith("AIza"):
    raise RuntimeError("Invalid or missing GOOGLE_API_KEY. Set it in HF Secrets.")

# Configuration
EMBED_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "gemini-3-flash-preview"  # Fixed: valid model name (was "gemini-3-flash-preview")
INDEX_PATH = "faiss_index.index"
META_PATH = "faiss_metadata.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_HISTORY = 6
SIMILARITY_THRESHOLD = 0.35
MAX_CONTEXT_CHARS = 2500
MIN_CHUNK_WORDS = 20
SUMMARY_KEYWORDS = [
    "what is the document about", "summarize", "overview", "main topic", 
    "explain this paper", "abstract", "what does it cover", "tell me about"
]

# Global state (initialized in lifespan)
text_splitter = None
embedder = None
chat_client = None
faiss_index = None
metadata = []
session_history = []
lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global text_splitter, embedder, chat_client, faiss_index, metadata
    
    print("⏳ Loading models...")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""], length_function=len, is_separator_regex=False
        )
        embedder = SentenceTransformer(EMBED_MODEL)
        chat_client = genai.Client(api_key=api_key)
        print("Models loaded.")
    except Exception as e:
        print(f"Model init failed: {e}")
        raise

    print("Loading FAISS index & metadata...")
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        faiss_index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r") as f:
            metadata = json.load(f)
        print(f"Loaded index with {len(metadata)} chunks.")
    else:
        print("No index found. Upload a PDF first via POST /upload.")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan, title="RAG Document Assistant", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["https://your-app.vercel.app"] before launch
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, Any]]] = []

class Citation(BaseModel):
    page: int
    snippet: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    metadata: Dict[str, Any]

# Helper: Clean PDF text
def clean_pdf_text(text: str) -> str:
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\b(BOS|EOS|FIG\.?|Figure\s*\d+)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(\w{3,})\s+([a-z]{2,4})\b', 
                  lambda m: m.group(1) + m.group(2) if len(m.group(1) + m.group(2)) <= 15 else m.group(0), 
                  text)
    
    fixes = {
        r'R ises': 'Rises', r'Farewellto': 'Farewell to', r'SunAlso': 'Sun Also',
        r'distributi on': 'distribution', r'generati on': 'generation', r'informati on': 'information',
        r'conta in': 'contain', r'retriev er': 'retriever', r'generat or': 'generator',
        r'th at': 'that', r'wh ich': 'which', r'for m': 'form', r'ar Xiv': 'arXiv',
    }
    for pattern, replacement in fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    lines = text.split('\n')
    cleaned = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: cleaned.append('\n'); continue
        if i < len(lines) - 1 and not re.search(r'[.!?]\s*$', line): cleaned.append(line + ' ')
        else: cleaned.append(line + '\n')
    text = ''.join(cleaned)
    return re.sub(r'\s{3,}', '  ', re.sub(r'\n{3,}', '\n\n', text)).strip()

# Helper: Citation snippet
def get_citation_snippet(text: str, max_len: int = 150) -> str:
    clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    clean = re.sub(r'\b(BOS|EOS|FIG\.?|Figure\s*\d+)\b', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\s{2,}', ' ', clean)
    sentences = re.split(r'(?<=[.!?])\s+', clean.strip())
    valid = [s.strip() for s in sentences if len(s.strip()) > 10 and not re.match(r'^[\s\.\,\;\:\!\?]+$', s)]
    first = valid[0] if valid else clean[:max_len]
    if len(first) > max_len:
        first = first[:max_len].rsplit(' ', 1)[0] + "..."
    return first.strip()

# Core: Ingest PDF
def ingest_pdf(pdf_path: str) -> int:
    global faiss_index, metadata
    if not all([embedder, text_splitter]):
        raise RuntimeError("Models not initialized.")
        
    reader = PdfReader(pdf_path)
    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip(): pages_text.append({"page": i + 1, "text": clean_pdf_text(text)})

    chunks = []
    for p in pages_text:
        for chunk in text_splitter.split_text(p["text"]):
            if chunk.strip() and len(chunk.split()) >= MIN_CHUNK_WORDS:
                chunks.append({"page": p["page"], "text": chunk, "word_count": len(chunk.split())})

    if not chunks: raise ValueError("No readable text in PDF.")

    vectors = embedder.encode([c["text"] for c in chunks], convert_to_numpy=True).astype("float32")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    faiss_index = faiss.IndexFlatIP(vectors.shape[1])
    faiss_index.add(vectors)
    faiss.write_index(faiss_index, INDEX_PATH)
    metadata = chunks
    with open(META_PATH, "w") as f: json.dump(metadata, f)
    
    print(f"Indexed {len(chunks)} chunks.")
    return len(chunks)

# Core: Context-Aware Retrieval
def retrieve(query: str, is_summary: bool, k: int) -> List[Dict]:
    global faiss_index, metadata, embedder
    if faiss_index is None or not metadata or embedder is None:
        return []
    
    fetch_k = 10 if is_summary else k * 2
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    distances, ids = faiss_index.search(query_vec, fetch_k)
    results, candidates = [], []
    
    for dist, idx in zip(distances[0], ids[0]):
        if idx == -1: continue
        txt = metadata[idx]["text"]
        if any(x in txt.lower() for x in ["references", "bibliography", "acknowledgments"]): continue
        if re.match(r'^\s*\[\d+\]', txt.strip()): continue
        if len(txt.split()) < MIN_CHUNK_WORDS: continue

        base_score = float(dist)
        page = metadata[idx]["page"]
        if page <= 3: base_score *= 1.15

        res = {"page": page, "text": txt, "score": base_score, "raw": float(dist)}
        candidates.append(res)
        if base_score >= SIMILARITY_THRESHOLD: results.append(res)

    if not results: results = sorted(candidates, key=lambda x: -x["score"])[:k]
    return sorted(results, key=lambda x: -x["score"])[:k]

# Endpoints
@app.get("/")
def root():
    return {"message": "RAG Document Assistant API", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "chunks_indexed": len(metadata) if metadata else 0}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    
    os.makedirs("uploads", exist_ok=True)
    temp_path = f"uploads/{file.filename}"
    with open(temp_path, "wb") as f: f.write(await file.read())
    
    try:
        async with lock:
            chunk_count = ingest_pdf(temp_path)
            session_history.clear()
            return {"status": "success", "chunks": chunk_count, "filename": file.filename}
    except Exception as e:
        raise HTTPException(500, f"Failed to index PDF: {str(e)}")
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, debug: bool = Query(False)):
    if chat_client is None:
        raise HTTPException(503, "LLM client not initialized.")
        
    async with lock:
        is_summary = any(kw in req.question.lower() for kw in SUMMARY_KEYWORDS)
        k = 6 if is_summary else 3
        
        history = req.history[-(MAX_HISTORY-1):] if req.history else []
        history.append({"role": "user", "parts": [{"text": req.question}]})
        if len(history) > MAX_HISTORY: history = history[-MAX_HISTORY:]
        
        context = retrieve(req.question, is_summary, k)
        
        if context:
            ctx_str = "\n\n".join([f"[Page {c['page']}]: {c['text']}" for c in context])
            if len(ctx_str) > MAX_CONTEXT_CHARS:
                ctx_str = ctx_str[:MAX_CONTEXT_CHARS] + "\n\n[...truncated...]"
            
            # ✅ FIXED INDENTATION BELOW
            if is_summary:
                prompt = f"""You are a precise AI assistant. Provide a HIGH-LEVEL SUMMARY of the document based ONLY on the provided context.
Cover the main topic, key contributions, methodology, and overall purpose.
If the context lacks a clear overview, state what you can infer and note limitations.
Always cite page numbers.

Context:
{ctx_str}

Question: {req.question}"""
            else:
                prompt = f"""You are a precise AI assistant. Answer ONLY using the provided context.
If the answer is not in the context, say "I don't know."
Be concise and clear. If multiple answers exist, explain clearly.
Always cite page numbers.

Context:
{ctx_str}

Question: {req.question}"""
        else:
            prompt = f"""You are a helpful assistant. The user asked about a document, but no relevant context was retrieved.
Please respond politely: acknowledge the limitation, offer general help if appropriate, and suggest rephrasing.
User question: {req.question}"""

        history[-1] = {"role": "user", "parts": [{"text": prompt}]}
        
        try:
            resp = chat_client.models.generate_content(model=CHAT_MODEL, contents=history)
            answer = resp.text
            history.append({"role": "model", "parts": [{"text": answer}]})
        except ClientError as e:
            raise HTTPException(502, f"LLM API Error: {str(e)}")
        
        citations = [
            {"page": c["page"], "snippet": get_citation_snippet(c["text"]), "score": round(c["score"], 3)} 
            for c in context
        ]
        meta = {"is_summary": is_summary, "chunks_used": len(context), "history_length": len(history)}
        if debug:
            meta["debug_context"] = context
        
        return ChatResponse(answer=answer, citations=citations, metadata=meta)