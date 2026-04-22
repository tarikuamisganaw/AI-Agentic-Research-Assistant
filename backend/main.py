import os, logging, warnings, asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

# Suppress noisy warnings 
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Local imports
from config import CHAT_MODEL, EMBED_MODEL, SUMMARY_KEYWORDS, MAX_HISTORY, MAX_CONTEXT_CHARS
from models import ChatRequest, Citation, ChatResponse
from utils import get_citation_snippet
import rag

# Load environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or not api_key.startswith("AIza"):
    raise RuntimeError("Invalid or missing GOOGLE_API_KEY. Set it in HF Secrets.")

# Global state
chat_client = None
lock = asyncio.Lock()
session_history = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models + load index on startup."""
    global chat_client
    
    print("Loading models...")
    try:
        rag.initialize_models(EMBED_MODEL)
        chat_client = genai.Client(api_key=api_key)
        print(" Models loaded.")
    except Exception as e:
        print(f"Model init failed: {e}")
        raise

    print("📂 Loading FAISS index & metadata...")
    if rag.load_index():
        print(f"Loaded index with {len(rag.metadata)} chunks.")
    else:
        print("No index found. Upload a PDF first via POST /upload.")
    yield
    print("🔄 Shutting down...")

app = FastAPI(lifespan=lifespan, title="RAG Document Assistant", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints
@app.get("/")
def root():
    return {"message": "RAG Document Assistant API", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "chunks_indexed": len(rag.metadata) if rag.metadata else 0}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    
    os.makedirs("uploads", exist_ok=True)
    temp_path = f"uploads/{file.filename}"
    with open(temp_path, "wb") as f: f.write(await file.read())
    
    try:
        async with lock:
            chunk_count = rag.ingest_pdf(temp_path)
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
        
        context = rag.retrieve(req.question, is_summary, k)
        
        if context:
            ctx_str = "\n\n".join([f"[Page {c['page']}]: {c['text']}" for c in context])
            if len(ctx_str) > MAX_CONTEXT_CHARS:
                ctx_str = ctx_str[:MAX_CONTEXT_CHARS] + "\n\n[...truncated...]"
            
           # FIX: Move conditional logic OUTSIDE f-string to avoid backslash-in-expression error
            if is_summary:
                instruction = "Provide a HIGH-LEVEL SUMMARY covering main topic, contributions, and methodology by explaining well.Avoid LaTeX, math notation ($...$), or academic symbols."
            else:
                instruction = 'Answer ONLY using the provided context.Avoid LaTeX, math notation ($...$), or academic symbols. Say "I don\'t know" if unsure.  Always cite page numbers.'
            
            prompt = f"""You are an AI assistant that explain concepts well. {instruction}
Context:
{ctx_str}
Question: {req.question}"""
        else:
            prompt = f"No relevant context found. Politely acknowledge this and offer general help. Question: {req.question}"

        history[-1] = {"role": "user", "parts": [{"text": prompt}]}
        
        try:
            resp = chat_client.models.generate_content(model=CHAT_MODEL, contents=history)
            answer = resp.text
            history.append({"role": "model", "parts": [{"text": answer}]})
        except ClientError as e:
            raise HTTPException(502, f"LLM API Error: {str(e)}")
        
        citations = [{"page": c["page"], "snippet": get_citation_snippet(c["text"]), "score": round(c["score"], 3)} for c in context]
        meta = {"is_summary": is_summary, "chunks_used": len(context), "history_length": len(history)}
        
        if debug: meta["debug_context"] = context
        
        return ChatResponse(answer=answer, citations=citations, metadata=meta)