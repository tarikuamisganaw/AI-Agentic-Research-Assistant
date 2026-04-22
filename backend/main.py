import os, logging, warnings, asyncio, time, json
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError, ClientError

# Suppress noisy warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

from config import CHAT_MODEL, EMBED_MODEL, MAX_HISTORY, MAX_CONTEXT_CHARS
from models import ChatRequest, Citation, ChatResponse
from utils import get_citation_snippet
import rag
from agent import verify_answer, needs_clarification

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or not api_key.startswith("AIza"):
    raise RuntimeError("Missing GOOGLE_API_KEY. Set in .env or HF Secrets.")

chat_client = genai.Client(api_key=api_key)
lock = asyncio.Lock()
session_history = []

def generate_with_retry(model: str, contents: list, max_retries: int = 3):
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return chat_client.models.generate_content(model=model, contents=contents)
        except ServerError as e:
            if e.code not in [503, 429]: raise
            time.sleep(delay * (2 ** attempt))
        except ClientError: raise
    raise RuntimeError("Gemini API unavailable after retries")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Initializing models...")
    rag.initialize_models(EMBED_MODEL)
    if os.path.exists(rag.INDEX_PATH) and os.path.exists(rag.META_PATH):
        rag.faiss_index = rag.faiss.read_index(rag.INDEX_PATH)
        with open(rag.META_PATH) as f: rag.metadata = json.load(f)
        print(f"Loaded index: {len(rag.metadata)} chunks")
    else:
        print("No index. Upload PDF first.")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan, title="AI Agentic Researcher", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "healthy", "chunks": len(rag.metadata) if rag.metadata else 0}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PDF only")
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"
    with open(path, "wb") as f: f.write(await file.read())
    try:
        async with lock:
            count = rag.ingest_pdf(path)
            session_history.clear()
            return {"status": "success", "chunks": count}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(path): os.remove(path)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, debug: bool = False):
    if not chat_client: raise HTTPException(503, "LLM not initialized")
    async with lock:
        is_summary = any(kw in req.question.lower() for kw in ["summarize", "overview", "abstract"])
        k = 6 if is_summary else 4
        history = (req.history or [])[-(MAX_HISTORY-1):]
        history.append({"role": "user", "parts": [{"text": req.question}]})
        
        context = rag.retrieve(req.question, is_summary, k)
        ctx_str = "\n".join([f"[P{c['page']}]: {c['text'][:300]}..." for c in context]) if context else "No context retrieved."
        if len(ctx_str) > MAX_CONTEXT_CHARS: ctx_str = ctx_str[:MAX_CONTEXT_CHARS] + "..."
        
        # 1. Generate draft
        prompt = f"You are an expert researcher. Answer ONLY using the context below. Avoid LaTeX, math notation ($...$), or academic symbols.Cite page numbers. If unsure, say so.\nContext:\n{ctx_str}\nQuestion: {req.question}"
        history[-1] = {"role": "user", "parts": [{"text": prompt}]}
        
        try:
            draft = generate_with_retry(CHAT_MODEL, history).text
        except Exception as e:
            raise HTTPException(502, f"LLM Error: {e}")
            
        # 2. Agent Verification
        verification = verify_answer(draft, ctx_str)
        
        # 3. Clarifying Fallback (if confidence is too low)
        if needs_clarification(verification):
            clarifying_prompt = f"The user asked: '{req.question}'. Retrieved context has low relevance. Ask a polite, specific clarifying question to help them refine their search."
            history.append({"role": "user", "parts": [{"text": clarifying_prompt}]})
            draft = generate_with_retry(CHAT_MODEL, history).text
            verification["status"] = "clarifying_question"
            
        citations = [{"page": c["page"], "snippet": get_citation_snippet(c["text"]), "score": c["score"]} for c in context]
        meta = {"is_summary": is_summary, "chunks_used": len(context), "verification": verification}
        if debug: meta["context_preview"] = ctx_str[:200]
        
        return ChatResponse(answer=draft, citations=citations, metadata=meta)