import os, logging, warnings, asyncio, time, json, re
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError, ClientError

# Suppress noisy warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

from config import CHAT_MODEL, EMBED_MODEL, MAX_HISTORY, MAX_CONTEXT_CHARS, QUERY_INTENT_TOP_K
from models import ChatRequest, ChatResponse
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

INTENT_LABELS = {
    "fact_lookup": "Fact lookup",
    "summary": "Summary",
    "deep_explanation": "Deep explanation",
    "comparison": "Comparison",
    "critical_analysis": "Critical analysis",
    "architecture": "Architecture / system design",
    "methodology": "Methodology",
    "limitations": "Limitations / risks",
}

INTENT_KEYWORDS = [
    ("comparison", ["compare", "comparison", "difference", "differentiate", "versus", " vs ", "distinction", "trade-off", "tradeoff"]),
    ("architecture", ["architecture", "system design", "pipeline", "component", "module", "workflow", "framework", "stack", "how is it built"]),
    ("methodology", ["methodology", "method", "approach", "experiment", "evaluation", "dataset", "protocol", "procedure", "how did they"]),
    ("limitations", ["limitation", "limitations", "risk", "risks", "challenge", "weakness", "failure", "constraint", "threat"]),
    ("critical_analysis", ["critique", "critical", "evaluate", "strength", "weakness", "assumption", "bias", "validity", "robustness"]),
    ("summary", ["summarize", "summary", "overview", "abstract", "main idea", "main thesis", "what is this about", "tl;dr"]),
    ("deep_explanation", ["explain", "why", "how does", "walk me through", "in detail", "deeply", "meaning", "interpret", "what does this document say"]),
]

INTENT_STYLE_GUIDE = {
    "fact_lookup": "Start with a concise direct answer, then give the supporting citation evidence.",
    "summary": "Use Overview, Key Finding, Detailed Explanation, and Evidence & Citations sections.",
    "deep_explanation": "Use Overview, Key Finding, Detailed Explanation, Implications, Limitations, and Evidence & Citations sections when supported.",
    "comparison": "Use Key Finding, Comparison / Differences, Detailed Explanation, and Evidence & Citations sections.",
    "critical_analysis": "Use Key Finding, Detailed Explanation, Implications, Limitations, and Evidence & Citations sections. Separate document facts from your supported interpretation.",
    "architecture": "Use Overview, Key Components, Detailed Explanation, Implications, Limitations, and Evidence & Citations sections.",
    "methodology": "Use Overview, Methodology, Detailed Explanation, Limitations, and Evidence & Citations sections.",
    "limitations": "Use Key Finding, Limitations / Risks, Evidence & Citations, and what remains uncertain.",
}

def detect_query_intent(question: str) -> str:
    normalized = f" {question.lower()} "
    for intent, keywords in INTENT_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return intent
    if re.match(r"^\s*(who|what|when|where|which|how many|how much)\b", question.lower()):
        return "fact_lookup"
    return "fact_lookup"

def format_context(context: List[Dict[str, Any]]) -> str:
    if not context:
        return "No context retrieved."
    blocks = []
    for idx, chunk in enumerate(context, start=1):
        text = re.sub(r"\s+", " ", chunk["text"]).strip()
        blocks.append(f"[Source {idx} | Page {chunk['page']} | Relevance {chunk['score']}]\n{text}")
    ctx_str = "\n\n".join(blocks)
    if len(ctx_str) > MAX_CONTEXT_CHARS:
        return ctx_str[:MAX_CONTEXT_CHARS].rsplit(" ", 1)[0] + "..."
    return ctx_str

def build_research_prompt(question: str, context: str, intent: str) -> str:
    style = INTENT_STYLE_GUIDE.get(intent, INTENT_STYLE_GUIDE["fact_lookup"])
    label = INTENT_LABELS.get(intent, "Fact lookup")
    return f"""You are a technical research analyst answering questions about a PDF.

Answer only from the retrieved document context below. Do not invent unsupported claims.
Every factual claim should be traceable to the context and include page citations like (Page 8).
If the context is incomplete, say exactly what is uncertain or missing.
Distinguish document facts from interpretation when you synthesize meaning or implications.
Use a confident expert tone: "The document proposes...", "The authors describe...", "The architecture introduces...", or "The key distinction is..." when those phrasings fit.
Avoid vague filler such as "Based on the provided context" unless you need to explain insufficient evidence.
Avoid LaTeX, math delimiters, and academic symbols that may render poorly.
Return clean markdown with concise section headings.

Detected question intent: {label}
Response style: {style}

Retrieved document context:
{context}

Question: {question}
"""

def relevance_label(context: List[Dict[str, Any]]) -> str:
    if not context:
        return "No document evidence found"
    best_score = max(c["score"] for c in context)
    if best_score >= 0.58:
        return "High relevance"
    if best_score >= 0.42:
        return "Moderate relevance"
    return "Exploratory match"

def build_followups(intent: str, context: List[Dict[str, Any]]) -> List[str]:
    primary_page = context[0]["page"] if context else None
    by_intent = {
        "fact_lookup": ["Explain this answer in context", "What evidence supports this?", "What limitations are mentioned?"],
        "summary": ["What are the key claims?", "Explain the methodology", "What are the limitations?"],
        "deep_explanation": ["Explain the architecture more deeply", "What assumptions does it rely on?", "Summarize the implications"],
        "comparison": ["Compare the approaches side by side", "What is the key distinction?", "Which option has stronger evidence?"],
        "critical_analysis": ["What are the main risks?", "Which claims are best supported?", "What evidence is missing?"],
        "architecture": ["Explain the architecture more deeply", "What are the key components?", "Where are the system limitations?"],
        "methodology": ["Explain the methodology step by step", "What data or experiments are used?", "What are the validity risks?"],
        "limitations": ["Which limitation is most important?", "How do the authors justify it?", "What evidence is missing?"],
    }
    followups = by_intent.get(intent, by_intent["fact_lookup"]).copy()
    if primary_page:
        followups[1] = f"What evidence on page {primary_page} supports this?"
    return followups[:3]

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
        intent = detect_query_intent(req.question)
        k = QUERY_INTENT_TOP_K.get(intent, QUERY_INTENT_TOP_K["fact_lookup"])
        history = (req.history or [])[-(MAX_HISTORY-1):]
        history.append({"role": "user", "parts": [{"text": req.question}]})
        
        context = rag.retrieve(req.question, intent, k)
        ctx_str = format_context(context)
        
        # 1. Generate draft
        prompt = build_research_prompt(req.question, ctx_str, intent)
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
        retrieved_sections = [
            {
                "page": c["page"],
                "score": c["score"],
                "preview": get_citation_snippet(c["text"], 110),
            }
            for c in context
        ]
        meta = {
            "intent": intent,
            "answer_type": INTENT_LABELS.get(intent, "Fact lookup"),
            "search_mode": "Semantic retrieval",
            "chunks_used": len(context),
            "top_k": k,
            "primary_source_page": context[0]["page"] if context else None,
            "relevance_label": relevance_label(context),
            "retrieved_sections": retrieved_sections,
            "suggested_followups": build_followups(intent, context),
            "verification": verification,
        }
        if debug: meta["context_preview"] = ctx_str[:200]
        
        return ChatResponse(answer=draft, citations=citations, metadata=meta)
