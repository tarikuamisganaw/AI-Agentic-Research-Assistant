import os
import re
import json
import time
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional: Silence Hugging Face warnings
import logging
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Diagnostic: Verify API key
if not api_key or not api_key.startswith("AIza"):
    print("Invalid or missing GOOGLE_API_KEY in .env")
    print("Get a new key: https://aistudio.google.com/app/apikey")
    print(f"Current value preview: {api_key[:10] if api_key else 'None'}...")
    exit(1)

# Initialize clients with explicit key
chat_client = genai.Client(api_key=api_key)

#Configuration
EMBED_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "gemini-3-flash-preview"
INDEX_PATH = "faiss_index.index"
META_PATH = "faiss_metadata.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_HISTORY = 6
SIMILARITY_THRESHOLD = 0.35
MAX_CONTEXT_CHARS = 2500  # Slightly larger for summaries
MIN_CHUNK_WORDS = 20

# Summary query detection keywords
SUMMARY_KEYWORDS = [
    "what is the document about", "summarize", "overview", "main topic", 
    "explain this paper", "abstract", "what does it cover", "tell me about"
]

# Initialize LangChain splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
    is_separator_regex=False,
)

# Load local embedding model
print(f"Loading embedding model: {EMBED_MODEL}...")
embedder = SentenceTransformer(EMBED_MODEL)
print("Embedding model ready.")

# FIX: Enhanced PDF text cleaning
def clean_pdf_text(text: str) -> str:
    """Fix common PDF extraction artifacts"""
    # 1. Join hyphenated words split across lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # 2. Fix CamelCase/mashed words from column extraction
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # 3. Remove PDF artifacts: BOS/EOS, figure labels
    text = re.sub(r'\b(BOS|EOS|FIG\.?|Figure\s*\d+)\b', '', text, flags=re.IGNORECASE)
    
    # 4. Remove standalone page numbers/headers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 5. Fix mid-sentence line breaks (keep paragraph breaks)
    lines = text.split('\n')
    cleaned = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            cleaned.append('\n')
            continue
        if i < len(lines) - 1 and not re.search(r'[.!?]\s*$', line):
            cleaned.append(line + ' ')
        else:
            cleaned.append(line + '\n')
    text = ''.join(cleaned)
    
    # 6. Collapse excess whitespace
    text = re.sub(r'\s{3,}', '  ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

# 1. Extract & Chunk PDF → Embed → Save to FAISS
def ingest_pdf(pdf_path: str) -> bool:
    reader = PdfReader(pdf_path)
    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            pages_text.append({"page": i + 1, "text": clean_pdf_text(text)})

    chunks = []
    for p in pages_text:
        page_chunks = text_splitter.split_text(p["text"])
        for chunk_text in page_chunks:
            if chunk_text.strip() and len(chunk_text.split()) >= MIN_CHUNK_WORDS:
                chunks.append({
                    "page": p["page"],
                    "text": chunk_text,
                    "word_count": len(chunk_text.split())
                })

    if not chunks:
        print("No readable text found in PDF.")
        return False

    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL}...")
    texts = [c["text"] for c in chunks]
    
    vectors = embedder.encode(texts, convert_to_numpy=True).astype("float32")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    dim = vectors.shape[1]
    print(f"Vector dimension: {dim}")

    index = faiss.IndexFlatIP(dim)  # Cosine similarity when normalized
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w") as f:
        json.dump(chunks, f)

    print(f"Indexed {len(chunks)} chunks. Ready to chat!")
    return True

# 🔍 2. Retrieve Relevant Chunks (Context-Aware)
def retrieve(query: str, is_summary: bool = False, k: int = 4) -> list[dict]:
    if not os.path.exists(INDEX_PATH):
        print("No FAISS index found. Upload a PDF first.")
        return []

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    # Fetch more candidates for summaries to capture broad context
    fetch_k = 10 if is_summary else k * 2
    
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    distances, ids = index.search(query_vec, fetch_k)
    
    results = []
    all_candidates = []
    
    for dist, idx in zip(distances[0], ids[0]):
        if idx == -1:
            continue
            
        chunk_text = metadata[idx]["text"]
        
        #FIX #3: Filter out reference/citation heavy chunks
        if any(x in chunk_text.lower() for x in ["references", "bibliography", "acknowledgments"]):
            continue
        if re.match(r'^\s*\[\d+\]', chunk_text.strip()):  # Starts with [1], [7], etc.
            continue
            
        if len(chunk_text.split()) < MIN_CHUNK_WORDS:
            continue
            
        # Base cosine score (higher = better)
        base_score = float(dist)
        
        # FIX #2: Boost early pages (abstract/intro usually live here)
        page = metadata[idx]["page"]
        if page <= 3:
            base_score *= 1.15  # 15% relevance boost for early pages
            
        result = {
            "page": page,
            "text": chunk_text,
            "score": base_score,
            "raw_score": float(dist)  # Keep original for debug
        }
        all_candidates.append(result)
        
        if base_score >= SIMILARITY_THRESHOLD:
            results.append(result)
    
    # Fallback: return up to k best matches if nothing passes threshold
    if not results and all_candidates:
        all_candidates = sorted(all_candidates, key=lambda x: -x["score"])
        results = all_candidates[:min(k, len(all_candidates))]
    
    # Sort by boosted score (descending) and return top k
    results = sorted(results, key=lambda x: -x["score"])[:k]
    return results

# FIX: Cleaner citation snippets
def get_citation_snippet(text: str, max_len: int = 150) -> str:
    clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    sentences = re.split(r'(?<=[.!?])\s+', clean.strip())
    first_sent = sentences[0].strip() if sentences else clean[:max_len]
    if len(first_sent) > max_len:
        first_sent = first_sent[:max_len].rsplit(' ', 1)[0] + "..."
    return first_sent

#3. Main CLI Chat Loop
print("RAG CLI Chat loaded. Type a PDF path to index, or ask a question.\n")
history = []

while True:
    user_input = input("👤 You: ").strip()
    if not user_input: continue
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    # Handle PDF upload path
    if user_input.endswith(".pdf") and os.path.exists(user_input):
        if ingest_pdf(user_input):
            history = []
        print()
        continue

    # Add user message to history
    history.append({"role": "user", "parts": [{"text": user_input}]})
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    print("Retrieving context & thinking...")
    try:
        #FIX #1: Detect summary-type questions
        is_summary = any(kw in user_input.lower() for kw in SUMMARY_KEYWORDS)
        k = 6 if is_summary else 3  # Broader context for summaries
        
        context = retrieve(user_input, is_summary=is_summary, k=k)
        
        #Debug logging
        print(f"\n Retrieved {len(context)} chunks (is_summary={is_summary}, k={k}):")
        if context:
            for c in context:
                print(f"Score: {c['score']:.3f} (raw: {c['raw_score']:.3f}) | Page {c['page']} | {c['text'][:60]}...")
        else:
            print("No relevant chunks found. Answering from general knowledge.")
        print()
        
        # FIX #4: Dynamic prompt based on query intent
        if context:
            ctx_str = "\n\n".join([f"[Page {c['page']}]: {c['text']}" for c in context])
            if len(ctx_str) > MAX_CONTEXT_CHARS:
                ctx_str = ctx_str[:MAX_CONTEXT_CHARS] + "\n\n[...truncated...]"
            
            if is_summary:
                prompt = f"""You are a precise AI assistant. Provide a HIGH-LEVEL SUMMARY of the document based ONLY on the provided context.
Cover the main topic, key contributions, methodology, and overall purpose.
If the context lacks a clear overview, state what you can infer and note limitations.
Always cite page numbers.

Context:
{ctx_str}

Question: {user_input}"""
            else:
                prompt = f"""You are a precise AI assistant. Answer ONLY using the provided context.
If the answer is not in the context, say "I don't know."
Be concise and clear. If multiple answers exist, explain clearly.
Always cite page numbers.

Context:
{ctx_str}

Question: {user_input}"""
        else:
            prompt = f"""You are a helpful assistant. The user asked about a document, but no relevant context was retrieved.
Please respond politely: acknowledge the limitation, offer general help if appropriate, and suggest rephrasing.
User question: {user_input}"""

        chat_history = history.copy()
        chat_history[-1] = {"role": "user", "parts": [{"text": prompt}]}
        
        resp = chat_client.models.generate_content(model=CHAT_MODEL, contents=chat_history)
        ai_text = resp.text
        print(f"Gemini: {ai_text}\n")
        
        history.append({"role": "model", "parts": [{"text": ai_text}]})

        # Show citations
        if context:
            print("Sources:")
            for c in context:
                snippet = get_citation_snippet(c["text"])
                print(f"  • Page {c['page']} (score: {c['score']:.3f}): {snippet}")
            print()

    except ClientError as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            print("Rate limited. Waiting 25s before retry...\n")
            time.sleep(25)
            history.pop()
            continue
        print(f"API Error: {e}\n")
        history.pop()
    except Exception as e:
        print(f"Error: {e}\n")
        history.pop()