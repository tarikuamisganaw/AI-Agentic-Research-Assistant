# Agentic Researcher Configuration
EMBED_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL =  "gemini-3-flash-preview"
INDEX_PATH = "faiss_index.index"
META_PATH = "faiss_metadata.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_HISTORY = 6
SIMILARITY_THRESHOLD = 0.35
MAX_CONTEXT_CHARS = 9000
MIN_CHUNK_WORDS = 20
SUMMARY_KEYWORDS = ["summarize", "overview", "abstract", "main topic", "what is this about"]
QUERY_INTENT_TOP_K = {
    "fact_lookup": 4,
    "summary": 8,
    "deep_explanation": 12,
    "comparison": 12,
    "critical_analysis": 10,
    "architecture": 12,
    "methodology": 10,
    "limitations": 8,
}
