# 🤖 AI Agentic Research Assistant

A self-verifying RAG system that evaluates answer confidence, routes to clarifying questions when uncertain, and grounds responses in uploaded PDFs. Built for accuracy, transparency, and production reliability.

🌐 **Live Demo**: [https://agentic-researcher.vercel.app](https://agentic-researcher.vercel.app)  
---

## ✨ Features

- 📄 **PDF Ingestion**: Automatic text cleaning, chunking, and FAISS vector indexing
- 🔍 **Context-Aware Retrieval**: Cosine similarity search with intent-based fetching & early-page boosting
- 🤖 **Self-Verifying Agent Loop**: Automatically scores answer confidence against retrieved context
- 🔄 **Smart Routing**: Returns direct answers when confident, triggers clarifying questions when uncertain
- 📊 **Transparent Metadata**: Confidence scores, verification status, and page-level citations included in every response
- 🛡️ **Resilient API**: Exponential backoff retry logic for LLM rate limits (503/429)
- ⚡ **Lightweight & Fast**: No heavy agent frameworks; <200ms verification overhead

---

## 🏗️ Architecture

```
User Query → Retrieve Context → LLM Draft Answer → 🔍 Verify Claims → 📊 Score Confidence
       ↓
  Confidence ≥ 45%? → Return Answer + Citations
       ↓ No
  Trigger Clarifying Fallback → Ask Follow-up → Return to User
```

**Tech Stack**:
- **Backend**: FastAPI, `sentence-transformers`, FAISS, Google Gemini API
- **Frontend**: React + Vite, TailwindCSS
- **Deployment**: Hugging Face Spaces (Docker), Vercel
- **Patterns**: Self-verification loop, conditional routing, retry with backoff, CORS-secured API

---

## 🚀 Local Setup

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/ai-agentic-researcher.git
cd ai-agentic-researcher/backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
Create `backend/.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run Backend
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
📖 API Docs: `http://localhost:8000/docs`

### 4. Run Frontend (if applicable)
```bash
cd ../frontend
npm install
npm run dev
```

---

## 🧪 Test the API
```bash
# Health check
curl https://yourusername-ai-agentic-researcher.hf.space/health

# Upload PDF
curl -X POST https://yourusername-ai-agentic-researcher.hf.space/upload \
  -F "file=@research_paper.pdf"

# Ask with verification metadata
curl -X POST https://yourusername-ai-agentic-researcher.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What methodology did they use?", "debug": true}'
```

---

## 📁 Project Structure
```
backend/
├── main.py          # FastAPI entry point, agent loop, retry logic
├── agent.py         # Self-verification & clarifying fallback logic
├── rag.py           # PDF ingestion, FAISS retrieval, embedding
├── config.py        # Constants & thresholds
├── models.py        # Pydantic request/response schemas
├── utils.py         # Text cleaning & citation snippet helpers
└── requirements.txt
frontend/            # React/Vite UI (separate deployment)
Dockerfile           # HF Spaces container config
```

---

## 🌐 Deployment
- **Backend**: Dockerized FastAPI deployed to Hugging Face Spaces (CPU, 16GB RAM)
- **Frontend**: Vite-optimized React app deployed to Vercel
- **CORS**: Configured for local dev + Vercel production domain
- **Secrets**: `GOOGLE_API_KEY` managed via HF Repository Secrets

---

## 🎯 Why This Architecture?
Traditional RAG systems output text blindly, leading to hallucination and low user trust. This project adds a **lightweight agentic verification layer** that:
1. Evaluates draft answers against source context
2. Returns a confidence score (`0–100%`)
3. Automatically routes to clarifying questions when confidence drops below threshold
4. Maintains sub-second latency without heavy agent frameworks

This pattern is used in production AI systems to balance accuracy, cost, and user experience.

---


