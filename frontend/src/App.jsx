import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import './index.css' // Ensure CSS is imported in case main.jsx misses it

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [citations, setCitations] = useState([])
  const [loading, setLoading] = useState(false)
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [showDebug, setShowDebug] = useState(false)
  const [toast, setToast] = useState(null) // { message, type }

  const answerRef = useRef(null)

  const showToast = (message, type = 'success') => {
    setToast({ message, type })
    setTimeout(() => setToast(null), 4000)
  }

  const handleUpload = async () => {
    if (!file) {
      showToast('Please select a PDF first', 'error')
      return
    }
    setUploading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData })
      const data = await res.json()
      if (data.status === 'success') {
        showToast(`Document indexed: ${data.chunks} chunks processed.`, 'success')
        setAnswer('')
        setCitations([])
      } else {
        showToast(`Upload failed: ${data.detail || JSON.stringify(data)}`, 'error')
      }
    } catch (e) {
      showToast(`Error: ${e.message}`, 'error')
    } finally {
      setUploading(false)
    }
  }

  const handleAsk = async () => {
    if (!question.trim()) return
    setLoading(true)
    setAnswer('')
    setCitations([])
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      })
      const data = await res.json()
      if (res.ok) {
        setAnswer(data.answer)
        setCitations(data.citations || [])
        // Scroll to answer after rendering
        setTimeout(() => {
          answerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
        }, 100)
      } else {
        setAnswer(`❌ Server Error: ${data.detail || JSON.stringify(data)}`)
      }
    } catch (e) {
      setAnswer(`❌ Connection Error: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">AI Agentic Research Assistant</h1>
        <p className="app-subtitle">Upload a document and uncover its insights instantly.</p>
      </header>

      {/* Upload Section */}
      <section className="section-panel">
        <label className="section-label">
          <i>📁</i> 1. Ingest Knowledge
        </label>
        <div className="upload-group">
          <input 
            type="file" 
            accept=".pdf" 
            onChange={e => setFile(e.target.files[0])} 
            className="file-input"
          />
          <button 
            onClick={handleUpload} 
            disabled={!file || uploading} 
            className="btn btn-secondary"
          >
            {uploading ? (
              <div className="loader-wrapper">
                Processing <div className="dot-flashing"></div>
              </div>
            ) : 'Index Document'}
          </button>
        </div>
      </section>

      {/* Chat Section */}
      <section className="section-panel">
        <label className="section-label">
          <i>✨</i> 2. Extract Insights
        </label>
        <textarea
          value={question}
          onChange={e => setQuestion(e.target.value)}
          placeholder="e.g., What is the main thesis of this document? Are there any critical limitations mentioned?"
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleAsk())}
          className="q-textarea"
          disabled={loading}
        />
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1rem' }}>
          <button 
            onClick={handleAsk} 
            disabled={loading || !question.trim()} 
            className="btn btn-primary"
          >
            {loading ? (
              <div className="loader-wrapper">
                Synthesizing <div className="dot-flashing"></div>
              </div>
            ) : 'Ask Assistant'}
          </button>
        </div>
      </section>

      {/* Answer Section */}
      {answer && (
        <section className="answer-panel" ref={answerRef}>
          <div className="answer-badge">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
            Generated Insight
          </div>
          
          <div className="markdown-body">
            <ReactMarkdown>
              {answer}
            </ReactMarkdown>
          </div>
          
          {/* Citations */}
          {citations.length > 0 && (
            <div className="citations-wrapper">
              <details>
                <summary className="citations-summary">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
                  Verified Sources ({citations.length})
                </summary>
                <div style={{ marginTop: '1rem' }}>
                  {citations.map((c, i) => (
                    <div key={i} className="citation-card">
                      <strong>Page {c.page}</strong>: {c.snippet}
                      {showDebug && <span className="debug-tag">sim: {c.score.toFixed(3)}</span>}
                    </div>
                  ))}
                </div>
              </details>
              
              <label className="debug-toggle">
                <input 
                  type="checkbox" 
                  checked={showDebug} 
                  onChange={e => setShowDebug(e.target.checked)} 
                />
                Show semantic similarity scores
              </label>
            </div>
          )}
        </section>
      )}

      {/* Toast Notifications */}
      {toast && (
        <div className={`toast ${toast.type}`}>
          {toast.type === 'success' ? '✅' : '❌'} {toast.message}
        </div>
      )}
    </div>
  )
}

export default App