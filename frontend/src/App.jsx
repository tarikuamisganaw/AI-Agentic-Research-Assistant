import { useEffect, useMemo, useRef, useState } from 'react'
import AnswerCard from './components/AnswerCard'
import FileUpload from './components/FileUpload'
import ProcessingSteps from './components/ProcessingSteps'
import RetrievalMetadata from './components/RetrievalMetadata'
import CitationPill from './components/CitationPill'
import './index.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const PROCESSING_STEPS = {
  upload: ['Extracting text', 'Chunking document', 'Creating embeddings'],
  ask: ['Searching semantic index', 'Synthesizing answer'],
}

function App() {
  const [question, setQuestion] = useState('')
  const [file, setFile] = useState(null)
  const [documentState, setDocumentState] = useState(null)
  const [answerHistory, setAnswerHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [showDebug, setShowDebug] = useState(false)
  const [toast, setToast] = useState(null)
  const [processing, setProcessing] = useState({ mode: null, status: 'idle', activeIndex: 0 })

  const answerRef = useRef(null)
  const toastTimerRef = useRef(null)
  const processingTimerRef = useRef(null)

  const latestEntry = answerHistory[0]
  const currentSteps = processing.mode ? PROCESSING_STEPS[processing.mode] : []

  const processingTitle = useMemo(() => {
    if (processing.mode === 'upload') return 'Indexing document'
    if (processing.mode === 'ask') return 'Generating answer'
    return 'Processing'
  }, [processing.mode])

  useEffect(() => {
    if (processing.status !== 'running' || !processing.mode) return undefined

    const timer = setInterval(() => {
      setProcessing(current => {
        const steps = PROCESSING_STEPS[current.mode] || []
        const lastStep = Math.max(steps.length - 1, 0)
        if (current.activeIndex >= lastStep) return current
        return { ...current, activeIndex: current.activeIndex + 1 }
      })
    }, 900)

    return () => clearInterval(timer)
  }, [processing.status, processing.mode])

  const showToast = (message, type = 'success') => {
    window.clearTimeout(toastTimerRef.current)
    setToast({ message, type })
    toastTimerRef.current = window.setTimeout(() => setToast(null), 4200)
  }

  const startProcessing = mode => {
    window.clearTimeout(processingTimerRef.current)
    setProcessing({ mode, status: 'running', activeIndex: 0 })
  }

  const finishProcessing = status => {
    setProcessing(current => {
      const steps = PROCESSING_STEPS[current.mode] || []
      return {
        ...current,
        status,
        activeIndex: Math.max(steps.length - 1, 0),
      }
    })
    processingTimerRef.current = window.setTimeout(() => {
      setProcessing({ mode: null, status: 'idle', activeIndex: 0 })
    }, status === 'complete' ? 850 : 1400)
  }

  const parseResponse = async response => {
    const text = await response.text()
    try {
      return text ? JSON.parse(text) : {}
    } catch {
      return { detail: text || 'Unexpected empty response' }
    }
  }

  const handleUpload = async () => {
    if (!file) {
      showToast('Please select a PDF first', 'error')
      return
    }

    setUploading(true)
    startProcessing('upload')

    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData })
      const data = await parseResponse(res)

      if (!res.ok || data.status !== 'success') {
        throw new Error(data.detail || 'Upload failed')
      }

      setDocumentState({ fileName: file.name, chunks: data.chunks })
      setAnswerHistory([])
      showToast(`Document indexed: ${data.chunks} chunks processed.`, 'success')
      finishProcessing('complete')
    } catch (error) {
      showToast(`Upload failed: ${error.message}`, 'error')
      finishProcessing('error')
    } finally {
      setUploading(false)
    }
  }

  const askQuestion = async (questionText = question) => {
    const trimmedQuestion = questionText.trim()
    if (!trimmedQuestion) return

    setQuestion(trimmedQuestion)
    setLoading(true)
    startProcessing('ask')

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmedQuestion }),
      })
      const data = await parseResponse(res)

      if (!res.ok) {
        throw new Error(data.detail || 'The assistant could not generate an answer.')
      }

      const entry = {
        id: `${Date.now()}-${trimmedQuestion}`,
        question: trimmedQuestion,
        answer: data.answer || 'The assistant returned an empty answer.',
        citations: data.citations || [],
        metadata: data.metadata || {},
      }

      setAnswerHistory(history => [entry, ...history])
      setQuestion('')
      finishProcessing('complete')
      window.setTimeout(() => {
        answerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 100)
    } catch (error) {
      const entry = {
        id: `${Date.now()}-error`,
        question: trimmedQuestion,
        answer: `Server Error: ${error.message}`,
        citations: [],
        metadata: {
          answer_type: 'Error',
          search_mode: 'Semantic retrieval',
          relevance_label: 'No answer generated',
          suggested_followups: [],
        },
      }
      setAnswerHistory(history => [entry, ...history])
      finishProcessing('error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="app-kicker">Document intelligence workspace</p>
          <h1 className="app-title">AI Agentic Research Assistant</h1>
        </div>
        <p className="app-subtitle">Upload a research PDF, ask precise questions, and review answers with page-grounded evidence.</p>
      </header>

      <main className="research-layout">
        <section className="workspace-column" aria-label="Research workspace">
          <section className="workspace-panel">
            <div className="panel-heading">
              <span>1</span>
              <div>
                <h2>Ingest Knowledge</h2>
                <p>{documentState ? `${documentState.fileName} indexed with ${documentState.chunks} chunks.` : 'Add a PDF to build the semantic index.'}</p>
              </div>
            </div>
            <FileUpload
              file={file}
              disabled={uploading}
              processing={uploading}
              onFileSelect={setFile}
              onFileReject={message => showToast(message, 'error')}
              onUpload={handleUpload}
            />
            {processing.mode === 'upload' && (
              <ProcessingSteps
                title={processingTitle}
                steps={currentSteps}
                activeIndex={processing.activeIndex}
                status={processing.status}
              />
            )}
          </section>

          <section className="workspace-panel">
            <div className="panel-heading">
              <span>2</span>
              <div>
                <h2>Extract Insights</h2>
                <p>Ask for facts, summaries, architecture, methodology, comparisons, or limitations.</p>
              </div>
            </div>

            <label className="sr-only" htmlFor="research-question">Research question</label>
            <textarea
              id="research-question"
              value={question}
              onChange={event => setQuestion(event.target.value)}
              placeholder="What is the main contribution, and what evidence supports it?"
              onKeyDown={event => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  askQuestion()
                }
              }}
              className="question-textarea"
              disabled={loading}
            />

            <div className="question-actions">
              <span>{loading ? 'Synthesizing a grounded answer...' : 'Ready for document-grounded analysis.'}</span>
              <button
                type="button"
                onClick={() => askQuestion()}
                disabled={loading || !question.trim()}
                className="btn btn-primary"
              >
                {loading ? 'Synthesizing' : 'Ask Assistant'}
              </button>
            </div>

            {processing.mode === 'ask' && (
              <ProcessingSteps
                title={processingTitle}
                steps={currentSteps}
                activeIndex={processing.activeIndex}
                status={processing.status}
              />
            )}
          </section>

          <section className="answer-history" aria-label="Answer history">
            {answerHistory.length === 0 ? (
              <div className="empty-state">
                <h2>Research answers will appear here</h2>
                <p>Each response is organized with a key finding, detailed explanation, citations, and follow-up prompts.</p>
              </div>
            ) : (
              answerHistory.map((entry, index) => (
                <div key={entry.id} ref={index === 0 ? answerRef : null}>
                  <AnswerCard entry={entry} loading={loading} onFollowUp={askQuestion} />
                </div>
              ))
            )}
          </section>
        </section>

        <aside className="source-rail" aria-label="Citation and retrieval details">
          <RetrievalMetadata
            metadata={latestEntry?.metadata || {}}
            citations={latestEntry?.citations || []}
            showDebug={showDebug}
            onToggleDebug={setShowDebug}
          />

          <section className="source-preview">
            <div className="source-preview-header">
              <h2>Evidence Preview</h2>
              {latestEntry?.citations?.length > 0 && <span>{latestEntry.citations.length} sources</span>}
            </div>

            {latestEntry?.citations?.length > 0 ? (
              <div className="source-preview-list">
                {latestEntry.citations.slice(0, 6).map((citation, index) => (
                  <article key={`${citation.page}-${index}`} className="source-card">
                    <CitationPill page={citation.page} score={citation.score} />
                    <p>{citation.snippet}</p>
                    {showDebug && <span className="score-tag">sim {Number(citation.score).toFixed(3)}</span>}
                  </article>
                ))}
              </div>
            ) : (
              <p className="source-empty">Sources found in the document will appear after the first answer.</p>
            )}
          </section>
        </aside>
      </main>

      {toast && (
        <div className={`toast ${toast.type}`} role="status" aria-live="polite">
          <span aria-hidden="true">{toast.type === 'success' ? '✓' : '!'}</span>
          {toast.message}
        </div>
      )}
    </div>
  )
}

export default App
