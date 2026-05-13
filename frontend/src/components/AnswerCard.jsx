import React from 'react'
import ReactMarkdown from 'react-markdown'
import CitationPill from './CitationPill'
import FollowUpQuestions from './FollowUpQuestions'

const citationPattern = /(\(?\bPage\s+\d+\b\)?)/gi

const stripMarkdown = text => text
  .replace(/[#*_`>~-]/g, '')
  .replace(/\[(.*?)\]\(.*?\)/g, '$1')
  .replace(/\s+/g, ' ')
  .trim()

const extractKeyFinding = answer => {
  if (!answer) return ''

  const headingMatch = answer.match(/(?:^|\n)#{1,4}\s*Key Finding\s*\n([\s\S]*?)(?=\n#{1,4}\s|\n\*\*[^*\n]+\*\*|$)/i)
  if (headingMatch?.[1]) return stripMarkdown(headingMatch[1]).slice(0, 420)

  const boldMatch = answer.match(/\*\*Key Finding\*\*:?\s*([\s\S]*?)(?=\n\n|\n\*\*[^*\n]+\*\*|$)/i)
  if (boldMatch?.[1]) return stripMarkdown(boldMatch[1]).slice(0, 420)

  const firstParagraph = answer
    .split(/\n{2,}/)
    .map(stripMarkdown)
    .find(Boolean)

  return (firstParagraph || stripMarkdown(answer)).slice(0, 420)
}

const renderCitationText = children => React.Children.map(children, child => {
  if (typeof child === 'string') {
    return child.split(citationPattern).map((part, index) => {
      const match = part.match(/\bPage\s+(\d+)\b/i)
      if (!match) return part
      return <CitationPill key={`${part}-${index}`} page={match[1]} compact />
    })
  }

  if (React.isValidElement(child) && child.props?.children) {
    return React.cloneElement(child, {
      ...child.props,
      children: renderCitationText(child.props.children),
    })
  }

  return child
})

function MarkdownContent({ answer }) {
  const components = {
    p: ({ children }) => <p>{renderCitationText(children)}</p>,
    li: ({ children }) => <li>{renderCitationText(children)}</li>,
    h2: ({ children }) => <h2>{renderCitationText(children)}</h2>,
    h3: ({ children }) => <h3>{renderCitationText(children)}</h3>,
    h4: ({ children }) => <h4>{renderCitationText(children)}</h4>,
    strong: ({ children }) => <strong>{renderCitationText(children)}</strong>,
  }

  return (
    <div className="markdown-body">
      <ReactMarkdown components={components}>{answer}</ReactMarkdown>
    </div>
  )
}

function AnswerCard({ entry, loading, onFollowUp }) {
  const suggestedQuestions = entry.metadata?.suggested_followups || []
  const keyFinding = extractKeyFinding(entry.answer)

  return (
    <article className="history-turn">
      <section className="user-question-card" aria-label="User question">
        <span className="turn-label">User Question</span>
        <p>{entry.question}</p>
      </section>

      <section className="answer-card" aria-label="AI answer">
        <div className="answer-card-header">
          <div>
            <span className="answer-eyebrow">AI Answer</span>
            <h2>{entry.metadata?.answer_type || 'Research answer'}</h2>
          </div>
          {entry.metadata?.primary_source_page && (
            <CitationPill page={entry.metadata.primary_source_page} compact />
          )}
        </div>

        <section className="answer-section key-finding-section">
          <h3>Key Finding</h3>
          <p>{keyFinding || 'The assistant could not extract a concise finding from this response.'}</p>
        </section>

        <section className="answer-section">
          <h3>Detailed Explanation</h3>
          <MarkdownContent answer={entry.answer} />
        </section>

        {entry.citations?.length > 0 && (
          <section className="answer-section evidence-section">
            <h3>Evidence / Citations</h3>
            <div className="citation-card-list">
              {entry.citations.map((citation, index) => (
                <article key={`${citation.page}-${index}`} className="citation-card">
                  <div className="citation-card-header">
                    <CitationPill page={citation.page} score={citation.score} />
                    {citation.score && <span>Relevance {Number(citation.score).toFixed(3)}</span>}
                  </div>
                  <p>{citation.snippet}</p>
                </article>
              ))}
            </div>
          </section>
        )}

        <FollowUpQuestions questions={suggestedQuestions} disabled={loading} onSelect={onFollowUp} />
      </section>
    </article>
  )
}

export default AnswerCard
