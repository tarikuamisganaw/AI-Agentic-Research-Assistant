import CitationPill from './CitationPill'

function RetrievalMetadata({ metadata = {}, citations = [], showDebug, onToggleDebug }) {
  const sections = metadata.retrieved_sections || []
  const chunksUsed = metadata.chunks_used ?? citations.length
  const searchMode = metadata.search_mode || 'Semantic retrieval completed'
  const answerType = metadata.answer_type || 'Document answer'
  const relevance = metadata.relevance_label || (citations.length ? 'Sources found in document' : 'Awaiting sources')

  return (
    <div className="metadata-panel">
      <div className="metadata-header">
        <h2>Retrieval Transparency</h2>
        <span>{relevance}</span>
      </div>

      <dl className="metadata-grid">
        <div>
          <dt>Search mode</dt>
          <dd>{searchMode}</dd>
        </div>
        <div>
          <dt>Answer type</dt>
          <dd>{answerType}</dd>
        </div>
        <div>
          <dt>Retrieved sections</dt>
          <dd>{chunksUsed || 'Sources found in document'}</dd>
        </div>
        <div>
          <dt>Primary source page</dt>
          <dd>
            {metadata.primary_source_page ? (
              <CitationPill page={metadata.primary_source_page} compact />
            ) : (
              'Not available'
            )}
          </dd>
        </div>
      </dl>

      {sections.length > 0 && (
        <div className="retrieved-list">
          {sections.slice(0, 5).map((section, index) => (
            <article key={`${section.page}-${index}`} className="retrieved-item">
              <div className="retrieved-item-header">
                <CitationPill page={section.page} score={section.score} compact />
                {showDebug && <span className="score-tag">sim {Number(section.score).toFixed(3)}</span>}
              </div>
              <p>{section.preview}</p>
            </article>
          ))}
        </div>
      )}

      {onToggleDebug && (
        <label className="debug-toggle">
          <input
            type="checkbox"
            checked={showDebug}
            onChange={event => onToggleDebug(event.target.checked)}
          />
          Show semantic similarity scores
        </label>
      )}
    </div>
  )
}

export default RetrievalMetadata
