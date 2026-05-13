function CitationPill({ page, score, compact = false }) {
  if (!page) return null

  const title = score ? `Page ${page} - relevance ${Number(score).toFixed(3)}` : `Page ${page}`

  return (
    <span className={`citation-pill ${compact ? 'citation-pill-compact' : ''}`} title={title}>
      <span aria-hidden="true">📄</span>
      <span>Page {page}</span>
    </span>
  )
}

export default CitationPill
