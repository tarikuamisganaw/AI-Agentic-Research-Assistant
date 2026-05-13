function StepIcon({ state }) {
  if (state === 'complete') {
    return (
      <span className="step-icon is-complete" aria-hidden="true">
        <svg viewBox="0 0 20 20">
          <path d="M4.5 10.5 8 14l7.5-8" />
        </svg>
      </span>
    )
  }

  if (state === 'active') {
    return <span className="step-icon step-spinner" aria-hidden="true" />
  }

  return <span className="step-icon" aria-hidden="true" />
}

function ProcessingSteps({ title = 'Processing', steps = [], activeIndex = 0, status = 'idle' }) {
  if (!steps.length || status === 'idle') return null

  return (
    <div className={`processing-card is-${status}`} role="status" aria-live="polite">
      <div className="processing-card-header">
        <span>{title}</span>
        <span>{status === 'complete' ? 'Complete' : status === 'error' ? 'Needs attention' : 'In progress'}</span>
      </div>
      <ol className="processing-steps">
        {steps.map((step, index) => {
          const state = status === 'complete' || index < activeIndex
            ? 'complete'
            : index === activeIndex
              ? 'active'
              : 'pending'

          return (
            <li key={step} className={`processing-step is-${state}`}>
              <StepIcon state={state} />
              <span>{step}</span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}

export default ProcessingSteps
