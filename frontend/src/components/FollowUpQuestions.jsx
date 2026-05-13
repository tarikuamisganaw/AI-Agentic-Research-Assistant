function FollowUpQuestions({ questions = [], disabled, onSelect }) {
  if (!questions.length) return null

  return (
    <div className="follow-up-section">
      <h3>Suggested Follow-Up Questions</h3>
      <div className="follow-up-list">
        {questions.slice(0, 3).map(question => (
          <button
            key={question}
            type="button"
            className="follow-up-chip"
            disabled={disabled}
            onClick={() => onSelect(question)}
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  )
}

export default FollowUpQuestions
