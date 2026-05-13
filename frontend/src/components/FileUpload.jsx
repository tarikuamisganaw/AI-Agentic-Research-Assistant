import { useRef, useState } from 'react'

const formatFileSize = bytes => {
  if (!bytes) return null
  const units = ['B', 'KB', 'MB', 'GB']
  let size = bytes
  let unitIndex = 0
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex += 1
  }
  return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`
}

function PdfIcon() {
  return (
    <svg className="pdf-icon" viewBox="0 0 48 48" role="img" aria-label="PDF file">
      <path d="M12 4h16l8 8v32H12z" />
      <path d="M28 4v8h8" />
      <text x="17" y="31">PDF</text>
    </svg>
  )
}

function FileUpload({ file, disabled, processing, onFileSelect, onFileReject, onUpload }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const selectFile = selectedFile => {
    if (!selectedFile) return
    const isPdf = selectedFile.type === 'application/pdf' || selectedFile.name.toLowerCase().endsWith('.pdf')
    if (!isPdf) {
      onFileReject?.('Please choose a PDF file.')
      return
    }
    onFileSelect(selectedFile)
  }

  const openPicker = () => {
    if (!disabled) inputRef.current?.click()
  }

  const handleKeyDown = event => {
    if (disabled) return
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      openPicker()
    }
  }

  const handleDrop = event => {
    event.preventDefault()
    if (disabled) return
    setDragging(false)
    selectFile(event.dataTransfer.files?.[0])
  }

  return (
    <div className="upload-stack">
      <div
        className={`file-dropzone ${dragging ? 'is-dragging' : ''} ${disabled ? 'is-disabled' : ''}`}
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Upload a research PDF"
        aria-disabled={disabled}
        onClick={openPicker}
        onKeyDown={handleKeyDown}
        onDragEnter={event => {
          event.preventDefault()
          if (!disabled) setDragging(true)
        }}
        onDragOver={event => event.preventDefault()}
        onDragLeave={event => {
          event.preventDefault()
          setDragging(false)
        }}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf"
          className="hidden-file-input"
          disabled={disabled}
          onChange={event => selectFile(event.target.files?.[0])}
        />

        <PdfIcon />
        <div className="dropzone-copy">
          <h2>Drop your research PDF here</h2>
          <p>or click to browse</p>
        </div>

        {file && (
          <div className="selected-file" aria-live="polite">
            <span className="selected-file-name">{file.name}</span>
            {formatFileSize(file.size) && <span>{formatFileSize(file.size)}</span>}
          </div>
        )}
      </div>

      <button
        type="button"
        className="btn btn-secondary"
        onClick={onUpload}
        disabled={!file || disabled}
      >
        {processing ? 'Indexing document' : 'Index Document'}
      </button>
    </div>
  )
}

export default FileUpload
