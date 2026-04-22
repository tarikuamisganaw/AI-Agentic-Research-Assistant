import re

def clean_pdf_text(text: str) -> str:
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\b(BOS|EOS|FIG\.?|Figure\s*\d+)\b', '', text, flags=re.IGNORECASE)
    lines = text.split('\n')
    cleaned = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: cleaned.append('\n'); continue
        if i < len(lines) - 1 and not re.search(r'[.!?]\s*$', line): cleaned.append(line + ' ')
        else: cleaned.append(line + '\n')
    text = ''.join(cleaned)
    return re.sub(r'\s{3,}', '  ', re.sub(r'\n{3,}', '\n\n', text)).strip()

def get_citation_snippet(text: str, max_len: int = 150) -> str:
    clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    sentences = re.split(r'(?<=[.!?])\s+', clean.strip())
    valid = [s.strip() for s in sentences if len(s.strip()) > 10]
    first = valid[0] if valid else clean[:max_len]
    return (first[:max_len].rsplit(' ', 1)[0] + "...") if len(first) > max_len else first.strip()