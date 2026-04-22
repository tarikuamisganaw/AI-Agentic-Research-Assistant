FROM python:3.10-slim

WORKDIR /code

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend/ ./backend/

# Set environment variables
ENV PYTHONPATH=/code
ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Run the this FastAPI app
# Note: main:app is "from backend/main.py import app"
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]