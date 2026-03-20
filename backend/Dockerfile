FROM python:3.11-slim

# Needed for faiss (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY app ./app

# Pre-download the ONNX embedding model so first requests aren't slow
# and the container works offline / without HuggingFace rate limits
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('sentence-transformers/all-MiniLM-L6-v2').embed(['warmup']))"

# Optional: create storage dirs (app also creates them)
RUN mkdir -p storage/uploads storage/extracted storage/chunks storage/index

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
