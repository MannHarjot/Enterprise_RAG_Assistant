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

# Optional: create storage dirs (app also creates them)
RUN mkdir -p storage/uploads storage/extracted storage/chunks storage/index

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
