# Enterprise RAG Assistant (PDF Q&A with Citations)

A Dockerized, cloud-deployable web app that lets users upload PDFs and ask questions with answers grounded in the documents (with citations). Includes an extractive fallback mode when an LLM key/quota isn’t available.

## Tech Stack
- **Backend:** FastAPI (Python)
- **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector DB:** FAISS
- **Frontend:** React (Vite)
- **Infra:** Docker + docker-compose
- **Features:** request tracing (request_id), multi-document global search

## Features
- Upload PDF → extract text per page → chunk into passages
- Index per-document (FAISS) or build **global** index across all PDFs
- Ask per-document or global questions
- Answers returned with citations: filename + page + chunk id
- Works without paid LLM (extractive fallback)

## Run with Docker
```bash
docker compose up --build
