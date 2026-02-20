from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pathlib import Path
import json
import os

import logging
import uuid

from dotenv import load_dotenv
from openai import OpenAIError

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pypdf import PdfReader


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

UPLOAD_DIR = Path("storage/uploads")
EXTRACT_DIR = Path("storage/extracted")
CHUNKS_DIR = Path("storage/chunks")
INDEX_DIR = Path("storage/index")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_INDEX_PATH = INDEX_DIR / "global.faiss"
GLOBAL_META_PATH = INDEX_DIR / "global.meta.json"

def list_all_chunk_files() -> list[Path]:
    return sorted(CHUNKS_DIR.glob("*.chunks.json"))


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id

    logger.info(f"[{request_id}] {request.method} {request.url.path}")

    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


def extractive_answer(question: str, hits: list[dict]) -> str:
    if not hits:
        return "I don’t know based on the documents I have."

    top = hits[:2]
    combined = "\n\n".join(
        [f"{h['snippet']} [S{i+1}]" for i, h in enumerate(top)]
    )

    return (
        "Based on the document passages, here’s the most relevant information:\n\n"
        f"{combined}"
    )


_model = None
def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


@app.get("/")
def root(request: Request):
    return {
        "request_id": request.state.request_id,
        "message": "Enterprise RAG Assistant is running 🚀",
    }


@app.get("/health")
def health_check(request: Request):
    return {"request_id": request.state.request_id, "status": "ok"}


@app.get("/documents")
def list_documents(request: Request):
    pdfs = sorted([p.name for p in UPLOAD_DIR.glob("*.pdf")])
    return {"request_id": request.state.request_id, "documents": pdfs}


def extract_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": i, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def load_chunks_for_pdf_stem(stem: str) -> list[dict]:
    chunks_file = CHUNKS_DIR / f"{stem}.chunks.json"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file.name}")
    data = json.loads(chunks_file.read_text(encoding="utf-8"))
    return data.get("chunks", [])


def build_faiss_index(chunks: list[dict]) -> tuple[faiss.Index, list[dict]]:
    texts = [c["text"] for c in chunks]
    model = get_model()
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    dim = emb.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(emb.astype(np.float32))

    meta = [{
        "filename": c["filename"],
        "page": c["page"],
        "chunk_id": c["chunk_id"],
        "text": c["text"][:500]
    } for c in chunks]

    return index, meta

def load_global_index() -> tuple[faiss.Index, list[dict]]:
    if not GLOBAL_INDEX_PATH.exists() or not GLOBAL_META_PATH.exists():
        raise FileNotFoundError("Global index not found. Run /index_global first.")

    index = faiss.read_index(str(GLOBAL_INDEX_PATH))
    meta = json.loads(GLOBAL_META_PATH.read_text(encoding="utf-8")).get("meta", [])
    return index, meta

def load_index(pdf_stem: str) -> tuple[faiss.Index, list[dict]]:
    index_path = INDEX_DIR / f"{pdf_stem}.faiss"
    meta_path = INDEX_DIR / f"{pdf_stem}.meta.json"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index not found. Run /index/{pdf_stem} first.")

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8")).get("meta", [])
    return index, meta

def search_global(query: str, k: int = 6) -> list[dict]:
    index, meta = load_global_index()

    model = get_model()
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    scores, ids = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        m = meta[idx]
        results.append({
            "score": float(score),
            "filename": m["filename"],
            "page": m["page"],
            "chunk_id": m["chunk_id"],
            "snippet": m["text"],
        })
    return results

def search(pdf_stem: str, query: str, k: int = 4) -> list[dict]:
    index, meta = load_index(pdf_stem)

    model = get_model()
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    scores, ids = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        m = meta[idx]
        results.append({
            "score": float(score),
            "filename": m["filename"],
            "page": m["page"],
            "chunk_id": m["chunk_id"],
            "snippet": m["text"],
        })
    return results


@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    safe_name = Path(file.filename).name
    pdf_path = UPLOAD_DIR / safe_name

    data = await file.read()
    pdf_path.write_bytes(data)

    pages = extract_pages(pdf_path)

    all_chunks = []
    for p in pages:
        for idx, c in enumerate(chunk_text(p["text"]), start=1):
            all_chunks.append({
                "filename": safe_name,
                "page": p["page"],
                "chunk_id": idx,
                "text": c
            })

    out_path = EXTRACT_DIR / f"{pdf_path.stem}.json"
    out_path.write_text(
        json.dumps({"filename": safe_name, "pages": pages}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    chunks_path = CHUNKS_DIR / f"{pdf_path.stem}.chunks.json"
    chunks_path.write_text(
        json.dumps({"filename": safe_name, "chunks": all_chunks}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    preview = ""
    for p in pages:
        if p["text"].strip():
            preview = p["text"][:300]
            break

    return {
        "request_id": request.state.request_id,
        "status": "saved_and_extracted",
        "filename": safe_name,
        "pages": len(pages),
        "chunks": len(all_chunks),
        "preview": preview,
        "extracted_json": out_path.name,
        "chunks_json": chunks_path.name,
    }


@app.post("/index_global")
def index_global(request: Request):
    chunk_files = list_all_chunk_files()
    if not chunk_files:
        raise HTTPException(status_code=400, detail="No chunk files found. Upload PDFs first.")

    all_chunks = []
    for f in chunk_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        all_chunks.extend(data.get("chunks", []))

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No chunks found to index.")

    index, meta = build_faiss_index(all_chunks)

    faiss.write_index(index, str(GLOBAL_INDEX_PATH))
    GLOBAL_META_PATH.write_text(json.dumps({"meta": meta}, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "request_id": request.state.request_id,
        "status": "indexed_global",
        "documents": len(chunk_files),
        "vectors": index.ntotal,
        "index_file": GLOBAL_INDEX_PATH.name,
        "meta_file": GLOBAL_META_PATH.name,
    }

@app.post("/index/{pdf_stem}")
def index_document(pdf_stem: str, request: Request):
    try:
        chunks = load_chunks_for_pdf_stem(pdf_stem)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks found to index.")

    index, meta = build_faiss_index(chunks)

    index_path = INDEX_DIR / f"{pdf_stem}.faiss"
    meta_path = INDEX_DIR / f"{pdf_stem}.meta.json"

    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps({"meta": meta}, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "request_id": request.state.request_id,
        "status": "indexed",
        "pdf_stem": pdf_stem,
        "vectors": index.ntotal,
        "index_file": index_path.name,
        "meta_file": meta_path.name
    }


class AskRequest(BaseModel):
    pdf_stem: str
    question: str
    top_k: int = 4


@app.post("/ask")
def ask(req: AskRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        hits = search(req.pdf_stem, req.question, k=req.top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "request_id": request.state.request_id,
        "question": req.question,
        "pdf_stem": req.pdf_stem,
        "matches": hits
    }

class AnswerGlobalRequest(BaseModel):
    question: str
    top_k: int = 6

@app.post("/answer_global")
def answer_global(req: AnswerGlobalRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        hits = search_global(req.question, k=req.top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Guardrail: weak retrieval
    if not hits or max(h["score"] for h in hits) < 0.03:
        return {
            "request_id": request.state.request_id,
            "answer": "I don’t know based on the documents I have.",
            "citations": [],
            "mode": "no_support"
        }

    # No paid LLM → extractive answer
    if client is None:
        return {
            "request_id": request.state.request_id,
            "answer": extractive_answer(req.question, hits),
            "citations": [
                {
                    "source_id": f"S{i+1}",
                    "filename": h["filename"],
                    "page": h["page"],
                    "chunk_id": h["chunk_id"],
                    "snippet": h["snippet"],
                    "score": h["score"],
                }
                for i, h in enumerate(hits)
            ],
            "mode": "extractive_global",
        }

    # Build sources for LLM
    context_lines = []
    for i, h in enumerate(hits, start=1):
        context_lines.append(
            f"[S{i}] ({h['filename']}, page {h['page']}) {h['snippet']}"
        )
    context = "\n\n".join(context_lines)

    instructions = (
        "You are a document Q&A assistant. Answer ONLY using the provided sources.\n"
        "If the answer is not contained in the sources, say: "
        "\"I don’t know based on the documents I have.\".\n"
        "Cite every factual statement using [S1], [S2], etc."
    )

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=instructions,
            input=f"Question: {req.question}\n\nSources:\n{context}",
        )
    except OpenAIError:
        return {
            "request_id": request.state.request_id,
            "answer": extractive_answer(req.question, hits),
            "citations": [
                {
                    "source_id": f"S{i+1}",
                    "filename": h["filename"],
                    "page": h["page"],
                    "chunk_id": h["chunk_id"],
                    "snippet": h["snippet"],
                    "score": h["score"],
                }
                for i, h in enumerate(hits)
            ],
            "mode": "extractive_global_fallback",
        }

    return {
        "request_id": request.state.request_id,
        "answer": response.output_text,
        "citations": [
            {
                "source_id": f"S{i+1}",
                "filename": h["filename"],
                "page": h["page"],
                "chunk_id": h["chunk_id"],
                "snippet": h["snippet"],
                "score": h["score"],
            }
            for i, h in enumerate(hits)
        ],
        "mode": "llm_global",
    }

class AnswerRequest(BaseModel):
    pdf_stem: str
    question: str
    top_k: int = 4


@app.post("/answer")
def answer(req: AnswerRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        hits = search(req.pdf_stem, req.question, k=req.top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not hits or max(h["score"] for h in hits) < 0.03:
        return {
            "request_id": request.state.request_id,
            "answer": "I don’t know based on the documents I have.",
            "citations": [],
            "matches": hits,
            "mode": "no_support"
        }

    if client is None:
        return {
            "request_id": request.state.request_id,
            "answer": extractive_answer(req.question, hits),
            "citations": [
                {
                    "source_id": f"S{i+1}",
                    "filename": h["filename"],
                    "page": h["page"],
                    "chunk_id": h["chunk_id"],
                    "snippet": h["snippet"],
                    "score": h["score"],
                }
                for i, h in enumerate(hits)
            ],
            "matches": hits,
            "mode": "extractive",
        }

    context_lines = []
    for i, h in enumerate(hits, start=1):
        context_lines.append(
            f"[S{i}] ({h['filename']}, page {h['page']}) {h['snippet']}"
        )
    context = "\n\n".join(context_lines)

    instructions = (
        "You are a document Q&A assistant. "
        "Answer ONLY using the provided sources.\n"
        "If the answer is not contained in the sources, say: "
        "\"I don’t know based on the documents I have.\".\n"
        "Cite every factual statement using [S1], [S2], etc."
    )

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=instructions,
            input=f"Question: {req.question}\n\nSources:\n{context}",
        )
    except OpenAIError:
        return {
            "request_id": request.state.request_id,
            "answer": extractive_answer(req.question, hits),
            "citations": [
                {
                    "source_id": f"S{i+1}",
                    "filename": h["filename"],
                    "page": h["page"],
                    "chunk_id": h["chunk_id"],
                    "snippet": h["snippet"],
                    "score": h["score"],
                }
                for i, h in enumerate(hits)
            ],
            "matches": hits,
            "mode": "extractive_fallback",
        }

    return {
        "request_id": request.state.request_id,
        "answer": response.output_text,
        "citations": [
            {
                "source_id": f"S{i+1}",
                "filename": h["filename"],
                "page": h["page"],
                "chunk_id": h["chunk_id"],
                "snippet": h["snippet"],
                "score": h["score"],
            }
            for i, h in enumerate(hits)
        ],
        "matches": hits,
        "mode": "llm",
    }


@app.get("/config")
def config_check(request: Request):
    return {
        "request_id": request.state.request_id,
        "model": OPENAI_MODEL,
        "has_api_key": bool(OPENAI_API_KEY),
    }

class AskGlobalRequest(BaseModel):
    question: str
    top_k: int = 6

@app.post("/ask_global")
def ask_global(req: AskGlobalRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        hits = search_global(req.question, k=req.top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "request_id": request.state.request_id,
        "question": req.question,
        "matches": hits
    }

