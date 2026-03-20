import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")
AUTH0_DOMAIN   = os.getenv("AUTH0_DOMAIN")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
ROLES_CLAIM    = "https://rag-assistant-api/roles"

# ── Storage ───────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("storage/uploads")
CHUNKS_DIR = Path("storage/chunks")
INDEX_DIR  = Path("storage/index")
for _d in (UPLOAD_DIR, CHUNKS_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

GLOBAL_INDEX_DIR = INDEX_DIR / "global"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")

# ── App ───────────────────────────────────────────────────────────────────────
def parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "")
    if raw.strip():
        return [o.strip() for o in raw.split(",") if o.strip()]
    return ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"]

app = FastAPI(title="Enterprise RAG Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Embeddings singleton ──────────────────────────────────────────────────────
_embeddings: FastEmbedEmbeddings | None = None

def get_embeddings() -> FastEmbedEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings

# ── Auth0 JWT validation ──────────────────────────────────────────────────────
security     = HTTPBearer()
_jwks_cache: dict | None = None

def _get_jwks() -> dict:
    global _jwks_cache
    if _jwks_cache is None:
        resp = httpx.get(
            f"https://{AUTH0_DOMAIN}/.well-known/jwks.json", timeout=10
        )
        resp.raise_for_status()
        _jwks_cache = resp.json()
    return _jwks_cache

def _verify_token(token: str) -> dict:
    if not AUTH0_DOMAIN or not AUTH0_AUDIENCE:
        raise HTTPException(status_code=500, detail="Auth0 not configured on server.")
    try:
        jwks   = _get_jwks()
        header = jwt.get_unverified_header(token)
        rsa_key = next(
            (
                {"kty": k["kty"], "kid": k["kid"], "use": k["use"], "n": k["n"], "e": k["e"]}
                for k in jwks["keys"]
                if k["kid"] == header["kid"]
            ),
            None,
        )
        if not rsa_key:
            raise HTTPException(status_code=401, detail="Matching public key not found.")
        return jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=AUTH0_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/",
        )
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    return _verify_token(credentials.credentials)

def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if "admin" not in user.get(ROLES_CLAIM, []):
        raise HTTPException(status_code=403, detail="Admin role required.")
    return user

# ── Request-ID middleware ─────────────────────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = rid
    logger.info("[%s] %s %s", rid, request.method, request.url.path)
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response

# ── Storage helpers ───────────────────────────────────────────────────────────
def _remove_path(p: Path):
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()

def cleanup_document(pdf_name: str) -> dict:
    safe  = Path(pdf_name).name
    stem  = Path(safe).stem
    paths = [
        UPLOAD_DIR / safe,
        CHUNKS_DIR / f"{stem}.chunks.json",
        INDEX_DIR  / stem,
    ]
    deleted = []
    for p in paths:
        if p.exists():
            _remove_path(p)
            deleted.append(p.name)
    return {"filename": safe, "deleted_files": deleted, "deleted": safe in deleted}

def cleanup_global_index():
    _remove_path(GLOBAL_INDEX_DIR)

# ── LangChain helpers ─────────────────────────────────────────────────────────
def _load_vectorstore(index_dir: Path) -> FAISS:
    return FAISS.load_local(
        str(index_dir), get_embeddings(), allow_dangerous_deserialization=True
    )

def _search(vectorstore: FAISS, question: str, k: int) -> list[dict]:
    results = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    return [
        {
            "score":    float(score),
            "filename": doc.metadata.get("filename", ""),
            "page":     doc.metadata.get("page", 0),
            "chunk_id": doc.metadata.get("chunk_id", 0),
            "snippet":  doc.page_content[:500],
        }
        for doc, score in results
    ]

_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a document Q&A assistant. Use the provided source passages to answer the question.\n"
        "Synthesize relevant information across sources — you do not need a verbatim quote to answer.\n"
        "Only say \"I don't know based on the documents I have.\" if the sources contain "
        "NO information relevant to the question whatsoever.\n"
        "Cite every factual claim using [S1], [S2], etc.\n\n"
        "Sources:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    ),
)

def _build_answer(question: str, hits: list[dict], request_id: str, mode: str) -> dict:
    citations = [{"source_id": f"S{i+1}", **h} for i, h in enumerate(hits)]

    if not OPENAI_API_KEY:
        answer = "Based on the document passages:\n\n" + "\n\n".join(
            f"{h['snippet']} [S{i+1}]" for i, h in enumerate(hits[:2])
        )
        return {"request_id": request_id, "answer": answer, "citations": citations, "mode": f"extractive_{mode}"}

    try:
        llm     = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
        context = "\n\n".join(
            f"[S{i+1}] ({h['filename']}, page {h['page']}) {h['snippet']}"
            for i, h in enumerate(hits)
        )
        chain  = _PROMPT | llm
        result = chain.invoke({"context": context, "question": question})
        answer = result.content if hasattr(result, "content") else str(result)
        return {"request_id": request_id, "answer": answer, "citations": citations, "mode": f"llm_{mode}"}
    except Exception:
        logger.exception("LLM call failed, using extractive fallback")
        answer = "Based on the document passages:\n\n" + "\n\n".join(
            f"{h['snippet']} [S{i+1}]" for i, h in enumerate(hits[:2])
        )
        return {"request_id": request_id, "answer": answer, "citations": citations, "mode": f"extractive_{mode}_fallback"}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root(request: Request):
    return {"request_id": request.state.request_id, "message": "Enterprise RAG Assistant"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/config")
def config_check(request: Request):
    return {
        "request_id": request.state.request_id,
        "model":       OPENAI_MODEL,
        "has_api_key": bool(OPENAI_API_KEY),
    }

@app.get("/documents")
def list_documents(request: Request, user: dict = Depends(get_current_user)):
    pdfs = sorted(p.name for p in UPLOAD_DIR.glob("*.pdf"))
    return {"request_id": request.state.request_id, "documents": pdfs}

# ── Upload (admin only) ───────────────────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(require_admin),
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    safe_name = Path(file.filename).name
    pdf_path  = UPLOAD_DIR / safe_name
    pdf_path.write_bytes(await file.read())

    # LangChain: load pages and split into chunks
    loader   = PyPDFLoader(str(pdf_path))
    pages    = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks   = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["filename"] = safe_name
        chunk.metadata["chunk_id"] = i + 1
        chunk.metadata["page"]     = int(chunk.metadata.get("page", 0)) + 1  # 1-indexed

    # Persist chunks as JSON for global indexing
    chunks_data = [
        {"text": c.page_content, "filename": c.metadata["filename"],
         "page": c.metadata["page"], "chunk_id": c.metadata["chunk_id"]}
        for c in chunks
    ]
    (CHUNKS_DIR / f"{pdf_path.stem}.chunks.json").write_text(
        json.dumps({"filename": safe_name, "chunks": chunks_data}, indent=2)
    )

    preview = chunks[0].page_content[:300] if chunks else ""
    return {
        "request_id": request.state.request_id,
        "status":     "uploaded",
        "filename":   safe_name,
        "pages":      len(pages),
        "chunks":     len(chunks),
        "preview":    preview,
    }

# ── Indexing ──────────────────────────────────────────────────────────────────
@app.post("/index/{pdf_stem}")
def index_document(pdf_stem: str, request: Request, user: dict = Depends(get_current_user)):
    chunks_file = CHUNKS_DIR / f"{pdf_stem}.chunks.json"
    if not chunks_file.exists():
        raise HTTPException(status_code=404, detail=f"No chunks found for '{pdf_stem}'. Upload PDF first.")

    data   = json.loads(chunks_file.read_text())
    chunks = data.get("chunks", [])
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks to index.")

    docs = [
        Document(
            page_content=c["text"],
            metadata={"filename": c["filename"], "page": c["page"], "chunk_id": c["chunk_id"]},
        )
        for c in chunks
    ]

    logger.info("[%s] Indexing %s (%d chunks)", request.state.request_id, pdf_stem, len(docs))
    vs = FAISS.from_documents(docs, get_embeddings())
    vs.save_local(str(INDEX_DIR / pdf_stem))

    return {"request_id": request.state.request_id, "status": "indexed", "pdf_stem": pdf_stem, "vectors": len(docs)}

@app.post("/index_global")
def index_global(request: Request, user: dict = Depends(get_current_user)):
    chunk_files = sorted(CHUNKS_DIR.glob("*.chunks.json"))
    if not chunk_files:
        raise HTTPException(status_code=400, detail="No PDFs uploaded yet.")

    all_docs = []
    for f in chunk_files:
        for c in json.loads(f.read_text()).get("chunks", []):
            all_docs.append(Document(
                page_content=c["text"],
                metadata={"filename": c["filename"], "page": c["page"], "chunk_id": c["chunk_id"]},
            ))

    if not all_docs:
        raise HTTPException(status_code=400, detail="No chunks found.")

    logger.info("[%s] Building global index (%d chunks)", request.state.request_id, len(all_docs))
    vs = FAISS.from_documents(all_docs, get_embeddings())
    vs.save_local(str(GLOBAL_INDEX_DIR))

    return {"request_id": request.state.request_id, "status": "indexed_global", "vectors": len(all_docs)}

# ── Answer ────────────────────────────────────────────────────────────────────
class AnswerRequest(BaseModel):
    pdf_stem: str
    question: str
    top_k:    int = 4

@app.post("/answer")
def answer(req: AnswerRequest, request: Request, user: dict = Depends(get_current_user)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    logger.info("[answer] question=%r pdf_stem=%r", req.question, req.pdf_stem)

    index_path = INDEX_DIR / req.pdf_stem
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Index not found. Run /index/{pdf_stem} first.")

    vs   = _load_vectorstore(index_path)
    hits = _search(vs, req.question, req.top_k)

    if not hits:
        return {"request_id": request.state.request_id, "answer": "I don't know based on the documents I have.", "citations": [], "mode": "no_support"}

    return _build_answer(req.question, hits, request.state.request_id, "doc")

class AnswerGlobalRequest(BaseModel):
    question: str
    top_k:    int = 6

@app.post("/answer_global")
def answer_global(req: AnswerGlobalRequest, request: Request, user: dict = Depends(get_current_user)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    logger.info("[answer_global] question=%r", req.question)

    if not GLOBAL_INDEX_DIR.exists():
        raise HTTPException(status_code=404, detail="Global index not found. Run /index_global first.")

    vs   = _load_vectorstore(GLOBAL_INDEX_DIR)
    hits = _search(vs, req.question, req.top_k)

    if not hits:
        return {"request_id": request.state.request_id, "answer": "I don't know based on the documents I have.", "citations": [], "mode": "no_support"}

    return _build_answer(req.question, hits, request.state.request_id, "global")

# ── Delete (admin only) ───────────────────────────────────────────────────────
@app.delete("/documents/{filename}")
def delete_document(filename: str, request: Request, user: dict = Depends(require_admin)):
    result = cleanup_document(filename)
    if not result["deleted"]:
        raise HTTPException(status_code=404, detail="Document not found.")
    cleanup_global_index()
    return {"request_id": request.state.request_id, "status": "deleted", **result}

@app.delete("/documents")
def delete_all_documents(request: Request, user: dict = Depends(require_admin)):
    pdfs    = [p.name for p in UPLOAD_DIR.glob("*.pdf")]
    deleted = [n for n in pdfs if cleanup_document(n)["deleted"]]
    cleanup_global_index()
    return {"request_id": request.state.request_id, "status": "deleted_all", "count": len(deleted), "deleted": deleted}

@app.get("/admin/stats")
def admin_stats(request: Request, user: dict = Depends(require_admin)):
    num_pdfs   = len(list(UPLOAD_DIR.glob("*.pdf")))
    num_chunks = len(list(CHUNKS_DIR.glob("*.chunks.json")))
    global_vectors = 0
    if GLOBAL_INDEX_DIR.exists():
        vs = _load_vectorstore(GLOBAL_INDEX_DIR)
        global_vectors = vs.index.ntotal
    return {
        "request_id":    request.state.request_id,
        "pdf_count":     num_pdfs,
        "chunk_files":   num_chunks,
        "global_vectors": global_vectors,
    }

# ── LangGraph Agentic RAG ─────────────────────────────────────────────────────
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

class RAGState(TypedDict):
    question:        str
    pdf_stem:        Optional[str]
    available_docs:  list[str]
    rewritten_query: str
    search_mode:     str        # "single" | "global"
    hits:            list[dict]
    answer:          str
    citations:       list[dict]
    reasoning:       str

def _node_route_and_rewrite(state: RAGState) -> RAGState:
    """Agent node: decide search strategy and rewrite query for better retrieval."""
    if not OPENAI_API_KEY:
        mode = "single" if state["pdf_stem"] else "global"
        return {**state, "rewritten_query": state["question"],
                "search_mode": mode, "reasoning": "Direct search (no LLM routing available)."}

    docs_list = ", ".join(state["available_docs"]) if state["available_docs"] else "none"
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)

    prompt = (
        f'You are a routing agent for a document Q&A system.\n'
        f'User question: "{state["question"]}"\n'
        f'Available documents: {docs_list}\n'
        f'Hinted document: {state["pdf_stem"] or "none"}\n\n'
        f'Tasks:\n'
        f'1. Rewrite the question to be more specific for semantic search (remove filler, add context).\n'
        f'2. Choose search_mode: "single" if question targets a specific hinted document, "global" otherwise.\n\n'
        f'Respond ONLY as valid JSON:\n'
        f'{{"rewritten_query": "...", "search_mode": "single|global", "reasoning": "one sentence explanation"}}'
    )

    try:
        result  = llm.invoke([HumanMessage(content=prompt)])
        raw     = result.content.strip()
        # GPT-4o sometimes wraps JSON in markdown fences — strip them
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()
        parsed  = json.loads(raw)
        return {
            **state,
            "rewritten_query": parsed.get("rewritten_query", state["question"]),
            "search_mode":     parsed.get("search_mode", "global"),
            "reasoning":       parsed.get("reasoning", ""),
        }
    except Exception:
        mode = "single" if state["pdf_stem"] else "global"
        return {**state, "rewritten_query": state["question"],
                "search_mode": mode, "reasoning": "Fallback routing used."}

def _node_retrieve(state: RAGState) -> RAGState:
    """Agent node: retrieve relevant chunks from FAISS."""
    query = state["rewritten_query"]

    if state["search_mode"] == "single" and state["pdf_stem"]:
        index_path = INDEX_DIR / state["pdf_stem"]
        if not index_path.exists():
            logger.warning("[agent] single index not found: %s", index_path)
            return {**state, "hits": []}
        vs   = _load_vectorstore(index_path)
        hits = _search(vs, query, 4)
    else:
        if not GLOBAL_INDEX_DIR.exists():
            logger.warning("[agent] global index not found")
            return {**state, "hits": []}
        vs   = _load_vectorstore(GLOBAL_INDEX_DIR)
        hits = _search(vs, query, 6)

    logger.info("[agent] retrieved %d chunks mode=%s", len(hits), state["search_mode"])
    return {**state, "hits": hits}

def _node_generate(state: RAGState) -> RAGState:
    """Agent node: generate grounded answer from retrieved context."""
    if not state["hits"]:
        return {**state, "answer": "I don't know based on the documents I have.", "citations": []}

    # Rewritten query was for retrieval only — answer the original user question
    result = _build_answer(state["question"], state["hits"], "", state["search_mode"])
    return {**state, "answer": result["answer"], "citations": result["citations"]}

def _build_rag_graph():
    g = StateGraph(RAGState)
    g.add_node("route",    _node_route_and_rewrite)
    g.add_node("retrieve", _node_retrieve)
    g.add_node("generate", _node_generate)
    g.set_entry_point("route")
    g.add_edge("route",    "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()

_rag_graph = None
def get_rag_graph():
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = _build_rag_graph()
    return _rag_graph

class AgentRequest(BaseModel):
    question: str
    pdf_stem: Optional[str] = None

@app.post("/agent/answer")
def agent_answer(req: AgentRequest, request: Request, user: dict = Depends(get_current_user)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    logger.info("[agent/answer] question=%r pdf_stem=%r", req.question, req.pdf_stem)

    available_docs = sorted(p.stem for p in UPLOAD_DIR.glob("*.pdf"))

    initial_state: RAGState = {
        "question":        req.question,
        "pdf_stem":        req.pdf_stem,
        "available_docs":  available_docs,
        "rewritten_query": req.question,
        "search_mode":     "single" if req.pdf_stem else "global",
        "hits":            [],
        "answer":          "",
        "citations":       [],
        "reasoning":       "",
    }

    result = get_rag_graph().invoke(initial_state)

    return {
        "request_id":      request.state.request_id,
        "answer":          result["answer"],
        "citations":       result["citations"],
        "reasoning":       result["reasoning"],
        "rewritten_query": result["rewritten_query"],
        "search_mode":     result["search_mode"],
        "mode":            "agent",
    }
