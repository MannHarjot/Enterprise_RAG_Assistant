# Enterprise RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application with Auth0 authentication, role-based access control, and a LangGraph agentic pipeline. Built as a portfolio project demonstrating production RAG engineering patterns.

---

## Live Demo

> **URL:** *(add Render URL once deployed)*

| Role | Email | Password |
|---|---|---|
| Admin | admin@demo.com | Admin@123 |
| Employee | employee@demo.com | Employee@123 |

**Admin** — upload PDFs, delete documents, ask questions
**Employee** — ask questions only (read-only access)

---

## What This Started As

The original version was a basic PDF Q&A app:

- FastAPI backend, React frontend, Docker Compose
- `sentence-transformers` (PyTorch) for embeddings
- FAISS for vector search
- Manual chunking pipeline (custom code)
- No authentication
- Deployed on Render free tier — **crashed on every PDF upload with OOM errors**
- "Index not found" error on global search

---

## What Changed and Why

### 1. Replaced PyTorch embeddings with fastembed (ONNX)

**Problem:** `torch` + `sentence-transformers` = ~500MB RAM. Render free tier = 512MB limit. The container hit the ceiling the moment embeddings were computed on upload and died.

**Fix:** Replaced with `fastembed` (ONNX runtime). Same model (`all-MiniLM-L6-v2`), ~50MB footprint, no PyTorch dependency. Container now runs at ~150MB steady state.

---

### 2. Refactored pipeline to LangChain

**Before:** Custom one-off code for PDF parsing, chunking, FAISS calls, and prompt building.

**After:** `PyPDFLoader` → `RecursiveCharacterTextSplitter` → `FAISS.from_documents()` → `PromptTemplate | ChatOpenAI` chain.

**Side effect discovered:** LangChain's `FAISS.save_local()` creates a **directory** (not a file) with `index.faiss` + `index.pkl` inside. The original cleanup code used `Path.unlink()` which threw errors. Fixed by switching to `shutil.rmtree()`.

---

### 3. Fixed the "Global Index Not Found" bug

**Problem:** The global search flow was:
1. Frontend calls `POST /index_global` → was blocked by `require_admin()` → returned 403
2. Frontend silently continued to `POST /answer_global` → no index existed → "Index not found"

**Fix:** Removed admin requirement from `/index_global`. Indexing reads already-uploaded chunks — it's not a write operation and shouldn't be admin-gated.

---

### 4. Added Auth0 authentication with RBAC

**Why:** Portfolio project accessible to recruiters — needed a way to demonstrate role separation without exposing admin controls.

**How:**
- Auth0 SPA application + API (`https://rag-assistant-api` audience)
- JWT validation in FastAPI using `python-jose` and Auth0's JWKS endpoint (RS256)
- Roles (`admin`, `employee`) injected into JWT via a Post-Login Action:
  ```javascript
  api.accessToken.setCustomClaim(`${namespace}/roles`, event.authorization.roles);
  ```
- Backend enforces roles via `require_admin()` dependency on upload/delete endpoints

**Problems encountered:**
- Auth0 "Unknown client" error — the original application was corrupted. Fixed by deleting and recreating the SPA application with a new Client ID.
- "not authorized to access resource server" — new SPA app wasn't linked to the API. Fixed by going to Application → APIs → Authorize.
- Vite bakes `import.meta.env.*` at **build time**, not runtime. Had to add `ARG` / `ENV` build args to the frontend Dockerfile so Auth0 config is embedded during `npm run build`.

---

### 5. Added LangGraph Agentic RAG

**Why:** Aligns with AI/Automation Data Scientist JD requirements. Demonstrates agentic reasoning on top of a RAG pipeline.

**Architecture — 3-node LangGraph:**

```
User question
     │
     ▼
┌─────────────────┐
│ route + rewrite │  LLM decides: single-doc or global search?
│                 │  Rewrites query for better FAISS retrieval
└────────┬────────┘  e.g. "what is this about" →
         │                "document scope topics and purpose"
         ▼
┌─────────────────┐
│    retrieve     │  FAISS similarity search using rewritten query
└────────┬────────┘  Returns top-k chunks with relevance scores
         │
         ▼
┌─────────────────┐
│    generate     │  GPT-4o answers the ORIGINAL user question
└─────────────────┘  using retrieved context + [S1][S2] citations
```

**Problems encountered:**

**Bug 1 — "Fallback routing used" every time**
GPT-4o returned the routing JSON wrapped in markdown fences (` ```json ... ``` `). `json.loads()` threw immediately and fell back. Fixed by stripping fences before parsing:
```python
if raw.startswith("```"):
    raw = raw.split("```", 2)[1]
    if raw.startswith("json"):
        raw = raw[4:]
    raw = raw.rsplit("```", 1)[0].strip()
```

**Bug 2 — Agent always returned "I don't know" despite finding relevant chunks**
`_node_generate` was passing `state["rewritten_query"]` to the LLM as the question. The rewritten query (e.g., "COMPSCI 4SD3 midterm exam structure content February 2025") was too specific — GPT-4o couldn't find a verbatim answer in the chunks and refused. Fix: pass `state["question"]` (the original user question) to generation. The rewritten query is only for FAISS retrieval.

**Bug 3 — Employee users always got "I don't know"**
The LLM prompt said `"Answer ONLY using the provided sources"` + `"If the answer is not in the sources, say I don't know"`. For broad questions like "what is this pdf about", the embedding model returns loosely-matched chunks with negative relevance scores. GPT-4o interpreted the strict prompt as "refuse to synthesize" and returned "I don't know" even though the context contained the answer.

Fixed by softening the prompt to allow synthesis:
```
"Synthesize relevant information across sources — you do not need a verbatim quote to answer."
"Only say I don't know if the sources contain NO information relevant to the question whatsoever."
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Frontend (React)                    │
│  Auth0 login → JWT → Bearer token on every request   │
│  3 modes: Single-doc │ Global (all PDFs) │ Agent      │
│  Agent Mode UI: shows routing decision, rewritten     │
│  query, and LLM reasoning in an "Agent Trace" panel  │
└─────────────────────┬────────────────────────────────┘
                      │ nginx reverse proxy
┌─────────────────────▼────────────────────────────────┐
│                  Backend (FastAPI)                    │
│                                                       │
│  JWT validation via Auth0 JWKS (RS256)                │
│  Admin: upload, delete  │  Employee: read/query only  │
│                                                       │
│  Upload  → PyPDFLoader → RecursiveCharacterTextSplit  │
│          → chunks saved as JSON + FAISS index         │
│                                                       │
│  Answer  → FAISS search → PromptTemplate | GPT-4o     │
│  Agent   → LangGraph: route → retrieve → generate    │
└──────────────────────────────────────────────────────┘
         Persisted via Docker named volume (backend_storage)
```

---

## Live Demo:
https://enterprise-rag-assistant-1.onrender.com

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, TailwindCSS |
| Auth | Auth0 (SPA + API), JWT RS256, RBAC via Post-Login Action |
| Backend | FastAPI, Python 3.11, Uvicorn |
| RAG Pipeline | LangChain: PyPDFLoader, RecursiveCharacterTextSplitter, FAISS, ChatOpenAI |
| Agent | LangGraph: StateGraph, TypedDict state, 3 nodes |
| Embeddings | fastembed ONNX — `all-MiniLM-L6-v2` (~50MB, no PyTorch) |
| Vector Store | FAISS (IndexFlatL2, saved to Docker named volume) |
| LLM | OpenAI GPT-4o |
| Deployment | Docker Compose, multi-stage builds, nginx, Render |

---

## Local Development

### Prerequisites
- Docker Desktop
- OpenAI API key
- Auth0 account (free tier)

### Setup

```bash
git clone https://github.com/MannHarjot/Enterprise_RAG_Assistant
cd Enterprise_RAG_Assistant
```

Create `backend/.env`:
```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_AUDIENCE=https://rag-assistant-api
CORS_ORIGINS=http://localhost:3000
```

Update `frontend/.env` and `docker-compose.yml` with your Auth0 Client ID.

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000/docs

### Auth0 Setup

1. Create a **Single Page Application** — set Allowed Callback/Logout/Web Origins to `http://localhost:3000`
2. Create an **API** with identifier `https://rag-assistant-api`, enable RBAC + Add Permissions in Access Token
3. Create roles: `admin`, `employee`. Create demo users, assign roles
4. Add a **Post-Login Action** (Flows → Login):
```javascript
exports.onExecutePostLogin = async (event, api) => {
  const ns = 'https://rag-assistant-api';
  if (event.authorization) {
    api.accessToken.setCustomClaim(`${ns}/roles`, event.authorization.roles);
  }
};
```

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/documents` | Any | List uploaded PDFs |
| POST | `/upload` | Admin | Upload and chunk a PDF |
| POST | `/index/{stem}` | Any | Build FAISS index for a document |
| POST | `/index_global` | Any | Build global FAISS index (all docs) |
| POST | `/answer` | Any | Answer from single-doc index |
| POST | `/answer_global` | Any | Answer from global index |
| POST | `/agent/answer` | Any | LangGraph agentic RAG answer |
| DELETE | `/documents/{name}` | Admin | Delete document + its index |
| DELETE | `/documents` | Admin | Delete all documents |
| GET | `/admin/stats` | Admin | Vector count stats |

---

## Deployment (Render)

1. Push repo to GitHub
2. Create **Web Service** for `./backend` — add all env vars from `backend/.env`, set `CORS_ORIGINS=https://your-frontend.onrender.com`
3. Create **Static Site** or **Web Service** for `./frontend` — pass `VITE_AUTH0_DOMAIN`, `VITE_AUTH0_CLIENT_ID`, `VITE_AUTH0_AUDIENCE` as build environment variables
4. Add production URLs to Auth0 app's Allowed Callback URLs, Logout URLs, and Web Origins
