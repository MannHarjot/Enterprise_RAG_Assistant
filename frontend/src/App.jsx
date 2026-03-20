import { useAuth0 } from "@auth0/auth0-react";
import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE        = import.meta.env.VITE_API_BASE_URL || "/api";
const ROLES_CLAIM     = "https://rag-assistant-api/roles";
const REQUEST_TIMEOUT = 120000; // 2 min — indexing can be slow

// ── Fetch helpers ─────────────────────────────────────────────────────────────
async function fetchWithTimeout(url, options = {}, ms = REQUEST_TIMEOUT) {
  const ctrl = new AbortController();
  const tid  = setTimeout(() => ctrl.abort(), ms);
  try {
    return await fetch(url, { ...options, signal: ctrl.signal });
  } catch (err) {
    if (err.name === "AbortError") throw new Error("Request timed out. Please try again.");
    throw err;
  } finally {
    clearTimeout(tid);
  }
}

async function apiFetch(url, options = {}, token, ms = REQUEST_TIMEOUT) {
  return fetchWithTimeout(
    url,
    { ...options, headers: { ...(options.headers || {}), Authorization: `Bearer ${token}` } },
    ms
  );
}

// ── Login screen ──────────────────────────────────────────────────────────────
function LoginScreen() {
  const { loginWithRedirect, isLoading } = useAuth0();

  const loginAs = (hint) =>
    loginWithRedirect({ authorizationParams: { login_hint: hint } });

  return (
    <div className="flex min-h-screen items-center justify-center bg-[radial-gradient(circle_at_top_left,_#1e293b_0%,_#020617_45%,_#020617_100%)] px-4">
      <div className="w-full max-w-md space-y-6">
        <div className="rounded-3xl border border-white/10 bg-white/5 p-8 shadow-2xl backdrop-blur">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-300">
            Enterprise RAG Assistant
          </p>
          <h1 className="mt-2 text-2xl font-semibold text-white">Sign in to continue</h1>
          <p className="mt-2 text-sm text-slate-400">
            Upload PDFs and ask questions with AI-powered answers and citations.
          </p>

          <button
            onClick={() => loginWithRedirect()}
            disabled={isLoading}
            className="mt-6 inline-flex w-full items-center justify-center rounded-xl border border-cyan-300/30 bg-cyan-500 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:opacity-50"
          >
            {isLoading ? "Loading…" : "Sign in with Auth0"}
          </button>
        </div>

        {/* Demo credentials */}
        <div className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur">
          <p className="mb-4 text-xs font-semibold uppercase tracking-widest text-slate-400">
            Demo Credentials
          </p>
          <div className="space-y-3">
            <DemoCard
              role="Admin"
              email="admin@demo.com"
              password="Admin@123"
              description="Upload PDFs, delete documents, ask questions"
              color="cyan"
              onLogin={() => loginAs("admin@demo.com")}
            />
            <DemoCard
              role="Employee"
              email="employee@demo.com"
              password="Employee@123"
              description="Ask questions only — read-only access"
              color="violet"
              onLogin={() => loginAs("employee@demo.com")}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function DemoCard({ role, email, password, description, color, onLogin }) {
  const accent = color === "cyan" ? "text-cyan-300 border-cyan-300/30 bg-cyan-500" : "text-violet-300 border-violet-300/30 bg-violet-500";
  const badge  = color === "cyan" ? "bg-cyan-400/10 text-cyan-300" : "bg-violet-400/10 text-violet-300";

  return (
    <div className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold ${badge}`}>
            {role}
          </span>
          <p className="mt-2 text-sm text-slate-300">{description}</p>
          <div className="mt-2 space-y-1 font-mono text-xs text-slate-400">
            <div><span className="text-slate-500">email </span>{email}</div>
            <div><span className="text-slate-500">pass  </span>{password}</div>
          </div>
        </div>
        <button
          onClick={onLogin}
          className={`shrink-0 rounded-xl border px-3 py-2 text-xs font-semibold text-slate-950 transition hover:opacity-90 ${accent}`}
        >
          Login
        </button>
      </div>
    </div>
  );
}

// ── Main app ──────────────────────────────────────────────────────────────────
export default function App() {
  const { isAuthenticated, isLoading, user, logout, getAccessTokenSilently } = useAuth0();

  const roles   = user?.[ROLES_CLAIM] ?? [];
  const isAdmin = roles.includes("admin");

  const fileInputRef  = useRef(null);
  const [documents,   setDocuments]   = useState([]);
  const [selectedDoc, setSelectedDoc] = useState("");
  const [question,    setQuestion]    = useState("");
  const [result,      setResult]      = useState(null);
  const [busy,        setBusy]        = useState(false);
  const [error,       setError]       = useState("");
  const [globalMode,  setGlobalMode]  = useState(false);
  const [agentMode,   setAgentMode]   = useState(false);
  const [file,        setFile]        = useState(null);

  const selectedStem = useMemo(
    () => (selectedDoc.endsWith(".pdf") ? selectedDoc.slice(0, -4) : selectedDoc),
    [selectedDoc]
  );

  const token = async () => getAccessTokenSilently();

  async function refreshDocs() {
    try {
      const t   = await token();
      const res = await apiFetch(`${API_BASE}/documents`, {}, t, 15000);
      const data = await res.json();
      const docs = data.documents || [];
      setDocuments(docs);
      if (docs.length && (!selectedDoc || !docs.includes(selectedDoc))) {
        setSelectedDoc(docs[0]);
      }
      if (!docs.length) setSelectedDoc("");
    } catch {
      // silent — user will see no docs listed
    }
  }

  useEffect(() => {
    if (isAuthenticated) refreshDocs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated]);

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#020617]">
        <div className="text-sm text-slate-400">Loading…</div>
      </div>
    );
  }

  if (!isAuthenticated) return <LoginScreen />;

  // ── Handlers ────────────────────────────────────────────────────────────────
  async function handleUpload(e) {
    e.preventDefault();
    setError(""); setResult(null);
    if (!file) { setError("Please choose a PDF first."); return; }
    setBusy(true);
    try {
      const t    = await token();
      const form = new FormData();
      form.append("file", file);
      const res  = await apiFetch(`${API_BASE}/upload`, { method: "POST", body: form }, t);
      if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail || `Upload failed (${res.status})`); }
      const data = await res.json();
      await refreshDocs();
      setSelectedDoc(data.filename);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setError(err.message || "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleAsk(e) {
    e.preventDefault();
    setError(""); setResult(null);
    if (!question.trim()) { setError("Type a question first."); return; }
    if (!agentMode && !globalMode && !selectedStem) { setError("No document selected."); return; }
    setBusy(true);
    try {
      const t = await token();

      // ── Agent mode ────────────────────────────────────────────────────────
      if (agentMode) {
        // Pre-build indices so the agent can use either single or global
        const indexPromises = [apiFetch(`${API_BASE}/index_global`, { method: "POST" }, t)];
        if (selectedStem) {
          indexPromises.push(
            apiFetch(`${API_BASE}/index/${encodeURIComponent(selectedStem)}`, { method: "POST" }, t)
          );
        }
        const indexResults = await Promise.all(indexPromises);
        for (const r of indexResults) {
          if (!r.ok) {
            const d = await r.json().catch(() => ({}));
            throw new Error(d.detail || `Index build failed (${r.status})`);
          }
        }

        const ansRes = await apiFetch(
          `${API_BASE}/agent/answer`,
          { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, pdf_stem: selectedStem || null }) },
          t
        );
        const data = await ansRes.json();
        if (!ansRes.ok) throw new Error(data.detail || `Agent request failed (${ansRes.status})`);
        setResult(data);
        return;
      }

      // ── Single-doc mode ───────────────────────────────────────────────────
      if (!globalMode) {
        const idxRes = await apiFetch(
          `${API_BASE}/index/${encodeURIComponent(selectedStem)}`,
          { method: "POST" },
          t
        );
        if (!idxRes.ok) {
          const d = await idxRes.json().catch(() => ({}));
          throw new Error(d.detail || `Indexing failed (${idxRes.status})`);
        }
        const ansRes = await apiFetch(
          `${API_BASE}/answer`,
          { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ pdf_stem: selectedStem, question, top_k: 4 }) },
          t
        );
        const data = await ansRes.json();
        if (!ansRes.ok) throw new Error(data.detail || `Request failed (${ansRes.status})`);
        setResult(data);
        return;
      }

      // ── Global mode ───────────────────────────────────────────────────────
      const idxRes = await apiFetch(`${API_BASE}/index_global`, { method: "POST" }, t);
      if (!idxRes.ok) {
        const d = await idxRes.json().catch(() => ({}));
        throw new Error(d.detail || `Global indexing failed (${idxRes.status})`);
      }
      const ansRes = await apiFetch(
        `${API_BASE}/answer_global`,
        { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, top_k: 6 }) },
        t
      );
      const data = await ansRes.json();
      if (!ansRes.ok) throw new Error(data.detail || `Request failed (${ansRes.status})`);
      setResult(data);
    } catch (err) {
      setError(err.message || "Ask failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleDeleteDoc(docName) {
    if (!window.confirm(`Delete "${docName}" and its index?`)) return;
    setError(""); setResult(null); setBusy(true);
    try {
      const t   = await token();
      const res = await apiFetch(`${API_BASE}/documents/${encodeURIComponent(docName)}`, { method: "DELETE" }, t);
      const d   = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(d.detail || `Delete failed (${res.status})`);
      if (selectedDoc === docName) setSelectedDoc("");
      await refreshDocs();
    } catch (err) {
      setError(err.message || "Delete failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleDeleteAll() {
    if (!documents.length) return;
    if (!window.confirm(`Delete all ${documents.length} PDF(s)?`)) return;
    setError(""); setResult(null); setBusy(true);
    try {
      const t   = await token();
      const res = await apiFetch(`${API_BASE}/documents`, { method: "DELETE" }, t);
      const d   = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(d.detail || `Delete all failed (${res.status})`);
      setSelectedDoc(""); setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      await refreshDocs();
    } catch (err) {
      setError(err.message || "Delete all failed.");
    } finally {
      setBusy(false);
    }
  }

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_#1e293b_0%,_#020617_45%,_#020617_100%)] px-4 py-8 text-slate-100 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl">

        {/* Header */}
        <header className="mb-6 rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl shadow-black/30 backdrop-blur md:p-8">
          <div className="flex flex-col gap-5 md:flex-row md:items-start md:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-300">
                Enterprise RAG Assistant
              </p>
              <h1 className="mt-2 text-2xl font-semibold tracking-tight text-white sm:text-4xl">
                Search your PDFs with grounded answers
              </h1>
              <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300 sm:text-base">
                Upload PDFs, pick a source, and ask questions with citation snippets and scoring details.
              </p>
            </div>
            <div className="flex flex-col gap-3 sm:min-w-72">
              <div className="grid grid-cols-2 gap-3">
                <StatCard label="Uploaded PDFs"  value={String(documents.length)} />
                <StatCard label="Search Mode"    value={agentMode ? "Agent" : globalMode ? "Global" : "Single"} />
              </div>
              {/* User info + logout */}
              <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-2">
                <div className="min-w-0">
                  <p className="truncate text-xs text-slate-300">{user?.email}</p>
                  <RoleBadge roles={roles} />
                </div>
                <button
                  onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })}
                  className="ml-3 shrink-0 rounded-lg border border-white/15 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:bg-white/10"
                >
                  Sign out
                </button>
              </div>
            </div>
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-[340px_minmax(0,1fr)]">
          <aside className="space-y-5">

            {/* Upload — admin only */}
            {isAdmin && (
              <section className={panelClass}>
                <div className="mb-4 flex items-center justify-between gap-3">
                  <h2 className={panelTitleClass}>Upload PDF</h2>
                  <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-slate-300">Admin</span>
                </div>
                <form onSubmit={handleUpload} className="space-y-3">
                  <input ref={fileInputRef} type="file" accept="application/pdf"
                    onChange={(e) => setFile(e.target.files?.[0] || null)} className="hidden" id="pdf-file-input" />
                  <button type="button" onClick={() => fileInputRef.current?.click()}
                    className="flex w-full items-center justify-center rounded-xl border border-dashed border-white/20 bg-slate-900/70 px-4 py-4 text-sm font-medium text-slate-100 transition hover:border-cyan-300/50 hover:bg-slate-900">
                    Choose PDF file
                  </button>
                  <div className="rounded-xl border border-white/10 bg-slate-950/70 px-3 py-2 text-sm text-slate-300">
                    {file ? file.name : "No file selected"}
                  </div>
                  <button type="submit" disabled={busy || !file} className={primaryButtonClass}>
                    {busy ? "Uploading…" : "Upload document"}
                  </button>
                </form>
              </section>
            )}

            {/* Documents */}
            <section className={panelClass}>
              <div className="mb-4 flex items-center justify-between gap-3">
                <h2 className={panelTitleClass}>Documents</h2>
                <div className="flex items-center gap-2">
                  <button onClick={refreshDocs} disabled={busy} className={secondaryButtonClass}>Refresh</button>
                  {isAdmin && (
                    <button onClick={handleDeleteAll} disabled={busy || !documents.length} className={dangerButtonClass}>
                      Delete all
                    </button>
                  )}
                </div>
              </div>

              <div className="mb-3 rounded-xl border border-white/10 bg-slate-950/70 p-3">
                <label className="flex items-start gap-3 text-sm text-slate-200">
                  <input type="checkbox" checked={globalMode} onChange={(e) => setGlobalMode(e.target.checked)}
                    className="mt-0.5 h-4 w-4 rounded border-white/20 bg-slate-900 text-cyan-400 focus:ring-cyan-400" />
                  <span>
                    Search across all PDFs
                    <span className="block text-xs text-slate-400">Queries the global index.</span>
                  </span>
                </label>
              </div>

              {documents.length === 0 ? (
                <div className="rounded-xl border border-white/10 bg-white/5 px-4 py-6 text-center text-sm text-slate-400">
                  {isAdmin ? "No PDFs uploaded yet." : "No PDFs available yet."}
                </div>
              ) : (
                <ul className="max-h-80 space-y-2 overflow-auto pr-1">
                  {documents.map((doc) => {
                    const isActive = doc === selectedDoc;
                    return (
                      <li key={doc}>
                        <div className={`rounded-xl border p-2 transition ${isActive ? "border-cyan-300/60 bg-cyan-400/10" : "border-white/10 bg-slate-950/60"}`}>
                          <div className="flex items-start gap-2">
                            <button type="button" onClick={() => setSelectedDoc(doc)} disabled={globalMode}
                              className={`min-w-0 flex-1 rounded-lg px-2 py-1.5 text-left transition ${globalMode ? "cursor-not-allowed opacity-50" : "hover:bg-white/5"}`}>
                              <div className="truncate text-sm font-medium text-white">{doc}</div>
                              <div className="mt-1 text-xs text-slate-400">{isActive ? "Selected" : "Click to select"}</div>
                            </button>
                            {isAdmin && (
                              <button type="button" onClick={(e) => { e.stopPropagation(); handleDeleteDoc(doc); }}
                                disabled={busy} className={dangerIconButtonClass}>
                                Delete
                              </button>
                            )}
                          </div>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
            </section>
          </aside>

          <main className="space-y-5">
            {/* Ask */}
            <section className={panelClass}>
              <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h2 className={panelTitleClass}>Ask a Question</h2>
                  <p className="text-sm text-slate-400">
                    {agentMode
                      ? "Agent mode — LLM routes query and rewrites for optimal retrieval."
                      : globalMode
                        ? "Global mode — searching across all uploaded PDFs."
                        : selectedDoc ? `Selected: ${selectedDoc}` : "Select a document or enable global mode."}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setAgentMode((v) => !v)}
                    className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold transition ${
                      agentMode
                        ? "border-violet-400/40 bg-violet-500/20 text-violet-300"
                        : "border-white/10 bg-white/5 text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    <span className={`h-1.5 w-1.5 rounded-full ${agentMode ? "bg-violet-400" : "bg-slate-500"}`} />
                    {agentMode ? "Agent ON" : "Agent OFF"}
                  </button>
                  <span className="inline-flex items-center rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-slate-300">
                    {busy ? "Processing…" : "Ready"}
                  </span>
                </div>
              </div>

              <form onSubmit={handleAsk} className="space-y-4">
                <label className="block">
                  <span className="mb-2 block text-sm font-medium text-slate-200">Question</span>
                  <textarea value={question} onChange={(e) => setQuestion(e.target.value)} rows={5}
                    placeholder={globalMode ? "Ask across all PDFs…" : "Ask about the selected PDF…"}
                    className="w-full rounded-2xl border border-white/10 bg-slate-950/80 px-4 py-3 text-sm leading-6 text-slate-100 placeholder:text-slate-500 focus:border-cyan-400 focus:outline-none" />
                </label>
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <p className="text-xs text-slate-400">Answers include citations: filename, page, chunk, and similarity score.</p>
                  <button type="submit" disabled={busy || documents.length === 0} className={`${primaryButtonClass} sm:w-auto`}>
                    {busy ? "Generating answer…" : "Ask question"}
                  </button>
                </div>
              </form>
            </section>

            {/* Error */}
            {error && (
              <section className="rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100 shadow-lg">
                <div className="font-semibold">Error</div>
                <div className="mt-1">{error}</div>
              </section>
            )}

            {/* Answer */}
            <section className={panelClass}>
              <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h2 className={panelTitleClass}>Answer</h2>
                  <p className="text-sm text-slate-400">Response and source evidence from the retrieval pipeline.</p>
                </div>
                {result && (
                  <div className="text-xs text-slate-400">
                    mode: <span className="font-semibold text-slate-200">{result.mode || "unknown"}</span>
                    <span className="mx-2">•</span>
                    id: <code className="rounded bg-black/30 px-1.5 py-0.5 text-slate-200">{result.request_id || "n/a"}</code>
                  </div>
                )}
              </div>

              {!result ? (
                <div className="rounded-2xl border border-dashed border-white/15 bg-slate-950/50 px-4 py-10 text-center text-sm text-slate-400">
                  Submit a question to see the answer and citations here.
                </div>
              ) : (
                <div className="space-y-5">
                  {/* Agent reasoning panel */}
                  {result.mode === "agent" && (
                    <div className="rounded-2xl border border-violet-400/20 bg-violet-500/10 p-4 text-sm sm:p-5">
                      <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-violet-300">Agent Trace</div>
                      <div className="space-y-2">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-xs font-medium text-slate-400">Search mode</span>
                          <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${result.search_mode === "single" ? "bg-cyan-400/15 text-cyan-300" : "bg-violet-400/15 text-violet-300"}`}>
                            {result.search_mode}
                          </span>
                        </div>
                        {result.rewritten_query && result.rewritten_query !== question && (
                          <div>
                            <span className="text-xs font-medium text-slate-400">Rewritten query </span>
                            <span className="text-xs text-slate-200 italic">"{result.rewritten_query}"</span>
                          </div>
                        )}
                        {result.reasoning && (
                          <div>
                            <span className="text-xs font-medium text-slate-400">Reasoning </span>
                            <span className="text-xs text-slate-300">{result.reasoning}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  <div className="rounded-2xl border border-white/10 bg-slate-950/70 p-4 text-sm leading-7 text-slate-100 sm:p-5">
                    <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-cyan-300">Answer</div>
                    <div className="whitespace-pre-wrap">{result.answer}</div>
                  </div>
                  <div>
                    <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-300">
                      Citations ({result.citations?.length || 0})
                    </h3>
                    {result.citations?.length ? (
                      <ul className="space-y-3">
                        {result.citations.map((c) => (
                          <li key={c.source_id} className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
                            <div className="flex flex-wrap items-center gap-2 text-sm text-slate-200">
                              <span className="rounded-md bg-cyan-400/10 px-2 py-0.5 font-semibold text-cyan-300">[{c.source_id}]</span>
                              <span className="font-medium">{c.filename}</span>
                              <span className="text-slate-400">page {c.page}</span>
                              <span className="text-slate-400">chunk {c.chunk_id}</span>
                              <span className="text-slate-400">score {typeof c.score === "number" ? c.score.toFixed(3) : c.score}</span>
                            </div>
                            <p className="mt-3 text-sm leading-6 text-slate-300">{c.snippet}</p>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <div className="rounded-xl border border-white/10 bg-white/5 px-4 py-4 text-sm text-slate-400">No citations returned.</div>
                    )}
                  </div>
                </div>
              )}
            </section>
          </main>
        </div>
      </div>
    </div>
  );
}

// ── Small components ──────────────────────────────────────────────────────────
function StatCard({ label, value }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 truncate text-lg font-semibold text-white">{value}</div>
    </div>
  );
}

function RoleBadge({ roles }) {
  if (!roles?.length) return null;
  const isAdmin = roles.includes("admin");
  return (
    <span className={`mt-0.5 inline-block rounded-full px-2 py-0.5 text-[10px] font-semibold ${isAdmin ? "bg-cyan-400/10 text-cyan-300" : "bg-violet-400/10 text-violet-300"}`}>
      {isAdmin ? "admin" : "employee"}
    </span>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const panelClass          = "rounded-3xl border border-white/10 bg-white/5 p-5 shadow-xl shadow-black/20 backdrop-blur md:p-6";
const panelTitleClass     = "text-lg font-semibold tracking-tight text-white";
const primaryButtonClass  = "inline-flex w-full items-center justify-center rounded-xl border border-cyan-300/30 bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50";
const secondaryButtonClass = "inline-flex items-center justify-center rounded-xl border border-white/15 bg-white/5 px-3 py-2 text-sm font-medium text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50";
const dangerButtonClass   = "inline-flex items-center justify-center rounded-xl border border-rose-300/20 bg-rose-500/10 px-3 py-2 text-sm font-medium text-rose-200 transition hover:bg-rose-500/20 disabled:cursor-not-allowed disabled:opacity-50";
const dangerIconButtonClass = "shrink-0 rounded-lg border border-rose-300/20 bg-rose-500/10 px-2.5 py-1.5 text-xs font-medium text-rose-200 transition hover:bg-rose-500/20 disabled:cursor-not-allowed disabled:opacity-50";
