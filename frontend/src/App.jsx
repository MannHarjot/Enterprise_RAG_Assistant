import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

export default function App() {
  const fileInputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState("");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [globalMode, setGlobalMode] = useState(false);

  const selectedStem = useMemo(() => {
    if (!selectedDoc) return "";
    return selectedDoc.endsWith(".pdf") ? selectedDoc.slice(0, -4) : selectedDoc;
  }, [selectedDoc]);

  async function refreshDocs() {
    const res = await fetch(`${API_BASE}/documents`);
    const data = await res.json();
    const nextDocs = data.documents || [];

    setDocuments(nextDocs);

    if (!nextDocs.length) {
      setSelectedDoc("");
      return;
    }

    if (!selectedDoc || !nextDocs.includes(selectedDoc)) {
      setSelectedDoc(nextDocs[0]);
    }
  }

  useEffect(() => {
    refreshDocs().catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleUpload(e) {
    e.preventDefault();
    setError("");
    setResult(null);

    if (!file) {
      setError("Please choose a PDF first.");
      return;
    }

    setBusy(true);
    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `Upload failed (${res.status})`);
      }

      const data = await res.json();
      await refreshDocs();
      setSelectedDoc(data.filename);
      setFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      setError(err.message || "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleAsk(e) {
    e.preventDefault();
    setError("");
    setResult(null);

    if (!question.trim()) {
      setError("Type a question first.");
      return;
    }

    if (!globalMode && !selectedStem) {
      setError("No document selected.");
      return;
    }

    setBusy(true);
    try {
      if (!globalMode) {
        await fetch(`${API_BASE}/index/${encodeURIComponent(selectedStem)}`, {
          method: "POST",
        });

        const res = await fetch(`${API_BASE}/answer`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pdf_stem: selectedStem,
            question,
            top_k: 4,
          }),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);

        setResult(data);
        return;
      }

      await fetch(`${API_BASE}/index_global`, { method: "POST" });

      const res = await fetch(`${API_BASE}/answer_global`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          top_k: 6,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);

      setResult(data);
    } catch (err) {
      setError(err.message || "Ask failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleDeleteDocument(docName) {
    const confirmed = window.confirm(`Delete "${docName}" and its extracted/indexed data?`);
    if (!confirmed) return;

    setError("");
    setResult(null);
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(docName)}`, {
        method: "DELETE",
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `Delete failed (${res.status})`);

      if (selectedDoc === docName) {
        setSelectedDoc("");
      }
      await refreshDocs();
    } catch (err) {
      setError(err.message || "Delete failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleDeleteAllDocuments() {
    if (!documents.length) return;

    const confirmed = window.confirm(
      `Delete all ${documents.length} uploaded PDF(s) and generated data?`
    );
    if (!confirmed) return;

    setError("");
    setResult(null);
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/documents`, { method: "DELETE" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `Delete all failed (${res.status})`);

      setSelectedDoc("");
      setFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      await refreshDocs();
    } catch (err) {
      setError(err.message || "Delete all failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_#1e293b_0%,_#020617_45%,_#020617_100%)] px-4 py-8 text-slate-100 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl">
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
                Upload one or more PDF documents, pick a source, and ask questions with citation
                snippets and scoring details.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3 sm:min-w-72">
              <StatCard label="Uploaded PDFs" value={String(documents.length)} />
              <StatCard label="Search Mode" value={globalMode ? "Global" : "Single"} />
            </div>
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-[340px_minmax(0,1fr)]">
          <aside className="space-y-5">
            <section className={panelClass}>
              <div className="mb-4 flex items-center justify-between gap-3">
                <h2 className={panelTitleClass}>Upload PDF</h2>
                <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-slate-300">
                  Step 1
                </span>
              </div>

              <form onSubmit={handleUpload} className="space-y-3">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="application/pdf"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                  className="hidden"
                  id="pdf-file-input"
                />

                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="flex w-full items-center justify-center rounded-xl border border-dashed border-white/20 bg-slate-900/70 px-4 py-4 text-sm font-medium text-slate-100 transition hover:border-cyan-300/50 hover:bg-slate-900"
                >
                  Choose PDF file
                </button>

                <div className="rounded-xl border border-white/10 bg-slate-950/70 px-3 py-2 text-sm text-slate-300">
                  {file ? file.name : "No file selected"}
                </div>

                <button type="submit" disabled={busy || !file} className={primaryButtonClass}>
                  {busy ? "Uploading..." : "Upload document"}
                </button>
              </form>
            </section>

            <section className={panelClass}>
              <div className="mb-4 flex items-center justify-between gap-3">
                <h2 className={panelTitleClass}>Documents</h2>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={refreshDocs}
                    disabled={busy}
                    className={secondaryButtonClass}
                  >
                    Refresh
                  </button>
                  <button
                    type="button"
                    onClick={handleDeleteAllDocuments}
                    disabled={busy || documents.length === 0}
                    className={dangerButtonClass}
                    title="Delete all uploaded PDFs and generated files"
                  >
                    Delete all
                  </button>
                </div>
              </div>

              <div className="mb-3 rounded-xl border border-white/10 bg-slate-950/70 p-3">
                <label className="flex items-start gap-3 text-sm text-slate-200">
                  <input
                    type="checkbox"
                    checked={globalMode}
                    onChange={(e) => setGlobalMode(e.target.checked)}
                    className="mt-0.5 h-4 w-4 rounded border-white/20 bg-slate-900 text-cyan-400 focus:ring-cyan-400"
                  />
                  <span>
                    Search across all PDFs
                    <span className="block text-xs text-slate-400">
                      Disables per-document selection and queries the global index.
                    </span>
                  </span>
                </label>
              </div>

              {documents.length === 0 ? (
                <div className="rounded-xl border border-white/10 bg-white/5 px-4 py-6 text-center text-sm text-slate-400">
                  No PDFs uploaded yet.
                </div>
              ) : (
                <ul className="max-h-80 space-y-2 overflow-auto pr-1">
                  {documents.map((doc) => {
                    const isActive = doc === selectedDoc;
                    return (
                      <li key={doc}>
                        <div
                          className={`rounded-xl border p-2 transition ${
                            isActive
                              ? "border-cyan-300/60 bg-cyan-400/10"
                              : "border-white/10 bg-slate-950/60"
                          }`}
                        >
                          <div className="flex items-start gap-2">
                            <button
                              type="button"
                              onClick={() => setSelectedDoc(doc)}
                              disabled={globalMode}
                              className={`min-w-0 flex-1 rounded-lg px-2 py-1.5 text-left transition ${
                                globalMode
                                  ? "cursor-not-allowed opacity-50"
                                  : "hover:bg-white/5"
                              }`}
                            >
                              <div className="truncate text-sm font-medium text-white">{doc}</div>
                              <div className="mt-1 text-xs text-slate-400">
                                {isActive ? "Selected" : "Click to select"}
                              </div>
                            </button>

                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteDocument(doc);
                              }}
                              disabled={busy}
                              className={dangerIconButtonClass}
                              title={`Delete ${doc}`}
                              aria-label={`Delete ${doc}`}
                            >
                              Delete
                            </button>
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
            <section className={panelClass}>
              <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h2 className={panelTitleClass}>Ask a Question</h2>
                  <p className="text-sm text-slate-400">
                    {globalMode
                      ? "Global mode is active. Your question will search across all uploaded PDFs."
                      : selectedDoc
                        ? `Selected document: ${selectedDoc}`
                        : "Select a document or enable global mode before asking a question."}
                  </p>
                </div>
                <span className="inline-flex items-center rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-slate-300">
                  {busy ? "Processing request..." : "Ready"}
                </span>
              </div>

              <form onSubmit={handleAsk} className="space-y-4">
                <label className="block">
                  <span className="mb-2 block text-sm font-medium text-slate-200">Question</span>
                  <textarea
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    rows={5}
                    placeholder={
                      globalMode
                        ? "Ask across all uploaded PDFs. Example: Summarize the compliance requirements mentioned in all documents."
                        : "Ask about the selected PDF. Example: What are the key terms and dates in this document?"
                    }
                    className="w-full rounded-2xl border border-white/10 bg-slate-950/80 px-4 py-3 text-sm leading-6 text-slate-100 placeholder:text-slate-500 focus:border-cyan-400 focus:outline-none"
                  />
                </label>

                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <p className="text-xs text-slate-400">
                    Answers include citations with filename, page number, chunk, and similarity score.
                  </p>
                  <button
                    type="submit"
                    disabled={busy || (!globalMode && documents.length === 0)}
                    className={`${primaryButtonClass} sm:w-auto`}
                  >
                    {busy ? "Generating answer..." : "Ask question"}
                  </button>
                </div>
              </form>
            </section>

            {error ? (
              <section className="rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100 shadow-lg shadow-black/10">
                <div className="font-semibold">Request error</div>
                <div className="mt-1">{error}</div>
              </section>
            ) : null}

            <section className={panelClass}>
              <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h2 className={panelTitleClass}>Answer</h2>
                  <p className="text-sm text-slate-400">
                    Response text and source evidence from the retrieval pipeline.
                  </p>
                </div>
                {result ? (
                  <div className="text-xs text-slate-400">
                    mode: <span className="font-semibold text-slate-200">{result.mode || "unknown"}</span>
                    <span className="mx-2">•</span>
                    request_id:{" "}
                    <code className="rounded bg-black/30 px-1.5 py-0.5 text-slate-200">
                      {result.request_id || "n/a"}
                    </code>
                  </div>
                ) : null}
              </div>

              {!result ? (
                <div className="rounded-2xl border border-dashed border-white/15 bg-slate-950/50 px-4 py-10 text-center text-sm text-slate-400">
                  Submit a question to see the generated answer and citations here.
                </div>
              ) : (
                <div className="space-y-5">
                  <div className="rounded-2xl border border-white/10 bg-slate-950/70 p-4 text-sm leading-7 text-slate-100 sm:p-5">
                    <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-cyan-300">
                      Answer Text
                    </div>
                    <div className="whitespace-pre-wrap">{result.answer}</div>
                  </div>

                  <div>
                    <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-300">
                      Citations ({result.citations?.length || 0})
                    </h3>
                    {result.citations?.length ? (
                      <ul className="space-y-3">
                        {result.citations.map((c) => (
                          <li
                            key={c.source_id}
                            className="rounded-2xl border border-white/10 bg-slate-950/60 p-4"
                          >
                            <div className="flex flex-wrap items-center gap-2 text-sm text-slate-200">
                              <span className="rounded-md bg-cyan-400/10 px-2 py-0.5 font-semibold text-cyan-300">
                                [{c.source_id}]
                              </span>
                              <span className="font-medium">{c.filename}</span>
                              <span className="text-slate-400">page {c.page}</span>
                              <span className="text-slate-400">chunk {c.chunk_id}</span>
                              <span className="text-slate-400">
                                score {typeof c.score === "number" ? c.score.toFixed(3) : c.score}
                              </span>
                            </div>
                            <p className="mt-3 text-sm leading-6 text-slate-300">{c.snippet}</p>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <div className="rounded-xl border border-white/10 bg-white/5 px-4 py-4 text-sm text-slate-400">
                        No citations returned.
                      </div>
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

function StatCard({ label, value }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 truncate text-lg font-semibold text-white">{value}</div>
    </div>
  );
}

const panelClass =
  "rounded-3xl border border-white/10 bg-white/5 p-5 shadow-xl shadow-black/20 backdrop-blur md:p-6";

const panelTitleClass = "text-lg font-semibold tracking-tight text-white";

const primaryButtonClass =
  "inline-flex w-full items-center justify-center rounded-xl border border-cyan-300/30 bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50";

const secondaryButtonClass =
  "inline-flex items-center justify-center rounded-xl border border-white/15 bg-white/5 px-3 py-2 text-sm font-medium text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50";

const dangerButtonClass =
  "inline-flex items-center justify-center rounded-xl border border-rose-300/20 bg-rose-500/10 px-3 py-2 text-sm font-medium text-rose-200 transition hover:bg-rose-500/20 disabled:cursor-not-allowed disabled:opacity-50";

const dangerIconButtonClass =
  "shrink-0 rounded-lg border border-rose-300/20 bg-rose-500/10 px-2.5 py-1.5 text-xs font-medium text-rose-200 transition hover:bg-rose-500/20 disabled:cursor-not-allowed disabled:opacity-50";
