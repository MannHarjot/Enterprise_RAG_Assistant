import { useEffect, useMemo, useState } from "react";

const API_BASE = "/api";

export default function App() {
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
    setDocuments(data.documents || []);
    if (!selectedDoc && (data.documents || []).length) {
      setSelectedDoc(data.documents[0]);
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

    // Only require a selected document in single-doc mode
    if (!globalMode && !selectedStem) {
      setError("No document selected.");
      return;
    }

    setBusy(true);
    try {
      if (!globalMode) {
        // Ensure per-doc index exists
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

      // Global mode: build global index then ask global
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

  return (
    <div style={{ maxWidth: 900, margin: "40px auto", fontFamily: "system-ui" }}>
      <h1 style={{ marginBottom: 8 }}>Enterprise RAG Assistant</h1>
      <p style={{ marginTop: 0, color: "#555" }}>
        Upload a PDF, then ask questions with citations.
      </p>

      <div style={{ display: "grid", gap: 16 }}>
        <section style={cardStyle}>
          <h2 style={h2}>1) Upload PDF</h2>
          <form onSubmit={handleUpload} style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <button disabled={busy} style={btnStyle}>
              {busy ? "Working..." : "Upload"}
            </button>
          </form>
        </section>

        <section style={cardStyle}>
          <h2 style={h2}>2) Ask a question</h2>

          <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <label style={{ fontWeight: 600 }}>Document:</label>
            <select
              value={selectedDoc}
              onChange={(e) => setSelectedDoc(e.target.value)}
              style={{ padding: 8, minWidth: 260 }}
              disabled={globalMode}
              title={globalMode ? "Disabled in global mode" : ""}
            >
              {documents.length === 0 ? (
                <option value="">No documents yet</option>
              ) : (
                documents.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))
              )}
            </select>

            <button disabled={busy} onClick={refreshDocs} style={btnStyleSecondary}>
              Refresh list
            </button>
          </div>

          <label style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 12 }}>
            <input
              type="checkbox"
              checked={globalMode}
              onChange={(e) => setGlobalMode(e.target.checked)}
            />
            Search across all PDFs (global)
          </label>

          <form onSubmit={handleAsk} style={{ marginTop: 12, display: "grid", gap: 10 }}>
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={globalMode ? "Ask across all PDFs..." : "Ask about selected PDF..."}
              style={{ padding: 10, fontSize: 14 }}
            />
            <button disabled={busy || (!globalMode && documents.length === 0)} style={btnStyle}>
              {busy ? "Working..." : "Ask"}
            </button>
          </form>
        </section>

        {error ? (
          <div style={{ ...cardStyle, borderColor: "#ffb3b3", background: "#fff5f5" }}>
            <strong>Error:</strong> {error}
          </div>
        ) : null}

        {result ? (
          <section style={cardStyle}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
              <h2 style={h2}>Answer</h2>
              <span style={{ color: "#666" }}>
                mode: <b>{result.mode || "unknown"}</b> • request_id:{" "}
                <code>{result.request_id || "n/a"}</code>
              </span>
            </div>

            <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5 }}>{result.answer}</div>

            <h3 style={{ marginTop: 18, marginBottom: 8 }}>Citations</h3>
            {result.citations?.length ? (
              <ul style={{ marginTop: 0 }}>
                {result.citations.map((c) => (
                  <li key={c.source_id} style={{ marginBottom: 10 }}>
                    <b>[{c.source_id}]</b> {c.filename}, page {c.page}, chunk {c.chunk_id}{" "}
                    <span style={{ color: "#666" }}>
                      (score {typeof c.score === "number" ? c.score.toFixed(3) : c.score})
                    </span>
                    <div style={{ color: "#333", marginTop: 4 }}>{c.snippet}</div>
                  </li>
                ))}
              </ul>
            ) : (
              <p style={{ color: "#666" }}>No citations returned.</p>
            )}
          </section>
        ) : null}
      </div>
    </div>
  );
}

const cardStyle = {
  border: "1px solid #e5e5e5",
  borderRadius: 12,
  padding: 16,
  boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
};

const btnStyle = {
  padding: "10px 14px",
  borderRadius: 10,
  border: "1px solid #111",
  background: "#111",
  color: "#fff",
  cursor: "pointer",
};

const btnStyleSecondary = {
  padding: "10px 14px",
  borderRadius: 10,
  border: "1px solid #ccc",
  background: "#fff",
  cursor: "pointer",
};

const h2 = { marginTop: 0, marginBottom: 10 };