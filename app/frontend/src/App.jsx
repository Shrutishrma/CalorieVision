import { useState, useEffect } from 'react'

const BACKEND_URL = 'http://localhost:8000'

export default function App() {
  const [status, setStatus] = useState(null)   // null | "ok" | "error"
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${BACKEND_URL}/health`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((data) => {
        setResponse(data)
        setStatus('ok')
      })
      .catch((err) => {
        setResponse({ error: err.message })
        setStatus('error')
      })
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="container">
      {/* ── Header ── */}
      <header className="header">
        <div className="logo">
          <span className="logo-icon">🏋️</span>
          <h1>CalorieVision</h1>
        </div>
        <p className="tagline">AI-powered exercise recognition &amp; calorie estimation</p>
      </header>

      {/* ── Connection card ── */}
      <main className="card">
        <h2 className="card-title">Backend Connection</h2>
        <p className="card-subtitle">Probing <code>{BACKEND_URL}/health</code></p>

        <div className={`status-badge ${loading ? 'loading' : status}`}>
          {loading && <span className="spinner" />}
          {!loading && (
            <span className="status-dot" />
          )}
          <span className="status-text">
            {loading
              ? 'Connecting…'
              : status === 'ok'
              ? '✓ Backend is healthy'
              : '✗ Backend unreachable'}
          </span>
        </div>

        {!loading && response && (
          <pre className="response-block">
            {JSON.stringify(response, null, 2)}
          </pre>
        )}

        {!loading && status === 'error' && (
          <p className="hint">
            Make sure the FastAPI server is running:<br />
            <code>uvicorn app.backend.main:app --reload --port 8000</code>
          </p>
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="footer">
        CalorieVision · CV Pipeline ＋ FastAPI ＋ React
      </footer>
    </div>
  )
}
