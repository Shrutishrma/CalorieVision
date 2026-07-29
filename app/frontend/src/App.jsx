/* app/frontend/src/App.jsx — CalorieVision React Dashboard */
import { useState, useEffect, useRef } from 'react'

const DEFAULT_URL = 'http://localhost:8000'
const ALT_URL = 'http://127.0.0.1:8000'

/** Format seconds as  1h 20m 15s  /  4m 30s  / 30s */
function fmtDuration(secs) {
  const s = Math.round(secs)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const rem = s % 60
  if (h > 0) return `${h}h ${m}m ${rem}s`
  if (m > 0) return `${m}m ${rem}s`
  return `${rem}s`
}

/** Segment colour palette indexed by label */
const LABEL_COLORS = {
  squat:            '#6366f1',
  pushup:           '#0ea5e9',
  jumping_jack:     '#f59e0b',
  lunge:            '#10b981',
  plank:            '#8b5cf6',
  burpee:           '#ef4444',
  mountain_climber: '#ec4899',
  high_knees:       '#14b8a6',
}
const DEFAULT_COLOR = '#64748b'
const labelColor = (label) => LABEL_COLORS[label] ?? DEFAULT_COLOR

/** Tooltip that floats near the hovered segment */
function SegmentTooltip({ seg, visible, x }) {
  if (!visible || !seg) return null
  return (
    <div className="timeline-tooltip" style={{ left: `${Math.min(x, 80)}%` }}>
      <div className="tt-label">{seg.label.replace(/_/g, ' ')}</div>
      <div className="tt-row"><span>⏱</span> {fmtDuration(seg.start_time)} → {fmtDuration(seg.end_time)}</div>
      <div className="tt-row"><span>⏳</span> Duration: {fmtDuration(seg.duration_secs)}</div>
      <div className="tt-row"><span>🎯</span> Confidence: {(seg.confidence * 100).toFixed(0)}%</div>
      <div className="tt-row"><span>🔬</span> Source: {seg.source}</div>
    </div>
  )
}

export default function App() {
  const [activeBackendUrl, setActiveBackendUrl] = useState(DEFAULT_URL)
  const [backendStatus, setBackendStatus] = useState('loading')
  const [manifest, setManifest] = useState([])
  const [selectedVideoId, setSelectedVideoId] = useState('IODxDxX7oi4')
  const [customUrl, setCustomUrl] = useState('')
  const [weightKg, setWeightKg] = useState(70)
  const [userTier, setUserTier] = useState('intermediate')

  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [errorMsg, setErrorMsg] = useState(null)

  // Timeline interaction state
  const [hoveredSeg, setHoveredSeg] = useState(null)
  const [tooltipX, setTooltipX] = useState(0)
  const [activeSeg, setActiveSeg] = useState(null)
  const timelineRef = useRef(null)

  // 1. Probe backend health & load manifest
  useEffect(() => {
    const probeUrl = (url) => {
      fetch(`${url}/health`)
        .then(res => res.json())
        .then(data => {
          if (data.status === 'ok') {
            setActiveBackendUrl(url)
            setBackendStatus('ok')
            fetchManifest(url)
          } else {
            setBackendStatus('error')
          }
        })
        .catch(() => {
          if (url === DEFAULT_URL) probeUrl(ALT_URL)
          else setBackendStatus('error')
        })
    }

    const fetchManifest = (url) => {
      fetch(`${url}/manifest`)
        .then(res => res.json())
        .then(data => {
          if (Array.isArray(data) && data.length > 0) {
            setManifest(data)
            setSelectedVideoId(data[0].youtube_id)
          }
        })
        .catch(err => console.warn('Manifest load warning:', err))
    }

    probeUrl(DEFAULT_URL)
  }, [])

  // Auto-analyze on initial load once healthy
  useEffect(() => {
    if (backendStatus === 'ok' && !result && !analyzing) {
      handleAnalyze()
    }
  }, [backendStatus])

  // 2. Trigger Workout Analysis
  const handleAnalyze = () => {
    setAnalyzing(true)
    setErrorMsg(null)
    setActiveSeg(null)
    setHoveredSeg(null)

    const payload = {
      video_id: selectedVideoId,
      video_url: customUrl.trim() || undefined,
      weight_kg: parseFloat(weightKg),
      user_tier: userTier,
    }

    fetch(`${activeBackendUrl}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then(res => {
        if (!res.ok) throw new Error(`API Error: ${res.status}`)
        return res.json()
      })
      .then(data => setResult(data))
      .catch(() =>
        setErrorMsg(`Failed to connect to backend (${activeBackendUrl}/analyze). Make sure the Uvicorn backend is running.`)
      )
      .finally(() => setAnalyzing(false))
  }

  // Timeline mouse handlers
  const handleBlockMouseMove = (e, seg, idx) => {
    if (!timelineRef.current) return
    const rect = timelineRef.current.getBoundingClientRect()
    const xPct = ((e.clientX - rect.left) / rect.width) * 100
    setHoveredSeg(seg)
    setTooltipX(xPct)
  }

  const handleBlockClick = (seg) => {
    setActiveSeg(prev => prev?.segment_id === seg.segment_id ? null : seg)
  }

  return (
    <div className="app-container">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="brand">
          <span className="brand-icon">🏋️‍♂️</span>
          <div>
            <h1 className="brand-title">CalorieVision</h1>
            <p className="brand-tagline">AI-Powered Exercise Recognition &amp; MET Calorie Estimation</p>
          </div>
        </div>

        <div className="header-status">
          <div className={`status-chip ${backendStatus}`}>
            <span className="dot" />
            <span>{backendStatus === 'ok' ? 'Backend Connected' : backendStatus === 'loading' ? 'Checking API…' : 'API Offline'}</span>
          </div>
          <a href={`${activeBackendUrl}/docs`} target="_blank" rel="noreferrer" className="docs-link">
            Swagger Docs ↗
          </a>
        </div>
      </header>

      {/* ── Controls & Input Panel ── */}
      <div className="panel-grid">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">📹 Workout Video Selector</h2>
            <p className="card-subtitle">Choose a pre-tagged demo video or paste a custom YouTube link</p>
          </div>

          <div className="form-group">
            <label className="form-label">Demo Catalogue Video ({manifest.length} Available)</label>
            <select
              className="input-select"
              value={selectedVideoId}
              onChange={(e) => {
                setSelectedVideoId(e.target.value)
                setCustomUrl('')
              }}
            >
              {manifest.map(item => (
                <option key={item.youtube_id} value={item.youtube_id}>
                  {item.description || item.label}
                  {item.duration_secs ? ` · ${fmtDuration(item.duration_secs)}` : ''}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Or Paste Direct YouTube URL</label>
            <input
              type="text"
              className="input-text"
              placeholder="https://youtu.be/..."
              value={customUrl}
              onChange={(e) => {
                const val = e.target.value
                setCustomUrl(val)
                let extractedId = ""
                if (val.includes("v=")) extractedId = val.split("v=")[1].split("&")[0]
                else if (val.includes("youtu.be/")) extractedId = val.split("youtu.be/")[1].split("?")[0]
                if (extractedId) {
                  const match = manifest.find(m => m.youtube_id === extractedId)
                  if (match) setSelectedVideoId(match.youtube_id)
                }
              }}
            />
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title">⚙️ Parameters</h2>
            <p className="card-subtitle">Personalise MET body weight &amp; intensity tier</p>
          </div>

          <div className="form-group">
            <label className="form-label">Body Weight (kg)</label>
            <div className="weight-row">
              <input
                type="range"
                className="weight-slider"
                min="40"
                max="150"
                value={weightKg}
                onChange={(e) => setWeightKg(e.target.value)}
              />
              <span className="weight-value">{weightKg} kg</span>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Intensity Tier</label>
            <div className="tier-selector">
              {['beginner', 'intermediate', 'advanced'].map(tier => (
                <button
                  key={tier}
                  className={`tier-btn ${userTier === tier ? 'active' : ''}`}
                  onClick={() => setUserTier(tier)}
                >
                  {tier.charAt(0).toUpperCase() + tier.slice(1)}
                </button>
              ))}
            </div>
          </div>

          <button
            className="btn-primary"
            onClick={handleAnalyze}
            disabled={analyzing || backendStatus !== 'ok'}
          >
            {analyzing ? (
              <><span className="spinner-icon">⚡</span> Analysing Pipeline…</>
            ) : (
              <>🚀 Analyse Workout</>
            )}
          </button>
        </div>
      </div>

      {/* ── Error Banner ── */}
      {errorMsg && (
        <div className="card" style={{ borderColor: 'var(--accent-rose)', background: 'rgba(244, 63, 94, 0.1)' }}>
          <strong style={{ color: 'var(--accent-rose)' }}>Error:</strong> {errorMsg}
        </div>
      )}

      {/* ── Results Dashboard ── */}
      {result && (
        <>
          {/* Top Metrics Cards */}
          <div className="metrics-grid">
            <div className="metric-card purple">
              <div className="metric-title">Total Calorie Burn ({userTier})</div>
              <div className="metric-number">
                {result.total_kcal[userTier] ? result.total_kcal[userTier].toFixed(1) : '0.0'}{' '}
                <span style={{ fontSize: '1rem', color: 'var(--accent-cyan)' }}>kcal</span>
              </div>
              <div className="metric-subtitle">
                Beginner: {result.total_kcal.beginner?.toFixed(1)} | Adv: {result.total_kcal.advanced?.toFixed(1)}
              </div>
            </div>

            <div className="metric-card cyan">
              <div className="metric-title">Total Workout Duration</div>
              <div className="metric-number">
                {fmtDuration(result.duration_secs)}
              </div>
              <div className="metric-subtitle">{(result.duration_secs / 60).toFixed(1)} minutes total</div>
            </div>

            <div className="metric-card emerald">
              <div className="metric-title">Recognised Segments</div>
              <div className="metric-number">{result.segments.length}</div>
              <div className="metric-subtitle">Pose + OCR Fused Pipeline</div>
            </div>

            <div className="metric-card amber">
              <div className="metric-title">Camera &amp; Video Setup</div>
              <div className="metric-number" style={{ fontSize: '1.25rem', textTransform: 'capitalize' }}>
                {result.tags?.camera_angle || 'single'} Angle
              </div>
              <div className="metric-subtitle">
                Captions: {result.tags?.caption_present ? 'Yes' : 'No'} | Subjects: {result.tags?.num_subjects || '1'}
              </div>
            </div>
          </div>

          {/* ── Interactive Workout Timeline ── */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">⏱️ Interactive Workout Segment Timeline</h2>
              <p className="card-subtitle">
                Hover over a segment to see details · Click to pin &amp; highlight it in the table below
              </p>
            </div>

            <div className="timeline-section">
              {/* Scrubber bar */}
              <div className="timeline-bar-container" ref={timelineRef} onMouseLeave={() => setHoveredSeg(null)}>
                {result.segments.map((seg, idx) => {
                  const widthPct = (seg.duration_secs / result.duration_secs) * 100
                  const color = labelColor(seg.label)
                  const isActive = activeSeg?.segment_id === seg.segment_id
                  return (
                    <div
                      key={idx}
                      className={`timeline-block ${isActive ? 'tl-active' : ''}`}
                      style={{
                        width: `${Math.max(widthPct, 3)}%`,
                        background: color,
                        boxShadow: isActive ? `0 0 0 3px white, 0 0 0 5px ${color}` : undefined,
                      }}
                      title=""
                      onMouseMove={(e) => handleBlockMouseMove(e, seg, idx)}
                      onClick={() => handleBlockClick(seg)}
                    >
                      {widthPct > 8 && (
                        <span className="tl-label-text">
                          {seg.label.replace(/_/g, ' ')}
                        </span>
                      )}
                    </div>
                  )
                })}

                {/* Floating tooltip */}
                <SegmentTooltip seg={hoveredSeg} visible={!!hoveredSeg} x={tooltipX} />
              </div>

              {/* Time axis ticks */}
              <div className="timeline-ticks">
                {[0, 0.25, 0.5, 0.75, 1.0].map(f => (
                  <span key={f} style={{ left: `${f * 100}%` }}>
                    {fmtDuration(result.duration_secs * f)}
                  </span>
                ))}
              </div>

              {/* Legend */}
              <div className="timeline-legend">
                {Object.entries(LABEL_COLORS).map(([label, color]) => (
                  <div key={label} className="legend-item">
                    <span className="legend-color" style={{ background: color }} />
                    <span>{label.replace(/_/g, ' ')}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Pinned segment detail panel */}
            {activeSeg && (
              <div className="pinned-segment-panel" style={{ borderColor: labelColor(activeSeg.label) }}>
                <div className="pinned-header" style={{ color: labelColor(activeSeg.label) }}>
                  📌 {activeSeg.label.replace(/_/g, ' ')}
                  <button className="pinned-close" onClick={() => setActiveSeg(null)}>✕</button>
                </div>
                <div className="pinned-grid">
                  <div><span>Start</span><strong>{fmtDuration(activeSeg.start_time)}</strong></div>
                  <div><span>End</span><strong>{fmtDuration(activeSeg.end_time)}</strong></div>
                  <div><span>Duration</span><strong>{fmtDuration(activeSeg.duration_secs)}</strong></div>
                  <div><span>Confidence</span><strong>{(activeSeg.confidence * 100).toFixed(0)}%</strong></div>
                  <div><span>Source</span><strong className={`badge badge-${activeSeg.source}`}>{activeSeg.source}</strong></div>
                  <div><span>Calories ({userTier})</span><strong>{activeSeg.kcal?.[userTier]?.toFixed(2)} kcal</strong></div>
                </div>
              </div>
            )}
          </div>

          {/* ── Detailed Segment Table ── */}
          <div className="panel-grid">
            <div className="card" style={{ gridColumn: 'span 2' }}>
              <div className="card-header">
                <h2 className="card-title">📊 Exercise Segment Breakdown &amp; MET Calories</h2>
                <p className="card-subtitle">Per-segment time windows, confidence scores, sources, and MET calorie tier values</p>
              </div>

              <div className="table-responsive">
                <table className="segments-table">
                  <thead>
                    <tr>
                      <th></th>
                      <th>Segment ID</th>
                      <th>Time Window</th>
                      <th>Exercise</th>
                      <th>Duration</th>
                      <th>Source</th>
                      <th>Confidence</th>
                      <th>Beginner</th>
                      <th>Intermediate</th>
                      <th>Advanced</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.segments.map(seg => {
                      const isHighlighted = activeSeg?.segment_id === seg.segment_id
                      return (
                        <tr
                          key={seg.segment_id}
                          className={isHighlighted ? 'row-highlighted' : ''}
                          onClick={() => handleBlockClick(seg)}
                          style={{ cursor: 'pointer' }}
                        >
                          <td>
                            <span
                              style={{
                                display: 'inline-block',
                                width: 10,
                                height: 10,
                                borderRadius: '50%',
                                background: labelColor(seg.label),
                              }}
                            />
                          </td>
                          <td style={{ fontFamily: 'monospace', fontWeight: 600 }}>{seg.segment_id}</td>
                          <td>{fmtDuration(seg.start_time)} → {fmtDuration(seg.end_time)}</td>
                          <td style={{ fontWeight: 700, textTransform: 'capitalize', color: 'var(--text-main)' }}>
                            {seg.label.replace(/_/g, ' ')}
                          </td>
                          <td>{fmtDuration(seg.duration_secs)}</td>
                          <td>
                            <span className={`badge badge-${seg.source}`}>{seg.source}</span>
                          </td>
                          <td>
                            <div className="conf-bar-bg">
                              <div className="conf-bar-fill" style={{ width: `${seg.confidence * 100}%` }} />
                            </div>
                            <span>{(seg.confidence * 100).toFixed(0)}%</span>
                          </td>
                          <td style={{ color: 'var(--text-muted)' }}>{seg.kcal.beginner?.toFixed(2)} kcal</td>
                          <td style={{ fontWeight: 700, color: 'var(--accent-cyan)' }}>{seg.kcal.intermediate?.toFixed(2)} kcal</td>
                          <td style={{ color: 'var(--accent-amber)' }}>{seg.kcal.advanced?.toFixed(2)} kcal</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* ── Fusion Disagreement Inspector ── */}
            {result.failure_events && result.failure_events.length > 0 && (
              <div className="card" style={{ gridColumn: 'span 2' }}>
                <div className="card-header">
                  <h2 className="card-title" style={{ color: 'var(--accent-amber)' }}>
                    🔍 Fusion Arbitration Log — Pose vs OCR Disagreements
                  </h2>
                  <p className="card-subtitle">
                    These are segments where the <strong>pose model</strong> and the <strong>on-screen OCR text</strong> disagreed on which exercise was happening.
                    The pipeline logs them and keeps the higher-confidence reading.
                    Saved to <code>eval/failure_log.jsonl</code> for offline review.
                  </p>
                </div>

                <div className="failure-log-list">
                  {result.failure_events.map((evt, idx) => (
                    <div key={idx} className="failure-item">
                      <div className="failure-header">
                        <span className="fail-badge">DISAGREE</span>
                        <span className="fail-seg-id">Segment <code>{evt.segment_id}</code></span>
                        <span className="fail-time">
                          {fmtDuration(evt.start_time)} → {fmtDuration(evt.end_time)}
                        </span>
                        {evt.timestamp && (
                          <span className="fail-ts">logged {new Date(evt.timestamp).toLocaleTimeString()}</span>
                        )}
                      </div>
                      <div className="failure-body">
                        <div className="fail-col pose-col">
                          <div className="fail-source-label">🦴 Pose Model said</div>
                          <div className="fail-exercise">{(evt.pose_label || '—').replace(/_/g, ' ')}</div>
                          <div className="fail-conf">{evt.pose_confidence != null ? `${(evt.pose_confidence * 100).toFixed(0)}% confidence` : ''}</div>
                        </div>
                        <div className="fail-vs">VS</div>
                        <div className="fail-col ocr-col">
                          <div className="fail-source-label">🔤 On-screen text said</div>
                          <div className="fail-exercise">{(evt.ocr_label || 'nothing').replace(/_/g, ' ')}</div>
                          <div className="fail-conf">{evt.ocr_confidence != null ? `${(evt.ocr_confidence * 100).toFixed(0)}% confidence` : ''}</div>
                        </div>
                        <div className="fail-arrow">→</div>
                        <div className="fail-col result-col">
                          <div className="fail-source-label">✅ Pipeline kept</div>
                          <div className="fail-exercise" style={{ color: 'var(--accent-cyan)' }}>
                            {(evt.fused_label || '—').replace(/_/g, ' ')}
                          </div>
                          <div className="fail-conf">{evt.fused_confidence != null ? `${(evt.fused_confidence * 100).toFixed(0)}% confidence` : ''}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {/* ── Footer ── */}
      <footer className="footer">
        CalorieVision Team 15 · CV Pipeline (MediaPipe + PyTorch) ＋ FastAPI ＋ React
      </footer>
    </div>
  )
}
