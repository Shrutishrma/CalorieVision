/* app/frontend/src/App.jsx — CalorieVision React Dashboard */
import { useState, useEffect } from 'react'
import './style.css'

/* ── helpers ─────────────────────────────────────────────────────────────────── */
function extractVideoId(url) {
  if (!url) return ''
  if (url.includes('v='))       return url.split('v=')[1].split('&')[0].split('?')[0]
  if (url.includes('youtu.be/'))return url.split('youtu.be/')[1].split('?')[0]
  if (url.includes('shorts/'))  return url.split('shorts/')[1].split('?')[0]
  return ''
}

function fmtTime(secs) {
  const s = Math.round(secs || 0)
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60
  if (h > 0) return `${h}h ${m}m ${r}s`
  if (m > 0) return `${m}m ${r}s`
  return `${r}s`
}

function fmtClock(secs) {
  const s = Math.round(secs || 0)
  const m = Math.floor(s / 60), r = s % 60
  return `${m}:${String(r).padStart(2,'0')}`
}

const MET = { squat:5,pushup:8,jumping_jack:8,lunge:5.5,plank:4,burpee:10,
              mountain_climber:9,high_knees:8.5,situp:5.5,jump_rope:10,
              bicycle_crunch:5.5,shoulder_press:5,rest:1,unknown:3 }
const SCALE = { beginner:0.85, intermediate:1.0, advanced:1.2 }

function computeCalories(segs, wt, tier) {
  return segs.reduce((s, seg) => s + (MET[seg.exercise]||3) * (SCALE[tier]||1) * wt * (seg.duration_secs/3600), 0)
}

const BACKENDS = ['http://localhost:8005','http://localhost:8000']

const STAGES = [
  { label:'Video Acquisition',      desc:'yt-dlp + ffmpeg download' },
  { label:'Pose Estimation',        desc:'MediaPipe — 33 3D landmarks/frame' },
  { label:'Motion Filtering',       desc:'Active vs rest frame separation' },
  { label:'Scene Cut Detection',    desc:'PySceneDetect shot boundaries' },
  { label:'Action Classification',  desc:'PyTorch LSTM · 95.3% val accuracy' },
  { label:'EasyOCR Captions',       desc:'On-screen text & rep counter extraction' },
  { label:'Multi-Signal Fusion',    desc:'Pose + OCR confidence arbitration' },
  { label:'Calorie Estimation',     desc:'MET × weight × duration per segment' },
]

function stageIndex(stageName='') {
  const s = stageName.toLowerCase()
  if (s.includes('0')||s.includes('video')||s.includes('init'))  return 0
  if (s.includes('1')||s.includes('pose'))                        return 1
  if (s.includes('2')||s.includes('motion'))                      return 2
  if (s.includes('3')||s.includes('scene'))                       return 3
  if (s.includes('4')||s.includes('lstm')||s.includes('classif')) return 4
  if (s.includes('5')||s.includes('ocr'))                         return 5
  if (s.includes('6')||s.includes('fusion'))                      return 6
  if (s.includes('7')||s.includes('calor')||s.includes('met'))    return 7
  return 0
}

/* ══════════════════════════════════════════════════════════════════════════════
   APP
══════════════════════════════════════════════════════════════════════════════ */
export default function App() {
  const [backendUrl, setBackendUrl]   = useState('')
  const [status, setStatus]           = useState('loading')
  const [manifest, setManifest]       = useState([])

  const [selectedId, setSelectedId]   = useState('')
  const [customUrl, setCustomUrl]     = useState('')
  const [weightKg, setWeightKg]       = useState(70)
  const [tier, setTier]               = useState('intermediate')

  const [analyzing, setAnalyzing]     = useState(false)
  const [jobId, setJobId]             = useState(null)
  const [progress, setProgress]       = useState(0)
  const [stageName, setStageName]     = useState('')

  const [result, setResult]           = useState(null)
  const [activeVideoId, setActiveVideoId] = useState('')
  const [activeSeg, setActiveSeg]     = useState(null)
  const [errorMsg, setErrorMsg]       = useState('')

  /* ── Probe backend ── */
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      for (const url of BACKENDS) {
        try {
          const d = await fetch(`${url}/health`).then(r=>r.json())
          if (d.status==='ok' && d.service==='CalorieVision API') {
            if (cancelled) return
            setBackendUrl(url); setStatus('ok')
            const mf = await fetch(`${url}/manifest`).then(r=>r.json())
            if (Array.isArray(mf) && mf.length) {
              setManifest(mf); setSelectedId(mf[0].youtube_id)
            }
            return
          }
        } catch (_) {}
      }
      if (!cancelled) setStatus('error')
    })()
    return () => { cancelled=true }
  }, [])

  /* ── Auto-analyse on connect ── */
  useEffect(() => {
    if (status==='ok' && backendUrl && !result && !analyzing && !jobId && selectedId) handleAnalyse()
  }, [status, backendUrl, selectedId])

  /* ── Poll job ── */
  useEffect(() => {
    if (!jobId || !backendUrl) return
    const iv = setInterval(async () => {
      try {
        const d = await fetch(`${backendUrl}/status/${jobId}`).then(r=>r.json())
        setProgress(d.progress || 0)
        setStageName(d.stage || '')
        if (d.status==='completed') {
          setResult(d.result); setAnalyzing(false); setJobId(null)
        } else if (d.status==='failed') {
          setErrorMsg(d.error||'Pipeline failed'); setAnalyzing(false); setJobId(null)
        }
      } catch (_) {}
    }, 1200)
    return () => clearInterval(iv)
  }, [jobId, backendUrl])

  /* ── Submit ── */
  const handleAnalyse = () => {
    if (!backendUrl) return
    const vid = customUrl.trim() ? extractVideoId(customUrl) : selectedId
    setActiveVideoId(vid || selectedId)
    setAnalyzing(true); setResult(null); setErrorMsg(''); setActiveSeg(null)
    setProgress(0.04); setStageName('Initializing')

    const payload = {
      video_id:  customUrl.trim() ? undefined : selectedId,
      video_url: customUrl.trim() || undefined,
      weight_kg: parseFloat(weightKg),
      user_tier: tier,
      force_recompute: false,
    }

    fetch(`${backendUrl}/analyze?sync=false`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload),
    })
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(d => {
        if (d.job_id)       setJobId(d.job_id)
        else if (d.segments){ setResult(d); setAnalyzing(false) }
        else                { setResult(d); setAnalyzing(false) }
      })
      .catch(() => {
        setErrorMsg(`Cannot reach ${backendUrl}. Is the backend running?`)
        setAnalyzing(false)
      })
  }

  /* ── Derived for results ── */
  const segments = result?.segments || []
  const nonRest  = segments.filter(s => s.exercise!=='rest' && s.exercise!=='unknown')
  const totalDur = result?.total_duration_secs || segments.reduce((s,x)=>s+x.duration_secs, 0)
  const activeDur= result?.active_duration_secs || nonRest.reduce((s,x)=>s+x.duration_secs, 0)
  const activeRatio = totalDur > 0 ? Math.round((activeDur/totalDur)*100) : 0

  /* Exercise breakdown map */
  const exMap = {}
  nonRest.forEach(s => {
    if (!exMap[s.exercise]) exMap[s.exercise]={dur:0, kcal:0, count:0}
    exMap[s.exercise].dur  += s.duration_secs
    exMap[s.exercise].kcal += s.calories || 0
    exMap[s.exercise].count++
  })
  const exList = Object.entries(exMap).sort((a,b)=>b[1].dur-a[1].dur)
  const maxDur = exList[0]?.[1]?.dur || 1

  const calBeg  = computeCalories(segments, weightKg, 'beginner')
  const calInt  = computeCalories(segments, weightKg, 'intermediate')
  const calAdv  = computeCalories(segments, weightKg, 'advanced')

  return (
    <div className="app">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="site-header">
        <div className="logo">
          <div className="logo-mark">🏋️</div>
          <div>
            <div className="logo-name">CalorieVision</div>
            <div className="logo-desc">Exercise Recognition · MET Calorie Estimation</div>
          </div>
        </div>
        <div className="header-right">
          <div className={`status-pill ${status}`}>
            <span className="status-dot" />
            {status==='ok' ? 'Backend Connected' : status==='error' ? 'Backend Offline' : 'Checking API…'}
          </div>
          {status==='ok' && (
            <a className="docs-link" href={`${backendUrl}/docs`} target="_blank" rel="noreferrer">
              API Docs ↗
            </a>
          )}
        </div>
      </header>

      {/* ── Input ──────────────────────────────────────────────────────────── */}
      <div className="input-section">
        <div className="section-eyebrow">Analyse a workout video</div>
        <div className="input-layout">

          {/* Left — video picker */}
          <div className="input-pane">
            <div className="pane-label">Select from catalogue</div>
            <select
              className="catalogue-select"
              value={selectedId}
              disabled={!!customUrl.trim()}
              onChange={e => { setSelectedId(e.target.value); setCustomUrl('') }}
            >
              {manifest.map(v => (
                <option key={v.youtube_id} value={v.youtube_id}>
                  {v.description} {v.duration_secs ? `(${fmtClock(v.duration_secs)})` : ''}
                </option>
              ))}
            </select>

            <div className="divider-or">or paste YouTube URL</div>

            <div className="url-input-row">
              <input
                className="url-input"
                type="text"
                placeholder="https://youtube.com/watch?v=... or /shorts/..."
                value={customUrl}
                onChange={e => {
                  setCustomUrl(e.target.value)
                  const vid = extractVideoId(e.target.value)
                  if (vid) { const m = manifest.find(x=>x.youtube_id===vid); if(m) setSelectedId(m.youtube_id) }
                }}
              />
              {customUrl.trim() && (
                <button className="btn-clear-url" onClick={() => setCustomUrl('')}>Clear</button>
              )}
            </div>
          </div>

          {/* Right — params */}
          <div className="input-pane" style={{display:'flex',flexDirection:'column',gap:'1.1rem'}}>
            <div>
              <div className="pane-label">Parameters</div>
            </div>

            <div className="param-group">
              <div className="param-label">Body Weight</div>
              <div className="weight-control">
                <input type="range" className="weight-slider" min="40" max="150"
                  value={weightKg} onChange={e=>setWeightKg(e.target.value)} />
                <span className="weight-val">{weightKg} kg</span>
              </div>
            </div>

            <div className="param-group">
              <div className="param-label">Intensity Tier</div>
              <div className="tier-group">
                {['beginner','intermediate','advanced'].map(t => (
                  <button key={t} className={`tier-btn ${tier===t?'active':''}`} onClick={()=>setTier(t)}>
                    {t.charAt(0).toUpperCase()+t.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div style={{marginTop:'auto'}}>
              <button
                className={`analyse-btn ${analyzing?'running':''}`}
                onClick={handleAnalyse}
                disabled={analyzing || status!=='ok'}
              >
                {analyzing
                  ? <><SpinIcon /> Analysing…</>
                  : <>▶ Analyse Workout Video</>
                }
              </button>
              {status!=='ok' && !analyzing && (
                <p style={{fontSize:'0.72rem',color:'var(--text-3)',marginTop:'0.5rem',textAlign:'center'}}>
                  Waiting for backend…
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ── Error ──────────────────────────────────────────────────────────── */}
      {errorMsg && <div className="error-banner">⚠ {errorMsg}</div>}

      {/* ── Loading Modal ──────────────────────────────────────────────────── */}
      {analyzing && (
        <PipelineModal progress={progress} stageName={stageName} />
      )}

      {/* ── Results ────────────────────────────────────────────────────────── */}
      {result && !analyzing && (
        <div className="results-section">

          {/* Section: Video + Stats */}
          <div className="section-header">
            <span className="section-title">Analysis Results</span>
            <span className="section-meta">
              {segments.length} segments · {[...new Set(nonRest.map(s=>s.exercise))].length} active exercise type(s)
            </span>
          </div>

          <div className="video-results-grid">
            {/* YouTube embed */}
            <div className="video-embed-block" style={{position:'relative'}}>
              <span className="video-embed-label">Source Video</span>
              <iframe
                src={`https://www.youtube.com/embed/${activeVideoId}?modestbranding=1&rel=0`}
                title="Workout Video"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>

            {/* Stats sidebar */}
            <div className="stats-sidebar">
              <div className="stat-block accent-green">
                <div className="stat-block-label">Calories Burned</div>
                <div className="stat-block-value">{Math.round(result.total_calories||0)}</div>
                <div className="stat-block-sub">kcal · {tier} tier</div>
              </div>

              <div className="stat-block accent-blue">
                <div className="stat-block-label">Active Workout</div>
                <div className="stat-block-value">{fmtTime(activeDur)}</div>
                <div className="stat-block-sub">{activeRatio}% of {fmtTime(totalDur)}</div>
              </div>

              <div className="tier-comparison">
                <div className="stat-block-label" style={{marginBottom:'0.6rem'}}>Calorie comparison</div>
                <div className={`tier-row ${tier==='beginner'?'active-tier':''}`}>
                  <span className="tier-row-label">🟢 Beginner</span>
                  <span className="tier-row-val">{Math.round(calBeg)} kcal</span>
                </div>
                <div className={`tier-row ${tier==='intermediate'?'active-tier':''}`}>
                  <span className="tier-row-label">🟡 Intermediate</span>
                  <span className="tier-row-val">{Math.round(calInt)} kcal</span>
                </div>
                <div className={`tier-row ${tier==='advanced'?'active-tier':''}`}>
                  <span className="tier-row-label">🔴 Advanced</span>
                  <span className="tier-row-val">{Math.round(calAdv)} kcal</span>
                </div>
              </div>

              <div className="model-chip">
                <div className="model-chip-icon">⚡</div>
                <div>
                  <div className="model-chip-name">PyTorch LSTM</div>
                  <div className="model-chip-acc">95.3% Validation Accuracy</div>
                </div>
              </div>
            </div>
          </div>

          {/* Section: Exercise Breakdown */}
          <div className="exercise-breakdown">
            <div className="section-header">
              <span className="section-title">Exercise Breakdown</span>
              <span className="section-meta">{nonRest.length} active segments</span>
            </div>
            <div className="breakdown-grid">
              {exList.map(([ex, data]) => (
                <div className="breakdown-row" key={ex}>
                  <div className="breakdown-label">
                    {ex.replace(/_/g,' ')}
                    <span>×{data.count}</span>
                  </div>
                  <div className="breakdown-bar-track">
                    <div
                      className={`breakdown-bar-fill bar-${ex}`}
                      style={{width:`${Math.round((data.dur/maxDur)*100)}%`}}
                    />
                  </div>
                  <div className="breakdown-time">{fmtTime(data.dur)}</div>
                  <div className="breakdown-kcal">{Math.round(data.kcal)} kcal</div>
                </div>
              ))}
            </div>
          </div>

          {/* Section: Timeline */}
          <div className="timeline-section">
            <div className="section-header">
              <span className="section-title">Exercise Timeline</span>
              <span className="section-meta">Click any segment for details</span>
            </div>

            <div className="timeline-track">
              {segments.map((seg, idx) => {
                const w = ((seg.end_secs - seg.start_secs) / (totalDur||1)) * 100
                const isActive = activeSeg?.start_secs === seg.start_secs
                return (
                  <div
                    key={idx}
                    className={`timeline-segment tl-${seg.exercise} ${seg.exercise==='rest'?'tl-rest':''} ${isActive?'tl-active':''}`}
                    style={{ width: `${Math.max(w, 0.3)}%` }}
                    title={`${seg.exercise} · ${fmtClock(seg.start_secs)}–${fmtClock(seg.end_secs)}`}
                    onClick={() => setActiveSeg(isActive ? null : seg)}
                  >
                    {w > 5 && seg.exercise !== 'rest' ? seg.exercise.replace(/_/g,' ') : ''}
                  </div>
                )
              })}
            </div>

            <div className="timeline-ticks">
              {[0,0.25,0.5,0.75,1].map(t => (
                <span key={t} className="tl-tick">{fmtClock(totalDur*t)}</span>
              ))}
            </div>

            {activeSeg && (
              <div className="segment-detail-bar">
                <div className="sd-item"><span className="sd-label">Exercise</span>
                  <span className="sd-value" style={{textTransform:'capitalize'}}>
                    {activeSeg.exercise.replace(/_/g,' ')}
                  </span>
                </div>
                <div className="sd-item"><span className="sd-label">Time</span>
                  <span className="sd-value">{fmtClock(activeSeg.start_secs)} → {fmtClock(activeSeg.end_secs)}</span>
                </div>
                <div className="sd-item"><span className="sd-label">Duration</span>
                  <span className="sd-value">{activeSeg.duration_secs}s</span>
                </div>
                <div className="sd-item"><span className="sd-label">Calories</span>
                  <span className="sd-value">{activeSeg.calories} kcal</span>
                </div>
                <div className="sd-item"><span className="sd-label">Confidence</span>
                  <span className="sd-value">{Math.round((activeSeg.confidence||0)*100)}%</span>
                </div>
              </div>
            )}
          </div>

          {/* Section: Segment Table */}
          <div className="segments-section">
            <div className="section-header">
              <span className="section-title">Segment Breakdown</span>
              <span className="section-meta">{segments.length} total segments</span>
            </div>
            <div className="table-scroll">
              <table className="seg-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Exercise</th>
                    <th>Start</th>
                    <th>End</th>
                    <th>Duration</th>
                    <th>Confidence</th>
                    <th>Calories</th>
                  </tr>
                </thead>
                <tbody>
                  {segments.map((seg, i) => {
                    const isActive = activeSeg?.start_secs === seg.start_secs
                    return (
                      <tr
                        key={i}
                        className={isActive ? 'seg-active' : ''}
                        onClick={() => setActiveSeg(isActive ? null : seg)}
                      >
                        <td className="mono" style={{color:'var(--text-3)'}}>{i+1}</td>
                        <td>
                          <span className={`ex-tag ex-${seg.exercise}`}>
                            {seg.exercise.replace(/_/g,' ')}
                          </span>
                        </td>
                        <td className="mono">{fmtClock(seg.start_secs)}</td>
                        <td className="mono">{fmtClock(seg.end_secs)}</td>
                        <td className="mono">{seg.duration_secs}s</td>
                        <td>
                          <div className="conf-bar">
                            <div className="conf-track">
                              <div className="conf-fill" style={{width:`${Math.round((seg.confidence||0)*100)}%`}} />
                            </div>
                            <span className="mono" style={{fontSize:'0.7rem',color:'var(--text-3)'}}>
                              {Math.round((seg.confidence||0)*100)}%
                            </span>
                          </div>
                        </td>
                        <td className="mono" style={{color:'var(--blue)'}}>{seg.calories} kcal</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Section: Failure Log */}
          <div className="failure-section">
            <div className="section-header">
              <span className="section-title">Pipeline Fallback Log</span>
              <span className="section-meta">Stages that required fallback</span>
            </div>
            {result.failure_events?.length > 0 ? (
              result.failure_events.map((ev, i) => (
                <div key={i} className="failure-item">
                  <strong>{ev.stage || 'Unknown'}</strong>{ev.reason ? ` — ${ev.reason}` : ''}
                  {ev.fallback && <span style={{color:'var(--text-3)'}}> (fallback: {ev.fallback})</span>}
                </div>
              ))
            ) : (
              <div className="all-ok-row">
                <span>✓</span> All pipeline stages completed successfully — no fallbacks required
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="site-footer">
        <span>CalorieVision · Team 15</span>
        <div className="footer-pills">
          <span className="footer-pill">MediaPipe Pose</span>
          <span className="footer-pill">PyTorch LSTM</span>
          <span className="footer-pill">EasyOCR</span>
          <span className="footer-pill">FastAPI</span>
          <span className="footer-pill">React</span>
        </div>
      </footer>
    </div>
  )
}

/* ── Spinner icon ─────────────────────────────────────────────────────────── */
function SpinIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"
      style={{animation:'spin 0.8s linear infinite', display:'block'}}>
      <style>{`@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}`}</style>
      <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
      <path d="M12 2a10 10 0 0 1 10 10" />
    </svg>
  )
}

/* ── Pipeline Modal ───────────────────────────────────────────────────────── */
function PipelineModal({ progress=0, stageName='' }) {
  const pct = Math.min(Math.round(progress*100), 99)
  const cur = stageIndex(stageName)

  return (
    <div className="modal-backdrop">
      <div className="modal-box">
        <div className="modal-header">
          <div className="modal-badge">
            <svg width="8" height="8" viewBox="0 0 8 8" style={{animation:'spin 1.5s linear infinite'}}>
              <style>{`@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}`}</style>
              <circle cx="4" cy="4" r="3" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="10 4" />
            </svg>
            Executing Pipeline
          </div>
          <span className="modal-pct">{pct}%</span>
        </div>

        <div className="modal-title">Analysing Workout Video</div>
        <div className="modal-sub">Running multi-stage computer vision pipeline</div>

        <div className="progress-track">
          <div className="progress-fill" style={{width:`${pct}%`}} />
        </div>

        <div className="stage-list">
          {STAGES.map((s, i) => {
            const done = i < cur, current = i === cur
            return (
              <div key={i} className={`stage-row ${done?'done':''} ${current?'current':''}`}>
                <div className="stage-icon">
                  {done ? '✓' : (current ? '→' : i+1)}
                </div>
                <div>
                  <div>Stage {i}: {s.label}</div>
                  {current && <div className="stage-desc">{s.desc}</div>}
                </div>
              </div>
            )
          })}
        </div>

        <div className="modal-note">
          ℹ PyTorch LSTM (95.3% Val Acc) · MediaPipe Pose 33-landmark 3D · EasyOCR text extraction
        </div>
      </div>
    </div>
  )
}
