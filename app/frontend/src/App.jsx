/* app/frontend/src/App.jsx — CalorieVision React Dashboard */
import { useState, useEffect } from 'react'
import Header from './components/Header'
import VideoSelector from './components/VideoSelector'
import PipelineProgress from './components/PipelineProgress'
import MetricsGrid from './components/MetricsGrid'
import Timeline from './components/Timeline'
import SegmentTable from './components/SegmentTable'
import FailureLog from './components/FailureLog'

const DEFAULT_URL = 'http://localhost:8000'
const ALT_URL = 'http://127.0.0.1:8000'

export default function App() {
  const [activeBackendUrl, setActiveBackendUrl] = useState(DEFAULT_URL)
  const [backendStatus, setBackendStatus] = useState('loading')
  const [manifest, setManifest] = useState([])

  const [selectedVideoId, setSelectedVideoId] = useState('IODxDxX7oi4')
  const [customUrl, setCustomUrl] = useState('')
  const [weightKg, setWeightKg] = useState(70)
  const [userTier, setUserTier] = useState('intermediate')
  const [videoDurationMins, setVideoDurationMins] = useState('')
  const [forceRecompute, setForceRecompute] = useState(false)

  const [analyzing, setAnalyzing] = useState(false)
  const [activeJobId, setActiveJobId] = useState(null)
  const [jobProgress, setJobProgress] = useState(0)
  const [jobStage, setJobStage] = useState('')

  const [result, setResult] = useState(null)
  const [errorMsg, setErrorMsg] = useState(null)
  const [activeSeg, setActiveSeg] = useState(null)

  // 1. Probe backend health & load manifest
  useEffect(() => {
    const probeUrl = (url) => {
      fetch(`${url}/health`)
        .then((res) => res.json())
        .then((data) => {
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
        .then((res) => res.json())
        .then((data) => {
          if (Array.isArray(data) && data.length > 0) {
            setManifest(data)
            setSelectedVideoId(data[0].youtube_id)
          }
        })
        .catch((err) => console.warn('Manifest load warning:', err))
    }

    probeUrl(DEFAULT_URL)
  }, [])

  // Auto-analyze on initial load once healthy
  useEffect(() => {
    if (backendStatus === 'ok' && !result && !analyzing && !activeJobId) {
      handleAnalyze()
    }
  }, [backendStatus])

  // 2. Poll job status if an active async job is running
  useEffect(() => {
    if (!activeJobId) return

    const interval = setInterval(() => {
      fetch(`${activeBackendUrl}/status/${activeJobId}`)
        .then((res) => res.json())
        .then((data) => {
          setJobProgress(data.progress || 0)
          setJobStage(data.stage || 'Processing')

          if (data.status === 'completed' && data.result) {
            setResult(data.result)
            setAnalyzing(false)
            setActiveJobId(null)
          } else if (data.status === 'failed') {
            setErrorMsg(`Pipeline job failed: ${data.error || 'Unknown error'}`)
            setAnalyzing(false)
            setActiveJobId(null)
          }
        })
        .catch((err) => {
          console.warn('Status poll error:', err)
        })
    }, 1500)

    return () => clearInterval(interval)
  }, [activeJobId, activeBackendUrl])

  // 3. Trigger Workout Analysis
  const handleAnalyze = () => {
    setAnalyzing(true)
    setErrorMsg(null)
    setActiveSeg(null)
    setJobProgress(0.05)
    setJobStage('Initializing')

    const durationMins =
      customUrl.trim() && videoDurationMins ? parseFloat(videoDurationMins) : undefined

    const payload = {
      video_id: selectedVideoId,
      video_url: customUrl.trim() || undefined,
      weight_kg: parseFloat(weightKg),
      user_tier: userTier,
      video_duration_mins: durationMins,
      force_recompute: forceRecompute,
    }

    // Default to sync=true for fast cached videos, falls back to async job polling
    fetch(`${activeBackendUrl}/analyze?sync=true`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`API Error: ${res.status}`)
        return res.json()
      })
      .then((data) => {
        if (data.status === 'completed' && data.segments) {
          setResult(data)
          setAnalyzing(false)
        } else if (data.job_id) {
          setActiveJobId(data.job_id)
        } else {
          setResult(data)
          setAnalyzing(false)
        }
      })
      .catch((err) => {
        setErrorMsg(
          `Failed to connect to backend (${activeBackendUrl}/analyze). Make sure the Uvicorn backend is running.`
        )
        setAnalyzing(false)
      })
  }

  return (
    <div className="app-container">
      <Header backendStatus={backendStatus} activeBackendUrl={activeBackendUrl} />

      <VideoSelector
        manifest={manifest}
        selectedVideoId={selectedVideoId}
        setSelectedVideoId={setSelectedVideoId}
        customUrl={customUrl}
        setCustomUrl={setCustomUrl}
        videoDurationMins={videoDurationMins}
        setVideoDurationMins={setVideoDurationMins}
        weightKg={weightKg}
        setWeightKg={setWeightKg}
        userTier={userTier}
        setUserTier={setUserTier}
        forceRecompute={forceRecompute}
        setForceRecompute={setForceRecompute}
        handleAnalyze={handleAnalyze}
        analyzing={analyzing}
        backendStatus={backendStatus}
      />

      {analyzing && <PipelineProgress progress={jobProgress} currentStage={jobStage} />}

      {errorMsg && (
        <div
          className="card"
          style={{
            borderColor: 'var(--accent-rose)',
            background: 'rgba(244, 63, 94, 0.1)',
            marginBottom: '1rem',
          }}
        >
          <strong style={{ color: 'var(--accent-rose)' }}>Error:</strong> {errorMsg}
        </div>
      )}

      {result && (
        <>
          <MetricsGrid result={result} userTier={userTier} />

          <Timeline
            result={result}
            activeSeg={activeSeg}
            setActiveSeg={setActiveSeg}
            userTier={userTier}
          />

          <div className="panel-grid">
            <SegmentTable
              result={result}
              activeSeg={activeSeg}
              setActiveSeg={setActiveSeg}
            />
            <FailureLog failureEvents={result.failure_events} />
          </div>
        </>
      )}

      <footer className="footer">
        CalorieVision Team 15 · Multi-Stage CV Pipeline (MediaPipe + PyTorch) ＋ FastAPI ＋ React
      </footer>
    </div>
  )
}
