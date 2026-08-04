import React from 'react'

function fmtDuration(secs) {
  const s = Math.round(secs || 0)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const rem = s % 60
  if (h > 0) return `${h}h ${m}m ${rem}s`
  if (m > 0) return `${m}m ${rem}s`
  return `${rem}s`
}

export default function MetricsGrid({ result, userTier }) {
  const isLstm = result.classifier_used === 'lstm'
  const classifierBadgeText = isLstm
    ? 'PyTorch LSTM (95.3% Val Acc)'
    : result.classifier_used
    ? result.classifier_used.toUpperCase().replace(/_/g, ' ')
    : 'POSE + OCR'

  const activeKcal = result.total_kcal[userTier] ? result.total_kcal[userTier].toFixed(1) : '0.0'

  return (
    <div className="metrics-hero-container">
      {/* Top Banner Hero */}
      <div className="metrics-hero-header">
        <div className="hero-title-group">
          <h2 className="hero-title">Analysis Results Summary</h2>
          <p className="hero-subtitle">
            Target Video: <strong>{result.video_title}</strong> ({fmtDuration(result.duration_secs)})
          </p>
        </div>

        <div className={`classifier-badge ${isLstm ? 'lstm-active' : ''}`}>
          <span className="badge-icon">{isLstm ? '🧠' : '📊'}</span>
          <div className="badge-text-group">
            <span className="badge-label">Active Classifier Model</span>
            <span className="badge-value">{classifierBadgeText}</span>
          </div>
        </div>
      </div>

      {/* Grid Cards */}
      <div className="metrics-grid">
        <div className="metric-card purple glow">
          <div className="metric-icon">🔥</div>
          <div className="metric-content">
            <div className="metric-title">Total Calorie Expenditure</div>
            <div className="metric-number">
              {activeKcal} <span className="unit">kcal</span>
            </div>
            <div className="metric-subtitle">
              User Weight: <strong>{result.weight_kg} kg</strong> · Tier: <strong className="tier-tag">{userTier}</strong>
            </div>
          </div>
          <div className="tier-pill-row">
            <span className={`tier-pill ${userTier === 'beginner' ? 'active' : ''}`}>
              Beg: {result.total_kcal.beginner?.toFixed(1)}
            </span>
            <span className={`tier-pill ${userTier === 'intermediate' ? 'active' : ''}`}>
              Int: {result.total_kcal.intermediate?.toFixed(1)}
            </span>
            <span className={`tier-pill ${userTier === 'advanced' ? 'active' : ''}`}>
              Adv: {result.total_kcal.advanced?.toFixed(1)}
            </span>
          </div>
        </div>

        <div className="metric-card cyan">
          <div className="metric-icon">⏱️</div>
          <div className="metric-content">
            <div className="metric-title">Workout Duration</div>
            <div className="metric-number">{fmtDuration(result.duration_secs)}</div>
            <div className="metric-subtitle">
              {(result.duration_secs / 60).toFixed(1)} mins total video timeline
            </div>
          </div>
        </div>

        <div className="metric-card emerald">
          <div className="metric-icon">🏋️‍♂️</div>
          <div className="metric-content">
            <div className="metric-title">Exercise Action Segments</div>
            <div className="metric-number">{result.segments.length}</div>
            <div className="metric-subtitle">
              Classified &amp; synchronized exercise blocks
            </div>
          </div>
        </div>

        <div className="metric-card amber">
          <div className="metric-icon">🎥</div>
          <div className="metric-content">
            <div className="metric-title">Video &amp; Vision Specs</div>
            <div className="metric-number" style={{ fontSize: '1.2rem', textTransform: 'capitalize' }}>
              {result.tags?.camera_angle || 'single'} Angle
            </div>
            <div className="metric-subtitle">
              Captions: {result.tags?.caption_present ? 'Detected' : 'None'} · Body Count: {result.tags?.num_subjects || '1'}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
