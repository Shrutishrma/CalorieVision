import React from 'react'

function fmtDuration(secs) {
  const s = Math.round(secs)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const rem = s % 60
  if (h > 0) return `${h}h ${m}m ${rem}s`
  if (m > 0) return `${m}m ${rem}s`
  return `${rem}s`
}

export default function MetricsGrid({ result, userTier }) {
  const classifierDisplay = result.classifier_used
    ? result.classifier_used.toUpperCase().replace(/_/g, ' ')
    : 'POSE + OCR'

  return (
    <div className="metrics-grid">
      <div className="metric-card purple">
        <div className="metric-title">Total Calorie Burn ({userTier})</div>
        <div className="metric-number">
          {result.total_kcal[userTier] ? result.total_kcal[userTier].toFixed(1) : '0.0'}{' '}
          <span style={{ fontSize: '1rem', color: 'var(--accent-cyan)' }}>kcal</span>
        </div>
        <div className="metric-subtitle">
          Beg: {result.total_kcal.beginner?.toFixed(1)} | Adv: {result.total_kcal.advanced?.toFixed(1)}
        </div>
      </div>

      <div className="metric-card cyan">
        <div className="metric-title">Total Workout Duration</div>
        <div className="metric-number">{fmtDuration(result.duration_secs)}</div>
        <div className="metric-subtitle">{(result.duration_secs / 60).toFixed(1)} minutes total</div>
      </div>

      <div className="metric-card emerald">
        <div className="metric-title">Recognised Segments</div>
        <div className="metric-number">{result.segments.length}</div>
        <div className="metric-subtitle">Classifier: <strong>{classifierDisplay}</strong></div>
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
  )
}
