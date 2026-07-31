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

export default function FailureLog({ failureEvents }) {
  if (!failureEvents || failureEvents.length === 0) return null

  return (
    <div className="card" style={{ gridColumn: 'span 2' }}>
      <div className="card-header">
        <h2 className="card-title" style={{ color: 'var(--accent-amber)' }}>
          🔍 Fusion Arbitration Log — Pose vs OCR Disagreements
        </h2>
        <p className="card-subtitle">
          These are segments where the <strong>pose model</strong> and the <strong>on-screen OCR text</strong> disagreed on which exercise was happening.
          The pipeline logs them and arbitrates using confidence weights. Saved to <code>eval/failure_log.jsonl</code>.
        </p>
      </div>

      <div className="failure-log-list">
        {failureEvents.map((evt, idx) => (
          <div key={idx} className="failure-item">
            <div className="failure-header">
              <span className="fail-badge">DISAGREE</span>
              <span className="fail-seg-id">
                Segment <code>{evt.segment_id}</code>
              </span>
              <span className="fail-time">
                {fmtDuration(evt.start_time)} → {fmtDuration(evt.end_time)}
              </span>
              {evt.timestamp && (
                <span className="fail-ts">
                  logged {new Date(evt.timestamp).toLocaleTimeString()}
                </span>
              )}
            </div>
            <div className="failure-body">
              <div className="fail-col pose-col">
                <div className="fail-source-label">🦴 Pose Model said</div>
                <div className="fail-exercise">{(evt.pose_label || '—').replace(/_/g, ' ')}</div>
                <div className="fail-conf">
                  {evt.pose_confidence != null ? `${(evt.pose_confidence * 100).toFixed(0)}% confidence` : ''}
                </div>
              </div>
              <div className="fail-vs">VS</div>
              <div className="fail-col ocr-col">
                <div className="fail-source-label">🔤 On-screen text said</div>
                <div className="fail-exercise">{(evt.ocr_label || 'nothing').replace(/_/g, ' ')}</div>
                <div className="fail-conf">
                  {evt.ocr_confidence != null ? `${(evt.ocr_confidence * 100).toFixed(0)}% confidence` : ''}
                </div>
              </div>
              <div className="fail-arrow">→</div>
              <div className="fail-col result-col">
                <div className="fail-source-label">✅ Pipeline kept</div>
                <div className="fail-exercise" style={{ color: 'var(--accent-cyan)' }}>
                  {(evt.fused_label || '—').replace(/_/g, ' ')}
                </div>
                <div className="fail-conf">
                  {evt.fused_confidence != null ? `${(evt.fused_confidence * 100).toFixed(0)}% confidence` : ''}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
