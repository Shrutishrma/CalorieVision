import React, { useRef, useState } from 'react'

function fmtDuration(secs) {
  const s = Math.round(secs)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const rem = s % 60
  if (h > 0) return `${h}h ${m}m ${rem}s`
  if (m > 0) return `${m}m ${rem}s`
  return `${rem}s`
}

const LABEL_COLORS = {
  // ── Original 12 ──
  squat: '#6366f1',
  pushup: '#0ea5e9',
  jumping_jack: '#f59e0b',
  lunge: '#10b981',
  plank: '#8b5cf6',
  burpee: '#ef4444',
  mountain_climber: '#ec4899',
  high_knees: '#14b8a6',
  situp: '#f97316',
  jump_rope: '#84cc16',
  bicycle_crunch: '#06b6d4',
  shoulder_press: '#a855f7',
  // ── Expanded 14 ──
  deadlift: '#dc2626',
  pull_up: '#2563eb',
  bench_press: '#7c3aed',
  tricep_dip: '#059669',
  leg_raise: '#d97706',
  wall_sit: '#4f46e5',
  box_jump: '#e11d48',
  russian_twist: '#0891b2',
  hip_thrust: '#be185d',
  calf_raise: '#65a30d',
  lateral_raise: '#9333ea',
  bicep_curl: '#0d9488',
  kettlebell_swing: '#ea580c',
  superman_hold: '#4338ca',
}
const DEFAULT_COLOR = '#64748b'
export const labelColor = (label) => LABEL_COLORS[label] ?? DEFAULT_COLOR

function SegmentTooltip({ seg, visible, x }) {
  if (!visible || !seg) return null
  return (
    <div className="timeline-tooltip" style={{ left: `${Math.min(x, 80)}%` }}>
      <div className="tt-label">{seg.label.replace(/_/g, ' ')}</div>
      <div className="tt-row">
        <span>⏱</span> {fmtDuration(seg.start_time)} → {fmtDuration(seg.end_time)}
      </div>
      <div className="tt-row">
        <span>⏳</span> Duration: {fmtDuration(seg.duration_secs || (seg.end_time - seg.start_time))}
      </div>
      <div className="tt-row">
        <span>🎯</span> Confidence: {(seg.confidence * 100).toFixed(0)}%
      </div>
      <div className="tt-row">
        <span>🔬</span> Source: {seg.source}
      </div>
    </div>
  )
}

export default function Timeline({ result, activeSeg, setActiveSeg, userTier }) {
  const [hoveredSeg, setHoveredSeg] = useState(null)
  const [tooltipX, setTooltipX] = useState(0)
  const timelineRef = useRef(null)

  const handleBlockMouseMove = (e, seg) => {
    if (!timelineRef.current) return
    const rect = timelineRef.current.getBoundingClientRect()
    const xPct = ((e.clientX - rect.left) / rect.width) * 100
    setHoveredSeg(seg)
    setTooltipX(xPct)
  }

  const handleBlockClick = (seg) => {
    setActiveSeg((prev) => (prev?.segment_id === seg.segment_id ? null : seg))
  }

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">⏱️ Interactive Workout Segment Timeline</h2>
        <p className="card-subtitle">
          Hover over a segment to see details · Click to pin &amp; highlight it in the table below
        </p>
      </div>

      <div className="timeline-section">
        <div
          className="timeline-bar-container"
          ref={timelineRef}
          onMouseLeave={() => setHoveredSeg(null)}
        >
          {result.segments.map((seg, idx) => {
            const segDur = seg.duration_secs || (seg.end_time - seg.start_time)
            const widthPct = (segDur / result.duration_secs) * 100
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
                onMouseMove={(e) => handleBlockMouseMove(e, seg)}
                onClick={() => handleBlockClick(seg)}
              >
                {widthPct > 8 && (
                  <span className="tl-label-text">{seg.label.replace(/_/g, ' ')}</span>
                )}
              </div>
            )
          })}

          <SegmentTooltip seg={hoveredSeg} visible={!!hoveredSeg} x={tooltipX} />
        </div>

        <div className="timeline-ticks">
          {[0, 0.25, 0.5, 0.75, 1.0].map((f) => (
            <span key={f} style={{ left: `${f * 100}%` }}>
              {fmtDuration(result.duration_secs * f)}
            </span>
          ))}
        </div>

        <div className="timeline-legend">
          {Object.entries(LABEL_COLORS).map(([label, color]) => (
            <div key={label} className="legend-item">
              <span className="legend-color" style={{ background: color }} />
              <span>{label.replace(/_/g, ' ')}</span>
            </div>
          ))}
        </div>
      </div>

      {activeSeg && (
        <div
          className="pinned-segment-panel"
          style={{ borderColor: labelColor(activeSeg.label) }}
        >
          <div
            className="pinned-header"
            style={{ color: labelColor(activeSeg.label) }}
          >
            📌 {activeSeg.label.replace(/_/g, ' ')}
            <button className="pinned-close" onClick={() => setActiveSeg(null)}>
              ✕
            </button>
          </div>
          <div className="pinned-grid">
            <div>
              <span>Start</span>
              <strong>{fmtDuration(activeSeg.start_time)}</strong>
            </div>
            <div>
              <span>End</span>
              <strong>{fmtDuration(activeSeg.end_time)}</strong>
            </div>
            <div>
              <span>Duration</span>
              <strong>{fmtDuration(activeSeg.duration_secs || (activeSeg.end_time - activeSeg.start_time))}</strong>
            </div>
            <div>
              <span>Confidence</span>
              <strong>{(activeSeg.confidence * 100).toFixed(0)}%</strong>
            </div>
            <div>
              <span>Source</span>
              <strong className={`badge badge-${activeSeg.source}`}>
                {activeSeg.source}
              </strong>
            </div>
            <div>
              <span>Calories ({userTier})</span>
              <strong>{activeSeg.kcal?.[userTier]?.toFixed(2)} kcal</strong>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
