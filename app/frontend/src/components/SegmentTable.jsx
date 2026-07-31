import React from 'react'
import { labelColor } from './Timeline'

function fmtDuration(secs) {
  const s = Math.round(secs)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const rem = s % 60
  if (h > 0) return `${h}h ${m}m ${rem}s`
  if (m > 0) return `${m}m ${rem}s`
  return `${rem}s`
}

export default function SegmentTable({ result, activeSeg, setActiveSeg }) {
  const handleBlockClick = (seg) => {
    setActiveSeg((prev) => (prev?.segment_id === seg.segment_id ? null : seg))
  }

  return (
    <div className="card" style={{ gridColumn: 'span 2' }}>
      <div className="card-header">
        <h2 className="card-title">📊 Exercise Segment Breakdown &amp; MET Calories</h2>
        <p className="card-subtitle">
          Per-segment time windows, confidence scores, sources, and MET calorie tier values
        </p>
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
            {result.segments.map((seg) => {
              const isHighlighted = activeSeg?.segment_id === seg.segment_id
              const segDur = seg.duration_secs || (seg.end_time - seg.start_time)
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
                  <td>
                    {fmtDuration(seg.start_time)} → {fmtDuration(seg.end_time)}
                  </td>
                  <td
                    style={{
                      fontWeight: 700,
                      textTransform: 'capitalize',
                      color: 'var(--text-main)',
                    }}
                  >
                    {seg.label.replace(/_/g, ' ')}
                  </td>
                  <td>{fmtDuration(segDur)}</td>
                  <td>
                    <span className={`badge badge-${seg.source}`}>{seg.source}</span>
                  </td>
                  <td>
                    <div className="conf-bar-bg">
                      <div
                        className="conf-bar-fill"
                        style={{ width: `${(seg.confidence || 0) * 100}%` }}
                      />
                    </div>
                    <span>{((seg.confidence || 0) * 100).toFixed(0)}%</span>
                  </td>
                  <td style={{ color: 'var(--text-muted)' }}>
                    {seg.kcal?.beginner?.toFixed(2)} kcal
                  </td>
                  <td style={{ fontWeight: 700, color: 'var(--accent-cyan)' }}>
                    {seg.kcal?.intermediate?.toFixed(2)} kcal
                  </td>
                  <td style={{ color: 'var(--accent-amber)' }}>
                    {seg.kcal?.advanced?.toFixed(2)} kcal
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
