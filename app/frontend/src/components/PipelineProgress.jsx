import React from 'react'

const STAGES = [
  { id: 'download', name: 'Resolving / Downloading Video' },
  { id: 'keypoints', name: 'MediaPipe Keypoint Extraction' },
  { id: 'motion_filter', name: 'Active / Rest Motion Filter' },
  { id: 'scene_detect', name: 'Scene Cut Detection' },
  { id: 'classification', name: 'Exercise Action Classifier' },
  { id: 'ocr', name: 'EasyOCR Text Detection' },
  { id: 'fusion', name: 'Multi-Signal Fusion' },
  { id: 'calorie', name: '3-Tier MET Calorie Calculation' },
]

export default function PipelineProgress({ progress, currentStage }) {
  const pct = Math.round((progress || 0) * 100)

  return (
    <div className="card pipeline-progress-card">
      <div className="card-header">
        <h2 className="card-title" style={{ color: 'var(--accent-cyan)' }}>
          ⚡ Multi-Stage CV Pipeline Executing…
        </h2>
        <p className="card-subtitle">
          Running real video stages: MediaPipe Pose → PySceneDetect → Classifier → EasyOCR → Fusion
        </p>
      </div>

      <div className="progress-bar-container" style={{ background: 'rgba(255,255,255,0.05)', borderRadius: '8px', padding: '4px', marginBottom: '1rem' }}>
        <div
          className="progress-bar-fill"
          style={{
            width: `${Math.max(pct, 5)}%`,
            height: '10px',
            borderRadius: '6px',
            background: 'linear-gradient(90deg, #6366f1, #0ea5e9, #10b981)',
            transition: 'width 0.3s ease',
          }}
        />
      </div>

      <div className="stage-status-text" style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '0.75rem' }}>
        Current Stage: <span style={{ color: 'var(--accent-cyan)' }}>{currentStage || 'Initializing…'}</span> ({pct}%)
      </div>

      <div className="stages-pills-grid" style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
        {STAGES.map((st) => {
          const isCurrent = currentStage && currentStage.toLowerCase().includes(st.id)
          return (
            <div
              key={st.id}
              className={`stage-pill ${isCurrent ? 'active' : ''}`}
              style={{
                padding: '4px 10px',
                borderRadius: '12px',
                fontSize: '0.8rem',
                background: isCurrent ? 'var(--accent-indigo)' : 'rgba(255,255,255,0.05)',
                color: isCurrent ? '#fff' : 'var(--text-muted)',
                border: isCurrent ? '1px solid var(--accent-cyan)' : '1px solid transparent',
              }}
            >
              {st.name}
            </div>
          )
        })}
      </div>
    </div>
  )
}
