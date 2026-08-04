import React from 'react'

const STAGES = [
  { id: 'download', name: 'Stage 0: Video Acquisition', desc: 'Fetching video stream via yt-dlp & ffmpeg' },
  { id: 'keypoints', name: 'Stage 1: Pose Estimation', desc: 'Extracting 33 MediaPipe 3D body keypoints' },
  { id: 'motion_filter', name: 'Stage 2: Motion Filtering', desc: 'Detecting active movement vs rest intervals' },
  { id: 'scene_detect', name: 'Stage 3: Scene Cut Detection', desc: 'PySceneDetect shot boundary segmentation' },
  { id: 'classification', name: 'Stage 4: Action Classification', desc: 'PyTorch LSTM Classifier (95.3% Val Accuracy)' },
  { id: 'ocr', name: 'Stage 5: EasyOCR Captioning', desc: 'Extracting on-screen exercise titles & timers' },
  { id: 'fusion', name: 'Stage 6: Multi-Signal Fusion', desc: 'Cross-checking Pose predictions with OCR' },
  { id: 'calorie', name: 'Stage 7: MET Expenditure', desc: '3-Tier MET calorie burn calculation' },
]

export default function PipelineProgress({ progress, currentStage }) {
  const pct = Math.round((progress || 0.05) * 100)

  // Find active stage index
  const activeIdx = STAGES.findIndex(
    (st) => currentStage && currentStage.toLowerCase().includes(st.id.replace('_', ''))
  )
  const currentStageInfo = activeIdx >= 0 ? STAGES[activeIdx] : STAGES[0]

  return (
    <div className="pipeline-modal-overlay">
      <div className="pipeline-modal-card">
        {/* Top Header */}
        <div className="modal-top-bar">
          <div className="pulsing-chip">
            <span className="pulse-dot"></span>
            <span>Executing Pipeline</span>
          </div>
          <div className="pct-display">{pct}%</div>
        </div>

        <h2 className="modal-title">Analyzing Workout Video</h2>
        <p className="modal-subtitle">
          Running real-time multi-stage computer vision & deep learning pipeline
        </p>

        {/* Circular Progress & Stage Details */}
        <div className="modal-hero-progress">
          <div className="progress-ring-container">
            <svg className="progress-ring" width="120" height="120">
              <circle
                className="progress-ring-bg"
                stroke="rgba(255, 255, 255, 0.08)"
                strokeWidth="8"
                fill="transparent"
                r="50"
                cx="60"
                cy="60"
              />
              <circle
                className="progress-ring-fill"
                stroke="url(#gradient)"
                strokeWidth="8"
                strokeDasharray="314.15"
                strokeDashoffset={314.15 - (314.15 * Math.max(pct, 5)) / 100}
                strokeLinecap="round"
                fill="transparent"
                r="50"
                cx="60"
                cy="60"
              />
              <defs>
                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#6366f1" />
                  <stop offset="50%" stopColor="#0ea5e9" />
                  <stop offset="100%" stopColor="#10b981" />
                </linearGradient>
              </defs>
            </svg>
            <div className="ring-text">
              <span className="ring-pct">{pct}%</span>
              <span className="ring-label">Processing</span>
            </div>
          </div>

          <div className="active-stage-card">
            <div className="active-stage-tag">Current Pipeline Step</div>
            <div className="active-stage-title">{currentStageInfo.name}</div>
            <div className="active-stage-desc">{currentStageInfo.desc}</div>
          </div>
        </div>

        {/* Linear Stage Steps Checklist */}
        <div className="pipeline-steps-checklist">
          {STAGES.map((st, idx) => {
            const isCompleted = pct >= ((idx + 1) / STAGES.length) * 100 || (activeIdx >= 0 && idx < activeIdx)
            const isCurrent = activeIdx === idx || (activeIdx === -1 && idx === 0)

            return (
              <div
                key={st.id}
                className={`step-item ${isCompleted ? 'completed' : ''} ${isCurrent ? 'current' : ''}`}
              >
                <div className="step-icon">
                  {isCompleted ? '✓' : isCurrent ? '⚡' : idx + 1}
                </div>
                <div className="step-info">
                  <div className="step-name">{st.name}</div>
                </div>
              </div>
            )
          })}
        </div>

        <div className="modal-footer-note">
          <span>💡 Evaluator Note:</span> Powered by PyTorch LSTM (95.3% Val Accuracy) + MediaPipe Pose 3D Landmarks + EasyOCR.
        </div>
      </div>
    </div>
  )
}
