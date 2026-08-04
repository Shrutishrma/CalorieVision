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

export default function VideoSelector({
  manifest,
  selectedVideoId,
  setSelectedVideoId,
  customUrl,
  setCustomUrl,
  videoDurationMins,
  setVideoDurationMins,
  weightKg,
  setWeightKg,
  userTier,
  setUserTier,
  handleAnalyze,
  analyzing,
  backendStatus,
}) {
  const isCustomUrlEntered = Boolean(customUrl && customUrl.trim().length > 5)

  return (
    <div className="panel-grid">
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">📹 Workout Video Selector</h2>
          <p className="card-subtitle">Select a preset catalogue video or paste any YouTube URL / Short</p>
        </div>

        <div className="form-group">
          <label className="form-label">Demo Workout Catalogue ({manifest.length} Videos Available)</label>
          <select
            className="input-select"
            value={selectedVideoId}
            disabled={isCustomUrlEntered}
            onChange={(e) => {
              setSelectedVideoId(e.target.value)
              setCustomUrl('')
            }}
          >
            {manifest.map((item) => (
              <option key={item.youtube_id} value={item.youtube_id}>
                {item.description || item.label}
                {item.duration_secs ? ` · ${fmtDuration(item.duration_secs)}` : ''}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Or Paste Direct YouTube URL / Short</label>
          <input
            type="text"
            className="input-text"
            placeholder="https://www.youtube.com/watch?v=... or https://youtube.com/shorts/..."
            value={customUrl}
            onChange={(e) => {
              const val = e.target.value
              setCustomUrl(val)
              let extractedId = ''
              if (val.includes('v=')) extractedId = val.split('v=')[1].split('&')[0]
              else if (val.includes('youtu.be/')) extractedId = val.split('youtu.be/')[1].split('?')[0]
              else if (val.includes('shorts/')) extractedId = val.split('shorts/')[1].split('?')[0]

              if (extractedId) {
                const match = manifest.find((m) => m.youtube_id === extractedId)
                if (match) setSelectedVideoId(match.youtube_id)
              }
            }}
          />
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">⚙️ Personalization &amp; Execution</h2>
          <p className="card-subtitle">Configure body weight and workout intensity tier</p>
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
            {['beginner', 'intermediate', 'advanced'].map((tier) => (
              <button
                key={tier}
                type="button"
                className={`tier-btn ${userTier === tier ? 'active' : ''}`}
                onClick={() => setUserTier(tier)}
              >
                {tier.charAt(0).toUpperCase() + tier.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <button
          type="button"
          className="btn-primary"
          onClick={handleAnalyze}
          disabled={analyzing}
          style={{ marginTop: '1rem' }}
        >
          {analyzing ? (
            <>
              <span className="spinner-icon">⚡</span> Executing Pipeline…
            </>
          ) : (
            <>🚀 Analyse Workout Video</>
          )}
        </button>
      </div>
    </div>
  )
}
