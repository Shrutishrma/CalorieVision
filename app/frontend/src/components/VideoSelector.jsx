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
  forceRecompute,
  setForceRecompute,
  handleAnalyze,
  analyzing,
  backendStatus,
}) {
  const isCustomUrlSelected =
    customUrl.trim() &&
    !manifest.find((m) => m.youtube_id === selectedVideoId && customUrl.includes(m.youtube_id))

  return (
    <div className="panel-grid">
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">📹 Workout Video Selector</h2>
          <p className="card-subtitle">Choose a pre-tagged demo video or paste a custom YouTube link</p>
        </div>

        <div className="form-group">
          <label className="form-label">Demo Catalogue Video ({manifest.length} Available)</label>
          <select
            className="input-select"
            value={selectedVideoId}
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
          <label className="form-label">Or Paste Direct YouTube URL</label>
          <input
            type="text"
            className="input-text"
            placeholder="https://youtu.be/..."
            value={customUrl}
            onChange={(e) => {
              const val = e.target.value
              setCustomUrl(val)
              let extractedId = ''
              if (val.includes('v=')) extractedId = val.split('v=')[1].split('&')[0]
              else if (val.includes('youtu.be/')) extractedId = val.split('youtu.be/')[1].split('?')[0]
              if (extractedId) {
                const match = manifest.find((m) => m.youtube_id === extractedId)
                if (match) setSelectedVideoId(match.youtube_id)
              }
            }}
          />
        </div>

        {isCustomUrlSelected && (
          <div className="form-group custom-duration-group">
            <label className="form-label">
              📏 Video Duration (minutes)
              <span className="duration-hint">Required for uncached URLs — how long is the workout?</span>
            </label>
            <div className="duration-input-row">
              <input
                id="video-duration-input"
                type="number"
                className="input-text duration-input"
                placeholder="e.g. 20"
                min="1"
                max="180"
                step="1"
                value={videoDurationMins}
                onChange={(e) => setVideoDurationMins(e.target.value)}
              />
              <span className="duration-unit">min</span>
            </div>
            {!videoDurationMins && (
              <p className="duration-warning">⚠️ Without a duration the pipeline defaults to 30 seconds</p>
            )}
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">⚙️ Parameters &amp; Execution</h2>
          <p className="card-subtitle">Personalise MET body weight, tier &amp; cache options</p>
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
                className={`tier-btn ${userTier === tier ? 'active' : ''}`}
                onClick={() => setUserTier(tier)}
              >
                {tier.charAt(0).toUpperCase() + tier.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <input
            type="checkbox"
            id="force-recompute"
            checked={forceRecompute}
            onChange={(e) => setForceRecompute(e.target.checked)}
          />
          <label htmlFor="force-recompute" style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Force re-run pipeline (ignore cached keypoints/scenes/OCR)
          </label>
        </div>

        <button
          className="btn-primary"
          onClick={handleAnalyze}
          disabled={analyzing || backendStatus !== 'ok'}
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
