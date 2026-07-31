import React from 'react'

export default function Header({ backendStatus, activeBackendUrl }) {
  return (
    <header className="app-header">
      <div className="brand">
        <span className="brand-icon">🏋️‍♂️</span>
        <div>
          <h1 className="brand-title">CalorieVision</h1>
          <p className="brand-tagline">AI-Powered Exercise Recognition &amp; MET Calorie Estimation</p>
        </div>
      </div>

      <div className="header-status">
        <div className={`status-chip ${backendStatus}`}>
          <span className="dot" />
          <span>
            {backendStatus === 'ok'
              ? 'Backend Connected'
              : backendStatus === 'loading'
              ? 'Checking API…'
              : 'API Offline'}
          </span>
        </div>
        <a href={`${activeBackendUrl}/docs`} target="_blank" rel="noreferrer" className="docs-link">
          Swagger Docs ↗
        </a>
      </div>
    </header>
  )
}
