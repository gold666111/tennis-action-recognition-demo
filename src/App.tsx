import React from 'react'
import CameraPose from './CameraPose'

export default function App() {
  return (
    <div className="app">
      <header>
        <h1>Tennis Action Recognition — Demo (Webcam)</h1>
        <p>Real-time pose extraction (MediaPipe) + in-browser training & TF.js inference</p>
      </header>
      <main>
        <CameraPose />
      </main>
      <footer>
        <small>Repository: gold666111/tennis-action-recognition-demo</small>
      </footer>
    </div>
  )
}
