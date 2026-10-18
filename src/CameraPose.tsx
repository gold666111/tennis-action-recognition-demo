import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Pose, Results } from '@mediapipe/pose'
import * as mpPose from '@mediapipe/pose'
import * as drawUtils from '@mediapipe/drawing_utils'

// Actions (fixed)
const ACTIONS = ['serve', 'forehand', 'backhand', 'volley', 'smash', 'ready']
const WINDOW_SIZE = 30 // frames
const TARGET_FPS = 15

// selected landmark indices (MediaPipe 33-keypoints)
const LANDMARK_INDICES = [11,12,13,14,15,16,23,24,25,26,27,28]
const FEATURE_DIM_PER_POINT = 3 // x,y,z
const FEATURE_DIM = LANDMARK_INDICES.length * FEATURE_DIM_PER_POINT * 2 // position + delta

type Sample = {
  label: number
  frames: number[][] // array of feature vectors per frame
}

export default function CameraPose() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [status, setStatus] = useState('init')
  const [selectedAction, setSelectedAction] = useState(0)
  const [samples, setSamples] = useState<Sample[]>([])
  const slidingWindow = useRef<number[][]>([])
  const [model, setModel] = useState<tf.LayersModel | null>(null)
  const [predictions, setPredictions] = useState<number[] | null>(null)
  const predictionHistory = useRef<number[][]>([])
  const lastSampleTs = useRef<number>(0)
  const lastLandmarks = useRef<number[] | null>(null)
  const recordingIndex = useRef<number | null>(null) // index of current recording sample in samples array

  useEffect(() => {
    startCamera()
    // try load model from indexeddb
    (async () => {
      try {
        const m = await tf.loadLayersModel('indexeddb://tennis-model-v1')
        setModel(m)
        console.log('Loaded model from IndexedDB')
      } catch (e) {
        console.log('No model in IndexedDB')
      }
    })()
  }, [])

  async function startCamera() {
    setStatus('starting-camera')
    const video = document.createElement('video')
    video.autoplay = true
    video.playsInline = true
    video.width = 640
    video.height = 480
    videoRef.current = video

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
      video.srcObject = stream
      await video.play()
      setStatus('camera-ready')
      initPose()
    } catch (err) {
      console.error(err)
      setStatus('camera-error')
    }
  }

  function initPose() {
    const pose = new Pose({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`})
    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    })
    pose.onResults(onResults)

    async function sendFrame(ts?: number) {
      if (!videoRef.current) return
      await pose.send({image: videoRef.current})
      requestAnimationFrame(sendFrame)
    }

    requestAnimationFrame(sendFrame)
  }

  function onResults(results: Results) {
    drawCanvas(results)
    const lm = results.poseLandmarks
    if (!lm) return

    // sampling at fixed FPS
    const now = performance.now()
    if (now - lastSampleTs.current < 1000 / TARGET_FPS) {
      return
    }
    lastSampleTs.current = now

    const features = normalizeAndExtractFeatures(lm)

    // sliding window (for inference)
    slidingWindow.current.push(features)
    if (slidingWindow.current.length > WINDOW_SIZE) slidingWindow.current.shift()

    // if collecting sample
    if (status === 'recording' && recordingIndex.current !== null) {
      setSamples(prev => {
        const copy = [...prev]
        const last = copy[recordingIndex.current!]
        if (last && last.frames.length < WINDOW_SIZE) {
          last.frames.push(features)
        }
        return copy
      })
    }

    // inference if model available and window filled
    if (model && slidingWindow.current.length === WINDOW_SIZE) {
      const input = tf.tensor(slidingWindow.current).expandDims(0) // [1, WINDOW_SIZE, feat]
      const pred = (model.predict(input) as tf.Tensor)
      const probs = (pred.arraySync() as number[][])[0]
      // smoothing
      predictionHistory.current.push(probs)
      if (predictionHistory.current.length > 5) predictionHistory.current.shift()
      const avg = averagePredictions(predictionHistory.current)
      setPredictions(avg)
      input.dispose(); pred.dispose()
    }

    lastLandmarks.current = features
  }

  function drawCanvas(results: Results) {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return
    const ctx = canvas.getContext('2d')!
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    ctx.save()
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height)

    if (results.poseLandmarks) {
      drawUtils.drawConnectors(ctx, results.poseLandmarks, mpPose.POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 2})
      drawUtils.drawLandmarks(ctx, results.poseLandmarks, {color: '#FF0000', lineWidth: 1})
    }
    ctx.restore()
  }

  function normalizeAndExtractFeatures(landmarks: any[]) {
    // select important points
    const pts = LANDMARK_INDICES.map(i => landmarks[i] || {x:0,y:0,z:0})
    const hip = { x: ((landmarks[23]?.x||0) + (landmarks[24]?.x||0)) / 2, y: ((landmarks[23]?.y||0) + (landmarks[24]?.y||0)) / 2 }
    const lShoulder = landmarks[11] || {x:0,y:0}
    const rShoulder = landmarks[12] || {x:0,y:0}
    const angle = Math.atan2(lShoulder.y - rShoulder.y, lShoulder.x - rShoulder.x)
    const cos = Math.cos(-angle), sin = Math.sin(-angle)
    const shoulderDist = Math.hypot(lShoulder.x - rShoulder.x, lShoulder.y - rShoulder.y) + 1e-6

    const pos: number[] = []
    for (const p of pts) {
      const dx = p.x - hip.x
      const dy = p.y - hip.y
      const rx = dx * cos - dy * sin
      const ry = dx * sin + dy * cos
      pos.push(rx / shoulderDist, ry / shoulderDist, p.z || 0)
    }

    // delta to previous frame
    const delta: number[] = []
    if (lastLandmarks.current) {
      for (let i=0;i<pos.length;i++) {
        delta.push(pos[i] - lastLandmarks.current[i])
      }
    } else {
      for (let i=0;i<pos.length;i++) delta.push(0)
    }

    return [...pos, ...delta]
  }

  function averagePredictions(arr: number[][]) {
    if (arr.length === 0) return null
    const sum = new Array(arr[0].length).fill(0)
    for (const a of arr) for (let i=0;i<a.length;i++) sum[i]+=a[i]
    return sum.map(s => s/arr.length)
  }

  function startRecording() {
    // create new sample placeholder
    const idx = samples.length
    setSamples(prev => [...prev, {label: selectedAction, frames: []}])
    recordingIndex.current = idx
    setStatus('recording')

    // stop recording when window full or after timeout
    const maxTime = Math.ceil((WINDOW_SIZE / TARGET_FPS) * 1000 + 500)
    setTimeout(() => {
      setStatus('camera-ready')
      recordingIndex.current = null
    }, maxTime)
  }

  function clearSamples() {
    setSamples([])
    setPredictions(null)
    setModel(null)
    predictionHistory.current = []
    // remove saved model
    tf.io.removeModel('indexeddb://tennis-model-v1').catch(()=>{})
  }

  async function trainModel() {
    if (samples.length === 0) return alert('No samples')
    // prepare dataset: each sample must have WINDOW_SIZE frames
    const xs: number[][] = []
    const ys: number[][] = []
    for (const s of samples) {
      if (s.frames.length !== WINDOW_SIZE) continue
      // s.frames is array of [feat] per frame; produce [WINDOW_SIZE, feat]
      xs.push(s.frames.flat())
      const one = new Array(ACTIONS.length).fill(0)
      one[s.label] = 1
      ys.push(one)
    }
    if (xs.length === 0) return alert('No complete samples (need full windows)')

    const featureDim = samples[0].frames[0].length
    const xvals = new Float32Array(xs.length * WINDOW_SIZE * featureDim)
    for (let i=0;i<xs.length;i++) {
      const flat = xs[i]
      for (let j=0;j<flat.length;j++) xvals[i * flat.length + j] = flat[j]
    }
    const xsTensor = tf.tensor(xvals, [xs.length, WINDOW_SIZE, featureDim])
    const ysTensor = tf.tensor2d(ys)

    const m = createConv1DModel(WINDOW_SIZE, featureDim, ACTIONS.length)

    setStatus('training')
    await m.fit(xsTensor, ysTensor, {epochs: 40, batchSize: Math.min(16, xs.length)})
    setModel(m)
    setStatus('camera-ready')
    xsTensor.dispose(); ysTensor.dispose()

    // save to indexeddb
    await m.save('indexeddb://tennis-model-v1')
    alert('Training finished. Model saved to IndexedDB and ready for real-time inference.')
  }

  function createConv1DModel(windowSize: number, featureDim: number, numClasses: number) {
    const model = tf.sequential()
    model.add(tf.layers.inputLayer({inputShape: [windowSize, featureDim]}))
    model.add(tf.layers.conv1d({filters: 32, kernelSize: 3, activation: 'relu'}))
    model.add(tf.layers.conv1d({filters: 64, kernelSize: 3, activation: 'relu'}))
    model.add(tf.layers.globalAveragePooling1d())
    model.add(tf.layers.dropout({rate: 0.3}))
    model.add(tf.layers.dense({units: numClasses, activation: 'softmax'}))
    model.compile({optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['acc']})
    return model
  }

  function exportSamples() {
    const data = JSON.stringify({window: WINDOW_SIZE, fps: TARGET_FPS, samples})
    const blob = new Blob([data], {type: 'application/json'})
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'tennis_samples.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  function importSamples(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    const reader = new FileReader()
    reader.onload = () => {
      try {
        const parsed = JSON.parse(reader.result as string)
        if (parsed.samples) setSamples(parsed.samples)
        alert('Samples imported')
      } catch (err) { alert('Invalid file') }
    }
    reader.readAsText(f)
  }

  async function loadModelFromFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    const url = URL.createObjectURL(f)
    try {
      const m = await tf.loadLayersModel(url)
      setModel(m)
      await m.save('indexeddb://tennis-model-v1')
      alert('Model loaded and saved to IndexedDB')
    } catch (err) {
      alert('Failed to load model')
    } finally {
      URL.revokeObjectURL(url)
    }
  }

  return (
    <div className="camera-pose">
      <div className="left">
        <div className="video-wrap">
          <video style={{display: 'none'}} ref={videoRef}></video>
          <canvas ref={canvasRef}></canvas>
        </div>
        <div className="predictions">
          <h3>Live prediction</h3>
          {predictions ? (
            <ul>
              {ACTIONS.map((a, i) => (
                <li key={a}>{a}: {(predictions[i] * 100).toFixed(1)}%</li>
              ))}
            </ul>
          ) : <p>—</p>}
          {predictions && (function(){
            const maxIdx = predictions.indexOf(Math.max(...predictions))
            const conf = predictions[maxIdx]
            if (conf > 0.6) return <div><strong>Detected:</strong> {ACTIONS[maxIdx]} ({(conf*100).toFixed(0)}%)</div>
            return <div><em>Confidence too low</em></div>
          })()}
        </div>
      </div>

      <div className="right">
        <section>
          <h3>Action selection</h3>
          <select value={selectedAction} onChange={e => setSelectedAction(Number(e.target.value))}>
            {ACTIONS.map((a,i) => <option key={a} value={i}>{a}</option>)}
          </select>
          <div className="controls">
            <button onClick={startRecording}>Record sample (auto stop after window)</button>
            <button onClick={trainModel}>Train in browser</button>
            <button onClick={clearSamples}>Clear samples & remove model</button>
          </div>
        </section>

        <section>
          <h3>Samples</h3>
          <p>Total samples: {samples.length}</p>
          <ol>
            {samples.map((s,idx) => (
              <li key={idx}>{ACTIONS[s.label]} — {s.frames.length}/{WINDOW_SIZE} frames</li>
            ))}
          </ol>
          <div style={{marginTop:8}}>
            <button onClick={exportSamples}>Export samples (JSON)</button>
            <input type="file" accept="application/json" onChange={importSamples} />
          </div>

          <div style={{marginTop:8}}>
            <label>Load TF.js model (.json): </label>
            <input type="file" accept="application/json" onChange={loadModelFromFile} />
          </div>
        </section>

        <section>
          <h3>Status</h3>
          <p>{status}</p>
        </section>

        <section>
          <h3>Instructions</h3>
          <ol>
            <li>Allow camera access. Aim to show full body (or at least upper body + racket).</li>
            <li>Select an action, press Record and perform the action — the demo captures ~{WINDOW_SIZE} frames at {TARGET_FPS}fps.</li>
            <li>Collect several samples per action (recommended 10+), then Train.</li>
            <li>After training, perform actions and watch live predictions. Model persists in browser IndexedDB.</li>
          </ol>
        </section>
      </div>
    </div>
  )
}
