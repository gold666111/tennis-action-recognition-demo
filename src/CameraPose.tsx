import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Pose, Results } from '@mediapipe/pose'
import * as mpPose from '@mediapipe/pose'
import * as drawUtils from '@mediapipe/drawing_utils'

// Actions (fixed)
const ACTIONS = ['serve', 'forehand', 'backhand', 'volley', 'smash', 'ready']
const WINDOW_SIZE = 30 // frames

type Sample = {
  label: number
  frames: number[][] // array of flattened landmarks per frame
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

  useEffect(() => {
    startCamera()
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

    async function sendFrame() {
      if (!videoRef.current) return
      await pose.send({image: videoRef.current})
      requestAnimationFrame(sendFrame)
    }

    sendFrame()
  }

  function onResults(results: Results) {
    drawCanvas(results)
    const lm = results.poseLandmarks
    if (!lm) return

    const flat = flattenLandmarks(lm)
    // maintain sliding window
    slidingWindow.current.push(flat)
    if (slidingWindow.current.length > WINDOW_SIZE) slidingWindow.current.shift()

    // if collecting sample
    if (status === 'recording') {
      // append to current last sample frames
      setSamples(prev => {
        const copy = [...prev]
        const last = copy[copy.length - 1]
        if (last && last.frames.length < WINDOW_SIZE) {
          last.frames.push(flat)
        }
        return copy
      })
    }

    // inference if model available and window filled
    if (model && slidingWindow.current.length === WINDOW_SIZE) {
      const input = tf.tensor([slidingWindow.current.flat()])
      const pred = (model.predict(input) as tf.Tensor)
      const probs = pred.arraySync() as number[][]
      setPredictions(probs[0])
      input.dispose()
      pred.dispose()
    }
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

  function flattenLandmarks(landmarks: any[]) {
    // normalize by hip center and scale by torso size
    const xs = landmarks.map(l => l.x)
    const ys = landmarks.map(l => l.y)
    const cx = (xs[23] + xs[24]) / 2 // left/right hip
    const cy = (ys[23] + ys[24]) / 2
    const torsoSize = Math.hypot(xs[11] - xs[12], ys[11] - ys[12]) + 1e-6

    const flat = landmarks.flatMap((p: any) => [ (p.x - cx)/torsoSize, (p.y - cy)/torsoSize, p.z || 0 ])
    return flat
  }

  function startRecording() {
    // create new sample placeholder
    setSamples(prev => [...prev, {label: selectedAction, frames: []}])
    setStatus('recording')
    // stop recording after enough frames
    setTimeout(() => {
      setStatus('camera-ready')
    }, WINDOW_SIZE * 40) // ~40ms per frame -> ~WINDOW_SIZE frames
  }

  function clearSamples() {
    setSamples([])
    setPredictions(null)
    setModel(null)
  }

  async function trainModel() {
    if (samples.length === 0) return alert('No samples')
    // prepare dataset: each sample must have WINDOW_SIZE frames
    const xs: number[][] = []
    const ys: number[][] = []
    for (const s of samples) {
      if (s.frames.length !== WINDOW_SIZE) continue
      xs.push(s.frames.flat())
      const one = new Array(ACTIONS.length).fill(0)
      one[s.label] = 1
      ys.push(one)
    }
    if (xs.length === 0) return alert('No complete samples (need full windows)')

    const inputDim = xs[0].length
    const xsTensor = tf.tensor2d(xs)
    const ysTensor = tf.tensor2d(ys)

    const m = tf.sequential()
    m.add(tf.layers.dense({units: 128, activation: 'relu', inputShape: [inputDim]}))
    m.add(tf.layers.dropout({rate: 0.3}))
    m.add(tf.layers.dense({units: ACTIONS.length, activation: 'softmax'}))

    m.compile({optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['acc']})

    setStatus('training')
    await m.fit(xsTensor, ysTensor, {epochs: 30, batchSize: Math.min(16, xs.length)})
    setModel(m)
    setStatus('camera-ready')
    xsTensor.dispose(); ysTensor.dispose()
    alert('Training finished. Model ready for real-time inference.')
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
            <button onClick={clearSamples}>Clear samples</button>
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
        </section>

        <section>
          <h3>Status</h3>
          <p>{status}</p>
        </section>

        <section>
          <h3>Instructions</h3>
          <ol>
            <li>Allow camera access. Aim to show full body (or at least upper body + racket).</li>
            <li>Select an action, press Record and perform the action — the demo captures ~30 frames.</li>
            <li>Collect several samples per action (recommended 10+), then Train.</li>
            <li>After training, perform actions and watch live predictions.</li>
          </ol>
        </section>
      </div>
    </div>
  )
}
