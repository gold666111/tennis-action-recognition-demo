# Tennis Action Recognition Demo

This is a first-version, in-browser tennis action recognition demo.

Features
- Real-time webcam capture
- MediaPipe Pose landmarks extraction
- In-browser sample collection (record ~30-frame windows per action)
- Train a small TensorFlow.js model in the browser
- Real-time inference using a sliding window

Actions (default): serve, forehand, backhand, volley, smash, ready

Quick start (local)

1. Clone

   git clone https://github.com/gold666111/tennis-action-recognition-demo.git
   cd tennis-action-recognition-demo

2. Install

   npm install

3. Run

   npm run dev

4. Open the page (usually http://localhost:5173) on a laptop or phone browser that allows camera access.

Usage
- Select an action, press Record to capture one sample (~30 frames). Repeat to collect multiple samples per action.
- When you have several samples across classes, press Train to train a small model in the browser.
- After training, live predictions should appear while you perform actions.

Notes & next steps
- This is a minimal first version. Next improvements:
  - Add example dataset and pre-trained model for immediate demo without collecting samples
  - Improve preprocessing (rotation normalization / scale invariance)
  - Use a recurrent or temporal model (1D conv / LSTM / Transformer)
  - Add server-side training & model hosting for persistence

