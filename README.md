# Tennis Action Recognition Demo

This repository contains a browser-based tennis action recognition demo.

What's new in this improved version
- Fixed sampling FPS and stable windowing
- Better landmark normalization (translation + rotation + scale)
- Uses a small 1D-CNN (conv1d) for temporal modeling
- Prediction smoothing and confidence thresholding
- Model persistence: saved/loaded from IndexedDB (tfjs) and file load option
- Export/import collected samples for offline training

Quick start
1. Install
   npm install
2. Run dev server
   npm run dev
3. Open http://localhost:5173 on your device and allow camera access

Training & demo
- Use the UI to record samples (recommended 10+ per action) and train in browser.
- After training the model is saved to IndexedDB automatically.
- You can export collected samples (JSON) and train a larger model offline (see scripts/)

Offline training
- If you want to train on desktop/server with more data, export samples to JSON and run a Python training script (examples in scripts/). After training convert the SavedModel to TF.js format:

  pip install tensorflow tensorflowjs
  python scripts/train_offline.py --data exported_samples.json --out saved_model
  tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_layers_model saved_model/web_model saved_model_tfjs

- Upload the generated web_model (model.json + shard .bin) to the web app using the "Load TF.js model" control.

