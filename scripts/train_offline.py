#!/usr/bin/env python3
# scripts/train_offline.py
# Example script: train a small model using exported samples JSON

import json
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()

with open(args.data,'r') as f:
    data = json.load(f)

samples = data.get('samples', [])
window = data.get('window', 30)

X = []
Y = []
for s in samples:
    if len(s['frames']) != window: continue
    X.append(np.array(s['frames']))
    y = np.zeros((6,))
    y[s['label']] = 1
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

print('Loaded', X.shape, Y.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window, X.shape[2])),
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=16)

model.save(args.out)
print('Saved model to', args.out)
