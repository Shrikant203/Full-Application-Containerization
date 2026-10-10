#!/usr/bin/env python3
"""
train_model.py - trains the deep-learning model served by the Flask API.

A small fully connected neural network (64 -> 64 ReLU -> 10 softmax) is trained from
scratch in NumPy on the UCI handwritten-digits dataset (8x8 images, 1,797 samples) and
exported to model/model.npz. The API only needs NumPy at inference time, which keeps the
Docker image small. scikit-learn is used here only to load the dataset (training-time
dependency, listed in requirements-train.txt and NOT installed in the image).

Usage:  python train_model.py
"""
import json
import os
from datetime import date

import numpy as np
from sklearn.datasets import load_digits

SEED, EPOCHS, BATCH, LR, HIDDEN = 42, 80, 32, 0.002, 64
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model.npz")


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def forward(x, p):
    h = np.maximum(0, x @ p["W1"] + p["b1"])
    return h, softmax(h @ p["W2"] + p["b2"])


def main():
    d = load_digits()
    X, y = d.data.astype(np.float64) / 16.0, d.target
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    cut = int(len(X) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    p = {"W1": rng.normal(0, np.sqrt(2 / 64), (64, HIDDEN)), "b1": np.zeros(HIDDEN),
         "W2": rng.normal(0, np.sqrt(2 / HIDDEN), (HIDDEN, 10)), "b2": np.zeros(10)}
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(val) for k, val in p.items()}
    t = 0
    for ep in range(1, EPOCHS + 1):
        order = rng.permutation(len(Xtr))
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            xb, yb = Xtr[b], ytr[b]
            h, pr = forward(xb, p)
            dz2 = pr.copy(); dz2[np.arange(len(b)), yb] -= 1; dz2 /= len(b)
            g = {"W2": h.T @ dz2, "b2": dz2.sum(0)}
            dh = dz2 @ p["W2"].T; dh[h <= 0] = 0
            g["W1"], g["b1"] = xb.T @ dh, dh.sum(0)
            t += 1
            for k in p:  # Adam update
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                p[k] -= LR * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
        if ep % 20 == 0 or ep == 1:
            tra = (forward(Xtr, p)[1].argmax(1) == ytr).mean()
            tea = (forward(Xte, p)[1].argmax(1) == yte).mean()
            print(f"epoch {ep:3d}  train acc {tra:.4f}  test acc {tea:.4f}")

    train_acc = float((forward(Xtr, p)[1].argmax(1) == ytr).mean())
    test_acc = float((forward(Xte, p)[1].argmax(1) == yte).mean())
    params = int(sum(v_.size for v_ in p.values()))

    # one held-out test example per digit, served by GET /samples for easy verification
    sx, sy = [], []
    for digit in range(10):
        j = np.where(yte == digit)[0][0]
        sx.append(d.data[te[j]]); sy.append(digit)

    meta = {"name": "digits-mlp", "dataset": "UCI handwritten digits (sklearn load_digits), 8x8 grayscale",
            "architecture": "Input 64 -> Dense 64 + ReLU -> Dense 10 + Softmax", "parameters": params,
            "input": "64 pixel values in the range 0-16 (flattened 8x8 image)", "classes": list(range(10)),
            "train_samples": int(len(tr)), "test_samples": int(len(te)),
            "train_accuracy": round(train_acc, 4), "test_accuracy": round(test_acc, 4),
            "epochs": EPOCHS, "seed": SEED, "trained_on": str(date.today()), "numpy_version": np.__version__}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, W1=p["W1"].astype(np.float32), b1=p["b1"].astype(np.float32),
             W2=p["W2"].astype(np.float32), b2=p["b2"].astype(np.float32),
             samples_x=np.array(sx, dtype=np.float32), samples_y=np.array(sy, dtype=np.int64),
             meta=np.array(json.dumps(meta)))
    print(json.dumps(meta, indent=2))
    print("saved", OUT, f"({os.path.getsize(OUT) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
