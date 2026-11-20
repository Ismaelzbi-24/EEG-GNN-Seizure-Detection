"""
Export GAT v4 + scaler + band-pass filter for the in-browser version of the site.

Writes:
  site/model/model.bin   all weights as little-endian float32, back to back
  site/model/model.json  where each tensor sits in model.bin, plus the scaler and the
                         exact FIR filter MNE uses for raw.filter(1, 300) at 1000 Hz

Usage (from webapp/):  .venv/Scripts/python tools/export_browser_model.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import mne

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
import pipeline  # noqa: E402

model_path = ROOT / "models" / "best_model_v4.pt"
scaler_path = ROOT / "models" / "scaler.pkl"
out_dir = ROOT / "site" / "model"
out_dir.mkdir(parents=True, exist_ok=True)

ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

tensors, blobs, offset = {}, [], 0
for name, t in state.items():
    if name.endswith("num_batches_tracked"):
        continue
    arr = t.detach().cpu().numpy().astype("<f4").ravel()
    tensors[name] = {"offset": offset, "shape": list(t.shape)}
    blobs.append(arr.tobytes())
    offset += arr.size

# Same filter raw.filter(1.0, 300.0, fir_design="firwin") builds after resampling to 1000 Hz.
h = mne.filter.create_filter(None, float(pipeline.TARGET_SFREQ), 1.0, 300.0,
                             fir_design="firwin", verbose=False)

meta = {
    "name": "GAT v4",
    "fold": float(ckpt.get("fold", 0)), "epoch": float(ckpt.get("epoch", 0)),
    "f1": float(ckpt.get("f1", 0)),
    "arch": {"in": 18, "hidden": 128, "heads": 4, "negative_slope": 0.2, "bn_eps": 1e-5},
    "tensors": tensors,
    "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
    "filter": h.tolist(),
    "params": {"target_sfreq": pipeline.TARGET_SFREQ, "window_sec": pipeline.WINDOW_SEC,
               "step_sec": pipeline.STEP_SEC, "corr_thresh": pipeline.CORR_THRESH,
               "max_channels": pipeline.MAX_CHANNELS},
}
(out_dir / "model.bin").write_bytes(b"".join(blobs))
(out_dir / "model.json").write_text(json.dumps(meta))
print(f"wrote {offset} weights ({offset * 4 / 1e6:.2f} MB), filter taps {len(h)}")
