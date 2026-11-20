"""
Run the Python pipeline (api/pipeline.py) on an EDF file and dump per-window details as JSON,
so the in-browser engine can be checked against it (tools/verify_browser.js).

Usage (from webapp/):  .venv/Scripts/python tools/reference_dump.py file.edf out.json [max_windows]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
import pipeline as pl  # noqa: E402

edf, out = sys.argv[1], sys.argv[2]
max_windows = int(sys.argv[3]) if len(sys.argv) > 3 else 40
model, scaler, _ = pl.load_model(str(ROOT / "models" / "best_model_v4.pt"), str(ROOT / "models" / "scaler.pkl"))

data, names, info = pl.load_edf(edf)
starts, total = pl.window_starts(data.shape[1], max_windows)
win = pl.WINDOW_SEC * pl.TARGET_SFREQ
windows = []
for s in starts:
    w = data[:, s:s + win]
    feats = pl.extract_node_features(w, pl.TARGET_SFREQ)
    ei = pl.build_edges(w)
    probs = pl.predict_window(model, scaler, w)
    windows.append({"start": int(s), "features": feats.astype(float).tolist(),
                    "edges": int(ei.shape[1] // 2), "probs": probs.astype(float).tolist()})
result = pl.run(edf, model, scaler, max_windows=max_windows)
Path(out).write_text(json.dumps({"names": names, "info": info, "windows": windows, "result": result}))
print(f"{len(starts)} windows, {len(names)} channels -> {out}")
