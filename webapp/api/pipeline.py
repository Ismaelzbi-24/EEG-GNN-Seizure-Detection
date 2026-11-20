"""
EEG-GNN inference pipeline (GAT v4).

Preprocessing is copied from code/01_single_patient.py so that inference sees
exactly what the model saw in training:
  resample -> 1000 Hz, band-pass 1-300 Hz, drop flat/NaN channels, keep first 64,
  60 s windows with a 30 s step, 18 features per channel, |Pearson r| > 0.3 edges,
  StandardScaler from training, clipped to [-10, 10].
The model is the same GATv4 used in code/04_train_recall_gat.py / ieegnet_app.py.
"""
import os
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy.stats import kurtosis, skew
from torch_geometric.nn import GATConv, global_mean_pool

warnings.filterwarnings("ignore")

TARGET_SFREQ = 1000
WINDOW_SEC = 60
STEP_SEC = 30
CORR_THRESH = 0.3
MAX_CHANNELS = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── MODEL (identical to training) ───────────────────────────────────────────
class GATv4(nn.Module):
    def __init__(self, in_channels=18, hidden=128, heads=4, dropout=0.25):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden * heads, hidden, heads=heads, dropout=dropout)
        self.gat3 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout)
        self.skip1 = nn.Linear(in_channels, hidden)
        self.skip2 = nn.Linear(hidden * heads, hidden)
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.bn2 = nn.BatchNorm1d(hidden * heads)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, x, ei, batch):
        s1 = self.skip1(x)
        x = F.elu(self.bn1(self.gat1(x, ei)))
        x = F.dropout(x, self.dropout, self.training)
        s2 = self.skip2(x)
        x = F.elu(self.bn2(self.gat2(x, ei)))
        x = F.dropout(x, self.dropout, self.training)
        x = F.elu(self.bn3(self.gat3(x, ei)) + s2 + s1)
        gc = global_mean_pool(x, batch)[batch]
        return self.classifier(torch.cat([x, gc], dim=1))


def load_model(model_path, scaler_path):
    ckpt = torch.load(model_path, weights_only=False, map_location=DEVICE)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = GATv4().to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    meta = {k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in (ckpt.items() if isinstance(ckpt, dict) else [])
            if k in ("fold", "epoch", "f1")}
    return model, scaler, meta


# ── FEATURES (identical to code/01_single_patient.py) ─────────────────────────
def _bandpower(sig, fs, fmin, fmax):
    f, psd = signal.welch(sig, fs=fs, nperseg=min(fs, len(sig)))
    idx = np.logical_and(f >= fmin, f <= fmax)
    return float(np.trapz(psd[idx], f[idx])) if idx.any() else 0.0


def extract_node_features(window, fs):
    feats = []
    for ch in window:
        f, psd = signal.welch(ch, fs=fs, nperseg=min(fs, len(ch)))
        total_p = float(np.trapz(psd, f)) + 1e-10
        psd_norm = psd / (total_p + 1e-10)
        psd_norm = psd_norm[psd_norm > 0]
        feats.append([
            float(np.mean(ch)), float(np.std(ch)), float(np.var(ch)),
            float(skew(ch)), float(kurtosis(ch)),
            float(np.sqrt(np.mean(ch ** 2))),
            float(np.sum(np.abs(np.diff(ch)))),
            float(np.sum(np.diff(np.sign(ch)) != 0)),
            _bandpower(ch, fs, 0.5, 4), _bandpower(ch, fs, 4, 8),
            _bandpower(ch, fs, 8, 13), _bandpower(ch, fs, 13, 30),
            _bandpower(ch, fs, 30, 80), _bandpower(ch, fs, 80, 150),
            _bandpower(ch, fs, 80, 300),
            total_p, float(f[np.argmax(psd)]),
            float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10))),
        ])
    return np.array(feats, dtype=np.float32)


def build_edges(window):
    n = window.shape[0]
    corr = np.nan_to_num(np.corrcoef(window), nan=0.0)
    iu, ju = np.triu_indices(n, k=1)
    keep = np.abs(corr[iu, ju]) > CORR_THRESH
    a, b = iu[keep], ju[keep]
    if a.size == 0:
        a = np.arange(n)
        b = (a + 1) % n
    src = np.stack([a, b], axis=1).ravel()
    dst = np.stack([b, a], axis=1).ravel()
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


# ── RUN ──────────────────────────────────────────────────────────────────────
class PipelineError(Exception):
    """A problem with the uploaded file that the user should see."""


def load_edf(path):
    import mne
    mne.set_log_level("ERROR")
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    except Exception as e:
        raise PipelineError(f"Could not read this file as EDF: {e}")
    orig_fs = float(raw.info["sfreq"])
    if orig_fs < 2 * 300:
        # The 1-300 Hz filter and 80-300 Hz features need fs >= 600 Hz.
        raise PipelineError(
            f"Sampling rate is {orig_fs:g} Hz. The model needs at least 600 Hz "
            "(intracranial EEG is usually 500-2000 Hz; scalp EEG is too slow).")
    if orig_fs != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ)
    raw.filter(1.0, 300.0, fir_design="firwin", verbose=False)
    data = raw.get_data()
    names = raw.ch_names
    valid = [i for i, ch in enumerate(data)
             if not np.all(np.isnan(ch)) and np.std(ch) > 1e-10]
    if len(valid) < 3:
        raise PipelineError("Fewer than 3 usable channels in this recording.")
    total_valid = len(valid)
    valid = valid[:MAX_CHANNELS]
    return data[valid], [names[i] for i in valid], {
        "original_sfreq": orig_fs,
        "duration_sec": round(data.shape[1] / TARGET_SFREQ, 1),
        "channels_total": len(names),
        "channels_usable": total_valid,
        "channels_used": len(valid),
    }


def window_starts(n_samples, max_windows):
    win, step = WINDOW_SEC * TARGET_SFREQ, STEP_SEC * TARGET_SFREQ
    if n_samples < win:
        raise PipelineError(f"Recording is shorter than {WINDOW_SEC} seconds.")
    # Same range as training: range(0, total - win, step); keep at least one window.
    starts = list(range(0, n_samples - win, step)) or [0]
    total = len(starts)
    if max_windows and len(starts) > max_windows:
        # Spread the windows evenly across the whole recording.
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in sorted(set(idx))]
    return starts, total


@torch.no_grad()
def predict_window(model, scaler, window):
    x = np.clip(scaler.transform(extract_node_features(window, TARGET_SFREQ)), -10, 10)
    x = torch.tensor(x, dtype=torch.float, device=DEVICE)
    ei = build_edges(window).to(DEVICE)
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=DEVICE)
    return torch.softmax(model(x, ei, batch), dim=1)[:, 1].cpu().numpy()


def run(path, model, scaler, max_windows=40, progress=lambda done, total, stage: None):
    progress(0, 1, "Reading and filtering the recording")
    data, names, info = load_edf(path)
    starts, total_windows = window_starts(data.shape[1], max_windows)
    win = WINDOW_SEC * TARGET_SFREQ
    per_window = []
    for k, s in enumerate(starts):
        progress(k, len(starts), f"Analysing minute-window {k + 1} of {len(starts)}")
        per_window.append(predict_window(model, scaler, data[:, s:s + win]))
    progress(len(starts), len(starts), "Done")
    P = np.stack(per_window)  # windows x channels
    mean_p = P.mean(axis=0)
    channels = [{
        "name": n,
        "prob": round(float(mean_p[i]), 4),
        "max_prob": round(float(P[:, i].max()), 4),
        "timeline": [round(float(v), 3) for v in P[:, i]],
    } for i, n in enumerate(names)]
    channels.sort(key=lambda c: c["prob"], reverse=True)
    info.update({
        "windows_total": total_windows,
        "windows_analysed": len(starts),
        "window_starts_sec": [s / TARGET_SFREQ for s in starts],
    })
    return {"channels": channels, "info": info}
