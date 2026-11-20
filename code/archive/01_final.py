"""
================================================
Omni-iEEG Dataset Processor
Task: SOZ (Seizure Onset Zone) Channel Detection
Dataset: 302 patients, BIDS format EDF files
================================================
Ye script kya karta hai:
1. Har patient ke EDF files padhta hai
2. channels.tsv se SOZ labels leta hai
3. 1-minute windows banata hai (paper ke mutabiq)
4. Node features extract karta hai
5. Graph banata hai (channels = nodes)
6. PyTorch Geometric format mein save karta hai
"""
import os
import warnings
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import mne
from scipy import signal
from torch_geometric.data import Data
from tqdm import tqdm
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')
# ==========================================
# CONFIG — apna path yahan set karo
# ==========================================
DATASET_PATH = "/home/pmilab/EEG_GNN_Project/Hiroki/Data/Data"
OUTPUT_FILE  = "/home/pmilab/EEG_GNN_Project/code/omni_ieeg_graphs.pt"
META_FILE    = "/home/pmilab/EEG_GNN_Project/code/omni_ieeg_meta.csv"
TARGET_SFREQ  = 1000    # Hz — paper ke mutabiq resample
WINDOW_SEC    = 60      # 1 minute window — paper ke mutabiq
STEP_SEC      = 30      # 50% overlap
CORR_THRESH   = 0.3     # edge threshold
MAX_CHANNELS  = 64      # memory limit

print("=" * 55)
print("  Omni-iEEG Graph Dataset Builder")
print("=" * 55)
print(f"  Dataset path : {DATASET_PATH}")
print(f"  Window size  : {WINDOW_SEC}s")
print(f"  Target sfreq : {TARGET_SFREQ}Hz")
print(f"  Output file  : {OUTPUT_FILE}")
print("=" * 55)

# ==========================================
# BAND POWER FEATURE
# ==========================================
def bandpower(sig, fs, fmin, fmax):
    """Ek frequency band ki power nikalo"""
    f, psd = signal.welch(sig, fs=fs, nperseg=min(fs, len(sig)))
    idx = np.logical_and(f >= fmin, f <= fmax)
    return float(np.trapz(psd[idx], f[idx])) if idx.any() else 0.0

# ==========================================
# NODE FEATURE EXTRACTION
# ==========================================
def extract_node_features(window, fs):
    from scipy.stats import skew, kurtosis as kurt
    features = []
    for ch_sig in window:
        mean_v   = float(np.mean(ch_sig))
        std_v    = float(np.std(ch_sig))
        var_v    = float(np.var(ch_sig))
        skew_v   = float(skew(ch_sig))
        kurt_v   = float(kurt(ch_sig))
        rms_v    = float(np.sqrt(np.mean(ch_sig ** 2)))
        ll_v     = float(np.sum(np.abs(np.diff(ch_sig))))
        zc_v     = float(np.sum(np.diff(np.sign(ch_sig)) != 0))
        delta_p  = bandpower(ch_sig, fs, 0.5, 4)
        theta_p  = bandpower(ch_sig, fs, 4,   8)
        alpha_p  = bandpower(ch_sig, fs, 8,  13)
        beta_p   = bandpower(ch_sig, fs, 13, 30)
        gamma_p  = bandpower(ch_sig, fs, 30, 80)
        hgamma_p = bandpower(ch_sig, fs, 80, 150)
        hfo_p    = bandpower(ch_sig, fs, 80, 300)
        f, psd = signal.welch(ch_sig, fs=fs, nperseg=min(fs, len(ch_sig)))
        total_p  = float(np.trapz(psd, f)) + 1e-10
        peak_f   = float(f[np.argmax(psd)])
        psd_norm = psd / (total_p + 1e-10)
        psd_norm = psd_norm[psd_norm > 0]
        sp_ent   = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))
        feat = [
            mean_v, std_v, var_v, skew_v, kurt_v,
            rms_v, ll_v, zc_v,
            delta_p, theta_p, alpha_p, beta_p,
            gamma_p, hgamma_p, hfo_p,
            total_p, peak_f, sp_ent
        ]
        features.append(feat)
    return np.array(features, dtype=np.float32)

# ==========================================
# GRAPH CONSTRUCTION
# ==========================================
def build_graph(window, fs):
    n_ch = window.shape[0]
    corr = np.corrcoef(window)
    corr = np.nan_to_num(corr, nan=0.0)
    src, dst = [], []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            if abs(corr[i, j]) > CORR_THRESH:
                src.extend([i, j])
                dst.extend([j, i])
    if len(src) == 0:
        for i in range(n_ch):
            src.extend([i, (i+1) % n_ch])
            dst.extend([(i+1) % n_ch, i])
    return torch.tensor([src, dst], dtype=torch.long)

# ==========================================
# FIND SOZ CHANNELS FROM TSV
# ==========================================
def get_soz_labels(channels_tsv_path, ch_names):
    labels = {ch: 0 for ch in ch_names}
    try:
        df = pd.read_csv(channels_tsv_path, sep='\t')
        name_col = None
        for c in ['name', 'channel', 'ch_name', 'Name']:
            if c in df.columns:
                name_col = c
                break
        soz_col = None
        for c in ['status', 'soz', 'SOZ', 'seizure_onset_zone', 'type']:
            if c in df.columns:
                soz_col = c
                break
        if name_col and soz_col:
            for _, row in df.iterrows():
                ch = str(row[name_col]).strip()
                status = str(row[soz_col]).strip().lower()
                is_soz = any(s in status for s in [
                    'soz', 'bad', 'seizure', 'onset', 'patholog', '1', 'yes'
                ])
                if ch in labels:
                    labels[ch] = 1 if is_soz else 0
    except Exception:
        pass
    return labels

# ==========================================
# PROCESS ONE PATIENT
# ==========================================
def process_patient(patient_dir, patient_id, dataset_source):
    graphs = []
    ieeg_dir = patient_dir / "ieeg"
    if not ieeg_dir.exists():
        for sub in patient_dir.rglob("ieeg"):
            ieeg_dir = sub
            break
    if not ieeg_dir.exists():
        return []
    edf_files = list(ieeg_dir.glob("*.edf"))
    if not edf_files:
        return []
    for edf_path in edf_files:
        try:
            tsv_path = edf_path.with_name(edf_path.stem + "_channels.tsv")
            if not tsv_path.exists():
                tsv_files = list(ieeg_dir.glob("*channels.tsv"))
                tsv_path = tsv_files[0] if tsv_files else None
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            fs = raw.info['sfreq']
            if fs != TARGET_SFREQ:
                raw.resample(TARGET_SFREQ)
                fs = TARGET_SFREQ
            raw.filter(1., 300., fir_design='firwin', verbose=False)
            data = raw.get_data()
            ch_names = raw.ch_names
            valid_idx = []
            for i, ch in enumerate(data):
                if not np.all(np.isnan(ch)) and np.std(ch) > 1e-10:
                    valid_idx.append(i)
            if len(valid_idx) < 3:
                continue
            valid_idx = valid_idx[:MAX_CHANNELS]
            data = data[valid_idx]
            ch_names = [ch_names[i] for i in valid_idx]
            n_ch = len(ch_names)
            if tsv_path and Path(tsv_path).exists():
                soz_map = get_soz_labels(Path(tsv_path), ch_names)
            else:
                soz_map = {ch: 0 for ch in ch_names}
            node_labels = torch.tensor(
                [soz_map[ch] for ch in ch_names], dtype=torch.long
            )
            win_size  = int(WINDOW_SEC * fs)
            step_size = int(STEP_SEC * fs)
            total_samples = data.shape[1]
            if total_samples < win_size:
                continue
            for start in range(0, total_samples - win_size, step_size):
                window = data[:, start:start + win_size]
                node_feats = extract_node_features(window, int(fs))
                edge_index = build_graph(window, int(fs))
                graph_label = int(node_labels.sum().item() > 0)
                graph = Data(
                    x          = torch.tensor(node_feats, dtype=torch.float),
                    edge_index = edge_index,
                    y          = torch.tensor([graph_label], dtype=torch.long),
                    node_y     = node_labels,
                    patient_id = patient_id,
                    dataset    = dataset_source,
                    n_channels = n_ch,
                )
                graphs.append(graph)
        except Exception as e:
            continue
    return graphs

# ==========================================
# MAIN — ALL PATIENTS
# ==========================================
def main():
    dataset_path = Path(DATASET_PATH)

    # ---- SKIP LIST — corrupted/bad patients ----
    SKIP_PATIENTS = [
        'sub-Multicenterjh102',  # corrupted EDF — silent segfault
        'sub-Multicenterjh107',  # no valid data
        'sub-Multicenterpt1',    # no valid data
        'sub-Multicenterumf002', # no valid data
        'sub-Multicenterumf003', # no valid data
        'sub-Multicenterumf004', # no valid data
        'sub-Multicenterumf005', # no valid data
        'sub-hupHUP112',         # no valid data
        'sub-hupHUP180',         # corrupted EDF — silent segfault
    ]

    patient_dirs = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir()
        and d.name.startswith('sub-')
        and d.name not in SKIP_PATIENTS
    ])

    print(f"\nTotal patients found : {len(patient_dirs)}")
    print(f"Skipping             : {len(SKIP_PATIENTS)} known bad patients")

    all_graphs = []
    meta_rows  = []
    failed     = []

    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_id = patient_dir.name

        if 'zurich' in patient_id.lower():
            source = 'zurich'
        elif 'openieeg' in patient_id.lower() or 'multicenter' in patient_id.lower():
            source = 'openieeg'
        elif 'sourcesink' in patient_id.lower():
            source = 'sourcesink'
        elif 'hup' in patient_id.lower():
            source = 'hup'
        else:
            source = 'unknown'

        actual_dir = patient_dir
        if (patient_dir / "ses-presurgery").exists():
            actual_dir = patient_dir
        elif (patient_dir / "ses-01").exists():
            actual_dir = patient_dir

        graphs = process_patient(actual_dir, patient_id, source)

        if graphs:
            all_graphs.extend(graphs)
            meta_rows.append({
                'patient_id': patient_id,
                'dataset': source,
                'n_graphs': len(graphs),
                'n_channels': graphs[0].n_channels,
                'has_soz': int(graphs[0].y.item()),
            })
        else:
            failed.append(patient_id)

    print(f"\n{'='*55}")
    print(f"  Total graphs created : {len(all_graphs)}")
    print(f"  Patients processed   : {len(meta_rows)}")
    print(f"  Patients failed      : {len(failed)}")

    if all_graphs:
        labels = [g.y.item() for g in all_graphs]
        soz_count    = sum(labels)
        normal_count = len(labels) - soz_count
        print(f"  SOZ graphs (1)       : {soz_count}")
        print(f"  Normal graphs (0)    : {normal_count}")
        print(f"  Feature dim          : {all_graphs[0].x.shape[1]}")
        print(f"{'='*55}")
        torch.save(all_graphs, OUTPUT_FILE)
        print(f"\n  Saved: {OUTPUT_FILE}")
        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv(META_FILE, index=False)
        print(f"  Metadata: {META_FILE}")

    if failed:
        print(f"\n  Failed patients ({len(failed)}):")
        for p in failed[:10]:
            print(f"    {p}")

    return all_graphs

if __name__ == "__main__":
    graphs = main()