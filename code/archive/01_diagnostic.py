"""
Diagnostic version — finds all hanging/failed patients
"""
import os
import signal
import warnings
import gc
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import mne
from scipy import signal as scipy_signal
from torch_geometric.data import Data
from tqdm import tqdm

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ==========================================
# CONFIG — same as your original
# ==========================================
DATASET_PATH = "/home/pmilab/EEG_GNN_Project/Hiroki/Data/Data"
OUTPUT_FILE  = "/home/pmilab/EEG_GNN_Project/code/omni_ieeg_graphs.pt"
META_FILE    = "/home/pmilab/EEG_GNN_Project/code/omni_ieeg_meta.csv"
TARGET_SFREQ  = 1000
WINDOW_SEC    = 60
STEP_SEC      = 30
CORR_THRESH   = 0.3
MAX_CHANNELS  = 64

TIMEOUT_SECONDS = 120  # 2 minutes max per patient

# ==========================================
# TIMEOUT HANDLER
# ==========================================
class PatientTimeout(Exception):
    pass

def timeout_handler(signum, frame):
    raise PatientTimeout("Patient processing timed out!")

# ==========================================
# ALL YOUR ORIGINAL FUNCTIONS (unchanged)
# ==========================================
def bandpower(sig, fs, fmin, fmax):
    f, psd = scipy_signal.welch(sig, fs=fs, nperseg=min(fs, len(sig)))
    idx = np.logical_and(f >= fmin, f <= fmax)
    return float(np.trapz(psd[idx], f[idx])) if idx.any() else 0.0

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
        f, psd = scipy_signal.welch(ch_sig, fs=fs, nperseg=min(fs, len(ch_sig)))
        total_p  = float(np.trapz(psd, f)) + 1e-10
        peak_f   = float(f[np.argmax(psd)])
        psd_norm = psd / (total_p + 1e-10)
        psd_norm = psd_norm[psd_norm > 0]
        sp_ent   = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))
        feat = [mean_v, std_v, var_v, skew_v, kurt_v,
                rms_v, ll_v, zc_v,
                delta_p, theta_p, alpha_p, beta_p,
                gamma_p, hgamma_p, hfo_p,
                total_p, peak_f, sp_ent]
        features.append(feat)
    return np.array(features, dtype=np.float32)

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
            node_labels = torch.tensor([soz_map[ch] for ch in ch_names], dtype=torch.long)
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
# MAIN — WITH TIMEOUT DETECTION
# ==========================================
def main():
    dataset_path = Path(DATASET_PATH)
    patient_dirs = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and d.name.startswith('sub-')
    ])

    print(f"\nTotal patients found: {len(patient_dirs)}")
    print(f"Timeout per patient : {TIMEOUT_SECONDS}s")
    print("=" * 55)

    success_list  = []
    failed_list   = []
    timeout_list  = []

    for i, patient_dir in enumerate(tqdm(patient_dirs, desc="Scanning patients")):
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

        print(f"\n[{i+1}/{len(patient_dirs)}] START: {patient_id}", flush=True)

        # Set timeout alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)

        try:
            graphs = process_patient(patient_dir, patient_id, source)
            signal.alarm(0)  # cancel alarm on success

            if graphs:
                success_list.append(patient_id)
                print(f"  OK — {len(graphs)} graphs created", flush=True)
            else:
                failed_list.append(patient_id)
                print(f"  SKIP — no valid graphs (no EDF or bad data)", flush=True)

        except PatientTimeout:
            signal.alarm(0)
            timeout_list.append((i+1, patient_id))
            print(f"  TIMEOUT — hung after {TIMEOUT_SECONDS}s!", flush=True)

        except Exception as e:
            signal.alarm(0)
            failed_list.append(patient_id)
            print(f"  ERROR — {e}", flush=True)

        finally:
            gc.collect()
            torch.cuda.empty_cache()

    # ==========================================
    # FINAL REPORT
    # ==========================================
    print("\n" + "=" * 55)
    print("           DIAGNOSTIC SUMMARY")
    print("=" * 55)
    print(f"  Total patients      : {len(patient_dirs)}")
    print(f"  Successful          : {len(success_list)}")
    print(f"  Failed (no data)    : {len(failed_list)}")
    print(f"  TIMED OUT (hung)    : {len(timeout_list)}")

    if timeout_list:
        print(f"\n  !! PROBLEMATIC PATIENTS (timed out) !!")
        for idx, pid in timeout_list:
            print(f"    [{idx}] {pid}")

    if failed_list:
        print(f"\n  Patients with no valid data:")
        for pid in failed_list:
            print(f"    {pid}")

    # Save report
    report = {
        "total": len(patient_dirs),
        "success": success_list,
        "failed": failed_list,
        "timed_out": [pid for _, pid in timeout_list],
    }
    report_path = "/home/pmilab/EEG_GNN_Project/code/diagnostic_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved: {report_path}")
    print("=" * 55)

if __name__ == "__main__":
    main()
