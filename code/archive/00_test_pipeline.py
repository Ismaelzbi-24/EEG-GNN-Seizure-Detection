

import warnings
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

import numpy as np
import torch
from scipy import signal
from scipy.stats import skew, kurtosis
from torch_geometric.data import Data

print("=" * 50)
print("  Pipeline Quick Test — Single Patient")
print("=" * 50)

EDF_PATH = "/home/pmilab/EEG_GNN_Project/Hiroki/Data/Data/sub-Multicenterpt15/ses-presurgery/ieeg/sub-Multicenterpt15_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg.edf"

# Step 1: Load
print("\n[1] Loading EDF...")
raw = mne.io.read_raw_edf(EDF_PATH, preload=True, verbose=False)
print(f"    Channels : {len(raw.ch_names)}")
print(f"    Duration : {raw.times[-1]:.1f}s")
print(f"    Sfreq    : {raw.info['sfreq']} Hz")

# Step 2: Filter
print("\n[2] Filtering 1-300 Hz...")
raw.filter(1., 300., fir_design='firwin', verbose=False)
data = raw.get_data()
fs   = int(raw.info['sfreq'])
print(f"    Data shape: {data.shape}")

# Step 3: One window
print("\n[3] Extracting 60s window...")
win_size = 60 * fs
window   = data[:, :win_size]
print(f"    Window shape: {window.shape}")

# Step 4: Node features
print("\n[4] Extracting node features...")
def bandpower(sig, fs, fmin, fmax):
    f, psd = signal.welch(sig, fs=fs, nperseg=min(fs, len(sig)))
    idx = np.logical_and(f >= fmin, f <= fmax)
    return float(np.trapz(psd[idx], f[idx])) if idx.any() else 0.0

feats = []
for ch in window[:10]:  # test 10 channels only
    feat = [
        np.mean(ch), np.std(ch), np.var(ch),
        float(skew(ch)), float(kurtosis(ch)),
        float(np.sqrt(np.mean(ch**2))),
        bandpower(ch, fs, 0.5, 4),
        bandpower(ch, fs, 4, 8),
        bandpower(ch, fs, 8, 13),
        bandpower(ch, fs, 13, 30),
        bandpower(ch, fs, 30, 80),
        bandpower(ch, fs, 80, 150),
        bandpower(ch, fs, 80, 300),
    ]
    feats.append(feat)

node_feats = torch.tensor(feats, dtype=torch.float)
print(f"    Node features shape: {node_feats.shape}")
print(f"    Features per node  : {node_feats.shape[1]}")

# Step 5: Graph edges
print("\n[5] Building graph edges...")
corr = np.corrcoef(window[:10])
src, dst = [], []
for i in range(10):
    for j in range(i+1, 10):
        if abs(corr[i,j]) > 0.3:
            src.extend([i, j])
            dst.extend([j, i])

edge_index = torch.tensor([src, dst], dtype=torch.long)
print(f"    Edges: {edge_index.shape[1]}")

# Step 6: PyG Data
print("\n[6] Creating PyG graph...")
graph = Data(
    x=node_feats,
    edge_index=edge_index,
    y=torch.tensor([1], dtype=torch.long)
)
print(f"    Graph: {graph}")

# Step 7: GPU test
print("\n[7] GPU test...")
from torch_geometric.nn import GATConv, global_mean_pool
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"    Device: {device}")

gat = GATConv(13, 32, heads=4).to(device)
x   = graph.x.to(device)
ei  = graph.edge_index.to(device)
out = gat(x, ei)
print(f"    GAT output shape: {out.shape}")

print("\n" + "=" * 50)
print("  ALL STEPS PASSED! Pipeline ready.")
print("  Ab 01_process_omni_ieeg.py chalao")
print("=" * 50)
