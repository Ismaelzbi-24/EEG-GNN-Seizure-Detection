import mne
import numpy as np

edf_path = "../Hiroki/Data/Data/sub-hupHUP060/ses-01/ieeg/sub-hupHUP060_ses-01_task-ictal_run-01_ieeg.edf"

# Load EEG
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Get data
data = raw.get_data()

print("EEG Shape:", data.shape)

# Window settings
sfreq = int(raw.info['sfreq'])   # 500 Hz
window_sec = 2
window_size = sfreq * window_sec

windows = []

for start in range(0, data.shape[1] - window_size, window_size):
    segment = data[:, start:start+window_size]
    windows.append(segment)

windows = np.array(windows)

print("Windows Shape:", windows.shape)

# =========================
# CREATE CORRELATION GRAPH
# =========================

sample_window = windows[0]   # first window

corr_matrix = np.corrcoef(sample_window)

print("Correlation Matrix Shape:", corr_matrix.shape)

# threshold weak edges
threshold = 0.7

adj_matrix = (np.abs(corr_matrix) > threshold).astype(int)

print("Adjacency Matrix Shape:", adj_matrix.shape)

print(adj_matrix)

import torch
from torch_geometric.data import Data

# =========================
# CREATE EDGE INDEX
# =========================

edge_index = np.array(np.nonzero(adj_matrix))

edge_index = torch.tensor(edge_index, dtype=torch.long)

# =========================
# NODE FEATURES
# =========================

node_features = torch.tensor(sample_window, dtype=torch.float)

# shape = [35 nodes, 1000 features]

# =========================
# CREATE GRAPH
# =========================

graph_data = Data(
    x=node_features,
    edge_index=edge_index
)

print(graph_data)
print("Node Features Shape:", graph_data.x.shape)
print("Edge Index Shape:", graph_data.edge_index.shape)

# =========================
# LABEL
# =========================

label = torch.tensor([1], dtype=torch.long)

graph_data.y = label

print("Graph Label:", graph_data.y)

# =========================
# CREATE ALL GRAPH SAMPLES
# =========================

all_graphs = []

for window in windows:

    # correlation graph
    corr_matrix = np.corrcoef(window)

    threshold = 0.7
    adj_matrix = (np.abs(corr_matrix) > threshold).astype(int)

    # edge index
    edge_index = np.array(np.nonzero(adj_matrix))
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # node features
    node_features = torch.tensor(window, dtype=torch.float)

    # label
    label = torch.tensor([1], dtype=torch.long)

    # graph object
    graph = Data(
        x=node_features,
        edge_index=edge_index,
        y=label
    )

    all_graphs.append(graph)

print("Total Graphs Created:", len(all_graphs))
print(all_graphs[0])