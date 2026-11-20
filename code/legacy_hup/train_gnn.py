import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
import mne

# ===================================
# LOAD EEG
# ===================================

edf_path = "../Hiroki/Data/Data/sub-hupHUP060/ses-01/ieeg/sub-hupHUP060_ses-01_task-ictal_run-01_ieeg.edf"

raw = mne.io.read_raw_edf(edf_path, preload=True)

data = raw.get_data()

# ===================================
# WINDOWING
# ===================================

sfreq = int(raw.info['sfreq'])

window_sec = 2
window_size = sfreq * window_sec

windows = []

for start in range(0, data.shape[1] - window_size, window_size):

    segment = data[:, start:start+window_size]

    windows.append(segment)

windows = np.array(windows)

# ===================================
# CREATE GRAPH DATASET
# ===================================

all_graphs = []

for window in windows:

    corr_matrix = np.corrcoef(window)

    threshold = 0.7

    adj_matrix = (np.abs(corr_matrix) > threshold).astype(int)

    edge_index = np.array(np.nonzero(adj_matrix))

    edge_index = torch.tensor(edge_index, dtype=torch.long)

    node_features = torch.tensor(window, dtype=torch.float)

    label = torch.tensor([1], dtype=torch.long)

    graph = Data(
        x=node_features,
        edge_index=edge_index,
        y=label
    )

    all_graphs.append(graph)

print("Total Graphs:", len(all_graphs))

# ===================================
# DATALOADER
# ===================================

loader = DataLoader(all_graphs, batch_size=16, shuffle=True)

# ===================================
# GAT MODEL
# ===================================

class GAT(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.gat1 = GATConv(1000, 64, heads=4)

        self.gat2 = GATConv(64 * 4, 64)

        self.classifier = torch.nn.Linear(64, 2)

    def forward(self, x, edge_index, batch):

        x = self.gat1(x, edge_index)

        x = F.elu(x)

        x = self.gat2(x, edge_index)

        x = F.elu(x)

        x = global_mean_pool(x, batch)

        x = self.classifier(x)

        return x

# ===================================
# DEVICE
# ===================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

# ===================================
# MODEL
# ===================================

model = GAT().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion = torch.nn.CrossEntropyLoss()

# ===================================
# TRAINING
# ===================================

for epoch in range(1, 11):

    model.train()

    total_loss = 0

    for batch in loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)

        loss = criterion(out, batch.y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} | Loss: {total_loss:.4f}")