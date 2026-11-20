import os
import mne
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("HUP Dataset Processing - RAW EEG Connectivity")

DATASET_PATH = "D:/FYP_Data/HUP_dataset"

participants = pd.read_csv(f"{DATASET_PATH}/participants.tsv", sep='\t')

def compute_connectivity(data):
    # data: [channels, time]
    corr_matrix = np.corrcoef(data)
    corr_matrix = np.nan_to_num(corr_matrix)

    edge_idx1, edge_idx2 = [], []

    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            if abs(corr_matrix[i, j]) > 0.3:
                edge_idx1.append(i)
                edge_idx2.append(j)

    return edge_idx1, edge_idx2

def process_patient(subject_id):
    try:
        subject_folder = f"sub-{subject_id}"
        subject_path = f"{DATASET_PATH}/{subject_folder}"

        edf_file = None
        for root, _, files in os.walk(subject_path):
            for f in files:
                if f.endswith('.edf'):
                    edf_file = os.path.join(root, f)
                    break
            if edf_file:
                break

        if not edf_file:
            return []

        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        data = raw.get_data()

        # filter valid channels
        valid_channels = [i for i in range(data.shape[0]) if not np.all(np.isnan(data[i]))]
        data = data[valid_channels[:64]]

        fs = int(raw.info['sfreq'])

        window_size = fs * 5  # 5 sec
        step = window_size // 2  # 50% overlap

        graphs = []

        for start in range(0, data.shape[1] - window_size, step):
            window = data[:, start:start+window_size]

            if window.shape[1] < window_size:
                continue

            # ======================
            # NODE FEATURES
            # ======================
            node_features = []
            for ch in window:
                node_features.append([
                    np.mean(ch),
                    np.std(ch),
                    np.max(ch),
                    np.min(ch),
                    np.median(ch),
                    np.percentile(ch, 25),
                    np.percentile(ch, 75),
                    np.var(ch)
                ])

            node_features = torch.tensor(node_features, dtype=torch.float)

            # ======================
            # RAW EEG CONNECTIVITY
            # ======================
            edge_idx1, edge_idx2 = compute_connectivity(window)

            if len(edge_idx1) == 0:
                continue

            edge_index = torch.tensor([edge_idx1, edge_idx2], dtype=torch.long)

            # label
            full_id = f"sub-{subject_id}"
            outcome = participants[participants['participant_id'] == full_id]['outcome'].values[0]
            y = torch.tensor([1 if outcome == 'S' else 0], dtype=torch.long)

            graph = Data(
                x=node_features,
                edge_index=edge_index,
                y=y
            )

            graph.subject = torch.tensor(int(subject_id.replace("HUP", "")))

            graphs.append(graph)

        return graphs

    except:
        return []

print("Processing dataset...")
all_graphs = []

for subject_id in tqdm(participants['participant_id'].str.replace('sub-', '')):
    g = process_patient(subject_id)
    all_graphs.extend(g)

torch.save(all_graphs, "hup_bnt_gnn_dataset_v4.pt")

print("Total graphs:", len(all_graphs))
print("Saved: hup_bnt_gnn_dataset_v4.pt")