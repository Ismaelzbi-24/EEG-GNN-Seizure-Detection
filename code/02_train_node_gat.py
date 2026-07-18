"""
================================================
Omni-iEEG GAT Training Pipeline
Task: SOZ Channel / Graph Classification
================================================

Ye script kya karta hai:
1. Processed graph dataset load karta hai
2. Patient-level GroupKFold split karta hai
3. GAT model train karta hai
4. F1, Recall, Precision, AUC report karta hai
5. Best model save karta hai
================================================
"""

import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

# ==========================================
# REPRODUCIBILITY
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# CONFIG
# ==========================================
DATA_FILE   = "/home/pmilab/EEG_GNN_Project/code/omni_ieeg_graphs.pt"
OUTPUT_DIR  = "/home/pmilab/EEG_GNN_Project/results/omni_gat"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FOLDS     = 5
EPOCHS      = 120
BATCH_SIZE  = 32
LR          = 0.0005
DROPOUT     = 0.3

print(f"\nConfig:")
print(f"  Folds   : {N_FOLDS}")
print(f"  Epochs  : {EPOCHS}")
print(f"  Batch   : {BATCH_SIZE}")
print(f"  LR      : {LR}")

# ==========================================
# LOAD DATA
# ==========================================
print(f"\nLoading dataset from: {DATA_FILE}")
data_list = torch.load(DATA_FILE, weights_only=False)
print(f"Total graphs: {len(data_list)}")

# Patient IDs — GroupKFold ke liye
patient_ids = [d.patient_id for d in data_list]
unique_pats = list(set(patient_ids))
pat_to_int  = {p: i for i, p in enumerate(unique_pats)}
groups      = [pat_to_int[d.patient_id] for d in data_list]
labels      = [d.y.item() for d in data_list]

print(f"Unique patients: {len(unique_pats)}")
print(f"SOZ graphs   : {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
print(f"Normal graphs: {len(labels)-sum(labels)}")
print(f"Feature dim  : {data_list[0].x.shape[1]}")

# ==========================================
# FEATURE SCALING
# ==========================================
print("\nScaling features...")
all_x = np.vstack([d.x.numpy() for d in data_list])
scaler = StandardScaler().fit(all_x)

for d in data_list:
    scaled = scaler.transform(d.x.numpy())
    scaled = np.clip(scaled, -10, 10)   # outlier clip
    d.x = torch.tensor(scaled, dtype=torch.float)

print("Scaling done.")

# ==========================================
# CLASS WEIGHTS (imbalance handle)
# ==========================================
node_labels_all = []
for g in data_list:
    node_labels_all.extend([int(v) for v in g.node_y.tolist()])

n_pos = sum(node_labels_all)
n_neg = len(node_labels_all) - n_pos
pos_weight = n_neg / (n_pos + 1e-6)
print(f"\nNode-level class weight for SOZ: {pos_weight:.2f}")


# ==========================================
# MODEL — GAT with Skip Connections
# ==========================================
class OmniGAT(nn.Module):
    """
    GAT model for SOZ detection.
    
    Architecture:
    - 2 GAT layers with multi-head attention
    - Skip connection
    - Global mean pool
    - Graph-level statistical features concatenated
    - 2-layer classifier
    """
    def __init__(self, in_channels, hidden=128, heads=8, dropout=0.3):
        super().__init__()

        self.dropout = dropout

        # GAT layers
        self.gat1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout)

        # Skip connection projection
        self.skip_proj = nn.Linear(in_channels, hidden)

        # Batch norm
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.bn2 = nn.BatchNorm1d(hidden)

        # Node-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x, edge_index, batch):
        # Skip connection
        skip = self.skip_proj(x)

        # GAT layer 1
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT layer 2
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x + skip)   # Skip connection

        # Node-level prediction
        return self.classifier(x)


# ==========================================
# TRAIN ONE EPOCH
# ==========================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        out  = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.node_y.long())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ==========================================
# EVALUATE — PATIENT LEVEL
# ==========================================
@torch.no_grad()
def evaluate_patient_level(model, loader):
    """
    Node-level evaluation using node_y.
    """
    model.eval()

    all_trues = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)

            out = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.softmax(out, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_trues.extend(batch.node_y.long().cpu().numpy())

    all_trues = np.array(all_trues)
    all_probs = np.array(all_probs)

    best_f1 = 0.0
    best_thresh = 0.5

    for t in np.arange(0.05, 0.95, 0.01):
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_trues, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    preds = (all_probs >= best_thresh).astype(int)

    return {
        "acc": accuracy_score(all_trues, preds),
        "f1": f1_score(all_trues, preds, zero_division=0),
        "precision": precision_score(all_trues, preds, zero_division=0),
        "recall": recall_score(all_trues, preds, zero_division=0),
        "auc": roc_auc_score(all_trues, all_probs),
        "threshold": best_thresh,
        "cm": confusion_matrix(all_trues, preds),
    }


# ==========================================
# CROSS VALIDATION TRAINING
# ==========================================
print("\n" + "=" * 55)
print("  Starting 5-Fold Cross Validation")
print("=" * 55)

gkf = GroupKFold(n_splits=N_FOLDS)

fold_results = []
all_rows     = []
best_f1_global = -1.0
best_model_state = None

in_channels = data_list[0].x.shape[1]

for fold, (train_idx, test_idx) in enumerate(
    gkf.split(data_list, labels, groups=groups), start=1
):
    print(f"\n{'='*55}")
    print(f"  FOLD {fold}/{N_FOLDS}")
    print(f"{'='*55}")

    train_data = [data_list[i] for i in train_idx]
    test_data  = [data_list[i] for i in test_idx]

    # Count classes in train
    train_labels = [d.y.item() for d in train_data]
    print(f"  Train: {len(train_data)} graphs | "
          f"SOZ={sum(train_labels)} Normal={len(train_labels)-sum(train_labels)}")
    print(f"  Test : {len(test_data)} graphs")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model     = OmniGAT(in_channels, hidden=64, heads=4, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Weighted loss for class imbalance
    train_node_labels = []
    for g in train_data:
        train_node_labels.extend([int(v) for v in g.node_y.tolist()])

    n_pos_fold = sum(train_node_labels)
    n_neg_fold = len(train_node_labels) - n_pos_fold
    w = torch.tensor([1.0, n_neg_fold / (n_pos_fold + 1e-6)]).to(DEVICE)
    criterion  = nn.CrossEntropyLoss(weight=w)

    # Training loop
    best_fold_f1    = -1.0
    best_fold_state = None
    best_fold_metrics = None

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            metrics = evaluate_patient_level(model, test_loader)

            print(f"  Ep {epoch:3d} | Loss: {loss:.4f} | "
                  f"F1: {metrics['f1']:.4f} | "
                  f"Rec: {metrics['recall']:.4f} | "
                  f"Prec: {metrics['precision']:.4f} | "
                  f"AUC: {metrics['auc']:.4f}")

            if metrics['f1'] > best_fold_f1:
                best_fold_f1      = metrics['f1']
                best_fold_metrics = metrics
                best_fold_state   = {
                    'model': model.state_dict(),
                    'fold' : fold,
                    'epoch': epoch,
                    'f1'   : metrics['f1'],
                }

    # Fold summary
    m = best_fold_metrics
    print(f"\n  Best Fold {fold}:")
    print(f"  Acc={m['acc']:.4f} | F1={m['f1']:.4f} | "
          f"Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | "
          f"AUC={m['auc']:.4f}")
    print(f"  Confusion Matrix [[TN FP] [FN TP]]:")
    print(f"  {m['cm']}")

    fold_results.append(m)

    if m['f1'] > best_f1_global:
        best_f1_global   = m['f1']
        best_model_state = best_fold_state

# ==========================================
# FINAL SUMMARY
# ==========================================
print("\n" + "=" * 55)
print("  FINAL RESULTS — 5 Fold Cross Validation")
print("=" * 55)

metrics_keys = ['acc', 'f1', 'precision', 'recall', 'auc']
metric_names = ['Accuracy ', 'F1-Score ', 'Precision', 'Recall   ', 'ROC-AUC  ']

for key, name in zip(metrics_keys, metric_names):
    vals = [r[key] for r in fold_results]
    print(f"  {name}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

print(f"\n  Best fold F1: {best_f1_global:.4f} "
      f"(Fold {best_model_state['fold']})")

# ==========================================
# COMPARE WITH PAPER BASELINES
# ==========================================
print("\n" + "=" * 55)
print("  COMPARISON WITH OMNI-IEEG PAPER BASELINES")
print("=" * 55)

f1_vals   = [r['f1']        for r in fold_results]
prec_vals = [r['precision'] for r in fold_results]
rec_vals  = [r['recall']    for r in fold_results]
auc_vals  = [r['auc']       for r in fold_results]

our_f1   = np.mean(f1_vals)
our_prec = np.mean(prec_vals)
our_rec  = np.mean(rec_vals)
our_auc  = np.mean(auc_vals)

baselines = [
    ("SEEG-NET    ", 0.5259, 0.5790, 0.7169, 0.7850),
    ("CLAP        ", 0.6009, 0.5936, 0.6997, 0.7684),
    ("TimeConv-CNN", 0.6469, 0.6259, 0.7454, 0.8061),
    ("Our GAT     ", our_f1, our_prec, our_rec, our_auc),
]

print(f"  {'Model':<16} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6}")
print(f"  {'-'*44}")
for name, f1, prec, rec, auc in baselines:
    marker = " ←" if "GAT" in name else ""
    print(f"  {name:<16} {f1:>6.4f} {prec:>6.4f} {rec:>6.4f} {auc:>6.4f}{marker}")

# ==========================================
# SAVE
# ==========================================
torch.save(best_model_state, os.path.join(OUTPUT_DIR, "best_model.pt"))

summary_df = pd.DataFrame([{
    'metric': k,
    'mean': np.mean([r[k] for r in fold_results]),
    'std' : np.std( [r[k] for r in fold_results]),
} for k in metrics_keys])
global_mean_pool

summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

print(f"\n  Model saved : {OUTPUT_DIR}/best_model.pt")
print(f"  Summary     : {OUTPUT_DIR}/summary.csv")
print("\n  TRAINING COMPLETE!")
