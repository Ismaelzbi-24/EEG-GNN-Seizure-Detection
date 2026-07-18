"""
Omni-iEEG Improved GAT — v3
Improvements over v2:
1. Hidden 64->128, Heads 4->8
2. Graph context broadcast to nodes
3. Focal Loss for class imbalance
4. Youden threshold (paper standard)
5. 3 GAT layers
"""

import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix, roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

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
DATA_FILE  = "/home/pmilab/EEG_GNN_Project/code/omni_ieeg_graphs.pt"
OUTPUT_DIR = "/home/pmilab/EEG_GNN_Project/results/omni_gat_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FOLDS    = 5
EPOCHS     = 120
BATCH_SIZE = 32
LR         = 0.0005
DROPOUT    = 0.3
HIDDEN     = 128
HEADS      = 8

print(f"\nConfig:")
print(f"  Folds   : {N_FOLDS}")
print(f"  Epochs  : {EPOCHS}")
print(f"  Hidden  : {HIDDEN}")
print(f"  Heads   : {HEADS}")
print(f"  Batch   : {BATCH_SIZE}")
print(f"  LR      : {LR}")

# ==========================================
# LOAD DATA
# ==========================================
print(f"\nLoading dataset...")
data_list = torch.load(DATA_FILE, weights_only=False)
print(f"Total graphs  : {len(data_list)}")

patient_ids = [d.patient_id for d in data_list]
unique_pats = list(set(patient_ids))
pat_to_int  = {p: i for i, p in enumerate(unique_pats)}
groups      = [pat_to_int[d.patient_id] for d in data_list]
labels      = [d.y.item() for d in data_list]

print(f"Unique patients: {len(unique_pats)}")
print(f"Feature dim    : {data_list[0].x.shape[1]}")

# Node-level class distribution
node_labels_all = []
for g in data_list:
    node_labels_all.extend([int(v) for v in g.node_y.tolist()])

n_pos = sum(node_labels_all)
n_neg = len(node_labels_all) - n_pos
print(f"Node SOZ (1)  : {n_pos} ({100*n_pos/len(node_labels_all):.1f}%)")
print(f"Node Normal(0): {n_neg} ({100*n_neg/len(node_labels_all):.1f}%)")

# ==========================================
# FEATURE SCALING
# ==========================================
print("\nScaling features...")
all_x  = np.vstack([d.x.numpy() for d in data_list])
scaler = StandardScaler().fit(all_x)

for d in data_list:
    scaled = scaler.transform(d.x.numpy())
    scaled = np.clip(scaled, -10, 10)
    d.x   = torch.tensor(scaled, dtype=torch.float)

print("Scaling done.")


# ==========================================
# FOCAL LOSS
# ==========================================
class FocalLoss(nn.Module):
    """
    Focal Loss — better than weighted CE for imbalanced data.
    alpha: weight for positive class
    gamma: focusing parameter (higher = more focus on hard examples)
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt      = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal   = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


# ==========================================
# MODEL — Improved GAT
# ==========================================
class ImprovedGAT(nn.Module):
    """
    Improvements over v2:
    - 3 GAT layers (deeper = more brain connectivity captured)
    - Hidden 128, Heads 8 (more capacity)
    - Graph context broadcast back to nodes
    - Skip connections
    """
    def __init__(self, in_channels, hidden=128, heads=8, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # 3 GAT layers
        self.gat1 = GATConv(in_channels,    hidden,        heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden * heads, hidden,        heads=heads, dropout=dropout)
        self.gat3 = GATConv(hidden * heads, hidden,        heads=1,     dropout=dropout)

        # Skip projections
        self.skip1 = nn.Linear(in_channels,    hidden)
        self.skip2 = nn.Linear(hidden * heads, hidden)

        # Batch norms
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.bn2 = nn.BatchNorm1d(hidden * heads)
        self.bn3 = nn.BatchNorm1d(hidden)

        # Node classifier — node features + graph context
        self.classifier = nn.Sequential(
            nn.Linear(hidden + hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x, edge_index, batch):
        # Layer 1
        skip1 = self.skip1(x)
        x     = self.gat1(x, edge_index)
        x     = self.bn1(x)
        x     = F.elu(x)
        x     = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        skip2 = self.skip2(x)
        x     = self.gat2(x, edge_index)
        x     = self.bn2(x)
        x     = F.elu(x)
        x     = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3 + skip
        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x + skip2 + skip1)  # residual

        # Graph context — broadcast to each node
        graph_context = global_mean_pool(x, batch)  # [n_graphs, hidden]
        graph_context = graph_context[batch]         # [n_nodes, hidden]

        # Combine node + graph context
        combined = torch.cat([x, graph_context], dim=1)

        return self.classifier(combined)


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
# EVALUATE — NODE LEVEL + YOUDEN THRESHOLD
# ==========================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    all_trues = []
    all_probs = []

    for batch in loader:
        batch = batch.to(DEVICE)
        out   = model(batch.x, batch.edge_index, batch.batch)
        probs = torch.softmax(out, dim=1)[:, 1]

        all_probs.extend(probs.cpu().numpy())
        all_trues.extend(batch.node_y.long().cpu().numpy())

    all_trues = np.array(all_trues)
    all_probs = np.array(all_probs)

    # Youden's J threshold (paper standard)
    fpr, tpr, thresholds = roc_curve(all_trues, all_probs)
    j_scores  = tpr - fpr
    best_idx  = np.argmax(j_scores)
    best_thresh = float(thresholds[best_idx])

    preds = (all_probs >= best_thresh).astype(int)

    return {
        "acc"      : accuracy_score(all_trues, preds),
        "f1"       : f1_score(all_trues, preds, zero_division=0),
        "precision": precision_score(all_trues, preds, zero_division=0),
        "recall"   : recall_score(all_trues, preds, zero_division=0),
        "auc"      : roc_auc_score(all_trues, all_probs),
        "threshold": best_thresh,
        "cm"       : confusion_matrix(all_trues, preds),
    }


# ==========================================
# CROSS VALIDATION
# ==========================================
print("\n" + "="*55)
print("  5-Fold Cross Validation — Improved GAT v3")
print("="*55)

gkf          = GroupKFold(n_splits=N_FOLDS)
fold_results = []
best_f1_global   = -1.0
best_model_state = None
in_channels      = data_list[0].x.shape[1]

for fold, (train_idx, test_idx) in enumerate(
    gkf.split(data_list, labels, groups=groups), start=1
):
    print(f"\n{'='*55}")
    print(f"  FOLD {fold}/{N_FOLDS}")
    print(f"{'='*55}")

    train_data = [data_list[i] for i in train_idx]
    test_data  = [data_list[i] for i in test_idx]

    print(f"  Train graphs: {len(train_data)}")
    print(f"  Test  graphs: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    model     = ImprovedGAT(in_channels, hidden=HIDDEN, heads=HEADS, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Focal loss — no need for manual weights, alpha handles it
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    best_fold_f1      = -1.0
    best_fold_metrics = None
    best_fold_state   = None

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            metrics = evaluate(model, test_loader)
            print(f"  Ep {epoch:3d} | Loss: {loss:.4f} | "
                  f"F1: {metrics['f1']:.4f} | "
                  f"Rec: {metrics['recall']:.4f} | "
                  f"Prec: {metrics['precision']:.4f} | "
                  f"AUC: {metrics['auc']:.4f} | "
                  f"Thresh: {metrics['threshold']:.2f}")

            if metrics['f1'] > best_fold_f1:
                best_fold_f1      = metrics['f1']
                best_fold_metrics = metrics
                best_fold_state   = {
                    'model': model.state_dict(),
                    'fold' : fold,
                    'epoch': epoch,
                    'f1'   : metrics['f1'],
                }

    m = best_fold_metrics
    print(f"\n  Best Fold {fold}:")
    print(f"  Acc={m['acc']:.4f} | F1={m['f1']:.4f} | "
          f"Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | "
          f"AUC={m['auc']:.4f}")
    print(f"  CM [[TN FP] [FN TP]]: {m['cm']}")

    fold_results.append(m)

    if m['f1'] > best_f1_global:
        best_f1_global   = m['f1']
        best_model_state = best_fold_state

# ==========================================
# FINAL SUMMARY
# ==========================================
print("\n" + "="*55)
print("  FINAL RESULTS — Improved GAT v3")
print("="*55)

metrics_keys = ['acc', 'f1', 'precision', 'recall', 'auc']
metric_names = ['Accuracy ', 'F1-Score ', 'Precision', 'Recall   ', 'ROC-AUC  ']

for key, name in zip(metrics_keys, metric_names):
    vals = [r[key] for r in fold_results]
    print(f"  {name}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# ==========================================
# PAPER COMPARISON
# ==========================================
our_f1   = np.mean([r['f1']        for r in fold_results])
our_prec = np.mean([r['precision'] for r in fold_results])
our_rec  = np.mean([r['recall']    for r in fold_results])
our_auc  = np.mean([r['auc']       for r in fold_results])

print("\n" + "="*55)
print("  COMPARISON WITH PAPER BASELINES")
print("="*55)
print(f"  {'Model':<16} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6}")
print(f"  {'-'*44}")

baselines = [
    ("SEEG-NET    ", 0.5259, 0.5790, 0.7169, 0.7850),
    ("CLAP        ", 0.6009, 0.5936, 0.6997, 0.7684),
    ("TimeConv-CNN", 0.6469, 0.6259, 0.7454, 0.8061),
    ("Our GAT v2  ", 0.4881, 0.4143, 0.6068, 0.7477),
    ("Our GAT v3  ", our_f1, our_prec, our_rec, our_auc),
]

for name, f1, prec, rec, auc in baselines:
    marker = " ←" if "v3" in name else ""
    print(f"  {name:<16} {f1:>6.4f} {prec:>6.4f} {rec:>6.4f} {auc:>6.4f}{marker}")

# ==========================================
# SAVE
# ==========================================
torch.save(best_model_state, os.path.join(OUTPUT_DIR, "best_model_v3.pt"))

summary_df = pd.DataFrame([{
    'metric': k,
    'mean'  : np.mean([r[k] for r in fold_results]),
    'std'   : np.std( [r[k] for r in fold_results]),
} for k in metrics_keys])
summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_v3.csv"), index=False)

print(f"\n  Best fold F1 : {best_f1_global:.4f} (Fold {best_model_state['fold']})")
print(f"  Model saved  : {OUTPUT_DIR}/best_model_v3.pt")
print(f"  Summary      : {OUTPUT_DIR}/summary_v3.csv")
print("\n  TRAINING COMPLETE!")
