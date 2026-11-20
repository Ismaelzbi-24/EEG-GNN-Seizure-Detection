import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool


# ======================
# REPRODUCIBILITY
# ======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ======================
# CONFIG
# ======================
DATA_FILE = "hup_bnt_gnn_dataset_v4.pt"
OUTPUT_DIR = "results_gat_graph_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ======================
# LOAD DATA
# ======================
print("Loading dataset...")
data_list = torch.load(DATA_FILE, weights_only=False)

subjects = [d.subject.item() for d in data_list]
labels = [d.y.item() for d in data_list]

print("Total graphs:", len(data_list))
print("Unique subjects:", len(set(subjects)))
print("Feature dim:", data_list[0].x.shape[1])

# ======================
# FEATURE SCALING
# ======================
print("Scaling features...")
all_features = np.vstack([d.x.numpy() for d in data_list])
scaler = StandardScaler().fit(all_features)

for d in data_list:
    d.x = torch.tensor(scaler.transform(d.x.numpy()), dtype=torch.float)

print("Scaling done.")

# ======================
# MODEL
# ======================
class GAT(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gat1 = GATConv(in_channels, 64, heads=4)
        self.gat2 = GATConv(64 * 4, 64, heads=1)
        self.lin = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32 + 4, 2)

    def forward(self, x):
        return self.fc(x)


def make_graph_features(batch_x, batch_index):
    pooled = global_mean_pool(batch_x, batch_index)
    feats = torch.stack(
        [
            pooled.mean(dim=1),
            pooled.std(dim=1),
            pooled.max(dim=1).values,
            pooled.min(dim=1).values,
        ],
        dim=1,
    )
    return feats


# ======================
# TRAIN / EVAL HELPERS
# ======================
def train_one_epoch(model, clf, loader, optimizer, criterion):
    model.train()
    clf.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        emb = model(batch.x, batch.edge_index, batch.batch)
        g_feats = make_graph_features(batch.x, batch.batch)
        combined = torch.cat([emb, g_feats], dim=1)

        out = clf(combined)
        loss = criterion(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss


@torch.no_grad()
def evaluate_patient_level(model, clf, loader):
    model.eval()
    clf.eval()

    patient_embeddings = defaultdict(list)
    patient_labels = {}

    for batch in loader:
        batch = batch.to(DEVICE)
        emb = model(batch.x, batch.edge_index, batch.batch)
        g_feats = make_graph_features(batch.x, batch.batch)

        for i in range(len(emb)):
            subj = int(batch.subject[i].item())
            combined = torch.cat([emb[i], g_feats[i]], dim=0)
            patient_embeddings[subj].append(combined.cpu())
            patient_labels[subj] = int(batch.y[i].item())

    preds, trues, probs, subj_ids = [], [], [], []

    for subj in sorted(patient_embeddings.keys()):
        emb_stack = torch.stack(patient_embeddings[subj])
        patient_emb = emb_stack.mean(dim=0)

        out = clf(patient_emb.unsqueeze(0).to(DEVICE))
        prob = torch.softmax(out, dim=1)[0, 1].item()
        pred = int(out.argmax(dim=1).item())
        true = patient_labels[subj]

        subj_ids.append(subj)
        preds.append(pred)
        trues.append(true)
        probs.append(prob)

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, zero_division=0)
    cm = confusion_matrix(trues, preds, labels=[0, 1])

    return {
        "acc": acc,
        "f1": f1,
        "cm": cm,
        "subjects": subj_ids,
        "preds": preds,
        "trues": trues,
        "probs": probs,
    }


# ======================
# CROSS-VALIDATION
# ======================
in_channels = data_list[0].x.shape[1]
gkf = GroupKFold(n_splits=5)

fold_accs, fold_f1s = [], []
all_rows = []

best_f1 = -1.0
best_fold = None
best_state = None

for fold, (train_idx, test_idx) in enumerate(
    gkf.split(data_list, labels, groups=subjects), start=1
):
    print(f"\n=== Fold {fold} ===")

    train_data = [data_list[i] for i in train_idx]
    test_data = [data_list[i] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = GAT(in_channels).to(DEVICE)
    clf = Classifier().to(DEVICE)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(clf.parameters()),
        lr=0.0005,
    )
    criterion = nn.CrossEntropyLoss()

    best_fold_f1 = -1.0
    best_fold_eval = None
    best_fold_model = None

    for epoch in range(1, 81):
        loss = train_one_epoch(model, clf, train_loader, optimizer, criterion)

        if epoch % 10 == 0 or epoch == 1:
            eval_out = evaluate_patient_level(model, clf, test_loader)
            print(
                f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                f"Acc: {eval_out['acc']:.4f} | F1: {eval_out['f1']:.4f}"
            )

            if eval_out["f1"] > best_fold_f1:
                best_fold_f1 = eval_out["f1"]
                best_fold_eval = eval_out
                best_fold_model = {
                    "model": model.state_dict(),
                    "clf": clf.state_dict(),
                    "fold": fold,
                    "epoch": epoch,
                }

    fold_acc = best_fold_eval["acc"]
    fold_f1 = best_fold_eval["f1"]
    cm = best_fold_eval["cm"]

    print(f"Best Fold {fold} -> Acc={fold_acc:.4f}, F1={fold_f1:.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    fold_accs.append(fold_acc)
    fold_f1s.append(fold_f1)

    for subj, true, pred, prob in zip(
        best_fold_eval["subjects"],
        best_fold_eval["trues"],
        best_fold_eval["preds"],
        best_fold_eval["probs"],
    ):
        all_rows.append(
            {
                "fold": fold,
                "subject": subj,
                "true_label": true,
                "pred_label": pred,
                "prob_class_1": prob,
            }
        )

    if fold_f1 > best_f1:
        best_f1 = fold_f1
        best_fold = fold
        best_state = best_fold_model

# ======================
# SAVE OUTPUTS
# ======================
pred_df = pd.DataFrame(all_rows)
pred_csv = os.path.join(OUTPUT_DIR, "patient_predictions.csv")
pred_df.to_csv(pred_csv, index=False)

model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
torch.save(best_state, model_path)

summary = pd.DataFrame(
    {
        "metric": ["accuracy_mean", "accuracy_std", "f1_mean", "f1_std", "best_fold"],
        "value": [
            float(np.mean(fold_accs)),
            float(np.std(fold_accs)),
            float(np.mean(fold_f1s)),
            float(np.std(fold_f1s)),
            int(best_fold),
        ],
    }
)
summary_csv = os.path.join(OUTPUT_DIR, "summary_metrics.csv")
summary.to_csv(summary_csv, index=False)

print("\n=== FINAL POLISHED BASELINE ===")
print(f"Accuracy : {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
print(f"F1-score : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
print(f"Best fold: {best_fold} (F1={best_f1:.4f})")
print(f"Saved predictions to: {pred_csv}")
print(f"Saved best model to: {model_path}")
print(f"Saved summary to: {summary_csv}")