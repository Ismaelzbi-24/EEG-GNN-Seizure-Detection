import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "/home/pmilab/EEG_GNN_Project/results/graphs"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# DATA
# ==========================================
folds          = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
fold_f1        = [0.6376, 0.7667, 0.5000, 0.6498, 0.5108]
fold_precision = [0.6182, 0.8387, 0.4049, 0.6456, 0.4152]
fold_recall    = [0.6583, 0.7060, 0.6534, 0.6540, 0.6638]
fold_auc       = [0.8495, 0.8924, 0.7575, 0.8582, 0.7997]

models     = ['SEEG-NET', 'CLAP', 'TimeConv\nCNN', 'GAT v2\n(Ours)', 'GAT v3\n(Ours)', 'GAT v4\n(Ours)']
f1_scores  = [0.5259, 0.6009, 0.6469, 0.4881, 0.6050, 0.6130]
precisions = [0.5790, 0.5936, 0.6259, 0.4143, 0.5730, 0.5845]
recalls    = [0.7169, 0.6997, 0.7454, 0.6068, 0.6549, 0.6671]
aucs       = [0.7850, 0.7684, 0.8061, 0.7477, 0.8272, 0.8315]

BASELINE_COLOR = '#5B8DB8'
OUR_COLOR      = '#E8734A'
COLORS = [BASELINE_COLOR, BASELINE_COLOR, BASELINE_COLOR,
          OUR_COLOR, OUR_COLOR, OUR_COLOR]

plt.rcParams.update({
    'font.family'  : 'DejaVu Sans',
    'font.size'    : 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi'   : 150,
})

# ==========================================
# GRAPH 1 — F1 Score Comparison
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#F8F9FA')
bars = ax.bar(models, f1_scores, color=COLORS, edgecolor='white',
              linewidth=1.2, width=0.6)
for bar, val in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(y=0.6469, color='#2ecc71', linestyle='--',
           linewidth=1.8, alpha=0.8, label='TimeConv-CNN best (0.6469)')
ax.set_title('F1-Score Comparison: Our GAT vs Paper Baselines', fontweight='bold', pad=12)
ax.set_ylabel('F1-Score')
ax.set_ylim(0, 0.92)
ax.legend(fontsize=10)
ax.set_facecolor('#FFFFFF')
ax.grid(axis='y', alpha=0.3)
baseline_patch = mpatches.Patch(color=BASELINE_COLOR, label='Paper Baselines')
our_patch      = mpatches.Patch(color=OUR_COLOR, label='Our GAT Models')
ax.legend(handles=[baseline_patch, our_patch,
          mpatches.Patch(color='#2ecc71', label='Best Baseline (0.6469)')],
          fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_f1_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 01_f1_comparison.png")

# ==========================================
# GRAPH 2 — Recall Comparison
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#F8F9FA')
bars = ax.bar(models, recalls, color=COLORS, edgecolor='white',
              linewidth=1.2, width=0.6)
for bar, val in zip(bars, recalls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(y=0.7454, color='#2ecc71', linestyle='--',
           linewidth=1.8, alpha=0.8, label='TimeConv-CNN best (0.7454)')
ax.set_title('Recall Comparison (Clinical Priority Metric)', fontweight='bold', pad=12)
ax.set_ylabel('Recall')
ax.set_ylim(0, 0.92)
ax.set_facecolor('#FFFFFF')
ax.grid(axis='y', alpha=0.3)
baseline_patch = mpatches.Patch(color=BASELINE_COLOR, label='Paper Baselines')
our_patch      = mpatches.Patch(color=OUR_COLOR, label='Our GAT Models')
ax.legend(handles=[baseline_patch, our_patch,
          mpatches.Patch(color='#2ecc71', label='Best Baseline (0.7454)')],
          fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_recall_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 02_recall_comparison.png")

# ==========================================
# GRAPH 3 — Precision Comparison
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#F8F9FA')
bars = ax.bar(models, precisions, color=COLORS, edgecolor='white',
              linewidth=1.2, width=0.6)
for bar, val in zip(bars, precisions):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(y=0.6259, color='#2ecc71', linestyle='--',
           linewidth=1.8, alpha=0.8)
ax.set_title('Precision Comparison', fontweight='bold', pad=12)
ax.set_ylabel('Precision')
ax.set_ylim(0, 1.0)
ax.set_facecolor('#FFFFFF')
ax.grid(axis='y', alpha=0.3)
baseline_patch = mpatches.Patch(color=BASELINE_COLOR, label='Paper Baselines')
our_patch      = mpatches.Patch(color=OUR_COLOR, label='Our GAT Models')
ax.legend(handles=[baseline_patch, our_patch,
          mpatches.Patch(color='#2ecc71', label='Best Baseline (0.6259)')],
          fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_precision_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 03_precision_comparison.png")

# ==========================================
# GRAPH 4 — AUC Comparison
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#F8F9FA')
bars = ax.bar(models, aucs, color=COLORS, edgecolor='white',
              linewidth=1.2, width=0.6)
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(y=0.8061, color='#2ecc71', linestyle='--',
           linewidth=1.8, alpha=0.8)
ax.set_title('ROC-AUC Comparison (Our Model Best!)', fontweight='bold', pad=12)
ax.set_ylabel('ROC-AUC')
ax.set_ylim(0.65, 0.97)
ax.set_facecolor('#FFFFFF')
ax.grid(axis='y', alpha=0.3)
baseline_patch = mpatches.Patch(color=BASELINE_COLOR, label='Paper Baselines')
our_patch      = mpatches.Patch(color=OUR_COLOR, label='Our GAT Models')
ax.legend(handles=[baseline_patch, our_patch,
          mpatches.Patch(color='#2ecc71', label='Best Baseline (0.8061)')],
          fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_auc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 04_auc_comparison.png")

# ==========================================
# GRAPH 5 — Fold-wise F1
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#F8F9FA')
x    = np.arange(len(folds))
bars = ax.bar(x, fold_f1, color=OUR_COLOR, edgecolor='white',
              linewidth=1.2, width=0.5, alpha=0.85)
for bar, val in zip(bars, fold_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
mean_f1 = np.mean(fold_f1)
ax.axhline(y=mean_f1, color='#2c3e50', linestyle='--',
           linewidth=2.2, label=f'Mean F1 = {mean_f1:.3f}')
ax.set_title('Fold-wise F1 Score — 5-Fold Cross Validation', fontweight='bold', pad=12)
ax.set_ylabel('F1-Score')
ax.set_xticks(x)
ax.set_xticklabels(folds, fontsize=12)
ax.set_ylim(0, 0.92)
ax.legend(fontsize=11)
ax.set_facecolor('#FFFFFF')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_foldwise_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 05_foldwise_f1.png")

# ==========================================
# GRAPH 6 — Fold-wise All Metrics
# ==========================================
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#F8F9FA')
x     = np.arange(len(folds))
width = 0.2
ax.bar(x - 1.5*width, fold_f1,        width, label='F1',        color='#E8734A')
ax.bar(x - 0.5*width, fold_precision, width, label='Precision', color='#5B8DB8')
ax.bar(x + 0.5*width, fold_recall,    width, label='Recall',    color='#2ecc71')
ax.bar(x + 1.5*width, fold_auc,       width, label='AUC',       color='#9b59b6')
ax.set_title('All Metrics per Fold — 5-Fold Cross Validation', fontweight='bold', pad=12)
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(folds, fontsize=12)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=11)
ax.set_facecolor('#FFFFFF')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_foldwise_all_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 06_foldwise_all_metrics.png")

# ==========================================
# GRAPH 7 — Radar Chart
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('#F8F9FA')
categories  = ['F1', 'Precision', 'Recall', 'AUC']
N           = len(categories)
angles      = [n / float(N) * 2 * np.pi for n in range(N)]
angles     += angles[:1]
our_values  = [0.6130, 0.5845, 0.6671, 0.8315] + [0.6130]
best_values = [0.6469, 0.6259, 0.7454, 0.8061] + [0.6469]
seeg_values = [0.5259, 0.5790, 0.7169, 0.7850] + [0.5259]
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
ax.set_ylim(0.4, 0.9)
ax.set_yticks([0.5, 0.6, 0.7, 0.8])
ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8'], fontsize=9)
ax.plot(angles, seeg_values,  'o-', linewidth=2, color='#95a5a6', label='SEEG-NET')
ax.fill(angles, seeg_values,  alpha=0.08, color='#95a5a6')
ax.plot(angles, best_values,  'o-', linewidth=2, color=BASELINE_COLOR, label='TimeConv-CNN')
ax.fill(angles, best_values,  alpha=0.12, color=BASELINE_COLOR)
ax.plot(angles, our_values,   'o-', linewidth=2.5, color=OUR_COLOR, label='Our GAT v4')
ax.fill(angles, our_values,   alpha=0.15, color=OUR_COLOR)
ax.set_title('Performance Radar Chart', fontweight='bold', pad=25, fontsize=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=11)
ax.set_facecolor('#FFFFFF')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_radar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 07_radar_chart.png")

# ==========================================
# GRAPH 8 — Confusion Matrix (Best Fold 2)
# ==========================================
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('#F8F9FA')
cm     = np.array([[113603, 3311], [13816, 29304]])
im     = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)
classes    = ['Normal (0)', 'SOZ (1)']
tick_marks = np.arange(2)
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes, fontsize=12)
ax.set_yticklabels(classes, fontsize=12)
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]:,}',
                ha='center', va='center', fontsize=13, fontweight='bold',
                color='white' if cm[i,j] > thresh else 'black')
ax.set_title('Confusion Matrix — Best Fold 2 (F1=0.7667)', fontweight='bold', pad=12)
ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
ax.set_facecolor('#FFFFFF')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 08_confusion_matrix.png")

# ==========================================
# GRAPH 9 — Model Progression
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#F8F9FA')
versions = ['GAT v2', 'GAT v3', 'GAT v4']
v_f1     = [0.4881,   0.6050,   0.6130]
v_rec    = [0.6068,   0.6549,   0.6671]
v_auc    = [0.7477,   0.8272,   0.8315]
x     = np.arange(len(versions))
width = 0.25
ax.bar(x - width, v_f1,  width, label='F1',     color='#E8734A')
ax.bar(x,         v_rec, width, label='Recall', color='#2ecc71')
ax.bar(x + width, v_auc, width, label='AUC',    color='#9b59b6')
for i, (f, r, a) in enumerate(zip(v_f1, v_rec, v_auc)):
    ax.text(i - width, f  + 0.01, f'{f:.3f}',  ha='center', fontsize=9, fontweight='bold')
    ax.text(i,         r  + 0.01, f'{r:.3f}',  ha='center', fontsize=9, fontweight='bold')
    ax.text(i + width, a  + 0.01, f'{a:.3f}',  ha='center', fontsize=9, fontweight='bold')
ax.set_title('Our Model Progression: v2 → v3 → v4', fontweight='bold', pad=12)
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(versions, fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.legend(fontsize=11)
ax.set_facecolor('#FFFFFF')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_model_progression.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 09_model_progression.png")

print("\nAll 9 graphs saved in:", OUTPUT_DIR)
