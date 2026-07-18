import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# DATA
# ==========================================

# Fold-wise results
folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
fold_f1        = [0.6376, 0.7667, 0.5000, 0.6498, 0.5108]
fold_precision = [0.6182, 0.8387, 0.4049, 0.6456, 0.4152]
fold_recall    = [0.6583, 0.7060, 0.6534, 0.6540, 0.6638]
fold_auc       = [0.8495, 0.8924, 0.7575, 0.8582, 0.7997]
fold_acc       = [0.8627, 0.8722, 0.7387, 0.8555, 0.7718]

# Paper baselines + our models
models     = ['SEEG-NET', 'CLAP', 'TimeConv\nCNN', 'GAT v2\n(Ours)', 'GAT v3\n(Ours)', 'GAT v4\n(Ours)']
f1_scores  = [0.5259,     0.6009, 0.6469,          0.4881,           0.6050,            0.6130]
precisions = [0.5790,     0.5936, 0.6259,           0.4143,           0.5730,            0.5845]
recalls    = [0.7169,     0.6997, 0.7454,           0.6068,           0.6549,            0.6671]
aucs       = [0.7850,     0.7684, 0.8061,           0.7477,           0.8272,            0.8315]

# Colors
BASELINE_COLOR = '#5B8DB8'
OUR_COLOR      = '#E8734A'
COLORS = [BASELINE_COLOR, BASELINE_COLOR, BASELINE_COLOR,
          OUR_COLOR, OUR_COLOR, OUR_COLOR]

plt.rcParams.update({
    'font.family'  : 'DejaVu Sans',
    'font.size'    : 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi'   : 150,
})

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#F8F9FA')
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ==========================================
# PLOT 1 — F1 Score Comparison
# ==========================================
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(models, f1_scores, color=COLORS, edgecolor='white',
               linewidth=1.2, width=0.6)

for bar, val in zip(bars, f1_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9,
             fontweight='bold')

ax1.axhline(y=0.6469, color='#2ecc71', linestyle='--',
            linewidth=1.5, alpha=0.7, label='TimeConv-CNN best (0.6469)')
ax1.set_title('F1-Score Comparison', fontweight='bold', pad=10)
ax1.set_ylabel('F1-Score')
ax1.set_ylim(0, 0.92)
ax1.legend(fontsize=8)
ax1.set_facecolor('#FFFFFF')
ax1.tick_params(axis='x', labelsize=8.5)
ax1.grid(axis='y', alpha=0.3)

# ==========================================
# PLOT 2 — Recall Comparison
# ==========================================
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(models, recalls, color=COLORS, edgecolor='white',
                linewidth=1.2, width=0.6)

for bar, val in zip(bars2, recalls):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9,
             fontweight='bold')

ax2.axhline(y=0.7454, color='#2ecc71', linestyle='--',
            linewidth=1.5, alpha=0.7, label='TimeConv-CNN best (0.7454)')
ax2.set_title('Recall Comparison\n(Clinical Priority)', fontweight='bold', pad=10)
ax2.set_ylabel('Recall')
ax2.set_ylim(0, 0.92)
ax2.legend(fontsize=8)
ax2.set_facecolor('#FFFFFF')
ax2.tick_params(axis='x', labelsize=8.5)
ax2.grid(axis='y', alpha=0.3)

# ==========================================
# PLOT 3 — AUC Comparison
# ==========================================
ax3 = fig.add_subplot(gs[0, 2])
bars3 = ax3.bar(models, aucs, color=COLORS, edgecolor='white',
                linewidth=1.2, width=0.6)

for bar, val in zip(bars3, aucs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9,
             fontweight='bold')

ax3.axhline(y=0.8061, color='#2ecc71', linestyle='--',
            linewidth=1.5, alpha=0.7, label='TimeConv-CNN best (0.8061)')
ax3.set_title('ROC-AUC Comparison\n(Our Model Best!)', fontweight='bold', pad=10)
ax3.set_ylabel('ROC-AUC')
ax3.set_ylim(0.65, 0.97)
ax3.legend(fontsize=8)
ax3.set_facecolor('#FFFFFF')
ax3.tick_params(axis='x', labelsize=8.5)
ax3.grid(axis='y', alpha=0.3)

# ==========================================
# PLOT 4 — Fold-wise F1 with error
# ==========================================
ax4 = fig.add_subplot(gs[1, 0])
x = np.arange(len(folds))

bars4 = ax4.bar(x, fold_f1, color='#E8734A', edgecolor='white',
                linewidth=1.2, width=0.5, alpha=0.85)

for bar, val in zip(bars4, fold_f1):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9,
             fontweight='bold')

mean_f1 = np.mean(fold_f1)
ax4.axhline(y=mean_f1, color='#2c3e50', linestyle='--',
            linewidth=2, label=f'Mean F1 = {mean_f1:.3f}')
ax4.set_title('Fold-wise F1 Score (5-Fold CV)', fontweight='bold', pad=10)
ax4.set_ylabel('F1-Score')
ax4.set_xticks(x)
ax4.set_xticklabels(folds)
ax4.set_ylim(0, 0.92)
ax4.legend(fontsize=9)
ax4.set_facecolor('#FFFFFF')
ax4.grid(axis='y', alpha=0.3)

# ==========================================
# PLOT 5 — Radar Chart
# ==========================================
ax5 = fig.add_subplot(gs[1, 1], polar=True)

categories  = ['F1', 'Precision', 'Recall', 'AUC']
N           = len(categories)
angles      = [n / float(N) * 2 * np.pi for n in range(N)]
angles     += angles[:1]

our_values  = [0.6130, 0.5845, 0.6671, 0.8315]
best_values = [0.6469, 0.6259, 0.7454, 0.8061]
our_values  += our_values[:1]
best_values += best_values[:1]

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax5.set_ylim(0.4, 0.9)
ax5.set_yticks([0.5, 0.6, 0.7, 0.8])
ax5.set_yticklabels(['0.5', '0.6', '0.7', '0.8'], fontsize=8)

ax5.plot(angles, best_values, 'o-', linewidth=2,
         color=BASELINE_COLOR, label='TimeConv-CNN')
ax5.fill(angles, best_values, alpha=0.15, color=BASELINE_COLOR)

ax5.plot(angles, our_values, 'o-', linewidth=2,
         color=OUR_COLOR, label='Our GAT v4')
ax5.fill(angles, our_values, alpha=0.15, color=OUR_COLOR)

ax5.set_title('Performance Radar\n(Our Model vs Best Baseline)',
              fontweight='bold', pad=20)
ax5.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)
ax5.set_facecolor('#FFFFFF')

# ==========================================
# PLOT 6 — Confusion Matrix (Best Fold 2)
# ==========================================
ax6 = fig.add_subplot(gs[1, 2])

cm = np.array([[113603,  3311],
               [ 13816, 29304]])

im = ax6.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax6)

classes    = ['Normal\n(0)', 'SOZ\n(1)']
tick_marks = np.arange(2)
ax6.set_xticks(tick_marks)
ax6.set_yticks(tick_marks)
ax6.set_xticklabels(classes, fontsize=11)
ax6.set_yticklabels(classes, fontsize=11)

thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax6.text(j, i, f'{cm[i, j]:,}',
                 ha='center', va='center', fontsize=11, fontweight='bold',
                 color='white' if cm[i, j] > thresh else 'black')

ax6.set_title('Confusion Matrix\n(Best Fold 2 — F1=0.7667)',
              fontweight='bold', pad=10)
ax6.set_xlabel('Predicted Label', fontweight='bold')
ax6.set_ylabel('True Label', fontweight='bold')
ax6.set_facecolor('#FFFFFF')

# ==========================================
# LEGEND + TITLE
# ==========================================
baseline_patch = mpatches.Patch(color=BASELINE_COLOR, label='Paper Baselines')
our_patch      = mpatches.Patch(color=OUR_COLOR,      label='Our GAT Models')

fig.legend(handles=[baseline_patch, our_patch],
           loc='upper center', ncol=2, fontsize=12,
           bbox_to_anchor=(0.5, 0.98),
           frameon=True, fancybox=True, shadow=True)

fig.suptitle('GATNet: SOZ Detection on Omni-iEEG Dataset — Results',
             fontsize=16, fontweight='bold', y=1.01)

plt.savefig('/home/pmilab/EEG_GNN_Project/results/results_graphs.png',
            dpi=150, bbox_inches='tight',
            facecolor='#F8F9FA')
print("Saved: results_graphs.png")
