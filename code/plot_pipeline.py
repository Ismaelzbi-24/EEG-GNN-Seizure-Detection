import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1]
BG = "#f8f9fa"
INK = "#1f2933"
MUTED = "#4b5563"

COLS = [
    ("1 · Data & Preprocessing", "#3b7dd8", "#e8f0fb"),
    ("2 · Graph Construction",   "#e8744f", "#fcebe4"),
    ("3 · GAT v4 Model",         "#9b59b6", "#f2e8f7"),
    ("4 · Evaluation & Output",  "#27ae60", "#e3f5ea"),
]

fig = plt.figure(figsize=(18, 9.2), dpi=110)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 180)
ax.set_ylim(0, 92)
ax.axis("off")

ax.text(90, 87.2, "EEG-GNN Pipeline: from raw iEEG to Seizure Onset Zone ranking",
        ha="center", va="center", fontsize=19, fontweight="bold", color=INK)
ax.text(90, 83.2, "Omni-iEEG multicenter benchmark  ·  310 patients  ·  13,704 brain graphs  ·  Graph Attention Network",
        ha="center", va="center", fontsize=11.5, color=MUTED)

CW, GAP, X0 = 40.0, 4.6, 3.0
TOP, BOT = 78.5, 3.0
xs = [X0 + i * (CW + GAP) for i in range(4)]


def box(x, y, w, h, title, body, edge, fill, tsize=11.5, bsize=10, tc=INK, bc=MUTED):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                       linewidth=1.6, edgecolor=edge, facecolor=fill)
    ax.add_patch(p)
    if body:
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + (h - 4.2) / 2, body, ha="center", va="center",
                fontsize=bsize, color=bc, linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", color=tc)


def down(x, y1, y2, color):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.6, color=color))


# column panels + headers
for (title, c, light), x in zip(COLS, xs):
    ax.add_patch(FancyBboxPatch((x, BOT), CW, TOP - BOT, boxstyle="round,pad=0,rounding_size=2",
                                linewidth=0, facecolor="white"))
    ax.add_patch(FancyBboxPatch((x, TOP - 6.5), CW, 6.5, boxstyle="round,pad=0,rounding_size=2",
                                linewidth=0, facecolor=c))
    ax.text(x + CW / 2, TOP - 3.25, title, ha="center", va="center",
            fontsize=13.5, fontweight="bold", color="white")

# arrows between columns
for i in range(3):
    xa = xs[i] + CW + 0.4
    xb = xs[i + 1] - 0.4
    ax.add_patch(FancyArrowPatch((xa, 42), (xb, 42), arrowstyle="simple,head_width=1.1,head_length=1.0",
                                 mutation_scale=14, color="#9aa5b1"))


def stack(ci, items, gap=3.2):
    """items: list of (title, body, height, style) top to bottom."""
    _, c, light = COLS[ci]
    x = xs[ci] + 2.5
    w = CW - 5
    y = TOP - 9.5
    centers = []
    for k, it in enumerate(items):
        t, b, h = it[:3]
        style = it[3] if len(it) > 3 else "normal"
        y -= h
        if style == "accent":
            box(x, y, w, h, t, b, c, c, tsize=12.5, bsize=11, tc="white", bc="white")
        elif style == "note":
            box(x, y, w, h, t, b, "#cbd2d9", BG, tsize=10.5, bsize=9.5)
        else:
            box(x, y, w, h, t, b, c, light)
        if k:
            down(x + w / 2, prev_bottom, y + h, c)
        prev_bottom = y
        y -= gap
    return


stack(0, [
    ("Raw iEEG recordings", "EDF files from the Omni-iEEG benchmark\n310 patients · 8 epilepsy centers", 12.5),
    ("Signal cleaning", "Resample to 1000 Hz\nBand-pass filter 1–300 Hz", 11.5),
    ("Channel selection", "Drop flat / NaN channels\nKeep up to 64 channels", 11.5),
    ("Sliding windows", "60 s windows · 50% overlap (30 s step)\nEach window becomes one graph", 11.5),])

stack(1, [
    ("Nodes = EEG channels", "18 features per channel", 8.5),
    ("Node features (18)",
     "Time (8): mean, std, variance, skewness,\nkurtosis, RMS, line length, zero crossings\n"
     "Band power (7): δ, θ, α, β, γ,\nripple 80–150 Hz, HFO 80–300 Hz\n"
     "Spectral (3): total power, peak frequency,\nspectral entropy", 22),
    ("Edges = functional connectivity", "Connect two channels when\n|Pearson r| > 0.3", 10.5),
    ("Normalisation", "StandardScaler, clipped to ±10\n→ 13,704 graphs", 10.5),
])

stack(2, [
    ("GAT layers 1 & 2", "8 attention heads · 192 hidden units\nSkip connections + BatchNorm", 11),
    ("GAT layer 3", "1 head · final node embeddings", 8.5),
    ("Node + graph context", "Node embedding ⊕ global mean pool", 8.5),
    ("Classifier (MLP)", "128 → 64 → 2 · softmax\nSOZ probability for every channel", 11),
    ("Training", "Focal Loss (α = 0.85, γ = 2) for rare SOZ nodes\nAdam lr 3e-4 · 150 epochs · batch 16", 11, "note"),
])

stack(3, [
    ("Patient-wise GroupKFold", "5 folds · no patient appears in\nboth training and test sets", 10.5),
    ("Recall-constrained threshold", "Maximise F1 subject to Recall ≥ 0.65\n(fallback: Youden's J)", 10.5),
    ("SOZ ranking", "Average probabilities across windows\n→ per-electrode ranking + heat map", 10.5),
    ("GAT v4 results", "ROC-AUC 0.8315\nF1 0.613  ·  Recall 0.667", 11, "accent"),
    ("Clinical use", "Ranked list of likely SOZ electrodes to\nsupport, not replace, the clinician", 9.5, "note"),
], gap=2.6)

fig.savefig(OUT, dpi=110, facecolor=BG)
print("saved", OUT)
