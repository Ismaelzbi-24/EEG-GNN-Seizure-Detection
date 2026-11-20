<div align="center">
<h1>🧠 EEG-GNN: Automated Seizure Onset Zone Detection</h1>
<h3>from Intracranial Electroencephalography for Improving Epilepsy Care</h3>

[![University](https://img.shields.io/badge/University-UET%20Taxila-blue?style=flat-square)](https://www.uettaxila.edu.pk/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8315%20🏆-gold?style=flat-square)]()
[![F1-Score](https://img.shields.io/badge/F1--Score-0.6130-red?style=flat-square)]()
[![Graphs](https://img.shields.io/badge/Graphs-13%2C704-blueviolet?style=flat-square)]()
[![Patients](https://img.shields.io/badge/Patients-310-teal?style=flat-square)]()

[Documentation](#-architecture) •
[Quick Start](#-quick-start) •
[Results](#-results) •
[Community](#-community) •
[Contributing](#-how-to-contribute)

</div>

---

**Drug-resistant epilepsy (DRE)** affects **30–40% of all epilepsy patients worldwide**. For these patients, accurate localization of the **Seizure Onset Zone (SOZ)** is the most critical step before surgical intervention — yet current manual analysis of intracranial EEG (iEEG) is time-consuming, subjective, and limited to specialized centers.

**EEG-GNN** is a fully automated, graph-based deep learning framework that models the brain as a functional connectivity graph and applies a **Graph Attention Network (GAT)** to detect seizure onset zones — achieving the **highest ROC-AUC (0.8315)** among all published baseline methods on the Omni-iEEG benchmark.

---

## Prerequisites

Before running EEG-GNN locally, please ensure:

- **Python 3.8 or later** installed
- **CUDA-capable GPU** (strongly recommended; 8 GB VRAM minimum)
- **16 GB RAM** minimum (32 GB recommended for full Omni-iEEG dataset)
- **PyTorch 2.0+** with matching CUDA toolkit
- **MNE-Python 1.4+** for EEG signal processing

> 💡 **Tip:** We recommend using a dedicated virtual environment (`venv` or `conda`) to avoid dependency conflicts. The project has been tested on Ubuntu 22.04 and Python 3.10. If you are on Windows, use WSL2 for best compatibility with MNE and PyTorch Geometric.

```bash
# Check your Python version
python3 --version   # should be 3.8+

# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 🗞️ Newest Feature — Omni-iEEG Multicenter Pipeline

The latest version of EEG-GNN scales from the single-center HUP dataset to the **Omni-iEEG multicenter benchmark** (ICLR 2026) — a standardized collection of 302 patients across 8 epilepsy centers worldwide.

The new pipeline introduces:
- **18-dimensional node features** (up from 8) including HFO proxy band power (80–300 Hz)
- **60-second sliding windows** with 50% overlap for richer temporal coverage
- **Manager-Worker fault-tolerant architecture** — processes 337 patients in parallel; individual failures do not stop the pipeline
- **Recall-constrained threshold optimization** — finds the optimal threshold satisfying Recall ≥ 0.65 before maximizing F1
Dataset:    Omni-iEEG (ICLR 2026)
Patients:   337 total → 310 successfully processed (92%)
Graphs:     13,704 generated
Features:   18-dimensional per EEG channel (node)
Window:     60 seconds · 50% overlap

![Pipeline Architecture](results/graphs/09_model_progression.png)
*Figure: GAT model performance progression from v2 → v3 → v4 across F1, Recall, and ROC-AUC.*

---

## Why EEG-GNN

| Problem | Current Approach | EEG-GNN Solution |
|---------|-----------------|-----------------|
| **Subjectivity** | Manual visual iEEG review | Objective GAT-based prediction |
| **Scalability** | Single-center models | Multicenter Omni-iEEG (8 centers) |
| **Data leakage** | Random train/test splits | GroupKFold patient-level separation |
| **Class imbalance** | Standard Cross-Entropy | Focal Loss on minority SOZ class |
| **Missed SOZ** | Fixed 0.5 threshold | Recall-constrained search (≥ 0.65) |
| **Fragile pipelines** | Single process aborts all | Manager-Worker isolates failures |

Here are key highlights of EEG-GNN:

- 🧠 **Graph-based brain modeling** — iEEG channels as nodes, Pearson correlation as edges
- ⚡ **State-of-the-art ROC-AUC 0.8315** — surpasses TimeConv-CNN (0.8061), SEEG-NET (0.7850), CLAP (0.7684)
- 🏥 **Clinically aligned** — high Recall (0.6671) minimizes missed seizure onset zones
- 🔬 **Fully reproducible** — complete pipeline code, preprocessing scripts, result plots included
- 🌐 **Web app** — upload an `.edf` recording and get a per-electrode SOZ ranking with a channel × time heat map

## ✨ Architecture

EEG-GNN processes raw iEEG recordings in four stages:

1. **Data & preprocessing**: EDF recordings are resampled to 1000 Hz, band-pass filtered (1–300 Hz), cleaned of flat channels (up to 64 kept), and cut into 60 s windows with 50% overlap.
2. **Graph construction**: each window becomes a graph. Channels are nodes with 18 features (time-domain, band power including HFO 80–300 Hz, and spectral). Edges connect channels with |Pearson r| > 0.3.
3. **GAT v4 model**: three Graph Attention layers (8 heads, 192 hidden units, skip connections) combine node embeddings with graph context, and an MLP outputs an SOZ probability per channel. It is trained with Focal Loss to handle the rare SOZ class.
4. **Evaluation & output**: 5-fold patient-wise GroupKFold, a recall-constrained threshold (Recall ≥ 0.65), and per-electrode SOZ ranking averaged across windows.

![Pipeline Architecture](results/graphs/pipeline_architecture.png)

For more on EEG-GNN's architecture, explore the source code in [`code/04_train_recall_gat.py`](code/04_train_recall_gat.py).

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/Ismaelzbi-24/EEG-GNN-Seizure-Detection.git
cd EEG-GNN-Seizure-Detection

# Install dependencies
pip install -r code/requirements.txt

# Step 1: Process all patients (Manager-Worker pipeline)
python code/01_safe_runner.py

# Step 2: Merge all graph files
python code/01_merge_graphs.py

# Step 3: Train GAT v4
python code/04_train_recall_gat.py

# Step 4: Visualize results
python code/plot_results.py
```

> 💡 **Tip:** The Manager-Worker architecture processes patients in parallel and isolates per-patient failures — a crash on one patient will not stop the rest. You can safely re-run the script to resume from failed patients.

---

## 🔒 Trained Model Access

The trained **GAT v4 weights are not included in this repository**. All code needed to train the model yourself is public (see Quick Start above).

If you would like the pretrained model for research or the web app, please contact us at **ismaelzabi302@gmail.com** with your name, affiliation and intended use.

The web app ([`webapp/`](webapp/)) expects the model files in `webapp/models/` and `webapp/site/model/`. See [`webapp/README.md`](webapp/README.md) for setup.

---

## 📊 Results

### Model Progression (v2 → v4)

![Model Progression](results/graphs/09_model_progression.png)

| Version | F1-Score | Recall | Precision | ROC-AUC | Key Change |
|---------|----------|--------|-----------|---------|------------|
| GAT v2 | 0.4881 | 0.6068 | 0.4143 | 0.7477 | Node-level GAT · Cross-Entropy Loss |
| GAT v3 | 0.6050 | 0.6549 | 0.5730 | 0.8272 | 3-layer GAT · Focal Loss · graph context |
| **GAT v4** | **0.6130** | **0.6671** | **0.5845** | **🏆 0.8315** | Recall-constrained threshold |

### F1-Score Comparison
![F1 Comparison](results/graphs/01_f1_comparison.png)

### Recall Comparison
![Recall Comparison](results/graphs/02_recall_comparison.png)

### ROC-AUC Comparison
![ROC-AUC Comparison](results/graphs/04_auc_comparison.png)

### Fold-wise Cross-Validation
![Foldwise F1](results/graphs/05_foldwise_f1.png)

### All Metrics Overview
![All Metrics](results/graphs/06_foldwise_all_metrics.png)

### Confusion Matrix
![Confusion Matrix](results/graphs/08_confusion_matrix.png)

### Per-Fold Results — GAT v4

| Fold | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|------|----------|----------|-----------|--------|---------|
| Fold 1 | 0.8627 | 0.6376 | 0.6182 | 0.6583 | 0.8495 |
| Fold 2 | 0.8722 | **0.7667** | **0.8387** | 0.7060 | **0.8924** |
| Fold 3 | 0.7387 | 0.5000 | 0.4049 | 0.6534 | 0.7575 |
| Fold 4 | 0.8555 | 0.6498 | 0.6456 | 0.6540 | 0.8582 |
| Fold 5 | 0.7718 | 0.5108 | 0.4152 | 0.6638 | 0.7997 |
| **Mean ± Std** | **0.8202 ± 0.054** | **0.6130 ± 0.099** | **0.5845 ± 0.162** | **0.6671 ± 0.020** | **0.8315 ± 0.039** |

### Comparison with Published Baselines

| Model | Type | F1 | Recall | ROC-AUC |
|-------|------|----|--------|---------|
| SEEG-NET | Segment-based | 0.5259 | 0.7169 | 0.7850 |
| CLAP | Segment-based | 0.6009 | 0.6997 | 0.7684 |
| TimeConv-CNN | Segment-based | 0.6469 | 0.7454 | 0.8061 |
| **Our GAT v4** | **Graph-based** | **0.6130** | **0.6671** | **🏆 0.8315** |

> 🏆 **EEG-GNN achieves the highest ROC-AUC (0.8315) across ALL methods** — surpassing TimeConv-CNN (0.8061). Best single fold: F1 = 0.7667.

---

## 🌟 Stay Ahead

⭐ **Star EEG-GNN on GitHub** to stay updated on new releases, dataset support, and model improvements.

[![Star on GitHub](https://img.shields.io/github/stars/Ismaelzbi-24/EEG-GNN-Seizure-Detection?style=social)](https://github.com/Ismaelzbi-24/EEG-GNN-Seizure-Detection)

**Upcoming work:**
- 🔜 Brain Network Transformer (BNT) extension for richer long-range dependencies
- 🔜 SHAP-based attention visualization for clinical interpretability
- 🔜 Multi-label SOZ classification (resected vs preserved vs SOZ)
- 🔜 Real-time iEEG processing support
- ✅ Web-based clinical decision support interface (see [`webapp/`](webapp/))

---

## 💬 Community

You can join the following groups or channels to discuss or ask questions about EEG-GNN:

- 🐛 **Report a bug or request a feature** — [Open a GitHub Issue](https://github.com/Ismaelzbi-24/EEG-GNN-Seizure-Detection/issues)
- 💬 **Ask a question or start a discussion** — [GitHub Discussions](https://github.com/Ismaelzbi-24/EEG-GNN-Seizure-Detection/discussions)
- 📧 **Contact the team directly** — ismaelzabi302@gmail.com
- 🎓 **UET Taxila — Dept. of Software Engineering** — [uettaxila.edu.pk](https://www.uettaxila.edu.pk/)

> 💡 **Tip:** Before opening an issue, please check existing issues and discussions — your question may already be answered!

---

## 👥 How to Contribute

We welcome contributions from the community! If you have found a problem in EEG-GNN or want to improve the pipeline, please open a [GitHub Issue](https://github.com/Ismaelzbi-24/EEG-GNN-Seizure-Detection/issues). To contribute code please see the [Contributing Guide](CONTRIBUTING.md) and we will gladly assist with your contribution.

1. **Fork** this repository
2. **Create a branch** for your feature or fix:
```bash
git checkout -b feature/your-feature-name
```
3. **Commit** your changes clearly:
```bash
git commit -m "feat: add spectral coherence edge weights"
```
4. **Push** and open a **Pull Request**

**Good first issues:**
- Add support for additional EEG datasets (Mayo, FNUSA)
- Implement coherence-based graph edge weights
- Add SHAP-based feature importance visualization
- Write unit tests for preprocessing scripts
- Improve documentation and usage examples

---

## 📁 Project Structure

```
EEG-GNN-Seizure-Detection/
│
├── code/                            # Active pipeline (run in this order)
│   ├── 01_safe_runner.py            # Manager: runs the worker per patient
│   ├── 01_single_patient.py         # Worker: per-patient graph building
│   ├── 01_merge_graphs.py           # Merge all per-patient .pt files
│   ├── 02_train_node_gat.py         # GAT v2 (node-level, most likely used)
│   ├── 02_train_omni_gat.py         # GAT v2 (patient-level variant)
│   ├── 03_train_improved_gat.py     # GAT v3 with Focal Loss
│   ├── 04_train_recall_gat.py       # GAT v4 recall-constrained (final)
│   ├── plot_results.py              # Result figures
│   ├── plot_individual.py           # Individual metric figures
│   ├── ieegnet_app.py               # SOZ web app (uses v4 model)
│   ├── omni_ieeg_meta.csv           # Dataset metadata
│   ├── requirements.txt             # Python dependencies
│   ├── archive/                     # Superseded / backup / test scripts
│   └── legacy_hup/                  # Early single-centre HUP experiments
│
├── logs/                            # Processing + training logs, diagnostic report
│
├── results/
│   ├── graphs/                      # All result plots + pipeline diagram
│   └── omni_gat*/                   # Per-version summaries (+ v4 scaler)
│
├── webapp/                          # SOZ detection web app
│   ├── api/                         # FastAPI server + preprocessing/GAT v4 pipeline
│   ├── site/                        # Website (runs the model in the browser)
│   ├── models/                      # Trained weights go here (available on request)
│   ├── test_data/                   # Synthetic EDF for testing
│   ├── tools/                       # Model export + browser/Python checks
│   ├── Dockerfile
│   ├── run_on_colab.ipynb
│   └── README.md
│
├── AUDIT_REPORT.md                  # Code and results audit
├── .gitignore
└── README.md
```

## 👨‍🎓 Team

| Name | Roll No | Role |
|------|---------|------|
| **Atif Khan** | 22-SE-79 | Graph construction & preprocessing |
| **Syed Ismael** | 22-SE-95 | GAT model development & evaluation |
| **Laraib Kanwal** | 22-SE-101 | Feature extraction & visualization |

**Supervisor:** Dr. Saima Zareen — Assistant Professor, Dept. of Software Engineering, UET Taxila

> 🌟 **Notable:** Co-Supervisor Dr. Syed Muhammad Anwar is currently Principal Investigator 
> at Children's National Hospital and Associate Professor at George Washington University 
> School of Medicine. He previously served as Associate Professor (Tenured) at UET Taxila 
> and is a Fulbright Research Fellow — bringing world-class expertise in surgical planning, 
> medical imaging, and computer vision to this project.

---

## 📝 Citation

If you use EEG-GNN in your research, please cite:

```bibtex
@thesis{eeg_gnn_uet2026,
  title     = {Automated Seizure Onset Zone Detection from Intracranial
               Electroencephalography for Improving Epilepsy Care},
  author    = { Syed Ismael, Atif Khan, Laraib},
  school    = {University of Engineering and Technology, Taxila},
  department= {Department of Software Engineering},
  year      = {2026},
  type      = {Bachelor's Final Year Project}
}
```

---

## 📜 License

EEG-GNN is under the [Apache 2.0 License](LICENSE).

---

<div align="center">

**⭐ Star this repo if it helped your research!**

Made with ❤️ at **UET Taxila** — Department of Software Engineering — FYP 2026

</div>
