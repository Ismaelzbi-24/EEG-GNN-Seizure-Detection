# AUDIT REPORT — EEG-GNN SOZ Detection (Omni-iEEG)

Audit date: 2026-09-24. Scope: every file in this repository (35 code/log/data files, 3 result CSVs, 1 scaler pickle, 11 PNGs, README).
Method: read-only. No code or data was modified. Checks were run on the logs, `omni_ieeg_meta.csv`, `diagnostic_report.json` and `scaler.pkl` using standalone scripts (stdlib only; numpy/torch are not installed on this machine). The graph dataset (`omni_ieeg_graphs.pt`), the raw Omni-iEEG data and the model checkpoints are **not** in the repo (they are git-ignored, see [.gitignore](.gitignore)), so nothing was re-trained.

**No manuscript file (pdf/docx/tex/md) exists in this workspace.** Task 3 therefore compares the code against [README.md](README.md). Please add the manuscript to the workspace and the comparison can be redone against it.

Line references are `file:line` relative to the repo root.

---

## 0. Executive summary (read this first)

| # | Finding | Status |
|---|---|---|
| 1 | The decision threshold (Recall ≥ 0.65, then max F1) is searched **on the test fold** | **LEAKAGE** |
| 2 | The best epoch/checkpoint is chosen by **test-fold F1** (16 candidate epochs per fold) | **LEAKAGE** |
| 3 | StandardScaler is fit on **all 797,837 nodes (train + test)**; confirmed from `scaler.pkl` | **LEAKAGE** (mild, unsupervised) |
| 4 | v2→v3→v4 were all developed and compared on the same test folds; there is no untouched hold-out set | **LEAKAGE** (design-level) |
| 5 | SOZ label parser reads the BIDS `status` column first, and treats any value containing `'bad'` (or `'1'`, `'yes'`, `'onset'`...) as SOZ | **CONCERN — potentially critical** (labels may partly be "bad channel" flags) |
| 6 | GroupKFold is by patient ID and patient-disjoint at ID level | OK |
| 7 | ...but 5 likely same-patient pairs exist under two IDs (Multicenter vs sourcesink), and they land in different folds ~83% of the time | **CONCERN** |
| 8 | Fold partition is not reproducible (depends on Python `set()` hash order); v3 and v4 were evaluated on **different** patient partitions | **CONCERN** |
| 9 | v3/v4 metrics are pooled per channel per 60-s window within a fold. There is **no patient-level aggregation**, although the README claims it | **CONCERN** |
| 10 | Baselines (SEEG-NET, CLAP, TimeConv-CNN) are **hard-coded numbers**, not re-run on our splits, so the comparison is not like-for-like | **CONCERN** |
| 11 | README ROC-AUC std is wrong: 0.039 is written, the correct value is 0.047 | Mismatch |
| 12 | With the final epoch (no test-based epoch selection), v4 AUC = 0.8256 and v3 AUC = 0.8281, so v4 is not better than v3 on AUC | Fact from logs |

Bottom line: the reported numbers (F1 0.6130, Recall 0.6671, AUC 0.8315) are **optimistically biased**, and the "state-of-the-art / highest ROC-AUC" claim is **not supported** as currently evaluated. Fixes are listed in §5 (Q&A) and §6.

---

## 1. Pipeline Summary

### 1.1 Which scripts produced the reported results

| Stage | Script actually used | Evidence |
|---|---|---|
| Per-patient processing | [code/01_safe_runner.py](code/01_safe_runner.py) → [code/01_single_patient.py](code/01_single_patient.py) | `safe_runner_log.txt` has 337 entries. Metadata `dataset='unknown'` for Detroit/UCLA can only come from the source mapping in [code/01_single_patient.py:201-210](code/01_single_patient.py#L201-L210), which has no `'openieeg'` branch |
| Merge | [code/01_merge_graphs.py](code/01_merge_graphs.py) | column layout of `omni_ieeg_meta.csv` matches [code/01_merge_graphs.py:24-30](code/01_merge_graphs.py#L24-L30) |
| GAT v2 | **Most likely** [code/02_train_node_gat.py](code/02_train_node_gat.py) (node-level). No v2 log exists | Both `02_*` scripts write to `results/omni_gat/summary.csv` ([code/02_train_node_gat.py:59](code/02_train_node_gat.py#L59), [code/02_train_omni_gat.py:55](code/02_train_omni_gat.py#L55)). v2 precision of 0.414 is only plausible for the node-level task (21% positives). The patient-level task has 84% positives, where predicting all-positive already gives F1 = 0.912 |
| GAT v3 | [code/03_train_improved_gat.py](code/03_train_improved_gat.py) | `training_v3_log.txt` config matches |
| GAT v4 | [code/04_train_recall_gat.py](code/04_train_recall_gat.py) | `training_v4_log.txt` config matches |
| Figures | [code/plot_individual.py](code/plot_individual.py), [code/plot_results.py](code/plot_results.py) | all numbers are **hard-coded**, not read from results |

Not used for the reported results: [code/archive/01_process_omni_ieeg.py](code/archive/01_process_omni_ieeg.py) is a byte-for-byte copy of the diagnostic script. It never saves graphs, even though the README calls it the "Main preprocessing script". Also unused: `01_final.py` (its run in `main_processing_log.txt` stops at 54/328), `01_clean.py`, `*_BACKUP*.py`, and the legacy single-centre HUP scripts (`preprocess.py`, `process_dataset_v4.py`, `train_gnn.py`, `train_gat_graph_features_polished.py`), which use 8 statistical features, 2–5 s windows and HUP surgical outcome labels.

### 1.2 Data: patients attempted / succeeded / failed

| Count | Meaning | Source |
|---|---|---|
| **302** | Omni-iEEG size as stated in the README ([README.md:54](README.md#L54)) and in an old docstring ([code/archive/01_final.py](code/archive/01_final.py) header, "Dataset: 302 patients"). Not derived from code | text only |
| **337** | `sub-*` directories found in the local data folder ([code/01_safe_runner.py:8-13](code/01_safe_runner.py#L8-L13)). Logged as "Total patients found: 337" | `safe_runner_log.txt`, `diagnostic_log.txt` |
| 309 OK / 7 FAIL / 21 TIMEOUT | Safe-runner outcome (309+7+21 = 337) | `logs/safe_runner_log.txt` lines 1–337 |
| **310** | Patients in `omni_ieeg_meta.csv` = the 309 OK **plus `sub-openieegUCLA21`**, which is logged as TIMEOUT but still has a `.pt` file | reconciliation script (§1.2 note) |
| **13,704** | Total graphs = sum of `n_graphs` in `omni_ieeg_meta.csv`. Confirmed by "Total graphs: 13704" in both training logs | `omni_ieeg_meta.csv`, `training_v4_log.txt:14` |

Failure reasons:
- **7 FAIL (no valid graphs):** `Multicenterjh107`, `Multicenterpt1`, `Multicenterumf002–005`, `hupHUP112`. The worker exits 1 when no graph is produced ([code/01_single_patient.py:218-220](code/01_single_patient.py#L218-L220)). The stderr tail is empty, so the precise reason is UNKNOWN. Possible causes are: no EDF found, < 3 valid channels, recording < 60 s, or an exception swallowed at [code/01_single_patient.py:177-179](code/01_single_patient.py#L177-L179).
- **21 TIMEOUT:** `hupHUP132`, 17 UCLA patients, and `zurich04–06` ([code/01_safe_runner.py:21](code/01_safe_runner.py#L21), `timeout=120`). The diagnostic run (`diagnostic_report.json`) processed **330/337 successfully with 0 timeouts**, including all 20 of these. So these patients were excluded for **compute time only**, not data quality. This biases the cohort against long/large recordings, and UCLA drops from ~51 to 34 patients.
- Why UCLA21 is in the dataset despite TIMEOUT: UNKNOWN. Most likely the `.pt` was written just before the kill, or came from an earlier run (the worker skips existing files, [code/01_single_patient.py:197-199](code/01_single_patient.py#L197-L199)).

Cohort composition (from `omni_ieeg_meta.csv`):

| Source tag | Patients | Graphs | Patients with ≥1 SOZ-labelled channel |
|---|---|---|---|
| `unknown` (actually openieeg Detroit 135 + UCLA 34) | 169 | 7,546 | 139 |
| `hup` | 56 | 2,513 | 48 |
| `sourcesink` | 37 | 608 | 34 |
| `openieeg` (actually the "Multicenter" folders) | 31 | 798 | 22 |
| `zurich` | 17 | 2,239 | 17 |
| **Total** | **310** | **13,704** | **260 (83.9%)** |

- Graphs per patient: min 1, median 17.5, max 317. Nine `rns` patients have 1 graph each; Zurich has 16% of all graphs from 5% of patients.
- Channels per patient: 15–64. **266/310 patients were truncated at 64 channels.**
- Node labels: 169,849 SOZ (21.3%) / 627,988 non-SOZ, counted over all node-windows ([training_v4_log.txt:17-18](logs/training_v4_log.txt)).
- **Likely duplicate subjects under two IDs:** `Multicenterjh103`/`sourcesinkjh103`, `Multicenterjh105`/`sourcesinkjh105`, `Multicenterpt01`/`sourcesinkpt1`, `Multicenterpt2`/`sourcesinkpt2`, `Multicenterpt3`/`sourcesinkpt3`. Also `Multicenterumf002–005` (failed) mirror `sourcesinkumf002–005` (included), and the Multicenter folder contains both `pt01` and `pt1`. Whether these are the same people is UNKNOWN from the code, but the naming strongly suggests it. This probably also explains part of the 337-vs-302 gap.

### 1.3 Preprocessing (all in [code/01_single_patient.py](code/01_single_patient.py))

| Step | What the code does | Line |
|---|---|---|
| Files | **All** `*.edf` in the `ieeg/` folder, whatever the BIDS `task` (filenames seen in code include `task-ictal`) | [123](code/01_single_patient.py#L123) |
| Sampling rate | Resample every file to **1000 Hz** with `raw.resample` (up- or down-sampling) | [23](code/01_single_patient.py#L23), [134-136](code/01_single_patient.py#L134-L136) |
| Filter | Band-pass **1–300 Hz**, FIR (firwin), applied after resampling | [137](code/01_single_patient.py#L137) |
| Notch | **None** | — |
| Re-referencing | **None** (uses whatever reference is in the EDF) | — |
| Channel type selection | **None**: every EDF channel, including any ECG/EKG/DC/scalp channels, becomes a node if not flat | — |
| Bad channels / artifacts | Drop a channel only if all-NaN or std ≤ 1e-10 (flat). No artifact rejection. BIDS `status=bad` is **not** used for exclusion; it is turned into label 1 (see §1.6) | [141-143](code/01_single_patient.py#L141-L143) |
| Channel cap | Keep the **first 64** valid channels in EDF order | [146](code/01_single_patient.py#L146) |
| Window | **60 s**, step **30 s** (50% overlap). `range(0, N - win, step)` drops a final window that would end exactly at N | [24-25](code/01_single_patient.py#L24-L25), [157-162](code/01_single_patient.py#L157-L162) |
| Normalization | None in preprocessing. At training time: a global `StandardScaler` over all nodes of all graphs, then clip to ±10 | [code/04_train_recall_gat.py:97-104](code/04_train_recall_gat.py#L97-L104) |
| channels.tsv fallback | If the run-specific TSV is missing, uses the **first** `*channels.tsv` in the folder, which may belong to a different run | [128-131](code/01_single_patient.py#L128-L131) |

### 1.4 Node features — exactly 18, in this order ([code/01_single_patient.py:37-69](code/01_single_patient.py#L37-L69))

Each feature is computed per channel per 60-s window (60,000 samples at 1000 Hz). Band powers use Welch PSD (`nperseg=1000` = 1 s, scipy defaults: Hann window, 50% overlap, 1 Hz resolution) integrated with `np.trapz` over the **inclusive** range `fmin ≤ f ≤ fmax` ([32-35](code/01_single_patient.py#L32-L35)). Units are volts (MNE default).

| # | Feature | Computation |
|---|---|---|
| 0 | mean | `np.mean` |
| 1 | std | `np.std` |
| 2 | variance | `np.var` |
| 3 | skewness | `scipy.stats.skew` |
| 4 | kurtosis | `scipy.stats.kurtosis` (Fisher/excess) |
| 5 | RMS | `sqrt(mean(x²))` |
| 6 | line length | `sum(|diff(x)|)`, **not** normalized by length |
| 7 | zero crossings | count of sign changes |
| 8 | delta power | 0.5–4 Hz (effectively 1–4 Hz after the 1 Hz high-pass) |
| 9 | theta power | 4–8 Hz |
| 10 | alpha power | 8–13 Hz |
| 11 | beta power | 13–30 Hz |
| 12 | gamma power | 30–80 Hz |
| 13 | high-gamma power | 80–150 Hz |
| 14 | **HFO proxy power** | **80–300 Hz** |
| 15 | total power | trapz over the whole PSD (0–500 Hz) + 1e-10 |
| 16 | peak frequency | argmax of the PSD (mean over dataset is 3.6 Hz, from `scaler.pkl`) |
| 17 | spectral entropy | `-Σ p log2 p` with `p = psd / total_p`. `total_p` is a trapz integral, so p does not sum to 1; this is not a normalized Shannon entropy |

HFO-proxy / Nyquist check:
- After resampling to 1000 Hz the Nyquist frequency is 500 Hz, so a 300 Hz upper edge is numerically valid.
- **However, resampling up cannot create content that was never recorded.** For any file recorded below 600 Hz, the 80–300 Hz band is partly or wholly empty. The repo's own comment says HUP was recorded at 500 Hz ([code/legacy_hup/preprocess.py:15](code/legacy_hup/preprocess.py#L15), Nyquist 250 Hz). The original sampling rate of each centre/file is **not** logged anywhere → UNKNOWN.
- There is no notch filter, so line-noise harmonics (100/150/200/250 Hz for 50 Hz mains; 120/180/240 Hz for 60 Hz) fall inside the high-gamma and HFO bands.
- This is band power, not HFO detection (no event detector). "HFO proxy" is the accurate wording; "HFO detection" would not be.
- Redundancy: std ≈ RMS (mean ≈ 0 after high-pass; identical means in `scaler.pkl`), variance = std², total power ≈ variance (Parseval), and high-gamma ⊂ HFO band. So the effective feature dimension is below 18.

### 1.5 Edges ([code/01_single_patient.py:71-85](code/01_single_patient.py#L71-L85))

- Pearson correlation of the full-band (1–300 Hz) 60-s window between every channel pair (`np.corrcoef`); NaN → 0.
- An edge is kept if **|r| > 0.3** (absolute value, so the sign is discarded) ([78](code/01_single_patient.py#L78)).
- Undirected (both directions stored). No top-k. The graph is not fully connected.
- **Unweighted:** the correlation value is not stored (no `edge_attr`) and is never passed to `GATConv`.
- Self-loops are not stored in `edge_index`, but `GATConv` adds them by default (`add_self_loops=True`).
- Fallback: if no pair passes the threshold, a **ring graph** in channel order is created ([81-84](code/01_single_patient.py#L81-L84)).

### 1.6 Labels ([code/01_single_patient.py:87-112](code/01_single_patient.py#L87-L112))

- **Per channel** (`node_y`), fixed for the whole patient: **every window of a patient carries identical node labels**. SOZ status is static and does not come from the window's content.
- Name column: first found of `name, channel, ch_name, Name`.
- Label column: first found of **`status`, `soz`, `SOZ`, `seizure_onset_zone`, `type`** ([97](code/01_single_patient.py#L97)). `status` is a standard BIDS column meaning channel quality (`good`/`bad`). If a TSV has both `status` and `soz`, **the `status` column is used and `soz` is ignored.**
- Positive if the lower-cased value **contains** any of `soz, bad, seizure, onset, patholog, 1, yes` ([105-106](code/01_single_patient.py#L105-L106)):
  - `bad` → SOZ = 1. A bad-quality channel would be labelled SOZ.
  - Substring matching: `non-soz`, `nsoz`, `no_onset`, `10`, `0.1` → 1. Values like `True`/`true` → 0.
  - Any exception silently produces **all-zero labels** ([110-111](code/01_single_patient.py#L110-L111)); a name mismatch between EDF and TSV also gives 0.
- Graph-level label `y = 1` if any channel is SOZ ([166](code/01_single_patient.py#L166)). It is used only by the v2 graph-level script and as the (ignored) `y` argument to `GroupKFold.split`.
- Whether the Omni-iEEG `channels.tsv` actually has a `status` column, and what values it holds, is **UNKNOWN** from the repo. This is the single most important item to check (command in §6).

### 1.7 Models

Parameter counts below are **computed analytically**, assuming PyTorch Geometric ≥ 2.3 `GATConv` (bias-free linear projection, `att_src`, `att_dst`, output bias) and counting BatchNorm affine parameters only. They were not verified by instantiation (torch is not installed here, and the PyG version is not pinned in [code/requirements.txt](code/requirements.txt)).

**GAT v4 as actually run** ([code/04_train_recall_gat.py:133-196](code/04_train_recall_gat.py#L133-L196), instantiated at [306](code/04_train_recall_gat.py#L306) with `hidden=128, heads=4, dropout=0.25`):
```
x[18] ─┬─ GATConv(18→128, heads=4, concat) → 512 → BN → ELU → Dropout(0.25)
       │     ├─ skip2 = Linear(512→128)
       │  GATConv(512→128, heads=4) → 512 → BN → ELU → Dropout(0.25)
       │  GATConv(512→128, heads=1) → 128 → BN
       └─ skip1 = Linear(18→128)
   h = ELU(gat3 + skip2 + skip1)                          [128]
   g = global_mean_pool(h)[batch]   (graph context broadcast to nodes)
   concat(h, g) [256] → Linear 128 → BN → ReLU → Drop → Linear 64 → ReLU → Drop → Linear 2
```
Attention dropout is 0.25 inside each GATConv. Output: per-node 2-class logits; softmax class-1 probability. **≈ 452,290 parameters.**

**GAT v3** ([code/03_train_improved_gat.py:133-194](code/03_train_improved_gat.py#L133-L194), `hidden=128, heads=8, dropout=0.3`): same 3-layer residual body (layer widths 1024), graph-context broadcast, classifier `Linear(256→64) → ReLU → Dropout → Linear(64→2)`. **≈ 1,359,170 parameters.**

**GAT v2 (node-level, most likely source)** ([code/02_train_node_gat.py:123-174](code/02_train_node_gat.py#L123-L174), instantiated `hidden=64, heads=4` at [284](code/02_train_node_gat.py#L284)): GATConv(18→64×4) → BN → ELU → Dropout → GATConv(256→64, 1 head) → BN → ELU(+ Linear skip) → `Linear(64→64) → ReLU → Dropout → Linear(64→2)`. No graph context. **≈ 28,098 parameters.**
(Alternative `02_train_omni_gat.py`: graph-level classification, mean-pool + 4 pooled statistics → classifier, ≈ 28,354 parameters, patient-level aggregation of window probabilities, [code/02_train_omni_gat.py:115-182](code/02_train_omni_gat.py#L115-L182).)

---

## 2. Exact Hyperparameters Table

| Item | v2 (`02_train_node_gat.py`) | v3 (`03_train_improved_gat.py`) | **v4 (`04_train_recall_gat.py`)** |
|---|---|---|---|
| Task / unit | node (channel × window) | node | node |
| GAT layers | 2 | 3 (+ residual) | 3 (+ residual) |
| Hidden | 64 | 128 | **128** (docstring says 192 — wrong) |
| Heads | 4 (layer 1), 1 | 8, 8, 1 | **4, 4, 1** (class default 8 is overridden) |
| Dropout (feature + attention) | 0.3 | 0.3 | 0.25 |
| Activation | ELU (GAT), ReLU (head) | ELU / ReLU | ELU / ReLU |
| Readout | none | mean-pool broadcast to nodes | mean-pool broadcast to nodes |
| Classifier | 64→64→2 | 256→64→2 | 256→128(BN)→64→2 |
| Params (analytic) | ≈ 28.1 k | ≈ 1.36 M | ≈ 452 k |
| Loss | weighted CE, w = [1, n_neg/n_pos] over train nodes ([289-296](code/02_train_node_gat.py#L289-L296)) | Focal α=0.75, γ=2.0 ([293](code/03_train_improved_gat.py#L293)) | **Focal α=0.85, γ=2.0** ([311](code/04_train_recall_gat.py#L311)) |
| Optimizer | Adam, wd 1e-4 | Adam, wd 1e-4 | Adam, wd 1e-4 ([307](code/04_train_recall_gat.py#L307)) |
| LR | 5e-4 | 5e-4 | **3e-4** |
| Scheduler | CosineAnnealing, T_max = 120 | Cosine, T_max = 120 | Cosine, T_max = 150 (stepped per epoch) |
| Epochs | 120 | 120 | **150** |
| Batch size (graphs) | 32 | 32 | **16** |
| Grad clipping | 1.0 | 1.0 | 1.0 |
| Early stopping | none; best of evals at epochs {1,10,20,…} **by test F1** | same | same (16 evals: 1,10,…,150) |
| Threshold | max-F1 search 0.05–0.95 on **test** | Youden J on **test** | Recall ≥ 0.65 then max F1, 0.05–0.94 on **test**; Youden fallback |
| CV | GroupKFold(5) by patient ID | same | same |
| Seeds | 42 (random, numpy, torch, cuda) | 42 | 42 ([34-39](code/04_train_recall_gat.py#L34-L39)); no cuDNN determinism flags; fold split not seeded (see §3) |
| Feature scaling | StandardScaler on all data, clip ±10 | same | same |
| Hardware (log) | UNKNOWN (no log) | RTX A5000 | RTX A5000 |

Items listed in the v4 docstring but **not implemented**: "Asymmetric class weights (1:5 instead of 1:3)", "Larger hidden size (128 → 192)", and the comment "minimum recall of 0.70" (the code uses 0.65) ([code/04_train_recall_gat.py:5-10](code/04_train_recall_gat.py#L5-L10), [243](code/04_train_recall_gat.py#L243), [254](code/04_train_recall_gat.py#L254)).

Preprocessing constants: fs 1000 Hz; band-pass 1–300 Hz; window 60 s; step 30 s; |r| > 0.3; max 64 channels; per-patient timeout 120 s.

---

## 3. Integrity Findings

### 3.1 GroupKFold patient separation — **OK** (at ID level), with two **CONCERN**s
- The split is `GroupKFold(n_splits=5).split(data_list, labels, groups=groups)`, where `groups` is the integer ID of each graph's `patient_id` ([code/04_train_recall_gat.py:76-79](code/04_train_recall_gat.py#L76-L79), [284-291](code/04_train_recall_gat.py#L284-L291)). By construction, no patient ID appears in both train and test.
- **Read-only check run:** a re-implementation of sklearn's greedy GroupKFold over `omni_ieeg_meta.csv` reproduces the logged fold sizes exactly: test graphs = 2741, 2741, 2741, 2741, 2740 (train 10963/10964, matching both logs). Every patient is in exactly one test fold (61–63 patients per fold). Also, across the five v4 folds the confusion matrices sum to 797,837 test nodes, exactly the total node count, so every node was tested exactly once.
- **CONCERN — reproducibility:** `unique_pats = list(set(patient_ids))` ([77](code/04_train_recall_gat.py#L77)) depends on Python's per-process string-hash randomization. With equal-sized groups, that changes which patients go to which fold. In 200 simulated orderings, 200 different partitions appeared. Evidence that this happened for real: v3 and v4 have **different test node counts per fold** (fold 1: 157,957 vs 158,226; fold 3: 167,062 vs 156,350). So **v3 and v4 were not evaluated on the same patients**, and `SEED=42` does not fix this.
- **CONCERN — duplicate subjects:** the 5 likely same-person pairs (§1.2) are treated as different groups. In simulation, such a pair ends up split across train and test 83% of the time. That would be subject-level leakage for those patients.

### 3.2 Threshold selection (Recall ≥ 0.65) — **LEAKAGE**
- `evaluate(model, test_loader)` ([code/04_train_recall_gat.py:322](code/04_train_recall_gat.py#L322)) collects the **test-fold** probabilities and labels, scans t ∈ [0.05, 0.95) for the highest F1 among thresholds with recall ≥ 0.65 ([248-256](code/04_train_recall_gat.py#L248-L256)), and reports metrics at that threshold. There is no validation split anywhere.
- Signature in the results: v4 recall is 0.653–0.706 in every fold (SD 0.020), sitting just above the 0.65 floor. This is the constraint being met on the test data itself.
- The same leakage exists in v3 (Youden J on test, [code/03_train_improved_gat.py:241-244](code/03_train_improved_gat.py#L241-L244)) and v2 (max-F1 on test, [code/02_train_node_gat.py:228-233](code/02_train_node_gat.py#L228-L233); [code/02_train_omni_gat.py:246-251](code/02_train_omni_gat.py#L246-L251)).
- Chosen thresholds (v4, best epoch): 0.59, 0.60, 0.55, 0.54, 0.53.
- ROC-AUC does not depend on the threshold, so it is not affected by this particular leak (but see 3.3).

### 3.3 Early stopping / model selection — **LEAKAGE**
- Every 10 epochs (and at epoch 1) the model is evaluated on the **test fold**, and the checkpoint/metrics with the highest test F1 are kept as that fold's result ([code/04_train_recall_gat.py:321-338](code/04_train_recall_gat.py#L321-L338)). Same in v3 ([303-320](code/03_train_improved_gat.py#L303-L320)) and v2.
- Best epochs chosen — v4: 40, 80, 50, 100, **1**; v3: 10, 30, 120, 90, 30.
- Size of the effect (from the logs, final epoch vs selected epoch):

| | Reported (test-selected epoch) | Fixed final epoch |
|---|---|---|
| v4 F1* | 0.6130 | 0.5878 |
| v4 ROC-AUC | 0.8315 | **0.8256** |
| v3 F1* | 0.6050 | 0.5853 |
| v3 ROC-AUC | 0.8272 | **0.8281** |

\*F1 in both columns still uses the test-tuned threshold, so even 0.588 is optimistic.

- Consequences: with no epoch cherry-picking, **v4 is not better than v3 on AUC**. The v3→v4 AUC gain of +0.004 is within selection noise. Both AUCs are still above the quoted TimeConv-CNN 0.8061, but that comparison is not like-for-like (3.6).
- The "best model" saved and used by all three apps is the fold with the highest **test** F1 (fold 2) ([code/04_train_recall_gat.py:349-351](code/04_train_recall_gat.py#L349-L351), [397](code/04_train_recall_gat.py#L397)). It was trained on ~4/5 of the patients, not all 310, although the app texts say "Trained on 310 patients".

### 3.4 Metric granularity and aggregation — **CONCERN**
- v3/v4: one sample = one channel in one 60-s window. All node-windows of all test patients in a fold are **pooled**, then accuracy/F1/precision/recall/AUC are computed once per fold ([code/04_train_recall_gat.py:231-274](code/04_train_recall_gat.py#L231-L274)).
- **No patient-level aggregation** (e.g. averaging each channel's probability across windows) exists in v3/v4. Patients with long recordings dominate: Zurich is 16% of graphs from 5% of patients. Overlapping windows of the same static channel label are highly correlated samples.
- Across folds: **macro average** of the 5 fold values, with `np.std` (population SD, ddof = 0) ([363-365](code/04_train_recall_gat.py#L363-L365)). Pooling all folds instead gives nearly the same F1 (0.6126, P 0.5634, R 0.6711).
- Only v2 graph-level (`02_train_omni_gat.py`) does patient-level aggregation (mean window probability per patient, [code/02_train_omni_gat.py:211-237](code/02_train_omni_gat.py#L211-L237)). That script answers a different question ("does this patient have any SOZ channel?"), with 84% positives.
- Reference points: predicting every node as SOZ gives node-level F1 = 0.351. For a patient-level task, predicting all-positive gives F1 = 0.912.

### 3.5 Normalization — **LEAKAGE** (mild)
- `StandardScaler().fit(all_x)` over all graphs before CV ([code/04_train_recall_gat.py:98-99](code/04_train_recall_gat.py#L98-L99); same in v2/v3).
- **Confirmed from the artifact:** `results/omni_gat_v4/scaler.pkl` has `n_samples_seen_ = 797,837` = 169,849 + 627,988, i.e. every node of every graph, train and test. It is an sklearn 1.7.2 StandardScaler with 18 features. Note that `04_train_recall_gat.py` never saves a scaler, so which script wrote this file is UNKNOWN.
- The leak is unsupervised (no labels involved) and probably small, but test statistics do reach training, and the fix is trivial: fit per fold on train only.

### 3.6 Baselines — **CONCERN (not like-for-like)**
- SEEG-NET, CLAP and TimeConv-CNN are not implemented anywhere in the repo (grep finds only names in plot labels). Their numbers are **hard-coded** tuples ([code/04_train_recall_gat.py:381-388](code/04_train_recall_gat.py#L381-L388), [code/03_train_improved_gat.py:363-369](code/03_train_improved_gat.py#L363-L369), [code/02_train_omni_gat.py:393-398](code/02_train_omni_gat.py#L393-L398), [code/plot_individual.py:23-26](code/plot_individual.py#L23-L26)). They are presumably copied from the Omni-iEEG paper; their source and evaluation protocol are UNKNOWN from the repo.
- Differences that make the comparison invalid as it stands:
  - different patients (our 310 local IDs after 27 exclusions vs the benchmark's cohort);
  - different splits (our 5-fold GroupKFold vs the benchmark's own splits);
  - different evaluation unit (our channel×window pooling vs the benchmark's definition);
  - different threshold protocol (ours tuned on test);
  - different labels (our substring parser);
  - possibly different recording types (we use all EDFs, including `task-ictal`).
- Even taking the numbers at face value, TimeConv-CNN beats v4 on F1 (0.6469 vs 0.6130) and all three baselines beat v4 on recall (0.7169 / 0.6997 / 0.7454 vs 0.6671).

### 3.7 README per-fold table recomputation — **mostly OK, one error**
Recomputed from the README fold rows ([README.md:160-165](README.md#L160-L165)). The fold rows match `training_v4_log.txt` best-fold lines exactly, and each fold's metrics also match a recomputation from its logged confusion matrix.

| Metric | Mean | SD (ddof = 0, as code) | SD (ddof = 1) | README | `summary_v4.csv` |
|---|---|---|---|---|---|
| Accuracy | 0.8202 | 0.0543 | 0.0607 | 0.8202 ± 0.054 ✓ | 0.8202 ± 0.0543 |
| F1 | 0.6130 | 0.0988 | 0.1105 | 0.6130 ± 0.099 ✓ | 0.6130 ± 0.0988 |
| Precision | 0.5845 | 0.1615 | 0.1806 | 0.5845 ± 0.162 ✓ | 0.5845 ± 0.1615 |
| Recall | 0.6671 | 0.0198 | 0.0221 | 0.6671 ± 0.020 ✓ | 0.6671 ± 0.0198 |
| ROC-AUC | 0.8315 | **0.0474** | 0.0530 | 0.8315 ± **0.039 ✗** | 0.8315 ± 0.0474 |

The README's AUC SD of 0.039 equals v3's AUC SD (0.0385), which looks like a copy error. For a paper, prefer the sample SD (ddof = 1) and say so.

### 3.8 Other integrity issues
| Issue | Status | Evidence |
|---|---|---|
| Model/threshold design iterated v2→v3→v4 on the same test folds; no held-out test set | LEAKAGE (design-level) | three scripts, same CV, all selecting on test |
| Label source (`status`/`bad`) | CONCERN, potentially critical | §1.6 |
| Cohort selection by 120 s timeout (21 processable patients excluded) | CONCERN (selection bias) | §1.2 |
| First-64-channel truncation (266/310 patients) may drop SOZ channels | CONCERN | [code/01_single_patient.py:146](code/01_single_patient.py#L146) |
| Global amplitude features without per-patient normalization, while SOZ prevalence differs by centre (e.g. Zurich 17/17 positive): the model could learn centre/amplifier identity | CONCERN | meta table §1.2 |
| All EDFs used, including ictal runs; ictal vs interictal split UNKNOWN | CONCERN / UNKNOWN | [code/archive/00_test_pipeline.py:18](code/archive/00_test_pipeline.py#L18), [code/archive/test_read.py:3](code/archive/test_read.py#L3) |
| Confusion-matrix figure (`08_confusion_matrix.png`, `results_graphs.png`) uses `[[113603, 3311], [13816, 29304]]`. This matches **neither** v4 fold 2 (`[[106581, 6499], [14075, 33800]]`) nor any v3 fold; its implied F1 is 0.7739, not the captioned 0.7667 | Mismatch | [code/plot_individual.py:237](code/plot_individual.py#L237), [code/plot_results.py:176-177](code/plot_results.py#L176-L177), [logs/training_v4_log.txt:78-79](logs/training_v4_log.txt) |
| All figures use hard-coded numbers, not saved results | CONCERN (traceability) | `plot_*.py` |
| `02_train_omni_gat.py` ends with a stray `CCC` line → NameError after saving | Minor bug | [code/02_train_omni_gat.py:423](code/02_train_omni_gat.py#L423) |

---

## 4. Code-vs-Manuscript Mismatches (manuscript absent → compared against README)

| # | README claim | What the code / logs show |
|---|---|---|
| 1 | "302 patients across 8 epilepsy centers" ([54](README.md#L54)) vs "337 total → 310" ([62](README.md#L62)) | 337 local folders; 310 used; 302 is never derived from code. "8 centres" cannot be verified: the IDs show site codes HUP, Detroit, UCLA, Zurich, NIH/pt, jh/PY, umf, ummc, rns, and mapping codes to institutions is not in the repo |
| 2 | "processes 337 patients **in parallel**" ([59](README.md#L59), [122](README.md#L122)) | Sequential `for` loop with blocking `subprocess.run` ([code/01_safe_runner.py:16-24](code/01_safe_runner.py#L16-L24)) |
| 3 | "310 successfully processed (92%)" | 309 OK in the runner log + 1 TIMEOUT patient with a file; 21 exclusions were timeouts, not bad data |
| 4 | "Recall-constrained threshold optimization … Recall ≥ 0.65" presented as a method | It is tuned on the test fold (leakage) |
| 5 | "five-stage pipeline … **patient-level aggregation**" ([92](README.md#L92)) | v3/v4 have no patient-level aggregation |
| 6 | "Data leakage → GroupKFold patient-level separation" ([78](README.md#L78)) | Patient split is OK by ID, but threshold, epoch and scaler leak test information; possible duplicate subjects |
| 7 | "State-of-the-art ROC-AUC 0.8315" / "highest ROC-AUC among all published baseline methods" ([26](README.md#L26), [86](README.md#L86), [176](README.md#L176)) | Baselines are not re-run and the protocol differs; the AUC depends on test-selected epochs (0.8256 at a fixed epoch) |
| 8 | "Clinically aligned — high Recall (0.6671) minimizes missed SOZ" ([87](README.md#L87)) | v4 recall is **lower** than all three baselines |
| 9 | "Fully reproducible" ([88](README.md#L88)) | Hard-coded `/home/pmilab/...` paths; fold partition not reproducible; unpinned requirements; no data or checkpoints; figures use hard-coded numbers |
| 10 | ROC-AUC ± 0.039 ([165](README.md#L165)) | ± 0.0474 (log, CSV, recomputation) |
| 11 | v2 = "Node-level GAT · Cross-Entropy Loss" | Consistent with `02_train_node_gat.py` (class-weighted CE), but no v2 log exists and both `02_*` scripts overwrite the same `summary.csv` |
| 12 | v4 "Key change: Recall-constrained threshold" | v4 also changed α 0.75→0.85, heads 8→4, batch 32→16, LR 5e-4→3e-4, epochs 120→150, dropout 0.3→0.25, a deeper classifier, **and a different fold partition**. The gain cannot be attributed to the threshold |
| 13 | v4 docstring: hidden 192, class weights 1:5, recall 0.70 | Hidden 128, no class weights, recall 0.65 |
| 14 | "01_process_omni_ieeg.py — Main preprocessing script" ([237](README.md#L237)) | It is the diagnostic script and saves no graphs |
| 15 | "Best single fold F1 = 0.7667" + confusion matrix "Best Fold 2" | The CM figure numbers do not match fold 2 (implied F1 = 0.7739) |
| 16 | Figure at line 67: alt text "Pipeline Architecture" | Shows `09_model_progression.png` |
| 17 | License: badge says **MIT** ([8](README.md#L8)), section says **Apache 2.0** ([303](README.md#L303)) | No `LICENSE` file exists |
| 18 | "see the Contributing Guide (CONTRIBUTING.md)" | File does not exist |
| 19 | Team table "Laraib Kanwal"; BibTeX `author = { Syed Ismael, Atif Khan, Laraib}` | Name inconsistent; BibTeX authors must be separated by `and`; author order differs from the team table |
| 20 | "Tested on Ubuntu 22.04 and Python 3.10", "MNE 1.4+", "8 GB VRAM minimum" | Not verifiable; logs show an RTX A5000 and `/home/pmilab` paths only |
| 21 | Apps: "Trained on 310 patients", "AUC 0.8315 ✓ Best" ([code/ieegnet_app.py:547](code/ieegnet_app.py#L547), [code/archive/soz_app_v2.py:400](code/archive/soz_app_v2.py#L400)) | The deployed checkpoint is one fold model (~248 training patients) picked by test F1. The app default threshold is 0.35 ([code/ieegnet_app.py:454](code/ieegnet_app.py#L454)), unlike the 0.53–0.60 used in evaluation |
| 22 | "ICLR 2026" for Omni-iEEG | Not verifiable from the repo |

---

## 5. Anticipated Questions & Answers

**Q1. How was the classification threshold chosen? Was the test set used?**
*Honest answer:* Yes. For every fold, the threshold (the highest-F1 threshold with recall ≥ 0.65) was searched on the same test-fold predictions that the metrics are reported on ([code/04_train_recall_gat.py:248-256](code/04_train_recall_gat.py#L248-L256), called at [322](code/04_train_recall_gat.py#L322)). That is why recall sits at 0.65–0.71 in every fold.
*Fix / what to write:* Inside each training fold, split off ~15% of the **patients** as a validation set (GroupShuffleSplit). Choose the threshold on validation, freeze it, and report test metrics. Until then, write "threshold optimised post hoc on the evaluation fold (optimistic)", or report only threshold-free AUC/AUPRC.

**Q2. How did you choose the number of epochs / early stopping?**
*Honest answer:* There was no early stopping. The model was evaluated on the test fold at 16 epochs, and the epoch with the best test F1 was reported ([321-338](code/04_train_recall_gat.py#L321-L338)). At a fixed final epoch, AUC is 0.8256 (v4) vs 0.8281 (v3), not 0.8315.
*Fix:* Early-stop on the inner validation split (Q1), or fix the epoch count in advance. Re-report.

**Q3. Is there patient-level data leakage?**
*Honest answer:* Patient IDs are cleanly separated by GroupKFold ([284-291](code/04_train_recall_gat.py#L284-L291)), verified by simulation against the logged fold sizes. Two caveats:
- about five subjects appear to exist under two IDs (Multicenter vs sourcesink: jh103, jh105, pt1/pt01, pt2, pt3), which could put the same person in train and test;
- the scaler is fit on all data (`scaler.pkl` saw all 797,837 nodes).

*Fix:* Deduplicate subjects (or group the pairs under one ID). Fit the scaler inside each fold. State the dedup rule in Methods.

**Q4. How exactly are SOZ labels obtained? Are you sure they are SOZ and not "bad channel" flags?**
*Honest answer:* Labels come from `channels.tsv`. The code reads the `status` column **before** the `soz` column and marks any value containing "bad", "1", "yes", "onset", etc. as SOZ ([code/01_single_patient.py:97-109](code/01_single_patient.py#L97-L109)). If Omni-iEEG's TSVs have a BIDS `status` (good/bad) column, part of our "SOZ" labels are bad-channel flags. This has not been verified yet.
*Fix:* Inspect the TSV headers (command in §6). Read the dataset's official SOZ column explicitly with exact value matching, exclude `status=bad` channels as inputs, regenerate the graphs, and re-run. This must be settled before submission.

**Q5. Are the baseline comparisons fair? Did you re-run SEEG-NET, CLAP and TimeConv-CNN?**
*Honest answer:* No. Their numbers are hard-coded from the literature ([381-388](code/04_train_recall_gat.py#L381-L388)). Our cohort, splits, evaluation unit, labels and threshold protocol all differ, so the comparison is indicative only.
*Fix:* Either run the official Omni-iEEG benchmark split and evaluation code with our model, or re-train at least one baseline (e.g. TimeConv-CNN) on our exact folds. Otherwise drop "state-of-the-art" and write "comparable to published baselines reported under a different protocol".

**Q6. Is the "highest ROC-AUC" claim justified?**
*Honest answer:* Not as evaluated:
- the 0.8315 depends on test-based epoch selection (0.8256 at a fixed epoch);
- the baselines are not like-for-like;
- the fold SD (0.047) is larger than the 0.025 gap to TimeConv-CNN;
- TimeConv-CNN is better on F1 and recall.

*Fix:* Remove the claim or rephrase it as above. If the re-run baselines support it, add a paired statistical test (e.g. DeLong on pooled predictions, or a Wilcoxon test over folds or patients).

**Q7. Are metrics per channel, per window, or per patient?**
*Honest answer:* Per channel per 60-s window, pooled within a fold, then averaged over folds (population SD) ([231-274](code/04_train_recall_gat.py#L231-L274), [363-365](code/04_train_recall_gat.py#L363-L365)). There is no patient-level aggregation, although the README says there is. SOZ labels are constant over time, so windows are strongly correlated.
*Fix:* Add a channel-level result (average each channel's probability over all its windows, then compute metrics) and a per-patient AUC distribution (median and IQR across patients). Report sample SD or 95% CI.

**Q8. What does the "HFO proxy (80–300 Hz)" feature actually measure? Does the sampling rate support it?**
*Honest answer:* It is Welch band power over 80–300 Hz after resampling to 1000 Hz ([code/01_single_patient.py:55](code/01_single_patient.py#L55), [134-137](code/01_single_patient.py#L134-L137)). It is not HFO event detection.
- Nyquist after resampling is 500 Hz, but files recorded below 600 Hz (e.g. HUP at 500 Hz according to [code/legacy_hup/preprocess.py:15](code/legacy_hup/preprocess.py#L15)) contain no real content up to 300 Hz.
- There is no notch filter, so mains harmonics fall in-band.

*Fix:* Log each file's native sampling rate. Either cap the band at min(300, 0.4·fs_native) or exclude low-fs centres from that feature. Add a 50/60 Hz notch with harmonics. Rename it "high-frequency band power".

**Q9. Why did v4 improve over v3, and is the improvement significant?**
*Honest answer:* v4 changed about eight things at once (α, heads, batch size, LR, epochs, dropout, classifier depth, threshold rule), and it ran on a **different patient partition**, because fold assignment depends on hash order ([77](code/04_train_recall_gat.py#L77)). The gain (+0.008 F1, +0.004 AUC) is far smaller than the fold SD (≈ 0.10 F1, 0.05 AUC). At a fixed epoch, v3 has the higher AUC.
*Fix:* Fix the folds (sort patient IDs, or save the fold assignments to disk). Run v3 and v4 on identical folds with several seeds. Do a one-change-at-a-time ablation. Report mean ± SD over seeds.

**Q10. Does the graph (GAT) actually help over the node features alone? Why is fold 5's best epoch epoch 1?**
*Honest answer:* Not tested. v4 already reaches AUC 0.82 after one epoch in folds 1 and 4, and fold 5's best is epoch 1 ([logs/training_v4_log.txt:32](logs/training_v4_log.txt), [140](logs/training_v4_log.txt)). That suggests most of the signal is in the per-channel features. Edges are unweighted |r| > 0.3 and sign is discarded ([code/01_single_patient.py:78](code/01_single_patient.py#L78)).
*Fix:* Add ablations with the same folds, validation and threshold protocol:
- logistic regression / MLP / gradient boosting on the 18 features;
- GAT with a shuffled or empty graph;
- GAT with correlation edge weights.

Also report feature importance (permutation, or SHAP on the MLP).

*Further likely questions (short answers):*
- **Cohort size and exclusions?** 337 folders → 310 patients. 7 had no valid data; 21 timed out at 120 s even though they were processable (selection bias; [code/01_safe_runner.py:21](code/01_safe_runner.py#L21)). Re-run with a higher timeout.
- **Channel handling?** Only the first 64 channels, no channel-type filtering, no re-referencing ([code/01_single_patient.py:146](code/01_single_patient.py#L146)). Disclose this, or run with all channels and a common average or bipolar reference.
- **Ictal vs interictal data?** All EDF runs are used, including `task-ictal` runs. Disclose, or restrict to the interictal task if that is what the benchmark does.
- **Class imbalance?** 21.3% positive node-windows. Handled with Focal loss α = 0.85, γ = 2. Also report AUPRC with the 0.213 prevalence baseline.

---

## 6. Open Unknowns

| Unknown | Why it matters | What is needed |
|---|---|---|
| Actual columns and values in Omni-iEEG `channels.tsv` (is there `status`? what does `soz` contain?) | Decides whether labels are SOZ or bad-channel flags (§1.6) | On the server: `head -3 $(find /home/pmilab/EEG_GNN_Project/Hiroki/Data/Data -name '*channels.tsv' \| head -5)` and `cut -f<status col> ... \| sort \| uniq -c` |
| Native sampling rate per file / centre | Validity of the 80–300 Hz feature | Log `raw.info['sfreq']` before resampling |
| Which BIDS tasks (ictal/interictal) were included | Protocol comparability with the benchmark | List EDF filenames per patient |
| Are the Multicenter/sourcesink pairs (jh103, jh105, pt1/pt01, pt2, pt3, umf002–005) the same people? | Cross-fold subject leakage; the 337 vs 302 count | Omni-iEEG participants.tsv / documentation |
| Why the 7 patients failed | Methods text | Re-run the worker for those IDs with stderr/stdout captured |
| Why `sub-openieegUCLA21` has a `.pt` despite TIMEOUT | Cohort definition | File timestamps in `patient_graphs/` |
| Which script produced GAT v2 (node vs graph level); its log | v2 numbers in the progression table | v2 training log |
| Which script produced `scaler.pkl`; which checkpoint/epoch the apps use | Deployment claims | Server files |
| Source and protocol of the baseline numbers (SEEG-NET 0.7850 etc.) | Like-for-like comparison | Citation and table from the Omni-iEEG paper |
| Number and identity of centres ("8") | Multicentre claim | Mapping site codes → institutions |
| Exact PyG/torch/sklearn/MNE versions | Parameter counts, reproducibility | `pip freeze` on the server (the scaler shows sklearn 1.7.2) |
| Source of the confusion matrix in the figures | Figure correctness | Replace with the logged fold-2 CM, or regenerate from saved predictions |
| Manuscript text | Task 3 was done against the README only | Add the manuscript file to this workspace |

---

## 7. Addendum — Manuscript 4.0 vs Code (added after `Manuscript 4.0.docx` was provided)

Source: `C:\Users\syed.ismael\OneDrive - TMG\Documents\Manuscript 4.0.docx` (read in full, including its 5 embedded images). Section names below refer to the manuscript.

### 7.1 Statements that match the code
- Final hyperparameters (§E) match v4 as run: 3 GAT layers, hidden 128, 4 heads, Adam with LR 3e-4 and weight decay 1e-4, dropout 0.25, 150 epochs, batch 16, seed 42, cosine T_max = 150, gradient clipping 1.0, Focal α = 0.85, γ = 2.0, threshold grid 0.05–0.95 with Recall ≥ 0.65 then max F1 ([code/04_train_recall_gat.py:53-59](code/04_train_recall_gat.py#L53-L59), [248-256](code/04_train_recall_gat.py#L248-L256), [306-311](code/04_train_recall_gat.py#L306-L311)).
- 60 s windows with a 30 s step, Pearson correlation with threshold 0.30, ~13.7k graphs, 310 patients, and skip connections + batch norm + graph-context broadcast + 3-layer MLP all match.
- The Table I values (mean) and the Table II F1/AUC numbers are the same as the README and the logs.

### 7.2 Mismatches / problems

| # | Manuscript says | Code / logs show | Severity |
|---|---|---|---|
| M1 | Abstract: "able to outperform several baseline approaches"; Contributions: "state-of-the-art performance" | TimeConv-CNN is better on F1 (0.6469) and all baselines are better on recall; the baseline numbers are copied, not re-run (§3.6) | High |
| M2 | §B: "undirected **weighted** graph G = (V, E, W)"; §Methodology: "weighted graph edges" | Edges are **unweighted**; correlation values are not stored or used ([code/01_single_patient.py:71-85](code/01_single_patient.py#L71-L85)) | High |
| M3 | §B: "connectivity maintained using a **nearest-neighbor strategy**" | Only if a window has **zero** edges, a ring over channel index order is added; no nearest-neighbour step ([81-84](code/01_single_patient.py#L81-L84)) | Medium |
| M4 | §A: "channels with insufficient metadata or **too much noise were removed**" | Only flat/NaN channels are removed; noisy channels are not; channels marked `bad` may be labelled SOZ; only the first 64 channels are kept ([141-146](code/01_single_patient.py#L141-L146), [97-106](code/01_single_patient.py#L97-L106)) | High |
| M5 | §B: "8-dimensional statistical features (mean, variance, RMS, skewness, kurtosis, line length, zero-crossing rate)" | That list has 7 items; **standard deviation** is missing. The 10 spectral features are delta, theta, alpha, beta, gamma, high-gamma (80–150), HFO proxy (80–300), total power, peak frequency, spectral entropy ([code/01_single_patient.py:62-68](code/01_single_patient.py#L62-L68)) | Low |
| M6 | Preprocessing not described | Code resamples to 1000 Hz, band-passes 1–300 Hz, uses no notch filter and no re-referencing, and uses all EDFs including ictal runs — none of this is in the paper | Medium |
| M7 | Methodology: "(G) Evaluation. **Patient-level model evaluation.**" | No patient-level aggregation; metrics are pooled over channel × window (§3.4). This sentence is also a fragment | High |
| M8 | §D: "one group of patients was used for **validation**" | That same fold is the **test** fold, and it is also used to pick the threshold and the best epoch — no separate validation set (§3.2–3.3) | **Critical** |
| M9 | §E: "threshold was selected … during evaluation" | True, but it does not say this was done on the test fold — a reviewer will read it as leakage | **Critical** |
| M10 | §E: "hyperparameters were tuned during the development process" | Tuned by comparing v2/v3/v4 on the same test folds | High |
| M11 | §D: "Each electrode was classified as either SOZ or non-SOZ" | Each electrode is classified **in every 60-s window**; metrics count electrode-windows, not electrodes | Medium |
| M12 | Table I: ROC-AUC 0.8315 **± 0.039** | ± 0.0474 (log/CSV); ± 0.053 as a sample SD | Medium |
| M13 | Table I caption: "Performance across various model versions from v2 to v4" | The table shows v4 only | Low |
| M14 | §Results A: "v3 with deeper layers, graph context, Focal Loss, **and GroupKFold**; v4 further tuned the attention heads and decision threshold" | v2 already used GroupKFold. v4 also changed α, LR, batch, epochs, dropout and classifier depth, and ran on a **different fold partition**. At a fixed epoch v4 AUC (0.8256) < v3 (0.8281) (§3.3) | High |
| M15 | §Results B: "recently published frameworks … tested on the Omni-iEEG benchmark"; "highest ROC-AUC" | Different protocol, patients, splits and labels; not like-for-like; the AUC gap (0.025) is smaller than the fold SD (0.047) | High |
| M16 | §A: "Omni-iEEG benchmark **of 310 patients** from eight centers" | 310 is **our processed subset** of 337 folders (7 failed, 21 timed out, §1.2), not the benchmark's size. "Eight centres" cannot be verified from the code | Medium |
| M17 | §Discussion: "approximately 2–3 hours for five-fold training" | No timing appears in any log → UNKNOWN (cannot confirm) | Low |
| M18 | §Discussion: "Focal Loss and recall-aware threshold optimization mitigate class imbalance" | The threshold is optimized on test data; the reported recall floor is guaranteed by construction | Medium |
| M19 | **Figure I** (image1) | It shows the **old HUP pipeline**, not this one: "5 s / 10 s windows" (a 60 s patch is pasted over it), "Coherence", "Edge weights = strength", features "max, min, median, percentiles, Hjorth, wavelet", "Graph-level features", "Window representation → mean/max/attention pooling → patient embedding", "Outcome prediction: seizure-free vs non-seizure-free", "Unsupervised attention-based node importance ranking". **None of these are in the v4 code** | High |
| M20 | Abbreviations list | SEEG is defined but "SEEG-NET" and "CLAP" are not; fine otherwise | Low |

### 7.3 Reference problems
- [8] is an exact duplicate of [2] (a surgery review) and is cited for "framework … iEEG of multiple centers", which does not fit.
- [13] duplicates [9]. Both are incomplete: title only, no authors, venue or year.
- [11] (MRAgent, an LLM agent for Mendelian randomization) and [12] (brain–visual alignment, arXiv 2606.25718) are cited for CNN/RNN iEEG claims. They are unrelated to the claims.
- [10] (EEG-GCNN) is a GCN paper, but it is cited for CNN/RNN deep learning.
- [1] WHO has no access date. No reference supports the DRE definition sentence ("at least two anti-seizure medications").
- [14] (HUP ds004100) is cited as the dataset, while the work uses Omni-iEEG. Clarify that HUP is a sub-cohort.
- Author order in the manuscript (Laraib Kanwal, Syed Ismael, Atif Khan, Saima Zareen, Syed Muhammad Anwar) differs from the README BibTeX.
