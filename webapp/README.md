---
title: EEG-GNN
emoji: 🧠
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# EEG-GNN web app

## Free version: the model runs in the browser (live)

`site/` now runs GAT v4 entirely in the visitor's browser (`site/engine.js` + `site/worker.js`), so the
site is plain static files: no server, no cost, and recordings are never uploaded.

- Live: https://huggingface.co/spaces/Ismael-zbi/eeg-gnn (static Space, free)
- `site/model/` holds the exported weights, scaler and MNE filter (not in git; the trained model is available on request from ismaelzabi302@gmail.com). Rebuild after retraining:
  `.venv/Scripts/python tools/export_browser_model.py`
- Check the browser engine against the Python pipeline on any EDF:
  `.venv/Scripts/python tools/reference_dump.py file.edf ref.json` then `node tools/verify_browser.js file.edf ref.json`
  (tested: probabilities within 0.0001 of Python, same edges, same ranking, incl. 2048 Hz files and >64 channels)
- Any static host works (Hugging Face static Space, GitHub Pages, Vercel): upload the contents of `site/`.

The server version below (FastAPI + Docker) still works for the lab server, but a Docker Space now needs HF PRO.

The public website plus a live model API. A visitor uploads an `.edf` iEEG recording, and the
GAT v4 model ranks every electrode by how likely it is to be in the seizure onset zone.

```
webapp/
├── api/
│   ├── main.py        FastAPI server: /health, POST /jobs (upload), GET /jobs/{id} (progress + result)
│   └── pipeline.py    Preprocessing + GATv4, copied from code/01_single_patient.py and code/04_train_recall_gat.py
├── site/              The website (plain HTML/CSS/JS). Served by the API at "/", or host it separately.
│   ├── index.html  styles.css  app.js
│   └── config.js      Where the site finds the API ("" = same server)
├── models/            Put best_model_v4.pt + scaler.pkl here (not in git)
├── requirements.txt
└── Dockerfile         For Hugging Face Spaces / any Docker host
```

## In-browser version (free static hosting, e.g. Vercel)

`site/` now runs the model itself, inside the visitor's browser, so no server is needed and the
recording never leaves their computer:

- `site/engine.js` is a JavaScript port of `api/pipeline.py` (EDF read, resample, the exact MNE
  1–300 Hz FIR kernel, the 18 features, correlation edges, scaler, GAT v4 forward pass).
- `site/worker.js` runs it in a Web Worker; `site/model/` holds the exported weights.
- Re-export after retraining: `.venv/Scripts/python tools/export_browser_model.py`
- Check it still matches Python: `python tools/reference_dump.py f.edf ref.json` then
  `node tools/verify_browser.js f.edf ref.json` (probabilities agree to within ~1e-4 on test files).
- Deploy: `vercel deploy site --prod` (or any static host: GitHub Pages, Netlify, a static HF Space).

The FastAPI server below still works and is unchanged.

## What happens to an upload

1. The file is streamed to a temp file on the server (max `MAX_UPLOAD_MB`, default 500 MB).
2. Same preprocessing as training: resample to 1000 Hz → 1–300 Hz band-pass → drop flat channels → first 64 channels →
   60-s windows, 30-s step → 18 features per channel → |Pearson r| > 0.3 edges → training StandardScaler, clipped ±10.
3. GAT v4 scores every channel in every window. Long files are capped at `MAX_WINDOWS` (default 40) windows spread
   evenly across the recording. The site shows the per-channel **average**, plus a channel × time heat map.
4. The temp file is deleted. Results live in memory for 1 hour, then they're gone.

The site polls `/jobs/{id}` every 0.7 s, so the progress bar moves window by window.

## 0. Get the model files

They are git-ignored and not public. Team members can copy them from the lab server into `webapp/models/`; anyone else can request them from ismaelzabi302@gmail.com.

```
results/omni_gat_v4/best_model_v4.pt   (on the server: /home/pmilab/EEG_GNN_Project/results/omni_gat_v4/)
results/omni_gat_v4/scaler.pkl         (already in this repo)
```

## Option A — run on the lab server (GPU, quickest)

```bash
cd eeg-gnn/webapp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # on a GPU box, install the CUDA build of torch instead of the CPU one
cd api && uvicorn main:app --host 0.0.0.0 --port 7860
```

Open `http://<server-ip>:7860`. That's the full site with live analysis. Only machines on the university network can reach it.
To share it publicly without opening firewall ports, use a tunnel, for example:

```bash
cloudflared tunnel --url http://localhost:7860     # prints a public https://….trycloudflare.com link
```

## Option B — Hugging Face Spaces (free, public, CPU)

1. Create a Space at huggingface.co → **New Space** → SDK **Docker** → blank.
2. Push the contents of `webapp/` to it (this README's header already configures the Space):
   ```bash
   git clone https://huggingface.co/spaces/<you>/eeg-gnn hf-space
   cp -r eeg-gnn/webapp/* hf-space/
   cd hf-space && git lfs install && git lfs track "models/*.pt" "models/*.pkl"
   git add . && git commit -m "EEG-GNN app" && git push
   ```
3. Wait for the build. The site and API are both at `https://<you>-eeg-gnn.hf.space`.

CPU speed: roughly a few seconds per minute-window (64 channels), so a 40-window file takes on the order of a minute or two.
Lower `MAX_WINDOWS` in the Space's settings → Variables if that's too slow.

## Hosting the site somewhere else (GitHub Pages, Vercel)

Deploy only `site/`, and set `site/config.js`:

```js
window.EEGGNN_API = "https://<you>-eeg-gnn.hf.space";
```

Then set `ALLOWED_ORIGINS=https://<your-site-domain>` on the API so only your site can call it.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `models/best_model_v4.pt` | checkpoint (`{'model': state_dict, ...}`) |
| `SCALER_PATH` | `models/scaler.pkl` | training StandardScaler |
| `MAX_UPLOAD_MB` | 500 | upload limit |
| `MAX_WINDOWS` | 40 | max 60-s windows analysed per file |
| `WORKERS` | 1 | files analysed at the same time |
| `ALLOWED_ORIGINS` | `*` | CORS origins, comma-separated |

## Important

- **Research only.** The page says so, and it must stay that way: the model is not a medical device.
- **Patient data.** A public Space sends files to Hugging Face's servers. Only upload anonymised or public recordings
  (e.g. Omni-iEEG). Real hospital data must stay on the lab server (Option A, no public tunnel).
- **Threshold.** The checkpoint doesn't store a threshold. The site starts at 0.50 (training picked ~0.49–0.62 per
  fold) and lets the user move it.
- **Known caveats** from `AUDIT_REPORT.md` (threshold/epoch chosen on test folds, scaler fit on all data) apply to these
  scores too.
