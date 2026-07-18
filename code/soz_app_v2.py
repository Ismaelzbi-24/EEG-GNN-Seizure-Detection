"""
GATNet SOZ Detection — Professional Dashboard UI v2
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import mne
import plotly.graph_objects as go
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

MODEL_PATH  = "/home/pmilab/EEG_GNN_Project/results/omni_gat_v4/best_model_v4.pt"
SCALER_PATH = "/home/pmilab/EEG_GNN_Project/results/omni_gat_v4/scaler.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# MODEL
# ==========================================
class GATv4(nn.Module):
    def __init__(self, in_channels, hidden=128, heads=4, dropout=0.25):
        super().__init__()
        self.dropout = dropout
        self.gat1  = GATConv(in_channels,    hidden,        heads=heads, dropout=dropout)
        self.gat2  = GATConv(hidden * heads, hidden,        heads=heads, dropout=dropout)
        self.gat3  = GATConv(hidden * heads, hidden,        heads=1,     dropout=dropout)
        self.skip1 = nn.Linear(in_channels,    hidden)
        self.skip2 = nn.Linear(hidden * heads, hidden)
        self.bn1   = nn.BatchNorm1d(hidden * heads)
        self.bn2   = nn.BatchNorm1d(hidden * heads)
        self.bn3   = nn.BatchNorm1d(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden + hidden, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 2)
        )

    def forward(self, x, edge_index, batch):
        skip1 = self.skip1(x)
        x = F.elu(self.bn1(self.gat1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        skip2 = self.skip2(x)
        x = F.elu(self.bn2(self.gat2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn3(self.gat3(x, edge_index)) + skip2 + skip1)
        gc = global_mean_pool(x, batch)[batch]
        return self.classifier(torch.cat([x, gc], dim=1))

print("Loading model...")
checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
model = GATv4(18, 128, 4, 0.25).to(DEVICE)
model.load_state_dict(checkpoint['model'])
model.eval()
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
print(f"Ready on {DEVICE}")

# ==========================================
# FEATURES
# ==========================================
def bandpower(sig, fs, fmin, fmax):
    f, psd = signal.welch(sig, fs=fs, nperseg=min(fs, len(sig)))
    idx = np.logical_and(f >= fmin, f <= fmax)
    return float(np.trapz(psd[idx], f[idx])) if idx.any() else 0.0

def extract_features(window, fs):
    features = []
    for ch in window:
        f, psd = signal.welch(ch, fs=fs, nperseg=min(fs, len(ch)))
        tp = float(np.trapz(psd, f)) + 1e-10
        pn = psd/tp; pn = pn[pn>0]
        features.append([
            float(np.mean(ch)), float(np.std(ch)), float(np.var(ch)),
            float(skew(ch)), float(kurtosis(ch)),
            float(np.sqrt(np.mean(ch**2))),
            float(np.sum(np.abs(np.diff(ch)))),
            float(np.sum(np.diff(np.sign(ch)) != 0)),
            bandpower(ch,fs,0.5,4), bandpower(ch,fs,4,8),
            bandpower(ch,fs,8,13), bandpower(ch,fs,13,30),
            bandpower(ch,fs,30,80), bandpower(ch,fs,80,150),
            bandpower(ch,fs,80,300), tp,
            float(f[np.argmax(psd)]),
            float(-np.sum(pn * np.log2(pn + 1e-10)))
        ])
    return np.array(features, dtype=np.float32)

def build_edges(window, thresh=0.3):
    n = window.shape[0]
    cor = np.nan_to_num(np.corrcoef(window))
    src, dst = [], []
    for i in range(n):
        for j in range(i+1, n):
            if abs(cor[i,j]) > thresh:
                src.extend([i,j]); dst.extend([j,i])
    if not src:
        for i in range(n):
            src.extend([i,(i+1)%n]); dst.extend([(i+1)%n,i])
    return torch.tensor([src,dst], dtype=torch.long)

# ==========================================
# INFERENCE
# ==========================================
@torch.no_grad()
def predict_file(file, threshold):
    if file is None:
        return [None]*6 + ["⚠️ Please upload a file first."]

    path = file.name
    threshold = float(threshold)

    try:
        if path.lower().endswith('.edf'):
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            fs  = raw.info['sfreq']
            if fs != 1000: raw.resample(1000); fs = 1000
            raw.filter(1., 300., fir_design='firwin', verbose=False)
            data = raw.get_data()
            ch_names = raw.ch_names
            valid = [i for i in range(data.shape[0])
                     if not np.all(np.isnan(data[i])) and np.std(data[i]) > 1e-10][:64]
            data = data[valid]
            ch_names = [ch_names[i] for i in valid]
            win = data[:, :min(60000, data.shape[1])]
            feats  = extract_features(win, int(fs))
            scaled = np.clip(scaler.transform(feats), -10, 10)
            x  = torch.tensor(scaled, dtype=torch.float).to(DEVICE)
            ei = build_edges(win).to(DEVICE)
            bt = torch.zeros(x.shape[0], dtype=torch.long).to(DEVICE)
            out   = model(x, ei, bt)
            probs = torch.softmax(out, dim=1)[:,1].cpu().numpy()

        elif path.lower().endswith('.pt'):
            from torch_geometric.loader import DataLoader
            graphs = torch.load(path, weights_only=False)
            if not isinstance(graphs, list): graphs = [graphs]
            loader = DataLoader(graphs, batch_size=32)
            probs_list = []
            for batch in loader:
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.batch)
                probs_list.extend(torch.softmax(out,dim=1)[:,1].cpu().numpy())
            probs    = np.array(probs_list)
            ch_names = [f"Ch_{i}" for i in range(len(probs))]
        else:
            return [None]*6 + ["❌ Unsupported format. Use .edf or .pt"]

        preds  = (probs >= threshold).astype(int)
        n_soz  = int(preds.sum())
        n_norm = len(preds) - n_soz
        n_tot  = len(preds)

        soz_channels = sorted(
            [(ch_names[i], float(probs[i])) for i in np.where(preds==1)[0]],
            key=lambda x: x[1], reverse=True
        )

        # ── STAT CARDS ──────────────────────────────
        cards_html = f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:8px">
  <div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);border-radius:14px;padding:20px;text-align:center;color:white">
    <div style="font-size:13px;opacity:.8;margin-bottom:6px">TOTAL CHANNELS</div>
    <div style="font-size:36px;font-weight:700">{n_tot}</div>
    <div style="font-size:12px;opacity:.7;margin-top:4px">Analyzed</div>
  </div>
  <div style="background:linear-gradient(135deg,#7f1d1d,#dc2626);border-radius:14px;padding:20px;text-align:center;color:white">
    <div style="font-size:13px;opacity:.8;margin-bottom:6px">SOZ DETECTED</div>
    <div style="font-size:36px;font-weight:700">{n_soz}</div>
    <div style="font-size:12px;opacity:.7;margin-top:4px">{100*n_soz/n_tot:.1f}% of total</div>
  </div>
  <div style="background:linear-gradient(135deg,#14532d,#16a34a);border-radius:14px;padding:20px;text-align:center;color:white">
    <div style="font-size:13px;opacity:.8;margin-bottom:6px">NORMAL CHANNELS</div>
    <div style="font-size:36px;font-weight:700">{n_norm}</div>
    <div style="font-size:12px;opacity:.7;margin-top:4px">{100*n_norm/n_tot:.1f}% of total</div>
  </div>
  <div style="background:linear-gradient(135deg,#312e81,#7c3aed);border-radius:14px;padding:20px;text-align:center;color:white">
    <div style="font-size:13px;opacity:.8;margin-bottom:6px">THRESHOLD</div>
    <div style="font-size:36px;font-weight:700">{threshold:.2f}</div>
    <div style="font-size:12px;opacity:.7;margin-top:4px">Detection cutoff</div>
  </div>
</div>"""

        # ── SOZ TABLE ───────────────────────────────
        rows = ""
        for rank, (ch, prob) in enumerate(soz_channels[:15], 1):
            pct   = prob * 100
            bar_w = int(pct)
            color = "#dc2626" if prob >= 0.7 else "#f59e0b" if prob >= 0.5 else "#ef4444"
            rows += f"""
<tr style="border-bottom:1px solid #f1f5f9">
  <td style="padding:10px 14px;font-weight:600;color:#64748b">#{rank}</td>
  <td style="padding:10px 14px;font-weight:600;color:#0f172a;font-family:monospace">{ch}</td>
  <td style="padding:10px 14px">
    <div style="background:#f1f5f9;border-radius:6px;height:10px;width:140px">
      <div style="background:{color};height:10px;border-radius:6px;width:{bar_w}%"></div>
    </div>
  </td>
  <td style="padding:10px 14px;font-weight:700;color:{color}">{prob:.3f}</td>
  <td style="padding:10px 14px">
    <span style="background:{color};color:white;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600">SOZ</span>
  </td>
</tr>"""

        table_html = f"""
<div style="background:white;border-radius:14px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-top:16px">
  <div style="padding:16px 20px;background:#f8fafc;border-bottom:1px solid #e2e8f0">
    <span style="font-weight:700;font-size:15px;color:#0f172a">🔴 Top SOZ Channels</span>
    <span style="margin-left:8px;font-size:13px;color:#64748b">(Ranked by probability)</span>
  </div>
  <table style="width:100%;border-collapse:collapse">
    <thead>
      <tr style="background:#f8fafc">
        <th style="padding:10px 14px;text-align:left;font-size:12px;color:#94a3b8;font-weight:600">RANK</th>
        <th style="padding:10px 14px;text-align:left;font-size:12px;color:#94a3b8;font-weight:600">CHANNEL</th>
        <th style="padding:10px 14px;text-align:left;font-size:12px;color:#94a3b8;font-weight:600">CONFIDENCE</th>
        <th style="padding:10px 14px;text-align:left;font-size:12px;color:#94a3b8;font-weight:600">SCORE</th>
        <th style="padding:10px 14px;text-align:left;font-size:12px;color:#94a3b8;font-weight:600">STATUS</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""

        clinical = f"""
<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:12px;padding:16px 20px;margin-top:16px">
  <div style="font-weight:700;color:#92400e;margin-bottom:6px">⚕️ Clinical Note</div>
  <div style="color:#78350f;font-size:14px;line-height:1.6">
    <b>{n_soz} channels</b> identified as potential Seizure Onset Zone with probability ≥ {threshold:.2f}.
    These channels may be involved in seizure generation and are recommended for further clinical review.
    <br><b>This is a computational prediction — not a final diagnosis.</b>
  </div>
</div>"""

        summary_html = cards_html + table_html + clinical

        # ── BAR CHART ───────────────────────────────
        colors_bar = ['#dc2626' if p >= threshold else '#16a34a' for p in probs]
        fig_bar = go.Figure(go.Bar(
            x=ch_names, y=probs,
            marker_color=colors_bar,
            marker_line_width=0,
            text=[f'{p:.2f}' for p in probs],
            textposition='outside',
            textfont=dict(size=9)
        ))
        fig_bar.add_hline(y=threshold, line_dash='dot', line_color='#f59e0b',
                          line_width=2, annotation_text=f'Threshold {threshold:.2f}',
                          annotation_font_color='#f59e0b')
        fig_bar.update_layout(
            title=dict(text='SOZ Probability per Channel', font=dict(size=16, color='#0f172a')),
            xaxis=dict(title='EEG Channel', tickfont=dict(size=9), tickangle=-45),
            yaxis=dict(title='SOZ Probability', range=[0, 1.15]),
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=50,r=30,t=60,b=100), height=420,
            shapes=[dict(type='rect', xref='paper', yref='y',
                        x0=0, x1=1, y0=threshold, y1=1.15,
                        fillcolor='rgba(220,38,38,0.04)', line_width=0)]
        )
        fig_bar.update_xaxes(showgrid=False, showline=True, linecolor='#e2e8f0')
        fig_bar.update_yaxes(showgrid=True, gridcolor='#f1f5f9', showline=True, linecolor='#e2e8f0')

        # ── PIE CHART ───────────────────────────────
        fig_pie = go.Figure(go.Pie(
            labels=['SOZ Channels', 'Normal Channels'],
            values=[n_soz, n_norm],
            marker=dict(colors=['#dc2626','#16a34a'],
                        line=dict(color='white', width=3)),
            hole=0.55,
            textinfo='label+percent',
            textfont=dict(size=13)
        ))
        fig_pie.add_annotation(
            text=f'<b>{n_soz}</b><br>SOZ',
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color='#dc2626')
        )
        fig_pie.update_layout(
            title=dict(text='Channel Distribution', font=dict(size=16, color='#0f172a')),
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=20,r=20,t=60,b=20), height=380,
            legend=dict(orientation='h', y=-0.1)
        )

        # ── HEATMAP ─────────────────────────────────
        cols = min(8, n_tot)
        rows_n = (n_tot + cols - 1) // cols
        mat = np.full((rows_n, cols), np.nan)
        nms = [['' for _ in range(cols)] for _ in range(rows_n)]
        for i, (p, c) in enumerate(zip(probs, ch_names)):
            r, col = divmod(i, cols)
            mat[r, col] = p
            nms[r][col] = c

        fig_heat = go.Figure(go.Heatmap(
            z=mat, text=nms,
            texttemplate='<b>%{text}</b><br>%{z:.2f}',
            textfont=dict(size=9),
            colorscale=[[0,'#dcfce7'],[0.3,'#fef9c3'],
                        [0.5,'#fde68a'],[0.7,'#fca5a5'],[1,'#dc2626']],
            zmin=0, zmax=1,
            colorbar=dict(title='Probability', tickformat='.1f',
                         thickness=15, len=0.8)
        ))
        fig_heat.update_layout(
            title=dict(text='Channel SOZ Probability Heatmap', font=dict(size=16, color='#0f172a')),
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=30,r=80,t=60,b=30),
            height=max(320, rows_n * 65),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )

        # ── CSV ─────────────────────────────────────
        df = pd.DataFrame({
            'Channel'        : ch_names,
            'SOZ_Probability': probs.round(4),
            'Prediction'     : ['SOZ' if p==1 else 'Normal' for p in preds],
            'Risk_Level'     : ['High' if p>=0.7 else 'Medium' if p>=0.5 else 'Low' if p>=threshold else 'Normal'
                                for p in probs]
        }).sort_values('SOZ_Probability', ascending=False)

        csv_path = '/tmp/soz_predictions.csv'
        df.to_csv(csv_path, index=False)

        return fig_bar, fig_pie, fig_heat, summary_html, csv_path, ""

    except Exception as e:
        import traceback
        return [None]*5 + [f"❌ Error: {str(e)}\n{traceback.format_exc()}"]


# ==========================================
# CSS
# ==========================================
CSS = """
* { box-sizing: border-box; }
body { background: #f1f5f9 !important; }
.gradio-container { 
    max-width: 1400px !important; 
    background: #f1f5f9 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1d4ed8 100%);
    border-radius: 18px;
    padding: 32px 40px;
    margin-bottom: 24px;
    color: white;
}
.sidebar-card {
    background: white;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
    border: 1px solid #e2e8f0;
    margin-bottom: 16px;
}
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
}
footer { display: none !important; }
.gr-button-primary {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px !important;
    letter-spacing: 0.3px !important;
}
"""

# ==========================================
# APP LAYOUT
# ==========================================
with gr.Blocks(css=CSS, title="GATNet — SOZ Detection") as app:

    # Header
    gr.HTML("""
    <div class="main-header">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px">
        <div style="font-size:42px">🧠</div>
        <div>
          <div style="font-size:26px;font-weight:800;letter-spacing:-0.5px">GATNet</div>
          <div style="font-size:14px;opacity:.75;margin-top:2px">Seizure Onset Zone Detection System</div>
        </div>
        <div style="margin-left:auto;text-align:right">
          <div style="background:rgba(255,255,255,.15);border-radius:8px;padding:8px 16px;font-size:13px">
            🟢 Model Active &nbsp;|&nbsp; AUC: 0.8315 &nbsp;|&nbsp; 310 Patients
          </div>
        </div>
      </div>
      <div style="display:flex;gap:24px;font-size:13px;opacity:.8">
        <span>📊 Graph Attention Network v4</span>
        <span>🏥 Omni-iEEG Dataset</span>
        <span>⚡ CUDA Accelerated</span>
        <span>🔬 Pre-Surgical Planning Tool</span>
      </div>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT SIDEBAR ──────────────────────────
        with gr.Column(scale=1, min_width=320):

            gr.HTML('<div style="font-weight:700;font-size:15px;color:#0f172a;margin-bottom:12px">📁 Input Configuration</div>')

            with gr.Group():
                file_input = gr.File(
                    label="Upload iEEG File",
                    file_types=['.edf', '.EDF', '.pt'],
                    elem_id="file_upload"
                )
                threshold = gr.Slider(
                    minimum=0.1, maximum=0.9,
                    value=0.35, step=0.05,
                    label="Detection Threshold",
                    info="↓ Lower = More Sensitive  |  Higher = More Specific ↑"
                )
                predict_btn = gr.Button(
                    "🔍  Run SOZ Detection",
                    variant="primary",
                    size="lg"
                )

            # Model Info Card
            gr.HTML("""
            <div class="sidebar-card" style="margin-top:16px">
              <div style="font-weight:700;font-size:14px;color:#0f172a;margin-bottom:14px">⚙️ Model Information</div>
              <div style="display:flex;flex-direction:column;gap:10px;font-size:13px">
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f1f5f9">
                  <span style="color:#64748b">Architecture</span>
                  <span style="font-weight:600;color:#0f172a">GAT v4 (3-layer)</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f1f5f9">
                  <span style="color:#64748b">Dataset</span>
                  <span style="font-weight:600;color:#0f172a">Omni-iEEG</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f1f5f9">
                  <span style="color:#64748b">Patients</span>
                  <span style="font-weight:600;color:#0f172a">310 (8 centers)</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f1f5f9">
                  <span style="color:#64748b">ROC-AUC</span>
                  <span style="font-weight:600;color:#16a34a">0.8315 ✓ Best</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f1f5f9">
                  <span style="color:#64748b">F1-Score</span>
                  <span style="font-weight:600;color:#0f172a">0.6130</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0">
                  <span style="color:#64748b">Task</span>
                  <span style="font-weight:600;color:#0f172a">SOZ Detection</span>
                </div>
              </div>
            </div>
            """)

            # Supported Formats
            gr.HTML("""
            <div class="sidebar-card">
              <div style="font-weight:700;font-size:14px;color:#0f172a;margin-bottom:12px">📋 Supported Formats</div>
              <div style="display:flex;flex-direction:column;gap:8px;font-size:13px">
                <div style="display:flex;align-items:center;gap:10px;padding:8px;background:#f0fdf4;border-radius:8px">
                  <span style="font-size:18px">📄</span>
                  <div>
                    <div style="font-weight:600;color:#166534">.EDF File</div>
                    <div style="color:#4b5563;font-size:12px">Raw iEEG recording</div>
                  </div>
                </div>
                <div style="display:flex;align-items:center;gap:10px;padding:8px;background:#eff6ff;border-radius:8px">
                  <span style="font-size:18px">🔷</span>
                  <div>
                    <div style="font-weight:600;color:#1e40af">.PT File</div>
                    <div style="color:#4b5563;font-size:12px">Processed graph data</div>
                  </div>
                </div>
              </div>
            </div>
            """)

            error_box = gr.Textbox(visible=False, label="Error")

        # ── MAIN CONTENT ──────────────────────────
        with gr.Column(scale=3):

            # Stats + Table
            summary_html = gr.HTML("""
            <div style="background:white;border-radius:14px;padding:48px;text-align:center;
                        box-shadow:0 1px 4px rgba(0,0,0,.06);border:1px solid #e2e8f0">
              <div style="font-size:48px;margin-bottom:16px">🧠</div>
              <div style="font-size:18px;font-weight:600;color:#0f172a;margin-bottom:8px">
                Ready for Analysis
              </div>
              <div style="font-size:14px;color:#64748b">
                Upload an iEEG file and click Run SOZ Detection
              </div>
            </div>
            """)

            # Charts
            with gr.Row():
                bar_plot  = gr.Plot(label="Channel Probabilities")
                pie_plot  = gr.Plot(label="SOZ Distribution")

            heat_plot = gr.Plot(label="Probability Heatmap")

            # Download
            with gr.Row():
                csv_out = gr.File(
                    label="📥 Download Full Report (CSV)",
                    visible=True
                )

    # Disclaimer
    gr.HTML("""
    <div style="background:white;border-radius:12px;padding:16px 24px;margin-top:16px;
                border:1px solid #e2e8f0;display:flex;align-items:center;gap:16px">
      <span style="font-size:22px">⚠️</span>
      <div style="font-size:13px;color:#475569">
        <b style="color:#0f172a">Research Use Only.</b>
        This tool provides computational predictions to assist clinical review.
        All results must be validated by qualified medical professionals before clinical application.
        &nbsp;|&nbsp; <b>FYP Project</b> — GATNet SOZ Detection &nbsp;|&nbsp; Omni-iEEG Dataset
      </div>
    </div>
    """)

    predict_btn.click(
        fn=predict_file,
        inputs=[file_input, threshold],
        outputs=[bar_plot, pie_plot, heat_plot, summary_html, csv_out, error_box]
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
