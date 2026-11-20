"""
SOZ Detection App — GATNet
Seizure Onset Zone Detection using Graph Attention Network
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
import plotly.express as px
from plotly.subplots import make_subplots
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ==========================================
# PATHS
# ==========================================
MODEL_PATH  = "/home/pmilab/EEG_GNN_Project/results/omni_gat_v4/best_model_v4.pt"
SCALER_PATH = "/home/pmilab/EEG_GNN_Project/results/omni_gat_v4/scaler.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# MODEL DEFINITION
# ==========================================
class GATv4(nn.Module):
    def __init__(self, in_channels, hidden=128, heads=4, dropout=0.25):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_channels,    hidden,        heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden * heads, hidden,        heads=heads, dropout=dropout)
        self.gat3 = GATConv(hidden * heads, hidden,        heads=1,     dropout=dropout)
        self.skip1 = nn.Linear(in_channels,    hidden)
        self.skip2 = nn.Linear(hidden * heads, hidden)
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.bn2 = nn.BatchNorm1d(hidden * heads)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden + hidden, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x, edge_index, batch):
        skip1 = self.skip1(x)
        x     = self.gat1(x, edge_index)
        x     = self.bn1(x)
        x     = F.elu(x)
        x     = F.dropout(x, p=self.dropout, training=self.training)
        skip2 = self.skip2(x)
        x     = self.gat2(x, edge_index)
        x     = self.bn2(x)
        x     = F.elu(x)
        x     = F.dropout(x, p=self.dropout, training=self.training)
        x     = self.gat3(x, edge_index)
        x     = self.bn3(x)
        x     = F.elu(x + skip2 + skip1)
        graph_context = global_mean_pool(x, batch)
        graph_context = graph_context[batch]
        combined = torch.cat([x, graph_context], dim=1)
        return self.classifier(combined)


# ==========================================
# LOAD MODEL & SCALER
# ==========================================
def load_model():
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
    model = GATv4(in_channels=18, hidden=128, heads=4, dropout=0.25).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def load_scaler():
    with open(SCALER_PATH, 'rb') as f:
        return pickle.load(f)

print("Loading model and scaler...")
model  = load_model()
scaler = load_scaler()
print(f"Model loaded! Device: {DEVICE}")


# ==========================================
# FEATURE EXTRACTION
# ==========================================
def bandpower(sig, fs, fmin, fmax):
    f, psd = signal.welch(sig, fs=fs, nperseg=min(fs, len(sig)))
    idx = np.logical_and(f >= fmin, f <= fmax)
    return float(np.trapz(psd[idx], f[idx])) if idx.any() else 0.0

def extract_features(window, fs):
    features = []
    for ch_sig in window:
        mean_v  = float(np.mean(ch_sig))
        std_v   = float(np.std(ch_sig))
        var_v   = float(np.var(ch_sig))
        skew_v  = float(skew(ch_sig))
        kurt_v  = float(kurtosis(ch_sig))
        rms_v   = float(np.sqrt(np.mean(ch_sig ** 2)))
        ll_v    = float(np.sum(np.abs(np.diff(ch_sig))))
        zc_v    = float(np.sum(np.diff(np.sign(ch_sig)) != 0))
        delta_p = bandpower(ch_sig, fs, 0.5, 4)
        theta_p = bandpower(ch_sig, fs, 4,   8)
        alpha_p = bandpower(ch_sig, fs, 8,  13)
        beta_p  = bandpower(ch_sig, fs, 13, 30)
        gamma_p = bandpower(ch_sig, fs, 30, 80)
        hgamma  = bandpower(ch_sig, fs, 80, 150)
        hfo_p   = bandpower(ch_sig, fs, 80, 300)
        f, psd  = signal.welch(ch_sig, fs=fs, nperseg=min(fs, len(ch_sig)))
        total_p = float(np.trapz(psd, f)) + 1e-10
        peak_f  = float(f[np.argmax(psd)])
        psd_n   = psd / total_p
        psd_n   = psd_n[psd_n > 0]
        sp_ent  = float(-np.sum(psd_n * np.log2(psd_n + 1e-10)))
        features.append([mean_v, std_v, var_v, skew_v, kurt_v,
                         rms_v, ll_v, zc_v, delta_p, theta_p,
                         alpha_p, beta_p, gamma_p, hgamma, hfo_p,
                         total_p, peak_f, sp_ent])
    return np.array(features, dtype=np.float32)

def build_edges(window, thresh=0.3):
    n   = window.shape[0]
    cor = np.corrcoef(window)
    cor = np.nan_to_num(cor)
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
# PREDICT FROM EDF
# ==========================================
def predict_edf(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    fs  = raw.info['sfreq']
    if fs != 1000:
        raw.resample(1000); fs = 1000
    raw.filter(1., 300., fir_design='firwin', verbose=False)
    data     = raw.get_data()
    ch_names = raw.ch_names
    valid    = [i for i in range(data.shape[0])
                if not np.all(np.isnan(data[i])) and np.std(data[i]) > 1e-10]
    valid    = valid[:64]
    data     = data[valid]
    ch_names = [ch_names[i] for i in valid]
    return data, ch_names, int(fs)

def predict_pt(pt_path):
    graphs = torch.load(pt_path, weights_only=False)
    if not isinstance(graphs, list):
        graphs = [graphs]
    return graphs, None

@torch.no_grad()
def run_inference_edf(data, ch_names, fs):
    win  = min(60*fs, data.shape[1])
    window = data[:, :win]
    feats  = extract_features(window, fs)
    scaled = scaler.transform(feats)
    scaled = np.clip(scaled, -10, 10)
    x      = torch.tensor(scaled, dtype=torch.float).to(DEVICE)
    ei     = build_edges(window).to(DEVICE)
    batch  = torch.zeros(x.shape[0], dtype=torch.long).to(DEVICE)
    out    = model(x, ei, batch)
    probs  = torch.softmax(out, dim=1)[:,1].cpu().numpy()
    return probs, ch_names

@torch.no_grad()
def run_inference_pt(graphs):
    from torch_geometric.loader import DataLoader
    loader = DataLoader(graphs, batch_size=32)
    all_probs = []
    all_names = []
    for batch in loader:
        batch = batch.to(DEVICE)
        out   = model(batch.x, batch.edge_index, batch.batch)
        probs = torch.softmax(out, dim=1)[:,1].cpu().numpy()
        all_probs.extend(probs)
    ch_names = [f"Ch_{i}" for i in range(len(all_probs))]
    return np.array(all_probs), ch_names


# ==========================================
# VISUALIZATIONS
# ==========================================
def make_probability_bar(ch_names, probs, threshold):
    colors = ['#E74C3C' if p >= threshold else '#2ECC71' for p in probs]
    fig = go.Figure(go.Bar(
        x=ch_names, y=probs,
        marker_color=colors,
        text=[f'{p:.2f}' for p in probs],
        textposition='outside'
    ))
    fig.add_hline(y=threshold, line_dash='dash',
                  line_color='orange', line_width=2,
                  annotation_text=f'Threshold ({threshold:.2f})')
    fig.update_layout(
        title='SOZ Probability per Channel',
        xaxis_title='EEG Channel',
        yaxis_title='SOZ Probability',
        yaxis_range=[0, 1.1],
        template='plotly_white',
        height=450,
        font=dict(size=11)
    )
    return fig

def make_pie_chart(n_soz, n_normal):
    fig = go.Figure(go.Pie(
        labels=['SOZ Channels', 'Normal Channels'],
        values=[n_soz, n_normal],
        marker_colors=['#E74C3C', '#2ECC71'],
        hole=0.4,
        textinfo='label+percent+value'
    ))
    fig.update_layout(
        title='Channel Distribution',
        template='plotly_white',
        height=350
    )
    return fig

def make_heatmap(ch_names, probs):
    n    = len(probs)
    cols = min(8, n)
    rows = (n + cols - 1) // cols
    mat  = np.zeros((rows, cols))
    nms  = [[''] * cols for _ in range(rows)]
    for i, (p, c) in enumerate(zip(probs, ch_names)):
        r, col = divmod(i, cols)
        mat[r, col] = p
        nms[r][col] = c
    fig = go.Figure(go.Heatmap(
        z=mat, text=nms,
        texttemplate='%{text}',
        colorscale=[[0,'#2ECC71'],[0.5,'#F39C12'],[1,'#E74C3C']],
        zmin=0, zmax=1,
        colorbar=dict(title='SOZ Prob')
    ))
    fig.update_layout(
        title='Channel SOZ Probability Heatmap',
        template='plotly_white',
        height=max(300, rows*60)
    )
    return fig


# ==========================================
# MAIN PREDICTION FUNCTION
# ==========================================
def predict(file, threshold):
    if file is None:
        return (None, None, None, None,
                "Please upload a file.", None)

    path = file.name
    threshold = float(threshold)

    try:
        if path.endswith('.edf') or path.endswith('.EDF'):
            data, ch_names, fs = predict_edf(path)
            probs, ch_names    = run_inference_edf(data, ch_names, fs)
        elif path.endswith('.pt'):
            graphs, _       = predict_pt(path)
            probs, ch_names = run_inference_pt(graphs)
        else:
            return (None, None, None, None,
                    "Unsupported file. Please upload .edf or .pt file.", None)

        preds   = (probs >= threshold).astype(int)
        n_soz   = int(preds.sum())
        n_total = len(preds)
        n_norm  = n_total - n_soz

        # Top SOZ channels
        soz_idx  = np.where(preds == 1)[0]
        soz_chs  = [(ch_names[i], probs[i]) for i in soz_idx]
        soz_chs  = sorted(soz_chs, key=lambda x: x[1], reverse=True)

        # Summary text
        summary = f"""
## 🧠 SOZ Detection Results

**Total Channels Analyzed:** {n_total}
**SOZ Channels Detected:** {n_soz} ({100*n_soz/n_total:.1f}%)
**Normal Channels:** {n_norm} ({100*n_norm/n_total:.1f}%)
**Threshold Used:** {threshold:.2f}

---

### 🔴 Top SOZ Channels (Highest Probability):
"""
        for i, (ch, prob) in enumerate(soz_chs[:10], 1):
            summary += f"\n{i}. **{ch}** — {prob:.3f} ({prob*100:.1f}%)"

        if n_soz == 0:
            summary += "\n\n✅ No SOZ channels detected at this threshold."
        else:
            summary += f"""

---

### 📋 Clinical Note:
These {n_soz} channels show elevated epileptic activity and may represent 
the Seizure Onset Zone. This is a computational prediction to assist 
clinical review — **not a final diagnosis**.
"""

        # CSV
        df = pd.DataFrame({
            'Channel'        : ch_names,
            'SOZ_Probability': probs.round(4),
            'Prediction'     : ['SOZ' if p == 1 else 'Normal' for p in preds]
        }).sort_values('SOZ_Probability', ascending=False)

        csv_path = '/tmp/soz_predictions.csv'
        df.to_csv(csv_path, index=False)

        # Figures
        fig_bar  = make_probability_bar(ch_names, probs, threshold)
        fig_pie  = make_pie_chart(n_soz, n_norm)
        fig_heat = make_heatmap(ch_names, probs)

        return fig_bar, fig_pie, fig_heat, summary, csv_path

    except Exception as e:
        return (None, None, None,
                f"Error: {str(e)}", None)


# ==========================================
# GRADIO UI
# ==========================================
with gr.Blocks(
    title="GATNet — SOZ Detection",
    theme=gr.themes.Soft(primary_hue="blue")
) as app:

    gr.Markdown("""
    # 🧠 GATNet: Seizure Onset Zone Detection
    ### Graph Attention Network for Pre-Surgical iEEG Analysis
    
    Upload an iEEG recording to identify potential **Seizure Onset Zone (SOZ)** channels.
    This tool assists in pre-surgical epilepsy planning by analyzing intracranial EEG signals.
    
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Input")
            file_input = gr.File(
                label="Upload iEEG File (.edf or .pt)",
                file_types=['.edf', '.EDF', '.pt']
            )
            threshold = gr.Slider(
                minimum=0.1, maximum=0.9,
                value=0.5, step=0.05,
                label="Detection Threshold",
                info="Lower = more sensitive (higher recall), Higher = more specific (higher precision)"
            )
            predict_btn = gr.Button(
                "🔍 Run SOZ Detection",
                variant="primary",
                size="lg"
            )

            gr.Markdown("""
            ### ℹ️ About
            **Model:** GAT v4 (3-layer Graph Attention Network)  
            **Dataset:** Omni-iEEG (310 patients, 8 centers)  
            **Task:** Node-level SOZ channel classification  
            **AUC:** 0.8315 (Best among all baselines)
            """)

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            summary_out = gr.Markdown("*Results will appear here after prediction...*")

    gr.Markdown("---")
    gr.Markdown("### 📈 Visualizations")

    with gr.Row():
        bar_plot  = gr.Plot(label="Channel Probabilities")
        pie_plot  = gr.Plot(label="SOZ Distribution")

    heat_plot = gr.Plot(label="Probability Heatmap")

    gr.Markdown("---")
    csv_out = gr.File(label="📥 Download Predictions CSV")

    gr.Markdown("""
    ---
    ### ⚠️ Disclaimer
    This tool is for **research purposes only**. All predictions should be 
    reviewed by qualified medical professionals before clinical use.
    
    **Developed by:** FYP Team | Omni-iEEG SOZ Detection Project
    """)

    predict_btn.click(
        fn=predict,
        inputs=[file_input, threshold],
        outputs=[bar_plot, pie_plot, heat_plot, summary_out, csv_out]
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
