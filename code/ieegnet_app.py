"""
iEEGNet — Automated SOZ Detection from Intracranial EEG
FYP Project | UET Taxila
"""

import gradio as gr
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, pickle, mne
import plotly.graph_objects as go
from torch_geometric.nn import GATConv, global_mean_pool
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings, io, base64
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

MODEL_PATH  = "/home/pmilab/EEG_GNN_Project/results/omni_gat_v4/best_model_v4.pt"
SCALER_PATH = "/home/pmilab/EEG_GNN_Project/results/omni_gat_v4/scaler.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── MODEL ────────────────────────────────────────────────────────────────────
class GATv4(nn.Module):
    def __init__(self, in_channels=18, hidden=128, heads=4, dropout=0.25):
        super().__init__()
        self.dropout = dropout
        self.gat1  = GATConv(in_channels,    hidden,        heads=heads, dropout=dropout)
        self.gat2  = GATConv(hidden*heads,   hidden,        heads=heads, dropout=dropout)
        self.gat3  = GATConv(hidden*heads,   hidden,        heads=1,     dropout=dropout)
        self.skip1 = nn.Linear(in_channels,  hidden)
        self.skip2 = nn.Linear(hidden*heads, hidden)
        self.bn1   = nn.BatchNorm1d(hidden*heads)
        self.bn2   = nn.BatchNorm1d(hidden*heads)
        self.bn3   = nn.BatchNorm1d(hidden)
        self.classifier   = nn.Sequential(
            nn.Linear(hidden*2,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,2))
    def forward(self, x, ei, batch):
        s1=self.skip1(x)
        x=F.elu(self.bn1(self.gat1(x,ei))); x=F.dropout(x,self.dropout,self.training)
        s2=self.skip2(x)
        x=F.elu(self.bn2(self.gat2(x,ei))); x=F.dropout(x,self.dropout,self.training)
        x=F.elu(self.bn3(self.gat3(x,ei))+s2+s1)
        gc=global_mean_pool(x,batch)[batch]
        return self.classifier(torch.cat([x,gc],dim=1))

print("Loading iEEGNet model...")
ckpt   = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
model  = GATv4().to(DEVICE); model.load_state_dict(ckpt['model']); model.eval()
with open(SCALER_PATH,'rb') as f: scaler=pickle.load(f)
print(f"Ready on {DEVICE}")

# ── SIGNAL PROCESSING ────────────────────────────────────────────────────────
def bp(sig,fs,lo,hi):
    f,psd=signal.welch(sig,fs=fs,nperseg=min(fs,len(sig)))
    idx=np.logical_and(f>=lo,f<=hi)
    return float(np.trapz(psd[idx],f[idx])) if idx.any() else 0.0

def extract(window,fs):
    feats=[]
    for ch in window:
        f,psd=signal.welch(ch,fs=fs,nperseg=min(fs,len(ch)))
        tp=float(np.trapz(psd,f))+1e-10; pn=psd/tp; pn=pn[pn>0]
        feats.append([float(np.mean(ch)),float(np.std(ch)),float(np.var(ch)),
                      float(skew(ch)),float(kurtosis(ch)),float(np.sqrt(np.mean(ch**2))),
                      float(np.sum(np.abs(np.diff(ch)))),float(np.sum(np.diff(np.sign(ch))!=0)),
                      bp(ch,fs,.5,4),bp(ch,fs,4,8),bp(ch,fs,8,13),bp(ch,fs,13,30),
                      bp(ch,fs,30,80),bp(ch,fs,80,150),bp(ch,fs,80,300),tp,
                      float(f[np.argmax(psd)]),float(-np.sum(pn*np.log2(pn+1e-10)))])
    return np.array(feats,dtype=np.float32)

def edges(window,th=0.3):
    n=window.shape[0]; cor=np.nan_to_num(np.corrcoef(window))
    src,dst=[],[]
    for i in range(n):
        for j in range(i+1,n):
            if abs(cor[i,j])>th: src.extend([i,j]); dst.extend([j,i])
    if not src:
        for i in range(n): src.extend([i,(i+1)%n]); dst.extend([(i+1)%n,i])
    return torch.tensor([src,dst],dtype=torch.long)

# ── INFERENCE ────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_edf(path):
    raw=mne.io.read_raw_edf(path,preload=True,verbose=False)
    fs=raw.info['sfreq']
    if fs!=1000: raw.resample(1000); fs=1000
    raw.filter(1.,300.,fir_design='firwin',verbose=False)
    data=raw.get_data(); chs=raw.ch_names
    valid=[i for i in range(data.shape[0]) if not np.all(np.isnan(data[i])) and np.std(data[i])>1e-10][:64]
    data=data[valid]; chs=[chs[i] for i in valid]
    win=data[:,:min(60000,data.shape[1])]
    sc=np.clip(scaler.transform(extract(win,int(fs))),-10,10)
    x=torch.tensor(sc,dtype=torch.float).to(DEVICE)
    ei=edges(win).to(DEVICE)
    bt=torch.zeros(x.shape[0],dtype=torch.long).to(DEVICE)
    probs=torch.softmax(model(x,ei,bt),dim=1)[:,1].cpu().numpy()
    return probs, chs

@torch.no_grad()
def infer_pt(path):
    from torch_geometric.loader import DataLoader
    gs=torch.load(path,weights_only=False)
    if not isinstance(gs,list): gs=[gs]
    loader=DataLoader(gs,batch_size=32); all_p=[]
    for b in loader:
        b=b.to(DEVICE)
        all_p.extend(torch.softmax(model(b.x,b.edge_index,b.batch),dim=1)[:,1].cpu().numpy())
    return np.array(all_p),[f"Ch_{i}" for i in range(len(all_p))]

# ── CHARTS ───────────────────────────────────────────────────────────────────
PURPLE="#7c3aed"; PINK="#ec4899"; BG="#1a1033"; CARD="#231642"; GRID="#2d1f55"

def chart_bar(chs,probs,thresh):
    cols=['#ec4899' if p>=thresh else '#6d28d9' for p in probs]
    fig=go.Figure(go.Bar(x=chs,y=probs,marker_color=cols,marker_line_width=0,
                         text=[f'{p:.2f}' for p in probs],textposition='outside',
                         textfont=dict(size=8,color='white')))
    fig.add_hline(y=thresh,line_dash='dot',line_color='#f59e0b',line_width=2,
                  annotation_text=f'Threshold {thresh:.2f}',
                  annotation_font=dict(color='#f59e0b',size=11))
    fig.update_layout(title=dict(text='SOZ Probability per Channel',font=dict(color='white',size=15)),
                      xaxis=dict(title='EEG Channel',tickfont=dict(color='#a78bfa',size=8),tickangle=-45,
                                 gridcolor=GRID,linecolor=GRID),
                      yaxis=dict(title='SOZ Probability',range=[0,1.2],tickfont=dict(color='#a78bfa'),
                                 gridcolor=GRID,linecolor=GRID),
                      plot_bgcolor=CARD,paper_bgcolor=BG,
                      margin=dict(l=50,r=30,t=60,b=100),height=400,
                      font=dict(color='white'))
    return fig

def chart_pie(n_soz,n_norm):
    fig=go.Figure(go.Pie(labels=['SOZ Channels','Normal Channels'],values=[n_soz,n_norm],
                         marker=dict(colors=['#ec4899','#7c3aed'],line=dict(color=BG,width=3)),
                         hole=0.55,textinfo='label+percent+value',
                         textfont=dict(size=12,color='white')))
    fig.add_annotation(text=f'<b>{n_soz}</b><br><span style="font-size:12px">SOZ</span>',
                       x=0.5,y=0.5,showarrow=False,font=dict(size=20,color='#ec4899'))
    fig.update_layout(title=dict(text='Channel Distribution',font=dict(color='white',size=15)),
                      plot_bgcolor=CARD,paper_bgcolor=BG,
                      margin=dict(l=20,r=20,t=60,b=40),height=380,
                      legend=dict(font=dict(color='white'),orientation='h',y=-0.1),
                      font=dict(color='white'))
    return fig

def chart_top10(chs,probs,thresh):
    soz_idx=np.where(probs>=thresh)[0]
    if len(soz_idx)==0:
        soz_idx=np.argsort(probs)[-10:][::-1]
    else:
        soz_idx=sorted(soz_idx,key=lambda i:probs[i],reverse=True)[:10]
    top_chs=[chs[i] for i in soz_idx]
    top_p  =[probs[i] for i in soz_idx]
    cols_b =['#ec4899' if p>=thresh else '#7c3aed' for p in top_p]
    fig=go.Figure(go.Bar(x=top_p,y=top_chs,orientation='h',
                         marker_color=cols_b,marker_line_width=0,
                         text=[f'{p:.3f}' for p in top_p],textposition='outside',
                         textfont=dict(color='white',size=10)))
    fig.update_layout(title=dict(text='Top SOZ Channels (Ranked)',font=dict(color='white',size=15)),
                      xaxis=dict(title='SOZ Probability',range=[0,1.15],
                                 tickfont=dict(color='#a78bfa'),gridcolor=GRID,linecolor=GRID),
                      yaxis=dict(tickfont=dict(color='#a78bfa',size=11),autorange='reversed'),
                      plot_bgcolor=CARD,paper_bgcolor=BG,
                      margin=dict(l=120,r=60,t=60,b=50),height=400,
                      font=dict(color='white'))
    return fig

def chart_heat(chs,probs):
    n=len(probs); cols=min(8,n); rows=(n+cols-1)//cols
    mat=np.full((rows,cols),np.nan); nms=[['' for _ in range(cols)] for _ in range(rows)]
    for i,(p,c) in enumerate(zip(probs,chs)):
        r,col=divmod(i,cols); mat[r,col]=p; nms[r][col]=c
    fig=go.Figure(go.Heatmap(z=mat,text=nms,
                             texttemplate='<b>%{text}</b><br>%{z:.2f}',
                             textfont=dict(size=8,color='white'),
                             colorscale=[[0,'#3b1f7a'],[0.3,'#6d28d9'],
                                         [0.6,'#be185d'],[1,'#ec4899']],
                             zmin=0,zmax=1,
                             colorbar=dict(title='Prob',tickfont=dict(color='white'),
                                          titlefont=dict(color='white'),thickness=12)))
    fig.update_layout(title=dict(text='Channel Probability Heatmap',font=dict(color='white',size=15)),
                      plot_bgcolor=CARD,paper_bgcolor=BG,
                      margin=dict(l=30,r=80,t=60,b=30),
                      height=max(300,rows*65),
                      xaxis=dict(showticklabels=False,showgrid=False),
                      yaxis=dict(showticklabels=False,showgrid=False),
                      font=dict(color='white'))
    return fig

# ── MAIN PREDICT ─────────────────────────────────────────────────────────────
def predict(file, threshold):
    if file is None:
        return [None]*4 + ["""
<div style='text-align:center;padding:60px;color:#a78bfa'>
  <div style='font-size:48px;margin-bottom:16px'>🧠</div>
  <div style='font-size:18px;font-weight:600'>Upload an iEEG file to begin analysis</div>
  <div style='font-size:14px;opacity:.7;margin-top:8px'>Supported: .edf, .EDF, .pt</div>
</div>""", None]

    threshold = float(threshold)
    try:
        p = file.name
        if p.lower().endswith('.edf'):
            probs, chs = infer_edf(p)
        elif p.lower().endswith('.pt'):
            probs, chs = infer_pt(p)
        else:
            return [None]*4 + ["❌ Unsupported format.", None]

        preds  = (probs >= threshold).astype(int)
        n_soz  = int(preds.sum())
        n_norm = len(preds) - n_soz
        n_tot  = len(preds)

        soz_sorted = sorted(
            [(chs[i], float(probs[i])) for i in np.where(preds==1)[0]],
            key=lambda x: x[1], reverse=True)

        # Status badge
        if n_soz == 0:
            status_html = """<div style='background:linear-gradient(135deg,#14532d,#166534);
                border-radius:12px;padding:16px 24px;display:flex;align-items:center;gap:16px'>
                <div style='font-size:32px'>✅</div>
                <div><div style='font-size:18px;font-weight:700;color:#4ade80'>No SOZ Channels Detected</div>
                <div style='color:#86efac;font-size:13px;margin-top:4px'>No channels exceeded the detection threshold</div></div></div>"""
        else:
            status_html = f"""<div style='background:linear-gradient(135deg,#7f1d1d,#991b1b);
                border-radius:12px;padding:16px 24px;display:flex;align-items:center;gap:16px'>
                <div style='font-size:32px'>⚠️</div>
                <div><div style='font-size:18px;font-weight:700;color:#fca5a5'>{n_soz} Potential SOZ Channel{"s" if n_soz>1 else ""} Detected</div>
                <div style='color:#fecaca;font-size:13px;margin-top:4px'>Channels showing elevated epileptic activity • Threshold: {threshold:.2f}</div></div></div>"""

        # Stat cards
        cards = f"""
<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0'>
  <div style='background:#231642;border:1px solid #3b1f7a;border-radius:12px;padding:16px;text-align:center'>
    <div style='font-size:11px;color:#a78bfa;letter-spacing:1px;margin-bottom:8px'>TOTAL CHANNELS</div>
    <div style='font-size:32px;font-weight:800;color:white'>{n_tot}</div>
    <div style='font-size:11px;color:#6d28d9;margin-top:4px'>Analyzed</div>
  </div>
  <div style='background:#231642;border:1px solid #be185d;border-radius:12px;padding:16px;text-align:center'>
    <div style='font-size:11px;color:#f9a8d4;letter-spacing:1px;margin-bottom:8px'>SOZ DETECTED</div>
    <div style='font-size:32px;font-weight:800;color:#ec4899'>{n_soz}</div>
    <div style='font-size:11px;color:#be185d;margin-top:4px'>{100*n_soz/n_tot:.1f}%</div>
  </div>
  <div style='background:#231642;border:1px solid #3b1f7a;border-radius:12px;padding:16px;text-align:center'>
    <div style='font-size:11px;color:#a78bfa;letter-spacing:1px;margin-bottom:8px'>NORMAL CHANNELS</div>
    <div style='font-size:32px;font-weight:800;color:#7c3aed'>{n_norm}</div>
    <div style='font-size:11px;color:#6d28d9;margin-top:4px'>{100*n_norm/n_tot:.1f}%</div>
  </div>
  <div style='background:#231642;border:1px solid #3b1f7a;border-radius:12px;padding:16px;text-align:center'>
    <div style='font-size:11px;color:#a78bfa;letter-spacing:1px;margin-bottom:8px'>THRESHOLD</div>
    <div style='font-size:32px;font-weight:800;color:white'>{threshold:.2f}</div>
    <div style='font-size:11px;color:#6d28d9;margin-top:4px'>Detection cutoff</div>
  </div>
</div>"""

        # SOZ table
        rows_html = ""
        for rank, (ch, prob) in enumerate(soz_sorted[:10], 1):
            bar_w = int(prob * 100)
            color = "#ec4899" if prob >= 0.7 else "#be185d" if prob >= 0.5 else "#9333ea"
            risk  = "High" if prob>=0.7 else "Medium" if prob>=0.5 else "Low"
            risk_bg = "#7f1d1d" if prob>=0.7 else "#78350f" if prob>=0.5 else "#4c1d95"
            rows_html += f"""
<tr style='border-bottom:1px solid #2d1f55'>
  <td style='padding:10px 14px;color:#6d28d9;font-weight:700'>#{rank}</td>
  <td style='padding:10px 14px;font-weight:700;color:white;font-family:monospace;font-size:13px'>{ch}</td>
  <td style='padding:10px 14px'>
    <div style='background:#2d1f55;border-radius:20px;height:8px;width:120px'>
      <div style='background:{color};height:8px;border-radius:20px;width:{bar_w}%'></div>
    </div>
  </td>
  <td style='padding:10px 14px;font-weight:700;color:{color};font-size:14px'>{prob:.3f}</td>
  <td style='padding:10px 14px'>
    <span style='background:{risk_bg};color:white;padding:3px 12px;border-radius:20px;font-size:11px;font-weight:600'>{risk}</span>
  </td>
</tr>"""

        table_html = f"""
<div style='background:#231642;border:1px solid #3b1f7a;border-radius:14px;overflow:hidden;margin-top:16px'>
  <div style='padding:14px 20px;background:#1a1033;border-bottom:1px solid #3b1f7a;
              display:flex;align-items:center;justify-content:space-between'>
    <span style='font-weight:700;font-size:14px;color:white'>🔴 Top SOZ Channels</span>
    <span style='font-size:12px;color:#a78bfa'>Ranked by probability</span>
  </div>
  <table style='width:100%;border-collapse:collapse'>
    <thead>
      <tr style='background:#1a1033'>
        <th style='padding:10px 14px;text-align:left;font-size:11px;color:#6d28d9;letter-spacing:1px'>RANK</th>
        <th style='padding:10px 14px;text-align:left;font-size:11px;color:#6d28d9;letter-spacing:1px'>CHANNEL</th>
        <th style='padding:10px 14px;text-align:left;font-size:11px;color:#6d28d9;letter-spacing:1px'>CONFIDENCE</th>
        <th style='padding:10px 14px;text-align:left;font-size:11px;color:#6d28d9;letter-spacing:1px'>SCORE</th>
        <th style='padding:10px 14px;text-align:left;font-size:11px;color:#6d28d9;letter-spacing:1px'>RISK</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""

        clinical = f"""
<div style='background:#1c1030;border:1px solid #f59e0b;border-radius:12px;
            padding:16px 20px;margin-top:16px'>
  <div style='font-weight:700;color:#fbbf24;margin-bottom:6px;font-size:14px'>⚕️ Clinical Note</div>
  <div style='color:#fde68a;font-size:13px;line-height:1.7'>
    <b>{n_soz} channel{"s" if n_soz!=1 else ""}</b> identified as potential Seizure Onset Zone 
    with SOZ probability ≥ {threshold:.2f}. These channels may be involved in seizure generation 
    and are recommended for further clinical review by a qualified epileptologist.<br>
    <span style='opacity:.8'>⚠️ This is a computational prediction to assist clinical review — 
    <b>not a final diagnosis.</b></span>
  </div>
</div>"""

        report_html = status_html + cards + table_html + clinical

        # CSV
        df = pd.DataFrame({'Channel':chs, 'SOZ_Probability':probs.round(4),
                           'Prediction':['SOZ' if p==1 else 'Normal' for p in preds],
                           'Risk_Level':['High' if p>=0.7 else 'Medium' if p>=0.5
                                         else 'Low' if p>=threshold else 'Normal' for p in probs]
                          }).sort_values('SOZ_Probability', ascending=False)
        csv_path = '/tmp/ieegnet_report.csv'
        df.to_csv(csv_path, index=False)

        return (chart_bar(chs,probs,threshold),
                chart_pie(n_soz,n_norm),
                chart_top10(chs,probs,threshold),
                chart_heat(chs,probs),
                report_html, csv_path)

    except Exception as e:
        import traceback
        err = f"<div style='color:#f87171;padding:20px;background:#1a1033;border-radius:8px'>❌ Error: {str(e)}</div>"
        return [None]*4 + [err, None]

# ── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body, .gradio-container { 
    background: #0f0a1f !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
.gradio-container { max-width: 1400px !important; }
.gr-button-primary {
    background: linear-gradient(135deg, #7c3aed, #ec4899) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 15px !important;
    padding: 14px 24px !important; letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
}
.gr-button-primary:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
footer { display: none !important; }
.gr-panel { background: #1a1033 !important; border: 1px solid #2d1f55 !important; border-radius: 14px !important; }
label { color: #a78bfa !important; font-size: 13px !important; }
input[type=range] { accent-color: #7c3aed !important; }
"""

# ── LAYOUT ───────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="iEEGNet — SOZ Detection") as app:

    # ── NAVBAR ──
    gr.HTML("""
    <nav style='background:#0f0a1f;border-bottom:1px solid #2d1f55;
                padding:0 40px;display:flex;align-items:center;
                height:64px;position:sticky;top:0;z-index:100'>
      <div style='display:flex;align-items:center;gap:10px'>
        <div style='font-size:26px'>🧠</div>
        <div style='font-size:18px;font-weight:800;
                    background:linear-gradient(135deg,#a78bfa,#ec4899);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
          iEEGNet
        </div>
      </div>
      <div style='display:flex;gap:32px;margin-left:48px'>
        <a href='#home' style='color:#a78bfa;text-decoration:none;font-size:14px;font-weight:500'>Home</a>
        <a href='#about' style='color:#6d28d9;text-decoration:none;font-size:14px'>About</a>
        <a href='#help' style='color:#6d28d9;text-decoration:none;font-size:14px'>Help</a>
      </div>
      <div style='margin-left:auto'>
        <span style='background:linear-gradient(135deg,#7c3aed,#ec4899);
                     color:white;padding:8px 18px;border-radius:8px;
                     font-size:13px;font-weight:600'>✦ GATNet v4</span>
      </div>
    </nav>
    """)

    # ── HOME HERO ──
    gr.HTML("""
    <div id='home' style='background:radial-gradient(ellipse at top,#2d1f55 0%,#0f0a1f 60%);
                          padding:80px 60px 60px;display:flex;align-items:center;gap:60px'>
      <div style='flex:1'>
        <div style='display:inline-block;background:#2d1f55;border:1px solid #7c3aed;
                    color:#a78bfa;padding:6px 16px;border-radius:20px;font-size:12px;
                    font-weight:600;letter-spacing:1px;margin-bottom:20px'>
          AI-POWERED EPILEPSY RESEARCH TOOL
        </div>
        <h1 style='font-size:42px;font-weight:800;line-height:1.2;color:white;margin-bottom:16px'>
          Automated <span style='background:linear-gradient(135deg,#a78bfa,#ec4899);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
          Seizure Onset Zone</span><br>Detection from iEEG
        </h1>
        <p style='color:#a78bfa;font-size:16px;line-height:1.7;max-width:520px;margin-bottom:32px'>
          Upload your intracranial EEG recording and let our Graph Attention Network 
          identify potential epileptogenic channels to assist pre-surgical planning.
        </p>
        <div style='display:flex;gap:32px'>
          <div style='text-align:center'>
            <div style='font-size:28px;font-weight:800;
                        background:linear-gradient(135deg,#a78bfa,#ec4899);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent'>0.83</div>
            <div style='font-size:12px;color:#6d28d9;margin-top:4px'>ROC-AUC Score</div>
          </div>
          <div style='text-align:center'>
            <div style='font-size:28px;font-weight:800;
                        background:linear-gradient(135deg,#a78bfa,#ec4899);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent'>310</div>
            <div style='font-size:12px;color:#6d28d9;margin-top:4px'>Patients Trained</div>
          </div>
          <div style='text-align:center'>
            <div style='font-size:28px;font-weight:800;
                        background:linear-gradient(135deg,#a78bfa,#ec4899);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent'>8</div>
            <div style='font-size:12px;color:#6d28d9;margin-top:4px'>Medical Centers</div>
          </div>
        </div>
      </div>
      <div style='flex:0 0 320px;text-align:center'>
        <div style='font-size:180px;filter:drop-shadow(0 0 40px #7c3aed)'>🧠</div>
      </div>
    </div>
    """)

    # ── ANALYSIS SECTION ──
    gr.HTML("""
    <div style='padding:40px 60px 20px'>
      <div style='display:inline-block;background:#2d1f55;border:1px solid #7c3aed;
                  color:#a78bfa;padding:6px 16px;border-radius:20px;font-size:12px;
                  font-weight:600;letter-spacing:1px;margin-bottom:12px'>
        ANALYSIS
      </div>
      <h2 style='font-size:28px;font-weight:700;color:white'>Upload & Analyze iEEG Recording</h2>
    </div>
    """)

    with gr.Row(elem_id="analysis", equal_height=False):
        with gr.Column(scale=1, min_width=340):
            with gr.Group():
                gr.HTML("""<div style='background:#1a1033;border:1px solid #2d1f55;
                    border-radius:14px;padding:20px 20px 0'>
                  <div style='font-size:14px;font-weight:700;color:white;margin-bottom:16px'>
                    📁 Upload iEEG File
                  </div>""")
                file_input = gr.File(label="Select File (.edf or .pt)",
                                     file_types=['.edf','.EDF','.pt'])
                threshold = gr.Slider(minimum=0.1, maximum=0.9, value=0.35,
                                      step=0.05, label="Detection Threshold",
                                      info="Lower = More Sensitive | Higher = More Specific")
                predict_btn = gr.Button("🔍  Analyze & Generate Report",
                                        variant="primary", size="lg")
                gr.HTML("</div>")

            gr.HTML("""
            <div style='background:#1a1033;border:1px solid #2d1f55;border-radius:14px;
                        padding:20px;margin-top:12px'>
              <div style='font-size:13px;font-weight:700;color:white;margin-bottom:14px'>
                ⚙️ Model Details
              </div>
              <div style='display:flex;flex-direction:column;gap:8px;font-size:12px'>
                <div style='display:flex;justify-content:space-between;
                            padding:8px;background:#0f0a1f;border-radius:8px'>
                  <span style='color:#6d28d9'>Architecture</span>
                  <span style='color:#a78bfa;font-weight:600'>3-Layer GAT v4</span>
                </div>
                <div style='display:flex;justify-content:space-between;
                            padding:8px;background:#0f0a1f;border-radius:8px'>
                  <span style='color:#6d28d9'>Dataset</span>
                  <span style='color:#a78bfa;font-weight:600'>Omni-iEEG</span>
                </div>
                <div style='display:flex;justify-content:space-between;
                            padding:8px;background:#0f0a1f;border-radius:8px'>
                  <span style='color:#6d28d9'>ROC-AUC</span>
                  <span style='color:#ec4899;font-weight:700'>0.8315 ✓ Best</span>
                </div>
                <div style='display:flex;justify-content:space-between;
                            padding:8px;background:#0f0a1f;border-radius:8px'>
                  <span style='color:#6d28d9'>F1-Score</span>
                  <span style='color:#a78bfa;font-weight:600'>0.6130</span>
                </div>
                <div style='display:flex;justify-content:space-between;
                            padding:8px;background:#0f0a1f;border-radius:8px'>
                  <span style='color:#6d28d9'>Patients</span>
                  <span style='color:#a78bfa;font-weight:600'>310 (8 Centers)</span>
                </div>
              </div>
            </div>
            """)

        with gr.Column(scale=3):
            report_html = gr.HTML("""
            <div style='background:#1a1033;border:1px solid #2d1f55;border-radius:14px;
                        padding:60px;text-align:center'>
              <div style='font-size:56px;margin-bottom:20px'>🧠</div>
              <div style='font-size:20px;font-weight:600;color:white;margin-bottom:8px'>
                Ready for Analysis
              </div>
              <div style='font-size:14px;color:#6d28d9'>
                Upload an iEEG file (.edf or .pt) and click Analyze
              </div>
            </div>
            """)

    # Charts
    gr.HTML("""<div style='padding:0 0 12px'>
      <div style='font-size:16px;font-weight:700;color:white'>📊 Diagnostic Visualizations</div>
    </div>""")

    with gr.Row():
        bar_plot  = gr.Plot()
        pie_plot  = gr.Plot()
    with gr.Row():
        top_plot  = gr.Plot()
        heat_plot = gr.Plot()

    csv_out = gr.File(label="📥 Download Diagnostic Report (CSV)")

    # ── ABOUT ──
    gr.HTML("""
    <div id='about' style='padding:60px;background:#0f0a1f;border-top:1px solid #2d1f55;margin-top:40px'>
      <div style='text-align:center;margin-bottom:48px'>
        <div style='display:inline-block;background:#2d1f55;border:1px solid #7c3aed;
                    color:#a78bfa;padding:6px 16px;border-radius:20px;font-size:12px;
                    font-weight:600;letter-spacing:1px;margin-bottom:16px'>ABOUT iEEGNet</div>
        <h2 style='font-size:32px;font-weight:800;color:white;margin-bottom:12px'>Our Mission</h2>
        <p style='color:#a78bfa;font-size:15px;max-width:600px;margin:0 auto;line-height:1.7'>
          iEEGNet aims to assist epilepsy surgery planning by automatically identifying 
          Seizure Onset Zones from intracranial EEG using state-of-the-art Graph Neural Networks.
        </p>
      </div>
      <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:24px;max-width:900px;margin:0 auto 48px'>
        <div style='background:#1a1033;border:1px solid #2d1f55;border-radius:16px;padding:28px;text-align:center'>
          <div style='font-size:36px;margin-bottom:14px'>🧠</div>
          <div style='font-size:15px;font-weight:700;color:white;margin-bottom:8px'>SOZ Detection</div>
          <div style='font-size:13px;color:#6d28d9;line-height:1.6'>Identify epileptogenic channels from iEEG recordings using Graph Attention Networks</div>
        </div>
        <div style='background:#1a1033;border:1px solid #2d1f55;border-radius:16px;padding:28px;text-align:center'>
          <div style='font-size:36px;margin-bottom:14px'>📊</div>
          <div style='font-size:15px;font-weight:700;color:white;margin-bottom:8px'>Multi-Center Data</div>
          <div style='font-size:13px;color:#6d28d9;line-height:1.6'>Trained on 310 patients from 8 leading epilepsy centers worldwide</div>
        </div>
        <div style='background:#1a1033;border:1px solid #2d1f55;border-radius:16px;padding:28px;text-align:center'>
          <div style='font-size:36px;margin-bottom:14px'>🏥</div>
          <div style='font-size:15px;font-weight:700;color:white;margin-bottom:8px'>Clinical Support</div>
          <div style='font-size:13px;color:#6d28d9;line-height:1.6'>Pre-surgical planning tool to assist epileptologists in treatment decisions</div>
        </div>
      </div>
      <div style='background:#1a1033;border:1px solid #2d1f55;border-radius:16px;padding:32px;max-width:900px;margin:0 auto'>
        <div style='font-size:15px;font-weight:700;color:white;margin-bottom:20px'>👨‍💻 Development Team</div>
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px'>
          <div style='background:#0f0a1f;border-radius:10px;padding:14px;text-align:center'>
            <div style='font-size:11px;color:#6d28d9;margin-bottom:4px'>22-SE-79</div>
            <div style='font-weight:600;color:white'>M. Atif</div>
          </div>
          <div style='background:#0f0a1f;border-radius:10px;padding:14px;text-align:center'>
            <div style='font-size:11px;color:#6d28d9;margin-bottom:4px'>22-SE-95</div>
            <div style='font-weight:600;color:white'>Syed Ismael</div>
          </div>
          <div style='background:#0f0a1f;border-radius:10px;padding:14px;text-align:center'>
            <div style='font-size:11px;color:#6d28d9;margin-bottom:4px'>22-SE-101</div>
            <div style='font-weight:600;color:white'>Laraib</div>
          </div>
        </div>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>
          <div style='background:#0f0a1f;border-radius:10px;padding:14px'>
            <div style='font-size:11px;color:#6d28d9;margin-bottom:4px'>SUPERVISOR</div>
            <div style='font-weight:600;color:#a78bfa'>Dr. Saima Zareen</div>
          </div>
          <div style='background:#0f0a1f;border-radius:10px;padding:14px'>
            <div style='font-size:11px;color:#6d28d9;margin-bottom:4px'>CO-SUPERVISOR</div>
            <div style='font-weight:600;color:#a78bfa'>Dr. Anwar</div>
          </div>
        </div>
        <div style='margin-top:12px;background:#0f0a1f;border-radius:10px;padding:14px;text-align:center'>
          <div style='font-size:11px;color:#6d28d9;margin-bottom:4px'>INSTITUTION</div>
          <div style='font-weight:600;color:white'>University of Engineering and Technology (UET), Taxila</div>
        </div>
      </div>
    </div>
    """)

    # ── HELP ──
    gr.HTML("""
    <div id='help' style='padding:60px;background:#0f0a1f;border-top:1px solid #2d1f55'>
      <div style='text-align:center;margin-bottom:48px'>
        <div style='display:inline-block;background:#2d1f55;border:1px solid #7c3aed;
                    color:#a78bfa;padding:6px 16px;border-radius:20px;font-size:12px;
                    font-weight:600;letter-spacing:1px;margin-bottom:16px'>HELP & SUPPORT</div>
        <h2 style='font-size:32px;font-weight:800;color:white'>How It Works</h2>
      </div>
      <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:0;max-width:900px;margin:0 auto 48px;position:relative'>
        <div style='position:absolute;top:40px;left:16%;right:16%;height:2px;
                    background:linear-gradient(90deg,#7c3aed,#ec4899);z-index:0'></div>
        <div style='text-align:center;position:relative;z-index:1'>
          <div style='width:80px;height:80px;border-radius:50%;
                      background:linear-gradient(135deg,#7c3aed,#ec4899);
                      display:flex;align-items:center;justify-content:center;
                      margin:0 auto 20px;font-size:28px;box-shadow:0 0 20px #7c3aed66'>
            📤
          </div>
          <div style='position:absolute;top:-8px;right:30%;width:24px;height:24px;
                      background:#ec4899;border-radius:50%;display:flex;align-items:center;
                      justify-content:center;font-size:11px;font-weight:800;color:white'>1</div>
          <div style='font-size:15px;font-weight:700;color:white;margin-bottom:8px'>Upload iEEG File</div>
          <div style='font-size:13px;color:#6d28d9;line-height:1.6;max-width:200px;margin:0 auto'>
            Upload your .edf or .pt intracranial EEG recording file
          </div>
        </div>
        <div style='text-align:center;position:relative;z-index:1'>
          <div style='width:80px;height:80px;border-radius:50%;
                      background:linear-gradient(135deg,#7c3aed,#ec4899);
                      display:flex;align-items:center;justify-content:center;
                      margin:0 auto 20px;font-size:28px;box-shadow:0 0 20px #7c3aed66'>
            🤖
          </div>
          <div style='position:absolute;top:-8px;right:30%;width:24px;height:24px;
                      background:#ec4899;border-radius:50%;display:flex;align-items:center;
                      justify-content:center;font-size:11px;font-weight:800;color:white'>2</div>
          <div style='font-size:15px;font-weight:700;color:white;margin-bottom:8px'>AI Analysis</div>
          <div style='font-size:13px;color:#6d28d9;line-height:1.6;max-width:200px;margin:0 auto'>
            GAT model analyzes brain connectivity and channel-level features
          </div>
        </div>
        <div style='text-align:center;position:relative;z-index:1'>
          <div style='width:80px;height:80px;border-radius:50%;
                      background:linear-gradient(135deg,#7c3aed,#ec4899);
                      display:flex;align-items:center;justify-content:center;
                      margin:0 auto 20px;font-size:28px;box-shadow:0 0 20px #7c3aed66'>
            📋
          </div>
          <div style='position:absolute;top:-8px;right:30%;width:24px;height:24px;
                      background:#ec4899;border-radius:50%;display:flex;align-items:center;
                      justify-content:center;font-size:11px;font-weight:800;color:white'>3</div>
          <div style='font-size:15px;font-weight:700;color:white;margin-bottom:8px'>Get Report</div>
          <div style='font-size:13px;color:#6d28d9;line-height:1.6;max-width:200px;margin:0 auto'>
            Receive detailed SOZ predictions with visualizations and downloadable report
          </div>
        </div>
      </div>
      <div style='max-width:900px;margin:0 auto'>
        <div style='font-size:15px;font-weight:700;color:white;margin-bottom:16px'>
          ❓ Frequently Asked Questions
        </div>
        <div style='display:flex;flex-direction:column;gap:8px'>
          <details style='background:#1a1033;border:1px solid #2d1f55;border-radius:10px;padding:16px;cursor:pointer'>
            <summary style='color:white;font-weight:600;font-size:14px;list-style:none'>
              What file formats are supported?
            </summary>
            <div style='color:#a78bfa;font-size:13px;margin-top:10px;line-height:1.6'>
              iEEGNet supports <b>.edf</b> (European Data Format) raw iEEG recordings and 
              <b>.pt</b> (PyTorch) pre-processed graph files. EDF files are resampled to 1000Hz 
              and filtered 1-300Hz automatically.
            </div>
          </details>
          <details style='background:#1a1033;border:1px solid #2d1f55;border-radius:10px;padding:16px;cursor:pointer'>
            <summary style='color:white;font-weight:600;font-size:14px;list-style:none'>
              How do I interpret the results?
            </summary>
            <div style='color:#a78bfa;font-size:13px;margin-top:10px;line-height:1.6'>
              Channels with SOZ probability above the threshold are marked as potential 
              Seizure Onset Zone. <b>High risk (≥0.7)</b> channels show strongest epileptic activity. 
              Lower the threshold for higher sensitivity, raise it for higher specificity.
            </div>
          </details>
          <details style='background:#1a1033;border:1px solid #2d1f55;border-radius:10px;padding:16px;cursor:pointer'>
            <summary style='color:white;font-weight:600;font-size:14px;list-style:none'>
              What threshold should I use?
            </summary>
            <div style='color:#a78bfa;font-size:13px;margin-top:10px;line-height:1.6'>
              Default threshold is <b>0.35</b> which balances sensitivity and specificity. 
              For higher recall (fewer missed SOZ), use 0.25-0.30. 
              For higher precision (fewer false positives), use 0.45-0.50.
            </div>
          </details>
          <details style='background:#1a1033;border:1px solid #2d1f55;border-radius:10px;padding:16px;cursor:pointer'>
            <summary style='color:white;font-weight:600;font-size:14px;list-style:none'>
              Is this a medical diagnosis tool?
            </summary>
            <div style='color:#a78bfa;font-size:13px;margin-top:10px;line-height:1.6'>
              <b>No.</b> iEEGNet is a research tool to assist clinical review. All predictions 
              must be validated by qualified epileptologists before clinical application. 
              This tool is intended for research and educational purposes only.
            </div>
          </details>
        </div>
      </div>
    </div>
    """)

    # ── FOOTER ──
    gr.HTML("""
    <footer style='background:#0a0717;border-top:1px solid #2d1f55;padding:40px 60px'>
      <div style='display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:40px;margin-bottom:32px'>
        <div>
          <div style='display:flex;align-items:center;gap:8px;margin-bottom:12px'>
            <span style='font-size:20px'>🧠</span>
            <span style='font-size:16px;font-weight:800;
                         background:linear-gradient(135deg,#a78bfa,#ec4899);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
              iEEGNet
            </span>
          </div>
          <p style='color:#6d28d9;font-size:13px;line-height:1.6;max-width:240px'>
            AI-powered SOZ detection from intracranial EEG for improved epilepsy care.
          </p>
        </div>
        <div>
          <div style='font-size:13px;font-weight:700;color:white;margin-bottom:14px;
                      letter-spacing:1px'>QUICK LINKS</div>
          <div style='display:flex;flex-direction:column;gap:8px'>
            <a href='#home' style='color:#6d28d9;text-decoration:none;font-size:13px'>Home</a>
            <a href='#about' style='color:#6d28d9;text-decoration:none;font-size:13px'>About</a>
            <a href='#help' style='color:#6d28d9;text-decoration:none;font-size:13px'>Help</a>
          </div>
        </div>
        <div>
          <div style='font-size:13px;font-weight:700;color:white;margin-bottom:14px;
                      letter-spacing:1px'>LEGAL</div>
          <div style='display:flex;flex-direction:column;gap:8px'>
            <span style='color:#6d28d9;font-size:13px'>Research Use Only</span>
            <span style='color:#6d28d9;font-size:13px'>Data Privacy</span>
            <span style='color:#6d28d9;font-size:13px'>Disclaimer</span>
          </div>
        </div>
        <div>
          <div style='font-size:13px;font-weight:700;color:white;margin-bottom:14px;
                      letter-spacing:1px'>INSTITUTION</div>
          <div style='color:#6d28d9;font-size:13px;line-height:1.6'>
            UET Taxila<br>Dept. of Software Engineering<br>FYP 2025-26
          </div>
        </div>
      </div>
      <div style='border-top:1px solid #2d1f55;padding-top:20px;
                  display:flex;justify-content:space-between;align-items:center'>
        <div style='color:#3b1f7a;font-size:12px'>© 2026 iEEGNet. All rights reserved.</div>
        <div style='color:#3b1f7a;font-size:12px'>
          Built with ❤️ for Pakistan's Healthcare | FYP Team — UET Taxila
        </div>
      </div>
    </footer>
    """)

    predict_btn.click(
        fn=predict,
        inputs=[file_input, threshold],
        outputs=[bar_plot, pie_plot, top_plot, heat_plot, report_html, csv_out]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860,
               share=False, inbrowser=True)
