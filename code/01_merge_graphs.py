"""
Merge all individual patient .pt files into one final dataset
"""
from pathlib import Path
import torch
import pandas as pd

GRAPHS_DIR  = "/home/pmilab/EEG_GNN_Project/code/patient_graphs"
OUTPUT_FILE = "/home/pmilab/EEG_GNN_Project/code/omni_ieeg_graphs.pt"
META_FILE   = "/home/pmilab/EEG_GNN_Project/code/omni_ieeg_meta.csv"

graphs_dir = Path(GRAPHS_DIR)
pt_files   = sorted(graphs_dir.glob("*.pt"))

print(f"Found {len(pt_files)} patient files — merging...")

all_graphs = []
meta_rows  = []

for pt_file in pt_files:
    try:
        graphs = torch.load(pt_file, weights_only=False)
        all_graphs.extend(graphs)
        meta_rows.append({
            'patient_id': pt_file.stem,
            'n_graphs'  : len(graphs),
            'n_channels': graphs[0].n_channels,
            'has_soz'   : int(graphs[0].y.item()),
            'dataset'   : graphs[0].dataset,
        })
        print(f"  {pt_file.stem}: {len(graphs)} graphs")
    except Exception as e:
        print(f"  ERROR loading {pt_file.stem}: {e}")

print(f"\n{'='*55}")
print(f"  Total graphs   : {len(all_graphs)}")
print(f"  Total patients : {len(meta_rows)}")

if all_graphs:
    labels       = [g.y.item() for g in all_graphs]
    soz_count    = sum(labels)
    normal_count = len(labels) - soz_count
    print(f"  SOZ graphs (1) : {soz_count}")
    print(f"  Normal graphs  : {normal_count}")
    print(f"  Feature dim    : {all_graphs[0].x.shape[1]}")
    print(f"{'='*55}")

    torch.save(all_graphs, OUTPUT_FILE)
    print(f"\n  Saved: {OUTPUT_FILE}")

    pd.DataFrame(meta_rows).to_csv(META_FILE, index=False)
    print(f"  Metadata: {META_FILE}")
