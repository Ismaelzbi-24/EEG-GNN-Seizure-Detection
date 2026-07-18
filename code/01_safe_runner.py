import subprocess
import sys
from pathlib import Path

DATASET_PATH = "/home/pmilab/EEG_GNN_Project/Hiroki/Data/Data"
LOG_FILE = "/home/pmilab/EEG_GNN_Project/code/safe_runner_log.txt"

patient_dirs = sorted([
    d.name for d in Path(DATASET_PATH).iterdir()
    if d.is_dir() and d.name.startswith('sub-')
])

print(f"Total patients: {len(patient_dirs)}")

with open(LOG_FILE, 'w') as log:
    for i, patient in enumerate(patient_dirs):
        print(f"[{i+1}/{len(patient_dirs)}] Processing {patient}...", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, '01_single_patient.py', patient],
                timeout=120,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                log.write(f"OK: {patient}\n")
                print(f"  OK", flush=True)
            else:
                log.write(f"FAIL: {patient} — {result.stderr[-200:]}\n")
                print(f"  FAILED: {result.stderr[-100:]}", flush=True)
        except subprocess.TimeoutExpired:
            log.write(f"TIMEOUT: {patient}\n")
            print(f"  TIMEOUT — skipping", flush=True)
        except Exception as e:
            log.write(f"ERROR: {patient} — {e}\n")
            print(f"  ERROR: {e}", flush=True)

print("Done! Check safe_runner_log.txt for results")