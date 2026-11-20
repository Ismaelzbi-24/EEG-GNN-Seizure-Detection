"""
Make a SYNTHETIC iEEG recording in EDF format, for testing the upload / pipeline only.

It is not real brain data and the model's output on it means nothing clinically.
Some channels get bursts of fast spiky activity so the result isn't completely flat.

Usage:  python make_test_edf.py [output.edf] [minutes] [channels]
Needs only the Python standard library.
"""
import math
import random
import struct
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_test.edf"
minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 3
n_ch = int(sys.argv[3]) if len(sys.argv) > 3 else 24
FS = 1000                      # Hz, the model's working rate
SECS = int(minutes * 60)
PMIN, PMAX = -2000.0, 2000.0   # microvolts
DMIN, DMAX = -32768, 32767
rnd = random.Random(42)

names = [f"{'RT' if i < n_ch // 2 else 'LF'}{i % (n_ch // 2) + 1}" for i in range(n_ch)]
spiky = set(range(3))          # first three channels get the "seizure-like" bursts


def field(s, n):
    return str(s)[:n].ljust(n).encode("ascii")


hdr = b"".join([
    field("0", 8), field("X X X SYNTHETIC_TEST", 80),
    field("Startdate 01-JAN-2026 X EEG-GNN synthetic test file", 80),
    field("01.01.26", 8), field("00.00.00", 8),
    field(256 + 256 * n_ch, 8), field("", 44),
    field(SECS, 8), field(1, 8), field(n_ch, 4),
])
for key, n in [("label", 16), ("trans", 80), ("dim", 8), ("pmin", 8), ("pmax", 8),
               ("dmin", 8), ("dmax", 8), ("pre", 80), ("ns", 8), ("res", 32)]:
    for i in range(n_ch):
        hdr += field({"label": names[i], "trans": "synthetic", "dim": "uV",
                      "pmin": int(PMIN), "pmax": int(PMAX), "dmin": DMIN, "dmax": DMAX,
                      "pre": "none", "ns": FS, "res": ""}[key], n)

phase = [[rnd.uniform(0, 6.28) for _ in range(4)] for _ in range(n_ch)]
freqs = [2.0, 6.0, 10.0, 22.0]
scale = (DMAX - DMIN) / (PMAX - PMIN)

with open(out, "wb") as f:
    f.write(hdr)
    for sec in range(SECS):
        burst = (sec % 40) >= 25          # 15 s of bursts every 40 s
        for c in range(n_ch):
            vals = []
            for k in range(FS):
                t = sec + k / FS
                v = sum((40 / (j + 1)) * math.sin(2 * math.pi * freqs[j] * t + phase[c][j]) for j in range(4))
                v += rnd.gauss(0, 12)
                if c in spiky and burst:
                    v += 160 * math.sin(2 * math.pi * 110 * t) + (300 if k % 125 < 6 else 0)
                d = int(round((v - PMIN) * scale + DMIN))
                vals.append(max(DMIN, min(DMAX, d)))
            f.write(struct.pack(f"<{FS}h", *vals))

print(f"Wrote {out}: {n_ch} channels, {SECS} s at {FS} Hz (synthetic, for testing only)")
