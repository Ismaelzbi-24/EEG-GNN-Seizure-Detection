/*
 * EEG-GNN in-browser engine (GAT v4).
 *
 * A JavaScript port of webapp/api/pipeline.py so the analysis runs on the visitor's own computer
 * and the recording is never uploaded. Each step copies what MNE / SciPy / PyTorch do in Python:
 *   EDF read (MNE units + calibration) -> resample to 1000 Hz (MNE FFT method) ->
 *   band-pass 1-300 Hz (the exact MNE FIR kernel, zero-phase, reflect_limited padding) ->
 *   drop flat channels, keep the first 64 -> 60 s windows, 30 s step -> 18 features per channel
 *   (SciPy Welch etc.) -> |Pearson r| > 0.3 edges -> training StandardScaler, clip ±10 -> GAT v4.
 *
 * Filtering is done per window with enough context around it, which gives the same numbers as
 * filtering the whole recording but needs far less memory.
 *
 * Works in a Web Worker (importScripts) and in Node (require) for testing.
 */
(function (root) {
  "use strict";

  class PipelineError extends Error {
    constructor(msg) { super(msg); this.name = "PipelineError"; this.userFacing = true; }
  }

  // ── FFT ──────────────────────────────────────────────────────────────────────
  // Mixed-radix Stockham for sizes made of small primes; Bluestein for everything else.
  const PLANS = new Map();

  function factorize(n) {
    const f = [];
    let m = n;
    while (m % 4 === 0) { f.push(4); m /= 4; }
    for (const p of [2, 3, 5]) while (m % p === 0) { f.push(p); m /= p; }
    for (let p = 7; m > 1 && p * p <= m; p += 2) while (m % p === 0) { f.push(p); m /= p; }
    if (m > 1) f.push(m);
    return f;
  }

  function getPlan(n) {
    let plan = PLANS.get(n);
    if (plan) return plan;
    const f = factorize(n);
    plan = f.some((p) => p > 13) ? makeBluestein(n) : makeStockham(n, f);
    if (PLANS.size > 24) PLANS.clear();
    PLANS.set(n, plan);
    return plan;
  }

  function makeStockham(n, factors) {
    const stages = [];
    let L = 1;
    for (const p of factors) {
      const Ls = L; L = Ls * p;
      const twr = new Float64Array(Ls * p), twi = new Float64Array(Ls * p);
      for (let j = 0; j < Ls; j++) for (let q = 0; q < p; q++) {
        const a = (-2 * Math.PI * j * q) / L;
        twr[j * p + q] = Math.cos(a); twi[j * p + q] = Math.sin(a);
      }
      const dr = new Float64Array(p * p), di = new Float64Array(p * p);
      for (let q = 0; q < p; q++) for (let q2 = 0; q2 < p; q2++) {
        const a = (-2 * Math.PI * ((q * q2) % p)) / p;
        dr[q * p + q2] = Math.cos(a); di[q * p + q2] = Math.sin(a);
      }
      stages.push({ p, Ls, L, r: n / L, twr, twi, dr, di });
    }
    return { kind: "stockham", n, stages, br: new Float64Array(n), bi: new Float64Array(n) };
  }

  function stockham(plan, re, im, inverse) {
    const n = plan.n;
    let xr = re, xi = im, yr = plan.br, yi = plan.bi;
    const s = inverse ? -1 : 1;
    const tr = new Float64Array(16), ti = new Float64Array(16);
    for (const st of plan.stages) {
      const { p, Ls, L, r, twr, twi } = st;
      const stride = Ls * r; // = n / p
      if (p === 2) {
        for (let j = 0; j < Ls; j++) {
          const wr = twr[j * 2 + 1], wi = s * twi[j * 2 + 1];
          for (let k = 0; k < r; k++) {
            const i0 = j + Ls * k, i1 = i0 + stride;
            const br = xr[i1] * wr - xi[i1] * wi, bi = xr[i1] * wi + xi[i1] * wr;
            const o = j + L * k;
            yr[o] = xr[i0] + br; yi[o] = xi[i0] + bi;
            yr[o + Ls] = xr[i0] - br; yi[o + Ls] = xi[i0] - bi;
          }
        }
      } else if (p === 4) {
        for (let j = 0; j < Ls; j++) {
          const w1r = twr[j * 4 + 1], w1i = s * twi[j * 4 + 1];
          const w2r = twr[j * 4 + 2], w2i = s * twi[j * 4 + 2];
          const w3r = twr[j * 4 + 3], w3i = s * twi[j * 4 + 3];
          for (let k = 0; k < r; k++) {
            const i0 = j + Ls * k, i1 = i0 + stride, i2 = i1 + stride, i3 = i2 + stride;
            const a0r = xr[i0], a0i = xi[i0];
            const a1r = xr[i1] * w1r - xi[i1] * w1i, a1i = xr[i1] * w1i + xi[i1] * w1r;
            const a2r = xr[i2] * w2r - xi[i2] * w2i, a2i = xr[i2] * w2i + xi[i2] * w2r;
            const a3r = xr[i3] * w3r - xi[i3] * w3i, a3i = xr[i3] * w3i + xi[i3] * w3r;
            const t0r = a0r + a2r, t0i = a0i + a2i, t1r = a0r - a2r, t1i = a0i - a2i;
            const t2r = a1r + a3r, t2i = a1i + a3i, t3r = a1r - a3r, t3i = a1i - a3i;
            const o = j + L * k;
            yr[o] = t0r + t2r; yi[o] = t0i + t2i;
            yr[o + 2 * Ls] = t0r - t2r; yi[o + 2 * Ls] = t0i - t2i;
            if (s > 0) {
              yr[o + Ls] = t1r + t3i; yi[o + Ls] = t1i - t3r;
              yr[o + 3 * Ls] = t1r - t3i; yi[o + 3 * Ls] = t1i + t3r;
            } else {
              yr[o + Ls] = t1r - t3i; yi[o + Ls] = t1i + t3r;
              yr[o + 3 * Ls] = t1r + t3i; yi[o + 3 * Ls] = t1i - t3r;
            }
          }
        }
      } else {
        const { dr, di } = st;
        for (let j = 0; j < Ls; j++) {
          for (let k = 0; k < r; k++) {
            for (let q = 0; q < p; q++) {
              const idx = j + Ls * k + stride * q;
              const wr = twr[j * p + q], wi = s * twi[j * p + q];
              tr[q] = xr[idx] * wr - xi[idx] * wi; ti[q] = xr[idx] * wi + xi[idx] * wr;
            }
            const o = j + L * k;
            for (let q2 = 0; q2 < p; q2++) {
              let sr = 0, si = 0;
              for (let q = 0; q < p; q++) {
                const cr = dr[q * p + q2], ci = s * di[q * p + q2];
                sr += tr[q] * cr - ti[q] * ci; si += tr[q] * ci + ti[q] * cr;
              }
              yr[o + Ls * q2] = sr; yi[o + Ls * q2] = si;
            }
          }
        }
      }
      let t = xr; xr = yr; yr = t; t = xi; xi = yi; yi = t;
    }
    if (xr !== re) { re.set(xr); im.set(xi); }
    void n;
  }

  function makeBluestein(n) {
    let m = 1;
    while (m < 2 * n - 1) m *= 2;
    const wr = new Float64Array(n), wi = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const a = (-Math.PI * ((k * k) % (2 * n))) / n;
      wr[k] = Math.cos(a); wi[k] = Math.sin(a);
    }
    const Br = new Float64Array(m), Bi = new Float64Array(m);
    Br[0] = wr[0]; Bi[0] = -wi[0];
    for (let k = 1; k < n; k++) { Br[k] = Br[m - k] = wr[k]; Bi[k] = Bi[m - k] = -wi[k]; }
    const inner = getPlan(m);
    stockham(inner, Br, Bi, false);
    return { kind: "bluestein", n, m, wr, wi, Br, Bi, inner, ar: new Float64Array(m), ai: new Float64Array(m) };
  }

  function bluestein(plan, re, im, inverse) {
    const { n, m, wr, wi, Br, Bi, inner, ar, ai } = plan;
    const s = inverse ? -1 : 1; // inverse = conj(forward(conj(x)))
    ar.fill(0); ai.fill(0);
    for (let k = 0; k < n; k++) {
      const xr = re[k], xi = s * im[k];
      ar[k] = xr * wr[k] - xi * wi[k]; ai[k] = xr * wi[k] + xi * wr[k];
    }
    stockham(inner, ar, ai, false);
    for (let k = 0; k < m; k++) {
      const r = ar[k] * Br[k] - ai[k] * Bi[k], i = ar[k] * Bi[k] + ai[k] * Br[k];
      ar[k] = r; ai[k] = i;
    }
    stockham(inner, ar, ai, true);
    for (let k = 0; k < n; k++) {
      const cr = ar[k] / m, ci = ai[k] / m;
      re[k] = cr * wr[k] - ci * wi[k];
      im[k] = s * (cr * wi[k] + ci * wr[k]);
    }
  }

  /** In-place complex FFT (unscaled; the inverse is not divided by n). */
  function fft(re, im, inverse) {
    const plan = getPlan(re.length);
    if (plan.kind === "stockham") stockham(plan, re, im, inverse);
    else bluestein(plan, re, im, inverse);
  }

  // ── EDF ──────────────────────────────────────────────────────────────────────
  const ascii = (u8, a, n) => {
    let s = "";
    for (let i = a; i < a + n; i++) s += String.fromCharCode(u8[i]);
    return s;
  };
  const num = (s) => {
    const v = parseFloat(s.trim().replace(",", "."));
    return Number.isFinite(v) ? v : NaN;
  };

  // EDF record durations are decimal strings, e.g. "1" or "0.5": keep them as an exact fraction.
  function decimalFraction(s) {
    s = s.trim();
    const m = /^(\d*)(?:\.(\d*))?$/.exec(s);
    if (!m) { const v = num(s); return [Math.round(v * 1e6), 1e6]; }
    const frac = m[2] || "";
    return [parseInt((m[1] || "0") + frac, 10), 10 ** frac.length];
  }
  const gcd = (a, b) => { a = Math.abs(a); b = Math.abs(b); while (b) [a, b] = [b, a % b]; return a; };

  function uniqueNames(names) {
    const count = {};
    names.forEach((n) => (count[n] = (count[n] || 0) + 1));
    const seen = {};
    return names.map((n) => {
      if (count[n] < 2) return n;
      const k = seen[n] || 0; seen[n] = k + 1;
      return `${n}-${k}`;
    });
  }

  async function readEdfHeader(src) {
    const h0 = await src.read(0, 256);
    if (h0.length < 256) throw new PipelineError("Could not read this file as EDF: the file is too short.");
    if (h0[0] === 0xff) throw new PipelineError("This looks like a BDF (24-bit) file. Please convert it to EDF first.");
    const ns = parseInt(ascii(h0, 252, 4).trim(), 10);
    const headerBytes = parseInt(ascii(h0, 184, 8).trim(), 10);
    if (!(ns > 0) || headerBytes !== 256 + 256 * ns)
      throw new PipelineError("Could not read this file as EDF: the header is not valid.");
    const h = await src.read(0, headerBytes);
    let pos = 256;
    const field = (len) => { const out = []; for (let i = 0; i < ns; i++) { out.push(ascii(h, pos, len)); pos += len; } return out; };
    const labels = field(16).map((s) => s.trim());
    field(80); // transducer
    const dims = field(8).map((s) => s.trim());
    const pmin = field(8).map(num), pmax = field(8).map(num);
    const dmin = field(8).map(num), dmax = field(8).map(num);
    field(80); // prefilter
    const nsamp = field(8).map((s) => parseInt(s.trim(), 10));

    let [durNum, durDen] = decimalFraction(ascii(h0, 244, 8));
    if (!(durNum > 0)) { durNum = 1; durDen = 1; } // MNE treats a 0 record length as 1 s
    const recSamples = nsamp.reduce((a, b) => a + b, 0);
    const recBytes = 2 * recSamples;
    const nRecHeader = parseInt(ascii(h0, 236, 8).trim(), 10);
    const nRecFile = Math.floor(Math.floor((src.size - headerBytes) / 2) / recSamples);
    const nRecords = nRecHeader === nRecFile ? nRecHeader : nRecFile;
    if (!(nRecords > 0)) throw new PipelineError("Could not read this file as EDF: it contains no data records.");

    const chOffset = []; // byte offset of each signal inside a record
    let acc = 0;
    for (let i = 0; i < ns; i++) { chOffset.push(acc); acc += 2 * nsamp[i]; }

    const sel = [];
    labels.forEach((l, i) => { if (l !== "EDF Annotations" && l !== "BDF Annotations") sel.push(i); });
    if (!sel.length) throw new PipelineError("This EDF file has no signal channels.");
    const nsMax = Math.max(...sel.map((i) => nsamp[i]));
    const chans = sel.map((i) => {
      const u = dims[i];
      const gain = ["μV", "µV", "uV"].includes(u) ? 1e-6 : u === "mV" ? 1e-3 : 1;
      const cal = (pmax[i] - pmin[i]) / (dmax[i] - dmin[i]);
      return { idx: i, ns: nsamp[i], byteOff: chOffset[i], cal, off: pmin[i] - dmin[i] * cal, gain };
    });
    const names = uniqueNames(sel.map((i) => labels[i]));
    chans.forEach((c, k) => (c.name = names[k]));
    return {
      headerBytes, recBytes, nRecords, nsMax, chans,
      // sampling frequency as an exact fraction: nsMax / (durNum / durDen)
      sfreqNum: nsMax * durDen, sfreqDen: durNum,
      sfreq: (nsMax * durDen) / durNum,
      nSamples: nRecords * nsMax,
    };
  }

  // Raw (physical, volts) samples [a, b) at the file's own rate for the given channels.
  async function readRaw(src, hdr, chans, a, b) {
    const { nsMax, recBytes, headerBytes, nRecords } = hdr;
    const r0 = Math.floor(a / nsMax);
    const r1 = Math.min(nRecords - 1, Math.floor((b - 1) / nsMax) + 1); // +1: neighbour for slow channels
    const bytes = await src.read(headerBytes + r0 * recBytes, (r1 - r0 + 1) * recBytes);
    const buf = bytes.byteOffset % 2 === 0 ? bytes : bytes.slice();
    const i16 = new Int16Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 2));
    const nRec = r1 - r0 + 1;
    return chans.map((c) => {
      const out = new Float64Array(b - a);
      const k = c.cal * c.gain, o = c.off * c.gain;
      const base = c.byteOff / 2;
      if (c.ns === nsMax) {
        for (let t = a; t < b; t++) {
          const r = Math.floor(t / nsMax) - r0, p = t - (r + r0) * nsMax;
          out[t - a] = i16[r * (recBytes / 2) + base + p] * k + o;
        }
      } else {
        // Slower channel: linear interpolation onto the fastest channel's time grid.
        const own = new Float64Array(nRec * c.ns);
        for (let r = 0; r < nRec; r++) for (let p = 0; p < c.ns; p++)
          own[r * c.ns + p] = i16[r * (recBytes / 2) + base + p] * k + o;
        for (let t = a; t < b; t++) {
          const x = (t / nsMax - r0) * c.ns, i = Math.floor(x), f = x - i;
          const v0 = own[Math.min(i, own.length - 1)], v1 = own[Math.min(i + 1, own.length - 1)];
          out[t - a] = v0 + (v1 - v0) * f;
        }
      }
      return out;
    });
  }

  // Standard deviation of every channel over the whole file (one streaming pass).
  async function channelStd(src, hdr, onProgress) {
    const { chans, recBytes, headerBytes, nRecords } = hdr;
    const per = Math.max(1, Math.floor((16 * 1048576) / recBytes));
    const n = chans.map(() => 0), s1 = chans.map(() => 0), s2 = chans.map(() => 0), ref = chans.map(() => null);
    for (let r0 = 0; r0 < nRecords; r0 += per) {
      const cnt = Math.min(per, nRecords - r0);
      const bytes = await src.read(headerBytes + r0 * recBytes, cnt * recBytes);
      const buf = bytes.byteOffset % 2 === 0 ? bytes : bytes.slice();
      const i16 = new Int16Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 2));
      chans.forEach((c, ci) => {
        const base = c.byteOff / 2;
        let a = s1[ci], b = s2[ci];
        if (ref[ci] === null) ref[ci] = i16[base];
        const rf = ref[ci];
        for (let r = 0; r < cnt; r++) {
          const st = r * (recBytes / 2) + base;
          for (let p = 0; p < c.ns; p++) { const d = i16[st + p] - rf; a += d; b += d * d; }
        }
        s1[ci] = a; s2[ci] = b; n[ci] += cnt * c.ns;
      });
      onProgress(Math.min(1, (r0 + cnt) / nRecords));
    }
    return chans.map((c, ci) => {
      const m = s1[ci] / n[ci];
      return Math.sqrt(Math.max(0, s2[ci] / n[ci] - m * m)) * Math.abs(c.cal * c.gain);
    });
  }

  // ── RESAMPLING (MNE's FFT method, boxcar window) ─────────────────────────────
  function fftResample(x, newLen) {
    const oldLen = x.length;
    const shorter = newLen < oldLen, useLen = shorter ? newLen : oldLen;
    const re = Float64Array.from(x), im = new Float64Array(oldLen);
    fft(re, im, false);
    const half = Math.floor(oldLen / 2) + 1;
    const hr = new Float64Array(Math.floor(newLen / 2) + 1), hi = new Float64Array(hr.length);
    for (let k = 0; k < Math.min(half, hr.length); k++) { hr[k] = re[k]; hi[k] = im[k]; }
    if (useLen % 2 === 0) {
      const nyq = useLen / 2, f = shorter ? 2 : 0.5;
      if (nyq < hr.length) { hr[nyq] *= f; hi[nyq] *= f; }
    }
    const scale = newLen / oldLen;
    // irfft(newLen): rebuild the Hermitian spectrum (DC and an even-length Nyquist bin are real).
    const yr = new Float64Array(newLen), yi = new Float64Array(newLen);
    yr[0] = hr[0] * scale;
    for (let k = 1; k < hr.length; k++) {
      if (newLen % 2 === 0 && k === newLen / 2) { yr[k] = hr[k] * scale; continue; }
      yr[k] = hr[k] * scale; yi[k] = hi[k] * scale;
      yr[newLen - k] = hr[k] * scale; yi[newLen - k] = -hi[k] * scale;
    }
    fft(yr, yi, true);
    for (let k = 0; k < newLen; k++) yr[k] /= newLen;
    return yr;
  }

  // ── SIGNAL ACCESS AT 1000 Hz ─────────────────────────────────────────────────
  // Returns samples [lo, hi) of the 1000 Hz signal for each channel. Outside the recording the
  // signal is mirrored the way MNE pads before filtering ("reflect_limited": 2*x[edge] - x[mirror]).
  async function signal1000(ctx, lo, hi) {
    const { n } = ctx;
    const cl = Math.max(0, lo), ch = Math.min(n, hi);
    const core = await core1000(ctx, cl, ch);
    return core.map((x) => {
      if (lo === cl && hi === ch) return x;
      const out = new Float64Array(hi - lo);
      for (let t = lo; t < hi; t++) {
        let v;
        if (t < 0) v = 2 * x[0 - cl] - x[-t - cl];
        else if (t >= n) v = 2 * x[n - 1 - cl] - x[2 * (n - 1) - t - cl];
        else v = x[t - cl];
        out[t - lo] = v;
      }
      return out;
    });
  }

  async function core1000(ctx, cl, ch) {
    const { src, hdr, used } = ctx;
    if (ctx.P === ctx.Q) return readRaw(src, hdr, used, cl, ch); // already 1000 Hz
    // ratio = P / Q (1000 Hz samples per original sample). Work on a block aligned so that both the
    // original and the resampled lengths are whole numbers, with ~2 s of context on each side.
    const { P, Q, nOrig } = ctx;
    const s0 = Math.floor(cl / P) * P, s1 = Math.ceil(ch / P) * P;
    const a0 = (s0 / P) * Q, a1 = (s1 / P) * Q;
    const M = Math.ceil((2 * hdr.sfreq) / Q) * Q;
    // Extend the right-hand context so the block length / Q only has factors 2, 3, 5 (fast FFT).
    let units = (a1 + 2 * M - a0) / Q;
    const smooth = (v) => { for (const p of [2, 3, 5]) while (v % p === 0) v /= p; return v === 1; };
    while (!smooth(units)) units++;
    const A = a0 - M, B = A + units * Q;
    const ra = Math.max(0, A), rb = Math.min(nOrig, B);
    const raw = await readRaw(src, hdr, used, ra, rb);
    const newLen = ((B - A) / Q) * P, drop = (M / Q) * P;
    return raw.map((x) => {
      const seg = new Float64Array(B - A);
      for (let t = A; t < B; t++) {
        let v;
        if (t < 0) v = 2 * x[0 - ra] - x[Math.min(-t, rb - 1) - ra];
        else if (t >= nOrig) v = 2 * x[nOrig - 1 - ra] - x[Math.max(2 * (nOrig - 1) - t, ra) - ra];
        else v = x[t - ra];
        seg[t - A] = v;
      }
      const y = fftResample(seg, newLen);
      return y.subarray(drop + (cl - s0), drop + (cl - s0) + (ch - cl));
    });
  }

  // ── BAND-PASS FILTER (zero-phase FIR, same kernel as MNE) ────────────────────
  const H_CACHE = new Map();
  function filterSpectrum(h, nfft) {
    let H = H_CACHE.get(nfft);
    if (H) return H;
    const re = new Float64Array(nfft), im = new Float64Array(nfft);
    re.set(h);
    fft(re, im, false);
    H = { re, im };
    H_CACHE.clear(); H_CACHE.set(nfft, H);
    return H;
  }

  // segs: arrays of length W + 2E (E = half filter length); returns the W filtered middle samples.
  function bandpass(segs, h, W) {
    const E = (h.length - 1) / 2;
    const L = segs[0].length;
    let nfft = 1;
    while (nfft < L + h.length - 1) nfft *= 2;
    const H = filterSpectrum(h, nfft);
    const out = new Array(segs.length);
    const re = new Float64Array(nfft), im = new Float64Array(nfft);
    for (let c = 0; c < segs.length; c += 2) {
      re.fill(0); im.fill(0);
      re.set(segs[c]);
      if (c + 1 < segs.length) im.set(segs[c + 1]);
      fft(re, im, false);
      for (let k = 0; k < nfft; k++) {
        const r = re[k] * H.re[k] - im[k] * H.im[k], i = re[k] * H.im[k] + im[k] * H.re[k];
        re[k] = r; im[k] = i;
      }
      fft(re, im, true);
      const a = new Float64Array(W), b = new Float64Array(W);
      for (let t = 0; t < W; t++) { a[t] = re[t + 2 * E] / nfft; b[t] = im[t + 2 * E] / nfft; }
      out[c] = a;
      if (c + 1 < segs.length) out[c + 1] = b;
    }
    return out;
  }

  // ── FEATURES (same 18 as code/01_single_patient.py) ──────────────────────────
  const HANN = new Map();
  function hann(n) {
    let w = HANN.get(n);
    if (!w) {
      w = new Float64Array(n);
      for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n); // periodic, like SciPy
      HANN.set(n, w);
    }
    return w;
  }

  // scipy.signal.welch(x, fs, nperseg): Hann, 50 % overlap, constant detrend, density, one-sided.
  function welch(x, fs, nperseg) {
    const w = hann(nperseg);
    let wss = 0;
    for (let i = 0; i < nperseg; i++) wss += w[i] * w[i];
    const scale = 1 / (fs * wss);
    const step = nperseg - Math.floor(nperseg / 2);
    const nseg = Math.floor((x.length - nperseg) / step) + 1;
    const nf = Math.floor(nperseg / 2) + 1;
    const psd = new Float64Array(nf);
    const re = new Float64Array(nperseg), im = new Float64Array(nperseg);
    const fillSeg = (arr, st) => {
      let m = 0;
      for (let i = 0; i < nperseg; i++) m += x[st + i];
      m /= nperseg;
      for (let i = 0; i < nperseg; i++) arr[i] = (x[st + i] - m) * w[i];
    };
    for (let sg = 0; sg < nseg; sg += 2) {
      fillSeg(re, sg * step);
      if (sg + 1 < nseg) fillSeg(im, (sg + 1) * step); else im.fill(0);
      fft(re, im, false);
      for (let k = 0; k < nf; k++) {
        const nk = (nperseg - k) % nperseg;
        const ar = (re[k] + re[nk]) / 2, ai = (im[k] - im[nk]) / 2;
        const br = (re[k] - re[nk]) / 2, bi = (im[k] + im[nk]) / 2;
        psd[k] += ar * ar + ai * ai;
        if (sg + 1 < nseg) psd[k] += br * br + bi * bi;
      }
    }
    const lastDoubled = nperseg % 2 === 0 ? nf - 1 : nf; // even length: Nyquist bin is not doubled
    for (let k = 0; k < nf; k++) {
      psd[k] = (psd[k] / nseg) * scale;
      if (k > 0 && k < lastDoubled) psd[k] *= 2;
    }
    const f = new Float64Array(nf);
    for (let k = 0; k < nf; k++) f[k] = (k * fs) / nperseg;
    return { f, psd };
  }

  function trapz(y, x, lo, hi) { // over indices [lo, hi]
    let s = 0;
    for (let i = lo; i < hi; i++) s += ((x[i + 1] - x[i]) * (y[i] + y[i + 1])) / 2;
    return s;
  }

  function bandpower(f, psd, fmin, fmax) {
    let lo = -1, hi = -1;
    for (let k = 0; k < f.length; k++) if (f[k] >= fmin && f[k] <= fmax) { if (lo < 0) lo = k; hi = k; }
    return lo < 0 ? 0 : trapz(psd, f, lo, hi);
  }

  function channelFeatures(ch, fs) {
    const n = ch.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += ch[i];
    mean /= n;
    let m2 = 0, m3 = 0, m4 = 0, sq = 0, ll = 0, zc = 0;
    for (let i = 0; i < n; i++) {
      const d = ch[i] - mean, d2 = d * d;
      m2 += d2; m3 += d2 * d; m4 += d2 * d2; sq += ch[i] * ch[i];
      if (i > 0) {
        ll += Math.abs(ch[i] - ch[i - 1]);
        if (Math.sign(ch[i]) !== Math.sign(ch[i - 1])) zc++;
      }
    }
    m2 /= n; m3 /= n; m4 /= n;
    const { f, psd } = welch(ch, fs, Math.min(fs, n));
    const totalP = trapz(psd, f, 0, f.length - 1) + 1e-10;
    let peak = 0, ent = 0;
    for (let k = 1; k < psd.length; k++) if (psd[k] > psd[peak]) peak = k;
    for (let k = 0; k < psd.length; k++) {
      const p = psd[k] / (totalP + 1e-10);
      if (p > 0) ent -= p * Math.log2(p + 1e-10);
    }
    return [
      mean, Math.sqrt(m2), m2, m3 / Math.pow(m2, 1.5), m4 / (m2 * m2) - 3,
      Math.sqrt(sq / n), ll, zc,
      bandpower(f, psd, 0.5, 4), bandpower(f, psd, 4, 8), bandpower(f, psd, 8, 13),
      bandpower(f, psd, 13, 30), bandpower(f, psd, 30, 80), bandpower(f, psd, 80, 150),
      bandpower(f, psd, 80, 300), totalP, f[peak], ent,
    ];
  }

  // Undirected edges where |Pearson r| > thresh (both directions), or a ring if there are none.
  function buildNeighbours(win, thresh) {
    const n = win.length, T = win[0].length;
    const z = win.map((x) => {
      let m = 0;
      for (let i = 0; i < T; i++) m += x[i];
      m /= T;
      const out = new Float64Array(T);
      let ss = 0;
      for (let i = 0; i < T; i++) { out[i] = x[i] - m; ss += out[i] * out[i]; }
      const sd = Math.sqrt(ss);
      for (let i = 0; i < T; i++) out[i] = sd > 0 ? out[i] / sd : NaN;
      return out;
    });
    const nb = Array.from({ length: n }, () => []);
    const R = new Float32Array(n * n);
    let edges = 0;
    for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) {
      let r = 0;
      const za = z[a], zb = z[b];
      for (let i = 0; i < T; i++) r += za[i] * zb[i];
      if (!Number.isFinite(r)) r = 0;
      r = Math.max(-1, Math.min(1, r));
      R[a * n + b] = R[b * n + a] = r;
      if (Math.abs(r) > thresh) { nb[a].push(b); nb[b].push(a); edges++; }
    }
    if (!edges) for (let a = 0; a < n; a++) { const b = (a + 1) % n; nb[a].push(b); nb[b].push(a); }
    return { nb, edges, R };
  }

  // Each electrode's k strongest links by |r| (deduplicated): [a, b, |r|] with a < b.
  function strongestLinks(R, n, k) {
    const seen = new Map();
    for (let a = 0; a < n; a++) {
      const best = [];
      for (let b = 0; b < n; b++) if (b !== a) best.push([Math.abs(R[a * n + b]), b]);
      best.sort((p, q) => q[0] - p[0]);
      best.slice(0, k).forEach(([r, b]) => seen.set(Math.min(a, b) * n + Math.max(a, b), Math.round(r * 1e3) / 1e3));
    }
    return [...seen].map(([key, r]) => [Math.floor(key / n), key % n, r]);
  }

  // ── MODEL (GAT v4, eval mode) ────────────────────────────────────────────────
  function loadModel(meta, buffer) {
    const all = new Float32Array(buffer);
    const T = {};
    for (const [name, t] of Object.entries(meta.tensors)) {
      const size = t.shape.reduce((a, b) => a * b, 1);
      T[name] = all.subarray(t.offset, t.offset + size);
    }
    return { meta, T, filter: Float64Array.from(meta.filter), scaler: meta.scaler, params: meta.params };
  }

  function linear(x, N, inD, W, b, outD) {
    const y = new Float64Array(N * outD);
    for (let n = 0; n < N; n++) {
      const xo = n * inD;
      for (let o = 0; o < outD; o++) {
        let s = b ? b[o] : 0;
        const wo = o * inD;
        for (let i = 0; i < inD; i++) s += x[xo + i] * W[wo + i];
        y[n * outD + o] = s;
      }
    }
    return y;
  }

  function batchNorm(x, N, D, T, pre, eps) {
    const w = T[pre + ".weight"], b = T[pre + ".bias"], m = T[pre + ".running_mean"], v = T[pre + ".running_var"];
    for (let n = 0; n < N; n++) for (let d = 0; d < D; d++) {
      const i = n * D + d;
      x[i] = ((x[i] - m[d]) / Math.sqrt(v[d] + eps)) * w[d] + b[d];
    }
    return x;
  }

  const elu = (x) => { for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = Math.exp(x[i]) - 1; return x; };
  const relu = (x) => { for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0; return x; };

  // PyG GATConv (concat heads, self-loops added, LeakyReLU 0.2, softmax over incoming edges).
  function gatConv(x, N, inD, T, pre, H, C, nb, slope) {
    const HC = H * C;
    const h = linear(x, N, inD, T[pre + ".lin.weight"], null, HC);
    const attS = T[pre + ".att_src"], attD = T[pre + ".att_dst"], bias = T[pre + ".bias"];
    const as = new Float64Array(N * H), ad = new Float64Array(N * H);
    for (let n = 0; n < N; n++) for (let k = 0; k < H; k++) {
      let s = 0, d = 0;
      for (let c = 0; c < C; c++) { const v = h[n * HC + k * C + c]; s += v * attS[k * C + c]; d += v * attD[k * C + c]; }
      as[n * H + k] = s; ad[n * H + k] = d;
    }
    const out = new Float64Array(N * HC);
    for (let i = 0; i < N; i++) {
      const src = [i, ...nb[i]];
      for (let k = 0; k < H; k++) {
        const e = src.map((j) => { const v = as[j * H + k] + ad[i * H + k]; return v > 0 ? v : v * slope; });
        const mx = Math.max(...e);
        let sum = 0;
        for (let q = 0; q < e.length; q++) { e[q] = Math.exp(e[q] - mx); sum += e[q]; }
        for (let q = 0; q < e.length; q++) {
          const a = e[q] / (sum + 1e-16), j = src[q];
          for (let c = 0; c < C; c++) out[i * HC + k * C + c] += a * h[j * HC + k * C + c];
        }
      }
      for (let d = 0; d < HC; d++) out[i * HC + d] += bias[d];
    }
    return out;
  }

  function forward(model, x, N, nb) {
    const { T } = model;
    const { in: F, hidden: Hd, heads, negative_slope: slope, bn_eps: eps } = model.meta.arch;
    const s1 = linear(x, N, F, T["skip1.weight"], T["skip1.bias"], Hd);
    const x1 = elu(batchNorm(gatConv(x, N, F, T, "gat1", heads, Hd, nb, slope), N, Hd * heads, T, "bn1", eps));
    const s2 = linear(x1, N, Hd * heads, T["skip2.weight"], T["skip2.bias"], Hd);
    const x2 = elu(batchNorm(gatConv(x1, N, Hd * heads, T, "gat2", heads, Hd, nb, slope), N, Hd * heads, T, "bn2", eps));
    const x3 = batchNorm(gatConv(x2, N, Hd * heads, T, "gat3", 1, Hd, nb, slope), N, Hd, T, "bn3", eps);
    for (let i = 0; i < x3.length; i++) x3[i] += s2[i] + s1[i];
    elu(x3);
    const gc = new Float64Array(Hd);
    for (let n = 0; n < N; n++) for (let d = 0; d < Hd; d++) gc[d] += x3[n * Hd + d] / N;
    const z = new Float64Array(N * 2 * Hd);
    for (let n = 0; n < N; n++) { z.set(x3.subarray(n * Hd, (n + 1) * Hd), n * 2 * Hd); z.set(gc, n * 2 * Hd + Hd); }
    let c = linear(z, N, 2 * Hd, T["classifier.0.weight"], T["classifier.0.bias"], 128);
    c = relu(batchNorm(c, N, 128, T, "classifier.1", eps));
    c = relu(linear(c, N, 128, T["classifier.4.weight"], T["classifier.4.bias"], 64));
    c = linear(c, N, 64, T["classifier.7.weight"], T["classifier.7.bias"], 2);
    const p = new Float64Array(N);
    for (let n = 0; n < N; n++) p[n] = 1 / (1 + Math.exp(c[n * 2] - c[n * 2 + 1]));
    return p;
  }

  // ── RUN ──────────────────────────────────────────────────────────────────────
  const tick = () => new Promise((r) => setTimeout(r, 0));

  /**
   * src: { size, read(offset, length) -> Promise<Uint8Array> }
   * Returns the same result object as the Python API: { channels: [...], info: {...} }.
   */
  async function run(src, model, opts = {}) {
    const maxWindows = opts.maxWindows ?? 40;
    const progress = opts.progress || (() => {});
    const debug = opts.debug ? { windows: [] } : null;
    const P0 = model.params;
    const FS = P0.target_sfreq, WIN = P0.window_sec * FS, STEP = P0.step_sec * FS;

    progress(0, "Reading the recording");
    const hdr = await readEdfHeader(src);
    if (hdr.sfreq < 2 * 300)
      throw new PipelineError(`Sampling rate is ${+hdr.sfreq.toFixed(3)} Hz. The model needs at least 600 Hz ` +
        "(intracranial EEG is usually 500-2000 Hz; scalp EEG is too slow).");

    // resampling ratio 1000 / sfreq as an exact fraction P / Q
    let P = FS * hdr.sfreqDen, Q = hdr.sfreqNum;
    const g = gcd(P, Q); P /= g; Q /= g;
    const nOrig = hdr.nSamples;
    const n = P === Q ? nOrig : Math.round((nOrig * P) / Q);

    const std = await channelStd(src, hdr, (f) => progress(0.02 + 0.1 * f, "Checking every channel"));
    const validIdx = [];
    std.forEach((s, i) => { if (s > 1e-10) validIdx.push(i); });
    if (validIdx.length < 3) throw new PipelineError("Fewer than 3 usable channels in this recording.");
    const usedIdx = validIdx.slice(0, P0.max_channels);
    const used = usedIdx.map((i) => hdr.chans[i]);

    if (n < WIN) throw new PipelineError(`Recording is shorter than ${P0.window_sec} seconds.`);
    let starts = [];
    for (let s = 0; s < n - WIN; s += STEP) starts.push(s);
    if (!starts.length) starts = [0];
    const totalWindows = starts.length;
    if (maxWindows && starts.length > maxWindows) {
      const idx = new Set();
      for (let k = 0; k < maxWindows; k++) idx.add(Math.round((k * (starts.length - 1)) / (maxWindows - 1)));
      starts = [...idx].sort((a, b) => a - b).map((i) => starts[i]);
    }

    const ctx = { src, hdr, used, P, Q, n, nOrig };
    const E = (model.filter.length - 1) / 2;
    const { mean: sm, scale: ss } = model.scaler;
    const per = [];
    // For the results visuals: each minute's strongest links, how often each pair was connected,
    // and every electrode's average scaled features.
    const NU = used.length;
    const windowLinks = [], pairCount = new Uint16Array(NU * NU), featSum = new Float64Array(NU * 18);
    for (let w = 0; w < starts.length; w++) {
      progress(0.12 + (0.88 * w) / starts.length, `Analysing minute-window ${w + 1} of ${starts.length}`);
      await tick();
      const s = starts[w];
      const segs = await signal1000(ctx, s - E, s + WIN + E);
      const win = bandpass(segs, model.filter, WIN);
      const N = win.length;
      const x = new Float64Array(N * 18);
      const raw = debug ? [] : null;
      win.forEach((ch, i) => {
        const f = channelFeatures(ch, FS);
        if (raw) raw.push(f);
        for (let k = 0; k < 18; k++) x[i * 18 + k] = Math.max(-10, Math.min(10, (f[k] - sm[k]) / ss[k]));
      });
      const { nb, edges, R } = buildNeighbours(win, P0.corr_thresh);
      const p = forward(model, x, N, nb);
      per.push(p);
      for (let q = 0; q < x.length; q++) featSum[q] += x[q];
      nb.forEach((list, a) => list.forEach((b) => { if (a < b) pairCount[a * N + b]++; }));
      windowLinks.push(strongestLinks(R, N, 3));
      if (debug) debug.windows.push({ start: s, features: raw, scaled: Array.from(x), neighbours: nb, corr: R, edges, probs: Array.from(p) });
    }
    progress(1, "Done");

    const channels = used.map((c, i) => {
      const tl = per.map((p) => p[i]);
      const mean = tl.reduce((a, b) => a + b, 0) / tl.length;
      return {
        name: c.name,
        index: i, // position in graph.windowLinks
        prob: Math.round(mean * 1e4) / 1e4,
        max_prob: Math.round(Math.max(...tl) * 1e4) / 1e4,
        timeline: tl.map((v) => Math.round(v * 1e3) / 1e3),
        features: Array.from({ length: 18 }, (_, k) => Math.round((featSum[i * 18 + k] / starts.length) * 1e3) / 1e3),
      };
    });
    channels.sort((a, b) => b.prob - a.prob);
    let connected = 0;
    for (let q = 0; q < pairCount.length; q++) if (pairCount[q] * 2 >= starts.length) connected++;
    const result = {
      channels,
      graph: { windowLinks, connectedPairs: connected, totalPairs: (NU * (NU - 1)) / 2 },
      info: {
        original_sfreq: hdr.sfreq,
        duration_sec: Math.round((n / FS) * 10) / 10,
        channels_total: hdr.chans.length,
        channels_usable: validIdx.length,
        channels_used: used.length,
        windows_total: totalWindows,
        windows_analysed: starts.length,
        window_starts_sec: starts.map((s) => s / FS),
      },
    };
    if (debug) result.debug = debug;
    return result;
  }

  const api = { run, loadModel, PipelineError, fft, welch, fftResample, readEdfHeader, _internal: { factorize, channelFeatures, bandpass, buildNeighbours, forward } };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.EEGEngine = api;
})(typeof self !== "undefined" ? self : globalThis);
