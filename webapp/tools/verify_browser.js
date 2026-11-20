// Compare the in-browser engine (site/engine.js) with the Python pipeline's output.
// Usage (from webapp/):  node tools/verify_browser.js file.edf reference.json [max_windows]
const fs = require("fs");
const path = require("path");
const E = require("../site/engine.js");

const [edf, refPath, mw] = process.argv.slice(2);
const site = path.join(__dirname, "..", "site", "model");
const meta = JSON.parse(fs.readFileSync(path.join(site, "model.json"), "utf8"));
const bin = fs.readFileSync(path.join(site, "model.bin"));
const model = E.loadModel(meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const ref = JSON.parse(fs.readFileSync(refPath, "utf8"));

const fd = fs.openSync(edf, "r");
const src = {
  size: fs.fstatSync(fd).size,
  read: async (off, len) => {
    const b = Buffer.alloc(len);
    const got = fs.readSync(fd, b, 0, len, off);
    return new Uint8Array(b.buffer, b.byteOffset, got);
  },
};

(async () => {
  const t0 = Date.now();
  const res = await E.run(src, model, { maxWindows: mw ? +mw : 40, debug: true });
  const secs = (Date.now() - t0) / 1000;
  const W = res.debug.windows, R = ref.windows;
  if (W.length !== R.length) throw new Error(`window count ${W.length} vs ${R.length}`);
  let featRel = 0, probAbs = 0, edgeDiff = 0, scaledAbs = 0;
  const { mean, scale } = meta.scaler;
  W.forEach((w, k) => {
    if (w.start !== R[k].start) throw new Error(`start ${w.start} vs ${R[k].start}`);
    edgeDiff = Math.max(edgeDiff, Math.abs(w.edges - R[k].edges));
    w.features.forEach((f, i) => f.forEach((v, j) => {
      const r = R[k].features[i][j];
      featRel = Math.max(featRel, Math.abs(v - r) / Math.max(Math.abs(r), 1e-30));
      const zs = (x) => Math.max(-10, Math.min(10, (x - mean[j]) / scale[j]));
      scaledAbs = Math.max(scaledAbs, Math.abs(zs(v) - zs(r)));
    }));
    w.probs.forEach((p, i) => (probAbs = Math.max(probAbs, Math.abs(p - R[k].probs[i]))));
  });
  const namesMatch = JSON.stringify(res.channels.map((c) => c.name).sort()) === JSON.stringify(ref.result.channels.map((c) => c.name).sort());
  const rankJs = res.channels.slice(0, 5).map((c) => c.name).join(","), rankPy = ref.result.channels.slice(0, 5).map((c) => c.name).join(",");
  console.log(JSON.stringify({
    file: path.basename(edf), seconds: secs, windows: W.length, channels: res.info.channels_used,
    max_feature_rel_err: featRel, max_scaled_feature_abs_err: scaledAbs, max_edge_count_diff: edgeDiff,
    max_prob_abs_err: probAbs, names_match: namesMatch, top5_js: rankJs, top5_py: rankPy,
    info_js: { ...res.info, window_starts_sec: undefined }, info_py: { ...ref.info },
  }, null, 1));
})().catch((e) => { console.error(e); process.exit(1); });
