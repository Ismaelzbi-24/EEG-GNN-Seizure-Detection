// Runs the EEG-GNN analysis off the main thread. The file is read straight from the visitor's disk.
/* global EEGEngine */
importScripts("engine.js");

let modelPromise = null;

// Fetch with a few retries, so a dropped connection doesn't leave the page without a model.
async function fetchRetry(url, kind) {
  let last;
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`${url} HTTP ${r.status}`);
      return kind === "json" ? await r.json() : await r.arrayBuffer();
    } catch (e) {
      last = e;
      await new Promise((res) => setTimeout(res, 500 * (attempt + 1)));
    }
  }
  throw last;
}

function getModel() {
  if (!modelPromise) {
    modelPromise = Promise.all([fetchRetry("model/model.json", "json"), fetchRetry("model/model.bin", "bin")])
      .then(([meta, bin]) => EEGEngine.loadModel(meta, bin));
    modelPromise.catch(() => { modelPromise = null; });
  }
  return modelPromise;
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === "load") {
    try {
      const m = await getModel();
      self.postMessage({ type: "ready", model: { name: m.meta.name, fold: m.meta.fold, epoch: m.meta.epoch, f1: m.meta.f1 } });
    } catch (err) {
      self.postMessage({ type: "loadError", message: String(err && err.message || err) });
    }
    return;
  }
  if (msg.type === "run") {
    const file = msg.file;
    try {
      const model = await getModel();
      const src = { size: file.size, read: (off, len) => file.slice(off, off + len).arrayBuffer().then((b) => new Uint8Array(b)) };
      const result = await EEGEngine.run(src, model, {
        maxWindows: msg.maxWindows || 40,
        progress: (p, stage) => self.postMessage({ type: "progress", p, stage }),
      });
      result.model = { name: model.meta.name, device: "your browser", fold: model.meta.fold, epoch: model.meta.epoch, f1: model.meta.f1 };
      self.postMessage({ type: "done", result });
    } catch (err) {
      const text = err && err.userFacing ? err.message : `Unexpected error while analysing: ${err && err.message || err}`;
      self.postMessage({ type: "error", message: text });
    }
  }
};
