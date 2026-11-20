(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Seeded RNG so drawings look the same on every visit.
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const SVGNS = "http://www.w3.org/2000/svg";

  // ── HERO WAVES ─────────────────────────────────────────────────────────────
  (function heroWaves() {
    const g = $("heroWaves");
    for (let c = 0; c < 6; c++) {
      const r = rng(200 + c);
      const ks = [3 + c, 7 + c * 2, 13 + c, 29 + c * 3].map((k) => ({ k, ph: r() * 6.28, a: 3 + r() * 9 }));
      const burst = 300 + c * 210;
      let d = "";
      for (let x = 0; x <= 3200; x += 4) {
        const xm = x % 1600;
        let y = 70 + c * 100;
        ks.forEach((o) => { y += o.a * Math.sin((o.k * 2 * Math.PI * xm) / 1600 + o.ph); });
        y += Math.exp(-(((xm - burst) / 70) ** 2)) * (c === 3 ? 42 : 22) * Math.sin(xm * 0.55);
        d += (x ? "L" : "M") + x + " " + y.toFixed(1) + " ";
      }
      const p = document.createElementNS(SVGNS, "path");
      p.setAttribute("d", d);
      p.setAttribute("stroke", c === 3 ? "#FF5A36" : "#2A3352");
      p.setAttribute("stroke-width", c === 3 ? "1.6" : "1.3");
      g.appendChild(p);
    }
  })();

  // ── SIMULATION ─────────────────────────────────────────────────────────────
  (function simulation() {
    const W = 520, H = 420;
    // Electrode positions on brain.webp: a right-hemisphere lateral render of the real
    // FreeSurfer fsaverage cortex (frontal lobe on the right). Baked from
    // expo_mockup/assets/mesh/brain_lh.js; first point = onset (right temporal lobe).
    const XY = [[253,322],[213,42],[468,154],[56,239],[386,315],[294,142],[128,154],[394,259],[351,69],[142,310],[263,290],[126,110],[481,265],[233,224],[347,351],[390,146],[215,90],[87,238],[305,233],[157,257],[199,162],[101,182],[448,204],[302,289],[278,73],[414,108],[333,96],[419,296],[280,372],[192,336],[159,94],[112,269],[237,257],[452,277],[259,192],[351,294],[274,99],[357,202],[86,290],[491,211],[101,138],[178,82],[334,307],[151,195],[439,160],[452,309],[370,94],[198,291],[240,128],[169,129],[113,195],[75,181],[344,262],[341,152],[399,191],[282,277],[133,272],[380,300],[194,229],[187,315],[128,224],[316,66]];
    const pts = XY.map(([x, y]) => ({ x, y }));
    const onset = pts[0], oi = 0;
    pts.forEach((p) => (p.d = Math.hypot(p.x - onset.x, p.y - onset.y)));
    const reach = (d) => (d < 26 ? 8 : 12 + d / 9);
    // Region labels from where each contact sits on the lateral view.
    const cnt = {};
    pts.forEach((p) => {
      const reg = p.y > 280 && p.x > 120 && p.x < 410 ? "RT" : p.x > 370 ? "RF" : p.x < 150 ? "RO" : "RP";
      cnt[reg] = (cnt[reg] || 0) + 1; p.name = reg + cnt[reg];
    });

    const brain = $("simBrain");
    brain.innerHTML = '<img class="cortex" src="brain.webp" alt="" draggable="false"><div class="on-cortex"><div class="sz-glow"></div></div><div class="shine"></div>';
    const glow = brain.querySelector(".sz-glow");
    glow.style.left = (onset.x / W) * 100 + "%"; glow.style.top = (onset.y / H) * 100 + "%";
    brain.setAttribute("role", "img");
    brain.setAttribute("aria-label", "Simulated brain network: a seizure starts at one spot in the right temporal lobe and spreads across the electrodes.");

    const seen = {}, edges = [];
    pts.forEach((p, i) => {
      pts.map((q, j) => ({ j, d: Math.hypot(q.x - p.x, q.y - p.y) }))
        .filter((o) => o.j !== i).sort((a, b) => a.d - b.d).slice(0, 3)
        .forEach((o) => {
          const key = Math.min(i, o.j) + "-" + Math.max(i, o.j);
          if (seen[key]) return; seen[key] = 1;
          const q = pts[o.j];
          const e = el("div", "edge");
          e.style.left = (p.x / W) * 100 + "%"; e.style.top = (p.y / H) * 100 + "%";
          e.style.width = (o.d / W) * 100 + "%";
          e.style.transform = `rotate(${(Math.atan2(q.y - p.y, q.x - p.x) * 180) / Math.PI}deg)`;
          brain.appendChild(e);
          edges.push({ i, j: o.j, e });
        });
    });
    const nodes = pts.map((p, i) => {
      const n = el("div", "node");
      n.style.left = (p.x / W) * 100 + "%"; n.style.top = (p.y / H) * 100 + "%";
      n.dataset.i = i;
      brain.appendChild(n);
      return n;
    });
    const ring = el("div", "ring"), tag = el("div", "tag", "AI: it started here");
    [ring, tag].forEach((x) => { x.style.left = (pts[oi].x / W) * 100 + "%"; x.style.top = (pts[oi].y / H) * 100 + "%"; x.hidden = true; brain.appendChild(x); });
    const tip = el("div", "ntip"); tip.hidden = true; brain.appendChild(tip);

    // ── hover: tilt the brain toward the cursor, light up an electrode and its links
    let hov = -1;
    const fine = window.matchMedia("(hover: hover)").matches;
    brain.addEventListener("pointermove", (ev) => {
      const b = brain.getBoundingClientRect();
      const mx = (ev.clientX - b.left) / b.width, my = (ev.clientY - b.top) / b.height;
      brain.style.setProperty("--mx", mx * 100 + "%"); brain.style.setProperty("--my", my * 100 + "%");
      if (fine && !reduceMotion) {
        brain.style.setProperty("--ry", ((mx - 0.5) * 9).toFixed(2) + "deg");
        brain.style.setProperty("--rx", ((0.5 - my) * 7).toFixed(2) + "deg");
      }
      // nearest electrode within ~18 px (in 520-wide units)
      const x = mx * W, y = my * H;
      let bi = -1, bd = 18;
      pts.forEach((p, i) => { const d = Math.hypot(p.x - x, p.y - y); if (d < bd) { bd = d; bi = i; } });
      if (bi !== hov) { hov = bi; render(); }
    });
    brain.addEventListener("pointerenter", () => brain.classList.add("live"));
    brain.addEventListener("pointerleave", () => {
      brain.classList.remove("live"); brain.style.setProperty("--rx", "0deg"); brain.style.setProperty("--ry", "0deg");
      hov = -1; render();
    });

    const byD = pts.map((_, i) => i).sort((a, b) => pts[a].d - pts[b].d);
    const chanIdx = [byD[0], byD[3], byD[Math.floor(byD.length * 0.35)], byD[Math.floor(byD.length * 0.65)], byD[byD.length - 1]];
    const strips = chanIdx.map((ci) => pts[ci].name).map((name, k) => {
      const start = reach(pts[chanIdx[k]].d), rr = rng(100 + k);
      let d = "";
      for (let x = 0; x <= 600; x += 2) {
        const tt = (x / 600) * 60, sz = tt >= start;
        const amp = sz ? 15 + 5 * Math.sin(tt * 0.7) : 4;
        const y = 23 + amp * Math.sin(x * (sz ? 0.9 : 0.25) + k) * (0.6 + rr() * 0.4) + (rr() - 0.5) * (sz ? 6 : 3);
        d += (x ? "L" : "M") + x + " " + y.toFixed(1) + " ";
      }
      const row = el("div", "strip", `<div class="lbl">${name}</div><div class="tr"><svg aria-hidden="true" viewBox="0 0 600 46" preserveAspectRatio="none"><path d="${d}" fill="none" stroke-width="1.2" vector-effect="non-scaling-stroke"/></svg><div class="fog"></div></div>`);
      $("simStrips").appendChild(row);
      return { start, lbl: row.querySelector(".lbl"), path: row.querySelector("path"), fog: row.querySelector(".fog") };
    });

    const PH = [
      [0, "Phase 1 · Calm", "#9FB8FF", "Everything looks normal.", "Brain areas chatter away at a steady rhythm. Nothing stands out yet."],
      [8, "Phase 2 · The spark", "#FF5A36", "One small spot misfires.", "A tiny patch starts firing abnormally. This is the seizure onset zone — the patch surgeons need to find."],
      [22, "Phase 3 · The storm", "#FF5A36", "It spreads in seconds.", "The abnormal activity races across the network. Soon almost every electrode looks “bad”, which is exactly why spotting where it started, by eye, is so hard."],
      [50, "Phase 4 · The answer", "#F2F1EC", "EEG-GNN points to the start.", "By reading the brain as a network — who is connected to whom, and how each area behaves — the model flags the electrodes most likely to be the origin."],
    ];
    let t = 0, playing = false, pred = null, timer = null;
    const PLAY = '<svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><path d="M4 2 L16 9 L4 16 Z" fill="currentColor"/></svg>';
    const PAUSE = '<svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><rect x="3" y="2" width="4" height="14" fill="currentColor"/><rect x="11" y="2" width="4" height="14" fill="currentColor"/></svg>';

    function render() {
      pts.forEach((p, i) => {
        const since = t - reach(p.d);
        let c = "#8C9AC8", s = 9;
        if (since >= 0 && since < 5) { c = "#FF5A36"; s = 14; } else if (since >= 5) { c = "#B04A30"; s = 10; }
        const n = nodes[i];
        if (i === oi && t >= 8) { c = "#FF5A36"; s = 16; n.classList.add("beat"); } else n.classList.remove("beat");
        n.style.background = c; n.style.width = n.style.height = s + "px";
        n.style.color = c;
        n.classList.toggle("hot", since >= 0 && since < 5);
        n.classList.toggle("hov", i === hov);
      });
      const hit = pts.map((p) => t >= reach(p.d));
      edges.forEach(({ i, j, e }) => {
        e.style.background = hit[i] && hit[j] ? "rgba(255,90,54,.42)" : "rgba(159,184,255,.2)";
        e.classList.toggle("lit", hov >= 0 && (i === hov || j === hov));
      });
      // cortex glow around the onset zone grows as the seizure spreads
      const g = t < 8 ? 0 : Math.min(1, (t - 8) / 30);
      glow.style.opacity = (g * 0.9).toFixed(3);
      glow.style.width = 14 + g * 58 + "%";
      if (hov >= 0) {
        const p = pts[hov], since = t - reach(p.d);
        const st = since < 0 ? "Normal activity" : since < 5 ? "Seizure arriving now" : "Taken over";
        const when = hov === oi ? "starts here at 0:08" : "reached at 0:" + String(Math.round(reach(p.d))).padStart(2, "0");
        tip.innerHTML = `<b>${p.name}</b><span>${st}</span><i>${when}</i>`;
        tip.style.left = (p.x / W) * 100 + "%"; tip.style.top = (p.y / H) * 100 + "%";
        tip.classList.toggle("below", p.y < 90);
        tip.hidden = false;
      } else tip.hidden = true;
      const show = pred === null ? t >= 50 : pred;
      ring.hidden = tag.hidden = !show;
      $("simPred").textContent = show ? "Hide AI answer" : "Show AI answer";
      $("simPred").setAttribute("aria-pressed", String(show));
      const ph = PH.filter((p) => t >= p[0]).pop();
      $("simTag").textContent = $("simTag2").textContent = ph[1];
      $("simDot").style.color = $("simTag2").style.color = ph[2];
      $("simTitle").textContent = ph[3]; $("simText").textContent = ph[4];
      const ss = Math.floor(t % 60);
      $("simClock").innerHTML = `0${Math.floor(t / 60)}:${ss < 10 ? "0" : ""}${ss} <i>/ 01:00</i>`;
      strips.forEach((s) => {
        const on = t >= s.start;
        s.path.setAttribute("stroke", on ? "#FF5A36" : "#5A6488");
        s.lbl.style.color = on ? "#FF5A36" : "#8A8F9C";
        s.fog.style.left = (t / 60) * 100 + "%";
      });
      $("simScrub").value = t;
      $("simPlay").innerHTML = playing ? PAUSE : PLAY;
      $("simPlay").setAttribute("aria-label", playing ? "Pause" : "Play");
    }
    function stop() { clearInterval(timer); timer = null; playing = false; }
    $("simPlay").addEventListener("click", () => {
      if (playing) { stop(); return render(); }
      if (t >= 60) { t = 0; pred = null; }
      playing = true;
      timer = setInterval(() => { t = Math.min(60, t + 0.25); if (t >= 60) stop(); render(); }, 50);
      render();
    });
    $("simScrub").addEventListener("input", (e) => { t = parseFloat(e.target.value); render(); });
    $("simPred").addEventListener("click", () => { pred = !(pred === null ? t >= 50 : pred); render(); });
    $("simEnd").addEventListener("click", () => { stop(); t = 60; pred = true; render(); });
    render();

    // Autoplay once when the section scrolls into view.
    if (!reduceMotion && "IntersectionObserver" in window) {
      const io = new IntersectionObserver((es) => {
        if (es[0].isIntersecting) { io.disconnect(); if (!playing && t === 0) $("simPlay").click(); }
      }, { threshold: 0.45 });
      io.observe($("simBrain"));
    }
  })();

  // ── VERSION BARS: fill + count up when scrolled into view ──────────────────
  (function versions() {
    const box = $("vRows");
    if (!box) return;
    const nums = [...box.querySelectorAll("b[data-v]")];
    const go = () => {
      box.classList.add("in");
      nums.forEach((b, k) => {
        const end = parseFloat(b.dataset.v);
        if (reduceMotion) { b.textContent = end.toFixed(1); return; }
        const t0 = performance.now() + k * 150, dur = 1300;
        const step = (now) => {
          const u = Math.max(0, Math.min(1, (now - t0) / dur)), e = 1 - Math.pow(1 - u, 3);
          b.textContent = (end * e).toFixed(1);
          if (u < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
      });
    };
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { io.disconnect(); go(); } }, { threshold: 0.4 });
      io.observe(box);
    } else go();
  })();

  // ── FAQ ────────────────────────────────────────────────────────────────────
  (function faq() {
    const items = [
      ["Does this replace the doctor?", "No. It is a second pair of eyes that suggests which electrodes to check first. Every decision stays with the medical team."],
      ["What file can I use?", "An intracranial EEG recording saved as .edf, at least one minute long and sampled at 600 Hz or more. Scalp EEG is too slow for the high-frequency features the model uses."],
      ["Is my recording uploaded anywhere?", "No. The model runs inside your browser, so the file never leaves your computer and nothing is stored."],
      ["What is a graph neural network?", "An AI built for networks — dots joined by lines — instead of photos or text. The brain is a network, so it is a natural fit."],
      ["Where does the training data come from?", "Omni-iEEG: a public research collection of brain recordings from epilepsy patients at several hospitals, including Detroit, UCLA, HUP and Zurich."],
      ["Can hospitals use it today?", "Not yet. It is a research prototype and needs more testing — especially on data from new hospitals — before any clinical use."],
    ];
    const box = $("faq");
    items.forEach(([q, a], i) => {
      const w = el("div");
      w.innerHTML = `<button type="button" aria-expanded="${i === 0}" aria-controls="faq${i}"><span>${esc(q)}</span><span aria-hidden="true">${i === 0 ? "−" : "+"}</span></button><p id="faq${i}" ${i === 0 ? "" : "hidden"}>${esc(a)}</p>`;
      const b = w.querySelector("button"), p = w.querySelector("p");
      b.addEventListener("click", () => {
        const open = b.getAttribute("aria-expanded") !== "true";
        b.setAttribute("aria-expanded", String(open));
        b.lastElementChild.textContent = open ? "−" : "+";
        p.hidden = !open;
      });
      box.appendChild(w);
    });
  })();

  // ── TRY IT: upload → poll → results ────────────────────────────────────────
  (function analyse() {
    const states = { upload: $("stUpload"), run: $("stRun"), err: $("stErr"), res: $("stRes") };
    const show = (k) => Object.entries(states).forEach(([n, e]) => (e.hidden = n !== k));
    // The model runs in this browser (worker.js + engine.js); nothing is uploaded.
    let ins = null, sel = null; // extra visuals (insights.js) and the selected electrode
    let ready = false, maxMb = 4000, result = null, fileName = "", showAll = false, t0 = 0, clockT = null, worker = null;

    function setApi(ok, text) {
      const led = document.querySelector(".led");
      led.className = "led " + (ok ? "ok" : "bad");
      $("apiText").textContent = text;
      $("drop").classList.toggle("disabled", !ok);
      $("file").disabled = !ok;
    }
    $("maxMb").textContent = maxMb;
    let onWorker = () => {};
    function makeWorker() {
      const w = new Worker("worker.js");
      w.onmessage = (e) => onWorker(e.data);
      w.onerror = (e) => onWorker({ type: "error", message: `The analysis stopped unexpectedly${e.message ? ": " + e.message : ""}.` });
      return w;
    }
    if (!window.Worker) setApi(false, "This browser can’t run the model. Please use a recent Chrome, Edge, Firefox or Safari.");
    else {
      worker = makeWorker();
      onWorker = (m) => {
        if (m.type === "ready") { ready = true; setApi(true, "Model ready · runs privately in your browser"); }
        else if (m.type === "loadError") setApi(false, `The model couldn’t be loaded: ${m.message}`);
      };
      worker.postMessage({ type: "load" });
    }

    const drop = $("drop");
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); if (ready) drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => { if (ready && e.dataTransfer.files[0]) start(e.dataTransfer.files[0]); });
    $("file").addEventListener("change", (e) => { if (e.target.files[0]) start(e.target.files[0]); e.target.value = ""; });
    $("errBack").addEventListener("click", () => show("upload"));
    $("again").addEventListener("click", () => show("upload"));

    function fail(msg) { stopClock(); $("errText").textContent = msg; show("err"); }
    function progress(p, stage) {
      const pct = Math.round(p * 100);
      $("runPct").textContent = pct + "%";
      $("runBar").firstElementChild.style.width = pct + "%";
      $("runBar").setAttribute("aria-valuenow", pct);
      $("runStage").textContent = stage;
    }
    function startClock() {
      t0 = Date.now();
      clockT = setInterval(() => { const s = Math.round((Date.now() - t0) / 1000); $("jobClock").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")} elapsed`; }, 500);
    }
    function stopClock() { clearInterval(clockT); }

    function start(file) {
      if (ins) { ins.destroy(); ins = null; }
      if (!/\.edf$/i.test(file.name)) return fail("That isn’t an .edf file. Please choose an EDF recording.");
      if (file.size > maxMb * 1048576) return fail(`That file is ${(file.size / 1048576).toFixed(0)} MB; the limit is ${maxMb} MB.`);
      fileName = file.name;
      $("runFile").textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
      show("run"); startClock(); progress(0, "Reading the file…");
      let finished = false;
      onWorker = (m) => {
        if (finished) return;
        if (m.type === "progress") progress(m.p, m.stage);
        else if (m.type === "done") {
          finished = true; stopClock(); result = m.result; showAll = false; sel = result.channels[0] ? result.channels[0].name : null;
          show("res");
          if (window.EEGInsights) {
            try { ins = window.EEGInsights.show(result, { threshold: parseFloat(thr.value), selected: sel, onSelect: (n) => { sel = n; renderResult(); } }); }
            catch (err) { console.error(err); $("netBand").hidden = $("fpBand").hidden = true; }
          }
          renderResult();
        }
        else if (m.type === "error" || m.type === "loadError") {
          finished = true; fail(m.message);
          worker.terminate(); worker = makeWorker(); // fresh worker after a crash
        }
      };
      worker.postMessage({ type: "run", file, maxWindows: 40 });
    }

    const thr = $("thr");
    thr.addEventListener("input", () => renderResult());
    $("showAll").addEventListener("click", () => { showAll = !showAll; renderResult(); });

    function heatColor(v) {
      // #141722 → #3A4466 → #FF5A36
      const stops = [[0, [20, 23, 34]], [0.5, [58, 68, 102]], [1, [255, 90, 54]]];
      const k = v <= 0.5 ? 0 : 1, [a, ca] = stops[k], [b, cb] = stops[k + 1];
      const f = (v - a) / (b - a);
      return `rgb(${ca.map((c, i) => Math.round(c + (cb[i] - c) * f)).join(",")})`;
    }

    function renderResult() {
      if (!result) return;
      const th = parseFloat(thr.value);
      $("thrVal").textContent = th.toFixed(2);
      const ch = result.channels, info = result.info;
      const flagged = ch.filter((c) => c.prob >= th);
      $("resFile").textContent = `${fileName} · ${info.duration_sec >= 120 ? (info.duration_sec / 60).toFixed(1) + " min" : info.duration_sec + " s"} · ${info.channels_used} electrodes`;
      $("resTitle").innerHTML = flagged.length
        ? `<span class="hot">${flagged.length}</span> electrode${flagged.length > 1 ? "s" : ""} flagged`
        : "No electrodes flagged";
      $("resSub").textContent = flagged.length
        ? `Most likely: ${flagged.slice(0, 4).map((c) => c.name).join(", ")}${flagged.length > 4 ? ` and ${flagged.length - 4} more` : ""}. These are the places to review first.`
        : "No electrode scored above the threshold. Try lowering the sensitivity threshold to see the strongest candidates.";

      const list = $("chanList"); list.innerHTML = "";
      const rows = showAll ? ch : ch.slice(0, 15);
      rows.forEach((c, i) => {
        const li = el("li", (c.prob >= th ? "flag" : "") + (c.name === sel ? " sel" : ""));
        li.tabIndex = 0; li.setAttribute("role", "button"); li.title = "Show " + c.name + " in the network and fingerprint below";
        li.addEventListener("click", () => { sel = c.name; renderResult(); });
        li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sel = c.name; renderResult(); } });
        li.innerHTML = `<span class="n">${String(i + 1).padStart(2, "0")}</span><span class="nm" title="${esc(c.name)}">${esc(c.name)}</span><span class="tr" aria-hidden="true"><i style="width:${(c.prob * 100).toFixed(1)}%"></i><b style="left:${th * 100}%"></b></span><span class="v">${(c.prob * 100).toFixed(0)}%</span>`;
        list.appendChild(li);
      });
      $("showAll").hidden = ch.length <= 15;
      $("showAll").textContent = showAll ? "Show top 15" : `Show all ${ch.length} electrodes`;

      const heat = $("heat"); heat.innerHTML = "";
      const nW = ch[0] ? ch[0].timeline.length : 0;
      ch.slice(0, 12).forEach((c) => {
        const row = el("div", "row");
        row.innerHTML = `<span class="lb" title="${esc(c.name)}">${esc(c.name)}</span>`;
        const cells = el("div", "cells");
        cells.style.gridTemplateColumns = `repeat(${nW}, minmax(0, 1fr))`;
        c.timeline.forEach((v, k) => {
          const cell = el("i");
          cell.style.background = heatColor(v);
          cell.title = `${c.name} · minute starting ${Math.round(info.window_starts_sec[k] / 60)}:${String(Math.round(info.window_starts_sec[k] % 60)).padStart(2, "0")} · ${(v * 100).toFixed(0)}%`;
          cells.appendChild(cell);
        });
        row.appendChild(cells);
        heat.appendChild(row);
      });

      const sampled = info.windows_analysed < info.windows_total ? ` (${info.windows_analysed} of ${info.windows_total}, evenly spread)` : "";
      if (ins) ins.update({ threshold: th, selected: sel });
      $("resModel").textContent = `${result.model.name}, run in ${result.model.device} · ${info.windows_analysed} minute-windows analysed${sampled}. Score = average over windows.`;
    }

    $("dlCsv").addEventListener("click", () => {
      if (!result) return;
      const th = parseFloat(thr.value);
      const lines = [["rank", "channel", "mean_probability", "max_probability", "flagged_at_threshold_" + th.toFixed(2)].join(",")];
      result.channels.forEach((c, i) => lines.push([i + 1, `"${c.name.replace(/"/g, '""')}"`, c.prob, c.max_prob, c.prob >= th ? 1 : 0].join(",")));
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
      a.download = fileName.replace(/\.edf$/i, "") + "_eeg-gnn.csv";
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    });
  })();
})();
