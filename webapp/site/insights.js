/*
 * Extra result visuals, all drawn from the real analysis result:
 *   A  summary tiles + "how clear is this result" verdict
 *   C  animated electrode network (per-minute scores and strongest links, GAT-style message dots)
 *   E  signal fingerprint: selected electrode's features vs the typical electrode
 * Usage: const ins = EEGInsights.show(result, { threshold, selected, onSelect(name) });
 *        ins.update({ threshold, selected }); ins.destroy();
 */
(() => {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const s = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const lerp = (a, b, t) => a + (b - a) * t;
  const pct = (v) => Math.round(v * 100) + "%";
  const median = (a) => { const b = a.slice().sort((x, y) => x - y), m = b.length >> 1; return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2; };
  const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const heat = (t) => { t = Math.max(0, Math.min(1, t)); const a = [58, 68, 102], b = [255, 90, 54]; return `rgb(${a.map((c, i) => Math.round(lerp(c, b[i], t))).join(",")})`; };
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ICON = {
    play: '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2 L14 8 L4 14 Z" fill="currentColor"/></svg>',
    pause: '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><rect x="3" y="2" width="3.5" height="12" fill="currentColor"/><rect x="9.5" y="2" width="3.5" height="12" fill="currentColor"/></svg>',
    replay: '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 3a5 5 0 1 1-4.6 3" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M2 2v4.5h4.5" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>',
  };

  let tip = null;
  const showTip = (ev, html) => {
    if (!tip) { tip = document.createElement("div"); tip.className = "rtip"; document.body.appendChild(tip); }
    tip.innerHTML = html; tip.hidden = false;
    tip.style.left = Math.min(ev.clientX + 14, innerWidth - 250) + "px"; tip.style.top = ev.clientY + 14 + "px";
  };
  const hideTip = () => { if (tip) tip.hidden = true; };

  const FEATS = [
    ["Amplitude", [[1, "Spread (std)"], [2, "Variance"], [5, "Signal size (RMS)"], [6, "Line length"], [0, "Mean level"]]],
    ["Wave shape", [[3, "Skewness"], [4, "Spikiness (kurtosis)"], [7, "Zero crossings"]]],
    ["Rhythms", [[8, "Delta 0.5–4 Hz"], [9, "Theta 4–8 Hz"], [10, "Alpha 8–13 Hz"], [11, "Beta 13–30 Hz"], [12, "Gamma 30–80 Hz"], [13, "Ripple 80–150 Hz"], [14, "Fast 80–300 Hz"], [15, "Total power"], [16, "Peak frequency"], [17, "Spectral entropy"]]],
  ];

  function show(result, opts) {
    const has = result.graph && result.channels.every((c) => Array.isArray(c.features) && c.index != null);
    $("netBand").hidden = $("fpBand").hidden = !has;
    // engine order (index) for graph data
    const byIdx = [];
    result.channels.forEach((c) => (byIdx[c.index ?? byIdx.length] = c));
    const N = byIdx.length, names = byIdx.map((c) => c.name), idxOf = Object.fromEntries(names.map((n, i) => [n, i]));
    const ranked = result.channels.map((c) => c.name), rankOf = Object.fromEntries(ranked.map((n, i) => [n, i + 1]));
    const st = { thr: opts.threshold, sel: idxOf[opts.selected] ?? idxOf[ranked[0]] };
    const select = (i) => opts.onSelect && opts.onSelect(names[i]);
    const ac = new AbortController(), sig = { signal: ac.signal };

    // ── A ────────────────────────────────────────────────────────────────
    function renderA() {
      const p = result.channels.map((c) => c.prob); // already sorted high → low
      const gap = p.length > 1 ? p[0] - p[1] : p[0], g5 = p[0] - p[Math.min(4, p.length - 1)];
      const flagged = p.filter((v) => v >= st.thr).length;
      $("sumTiles").innerHTML = [
        [ranked[0], `Top electrode · ${pct(p[0])}`],
        [`+${(gap * 100).toFixed(1)} pts`, p.length > 1 ? `Lead over #2 (${esc(ranked[1])})` : "Only one electrode"],
        [`${pct(p[p.length - 1])}–${pct(p[0])}`, "Range of all scores"],
        [`${flagged} of ${p.length}`, `Flagged at ${st.thr.toFixed(2)}`],
      ].map(([v, l]) => `<div><b>${esc(v)}</b><span>${l}</span></div>`).join("");
      let col, chip, text;
      if (gap >= 0.15) { col = "#5FD39A"; chip = "Clear leader"; text = `${ranked[0]} stands well above every other electrode. Review it first.`; }
      else if (g5 >= 0.1) { col = "var(--ice)"; chip = "A few stand out"; text = "The top 5 electrodes are ahead of the rest but close to each other. Review them as a group."; }
      else { col = "var(--hot)"; chip = "Close call"; text = `The top 5 electrodes are within ${(g5 * 100).toFixed(1)} points of each other, so no single electrode stands out. Treat this ranking as weak evidence.`; }
      const v = $("verdict"); v.style.setProperty("--vcol", col);
      v.innerHTML = `<span class="chip">${chip}</span><p>${esc(text)}</p>`;
    }

    // ── E ────────────────────────────────────────────────────────────────
    const pickEl = $("fpPick");
    pickEl.innerHTML = ranked.map((n) => `<option value="${idxOf[n]}">#${rankOf[n]} ${esc(n)} · ${pct(byIdx[idxOf[n]].prob)}</option>`).join("");
    pickEl.addEventListener("change", () => select(+pickEl.value), sig);
    const typical = has ? [...Array(18).keys()].map((k) => median(byIdx.map((c) => c.features[k]))) : [];
    function renderE() {
      if (!has) return;
      pickEl.value = st.sel;
      const svg = $("fp"); svg.innerHTML = "";
      const f = byIdx[st.sel].features, diff = f.map((v, k) => v - typical[k]);
      const lim = Math.max(1, Math.ceil(Math.max(...diff.map(Math.abs)) * 2) / 2);
      const L = 230, R = 960, mid = (L + R) / 2, x = (v) => mid + (v / lim) * ((R - L) / 2);
      let yy = 34; const rows = [];
      FEATS.forEach(([g, list]) => { rows.push({ head: g, y: yy }); yy += 22; list.forEach(([k, name]) => { rows.push({ k, name, y: yy }); yy += 25; }); yy += 10; });
      svg.setAttribute("viewBox", `0 0 1000 ${yy + 30}`);
      [-lim, -lim / 2, 0, lim / 2, lim].forEach((t) => {
        s("line", { x1: x(t), x2: x(t), y1: 20, y2: yy, stroke: "var(--line)", "stroke-width": 1 }, svg);
        s("text", { x: x(t), y: yy + 18, "text-anchor": "middle", class: "ax" }, svg).textContent = (t > 0 ? "+" : "") + (+t.toFixed(2));
      });
      s("line", { x1: mid, x2: mid, y1: 20, y2: yy, stroke: "var(--mute)", "stroke-width": 1 }, svg);
      const big = diff.map((v, k) => [Math.abs(v), k]).sort((a, b) => b[0] - a[0]).slice(0, 2).map((d) => d[1]);
      rows.forEach((r) => {
        if (r.head) { s("text", { x: 0, y: r.y, class: "ax" }, svg).textContent = r.head.toUpperCase(); return; }
        const v = diff[r.k], w = Math.abs(x(v) - mid);
        s("text", { x: 0, y: r.y + 4 }, svg).textContent = r.name;
        if (w > 0.5) {
          const h = 14, rr = Math.min(4, w), y0 = r.y - h / 2, e = v >= 0 ? mid + w : mid - w, sg = v >= 0 ? 1 : -1;
          const d = `M${mid},${y0}H${e - sg * rr}Q${e},${y0} ${e},${y0 + rr}V${y0 + h - rr}Q${e},${y0 + h} ${e - sg * rr},${y0 + h}H${mid}Z`;
          const p = s("path", { d, fill: v >= 0 ? "var(--hot)" : "var(--ice)" }, svg);
          p.addEventListener("mousemove", (ev) => showTip(ev, `<b>${esc(r.name)}</b>${esc(names[st.sel])}: ${f[r.k].toFixed(2)} · typical ${typical[r.k].toFixed(2)} · difference ${(v >= 0 ? "+" : "") + v.toFixed(2)}`));
          p.addEventListener("mouseleave", hideTip);
        }
        if (big.includes(r.k) && Math.abs(v) >= 0.01) s("text", { x: v >= 0 ? x(v) + 8 : x(v) - 8, y: r.y + 4, "text-anchor": v >= 0 ? "start" : "end" }, svg).textContent = (v >= 0 ? "+" : "") + v.toFixed(2);
      });
      s("text", { x: x(-lim), y: 12, class: "ax" }, svg).textContent = "← lower than typical";
      s("text", { x: x(lim), y: 12, "text-anchor": "end", class: "ax" }, svg).textContent = "higher than typical →";
      $("fpTbl").innerHTML = `<thead><tr><th>Feature</th><th>${esc(names[st.sel])}</th><th>Typical electrode</th><th>Difference</th></tr></thead><tbody>` +
        FEATS.flatMap(([, l]) => l).map(([k, n]) => `<tr><td>${n}</td><td>${f[k].toFixed(3)}</td><td>${typical[k].toFixed(3)}</td><td>${(diff[k] >= 0 ? "+" : "") + diff[k].toFixed(3)}</td></tr>`).join("") + "</tbody>";
    }

    // ── C ────────────────────────────────────────────────────────────────
    const net = has ? makeNet() : null;
    function makeNet() {
      const svg = $("net"); svg.innerHTML = "";
      const Wd = 640, Ht = 520, CX = 320, CY = 260, NT = byIdx[0].timeline.length, WL = result.graph.windowLinks;
      const tAll = byIdx.flatMap((c) => c.timeline), tmin = Math.min(...tAll), tmax = Math.max(...tAll);
      const heatT = (v) => heat((v - tmin) / (tmax - tmin || 1));
      const scoreAt = (i, t) => { const a = Math.floor(t), b = Math.min(NT - 1, a + 1); return lerp(byIdx[i].timeline[a], byIdx[i].timeline[b], t - a); };
      const starts = result.info.window_starts_sec;
      const fmt = (k) => { const v = starts[k]; return `${Math.floor(v / 60)}:${String(Math.round(v % 60)).padStart(2, "0")}`; };
      const gL = s("g", {}, svg), gP = s("g", {}, svg), gN = s("g", {}, svg);
      let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
      const P = names.map((_, i) => ({ x: CX + 190 * Math.cos((i / N) * 6.283) + rnd() * 10, y: CY + 190 * Math.sin((i / N) * 6.283) + rnd() * 10, vx: 0, vy: 0 }));
      let links = WL[0], linkEls = [], linkK = -1, t = 0, playing = false, drag = -1, downAt = null, last = 0, raf = 0, pulses = [], visible = true;
      let sparkDot = null, sx = null, sy = null;

      function step() {
        const F = P.map(() => [0, 0]);
        for (let a = 0; a < N; a++) for (let b = a + 1; b < N; b++) {
          const dx = P[a].x - P[b].x, dy = P[a].y - P[b].y, d2 = dx * dx + dy * dy + 0.01;
          if (d2 > 8100) continue; // repel only nearby electrodes
          const f = 160 / d2;
          F[a][0] += dx * f; F[a][1] += dy * f; F[b][0] -= dx * f; F[b][1] -= dy * f;
        }
        links.forEach(([a, b, w]) => {
          const dx = P[b].x - P[a].x, dy = P[b].y - P[a].y, d = Math.hypot(dx, dy) || 1, f = (d - 36) * 0.08 * w;
          F[a][0] += (dx / d) * f; F[a][1] += (dy / d) * f; F[b][0] -= (dx / d) * f; F[b][1] -= (dy / d) * f;
        });
        let e = 0;
        P.forEach((p, i) => {
          if (i === drag) { p.vx = p.vy = 0; return; }
          p.vx = (p.vx + F[i][0] + (CX - p.x) * 0.01) * 0.82; p.vy = (p.vy + F[i][1] + (CY - p.y) * 0.01) * 0.82;
          const sp = Math.hypot(p.vx, p.vy); if (sp > 6) { p.vx *= 6 / sp; p.vy *= 6 / sp; }
          p.x = Math.max(22, Math.min(Wd - 22, p.x + p.vx)); p.y = Math.max(22, Math.min(Ht - 22, p.y + p.vy)); e += sp;
        });
        return e / N;
      }
      for (let i = 0; i < 350; i++) step();

      const nodes = names.map((n, i) => {
        const g = s("g", { class: "nd", tabindex: 0, role: "button" }, gN);
        const c = s("circle", { stroke: "var(--panel)", "stroke-width": 2 }, g);
        const tx = s("text", { x: 12, y: 4 }, g); tx.textContent = n;
        g.addEventListener("pointerdown", (e) => { drag = i; downAt = [e.clientX, e.clientY]; svg.setPointerCapture(e.pointerId); svg.classList.add("dragging"); hideTip(); });
        g.addEventListener("mousemove", (e) => { if (drag < 0) showTip(e, `<b>${esc(n)}</b>${pct(scoreAt(i, t))} this minute · average ${pct(byIdx[i].prob)}`); });
        g.addEventListener("mouseleave", hideTip);
        g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(i); } });
        return { g, c, tx };
      });
      const toSvg = (e) => { const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; return pt.matrixTransform(svg.getScreenCTM().inverse()); };
      svg.addEventListener("pointermove", (e) => { if (drag < 0) return; const q = toSvg(e); P[drag].x = Math.max(22, Math.min(Wd - 22, q.x)); P[drag].y = Math.max(22, Math.min(Ht - 22, q.y)); wake(); }, sig);
      const endDrag = (e) => {
        if (drag < 0) return;
        const moved = downAt ? Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) : 99, i = drag;
        drag = -1; svg.classList.remove("dragging");
        if (moved < 5) select(i);
        wake();
      };
      svg.addEventListener("pointerup", endDrag, sig); svg.addEventListener("pointercancel", endDrag, sig);

      const play = $("netPlay"), scrub = $("netScrub");
      scrub.max = Math.max(0, NT - 1); scrub.value = 0; scrub.disabled = NT < 2; play.disabled = NT < 2;
      const setBtn = () => { const end = t >= NT - 1; play.innerHTML = playing ? ICON.pause : end && NT > 1 ? ICON.replay : ICON.play; play.setAttribute("aria-label", playing ? "Pause" : end ? "Replay" : "Play through the recording"); };
      play.addEventListener("click", () => { if (playing) playing = false; else { if (t >= NT - 1) t = 0; playing = true; } setBtn(); wake(); }, sig);
      scrub.addEventListener("input", () => { t = +scrub.value; playing = false; setBtn(); wake(); }, sig);

      function setLinks(k) {
        linkK = k; links = WL[k]; gL.innerHTML = ""; gP.innerHTML = "";
        linkEls = links.map(([a, b, w]) => ({ a, b, w, el: s("line", { "stroke-linecap": "round" }, gL) }));
        pulses = reduce ? [] : linkEls.filter((l) => l.a === st.sel || l.b === st.sel)
          .flatMap((l, j) => [0, 0.5].map((ph) => ({ from: l.a === st.sel ? l.b : l.a, w: l.w, ph: ph + j * 0.13, el: s("circle", { r: 3, fill: "var(--fg)" }, gP) })));
        side();
      }
      function side() {
        const k = Math.round(t), sel = st.sel;
        $("netWin").textContent = `Minute-window ${k + 1} of ${NT} · from ${fmt(k)}`;
        const sc = byIdx.map((c) => c.timeline[k]), best = sc.indexOf(Math.max(...sc));
        $("netTop").textContent = `Highest this minute: ${names[best]} · ${pct(sc[best])}`;
        $("netSel").textContent = `${names[sel]} · ${pct(byIdx[sel].timeline[k])} this minute · rank #${rankOf[names[sel]]} overall`;
        const mine = linkEls.filter((l) => l.a === sel || l.b === sel).map((l) => [l.a === sel ? l.b : l.a, l.w]).sort((p, q) => q[1] - p[1]);
        $("netLinks").innerHTML = mine.map(([j, w]) => `<li><span>${esc(names[j])} · ${pct(byIdx[j].timeline[k])}</span><span>r ${w.toFixed(2)}</span></li>`).join("");
        $("netStats").textContent = `In the graph the model used, ${result.graph.connectedPairs.toLocaleString()} of ${result.graph.totalPairs.toLocaleString()} electrode pairs were connected (|r| > 0.3) in most minutes. Only each electrode's 3 strongest links are drawn.`;
        const sp = $("netSpark"); sp.innerHTML = "";
        const tl = byIdx[sel].timeline; sx = (q) => 8 + (q / Math.max(1, NT - 1)) * 284; sy = (v) => 52 - ((v - tmin) / (tmax - tmin || 1)) * 44;
        if (st.thr >= tmin && st.thr <= tmax) s("line", { x1: 8, x2: 292, y1: sy(st.thr), y2: sy(st.thr), stroke: "var(--line)", "stroke-width": 1 }, sp);
        s("polyline", { points: tl.map((v, q) => `${sx(q)},${sy(v)}`).join(" "), fill: "none", stroke: "var(--hot)", "stroke-width": 2, "stroke-linejoin": "round" }, sp);
        sparkDot = s("circle", { r: 4, fill: "var(--hot)", stroke: "var(--well)", "stroke-width": 2 }, sp);
      }
      function wake() { if (!raf && visible) raf = requestAnimationFrame(frame); }
      function frame(now) {
        raf = 0;
        const dt = last ? Math.min(64, now - last) : 16; last = now;
        if (playing) { t += dt / 2400; if (t >= NT - 1) { t = NT - 1; playing = false; setBtn(); } scrub.value = t; }
        const k = Math.round(t);
        if (k !== linkK) setLinks(k);
        $("netClock").textContent = NT > 1 ? `${fmt(Math.floor(t))} – ${fmt(Math.min(NT - 1, Math.floor(t) + 1))}` : fmt(0);
        const energy = step(), sel = st.sel;
        const top5 = new Set(names.map((_, i) => [scoreAt(i, t), i]).sort((a, b) => b[0] - a[0]).slice(0, 5).map((q) => q[1]));
        linkEls.forEach((l) => {
          const on = l.a === sel || l.b === sel;
          l.el.setAttribute("x1", P[l.a].x); l.el.setAttribute("y1", P[l.a].y); l.el.setAttribute("x2", P[l.b].x); l.el.setAttribute("y2", P[l.b].y);
          l.el.setAttribute("stroke", on ? "var(--fg-2)" : "var(--line)"); l.el.setAttribute("stroke-width", on ? 1.6 : 1);
          l.el.setAttribute("opacity", on ? 0.85 : (0.2 + 0.6 * l.w).toFixed(2));
        });
        pulses.forEach((p) => {
          const u = (now / 1500 + p.ph) % 1, a = P[p.from], b = P[sel];
          p.el.setAttribute("cx", lerp(a.x, b.x, u)); p.el.setAttribute("cy", lerp(a.y, b.y, u));
          p.el.setAttribute("opacity", (Math.sin(u * Math.PI) * (0.25 + 0.75 * p.w * p.w)).toFixed(2));
        });
        nodes.forEach(({ g, c, tx }, i) => {
          const v = scoreAt(i, t), isSel = i === sel;
          g.setAttribute("transform", `translate(${P[i].x.toFixed(1)},${P[i].y.toFixed(1)})`);
          c.setAttribute("fill", heatT(v));
          c.setAttribute("r", isSel ? 10 + (reduce ? 0 : 1.2 * Math.sin(now / 260)) : v >= st.thr ? 7 : 5.5);
          c.setAttribute("stroke", isSel ? "var(--fg)" : "var(--panel)");
          tx.style.display = top5.has(i) || isSel ? "" : "none";
          g.setAttribute("aria-label", `${names[i]}, ${pct(v)} this minute`);
        });
        if (sparkDot) { sparkDot.setAttribute("cx", sx(t)); sparkDot.setAttribute("cy", sy(scoreAt(sel, t))); }
        if (playing || drag >= 0 || pulses.length || energy > 0.02) wake(); else last = 0;
      }
      // pause the loop while the network is off screen
      const io = "IntersectionObserver" in window ? new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible) wake(); else { cancelAnimationFrame(raf); raf = 0; last = 0; } }) : null;
      if (io) io.observe(svg);
      setBtn();
      return {
        refresh() { setLinks(Math.round(t)); wake(); },
        destroy() { cancelAnimationFrame(raf); raf = 0; if (io) io.disconnect(); playing = false; },
      };
    }

    function renderAll() { renderA(); renderE(); if (net) net.refresh(); }
    renderAll();
    return {
      update(o) {
        if (o.threshold != null) st.thr = o.threshold;
        if (o.selected != null && idxOf[o.selected] != null) st.sel = idxOf[o.selected];
        renderAll();
      },
      destroy() { ac.abort(); if (net) net.destroy(); hideTip(); },
    };
  }

  window.EEGInsights = { show };
})();
