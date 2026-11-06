(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const fmt = (v) => v == null || Number.isNaN(v) ? "–" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
  const short = (m) => (m.includes("/") ? m.split("/").slice(1).join("/") : m);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ============================================================ colour
  // blue ramp: pale = strongly oppose, deep = strongly support
  const BLUES = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"];
  const blue = d3.scaleLinear().domain(d3.range(BLUES.length).map((i) => -2 + (4 * i) / (BLUES.length - 1))).range(BLUES).interpolate(d3.interpolateLab).clamp(true);
  const opinionColor = (v) => (v == null ? "#d9dde4" : blue(v));
  const SHAPES = [d3.symbolCircle, d3.symbolSquare, d3.symbolTriangle, d3.symbolDiamond, d3.symbolStar, d3.symbolWye, d3.symbolCross];

  // ============================================================ tooltip
  const tip = $("tooltip");
  function showTip(evt, rows) {
    tip.replaceChildren();
    rows.forEach(([value, label, color]) => {
      const r = el("div", "row");
      if (color) { const k = el("span", "key"); k.style.background = color; r.appendChild(k); }
      r.appendChild(el("span", "tv", value));
      if (label) r.appendChild(el("span", "tl", label));
      tip.appendChild(r);
    });
    tip.style.display = "block";
    tip.style.left = Math.min(evt.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = Math.min(evt.clientY + 14, innerHeight - tip.offsetHeight - 8) + "px";
  }
  const hideTip = () => { tip.style.display = "none"; };

  // ============================================================ viewer state
  let DATA = null, runId = null, t = 0, playing = false, speed = 1.2, selected = null, followLive = false;
  let lastFrame = null, lastStepDrawn = -1;

  const W = 900, H = 600;
  const netSvg = d3.select("#net").append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img").attr("aria-label", "Agent network");
  const guide = netSvg.append("g");
  const gEdges = netSvg.append("g"), gInteractions = netSvg.append("g"), gNodes = netSvg.append("g");
  let nodes = [], links = [], nodeSel = null, sim = null;

  const D = () => (DATA ? DATA.opinions.length - 1 : 0);
  function opinionAt(i, time) {
    const d0 = Math.floor(time), d1 = Math.min(D(), d0 + 1), f = time - d0;
    const a = DATA.opinions[d0][i], b = DATA.opinions[d1][i];
    if (a == null) return b; if (b == null) return a;
    return a + (b - a) * f;
  }
  const edgesAt = () => DATA.edges; // the network is static for now (no rewiring)
  const xTarget = (v) => W / 2 + (v ?? 0) * (W * 0.19);

  // draw the opinion guide once
  function drawGuide() {
    guide.selectAll("*").remove();
    const labels = [[-2, "strongly oppose"], [-1, "oppose"], [0, "neutral"], [1, "support"], [2, "strongly support"]];
    labels.forEach(([v, text]) => {
      guide.append("line").attr("x1", xTarget(v)).attr("x2", xTarget(v)).attr("y1", 28).attr("y2", H - 8)
        .attr("stroke", css("--line")).attr("stroke-dasharray", v === 0 ? null : "2 5");
      guide.append("text").attr("x", xTarget(v)).attr("y", 16).attr("text-anchor", "middle")
        .attr("fill", css("--muted")).attr("font-size", 12).text(text);
    });
  }

  function setupNetwork() {
    const n = DATA.agents.length;
    const deg = new Array(n).fill(0);
    edgesAt(0).forEach(([u, v]) => { deg[u]++; deg[v]++; });
    const size = d3.scaleSqrt().domain([0, d3.max(deg) || 1]).range([90, 380]);
    const prev = new Map(nodes.map((d) => [d.id, d]));
    nodes = DATA.agents.map((a) => {
      const old = prev.get(a.id);
      const [px, py] = DATA.positions[a.id];
      return { id: a.id, model: a.model, area: size(deg[a.id]), x: old ? old.x : xTarget(DATA.opinions[0][a.id]), y: old ? old.y : 40 + py * (H - 80), vx: 0, vy: 0 };
    });
    const models = DATA.models;
    const shapeOf = (m) => SHAPES[models.indexOf(m) % SHAPES.length];

    nodeSel = gNodes.selectAll("g.node").data(nodes, (d) => d.id).join((enter) => {
      const g = enter.append("g").attr("class", "node").attr("tabindex", 0).style("cursor", "pointer");
      g.append("circle").attr("class", "hit").attr("r", 13).attr("fill", "transparent");
      g.append("path").attr("class", "mark");
      return g;
    });
    nodeSel.select("path.mark").attr("d", (d) => d3.symbol(shapeOf(d.model), d.area)()).attr("stroke-width", 1.5);
    nodeSel
      .on("pointermove", (evt, d) => {
        const p = DATA.agent_steps[Math.round(t)][d.id];
        showTip(evt, [[fmt(opinionAt(d.id, t)), `Agent ${d.id} · ${short(d.model)}`], ...(p && p.interactions.length ? [[String(p.interactions.length), "interactions this step"]] : [])]);
      })
      .on("pointerleave", hideTip)
      .on("click", (evt, d) => select(d.id))
      .on("keydown", (evt, d) => { if (evt.key === "Enter") select(d.id); })
      .call(d3.drag()
        .on("start", (evt, d) => { if (!evt.active) sim.alphaTarget(0.2).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (evt, d) => { d.fx = evt.x; d.fy = evt.y; })
        .on("end", (evt, d) => { if (!evt.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));

    if (sim) sim.stop();
    sim = d3.forceSimulation(nodes)
      .alphaDecay(0.03).velocityDecay(0.35)
      .force("link", d3.forceLink([]).id((d) => d.id).distance(30).strength(0.05))
      .force("charge", d3.forceManyBody().strength(-30).distanceMax(160))
      .force("collide", d3.forceCollide((d) => Math.sqrt(d.area) / 1.6 + 2))
      .force("x", d3.forceX((d) => xTarget(opinionAt(d.id, t))).strength(0.22))
      .force("y", d3.forceY(H / 2 + 10).strength(0.035))
      .on("tick", drawPositions);
    lastStepDrawn = -1;
    updateLinks();

    // legend
    const leg = $("net-legend"); leg.replaceChildren();
    const ramp = el("span", "item");
    const rs = d3.select(ramp).append("svg").attr("width", 180).attr("height", 24);
    const grad = rs.append("defs").append("linearGradient").attr("id", "ramp");
    BLUES.forEach((c, i) => grad.append("stop").attr("offset", i / (BLUES.length - 1)).attr("stop-color", c));
    rs.append("rect").attr("x", 0).attr("y", 2).attr("width", 180).attr("height", 8).attr("rx", 4).attr("fill", "url(#ramp)").attr("stroke", css("--line-2"));
    rs.append("text").attr("x", 0).attr("y", 22).attr("fill", css("--muted")).attr("font-size", 11).text("oppose −2");
    rs.append("text").attr("x", 180).attr("y", 22).attr("text-anchor", "end").attr("fill", css("--muted")).attr("font-size", 11).text("+2 support");
    leg.appendChild(ramp);
    models.forEach((m) => {
      const it = el("span", "item");
      d3.select(it).append("svg").attr("width", 14).attr("height", 14).append("path").attr("transform", "translate(7,7)")
        .attr("d", d3.symbol(shapeOf(m), 90)()).attr("fill", css("--ink-2"));
      it.appendChild(el("span", null, short(m)));
      leg.appendChild(it);
    });
    leg.appendChild(el("span", "hint", "Horizontal position follows opinion; links pull neighbours together. Drag nodes to move them."));
  }

  function updateLinks() {
    const step = Math.floor(t);
    if (step === lastStepDrawn) return;
    lastStepDrawn = step;
    links = edgesAt(step).map(([s, tt]) => ({ source: s, target: tt, key: s < tt ? `${s}-${tt}` : `${tt}-${s}` }));
    sim.force("link").links(links);
    gEdges.selectAll("line").data(links, (d) => d.key).join(
      (enter) => enter.append("line").attr("stroke", css("--edge")).attr("stroke-width", 1).attr("stroke-opacity", 0)
        .call((e) => e.transition().duration(reduceMotion ? 0 : 500).attr("stroke-opacity", 0.8)),
      (update) => update,
      (exit) => exit.transition().duration(reduceMotion ? 0 : 400).attr("stroke-opacity", 0).remove()
    );
    sim.alpha(Math.max(sim.alpha(), 0.25)).restart();
  }

  function drawPositions() {
    gEdges.selectAll("line").attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y).attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
    nodeSel.attr("transform", (d) => `translate(${d.x},${d.y})`);
    if (selected != null) {
      const p = DATA.agent_steps[Math.round(t)][selected];
      const me = nodes[selected];
      gInteractions.selectAll("line").data(p ? p.interactions : [], (j) => j).join("line")
        .attr("stroke", css("--ink")).attr("stroke-width", 1.8).attr("stroke-linecap", "round")
        .attr("x1", (j) => nodes[j].x).attr("y1", (j) => nodes[j].y).attr("x2", me.x).attr("y2", me.y);
    } else gInteractions.selectAll("line").remove();
  }

  function paintNodes() {
    const step = Math.round(t);
    const p = selected != null ? DATA.agent_steps[step][selected] : null;
    const partners = new Set(p ? p.interactions : []);
    nodeSel.select("path.mark")
      .attr("fill", (d) => opinionColor(opinionAt(d.id, t)))
      .attr("stroke", (d) => (d.id === selected ? css("--ink") : partners.has(d.id) ? css("--ink-2") : css("--node-ring")))
      .attr("stroke-width", (d) => (d.id === selected ? 2.5 : partners.has(d.id) ? 2 : 1.2));
    gEdges.selectAll("line")
      .attr("stroke", (d) => selected != null && (d.source.id === selected || d.target.id === selected) ? css("--ink-2") : css("--edge"));
    nodeSel.attr("aria-label", (d) => `Agent ${d.id}, ${short(d.model)}, opinion ${fmt(opinionAt(d.id, t))}`);
  }

  // ============================================================ line charts
  const CW = 380, CH = 170, cm = { t: 10, r: 14, b: 24, l: 38 };
  let cursors = [];
  function lineChart(sel, series, yDomain, label, endLabels) {
    const box = d3.select(sel); box.selectAll("*").remove();
    const svg = box.append("svg").attr("viewBox", `0 0 ${CW} ${CH}`).attr("role", "img").attr("aria-label", label);
    const right = endLabels ? 70 : cm.r;
    const x = d3.scaleLinear().domain([0, Math.max(1, D())]).range([cm.l, CW - right]);
    const y = d3.scaleLinear().domain(yDomain).nice().range([CH - cm.b, cm.t]);
    svg.append("g").attr("class", "gridline").attr("transform", `translate(${cm.l},0)`).call(d3.axisLeft(y).ticks(4).tickSize(-(CW - cm.l - right)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${CH - cm.b})`).call(d3.axisBottom(x).ticks(Math.min(D(), 6)).tickFormat(d3.format("d")).tickSizeOuter(0));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${cm.l},0)`).call(d3.axisLeft(y).ticks(4).tickSizeOuter(0)).call((g) => g.select(".domain").remove());
    const line = d3.line().defined((d) => d[1] != null).x((d) => x(d[0])).y((d) => y(d[1])).curve(d3.curveMonotoneX);
    series.forEach((s) => {
      svg.append("path").datum(s.values).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2).attr("stroke-linejoin", "round").attr("stroke-linecap", "round").attr("d", line);
      if (endLabels) {
        const last = [...s.values].reverse().find((d) => d[1] != null);
        if (last) svg.append("text").attr("x", x(last[0]) + 6).attr("y", y(last[1]) + 4).attr("font-size", 11).attr("fill", css("--ink-2")).text(s.name);
      }
    });
    const cursor = svg.append("line").attr("y1", cm.t).attr("y2", CH - cm.b).attr("stroke", css("--accent")).attr("stroke-width", 1.5);
    const dot = svg.append("g");
    const hover = svg.append("line").attr("y1", cm.t).attr("y2", CH - cm.b).attr("stroke", css("--muted")).attr("stroke-dasharray", "2 3").style("display", "none");
    cursors.push({ cursor, dot, series, x, y });
    svg.append("rect").attr("x", cm.l).attr("y", 0).attr("width", CW - cm.l - right).attr("height", CH).attr("fill", "transparent").style("cursor", "pointer")
      .on("pointermove", (evt) => {
        const [mx] = d3.pointer(evt); const d = Math.max(0, Math.min(D(), Math.round(x.invert(mx))));
        hover.style("display", null).attr("x1", x(d)).attr("x2", x(d));
        showTip(evt, [[`Step ${d}`, ""], ...series.map((s) => [fmt(s.values[d] ? s.values[d][1] : null), s.name, s.color])]);
      })
      .on("pointerleave", () => { hover.style("display", "none"); hideTip(); })
      .on("click", (evt) => { const [mx] = d3.pointer(evt); stop(); followLive = false; setTime(Math.round(x.invert(mx))); });
  }
  function moveCursors() {
    cursors.forEach(({ cursor, dot, series, x, y }) => {
      cursor.attr("x1", x(t)).attr("x2", x(t));
      dot.selectAll("circle").data(series).join("circle").attr("r", 4).attr("fill", (s) => s.color).attr("stroke", css("--surface")).attr("stroke-width", 2)
        .attr("cx", x(t)).attr("cy", (s) => {
          const d0 = Math.floor(t), d1 = Math.min(D(), d0 + 1), f = t - d0;
          const a = s.values[d0] && s.values[d0][1], b = s.values[d1] && s.values[d1][1];
          const v = a == null ? b : b == null ? a : a + (b - a) * f;
          return v == null ? -99 : y(v);
        });
    });
  }

  function buildCharts() {
    cursors = [];
    const ms = (key) => DATA.metrics.map((r) => [r.step, r[key]]);
    const mx = (key) => d3.max(DATA.metrics, (r) => r[key]) || 1;
    const ink = css("--accent");
    lineChart("#c-pol", [{ name: "polarization", color: ink, values: ms("polarization") }], [0, mx("polarization")], "Polarization over time");
    lineChart("#c-dis", [{ name: "disagreement", color: ink, values: ms("global_disagreement") }], [0, mx("global_disagreement")], "Disagreement over time");
    lineChart("#c-nci", [{ name: "neighbour correlation", color: ink, values: ms("neighbor_correlation") }], [-1, 1], "Neighbour correlation over time");
    const models = DATA.models;
    let series;
    if (models.length > 1) {
      $("means-title").textContent = "Mean opinion by model";
      series = models.map((m, i) => ({ name: short(m), color: css(`--s${(i % 8) + 1}`), values: ms(`mean_opinion[${m}]`) }));
    } else {
      $("means-title").textContent = "Mean opinion";
      series = [{ name: "mean", color: ink, values: ms("mean_opinion") }];
    }
    lineChart("#c-means", series, [-2, 2], "Mean opinion over time", models.length > 1);
    const ml = $("means-legend"); ml.replaceChildren();
    if (models.length > 1) series.forEach((s) => {
      const it = el("span", "item"); const k = el("span"); k.style.cssText = `display:inline-block;width:14px;height:2px;border-radius:1px;background:${s.color}`;
      it.append(k, el("span", null, s.name)); ml.appendChild(it);
    });
    buildHist();
  }

  // ============================================================ histogram
  const HW = 380, HH = 170, hm = { t: 10, r: 8, b: 24, l: 30 };
  const bins = d3.range(-2, 2, 0.5);
  let hsvg, hBars, hy;
  function buildHist() {
    const box = d3.select("#c-hist"); box.selectAll("*").remove();
    hsvg = box.append("svg").attr("viewBox", `0 0 ${HW} ${HH}`).attr("role", "img").attr("aria-label", "Opinion distribution");
    const hx = d3.scaleBand().domain(bins).range([hm.l, HW - hm.r]).paddingInner(0.1);
    let mxc = 1;
    for (let d = 0; d <= D(); d++) mxc = Math.max(mxc, d3.max(counts(d)));
    hy = d3.scaleLinear().domain([0, mxc]).nice().range([HH - hm.b, hm.t]);
    hsvg.append("g").attr("class", "gridline").attr("transform", `translate(${hm.l},0)`).call(d3.axisLeft(hy).ticks(4).tickSize(-(HW - hm.l - hm.r)).tickFormat(""));
    hsvg.append("g").attr("class", "axis").attr("transform", `translate(${hm.l},0)`).call(d3.axisLeft(hy).ticks(4).tickFormat(d3.format("d")).tickSizeOuter(0)).call((g) => g.select(".domain").remove());
    hsvg.append("g").attr("class", "axis").attr("transform", `translate(0,${HH - hm.b})`).call(d3.axisBottom(hx).tickFormat((b) => (b >= 0 ? "+" : "−") + Math.abs(b).toFixed(1)).tickSizeOuter(0));
    hBars = hsvg.append("g").selectAll("path").data(bins).join("path").attr("fill", (b) => blue(b + 0.25)).attr("stroke", css("--node-ring")).attr("stroke-width", 0.8)
      .datum((b) => ({ b, x: hx(b), w: hx.bandwidth(), c: 0 }))
      .on("pointermove", (evt, d) => showTip(evt, [[String(Math.round(d.c)), `agents between ${fmt(d.b)} and ${fmt(d.b + 0.5)}`]]))
      .on("pointerleave", hideTip);
  }
  function counts(time) {
    const c = new Array(bins.length).fill(0);
    for (let i = 0; i < DATA.agents.length; i++) {
      const v = opinionAt(i, time); if (v == null) continue;
      c[Math.max(0, Math.min(bins.length - 1, Math.floor((v + 2) / 0.5)))]++;
    }
    return c;
  }
  function drawHist() {
    const c = counts(t);
    hBars.each((d, i) => { d.c = c[i]; }).attr("d", (d) => {
      const h = hy(0) - hy(d.c), y0 = hy(d.c), r = Math.min(4, d.w / 2, h);
      return h <= 0 ? "" : `M${d.x},${y0 + h}V${y0 + r}Q${d.x},${y0} ${d.x + r},${y0}H${d.x + d.w - r}Q${d.x + d.w},${y0} ${d.x + d.w},${y0 + r}V${y0 + h}Z`;
    });
  }

  // ============================================================ agent panel & table
  function select(id) { selected = selected === id ? null : id; lastPanel = ""; paintNodes(); drawPositions(); drawPanel(); }
  let lastPanel = "";
  function drawPanel() {
    const step = Math.round(t);
    const key = `${selected}|${step}|${D()}`;
    if (key === lastPanel) return; lastPanel = key;
    const box = $("agent"); box.replaceChildren();
    if (selected == null) {
      box.appendChild(el("p", "hint", "Click a node to follow one agent: whom it interacted with at each step, how it reasoned, and the message it shared."));
      return;
    }
    const a = DATA.agents[selected];
    box.appendChild(el("div", "agent-head", `Agent ${a.id}`));
    box.appendChild(el("div", "agent-meta", `Model: ${short(a.model)}`));

    const SW = 320, SH = 96, sm = { t: 8, r: 8, b: 18, l: 26 };
    const sx = d3.scaleLinear().domain([0, Math.max(1, D())]).range([sm.l, SW - sm.r]);
    const sy = d3.scaleLinear().domain([-2, 2]).range([SH - sm.b, sm.t]);
    const s = d3.select(box).append("svg").attr("viewBox", `0 0 ${SW} ${SH}`).style("width", "100%").style("margin-top", "10px");
    s.append("g").attr("class", "gridline").attr("transform", `translate(${sm.l},0)`).call(d3.axisLeft(sy).tickValues([-2, 0, 2]).tickSize(-(SW - sm.l - sm.r)).tickFormat(""));
    s.append("g").attr("class", "axis").attr("transform", `translate(${sm.l},0)`).call(d3.axisLeft(sy).tickValues([-2, 0, 2]).tickFormat(d3.format("+d"))).call((g) => g.select(".domain").remove());
    s.append("g").attr("class", "axis").attr("transform", `translate(0,${SH - sm.b})`).call(d3.axisBottom(sx).ticks(Math.min(D(), 5)).tickFormat(d3.format("d")).tickSizeOuter(0));
    const vals = DATA.opinions.map((row, d) => [d, row[a.id]]);
    s.append("path").datum(vals).attr("fill", "none").attr("stroke", css("--accent")).attr("stroke-width", 2)
      .attr("d", d3.line().defined((d) => d[1] != null).x((d) => sx(d[0])).y((d) => sy(d[1])).curve(d3.curveMonotoneX));
    s.append("circle").attr("cx", sx(step)).attr("cy", sy(DATA.opinions[step][a.id] ?? 0)).attr("r", 5).attr("fill", opinionColor(DATA.opinions[step][a.id])).attr("stroke", css("--ink")).attr("stroke-width", 1.5);

    const kv = el("div", "kv"); kv.append(el("span", "hint", `Step ${step}`), el("b", null, fmt(DATA.opinions[step][a.id])));
    if (step > 0) {
      const prev = DATA.opinions[step - 1][a.id], cur = DATA.opinions[step][a.id];
      if (prev != null && cur != null) kv.append(el("span", "hint", `${cur - prev >= 0 ? "▲" : "▼"} ${Math.abs(cur - prev).toFixed(2)} since step ${step - 1}`));
    }
    box.appendChild(kv);
    const p = DATA.agent_steps[step][a.id];
    if (!p) return;
    // a separate scorer's rating of this step's text, next to the agent's own (scorer.py)
    if (p.scores) {
      const sc = el("div", "scores");
      sc.appendChild(el("div", "label", "Self-rating and scorer"));
      const row = (name, value, title) => { const r = el("div", "kv"); r.append(el("span", "hint", name), el("b", null, value)); if (title) r.title = title; sc.appendChild(r); };
      row("Self-rating", step === 0 ? "— (given)" : fmt(DATA.opinions[step][a.id]));
      Object.entries(p.scores).forEach(([label, v]) => {
        const [model, field] = label.split(" · ");
        row(`Scorer: ${short(model)} (${field})`, typeof v === "number" ? (v > 0 ? "+" : "") + v : v, label);
      });
      box.appendChild(sc);
    }
    if (p.interactions.length) {
      box.appendChild(el("div", "label", "Interacted with"));
      const f = el("div", "interactions");
      p.interactions.forEach((j) => {
        const b = el("button", null, `Agent ${j}  ${fmt(step > 0 ? DATA.opinions[step - 1][j] : null)}`);
        b.type = "button";
        const prev = DATA.agent_steps[step - 1] && DATA.agent_steps[step - 1][j];
        b.title = prev ? prev.message : ""; // the message j shared at the previous step
        b.addEventListener("click", () => select(j));
        f.appendChild(b);
      });
      box.appendChild(f);
    }
    if (p.reasoning) { box.appendChild(el("div", "label", "Reasoning")); box.appendChild(el("p", "reasoning", p.reasoning)); }
    if (p.message) { box.appendChild(el("div", "label", "Message")); box.appendChild(el("p", "message", p.message)); }
    if (p.error) box.appendChild(el("div", "agent-error", "Model reply could not be used: " + p.error));
  }

  let lastTable = "";
  function drawTable() {
    const step = Math.round(t), key = `${step}|${D()}`;
    if (key === lastTable) return; lastTable = key;
    const deg = new Array(DATA.agents.length).fill(0);
    edgesAt(step).forEach(([u, v]) => { deg[u]++; deg[v]++; });
    const tb = $("tbody"); tb.replaceChildren();
    DATA.agents.forEach((a) => {
      const tr = el("tr"); const p = DATA.agent_steps[step][a.id];
      [[a.id, "num"], [short(a.model)], [fmt(DATA.opinions[step][a.id]), "num"], [deg[a.id], "num"], [p ? p.message : ""]]
        .forEach(([v, c]) => tr.appendChild(el("td", c || null, String(v))));
      tr.style.cursor = "pointer"; tr.addEventListener("click", () => select(a.id));
      tb.appendChild(tr);
    });
  }

  // ============================================================ time & playback
  const scrub = $("scrub");
  function setTime(v) {
    if (!DATA || !sim) return; // a live run with no finished step yet
    t = Math.max(0, Math.min(D(), v));
    scrub.value = D() ? t / D() : 0;
    $("step").textContent = String(Math.round(t));
    updateLinks();
    sim.force("x").x((d) => xTarget(opinionAt(d.id, t)));
    if (sim.alpha() < 0.08) sim.alpha(0.08).restart();
    paintNodes(); drawHist(); moveCursors(); drawPanel(); drawTable();
  }
  function frame(now) {
    if (playing) {
      const dt = lastFrame == null ? 0 : (now - lastFrame) / 1000;
      let next = t + dt * speed;
      if (next >= D()) { next = D(); if (!followLive) stop(); }
      setTime(next);
    }
    lastFrame = now;
    requestAnimationFrame(frame);
  }
  const PLAY_ICON = '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M4.2 3.2v9.6a.8.8 0 0 0 1.2.7l7.6-4.8a.8.8 0 0 0 0-1.4L5.4 2.5A.8.8 0 0 0 4.2 3.2z" fill="currentColor"/></svg>';
  const PAUSE_ICON = '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><rect x="3.5" y="2.5" width="3" height="11" rx="1.2" fill="currentColor"/><rect x="9.5" y="2.5" width="3" height="11" rx="1.2" fill="currentColor"/></svg>';
  function play() { if (t >= D() && !followLive) setTime(0); playing = true; $("play").innerHTML = PAUSE_ICON; $("play").setAttribute("aria-label", "Pause"); }
  function stop() { playing = false; $("play").innerHTML = PLAY_ICON; $("play").setAttribute("aria-label", "Play"); }
  $("play").innerHTML = PLAY_ICON;
  $("play").addEventListener("click", () => (playing ? stop() : play()));
  scrub.addEventListener("input", () => { stop(); followLive = false; setTime(+scrub.value * D()); });
  // speed: slow / normal / fast as a three-icon toggle
  $("speed").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    speed = +b.dataset.speed;
    $("speed").querySelectorAll("button").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
  }));
  document.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(e.target.tagName) || !DATA) return;
    if (e.key === "ArrowRight") { stop(); setTime(Math.floor(t) + 1); }
    else if (e.key === "ArrowLeft") { stop(); setTime(Math.ceil(t) - 1); }
    else if (e.key === " ") { e.preventDefault(); playing ? stop() : play(); }
    else if (e.key === "Escape") { selected = null; lastPanel = ""; paintNodes(); drawPositions(); drawPanel(); }
  });

  // ============================================================ loading a run
  // live progress of a running job: which step is being computed, replies so far, time
  const clock = (sec) => { sec = Math.round(sec); return sec < 60 ? `${sec}s` : `${Math.floor(sec / 60)}m ${String(sec % 60).padStart(2, "0")}s`; };
  function progressParts(s) {
    const now = s.step + 1; // the step being computed (step 0 = first messages)
    const what = now === 0 ? "Writing first messages" : `Step ${now} of ${s.steps}`;
    const parts = [];
    if (s.agents && (s.replies > 0 || s.step < 0)) parts.push(`${s.replies}/${s.agents} agents replied` + (s.failed ? ` (${s.failed} failed)` : ""));
    if (s.elapsed != null) {
      parts.push(`${clock(s.elapsed)} elapsed`);
      const done = (s.step + 1) * s.agents + s.replies, total = (s.steps + 1) * s.agents;
      if (s.step >= 0 && done > 0) parts.push(`~${clock((s.elapsed / done) * (total - done))} left`);
    }
    return { what, parts };
  }
  function showStarting(s) {
    $("empty").hidden = false; $("view").hidden = true;
    const { what, parts } = s ? progressParts(s) : { what: "Starting", parts: [] };
    $("empty").querySelector("h1").textContent = `${what}…`;
    $("empty").querySelector("p").textContent = [...parts, "the first replies are slow while Ollama loads the models"].join(" · ");
  }
  let LIVE = null; // status of the open run while it is running, else null
  function stepOf() { // "/ 2" for a finished run, "/ 2 so far, of 5" while running
    $("of").textContent = LIVE && DATA ? ` / ${D()} so far, of ${LIVE.steps}` : ` / ${DATA ? D() : 0}`;
  }
  function statusPill(s) {
    const pill = $("status"); pill.className = "pill " + s.status;
    pill.hidden = s.status === "done"; // only worth a chip while running, or when it went wrong
    LIVE = s.status === "running" ? s : null;
    $("live").hidden = !LIVE;
    if (LIVE) {
      const { what, parts } = progressParts(s);
      pill.textContent = "Running";
      $("live-what").textContent = `Run in progress: ${what.toLowerCase()}`;
      $("live-parts").textContent = parts.length ? "· " + parts.join(" · ") : "";
      const total = (s.steps + 1) * s.agents, done = (s.step + 1) * s.agents + s.replies;
      $("live-fill").style.width = `${total ? Math.min(100, (100 * done) / total) : 0}%`;
    }
    else pill.textContent = s.status[0].toUpperCase() + s.status.slice(1);
    stepOf();
    $("cancel").hidden = s.status !== "running";
    if (s.error) pill.title = s.error;
  }
  function showData(data, keepTime) {
    const first = !DATA || !keepTime;
    DATA = data;
    $("empty").hidden = true; $("view").hidden = false;
    $("title").textContent = data.name;
    const topic = $("topic"); topic.replaceChildren();
    topic.classList.toggle("none", !data.topic);
    topic.textContent = data.topic || "None: numeric algorithm, opinions only";
    renderRunConfig(data);
    renderReport(data);
    stepOf();
    if (first) { selected = null; nodes = []; t = 0; drawGuide(); }
    t = Math.min(t, D()); // never point past the data we have
    setupNetwork(); buildCharts(); lastPanel = ""; lastTable = "";
    setTime(first ? 0 : followLive ? D() : Math.min(t, D()));
  }



  // ---------------------------------------------------------------- app mode
  async function api(path, opts) {
    const r = await fetch(path, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
    return body;
  }
  // the URL keeps what is open, so a reload lands in the same place:
  // /lab?tab=new, /lab?tab=runs, /lab?run=<id>. mode: "push" | "replace" | null (leave the URL alone)
  function setUrl(params, mode) {
    if (!mode) return;
    const url = `${location.pathname}?${new URLSearchParams(params)}`;
    if (url === location.pathname + location.search) return;
    history[mode === "push" ? "pushState" : "replaceState"](null, "", url);
  }
  function route() {
    const q = new URLSearchParams(location.search);
    if (q.get("run")) openRun(q.get("run"), null);
    else { if (runId) closeRun(null); setDrawer(q.get("tab") === "runs" ? "runs" : "new", "replace"); }
  }
  window.addEventListener("popstate", route);

  let pollTimer = null, lastSeenStep = -2;
  async function openRun(id, mode = "push") {
    setDrawer(null);
    setUrl({ run: id }, mode);
    $("labcard").hidden = true; $("run-nav").hidden = false; document.body.classList.add("run-open");
    runId = id; lastSeenStep = -2; clearTimeout(pollTimer); stop();
    renderRunsList();
    let res;
    try { res = await api(`/api/runs/${encodeURIComponent(id)}`); }
    catch (e) { if (id === runId) { closeRun(null); setDrawer("runs", "replace"); } return; } // deleted or unknown run
    if (id !== runId) return;
    statusPill(res.status);
    if (res.data) showData(res.data, false);
    else showStarting(res.status.status === "running" ? res.status : null);
    if (res.status.status === "running") { followLive = true; play(); poll(); }
    else if (res.data) play();
  }
  async function poll() {
    if (!runId) return;
    const id = runId;
    try {
      const s = await api(`/api/runs/${encodeURIComponent(id)}/status`);
      if (id !== runId) return;
      if (s.status === "running" && !DATA) showStarting(s);
      else statusPill(s);
      if (s.step !== lastSeenStep || s.status !== "running") {
        lastSeenStep = s.step;
        const res = await api(`/api/runs/${encodeURIComponent(id)}`);
        if (id !== runId) return;
        statusPill(res.status);
        if (res.data) showData(res.data, !!DATA && DATA.name === res.data.name);
      }
      if (s.status === "running") pollTimer = setTimeout(poll, 800);
      else {
        followLive = false; loadRuns();
        if (!DATA) { $("empty").querySelector("h1").textContent = `Run ${s.status}`; $("empty").querySelector("p").textContent = s.error || "No step was saved."; }
      }
    } catch (e) { pollTimer = setTimeout(poll, 2000); }
  }
  const EMPTY_TEXT = [$("empty").querySelector("h1").textContent, $("empty").querySelector("p").textContent];
  function closeRun(mode = "push") {
    runId = null; clearTimeout(pollTimer); stop(); if (sim) sim.stop(); followLive = false; LIVE = null; DATA = null;
    $("view").hidden = true; $("empty").hidden = true; // nothing behind the lab card when no run is open
    $("empty").querySelector("h1").textContent = EMPTY_TEXT[0]; $("empty").querySelector("p").textContent = EMPTY_TEXT[1];
    $("run-nav").hidden = true; $("labcard").hidden = false; document.body.classList.remove("run-open");
    if (mode) setDrawer("runs", mode);
  }
  $("back").addEventListener("click", () => closeRun("push"));
  $("cancel").addEventListener("click", () => api(`/api/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" }));

  // runs list
  let RUNS = [];
  async function loadRuns() { try { RUNS = await api("/api/runs"); renderRunsList(); suggestName(); } catch (e) { /* server gone */ } }

  // default run name: "Run N", one past the highest "Run N" so far; left alone once the user types a name
  let nameTouched = false;
  $("name").addEventListener("input", () => { nameTouched = true; });
  function suggestName() {
    if (nameTouched) return;
    const nums = (RUNS || []).map((r) => /^Run (\d+)$/i.exec(r.name || "")).filter(Boolean).map((m) => +m[1]);
    $("name").value = `Run ${nums.length ? Math.max(...nums) + 1 : 1}`;
  }
  function startedAt(stamp) { // "20261103-142012" -> "3 Nov 2026, 14:20"
    const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/.exec(stamp || "");
    if (!m) return "";
    const d = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]);
    return `${d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })}, ${m[4]}:${m[5]}`;
  }
  function duration(sec) { // 38 s · 4 min 12 s · 1 h 02 min; "—" when not saved
    if (sec == null) return "—";
    const s = Math.round(sec), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h ? `${h} h ${String(m).padStart(2, "0")} min` : m ? `${m} min ${s % 60} s` : `${s} s`;
  }
  const COPY = '<svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true"><rect x="5.5" y="5.5" width="8" height="8" rx="1.6" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M10.5 3.5v-.4A1.1 1.1 0 0 0 9.4 2H3.1A1.1 1.1 0 0 0 2 3.1v6.3a1.1 1.1 0 0 0 1.1 1.1h.4" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>';
  const TRASH = '<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 4h11M6 4V2.5h4V4M4 4l.7 9.5h6.6L12 4M6.8 6.5v4.5M9.2 6.5v4.5" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  // compact table (name, type, started, status); a click expands the row with the details and actions
  let expandedRun = null;
  const openSweeps = new Set(); // sweeps (runs/<sweep>/) whose runs are shown
  function renderRunsList() {
    const box = $("runs"); box.replaceChildren();
    if (!RUNS.length) { box.appendChild(el("p", "hint", "No runs yet.")); return; }
    const wrap = el("div", "runs-wrap"), table = el("table", "runs-table"), head = el("thead"), body = el("tbody"), hr = el("tr");
    ["Name", "Type", "Started", "Status", ""].forEach((h) => hr.appendChild(el("th", null, h)));
    head.appendChild(hr);
    // runs of a sweep are grouped under one row per sweep, closed until clicked
    const sweeps = new Map();
    RUNS.filter((r) => r.sweep).forEach((r) => { if (!sweeps.has(r.sweep)) sweeps.set(r.sweep, []); sweeps.get(r.sweep).push(r); });
    if (runId && runId.includes("/")) openSweeps.add(runId.split("/")[0]);
    sweeps.forEach((runs, name) => {
      const open = openSweeps.has(name), tr = el("tr", "sweep-row" + (open ? " open" : ""));
      const td = el("td"); td.colSpan = 5;
      const count = (st) => runs.filter((r) => r.status === st).length;
      const parts = [`${runs.length} runs`, `${count("done")} done`];
      if (count("running")) parts.push(`${count("running")} running`);
      if (count("failed")) parts.push(`${count("failed")} failed`);
      const rest = runs.length - count("done") - count("running") - count("failed");
      if (rest) parts.push(`${rest} cancelled or incomplete`);
      const chev = el("span", "row-chev"); chev.setAttribute("aria-hidden", "true");
      td.append(el("span", "sweep-tag", "Sweep"), el("span", "sweep-name", name), el("span", "hint", parts.join(" · ")), chev);
      tr.appendChild(td); tr.tabIndex = 0; tr.setAttribute("aria-expanded", String(open));
      const toggle = () => { open ? openSweeps.delete(name) : openSweeps.add(name); renderRunsList(); };
      tr.addEventListener("click", toggle);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
      body.appendChild(tr);
      if (open) runs.forEach((r) => runRow(r, body, true));
    });
    RUNS.filter((r) => !r.sweep).forEach((r) => runRow(r, body, false));
    table.append(head, body); wrap.appendChild(table); box.appendChild(wrap);
  }
  function runRow(r, body, inSweep) {
    {
      const llm = r.engine === "llm", open = expandedRun === r.id, tr = el("tr", (open ? "run-row open" : "run-row") + (inSweep ? " in-sweep" : ""));
      if (r.id === runId) tr.setAttribute("aria-current", "true");
      const tdName = el("td", "name"), tdType = el("td"), tdStatus = el("td");
      tdName.textContent = r.name;
      tdType.appendChild(el("span", "type-tag " + (llm ? "llm" : "num"), llm ? "LLM agents" : "Numeric"));
      tdStatus.appendChild(el("span", "pill " + r.status, r.status[0].toUpperCase() + r.status.slice(1)));
      if (llm && r.failed_pct != null && (r.failed_pct > 0 || r.status === "failed")) {
        const fp = el("span", "fail-pct" + (r.failed_pct > 5 ? " bad" : ""), `${rnum(r.failed_pct, 1)}% failed`);
        fp.title = "Share of replies that could not be used (see the run report)";
        tdStatus.appendChild(fp);
      }
      const chev = el("button", "row-chev"); chev.type = "button";
      chev.setAttribute("aria-expanded", String(open)); chev.setAttribute("aria-label", `${open ? "Hide" : "Show"} details of ${r.name}`);
      const tdChev = el("td", "actions"); tdChev.appendChild(chev);
      tr.append(tdName, tdType, el("td", "nowrap", startedAt(r.started)), tdStatus, tdChev);
      tr.addEventListener("click", () => { expandedRun = open ? null : r.id; renderRunsList(); });
      body.appendChild(tr);
      if (open) body.appendChild(runDetails(r, llm));
    }
  }
  function runDetails(r, llm) {
    const tr = el("tr", "run-details"), td = el("td"); td.colSpan = 5;
    const dl = el("dl");
    [["Duration", duration(r.status === "running" ? r.elapsed : r.seconds)], ["Agents", r.n], ["Steps", r.steps ?? "—"], ["Seed", r.seed ?? "—"],
      ...(r.replicate ? [["Replicate", r.replicate]] : []),
      ...(llm ? [["Failed replies", r.failed_pct == null ? NR : `${rnum(r.failed_pct, 1)}%`]] : []),
      ["Network", (TOPO_LABEL[r.topology] || r.topology || "").split(" (")[0]],
      [llm ? "Models" : "Rule", llm ? r.models.map(short).join(", ") : ALGO_LABEL[(r.models[0] || "").split("/")[1]] || ""]]
      .forEach(([k, v]) => { const d = el("div"); d.append(el("dt", null, k), el("dd", null, String(v))); dl.appendChild(d); });
    const bar = el("div", "run-actions");
    const del = el("button", "danger-btn"); del.type = "button"; del.innerHTML = TRASH; del.append("Delete");
    del.disabled = r.status === "running"; if (del.disabled) del.title = "Stop the run before deleting it";
    del.addEventListener("click", () => deleteRun(r));
    const go = el("button", "primary", "Open run"); go.type = "button";
    go.addEventListener("click", () => openRun(r.id));
    const dup = el("button", "btn-ghost"); dup.type = "button"; dup.innerHTML = COPY; dup.append("Duplicate");
    dup.title = "Start a new run with this configuration, ready to change";
    dup.disabled = !r.config; dup.addEventListener("click", () => duplicateRun(r));
    bar.append(go, dup, del);
    td.append(dl, bar); tr.appendChild(td);
    return tr;
  }
  async function deleteRun(r) {
    if (!confirm(`Delete "${r.name}"?\n\nIts run folder is removed from disk and cannot be restored.`)) return;
    try { await api(`/api/runs/${encodeURIComponent(r.id)}`, { method: "DELETE" }); }
    catch (err) { alert(`Could not delete "${r.name}": ${err.message}`); }
    loadRuns();
  }

  // lab card: "New run" and "Previous runs" expand the card under the header
  let drawerOpen = null;
  function setDrawer(which, mode) {
    drawerOpen = which;
    if (which) setUrl({ tab: which }, mode);
    $("labcard").classList.toggle("open", !!which);
    $("tab-new").setAttribute("aria-pressed", String(which === "new"));
    $("tab-runs").setAttribute("aria-pressed", String(which === "runs"));
    if (which) { $("pane-new").hidden = which !== "new"; $("pane-runs").hidden = which !== "runs"; }
    if (which === "runs") loadRuns();
  }
  // one of the two is always selected: clicking the open one, Escape or clicking outside do not close it
  $("tab-new").addEventListener("click", () => setDrawer("new", "replace"));
  $("tab-runs").addEventListener("click", () => setDrawer("runs", "replace"));

  // ---------------------------------------------------------------- form
  let OPTIONS = { ollama_models: [] };
  let engine = "llm";
  function setEngine(e) {
    engine = e;
    $("eng-llm").setAttribute("aria-pressed", e === "llm"); $("eng-num").setAttribute("aria-pressed", e === "numeric");
    syncVisibility();
    if (typeof renderWizard === "function") renderWizard();
    if (typeof scheduleChecks === "function") scheduleChecks();
  }
  $("eng-llm").addEventListener("click", () => setEngine("llm"));
  $("eng-num").addEventListener("click", () => setEngine("numeric"));

  function syncVisibility() {
    const topo = $("topology").value;
    document.querySelectorAll("[data-topo]").forEach((n) => { n.hidden = n.dataset.topo !== topo; });
    const dist = $("dist").value;
    $("dist-params").hidden = dist === "uniform";
    $("mean-label").firstChild.textContent = dist === "bimodal" ? "Camp position (±) " : "Mean ";
    const algo = $("algo").value;
    document.querySelectorAll("[data-algo=bc]").forEach((n) => { n.hidden = algo !== "bounded_confidence"; });
    document.querySelectorAll("[data-algo=fj]").forEach((n) => { n.hidden = algo !== "friedkin_johnsen"; });
  }
  ["topology", "dist", "algo"].forEach((id) => $(id).addEventListener("change", syncVisibility));
  // choice cards write their value into the hidden control named by data-for (#topology, #dist)
  document.querySelectorAll(".pick-cards").forEach((group) => group.querySelectorAll(".pick-card").forEach((card) => card.addEventListener("click", () => {
    group.querySelectorAll(".pick-card").forEach((c) => c.setAttribute("aria-pressed", String(c === card)));
    const target = $(group.dataset.for);
    target.value = card.dataset.value;
    target.dispatchEvent(new Event("change", { bubbles: true }));
  })));

  function modelOptions(select, value) {
    select.replaceChildren(...OPTIONS.ollama_models.map((m) => new Option(short(m), m)));
    if (value && !OPTIONS.ollama_models.includes(value)) select.appendChild(new Option(`${short(value)} (not installed here)`, value));
    if (value || select.options.length) select.value = value || select.options[0].value;
  }
  function addModelRow(name, weight) {
    const row = el("div", "model-row");
    const sel = document.createElement("select"); sel.setAttribute("aria-label", "Model");
    const w = document.createElement("input"); w.type = "number"; w.min = "0.1"; w.step = "any"; w.value = weight ?? 1; w.setAttribute("aria-label", "Weight");
    const count = el("span", "agents");
    const rm = el("button", "icon-btn", "×"); rm.type = "button"; rm.setAttribute("aria-label", "Remove model");
    modelOptions(sel, name ?? OPTIONS.ollama_models[0]);
    w.addEventListener("input", updateCounts);
    sel.addEventListener("change", syncModelChoices);
    rm.addEventListener("click", () => { if ($("models").children.length > 1) { row.remove(); updateCounts(); syncModelChoices(); scheduleChecks(); } });
    row.append(sel, w, count, rm);
    $("models").appendChild(row);
    updateCounts();
    syncModelChoices();
  }
  $("add-model").addEventListener("click", () => {
    const used = readModels().map((m) => m.name);
    const free = OPTIONS.ollama_models.find((m) => !used.includes(m));
    if (free) addModelRow(free, 1);
  });
  // a model can be picked once: options taken by another row are disabled, and
  // "+ Add model" disappears once every installed model is in the list
  function syncModelChoices() {
    const sels = [...$("models").querySelectorAll("select")];
    sels.forEach((sel) => [...sel.options].forEach((o) => { o.disabled = sels.some((other) => other !== sel && other.value === o.value); }));
    $("add-model").hidden = OPTIONS.ollama_models.every((m) => sels.some((sel) => sel.value === m));
  }
  $("n").addEventListener("input", updateCounts);

  // agents per model: exact shares by weight, largest-remainder rounding (as assign_models)
  function updateCounts() {
    const rows = [...$("models").children];
    const n = Math.max(0, Math.floor(+$("n").value || 0));
    const w = rows.map((r) => +r.querySelector("input[type=number]").value || 1);
    const total = w.reduce((a, b) => a + b, 0);
    const raw = w.map((x) => (x / total) * n);
    const counts = raw.map(Math.floor);
    raw.map((x, i) => [x - counts[i], i]).sort((a, b) => b[0] - a[0])
      .slice(0, n - counts.reduce((a, b) => a + b, 0)).forEach(([, i]) => { counts[i] += 1; });
    rows.forEach((r, i) => { r.querySelector(".agents").textContent = counts[i]; });
  }

  // installed models, the notice above them, and one-click downloads of the suggested ones
  function renderModels() {
    const note = $("ollama-note"), has = OPTIONS.ollama_models.length > 0;
    note.replaceChildren();
    if (OPTIONS.ollama_error) {
      note.append(OPTIONS.ollama_error);
      if (OPTIONS.ollama_missing) {
        note.append(" ");
        const a = el("a", null, "How to set it up"); a.href = "/docs#quick-start"; a.target = "_blank"; a.rel = "noopener";
        note.appendChild(a);
      }
    } else if (!has) note.append("No models installed yet. Download one below to start.");
    note.hidden = !note.childNodes.length;
    $("models-table").hidden = !has;
    if (has && !$("models").children.length) addModelRow(OPTIONS.ollama_models[0], 1);
    [...$("models").children].forEach((row) => { const sel = row.querySelector("select"); modelOptions(sel, sel.value); });
    syncModelChoices();
    renderLibrary();
  }
  // rows are built once and updated in place: rebuilding them on every poll would swallow clicks
  const libRows = new Map();
  function renderLibrary() {
    const dl = OPTIONS.downloads || {};
    const todo = OPTIONS.ollama_error ? [] : (OPTIONS.suggested || []).filter((m) => !OPTIONS.ollama_models.includes(`ollama_chat/${m.name}`));
    $("library-box").hidden = !todo.length;
    for (const [name, r] of libRows) if (!todo.some((m) => m.name === name)) { r.root.remove(); libRows.delete(name); }
    todo.forEach((m) => {
      let r = libRows.get(m.name);
      if (!r) {
        r = { root: el("div", "lib-row"), meta: el("div", "lib-meta"), btn: el("button", "btn-ghost"), bar: el("div", "lib-bar"), fill: el("span"), err: el("p", "error"), busy: false };
        const info = el("div"); info.append(el("div", "lib-name", m.label), r.meta);
        r.btn.type = "button"; r.bar.appendChild(r.fill);
        r.btn.addEventListener("click", () => (r.busy ? cancelPull(m.name) : pullModel(m.name)));
        r.root.append(info, r.btn, r.bar, r.err);
        $("library").appendChild(r.root); libRows.set(m.name, r);
      }
      const d = dl[m.name];
      r.busy = !!d && d.status === "downloading";
      r.meta.textContent = r.busy ? `Downloading… ${Math.round(d.progress * 100)}% of ${m.size}` : `${m.by} · ${m.size} · ${m.name}`;
      r.btn.textContent = r.busy ? "Cancel" : d && d.status === "failed" ? "Retry" : "Download";
      r.bar.hidden = !r.busy; r.fill.style.width = `${r.busy ? d.progress * 100 : 0}%`;
      r.err.hidden = !(d && d.error); r.err.textContent = d && d.error ? d.error : "";
    });
  }
  let pullTimer = null;
  const pulled = new Set(); // downloads already reflected in the model list
  async function pullModel(name) {
    try { OPTIONS.downloads = await api("/api/models/pull", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) }); }
    catch (err) { OPTIONS.downloads = { ...(OPTIONS.downloads || {}), [name]: { status: "failed", progress: 0, error: err.message } }; }
    renderLibrary();
    if (!pullTimer) pullTimer = setInterval(pollDownloads, 800);
  }
  async function cancelPull(name) {
    try { OPTIONS.downloads = await api("/api/models/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) }); }
    catch (err) { OPTIONS.downloads[name] = { ...OPTIONS.downloads[name], error: `Could not cancel: ${err.message}. Restart disensa app if it was started before this update.` }; }
    renderLibrary();
  }
  async function pollDownloads() {
    try { OPTIONS.downloads = await api("/api/models/pull"); } catch (e) { return; }
    const states = Object.values(OPTIONS.downloads);
    const finished = Object.keys(OPTIONS.downloads).filter((k) => OPTIONS.downloads[k].status === "done" && !pulled.has(k));
    finished.forEach((k) => pulled.add(k));
    if (finished.length) {
      try { OPTIONS = await api("/api/options"); } catch (e) { /* keep the old list */ }
      renderModels(); scheduleChecks();
    } else renderLibrary();
    if (!states.some((d) => d.status === "downloading")) { clearInterval(pullTimer); pullTimer = null; }
  }

  function readModels() {
    return [...$("models").children].map((row) => {
      const sel = row.querySelector("select"), w = row.querySelector("input[type=number]");
      return { name: sel.value, weight: +w.value || 1 };
    }).filter((m) => m.name);
  }
  const num = (id) => +$(id).value;

  // the form -> an ExperimentConfig (see config.py); `dynamics.type` is the engine switch
  function buildConfig() {
    const topo = $("topology").value;
    const network = { topology: topo, n: num("n") };
    if (topo === "watts_strogatz") Object.assign(network, { k: num("k"), beta: num("beta") });
    if (topo === "barabasi_albert") network.m = num("m");
    if (topo === "erdos_renyi") network.p = num("p");
    const cfg = {
      name: $("name").value.trim() || "run",
      seed: num("seed"),
      network,
      opinions: { distribution: $("dist").value, mean: num("mean"), std: num("std") },
    };
    if (engine === "numeric") {
      cfg.steps = num("steps");
      cfg.dynamics = { type: "numeric", rule: $("algo").value, epsilon: num("eps"), mu: num("mu"), stubbornness: num("lam") };
      return cfg;
    }
    const models = readModels();
    cfg.steps = num("llm-steps");
    cfg.dynamics = {
      type: "llm",
      statement: $("statement").value.trim(),
      models,
      interactions: { min: num("imin"), max: num("imax") },
      max_concurrency: 4,
    };
    return cfg;
  }

  // a saved config -> the form, the inverse of buildConfig ("Duplicate" in Previous runs)
  function pickCard(id, value) {
    if (value == null) return;
    $(id).value = value;
    document.querySelectorAll(`.pick-cards[data-for="${id}"] .pick-card`).forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.value === value)));
  }
  function fillForm(c) {
    const set = (id, v) => { if (v != null) $(id).value = v; };
    const nw = c.network || {}, op = c.opinions || {}, d = c.dynamics || {};
    set("name", `${c.name} (copy)`); nameTouched = true;
    set("seed", c.seed);
    pickCard("topology", nw.topology); set("n", nw.n); set("k", nw.k); set("beta", nw.beta); set("m", nw.m); set("p", nw.p);
    pickCard("dist", op.distribution); set("mean", op.mean); set("std", op.std);
    if (d.type === "numeric") {
      pickCard("algo", d.rule); set("steps", c.steps); set("eps", d.epsilon); set("mu", d.mu); set("lam", d.stubbornness);
    } else {
      set("statement", d.statement); set("llm-steps", c.steps);
      if (d.interactions) { set("imin", d.interactions.min); set("imax", d.interactions.max); }
      $("models").replaceChildren();
      (d.models || []).forEach((m) => addModelRow(m.name, m.weight));
      if (!$("models").children.length && OPTIONS.ollama_models.length) addModelRow(OPTIONS.ollama_models[0], 1);
    }
    setEngine(d.type === "numeric" ? "numeric" : "llm"); // also syncs the visible fields and re-runs the checks
    updateCounts();
  }
  function duplicateRun(r) {
    fillForm(r.config);
    setDrawer("new", "replace");
    furthest = stepsFor().length - 1; // every step is already filled in, so any step can be jumped to
    goStep("review");
  }

  // ---------------------------------------------------------------- new-run wizard
  const STEP_TITLES = { start: "Start", network: "Network", opinions: "Initial opinions", algorithm: "Algorithm",
    models: "Topic and models", interactions: "Interactions", review: "Review" };
  const stepsFor = () => engine === "numeric"
    ? ["start", "network", "opinions", "algorithm", "review"]
    : ["start", "network", "opinions", "models", "interactions", "review"];
  let stepKey = "start", furthest = 0;
  const sectionOf = (key) => document.querySelector(`.wiz-step[data-step="${key}"]`);

  function renderWizard() {
    const steps = stepsFor();
    if (!steps.includes(stepKey)) stepKey = steps[Math.min(steps.length - 1, furthest)] || "start";
    const i = steps.indexOf(stepKey);
    furthest = Math.min(Math.max(furthest, i), steps.length - 1);
    document.querySelectorAll(".wiz-step").forEach((sec) => { sec.hidden = sec.dataset.step !== stepKey; });
    $("wiz-count").textContent = `Step ${i + 1} of ${steps.length}`;
    const bar = $("wiz-progress"); bar.replaceChildren();
    steps.forEach((k, j) => {
      const b = el("button"); b.type = "button";
      b.className = j === i ? "current" : j <= furthest ? "done" : "";
      b.disabled = j > furthest;
      b.setAttribute("aria-label", `${STEP_TITLES[k]}${j === i ? " (current step)" : ""}`);
      b.title = STEP_TITLES[k];
      b.addEventListener("click", () => goStep(k));
      bar.appendChild(b);
    });
    $("wiz-back").style.visibility = i === 0 ? "hidden" : "visible";
    const last = stepKey === "review";
    $("wiz-next").hidden = last; $("run-btn").hidden = !last;
    $("wiz-next").textContent = steps[i + 1] === "review" ? "Review" : "Next";
    if (last) renderReview();
  }
  function goStep(key) {
    $("form-error").textContent = "";
    stepKey = key; renderWizard();
    const scroller = document.querySelector(".drawer-scroll"); if (scroller) scroller.scrollTop = 0;
  }
  function stepValid() {
    const sec = sectionOf(stepKey);
    for (const input of sec.querySelectorAll("input, select, textarea")) {
      if (input.offsetParent !== null && !input.checkValidity()) { input.reportValidity(); return false; }
    }
    if (stepKey === "models" && !readModels().length) { $("form-error").textContent = "Add at least one model."; return false; }
    return true;
  }
  function next() {
    if (!stepValid()) return;
    const steps = stepsFor(), i = steps.indexOf(stepKey);
    furthest = Math.max(furthest, i + 1);
    goStep(steps[Math.min(i + 1, steps.length - 1)]);
  }
  $("wiz-next").addEventListener("click", next);
  $("wiz-back").addEventListener("click", () => { const steps = stepsFor(); goStep(steps[Math.max(0, steps.indexOf(stepKey) - 1)]); });
  // Enter moves to the next step instead of submitting early
  $("form").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.target.tagName === "INPUT" && stepKey !== "review") { e.preventDefault(); next(); }
  });

  const TOPO_LABEL = { watts_strogatz: "Small-world (Watts–Strogatz)", barabasi_albert: "Scale-free (Barabási–Albert)", erdos_renyi: "Random (Erdős–Rényi)" };
  const ALGO_LABEL = { bounded_confidence: "Bounded confidence", degroot: "DeGroot", friedkin_johnsen: "Friedkin–Johnsen" };
  const optText = (id) => { const sel = $(id); return sel.options[sel.selectedIndex] ? sel.options[sel.selectedIndex].text : sel.value; };
  const DIST_LABEL = { uniform: "Uniform", normal: "Normal", bimodal: "Two camps" };
  // [step key, [[label, value], ...]] for a config; `counts` (model -> agents) switches to the run view
  function configBlocks(c, counts) {
    const nw = c.network, op = c.opinions || {}, d = c.dynamics, numeric = d.type !== "llm";
    const blocks = [
      ["start", [["Run name", c.name], ["Seed", c.seed], ["Agents", numeric ? "Numeric algorithms" : "LLM agents"]]],
      ["network", [["Topology", TOPO_LABEL[nw.topology] || nw.topology], ["Agents", nw.n],
        ...(nw.topology === "watts_strogatz" ? [["Neighbours k", nw.k], ["Rewire β", nw.beta]] : nw.topology === "barabasi_albert" ? [["Links per new node m", nw.m]] : [["Edge probability p", nw.p]])]],
      ["opinions", [["Distribution", DIST_LABEL[op.distribution] || op.distribution], ...(op.distribution === "uniform" ? [] : [[op.distribution === "bimodal" ? "Camp position" : "Mean", op.distribution === "bimodal" ? `±${op.mean}` : op.mean], ["Spread", op.std]])]],
    ];
    if (numeric) {
      blocks.push(["algorithm", [["Rule", ALGO_LABEL[d.rule] || d.rule], ["Steps", c.steps],
        ...(d.rule === "bounded_confidence" ? [["Confidence ε", d.epsilon], ["Rate μ", d.mu]]
          : d.rule === "friedkin_johnsen" ? [["Stubbornness λ", d.stubbornness]] : [])]]);
    } else {
      const models = counts
        ? d.models.map((m) => [short(m.name), `${counts[m.name] || 0} agents`])
        : [["Models", d.models.map((m) => `${short(m.name)} ×${m.weight}`).join(", ")]];
      blocks.push(["models", counts ? models : [["Statement", d.statement], ...models]]);
      blocks.push(["interactions", [["Steps", c.steps], ["Interactions per step", `${d.interactions.min}–${d.interactions.max}`]]]);
    }
    return blocks;
  }
  // the run's settings as four short sections: a main value, then the details in one muted line
  function renderRunConfig(data) {
    const c = data.config, grid = $("config-grid");
    grid.replaceChildren(); grid.hidden = !c;
    if (!c) return;
    const nw = c.network, op = c.opinions || {}, d = c.dynamics, llm = d.type === "llm";
    const join = (...xs) => xs.filter((x) => x != null && x !== "").join(" · ");
    const section = (title, main, sub, cls) => {
      const box = el("div", "spec" + (cls ? " " + cls : ""));
      box.appendChild(el("span", "meta-label", title));
      if (main != null) box.appendChild(el("div", "spec-main", main));
      if (sub) box.appendChild(el("div", "spec-sub", sub));
      grid.appendChild(box); return box;
    };
    section("Network", (TOPO_LABEL[nw.topology] || nw.topology).split(" (")[0], join(`${nw.n} agents`,
      nw.topology === "watts_strogatz" ? `k = ${nw.k} · β = ${nw.beta}` : nw.topology === "barabasi_albert" ? `${nw.m} links per new node` : `p = ${nw.p}`));
    section("Initial opinions", DIST_LABEL[op.distribution] || op.distribution,
      op.distribution === "bimodal" ? `at ±${op.mean} · spread ${op.std}` : op.distribution === "normal" ? `mean ${op.mean} · spread ${op.std}` : "from −2 to +2");
    if (llm) {
      // one row per model, coloured as in "Mean opinion by model", with its share of the agents
      const counts = {}; data.agents.forEach((a) => { counts[a.model] = (counts[a.model] || 0) + 1; });
      const box = section("Models", null, null, "spec-models"), n = data.agents.length || 1;
      data.models.forEach((m, i) => {
        const color = data.models.length > 1 ? css(`--s${(i % 8) + 1}`) : css("--accent");
        const row = el("div", "model-share"), bar = el("span", "share-bar"), fill = el("span");
        fill.style.width = `${(100 * (counts[m] || 0)) / n}%`; fill.style.background = color; bar.appendChild(fill);
        const name = el("span", "share-name", short(m)); name.title = m;
        row.append(name, bar, el("span", "share-count", `${counts[m] || 0}`));
        box.appendChild(row);
      });
      // sampling settings, read-only (config file only); older runs did not record all of them
      const sampling = (m) => [
        `temperature ${m.temperature ?? "not recorded"}`,
        `top_p ${m.top_p ?? "not recorded"}`,
        `top_k ${m.top_k ?? "not recorded"}`,
        m.think == null ? "thinking not recorded" : m.think ? "thinking on" : "thinking off",
      ].join(" · ");
      const lines = [...new Set(d.models.map(sampling))];
      lines.forEach((line) => {
        const who = lines.length > 1 ? d.models.filter((m) => sampling(m) === line).map((m) => short(m.name)).join(", ") + ": " : "";
        box.appendChild(el("div", "spec-sub", who + line));
      });
    } else {
      section("Update rule", ALGO_LABEL[d.rule] || d.rule,
        d.rule === "bounded_confidence" ? `ε = ${d.epsilon} · μ = ${d.mu}` : d.rule === "friedkin_johnsen" ? `λ = ${d.stubbornness}` : "");
    }
    section("Simulation", `${c.steps} steps`, join(llm && d.interactions ? `${d.interactions.min}–${d.interactions.max} interactions per step` : "", `seed ${c.seed}`, c.replicate ? `replicate ${c.replicate}` : ""));
  }
  // the run report (report.py): how far the run can be trusted, and what it cost.
  // Older runs did not record every number: those show "not recorded".
  const NR = "not recorded";
  const rnum = (v, digits = 0) => v == null ? NR : Number(v).toLocaleString(undefined, { maximumFractionDigits: digits });
  const KIND_LABEL = { provider: "server error", invalid_json: "not JSON", missing_belief: "no belief", invalid_belief: "belief not a number" };
  function renderReport(data) {
    const r = data.report, card = $("report-card"), grid = $("report-grid");
    grid.replaceChildren();
    const reason = $("report-reason");
    reason.hidden = !data.reason; reason.textContent = data.reason ? `Stopped: ${data.reason}` : "";
    card.hidden = !r || data.config.dynamics.type !== "llm";
    if (card.hidden) return;
    const section = (title, main, subs, bad) => {
      const box = el("div", "spec");
      box.appendChild(el("span", "meta-label", title));
      box.appendChild(el("div", "spec-main" + (bad ? " bad" : ""), main));
      (subs || []).filter(Boolean).forEach((sub) => box.appendChild(el("div", "spec-sub", sub)));
      grid.appendChild(box);
    };
    const kinds = Object.entries(r.failed_by_kind || {}).map(([k, v]) => `${KIND_LABEL[k] || k} ${v}`).join(" · ");
    const models = Object.entries(r.failed_by_model || {});
    section("Failed replies", `${rnum(r.failed_pct, 1)}%`, [
      `${rnum(r.failed)} of ${rnum(r.calls)} replies`, r.failed ? kinds : "",
      ...(r.failed && models.length > 1 ? models.map(([m, x]) => `${short(m)}: ${x.failed}/${x.calls} (${rnum(x.pct, 1)}%)`) : []),
    ], r.failed_pct > 5);
    section("Tokens", r.prompt_tokens == null ? NR : `${rnum(r.prompt_tokens)} in · ${rnum(r.completion_tokens)} out`,
      [r.prompt_tokens_mean == null ? "" : `${rnum(r.prompt_tokens_mean)} in · ${rnum(r.completion_tokens_mean)} out per reply`]);
    section("Time", duration(r.seconds), [
      r.seconds_per_call == null ? `per model call: ${NR}` : `${rnum(r.seconds_per_call, 2)} s per model call`,
      r.cache_hits == null ? `cache: ${NR}` : `${rnum(r.cache_hits)} of ${rnum(r.calls)} replies from the cache`,
    ]);
    section("Out of range", r.out_of_range == null ? NR : `${rnum(r.out_of_range)}`, [r.out_of_range == null ? "" : "beliefs outside −2…+2, clipped"]);
    const g = r.granularity;
    const rated = g ? g.whole + g.half + g.other : 0;
    if (!g) section("Belief granularity", NR);
    else if (!rated) section("Belief granularity", "—", ["no valid self-ratings yet"]);
    else {
      const pct = (v) => `${Math.round((100 * v) / rated)}%`;
      section("Belief granularity", `${pct(g.whole)} whole · ${pct(g.half)} half · ${pct(g.other)} other`, [`${rnum(rated)} valid self-ratings, steps ≥ 1`]);
    }
  }
  function renderReview() {
    const blocks = configBlocks(buildConfig());
    const box = $("review"); box.replaceChildren();
    blocks.forEach(([key, rows]) => {
      const card = el("div", "review-block");
      const head = el("header"); head.appendChild(el("h4", null, STEP_TITLES[key]));
      const edit = el("button", "edit-link", "Edit"); edit.type = "button"; edit.addEventListener("click", () => goStep(key));
      head.appendChild(edit); card.appendChild(head);
      const dl = el("dl");
      rows.forEach(([k, v]) => { dl.append(el("dt", null, k), el("dd", null, String(v))); });
      card.appendChild(dl); box.appendChild(card);
    });
  }

  // ---------------------------------------------------------------- warnings (checks.py)
  // Settings that are allowed but probably not intended, shown under the step they belong to
  // and again on the review step. Each links to the docs section that explains it.
  const checkStep = (field) => field.startsWith("network.") ? "network"
    : field.startsWith("opinions.") ? "opinions"
    : field.startsWith("dynamics.interactions") ? "interactions"
    : field.startsWith("dynamics.") ? "algorithm" : null;
  let CHECKS = [], checkTimer = null, checkSeq = 0;
  async function runChecks() {
    const seq = ++checkSeq;
    try {
      const res = await api("/api/check", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ config: buildConfig() }) });
      if (seq !== checkSeq) return; // a newer check is on its way
      CHECKS = res.checks || []; renderChecks();
    } catch (e) { /* server gone: no advice */ }
  }
  function scheduleChecks() { clearTimeout(checkTimer); checkTimer = setTimeout(runChecks, 200); }
  function checkNode(c) {
    const box = el("div", c.level === "error" ? "check check-error" : "notice check");
    box.appendChild(document.createTextNode(c.message + " "));
    if (c.docs) {
      const a = document.createElement("a");
      a.href = `/docs#${c.docs}`; a.target = "_blank"; a.rel = "noopener"; a.textContent = "Read more →";
      a.title = c.docs_title;
      box.appendChild(a);
    }
    return box;
  }
  function renderChecks() {
    document.querySelectorAll("[data-checks]").forEach((box) => {
      const step = box.dataset.checks;
      box.replaceChildren(...CHECKS.filter((c) => step === "review" || checkStep(c.field) === step).map(checkNode));
    });
  }
  $("form").addEventListener("input", scheduleChecks);
  $("form").addEventListener("change", scheduleChecks);

  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("form-error").textContent = "";
    const btn = $("run-btn"); btn.disabled = true; btn.textContent = "Starting…";
    try {
      const res = await api("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ config: buildConfig() }) });
      DATA = null;
      nameTouched = false;
      await loadRuns();
      openRun(res.id);
    } catch (err) {
      $("form-error").textContent = err.message;
    } finally {
      btn.disabled = false; btn.textContent = "Run simulation";
    }
  });

  async function init() {
    try { OPTIONS = await api("/api/options"); } catch (e) { OPTIONS = { ollama_models: [], suggested: [] }; $("form-error").textContent = "Cannot reach the local server. Start it again from Terminal."; }
    renderModels();
    syncVisibility();
    renderWizard();
    runChecks();
    await loadRuns();
    route();
    requestAnimationFrame(frame);
  }
  init();
})();
