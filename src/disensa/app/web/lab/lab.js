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
    if ($("table-card").hidden) return;
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
  function play() { if (t >= D() && !followLive) setTime(0); playing = true; $("play").textContent = "❚❚"; $("play").setAttribute("aria-label", "Pause"); }
  function stop() { playing = false; $("play").textContent = "▶"; $("play").setAttribute("aria-label", "Play"); }
  $("play").addEventListener("click", () => (playing ? stop() : play()));
  scrub.addEventListener("input", () => { stop(); followLive = false; setTime(+scrub.value * D()); });
  $("speed").addEventListener("change", (e) => { speed = +e.target.value; });
  $("table-toggle").addEventListener("click", () => { $("table-card").hidden = !$("table-card").hidden; lastTable = ""; drawTable(); });
  document.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(e.target.tagName) || !DATA) return;
    if (e.key === "ArrowRight") { stop(); setTime(Math.floor(t) + 1); }
    else if (e.key === "ArrowLeft") { stop(); setTime(Math.ceil(t) - 1); }
    else if (e.key === " ") { e.preventDefault(); playing ? stop() : play(); }
    else if (e.key === "Escape") { selected = null; lastPanel = ""; paintNodes(); drawPositions(); drawPanel(); }
  });

  // ============================================================ loading a run
  function statusPill(s) {
    const pill = $("status"); pill.className = "pill " + s.status;
    pill.textContent = s.status === "running" ? `Running · step ${Math.max(0, s.step)} of ${s.steps}` : s.status[0].toUpperCase() + s.status.slice(1);
    $("cancel").hidden = s.status !== "running";
    if (s.error) pill.title = s.error;
  }
  function showData(data, keepTime) {
    const first = !DATA || !keepTime;
    DATA = data;
    $("empty").hidden = true; $("view").hidden = false;
    $("title").textContent = data.name;
    $("topic").textContent = data.topic ? `“${data.topic}”` : "Numeric model: opinions only, no text.";
    const chips = $("chips"); chips.replaceChildren();
    [`${data.agents.length} agents`, data.topology.replace("_", " "), `seed ${data.seed}`, ...data.models.map(short)].forEach((c) => chips.appendChild(el("span", "chip", c)));
    $("of").textContent = ` / ${D()}`;
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
  let pollTimer = null, lastSeenStep = -2;
  async function openRun(id) {
    setDrawer(null);
    runId = id; lastSeenStep = -2; clearTimeout(pollTimer); stop();
    renderRunsList();
    const res = await api(`/api/runs/${encodeURIComponent(id)}`);
    statusPill(res.status);
    if (res.data) showData(res.data, false);
    else { $("empty").hidden = false; $("view").hidden = true; $("empty").querySelector("h1").textContent = "Starting…"; }
    if (res.status.status === "running") { followLive = true; play(); poll(); }
    else if (res.data) play();
  }
  async function poll() {
    if (!runId) return;
    const id = runId;
    try {
      const s = await api(`/api/runs/${encodeURIComponent(id)}/status`);
      if (id !== runId) return;
      if (s.status === "running" && !DATA) { $("empty").hidden = false; $("view").hidden = true; $("empty").querySelector("h1").textContent = `Starting… step ${Math.max(0, s.step)} of ${s.steps}`; }
      else statusPill(s);
      if (s.step !== lastSeenStep || s.status !== "running") {
        lastSeenStep = s.step;
        const res = await api(`/api/runs/${encodeURIComponent(id)}`);
        if (id !== runId) return;
        statusPill(res.status);
        if (res.data) showData(res.data, !!DATA && DATA.name === res.data.name);
      }
      if (s.status === "running") pollTimer = setTimeout(poll, 800);
      else { followLive = false; loadRuns(); }
    } catch (e) { pollTimer = setTimeout(poll, 2000); }
  }
  $("cancel").addEventListener("click", () => api(`/api/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" }));

  // runs list
  let RUNS = [];
  async function loadRuns() { try { RUNS = await api("/api/runs"); renderRunsList(); } catch (e) { /* server gone */ } }
  function renderRunsList() {
    const box = $("runs"); box.replaceChildren();
    if (!RUNS.length) { box.appendChild(el("p", "hint", "No runs yet.")); return; }
    RUNS.forEach((r) => {
      const b = el("button", "run-item"); b.type = "button";
      if (r.id === runId) b.setAttribute("aria-current", "true");
      const top = el("div", "t"); top.append(el("span", null, r.name), el("span", "pill " + r.status, r.status));
      const when = r.started ? `${r.started.slice(0, 4)}-${r.started.slice(4, 6)}-${r.started.slice(6, 8)} ${r.started.slice(9, 11)}:${r.started.slice(11, 13)}` : "";
      b.append(top, el("div", "m", `${when} · ${r.n} agents · ${(r.topology || "").replace("_", " ")}`), el("div", "m", r.models.map(short).join(", ")));
      b.addEventListener("click", () => openRun(r.id));
      box.appendChild(b);
    });
  }

  // lab card: "New run" and "Runs" expand the card under the header; clicking the open one closes it
  let drawerOpen = null;
  function setDrawer(which) {
    drawerOpen = which;
    $("labcard").classList.toggle("open", !!which);
    $("tab-new").setAttribute("aria-pressed", String(which === "new"));
    $("tab-runs").setAttribute("aria-pressed", String(which === "runs"));
    if (which) { $("pane-new").hidden = which !== "new"; $("pane-runs").hidden = which !== "runs"; }
    if (which === "runs") loadRuns();
  }
  const showTab = (which) => setDrawer(drawerOpen === which ? null : which);
  $("tab-new").addEventListener("click", () => showTab("new"));
  $("tab-runs").addEventListener("click", () => showTab("runs"));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && drawerOpen) setDrawer(null); });
  document.addEventListener("pointerdown", (e) => { if (drawerOpen && !$("labcard").contains(e.target)) setDrawer(null); });

  // ---------------------------------------------------------------- form
  let OPTIONS = { ollama_models: [], fake_models: [], presets: [] };
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

  function modelOptions(select, value) {
    select.replaceChildren();
    const groups = [["Ollama (on this Mac)", OPTIONS.ollama_models], ["Offline test models", OPTIONS.fake_models]];
    groups.forEach(([label, list]) => {
      if (!list.length) return;
      const g = document.createElement("optgroup"); g.label = label;
      list.forEach((m) => g.appendChild(new Option(short(m) + (m.startsWith("fake/") ? " (fake)" : ""), m)));
      select.appendChild(g);
    });
    if (value && ![...select.options].some((o) => o.value === value)) {
      select.appendChild(new Option(`${short(value)} (not installed here)`, value));
    }
    select.appendChild(new Option("Other (type a model name)…", "__custom__"));
    select.value = value == null ? select.options[0].value : value;
    return true;
  }
  function addModelRow(name, weight) {
    const row = el("div", "model-row");
    const sel = document.createElement("select"); sel.setAttribute("aria-label", "Model");
    const w = document.createElement("input"); w.type = "number"; w.min = "0.1"; w.step = "any"; w.value = weight ?? 1; w.title = "Weight: share of agents"; w.setAttribute("aria-label", "Weight");
    const rm = el("button", "icon-btn", "×"); rm.type = "button"; rm.setAttribute("aria-label", "Remove model");
    const custom = document.createElement("input"); custom.type = "text"; custom.className = "custom"; custom.placeholder = "e.g. anthropic/claude-haiku-4-5 or ollama_chat/gemma2:9b";
    const known = modelOptions(sel, name ?? (OPTIONS.ollama_models[0] || "fake/bounded"));
    custom.hidden = known; if (!known && name) custom.value = name;
    sel.addEventListener("change", () => { custom.hidden = sel.value !== "__custom__"; });
    rm.addEventListener("click", () => { if ($("models").children.length > 1) row.remove(); });
    row.append(sel, w, rm, custom);
    $("models").appendChild(row);
  }
  $("add-model").addEventListener("click", () => addModelRow(null, 1));

  function readModels() {
    return [...$("models").children].map((row) => {
      const sel = row.querySelector("select"), custom = row.querySelector("input.custom"), w = row.querySelector("input[type=number]");
      const name = sel.value === "__custom__" ? custom.value.trim() : sel.value;
      return { name, weight: +w.value || 1 };
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
      name: ($("name").value.trim() || "run").replace(/[^A-Za-z0-9_-]+/g, "_"),
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
      max_concurrency: models.every((m) => m.name.startsWith("fake/")) ? 64 : 4,
    };
    return cfg;
  }

  // a preset config -> the form
  function applyConfig(c) {
    const d = c.dynamics || {};
    setEngine(d.type === "numeric" ? "numeric" : "llm");
    const set = (id, v) => { if (v != null) $(id).value = v; };
    set("name", c.name); set("seed", c.seed);
    const nw = c.network || {};
    set("topology", nw.topology); set("n", nw.n); set("k", nw.k); set("beta", nw.beta); set("m", nw.m); set("p", nw.p);
    const op = c.opinions || {};
    set("dist", op.distribution || "uniform"); set("mean", op.mean ?? 0); set("std", op.std ?? 1);
    if (d.type === "numeric") { set("algo", d.rule); set("steps", c.steps); set("eps", d.epsilon); set("mu", d.mu); set("lam", d.stubbornness ?? 1); }
    else {
      set("llm-steps", c.steps); set("statement", d.statement);
      set("imin", d.interactions && d.interactions.min); set("imax", d.interactions && d.interactions.max);
      if (d.models && d.models.length) { $("models").replaceChildren(); d.models.forEach((m) => addModelRow(m.name, m.weight)); }
    }
    syncVisibility();
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
  function renderReview() {
    const c = buildConfig(), nw = c.network, op = c.opinions;
    const blocks = [
      ["start", [["Run name", c.name], ["Seed", c.seed], ["Agents", engine === "numeric" ? "Numeric model" : "LLM agents"], ["Preset", $("preset").value ? $("preset").value.replace(/\.yaml$/, "") : "Custom"]]],
      ["network", [["Topology", TOPO_LABEL[nw.topology]], ["Agents", nw.n],
        ...(nw.topology === "watts_strogatz" ? [["Neighbours k", nw.k], ["Rewire β", nw.beta]] : nw.topology === "barabasi_albert" ? [["Links per new node m", nw.m]] : [["Edge probability p", nw.p]])]],
      ["opinions", [["Distribution", optText("dist")], ...(op.distribution === "uniform" ? [] : [[op.distribution === "bimodal" ? "Camp position" : "Mean", `±${op.mean}`.replace("±", op.distribution === "bimodal" ? "±" : "")], ["Spread", op.std]])]],
    ];
    const d = c.dynamics;
    if (engine === "numeric") {
      blocks.push(["algorithm", [["Rule", ALGO_LABEL[d.rule]], ["Steps", c.steps],
        ...(d.rule === "bounded_confidence" ? [["Confidence ε", d.epsilon], ["Rate μ", d.mu]]
          : d.rule === "friedkin_johnsen" ? [["Stubbornness λ", d.stubbornness]] : [])]]);
    } else {
      blocks.push(["models", [["Statement", d.statement], ["Models", d.models.map((m) => `${short(m.name)} ×${m.weight}`).join(", ")]]]);
      blocks.push(["interactions", [["Steps", c.steps], ["Interactions per step", `${d.interactions.min}–${d.interactions.max}`]]]);
    }
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

  $("preset").addEventListener("change", (e) => {
    const p = OPTIONS.presets.find((x) => x.file === e.target.value);
    if (p) applyConfig(p.config);
    renderWizard();
  });

  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("form-error").textContent = "";
    const btn = $("run-btn"); btn.disabled = true; btn.textContent = "Starting…";
    try {
      const res = await api("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ config: buildConfig() }) });
      DATA = null;
      await loadRuns();
      openRun(res.id);
    } catch (err) {
      $("form-error").textContent = err.message;
    } finally {
      btn.disabled = false; btn.textContent = "Run simulation";
    }
  });

  async function init() {
    try { OPTIONS = await api("/api/options"); } catch (e) { $("form-error").textContent = "Cannot reach the local server. Start it again from Terminal."; }
    if (OPTIONS.ollama_error) { $("ollama-note").hidden = false; $("ollama-note").textContent = OPTIONS.ollama_error + " Offline test models still work."; }
    OPTIONS.presets.forEach((p) => $("preset").appendChild(new Option(p.file.replace(/\.yaml$/, ""), p.file)));
    const start = OPTIONS.presets.find((p) => p.file === "demo_camps.yaml");
    if (start) { applyConfig(start.config); $("preset").value = start.file; }
    else addModelRow(OPTIONS.ollama_models[0] || "fake/tribal", 1);
    syncVisibility();
    renderWizard();
    runChecks();
    await loadRuns();
    setDrawer("new");
    requestAnimationFrame(frame);
  }
  init();
})();
