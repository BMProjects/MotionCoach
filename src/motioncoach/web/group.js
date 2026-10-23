"use strict";
// Multi-person overview (docs/MULTI_PERSON_DESIGN.md §5b): identities on the video, people list, person x section
// matrix, synchronisation, floor layout, consistency, and identity review (merge / split / role). Each performer's
// detail opens in the single-person workbench (app.js openReport with ?person=).
// Uses the helpers of app.js: $, $$, el, api, css, show, refreshTasks.
// All top-level names here are global next to app.js: group-only names carry a "g" prefix where app.js has the same word.

const G = { id: null, job: null, g: null, tab: "people", edits: [], pick: new Set(), raf: 0, poll: null, tl: null, nowK: -1 };

/** Practitioners in identification order get greens from saturated (first) to pale (last); excluded people red.
 *  The number label always identifies the person (a saturation ramp cannot tell many people apart on its own). */
function idColor(id, g = G.g) {
  const perf = g.identities.filter((i) => i.performer).map((i) => i.id).sort((a, b) => a - b);
  const k = perf.indexOf(id);
  if (k < 0) return css("--bad");
  const f = perf.length > 1 ? k / (perf.length - 1) : 0;
  return `hsl(142, ${Math.round(88 - 62 * f)}%, ${Math.round(30 + 22 * f)}%)`;
}
const fmtS = (v) => (v === null || v === undefined ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(1)} s`);
const mmss = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const personOf = (id) => G.g.persons.find((p) => p.id === id);
const segShort = (name) => { const m = name.match(/^第(.)式/); return m ? `${m[1]}式` : name.slice(0, 3); };
const identity = (id) => G.g.identities.find((i) => i.id === id);
const svgEl = (tag, attrs = {}, ...kids) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) e.setAttribute(k, v);
  for (const k of kids) e.append(k instanceof Node ? k : document.createTextNode(k));
  return e;
};

async function openGroup(id, tab = null) {
  show("view-group");
  if (tab) G.tab = tab;
  clearTimeout(G.poll);
  const job = await api(`/api/jobs/${id}`).catch(() => null);
  if (!job || job.status !== "done") {                       // re-running after identity edits, or not finished yet
    $("#g-title").textContent = "集体分析";
    $("#g-sub").textContent = job ? `正在计算：${jobProgressText(job)}` : "找不到这条分析";
    if (job && (job.status === "queued" || job.status === "running")) G.poll = setTimeout(() => openGroup(id, tab), 1500);
    return;
  }
  const g = await api(`/api/jobs/${id}/group`);
  Object.assign(G, { id, job, g, edits: [], pick: new Set() });
  renderGroupHead(); gRenderFindings(); gRenderSections(); renderPeople(); renderIds(); renderMatrix(); renderSync(); renderLayout(); renderCons();
  setupGroupVideo(); gSetupTimeline();
  $$(".g-tabs button").forEach((b) => { b.onclick = () => { history.replaceState(null, "", `#/group/${id}${b.dataset.tab === "ids" ? "/ids" : ""}`); setTab(b.dataset.tab); }; });
  setTab(G.tab);
}
function setTab(tab) {
  G.tab = tab;
  $$(".g-tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  $("#g-people").classList.toggle("hidden", tab !== "people");
  $("#g-ids").classList.toggle("hidden", tab !== "ids");
}

function renderGroupHead() {
  const g = G.g, perf = g.identities.filter((i) => i.performer), done = g.persons.filter((p) => p.status === "done");
  $("#g-title").textContent = `${g.profile.title || g.reference.title} · 集体分析`;
  $("#g-sub").textContent = `${G.job.user_filename || g.video.filename} · 识别 ${g.identities.length} 人，练习者 ${perf.length} 人（完成比较 ${done.length} 人）· ${g.model.backend}（${g.model.device === "cpu" ? "CPU" : "GPU"}）`;
  const badges = $("#g-badges"); badges.replaceChildren();
  if (g.profile.status === "draft") badges.append(el("span", { class: "badge warn" }, "⚠ 草案规则 · 仅供算法验证"));
  const joins = g.identities.reduce((n, i) => n + i.joins.length, 0);
  if (joins) badges.append(el("button", { class: "badge", onclick: () => setTab("ids") }, `身份拼接点 ${joins} 处 · 去核对`));
  $("#g-ids-flag").textContent = joins ? `⚠ ${joins}` : "";
  const notes = g.persons.filter((p) => p.status === "error").map((p) => `#${p.id} 比较失败：${p.error}`);
  const box = $("#g-notes"); box.replaceChildren(...notes.map((n) => el("p", {}, n))); box.classList.toggle("hidden", !notes.length);
}

// ---------------------------------------------------------------- people list
function renderPeople() {
  const g = G.g, sy = (g.group && g.group.sync) || {};
  const ids = [...g.identities].sort((a, b) => (b.performer - a.performer) || a.id - b.id);
  const box = $("#g-people"); box.replaceChildren();
  for (const i of ids) {
    const p = personOf(i.id), lay = i.layout || {}, s = sy[i.id] || {};
    const c = (p && p.counts) || {}, ev = (c.good || 0) + (c.warn || 0) + (c.bad || 0);
    const stats = i.performer
      ? [p && p.status === "done" ? `达标 ${c.good || 0}/${ev}` : (p && p.reason) || "未完成比较", `同步 ${fmtS(s.offset_median_s)}`]
      : ["已排除"];
    if (lay.z_m) stats.push(`约 ${lay.z_m.toFixed(1)} m 远` + (lay.height_m ? ` · 身高约 ${lay.height_m.toFixed(2)} m` : ""));
    const role = i.performer ? { label: "排除", performer: false } : { label: "设为练习者", performer: true };
    box.append(el("div", { class: `g-person${i.performer ? "" : " out"}`, "data-id": i.id },
      el("div", { class: "g-person-head" },
        el("span", { class: "who-dot", style: `background:${i.performer ? idColor(i.id) : "transparent"};border-color:${idColor(i.id)}` }),
        el("b", {}, `#${i.id}`), el("span", { class: "hint" }, i.performer ? "练习者" : "排除")),
      el("div", { class: "g-person-stats" }, stats.join(" · ")),
      i.screening.reasons.length ? el("div", { class: "hint" }, i.screening.reasons.join("；")) : null,
      el("div", { class: "g-person-act" },
        i.report ? el("a", { class: "button", href: `#/job/${G.id}/p/${i.id}` }, "详细对比") : null,
        el("button", { onclick: () => queueEdit({ op: "role", id: i.id, performer: role.performer }, `#${i.id} ${role.label}`) }, role.label),
        el("button", { onclick: () => seekGroup((i.runs[0] || [0])[0]) }, "定位"))));
  }
  box.append(editBar());
}

// ---------------------------------------------------------------- identity review
function renderIds() {
  const g = G.g, dur = g.video.duration || 1, box = $("#g-ids");
  box.replaceChildren(el("p", { class: "hint" },
    "每条横条是一个身份被看到的时段；⚠ 为两段跟踪被拼接的位置，点击可跳到该处检查是否串人。勾选两个身份可合并，勾选一个可在当前时刻拆分。"));
  for (const i of [...g.identities].sort((a, b) => a.id - b.id)) {
    const bar = el("div", { class: "g-bar", title: "点击跳转" });
    bar.onclick = (e) => { const r = bar.getBoundingClientRect(); seekGroup(((e.clientX - r.left) / r.width) * dur); };
    for (const [a, b] of i.runs) bar.append(el("span", { class: "g-run", style: `left:${(a / dur) * 100}%;width:${Math.max(0.3, ((b - a) / dur) * 100)}%;background:${idColor(i.id)};opacity:${i.performer ? 1 : 0.45}` }));
    for (const t of i.joins) bar.append(el("button", { class: "g-join", style: `left:${(t / dur) * 100}%`, title: `拼接点 ${mmss(t)}`,
      onclick: (e) => { e.stopPropagation(); seekGroup(Math.max(0, t - 1.5)); } }, "⚠"));
    bar.append(el("span", { class: "g-cursor" }));
    const cb = el("input", { type: "checkbox", id: `g-pick-${i.id}`, onchange: (e) => { e.target.checked ? G.pick.add(i.id) : G.pick.delete(i.id); } });
    box.append(el("div", { class: `g-idrow${i.performer ? "" : " out"}` },
      el("label", { class: "g-idlab", for: `g-pick-${i.id}` }, cb, el("span", { class: "who-dot", style: `background:${idColor(i.id)}` }), `#${i.id}`),
      bar));
  }
  box.append(el("div", { class: "g-id-actions" },
    el("button", { onclick: () => {
      const ids = [...G.pick].sort((a, b) => a - b);
      if (ids.length !== 2) return note("合并需要恰好勾选两个身份。");
      queueEdit({ op: "merge", from: ids[1], into: ids[0] }, `#${ids[1]} 并入 #${ids[0]}`);
    } }, "合并所选（并入编号小的）"),
    el("button", { onclick: () => {
      const ids = [...G.pick];
      if (ids.length !== 1) return note("拆分需要恰好勾选一个身份。");
      const t = $("#g-video").currentTime;
      queueEdit({ op: "split", id: ids[0], at: +t.toFixed(1) }, `#${ids[0]} 在 ${mmss(t)} 拆分`);
    } }, "在当前时刻拆分所选")), editBar());
}
function note(msg) { $$(".g-edit-msg").forEach((e) => { e.textContent = msg; }); }
function queueEdit(edit, label) {
  G.edits.push({ edit, label });
  $$(".g-editbar").forEach((b) => b.replaceWith(editBar()));
}
function editBar() {
  const bar = el("div", { class: "g-editbar" + (G.edits.length ? " has" : "") });
  if (G.edits.length) {
    bar.append(el("div", {}, el("b", {}, "待应用的修改："), G.edits.map((e) => e.label).join("；")),
      el("div", { class: "g-person-act" },
        el("button", { class: "primary", onclick: applyEdits }, "应用并重新计算"),
        el("button", { onclick: () => { G.edits = []; $$(".g-editbar").forEach((b) => b.replaceWith(editBar())); } }, "撤销")));
  }
  bar.append(el("p", { class: "msg g-edit-msg" }));
  return bar;
}
async function applyEdits() {
  try {
    await api(`/api/jobs/${G.id}/identities`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(G.edits.map((e) => e.edit)) });
  } catch (e) { return note("无法应用：" + e.message); }
  G.edits = [];
  refreshTasks();
  openGroup(G.id);
}

// ---------------------------------------------------------------- person x section matrix
function renderMatrix() {
  const g = G.g, tb = $("#g-matrix"); tb.replaceChildren();
  const done = g.persons.filter((p) => p.status === "done");
  if (!done.length) { tb.append(el("tr", {}, el("td", { class: "hint" }, "没有完成比较的练习者"))); return; }
  const segs = new Map();
  for (const p of done) for (const s of p.segments) if (!segs.has(s.id)) segs.set(s.id, s);
  const cols = [...segs.values()].sort((a, b) => a.ref_start - b.ref_start);
  tb.append(el("tr", {}, el("th", {}, ""), ...cols.map((s) => el("th", { title: s.name }, segShort(s.name)))));
  for (const p of done.sort((a, b) => a.id - b.id)) {
    const tr = el("tr", {}, el("th", { class: "g-rowhead" }, el("span", { class: "who-dot", style: `background:${idColor(p.id)}` }), `#${p.id}`));
    for (const col of cols) {
      const s = p.segments.find((x) => x.id === col.id);
      if (!s) { tr.append(el("td", { class: "st-na", title: "该练习者的分析区间不含此段" }, "")); continue; }
      const c = s.counts, ev = (c.good || 0) + (c.warn || 0) + (c.bad || 0);
      const cls = c.bad ? "st-bad" : c.warn ? "st-warn" : ev ? "st-good" : c.imprecise ? "st-imprecise" : "st-unmeasurable";
      const tip = `#${p.id} · ${s.name}\n达标 ${c.good || 0}/${ev}` + (c.warn ? ` · 需注意 ${c.warn}` : "") + (c.bad ? ` · 偏差明显 ${c.bad}` : "")
        + (c.imprecise ? ` · 测量不确定 ${c.imprecise}` : "") + (c.unmeasurable ? ` · 不可评价 ${c.unmeasurable}` : "") + `\n检出 ${s.reps} 次重复`;
      tr.append(el("td", { class: cls, title: tip, onclick: () => { location.hash = `#/job/${G.id}/p/${p.id}@${s.ref_start}`; } },
        ev ? `${c.good || 0}/${ev}` : "—", c.imprecise ? el("small", {}, ` ±${c.imprecise}`) : null));
    }
    tb.append(tr);
  }
  const lg = $("#g-matrix-legend"); lg.replaceChildren();
  for (const [s, lab] of [["good", "全部达标"], ["warn", "有需注意"], ["bad", "有偏差明显"], ["imprecise", "仅测量不确定 ±"], ["unmeasurable", "不可评价"], ["na", "不在分析区间"]])
    lg.append(el("span", {}, el("i", { class: `st-${s}` }), lab));
  lg.append(el("span", {}, "格中数字 = 达标数/可评价数；点击打开该练习者在该段的详细对比"));
}

// ---------------------------------------------------------------- synchronisation (offset vs reference time)
function renderSync() {
  const box = $("#g-sync"); box.replaceChildren();
  const ser = G.g.group && G.g.group.sync_series;
  if (!ser || !ser.t.length || Object.keys(ser.offset).length < 2) { box.append(el("p", { class: "hint" }, "至少需要两位完成比较的练习者。")); return; }
  const W = 560, H = 240, m = { l: 44, r: 54, t: 10, b: 28 };
  const all = Object.values(ser.offset).flat().filter((v) => v !== null).map(Math.abs).sort((a, b) => a - b);
  const p90 = all[Math.floor(0.9 * (all.length - 1))] || 1;
  const lim = [1, 2, 5, 10, 20, 30, 60, 120].find((v) => v >= p90) || 120;   // beyond: clipped at the edge, see note
  const t0 = ser.t[0], t1 = ser.t[ser.t.length - 1];
  const X = (t) => m.l + ((t - t0) / Math.max(1e-6, t1 - t0)) * (W - m.l - m.r), Y = (v) => m.t + ((lim - Math.max(-lim, Math.min(lim, v))) / (2 * lim)) * (H - m.t - m.b);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "g-svg", role: "img", "aria-label": "各练习者相对集体的时间差" });
  for (const v of [-lim, -lim / 2, 0, lim / 2, lim]) {
    svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: v === 0 ? "g-zero" : "g-grid" }),
      svgEl("text", { x: m.l - 6, y: Y(v) + 4, class: "g-tick", "text-anchor": "end" }, `${v > 0 ? "+" : ""}${v % 1 ? v.toFixed(1) : v} s`));
  }
  for (const tt of niceTicks(t0, t1, 6)) svg.append(svgEl("text", { x: X(tt), y: H - 8, class: "g-tick", "text-anchor": "middle" }, `${Math.round(tt)} s`));
  const ends = [];
  for (const [id, vals] of Object.entries(ser.offset)) {
    let d = "", pen = false, last = null;
    vals.forEach((v, k) => { if (v === null) { pen = false; return; } d += `${pen ? "L" : "M"}${X(ser.t[k]).toFixed(1)},${Y(v).toFixed(1)}`; pen = true; last = [ser.t[k], v]; });
    svg.append(svgEl("path", { d, class: "g-line", stroke: idColor(+id) }));
    if (last) ends.push({ id: +id, y: Y(last[1]) });
  }
  ends.sort((a, b) => a.y - b.y).forEach((e, k, arr) => { if (k && e.y - arr[k - 1].y < 13) e.y = arr[k - 1].y + 13; });   // direct labels, no overlap
  for (const e of ends) svg.append(svgEl("text", { x: W - m.r + 6, y: e.y + 4, class: "g-lab", fill: css("--ink2") }, `#${e.id}`),
    svgEl("rect", { x: W - m.r + 1, y: e.y - 1, width: 4, height: 4, fill: idColor(e.id) }));
  // hover: crosshair + values
  const hair = svgEl("line", { y1: m.t, y2: H - m.b, class: "g-hair", visibility: "hidden" });
  svg.append(hair, svgEl("text", { x: m.l + 6, y: m.t + 10, class: "g-tick", "text-anchor": "start" }, "晚 ↑ / 早 ↓"));
  const tip = el("div", { class: "tip hidden" });
  const wrap = el("div", { class: "g-chart-wrap" }, svg, tip);
  svg.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect(), x = ((ev.clientX - r.left) / r.width) * W;
    const t = t0 + ((x - m.l) / (W - m.l - m.r)) * (t1 - t0);
    if (t < t0 || t > t1) { hair.setAttribute("visibility", "hidden"); tip.classList.add("hidden"); return; }
    const k = Math.round(((t - t0) / (t1 - t0)) * (ser.t.length - 1));
    hair.setAttribute("x1", X(ser.t[k])); hair.setAttribute("x2", X(ser.t[k])); hair.setAttribute("visibility", "visible");
    tip.replaceChildren(el("b", {}, `${ser.axis === "video" ? "视频" : "标准"} ${ser.t[k].toFixed(0)} s`), ...Object.entries(ser.offset).map(([id, v]) =>
      el("div", {}, el("span", { class: "who-dot", style: `background:${idColor(+id)}` }), `#${id}  ${fmtS(v[k])}`)));
    tip.classList.remove("hidden");
    tip.style.left = `${Math.min(r.width - 140, (X(ser.t[k]) / W) * r.width + 10)}px`; tip.style.top = "8px";
  });
  svg.addEventListener("pointerleave", () => { hair.setAttribute("visibility", "hidden"); tip.classList.add("hidden"); });
  box.append(wrap);
  const over = Object.entries(ser.offset).filter(([, v]) => v.some((x) => x !== null && Math.abs(x) > lim)).map(([id]) => `#${id}`);
  if (over.length) box.append(el("p", { class: "hint" }, `${over.join("、")} 的时间差超出 ±${lim} s，超出部分画在边缘；悬停查看数值。`));
}
function niceTicks(a, b, n) {
  const step = [5, 10, 15, 30, 60, 120, 300].find((s) => (b - a) / s <= n) || 600;
  const out = [];
  for (let t = Math.ceil(a / step) * step; t <= b; t += step) out.push(t);
  return out;
}

// ---------------------------------------------------------------- floor layout (top view)
function renderLayout() {
  const box = $("#g-layout"); box.replaceChildren();
  const pts = G.g.identities.filter((i) => i.layout && i.layout.z_m);
  if (!pts.length) { box.append(el("p", { class: "hint" }, "无法估计站位。")); return; }
  const W = 560, H = 300, m = 34;
  const xs = pts.map((i) => i.layout.x_m), zs = pts.map((i) => i.layout.z_m);
  const xr = Math.max(1.5, ...xs.map(Math.abs)) * 1.15, zmax = Math.max(...zs) * 1.12;
  const X = (x) => W / 2 + (x / xr) * (W / 2 - m), Y = (z) => H - m - (z / zmax) * (H - 2 * m);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "g-svg", role: "img", "aria-label": "站位俯视示意" });
  for (let z = 1; z < zmax; z += zmax > 12 ? 4 : 2) svg.append(svgEl("line", { x1: m, x2: W - m, y1: Y(z), y2: Y(z), class: "g-grid" }),
    svgEl("text", { x: m - 4, y: Y(z) + 4, class: "g-tick", "text-anchor": "end" }, `${z} m`));
  svg.append(svgEl("path", { d: `M${W / 2 - 7},${H - 6} L${W / 2 + 7},${H - 6} L${W / 2},${H - 18} Z`, fill: css("--ink2") }),
    svgEl("text", { x: W / 2 + 12, y: H - 6, class: "g-tick" }, "镜头"));
  const placed = [];
  const spot = (x, y) => {                                   // label above, below, right or left: the first free one
    for (const [dx, dy, a] of [[0, -14, "middle"], [0, 22, "middle"], [14, 5, "start"], [-14, 5, "end"]]) {
      const bx = a === "middle" ? x + dx - 14 : a === "start" ? x + dx : x + dx - 28, by = y + dy - 11;
      if (!placed.some((q) => Math.abs(q[0] - bx) < 30 && Math.abs(q[1] - by) < 14)) { placed.push([bx, by]); return [x + dx, y + dy, a]; }
    }
    return [x, y - 14, "middle"];
  };
  for (const i of [...pts].sort((a, b) => b.performer - a.performer)) {
    const c = idColor(i.id), x = X(i.layout.x_m), y = Y(i.layout.z_m);
    const g = svgEl("g", { class: "g-dot" }, svgEl("title", {}, `#${i.id} ${i.performer ? "练习者" : "排除"} · 横向 ${i.layout.x_m} m · 距镜头 ${i.layout.z_m} m`
      + (i.layout.height_m ? ` · 身高约 ${i.layout.height_m} m` : "")));
    g.append(svgEl("circle", { cx: x, cy: y, r: 9, fill: i.performer ? c : css("--card"), stroke: c, "stroke-width": 2, "stroke-dasharray": i.performer ? null : "3 2" }),
      (([lx, ly, a]) => svgEl("text", { x: lx, y: ly, class: "g-lab", "text-anchor": a, fill: css("--ink") }, `#${i.id}`))(spot(x, y)));
    svg.append(g);
  }
  box.append(svg);
}

// ---------------------------------------------------------------- consistency heat (section x indicator)
function renderCons() {
  const tb = $("#g-cons"); tb.replaceChildren();
  const cons = (G.g.group && G.g.group.consistency) || {};
  const segs = new Map();
  for (const p of G.g.persons.filter((x) => x.status === "done")) for (const s of p.segments) if (!segs.has(s.id)) segs.set(s.id, s);
  const cols = [...segs.values()].sort((a, b) => a.ref_start - b.ref_start).filter((s) => cons[s.id] && Object.keys(cons[s.id]).length);
  if (!cols.length) { tb.append(el("tr", {}, el("td", { class: "hint" }, "至少需要两位练习者覆盖同一段。"))); return; }
  tb.append(el("tr", {}, el("th", {}, ""), ...cols.map((s) => el("th", { title: s.name }, segShort(s.name)))));
  for (const f of G.g.features) {
    const tr = el("tr", {}, el("th", { class: "g-rowhead" }, f.label));
    for (const s of cols) {
      const v = cons[s.id][f.name];
      const a = v === undefined ? 0 : Math.min(1, v / 1.5);        // 1.5 tolerance units = darkest
      tr.append(el("td", { class: "g-cons-cell", title: v === undefined ? "少于两人可比" : `${s.name} · ${f.label}：离散 ${v.toFixed(2)} 个允许偏差`,
        style: v === undefined ? "" : `background:color-mix(in srgb, var(--s1) ${Math.round(15 + 70 * a)}%, var(--card));color:${a > 0.6 ? "#fff" : "inherit"}` },
        v === undefined ? "" : v.toFixed(1)));
    }
    tb.append(tr);
  }
}

// ---------------------------------------------------------------- video with identity boxes
function setupGroupVideo() {
  const v = $("#g-video"), g = G.g, [fw, fh] = g.video.frame_size;
  $("#g-stage").style.aspectRatio = `${fw} / ${fh}`;
  if (v.getAttribute("src") !== g.video.video_url) v.src = g.video.video_url;
  $("#g-speed").onchange = (e) => { v.playbackRate = +e.target.value; };
  $("#g-play").onclick = () => (v.paused ? v.play() : v.pause());
  v.onplay = () => { $("#g-play").textContent = "❚❚ 暂停"; };
  v.onpause = () => { $("#g-play").textContent = "▶ 播放"; };
  $("#g-show-out").onchange = drawGroup;
  $("#g-canvas").onclick = (e) => {
    const hit = boxesAt(v.currentTime).find((b) => inside(b, e));
    if (!hit) return;
    const i = identity(hit.id);
    if (i && i.report) location.hash = `#/job/${G.id}/p/${i.id}`;
    else { setTab("people"); const row = $(`.g-person[data-id="${hit.id}"]`); if (row) { row.scrollIntoView({ block: "nearest" }); row.classList.add("flash"); setTimeout(() => row.classList.remove("flash"), 900); } }
  };
  cancelAnimationFrame(G.raf);
  const loop = () => { if (!$("#view-group").classList.contains("hidden")) { drawGroup(); gDrawTimeline(); gRenderNow(); G.raf = requestAnimationFrame(loop); } };
  G.raf = requestAnimationFrame(loop);
}
function seekGroup(t) { const v = $("#g-video"); v.currentTime = Math.max(0, t); }
function displayRect() {
  const c = $("#g-canvas"), [fw, fh] = G.g.video.frame_size;
  const W = c.clientWidth, H = c.clientHeight, s = Math.min(W / fw, H / fh);
  return { s, ox: (W - fw * s) / 2, oy: (H - fh * s) / 2, W, H };
}
function boxesAt(t) {
  const out = [], showOut = $("#g-show-out").checked;
  for (const [id, tr] of Object.entries(G.g.tracks)) {
    const i = identity(+id);
    if (!i || (!i.performer && !showOut) || !tr.t.length) continue;
    let lo = 0, hi = tr.t.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (tr.t[mid] < t) lo = mid + 1; else hi = mid; }
    const k = lo > 0 && Math.abs(tr.t[lo - 1] - t) < Math.abs(tr.t[lo] - t) ? lo - 1 : lo;
    if (Math.abs(tr.t[k] - t) <= 0.35) out.push({ id: +id, b: tr.b[k], performer: i.performer });
  }
  return out;
}
function inside(b, e) {
  const r = displayRect(), c = $("#g-canvas").getBoundingClientRect(), x = e.clientX - c.left, y = e.clientY - c.top;
  return x >= r.ox + b.b[0] * r.s && x <= r.ox + b.b[2] * r.s && y >= r.oy + b.b[1] * r.s && y <= r.oy + b.b[3] * r.s;
}
function drawGroup() {
  const v = $("#g-video"), c = $("#g-canvas");
  const dpr = window.devicePixelRatio || 1, r = displayRect();
  if (c.width !== Math.round(r.W * dpr) || c.height !== Math.round(r.H * dpr)) { c.width = Math.round(r.W * dpr); c.height = Math.round(r.H * dpr); }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.W, r.H);
  const t = v.currentTime || 0;
  $("#g-time").textContent = `${mmss(t)} / ${mmss(G.g.video.duration || 0)}`;
  $$(".g-cursor").forEach((e) => { e.style.left = `${(t / (G.g.video.duration || 1)) * 100}%`; });
  ctx.font = "600 13px system-ui, sans-serif";
  for (const b of boxesAt(t)) {
    const col = idColor(b.id), x = r.ox + b.b[0] * r.s, y = r.oy + b.b[1] * r.s, w = (b.b[2] - b.b[0]) * r.s, h = (b.b[3] - b.b[1]) * r.s;
    ctx.setLineDash(b.performer ? [] : [5, 4]);
    ctx.lineWidth = 2; ctx.strokeStyle = col;
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    const label = `#${b.id}${b.performer ? "" : " 排除"}`, tw = ctx.measureText(label).width + 10;
    ctx.fillStyle = col;
    ctx.fillRect(x, Math.max(0, y - 20), tw, 20);
    ctx.fillStyle = "#fff"; ctx.fillText(label, x + 5, Math.max(14, y - 6));
  }
}


// ================================================================ timeline workbench (group indicators over video time)
const GTL = { gutter: 74, rowH: 13, ruler: 18, band: 20, curve: 56, sync: 34, gap: 6 };
const tlData = () => (G.g.group && G.g.group.timeline) || null;
const curPerf = () => G.g.identities.filter((i) => i.performer && i.report).sort((a, b) => a.id - b.id);
const nearestK = (arr, t) => { let lo = 0, hi = arr.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < t) lo = m + 1; else hi = m; } return lo; };
const gSegName = (sid) => { const p = G.g.persons.find((x) => (x.segments || []).some((s) => s.id === sid)); const s = p && p.segments.find((x) => x.id === sid); return s ? s.name : sid; };

/** sync spread across performers per window (standard deviation of their offsets) */
function syncSpread() {
  const ser = G.g.group && G.g.group.sync_series;
  if (!ser) return null;
  const ids = Object.keys(ser.offset);
  return { t: ser.t, v: ser.t.map((_, k) => {
    const xs = ids.map((i) => ser.offset[i][k]).filter((x) => x !== null);
    if (xs.length < 2) return null;
    const m = xs.reduce((a, b) => a + b, 0) / xs.length;
    return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
  }) };
}

function gSetupTimeline() {
  const c = $("#g-tl-canvas"), tl = tlData();
  G.nowK = -1;
  if (!tl) { $("#g-tl").replaceChildren(el("p", { class: "hint" }, "该记录生成于旧版本，没有时间轴数据；请重新分析。")); return; }
  G.tl = { spread: syncSpread(), rows: curPerf() };
  $("#g-tl-rows-n").textContent = G.tl.rows.length;
  const rowsBox = $("#g-tl-rows");
  rowsBox.onchange = () => { gTimelineHeight(); gDrawTimeline(); };
  gTimelineHeight();
  const toT = (e) => { const r = c.getBoundingClientRect(); const x = e.clientX - r.left - GTL.gutter;
    return Math.max(0, Math.min(1, x / (r.width - GTL.gutter))) * (G.g.video.duration || 1); };
  let drag = false;
  c.onpointerdown = (e) => { drag = true; c.setPointerCapture(e.pointerId); seekGroup(toT(e)); };
  c.onpointermove = (e) => { if (drag) seekGroup(toT(e)); timelineTip(e, toT(e)); };
  c.onpointerup = () => { drag = false; };
  c.onpointerleave = () => $("#g-tl-tip").classList.add("hidden");
}
/** per-performer rows only when expanded: with many people they would push everything else off the screen */
const gRowsOpen = () => $("#g-tl-rows").checked;
function gTimelineHeight() {
  const rows = gRowsOpen() ? G.tl.rows.length * GTL.rowH : 0;
  $("#g-tl-canvas").style.height = `${GTL.ruler + GTL.band + GTL.gap + GTL.curve + GTL.gap + GTL.sync + GTL.gap + rows + 4}px`;
}
function timelineTip(e, t) {
  const tl = tlData(), tip = $("#g-tl-tip"), k = nearestK(tl.t, t), c = $("#g-tl-canvas").getBoundingClientRect();
  const y = e.clientY - c.top, rows0 = GTL.ruler + GTL.band + GTL.gap + GTL.curve + GTL.gap + GTL.sync + GTL.gap;
  const lines = [el("b", {}, `${mmss(t)} · ${tl.section[k] ? gSegName(tl.section[k]) : "分析区间外"}`),
    el("div", {}, `整齐度（离散）${tl.consistency[k] ?? "—"}`)];
  if (gRowsOpen() && y >= rows0) {
    const row = G.tl.rows[Math.floor((y - rows0) / GTL.rowH)];
    if (row) lines.push(el("div", {}, `#${row.id} 偏离集体 ${tl.deviation[row.id][k] ?? "—"}`));
  }
  tip.replaceChildren(...lines);
  tip.classList.remove("hidden");
  tip.style.left = `${Math.min(c.width - 180, e.clientX - c.left + 12)}px`; tip.style.top = `${Math.max(0, y - 10)}px`;
}
function gDrawTimeline() {
  const tl = tlData(), c = $("#g-tl-canvas");
  if (!tl || !G.tl) return;
  const dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
  if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const dur = G.g.video.duration || 1, x0 = GTL.gutter, pw = W - x0, X = (t) => x0 + (t / dur) * pw;
  const ink2 = css("--ink2"), muted = css("--muted"), line = css("--line");
  ctx.font = "11px system-ui, sans-serif"; ctx.textBaseline = "middle";
  // ruler
  for (const t of niceTicks(0, dur, 10)) { ctx.fillStyle = muted; ctx.fillText(`${Math.round(t)}s`, X(t) + 2, 9); ctx.fillStyle = line; ctx.fillRect(X(t), 14, 1, 4); }
  let y = GTL.ruler;
  // sections (common axis)
  const bounds = (G.g.common_axis && G.g.common_axis.section_bounds) || {};
  ctx.fillStyle = ink2; ctx.fillText("动作段", 4, y + GTL.band / 2);
  for (const [sid, [a, b]] of Object.entries(bounds)) {
    ctx.fillStyle = css("--tl-bg") || line; ctx.fillRect(X(a), y + 2, Math.max(1, X(b) - X(a) - 2), GTL.band - 4);
    ctx.strokeStyle = css("--axis"); ctx.strokeRect(X(a) + 0.5, y + 2.5, Math.max(1, X(b) - X(a) - 3), GTL.band - 5);
    ctx.fillStyle = css("--ink"); ctx.save(); ctx.beginPath(); ctx.rect(X(a), y, X(b) - X(a), GTL.band); ctx.clip();
    ctx.fillText(gSegName(sid), X(a) + 5, y + GTL.band / 2); ctx.restore();
  }
  y += GTL.band + GTL.gap;
  // consistency curve (spread across performers, tolerance units; dashed line = one tolerance)
  const cv = tl.consistency, top = Math.max(1.2, ...cv.filter((v) => v !== null).sort((a, b) => a - b).slice(-Math.ceil(cv.length * 0.02) - 1, -1));
  const Yc = (v) => y + GTL.curve - (Math.min(v, top) / top) * (GTL.curve - 4);
  ctx.fillStyle = ink2; ctx.fillText("整齐度", 4, y + 12); ctx.fillStyle = muted; ctx.fillText("离散↑", 4, y + 26);
  ctx.strokeStyle = line; ctx.beginPath(); ctx.moveTo(x0, y + GTL.curve); ctx.lineTo(W, y + GTL.curve); ctx.stroke();
  ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(x0, Yc(1)); ctx.lineTo(W, Yc(1)); ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath(); let pen = false;
  tl.t.forEach((t, k) => { const v = cv[k]; if (v === null) { pen = false; return; } pen ? ctx.lineTo(X(t), Yc(v)) : ctx.moveTo(X(t), Yc(v)); pen = true; });
  ctx.strokeStyle = css("--s1"); ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1;
  y += GTL.curve + GTL.gap;
  // sync spread (seconds)
  const sp = G.tl.spread;
  ctx.fillStyle = ink2; ctx.fillText("同步离散", 4, y + GTL.sync / 2);
  ctx.strokeStyle = line; ctx.beginPath(); ctx.moveTo(x0, y + GTL.sync); ctx.lineTo(W, y + GTL.sync); ctx.stroke();
  if (sp) {
    const smax = Math.max(1, ...sp.v.filter((v) => v !== null));
    ctx.fillStyle = css("--s2");
    sp.t.forEach((t, k) => { const v = sp.v[k]; if (v === null) return; const h = (v / smax) * (GTL.sync - 4); ctx.fillRect(X(t) - 1.5, y + GTL.sync - h, 3, h); });
    ctx.fillStyle = muted; ctx.fillText(`${smax.toFixed(1)}s`, W - 34, y + 7);
  }
  y += GTL.sync + GTL.gap;
  // per-performer deviation rows (sequential: deeper = further from the group)
  const bad = css("--bad"), card = css("--card");
  (gRowsOpen() ? G.tl.rows : []).forEach((i, r) => {
    const yy = y + r * GTL.rowH, dv = tl.deviation[i.id] || [];
    ctx.fillStyle = idColor(i.id); ctx.fillRect(4, yy + 2, 7, GTL.rowH - 4);
    ctx.fillStyle = ink2; ctx.fillText(`#${i.id}`, 14, yy + GTL.rowH / 2);
    tl.t.forEach((t, k) => {
      const v = dv[k];
      if (v === null || v === undefined) return;
      const a = Math.min(1, v / 1.5);
      ctx.fillStyle = `color-mix(in srgb, ${bad} ${Math.round(8 + 85 * a)}%, ${card})`;
      ctx.fillRect(X(t), yy + 1, Math.max(1, pw / tl.t.length + 0.5), GTL.rowH - 2);
    });
  });
  // playhead
  const t = $("#g-video").currentTime || 0;
  ctx.fillStyle = css("--ink"); ctx.fillRect(X(t) - 1, 0, 2, H);
}

/** the current moment: section, uniformity, the least uniform indicators, who deviates most, sync */
function gRenderNow() {
  const tl = tlData(), box = $("#g-now");
  if (!tl) { box.replaceChildren(); return; }
  const t = $("#g-video").currentTime || 0, k = nearestK(tl.t, t);
  if (k === G.nowK) return;
  G.nowK = k;
  const feats = G.g.features.map((f) => ({ ...f, v: (tl.spread[f.name] || [])[k] })).filter((f) => f.v !== null && f.v !== undefined)
    .sort((a, b) => b.v - a.v).slice(0, 5);
  const fmax = Math.max(1.5, ...feats.map((f) => f.v));
  const people = curPerf().map((i) => ({ i, v: (tl.deviation[i.id] || [])[k] })).filter((x) => x.v !== null && x.v !== undefined)
    .sort((a, b) => b.v - a.v);
  const ser = G.g.group.sync_series, sk = ser ? nearestK(ser.t, t) : -1;
  box.replaceChildren(
    el("h2", {}, "当前时刻 ", el("span", { class: "hint mono" }, `${mmss(t)} · ${tl.section[k] ? gSegName(tl.section[k]) : "分析区间外"}`)),
    el("div", { class: "g-now-big" }, el("b", {}, tl.consistency[k] === null ? "—" : tl.consistency[k].toFixed(2)),
      el("span", {}, " 个允许偏差：同一时刻练习者之间的典型差异（越小越整齐）"), el("div", { class: "hint" }, `可比人数 ${tl.seen[k]}`)),
    el("h3", {}, "最不整齐的指标"),
    ...feats.map((f) => el("div", { class: "g-bar-row" }, el("span", {}, f.label),
      el("span", { class: "g-hbar" }, el("i", { style: `width:${Math.round((f.v / fmax) * 100)}%` })), el("span", { class: "mono" }, f.v.toFixed(2)))),
    el("h3", {}, "此刻偏离集体最多"),
    el("div", { class: "g-now-people" }, ...people.slice(0, 6).map(({ i, v }) => el("a", { class: "chip", href: `#/job/${G.id}/p/${i.id}` },
      el("i", { style: `background:${idColor(i.id)}` }), `#${i.id} ${v.toFixed(2)}` +
      (ser && sk >= 0 && ser.offset[i.id] && ser.offset[i.id][sk] !== null ? ` · ${fmtS(ser.offset[i.id][sk])}` : "")))),
    el("p", { class: "hint" }, "偏离 = 个人各指标与集体中位之差（允许偏差为单位，平均）；s = 相对集体的早晚。点击打开该练习者的详细对比。"));
}

/** cards: the least uniform moments (peaks of the spread, one per 10 s, inside the analysed sections) */
function gRenderFindings() {
  const tl = tlData(), box = $("#g-findings"), sum = $("#g-summary");
  box.replaceChildren(); sum.replaceChildren();
  if (!tl) return;
  const cv = tl.consistency, cand = [];
  for (let k = 0; k < cv.length; k++) {
    if (cv[k] === null || !tl.section[k]) continue;
    let peak = true;
    for (let j = Math.max(0, k - 20); j <= Math.min(cv.length - 1, k + 20); j++) if (cv[j] !== null && cv[j] > cv[k]) { peak = false; break; }
    if (peak) cand.push(k);
  }
  cand.sort((a, b) => cv[b] - cv[a]);
  cand.slice(0, 3).forEach((k, n) => {
    const f = G.g.features.map((x) => ({ ...x, v: (tl.spread[x.name] || [])[k] })).filter((x) => x.v !== null).sort((a, b) => b.v - a.v)[0];
    box.append(el("button", { class: "finding warn", onclick: () => seekGroup(tl.t[k]) },
      el("div", { class: "rank" }, el("b", {}, String(n + 1)), `${gSegName(tl.section[k])} · ${mmss(tl.t[k])}`),
      el("div", { class: "what" }, "最不整齐的时刻"),
      el("div", { class: "num" }, cv[k].toFixed(2), el("small", {}, "个允许偏差")),
      el("div", { class: "how" }, f ? `差异最大的是${f.label}（${f.v.toFixed(2)}）` : "")));
  });
  if (!cand.length) box.append(el("div", { class: "finding" }, el("div", { class: "what" }, "没有可比较的时刻")));
  const perSec = {};
  tl.t.forEach((_, k) => { const s = tl.section[k]; if (s && cv[k] !== null) (perSec[s] = perSec[s] || []).push(cv[k]); });
  const med = (a) => { const b = [...a].sort((x, y) => x - y); return b[Math.floor(b.length / 2)]; };
  const sy = Object.values((G.g.group && G.g.group.sync) || {}).map((x) => x.offset_median_s).filter((x) => x !== null).map(Math.abs);
  sum.append(el("div", {}, el("span", { class: "big" }, String(curPerf().length)), " 位练习者参与比较"),
    ...Object.entries(perSec).map(([s, a]) => el("div", { class: "row" }, `${gSegName(s)} 整齐度`, el("b", {}, med(a).toFixed(2)))),
    el("div", { class: "row" }, "同步偏差（中位）", el("b", {}, sy.length ? `${med(sy).toFixed(1)} s` : "—")),
    el("div", { class: "row", title: "整齐度 = 同一时刻练习者指标的标准差 / 允许偏差" }, "说明", el("b", {}, "数值越小越整齐")));
}


// ================================================================ "修正动作段": the group's common sections, editable
async function gRenderSections() {
  const box = $("#g-sections"), ax = G.g.common_axis || {};
  const refs = await api("/api/references").catch(() => []);
  const all = ((refs.find((r) => r.id === G.g.reference.id) || {}).segments || []);
  const how = { vote: `由 ${(ax.voters || []).length} 位画面最好的练习者各自分段，多数认定`, manual: "人工指定",
    consensus: "集体共识骨架（备用方法）" }[ax.method] || "未能确定";
  const rows = Object.entries(ax.section_bounds || {}).map(([sid, [a, b]]) => ({ sid, a, b }));
  const list = el("div", { class: "g-sec-list" });
  const draw = () => {
    list.replaceChildren(...rows.map((r, k) => el("div", { class: "fix-row" },
      el("b", { class: "g-sec-name" }, (all.find((x) => x.id === r.sid) || { name: r.sid }).name),
      el("label", {}, "起 ", el("input", { type: "number", step: "0.5", min: "0", value: (+r.a).toFixed(1), oninput: (e) => { r.a = +e.target.value; } })),
      el("label", {}, "止 ", el("input", { type: "number", step: "0.5", min: "0", value: (+r.b).toFixed(1), oninput: (e) => { r.b = +e.target.value; } })),
      el("button", { onclick: () => { const t = +$("#g-video").currentTime.toFixed(1); r.a = t; draw(); } }, "起=当前"),
      el("button", { onclick: () => { const t = +$("#g-video").currentTime.toFixed(1); r.b = t; draw(); } }, "止=当前"),
      el("button", { onclick: () => { rows.splice(k, 1); draw(); } }, "删除"))));
  };
  draw();
  const add = el("select", {}, ...all.filter((x) => !rows.some((r) => r.sid === x.id)).map((x) => el("option", { value: x.id }, x.name)));
  const msg = el("p", { class: "msg" });
  const post = async (sections) => {
    try {
      await api(`/api/jobs/${G.id}/sections`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sections }) });
      refreshTasks(); openGroup(G.id);
    } catch (e) { msg.textContent = "无法应用：" + e.message; }
  };
  box.replaceChildren(
    el("p", { class: "hint" }, `当前分段：${how}。${ax.votes ? "各段得票：" + Object.entries(ax.votes).filter(([, v]) => v).map(([k, v]) => `${(all.find((x) => x.id === k) || { name: k }).name} ${v}`).join("，") + "。" : ""}`
      + "如果某段识别错了或漏掉了，可在此修改起止（视频秒数）、删除或添加，然后重新计算；所有练习者都按修改后的分段比较。"),
    list,
    el("div", { class: "fix-row" }, "添加动作段 ", add, el("button", { onclick: () => {
      if (!add.value) return;
      const t = +$("#g-video").currentTime.toFixed(1);
      rows.push({ sid: add.value, a: t, b: Math.min(G.g.video.duration, t + 20) });
      rows.sort((x, y) => all.findIndex((s) => s.id === x.sid) - all.findIndex((s) => s.id === y.sid));
      add.querySelector(`option[value="${add.value}"]`).remove(); draw();
    } }, "从当前时刻添加")),
    el("div", { class: "fix-row" },
      el("button", { class: "primary", onclick: () => {
        if (rows.some((r) => !(r.b > r.a))) { msg.textContent = "每段的止应晚于起。"; return; }
        post(Object.fromEntries(rows.map((r) => [r.sid, [r.a, r.b]])));
      } }, "应用并重新计算"),
      ax.method === "manual" ? el("button", { onclick: () => post(null) }, "恢复自动分段") : null),
    msg);
}
