"use strict";
// motioncoach client: new analysis -> progress -> report workbench (timeline-driven) + developer bench.

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids) if (k !== null && k !== undefined && k !== false) e.append(k instanceof Node ? k : document.createTextNode(k));
  return e;
};
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const pct = (x) => (x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const JOINT_CN = { nose: "头", left_shoulder: "左肩", right_shoulder: "右肩", left_elbow: "左肘", right_elbow: "右肘",
  left_wrist: "左腕", right_wrist: "右腕", left_hip: "左髋", right_hip: "右髋", left_knee: "左膝", right_knee: "右膝",
  left_ankle: "左踝", right_ankle: "右踝" };
const STAGES = [["check", "检查画面中是否有人"], ["ingest", "视频规范化（旋转、编码）"], ["locate", "全片姿态识别 + 定位共同动作区间"], ["poses_ref", "标准视频姿态（区间内）"],
  ["poses_user", "练习者视频姿态（区间内）"], ["features", "关节特征"], ["align", "动作时间对应 + 左右判断"], ["compare", "逐关节比较、骨架归一化、差异聚类"]];
const SERIES = ["--p1", "--p2", "--p3", "--p4", "--p5", "--p6"];   // pattern hues avoid the identity pair and status colours
const STATUS_VAR = { good: "--good", warn: "--warn", bad: "--bad", unmeasurable: "--unm", missing: "--unm", na: "--na", uncertain: "--unc", imprecise: "--imp", out: "--track" };
const STATUS_GLYPH = { good: "✓", warn: "!", bad: "!!", unmeasurable: "?", na: "—", uncertain: "≈", imprecise: "±", missing: "?" };
const confidence = (c) => (c < 0.7 ? "高" : c < 1.2 ? "中" : "低");
const CFG = { dev_mode: false, default_backend: "mediapipe-full", default_backend_label: "", default_height_cm: 163.9 };

// ------------------------------------------------------------------ routing
function show(id) {
  // leaving a page stops its playback: returning to the group overview or the record list starts paused
  $$(".view").forEach((v) => { if (v.id !== id) v.querySelectorAll("video").forEach((x) => x.pause()); });
  $$(".view").forEach((v) => v.classList.toggle("hidden", v.id !== id));
  const nav = { "view-bench": "bench", "view-jobs": "jobs", "view-report": "jobs", "view-group": "jobs" }[id] || "new";
  $$(".topbar nav a").forEach((a) => a.classList.toggle("on", a.dataset.nav === nav));
}
async function route() {
  const h = location.hash;
  let m;
  if ((m = h.match(/^#\/job\/([0-9a-z_]{6,40})(?:\/p\/(\d+))?(?:@([0-9.]+))?$/))) { R.deepT = m[3] ? +m[3] : null; return openJob(m[1], m[2] === undefined ? null : +m[2]); }
  if ((m = h.match(/^#\/group\/([0-9a-z_]{6,40})(\/ids)?$/))) return openGroup(m[1], m[2] ? "ids" : "people");
  if ((m = h.match(/^#\/bench(?:\/([A-Za-z0-9_.-]+))?$/))) return openBench(m[1]);
  if (h === "#/jobs") return openJobs();
  show("view-new");
  initNew();
}
window.addEventListener("hashchange", route);

// ------------------------------------------------------------------ new analysis
const form = { ref: null, upload: null, local: null, refs: [] };
let newInit = false;
function fillSegSelects() {
  const r = form.refs.find((x) => x.id === form.ref);
  for (const sel of [$("#seg-from"), $("#seg-to")]) sel.replaceChildren(...(r ? r.segments : []).map((s) => el("option", { value: s.id }, s.name)));
  if (r) $("#seg-to").selectedIndex = Math.min(1, r.segments.length - 1);
}
async function initNew() {
  loadJobs();
  if (newInit) return;
  newInit = true;
  form.refs = await api("/api/references");
  const list = $("#ref-list");
  form.refs.forEach((r, i) => {
    const item = el("div", { class: "ref-item" + (i === 0 ? " sel" : "") },
      el("div", {}, r.title, r.status === "draft" ? el("span", { class: "badge warn" }, "草案") : null),
      el("div", { class: "meta" }, `${r.profile_title} · ${r.view === "front" ? "正面" : r.view} · ${r.segments.length} 段` + (r.available ? "" : " · 视频缺失")));
    item.onclick = () => { form.ref = r.id; list.querySelectorAll(".ref-item").forEach((x) => x.classList.remove("sel")); item.classList.add("sel"); fillSegSelects(); check(); };
    list.append(item);
    if (i === 0) form.ref = r.id;
  });
  fillSegSelects();
  $("#backend-default").textContent = `姿态识别：${CFG.default_backend_label || CFG.default_backend}（服务器默认`
    + (CFG.bounds_backend ? `；动作段边界用 ${CFG.bounds_backend}` : "") + "）。"
    + "长度按身体比例显示；填写本人身高后另显示估算的厘米值。";
  if (CFG.dev_mode) {
    const backends = await api("/api/backends");
    const bs = $("#backend");
    for (const b of backends) bs.append(el("option", { value: b.name, disabled: !b.available },
      `${b.label}（${b.dims}，${b.license}）${b.available ? "" : " — " + (b.status === "candidate" ? "候选" : "不可用")}`));
    bs.value = CFG.default_backend;
    const note = () => { const b = backends.find((x) => x.name === bs.value); $("#backend-note").textContent = b ? [b.note, "GPU：" + b.gpu, b.reason].filter(Boolean).join("；") : ""; };
    bs.onchange = note; note();
  }
  const locals = await api("/api/local-videos");
  const ll = $("#local-list");
  for (const v of locals) {
    const b = el("button", { title: v.name }, `${v.name}  (${v.size_mb} MB)`);
    b.onclick = () => { form.local = v.name; form.upload = null; ll.querySelectorAll("button").forEach((x) => x.classList.remove("sel")); b.classList.add("sel"); $("#drop-text").textContent = "已选：" + v.name; $("#drop").classList.add("has"); check(); };
    ll.append(b);
  }
  const drop = $("#drop");
  drop.addEventListener("dragover", (e) => e.preventDefault());
  drop.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) doUpload(e.dataTransfer.files[0]); });
  $("#file").onchange = (e) => e.target.files[0] && doUpload(e.target.files[0]);
  $("#start").onclick = start;
  check();
}
function check() { $("#start").disabled = !(form.ref && (form.upload || form.local || form.uploading)); }
function uploadFile(file, onProgress) {   // XHR (not fetch) for upload progress
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", "/api/uploads");
    x.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total, e.loaded, e.total); };
    x.onload = () => {
      let d = {}; try { d = JSON.parse(x.responseText); } catch { /* non-JSON error page */ }
      if (x.status >= 200 && x.status < 300) resolve(d); else reject(new Error(d.detail || `${x.status} ${x.statusText}`));
    };
    x.onerror = () => reject(new Error("网络中断"));
    const fd = new FormData(); fd.append("file", file); x.send(fd);
  });
}
// rough live speed: bytes over the last ~5 s; "stalled" when nothing moved for 10 s
const fmtMB = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(b >= 1e8 ? 0 : 1)} MB`);
function uploadStatus(t) {
  const now = Date.now(), pct = `${Math.round(t.frac * 100)}%`;
  if (!t.samples || t.samples.length < 2) return `${pct} · 正在建立连接…`;
  const last = t.samples[t.samples.length - 1];
  if (now - last.t > 10000) return `${pct} · 已 ${Math.round((now - last.t) / 1000)} 秒没有进度：网络可能中断，或页面被锁屏 / 切到后台挂起`;
  const first = t.samples.find((x) => now - x.t <= 5000) || t.samples[0];
  const speed = (last.b - first.b) / Math.max(0.5, (last.t - first.t) / 1000);   // bytes / s
  const left = speed > 0 ? (t.total - last.b) / speed : Infinity;
  const eta = !isFinite(left) ? "" : left < 60 ? ` · 约 ${Math.max(1, Math.round(left))} 秒` : ` · 约 ${Math.round(left / 60)} 分钟`;
  const slow = speed < 0.3e6 ? " · 网络较慢" : "";
  return `${pct}（${fmtMB(last.b)} / ${fmtMB(t.total)}）· ${(speed / 1e6).toFixed(speed < 1e6 ? 2 : 1)} MB/s${eta}${slow}`;
}
// non-blocking advice from the file's own metadata (docs/VIDEO_SPEC.md §2.1); the upload goes ahead regardless
function adviseFile(file) {
  const hint = $("#file-hint"); hint.textContent = ""; hint.classList.add("hidden");
  const v = document.createElement("video"), url = URL.createObjectURL(file);
  v.preload = "metadata"; v.muted = true;
  v.onloadedmetadata = () => {
    const short = Math.min(v.videoWidth, v.videoHeight), dur = v.duration, mbps = isFinite(dur) && dur > 0 ? file.size * 8 / dur / 1e6 : 0;
    const tips = [];
    if (short > 1100) tips.push(`分辨率 ${v.videoWidth}×${v.videoHeight}：系统只用到 1080p，4K 体积约为 1080p 的 4 倍`);
    if (mbps > 30) tips.push(`码率约 ${Math.round(mbps)} Mb/s（多为 4K 或 60 fps）`);
    if (tips.length) {
      hint.textContent = `提示：${tips.join("；")}。下次建议用 1080p 30 fps 拍摄，上传更快、结果不变。本次照常上传。`;
      hint.classList.remove("hidden");
    }
    URL.revokeObjectURL(url);
  };
  v.onerror = () => URL.revokeObjectURL(url);
  v.src = url;
}
function doUpload(file) {
  adviseFile(file);
  const task = { key: "u" + Date.now(), name: file.name, frac: 0, state: "uploading", total: file.size, samples: [] };
  TASKS.uploads.set(task.key, task);
  form.upload = null; form.local = null;
  $("#drop").classList.add("has");
  const show = () => {
    if (form.uploading === p) $("#drop-text").textContent = `上传中：${file.name} · ${uploadStatus(task)}`;
    renderTasks();
  };
  const p = uploadFile(file, (f, loaded) => {
    task.frac = f;
    task.samples.push({ t: Date.now(), b: loaded });
    if (task.samples.length > 200) task.samples.splice(0, 100);
    show();
  });
  const tick = setInterval(show, 1000);   // keeps the stall notice honest when no progress events arrive
  form.uploading = p; form.uploadTask = task;
  show();
  p.finally(() => clearInterval(tick));
  p.then((r) => {
    task.state = "uploaded"; task.frac = 1;
    if (form.uploading === p) { form.upload = r.upload_id; form.uploading = null; $("#drop-text").textContent = `已上传：${file.name}（${r.size_mb} MB）`; }
  }, (e) => {
    task.state = "error"; task.error = "上传失败：" + e.message;
    if (form.uploading === p) { form.uploading = null; $("#drop-text").textContent = task.error; $("#drop").classList.remove("has"); }
  }).finally(() => { check(); renderTasks(); });
  renderTasks(); check();
}
function resetForm() {
  Object.assign(form, { upload: null, local: null, uploading: null, uploadTask: null });
  $("#file").value = ""; $("#drop").classList.remove("has"); $("#drop-text").textContent = "点击选择或拖入视频文件";
  $$("#local-list button").forEach((x) => x.classList.remove("sel")); $("#start-msg").textContent = ""; check();
}
async function start() {
  $("#start").disabled = true;
  const scope = document.querySelector('input[name="scope"]:checked').value;
  const body = {
    reference_id: form.ref, user: form.upload || form.uploading ? { upload_id: form.upload, filename: form.uploadTask && form.uploadTask.name } : { local: form.local },
    user_mirrored: document.querySelector('input[name="mirror"]:checked').value, mirror_practice: $("#mirror-practice").checked,
    sample_fps: +$("#fps").value, backend: CFG.dev_mode ? $("#backend").value : CFG.default_backend,
    scope_mode: scope, seg_from: scope === "auto" ? null : $("#seg-from").value, seg_to: scope === "auto" ? null : $("#seg-to").value,
    mode: document.querySelector('input[name="people"]:checked').value,
    height_cm: +$("#height-cm").value || null,
    user_start: scope === "manual" ? +$("#u-start").value : null, user_end: scope === "manual" ? +$("#u-end").value : null,
  };
  // the job runs in the background: go to the history page, which shows its progress next to earlier results
  const uploading = form.uploading, task = form.uploadTask;
  if (uploading) { task.state = "pending"; task.thenAnalyse = true; }
  resetForm();
  location.hash = "#/jobs";
  try {
    if (uploading) body.user.upload_id = (await uploading).upload_id;
    const job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (task) TASKS.uploads.delete(task.key);
    TASKS.jobs.set(job.id, job);
  } catch (e) {
    if (task) { task.state = "error"; task.error = task.error || "无法开始分析：" + e.message; }
    else TASKS.uploads.set("e" + Date.now(), { name: "新分析", state: "error", error: "无法开始分析：" + e.message });
  }
  refreshTasks();
}
window.addEventListener("beforeunload", (e) => {   // closing the page would abort an upload
  if ([...TASKS.uploads.values()].some((t) => t.state === "uploading" || t.state === "pending")) { e.preventDefault(); e.returnValue = ""; }
});
async function loadJobs(list) {
  const jobs = list || await api("/api/jobs").catch(() => []);
  const box = $("#job-list");
  box.replaceChildren();
  if (!jobs.length) box.append(el("p", { class: "msg" }, "暂无"));
  for (const j of jobs.slice(0, 6)) {
    const d = new Date(j.created * 1000), sm = j.summary;
    const active = j.status === "running" || j.status === "queued";
    box.append(el("a", { href: active ? "#/jobs" : jobHref(j) }, el("span", {}, (j.user_filename || "").slice(0, 30) + (sm ? ` · ${sm.segments}` : "")),
      el("span", { class: "st" }, `${active ? jobProgressText(j) : JOB_STATUS[j.status] || j.status} · ${d.toLocaleString()}`)));
  }
}
const JOB_STATUS = { done: "完成", error: "失败", running: "进行中", queued: "排队", rejected: "无法分析" };
const jobStage = (j) => (j.status === "queued" ? "排队等待" : (STAGES.find(([k]) => k === j.stage) || [, "准备中"])[1]);
const jobProgressText = (j) => `${jobStage(j)} · ${Math.round((j.progress || 0) * 100)}%`;

// ------------------------------------------------------------------ background tasks: uploads + analyses, visible on every page
const TASKS = { uploads: new Map(), jobs: new Map(), timer: null, open: false };
async function refreshTasks() {
  clearTimeout(TASKS.timer);
  const jobs = await api("/api/jobs").catch(() => null);
  if (jobs) {
    for (const j of jobs) {
      const known = TASKS.jobs.get(j.id);
      if (j.status === "queued" || j.status === "running" || known) TASKS.jobs.set(j.id, { ...j, dismissed: known && known.dismissed });
    }
    if (!$("#view-jobs").classList.contains("hidden")) { jobsCache = jobs; renderJobs(); }
    if (!$("#view-new").classList.contains("hidden")) loadJobs(jobs);
  }
  renderTasks();
  const busy = [...TASKS.jobs.values()].some((j) => j.status === "queued" || j.status === "running")
    || [...TASKS.uploads.values()].some((t) => t.state === "uploading" || t.state === "pending");
  if (busy) TASKS.timer = setTimeout(refreshTasks, 1500);
}
function renderTasks() {
  const ups = [...TASKS.uploads.values()], jobs = [...TASKS.jobs.values()].filter((j) => !j.dismissed);
  const active = ups.filter((t) => t.state !== "error" && t.state !== "uploaded").length + jobs.filter((j) => j.status === "queued" || j.status === "running").length;
  const done = jobs.filter((j) => j.status === "done").length, failed = jobs.filter((j) => j.status === "error" || j.status === "rejected").length + ups.filter((t) => t.state === "error").length;
  const chip = $("#tasks-chip"), panel = $("#tasks-panel");
  chip.classList.toggle("hidden", !(ups.length || jobs.length));
  chip.classList.toggle("busy", active > 0);
  chip.textContent = active ? `⟳ 分析中 ${active} 项` : failed ? `! ${failed} 项未完成分析` : `✓ ${done} 项分析完成`;
  panel.classList.toggle("hidden", !TASKS.open || !(ups.length || jobs.length));
  const item = (name, sub, frac, state, actions) => el("div", { class: `task ${state}` },
    el("div", { class: "task-name", title: name }, name), el("div", { class: "task-sub" }, sub),
    frac !== null ? el("div", { class: "bar thin" }, el("div", { style: `width:${Math.round(frac * 100)}%` })) : null,
    actions ? el("div", { class: "task-act" }, ...actions) : null);
  const rows = [];
  for (const t of ups) {
    const sub = t.state === "error" ? t.error : t.state === "uploaded" ? "已上传，等待点击“开始分析”"
      : `上传中 ${uploadStatus(t)}${t.thenAnalyse ? "，完成后自动开始分析" : ""}`;
    rows.push(item(t.name, sub, t.state === "uploading" || t.state === "pending" ? t.frac : null, t.state,
      t.state === "error" || t.state === "uploaded" ? [el("button", { onclick: () => { TASKS.uploads.delete(t.key); renderTasks(); } }, "关闭")] : null));
  }
  for (const j of jobs) {
    const act = j.status === "queued" || j.status === "running";
    rows.push(item(j.user_filename || j.id, act ? jobProgressText(j) : j.status === "done" ? "分析完成"
      : j.status === "rejected" ? "无法分析：" + (j.reason || "") : "分析失败：" + (j.error || ""), act ? j.progress || 0 : null, j.status,
      act ? null : [j.status === "done" ? el("a", { class: "button", href: jobHref(j) }, "打开报告") : null,
        el("button", { onclick: () => { TASKS.jobs.set(j.id, { ...j, dismissed: true }); renderTasks(); } }, "关闭")]));
  }
  panel.replaceChildren(el("div", { class: "task-head" }, "后台任务", el("a", { href: "#/jobs" }, "分析记录 →")), ...rows);
}

// ------------------------------------------------------------------ history
let jobsCache = [];
async function openJobs() {
  show("view-jobs");
  jobsCache = await api("/api/jobs").catch(() => []);
  const q = $("#jobs-q");
  q.oninput = renderJobs;
  renderJobs();
  refreshTasks();
}
function renderJobs() {
  const q = $("#jobs-q").value.trim().toLowerCase(), box = $("#jobs-table");
  const rows = jobsCache.filter((j) => !q || JSON.stringify([j.user_filename, j.summary]).toLowerCase().includes(q));
  box.replaceChildren();
  if (!rows.length) { box.append(el("p", { class: "msg" }, jobsCache.length ? "没有匹配的记录。" : "还没有分析记录。")); return; }
  const t = el("table", { class: "jobs" }, el("tr", {}, ...["时间", "练习者视频", "分析区间", "结果", "状态", ""].map((h) => el("th", {}, h))));
  const del = async (e, j) => {
    e.stopPropagation();
    if (!confirm(`删除这条分析记录？\n${j.user_filename || j.id} · ${new Date(j.created * 1000).toLocaleString()}\n报告与姿态数据将被删除，不可恢复。`)) return;
    try { await api(`/api/jobs/${j.id}`, { method: "DELETE" }); } catch (err) { alert("删除失败：" + err.message); return; }
    jobsCache = jobsCache.filter((x) => x.id !== j.id); TASKS.jobs.delete(j.id); renderJobs(); renderTasks();
  };
  for (const j of rows) {
    const d = new Date(j.created * 1000), sm = j.summary || {}, old = sm.schema && !/0\.[345]$/.test(String(sm.schema));
    const open = () => { location.hash = jobHref(j); };
    const active = j.status === "running" || j.status === "queued";
    t.append(el("tr", { class: j.status === "done" ? "ok" : active ? "active" : "", tabindex: j.status === "done" ? "0" : null,
      onclick: j.status === "done" ? open : null,
      onkeydown: (e) => { if (e.key === "Enter") open(); } },
      el("td", { class: "nowrap" }, d.toLocaleString()),
      el("td", {}, el("b", {}, j.user_filename || "—"), isGroup(j) ? el("span", { class: "pill group" }, "多人") : null, el("div", { class: "hint" }, sm.reference || "")),
      el("td", {}, sm.segments || "—", sm.ref_range ? el("div", { class: "hint" }, `标准 ${sm.ref_range[0].toFixed(0)}–${sm.ref_range[1].toFixed(0)} s ↔ 练习者 ${sm.user_range[0].toFixed(0)}–${sm.user_range[1].toFixed(0)} s`) : null),
      el("td", {}, old ? el("span", { class: "hint" }, "旧版本报告，请重新分析") : (sm.pass_text || "—"), sm.top_issue && !old ? el("div", { class: "hint clamp" }, "重点：" + sm.top_issue) : null),
      el("td", { class: "nowrap" }, el("span", { class: `pill ${j.status}` }, JOB_STATUS[j.status] || j.status), j.error ? el("div", { class: "hint" }, j.error.slice(0, 80)) : null,
        j.reason ? el("div", { class: "hint reason" }, j.reason) : null,
        active ? el("div", { class: "job-prog" }, el("div", { class: "bar thin" }, el("div", { style: `width:${Math.round((j.progress || 0) * 100)}%` })), el("div", { class: "hint" }, jobProgressText(j))) : null),
      el("td", { class: "nowrap actions" }, j.status === "done" ? el("a", { href: jobHref(j), class: "button" }, isGroup(j) ? "集体总览" : "打开报告") : null,
        active ? null : el("button", { class: "danger", title: "删除这条分析记录", onclick: (e) => del(e, j) }, "删除"))));
  }
  box.append(t);
}

// unfinished analyses have no page of their own: their progress is shown in the history and the task tray
const isGroup = (j) => ((j && j.request) || {}).mode === "group";
const jobHref = (j) => (isGroup(j) ? `#/group/${j.id}` : `#/job/${j.id}`);
async function openJob(id, person = null) {
  const job = await api(`/api/jobs/${id}`).catch((e) => ({ status: "error", error: e.message }));
  if (job.status === "done" && isGroup(job) && person === null) { location.replace(`#/group/${id}`); return; }
  if (job.status === "done") return openReport(id, person);
  if (job.id) TASKS.jobs.set(job.id, job);
  location.replace("#/jobs");
  refreshTasks();
}

/** simple (default for everyone): cards + paired videos + skeleton comparison; detailed: the full workbench */
function setDetail(on) {
  $("#view-report").classList.toggle("simple", !on);
  $$("#r-mode button").forEach((b) => b.classList.toggle("on", (b.dataset.m === "detail") === on));
  try { localStorage.setItem("mc-detail", on ? "1" : "0"); } catch (e) { /* storage unavailable: per-visit only */ }
  if (on) requestAnimationFrame(redrawAll);
}
function initDetail() {
  let on = false;
  try { on = localStorage.getItem("mc-detail") === "1"; } catch (e) { /* default */ }
  $$("#r-mode button").forEach((b) => { b.onclick = () => setDetail(b.dataset.m === "detail"); });
  setDetail(on);
}

/** "这里识别不对": the user names the section and its time in the practitioner video, and a new analysis runs
 *  with that interval fixed (manual scope); the original report stays as it was. */
async function openCorrection(segId) {
  const box = $("#r-correct");
  if (R.person !== null && R.person !== undefined) {
    box.replaceChildren(el("p", {}, "多人分析的动作段由集体统一确定：请在集体总览的“修正动作段”中修改。"),
      el("a", { class: "button", href: `#/group/${R.jobId}` }, "去集体总览"));
    box.classList.remove("hidden"); return;
  }
  const job = await api(`/api/jobs/${R.jobId}`);
  const refs = await api("/api/references");
  const segs = (refs.find((r) => r.id === job.request.reference_id) || { segments: [] }).segments;
  const cur = R.rep.segments.find((s) => s.id === segId) || R.rep.segments[0];
  const opts = (sel) => segs.map((s) => el("option", { value: s.id, selected: s.id === sel }, s.name));
  const from = el("select", { id: "fix-from" }, ...opts(cur.id)), to = el("select", { id: "fix-to" }, ...opts(cur.id));
  const a = el("input", { id: "fix-a", type: "number", step: "0.5", min: "0", value: cur.user_start.toFixed(1) });
  const b = el("input", { id: "fix-b", type: "number", step: "0.5", min: "0", value: cur.user_end.toFixed(1) });
  const msg = el("p", { class: "msg" });
  const useNow = (inp) => el("button", { onclick: () => { inp.value = vUser().currentTime.toFixed(1); } }, "取当前画面时间");
  box.replaceChildren(
    el("h3", {}, "修正动作段"),
    el("p", { class: "hint" }, "选择这段时间里练习者实际做的动作段，并给出它在练习者视频中的起止（可先拖动画面，再点“取当前画面时间”）。将按此重新分析，原报告保留。"),
    el("div", { class: "fix-row" }, el("label", {}, "动作段 ", from, " 到 ", to)),
    el("div", { class: "fix-row" }, el("label", {}, "练习者视频 ", a, " 秒"), useNow(a), el("label", {}, " 至 ", b, " 秒"), useNow(b)),
    el("div", { class: "fix-row" },
      el("button", { class: "primary", onclick: async () => {
        const body = { ...job.request, scope_mode: "manual", seg_from: from.value, seg_to: to.value, user_start: +a.value, user_end: +b.value };
        if (!(body.user_end > body.user_start)) { msg.textContent = "结束时间应晚于开始时间。"; return; }
        try { const j = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
          TASKS.jobs.set(j.id, j); refreshTasks(); location.hash = "#/jobs"; }
        catch (e) { msg.textContent = "无法开始：" + e.message; }
      } }, "按此重新分析"),
      el("button", { onclick: () => box.classList.add("hidden") }, "取消")),
    msg);
  box.classList.remove("hidden");
  box.scrollIntoView({ block: "nearest" });
}

/** joints filled across a short gap at sample i (pose payload "filled": [[sample, joint], ...]), as a Set */
function filledAt(P, i) {
  if (!P.filledBy) { P.filledBy = new Map(); for (const [a, b] of P.filled || []) { if (!P.filledBy.has(a)) P.filledBy.set(a, new Set()); P.filledBy.get(a).add(b); } }
  return P.filledBy.get(i) || new Set();
}

// ================================================================== report workbench
const R = { rep: null, poses: {}, seg: null, feat: null, master: "user", raf: 0, normMode: "now", heatOpen: false, view: null, hoverX: null, sel: null };

/** in a group job the workbench shows one performer: back link to the overview and a performer switcher */
async function personHeader(id, person) {
  const back = $("#r-back"), box = $("#r-person");
  if (person === null) { back.href = "#/jobs"; back.textContent = "← 分析记录"; box.classList.add("hidden"); return; }
  back.href = `#/group/${id}`; back.textContent = "← 集体总览";
  const g = await api(`/api/jobs/${id}/group`).catch(() => null);
  if (!g) return;
  const sel = el("select", { id: "r-person-sel", "aria-label": "切换练习者",
    onchange: (e) => { location.hash = `#/job/${id}/p/${e.target.value}`; } },
    ...g.identities.filter((i) => i.report).map((i) => el("option", { value: i.id, selected: i.id === person }, `练习者 #${i.id}`)));
  box.replaceChildren(el("span", { class: "who-dot", style: `background:${idColor(person, g)}` }), "当前：", sel);
  box.classList.remove("hidden");
}
async function openReport(id, person = null) {
  show("view-report");
  vRef().pause(); vUser().pause();
  const q = person === null ? "" : `?person=${person}`;
  const [rep, pr, pu] = await Promise.all([api(`/api/jobs/${id}/report${q}`), api(`/api/jobs/${id}/poses/ref${q}`), api(`/api/jobs/${id}/poses/user${q}`)]);
  personHeader(id, person);
  if (!/0\.[345]$/.test(String(rep.schema))) {
    $("#r-title").textContent = "旧版本报告";
    $("#r-scope").textContent = "该报告由早期版本生成，缺少有效性状态等数据，请重新分析。";
    return;
  }
  // practitioner-led by default: each practitioner repetition replays the matching reference repetition
  Object.assign(R, { rep, poses: { ref: pr, user: pu }, jobId: id, person, master: "user", curRep: null });
  $("#r-correct").classList.add("hidden");
  initDetail();
  $("#master").value = "user";
  R.featIdx = Object.fromEntries(rep.features.map((f, i) => [f.name, i]));
  R.T = rep.series.t_ref;
  R.dt = R.T.length > 1 ? (R.T[R.T.length - 1] - R.T[0]) / (R.T.length - 1) : 0.1;
  R.segOf = R.T.map((t) => rep.segments.findIndex((s) => t >= s.ref_start && t < s.ref_end));
  buildMaps();
  R.view = [...rep.scope.ref_range];
  renderHead(); renderFindings(); renderDetails(); setupPlayer();
  $("#csv").href = `/api/jobs/${id}/export.csv${q}`;
  const first = rep.summary.top_issues[0];
  select(first ? first.segment : rep.segments[0].id, first ? first.feature : rep.features[0].name, { keepView: true });
  const t0 = R.deepT ?? (first ? first.time : rep.scope.ref_range[0]);
  whenReady(() => seekRef(t0 + 0.001));
  requestAnimationFrame(redrawAll);
}
function whenReady(fn) {
  const vs = [vRef(), vUser()];
  if (vs.every((v) => v.readyState >= 1)) return fn();
  Promise.all(vs.map((v) => v.readyState >= 1 ? 0 : new Promise((r) => v.addEventListener("loadedmetadata", r, { once: true })))).then(fn);
}
const segById = (id) => R.rep.segments.find((s) => s.id === id);
const featBy = (n) => R.rep.features[R.featIdx[n]];
// lengths in torso units can also be shown in cm (report.scale: adult-average height unless stated)
const LENGTH_FEATURES = new Set(["l_wrist_height", "r_wrist_height", "stance_width", "pelvis_height"]);
const cmShown = () => !!(R.rep && R.rep.scale && R.rep.scale.reliable && R.rep.scale.method === "stated_height");
const toCm = (f, v) => (cmShown() && LENGTH_FEATURES.has(f.name) && v !== null && v !== undefined
  ? v * R.rep.scale.torso_cm : null);
const fmtVal = (f, v, unit = false) => {
  if (v === null || v === undefined) return "—";
  if (f.unit === "deg") return `${Math.round(v)}${unit ? "°" : ""}`;
  const cm = unit ? toCm(f, v) : null;
  if (cm !== null) return `约 ${Math.round(cm)} cm（${v.toFixed(2)} 躯干长）`;
  return `${v.toFixed(2)}${unit ? (f.name === "head_turn" ? " 肩宽" : " 躯干长") : ""}`;
};
function scaleNote(f) {   // how centimetres were obtained, shown next to length indicators
  const sc = R.rep && R.rep.scale;
  if (!sc || !LENGTH_FEATURES.has(f.name) || !cmShown()) return "";
  if (!sc.reliable) return ` 厘米换算不可用：${sc.reason || "比例尺不可靠"}。`;
  return ` 厘米按${sc.method === "average_height" ? `成年人平均身高 ${sc.height_cm} cm ` : `身高 ${sc.height_cm} cm `}换算（${sc.frames_used} 个直立画面，约 ±${Math.round(sc.rel_uncertainty * 100)}%）。`;
}
const fmtSigned = (f, v) => (v === null || v === undefined ? "—" : (v > 0 ? "+" : "") + fmtVal(f, v, true));

/** status of feature f at reference sample i — the single place the client derives an instantaneous status */
function statusAt(i, fname) {
  const si = R.segOf[i];
  if (si < 0) return "out";
  const cell = R.rep.segments[si].features[fname];
  if (cell.status === "na") return "na";
  if (cell.status === "uncertain" || cell.status === "imprecise") return cell.status;
  const a = R.rep.series.ref[fname][i], b = userSeries(fname)[i];
  if (a === null || b === null) return "missing";
  const f = featBy(fname), d = Math.abs(b - a);
  return d <= f.tolerance.good ? "good" : d <= f.tolerance.warn ? "warn" : "bad";
}

// --- header, badges, findings, summary
function segName(id) { const s = R.rep.segments.find((x) => x.id === id) || (R.rep.scope.candidates || []).find((c) => c.segment === id); return s ? s.name : id; }
function renderHead() {
  const rep = R.rep, sc = rep.scope;
  const names = sc.segments.length > 1 ? `${segName(sc.segments[0])} — ${segName(sc.segments[sc.segments.length - 1])}` : segName(sc.segments[0]);
  $("#r-title").textContent = `${rep.profile.title} · ${names}`;
  $("#r-scope").textContent = `分析区间：标准 ${sc.ref_range[0].toFixed(0)}–${sc.ref_range[1].toFixed(0)} 秒 ↔ 练习者 ${sc.user_range[0].toFixed(0)}–${sc.user_range[1].toFixed(0)} 秒 · ${rep.model.backend}（${rep.model.device === "cpu" ? "CPU" : "GPU"}）`
    + (rep.model.bounds_backend && rep.model.bounds_backend !== rep.model.backend ? ` · 段边界 ${rep.model.bounds_backend}` : "");
  const notes = [];
  if (rep.left_right.text) notes.push("左右方向：" + rep.left_right.text);
  notes.push(...rep.quality.warnings);
  const box = $("#r-notes"); box.replaceChildren(...notes.map((n) => el("p", {}, n)));
  const badges = $("#r-badges"); badges.replaceChildren();
  if (rep.profile.status === "draft" || rep.reference.status === "draft") badges.append(el("span", { class: "badge warn" }, "⚠ 草案规则 · 仅供算法验证"));
  if (sc.needs_review) badges.append(el("span", { class: "badge err" }, "区间待人工核查"));
  const scl = rep.scale;
  if (scl) {   // lengths are shown in body proportions; centimetres only from the practitioner's own height
    const txt = cmShown()
      ? `长度另以厘米显示：按填写的身高 ${scl.height_cm} cm 换算（${scl.frames_used} 个直立画面，约 ±${Math.round(scl.rel_uncertainty * 100)}%），仅供参考；比较与判定按身体比例进行。`
      : scl.method === "stated_height"
        ? `厘米换算不可用：${scl.reason || "比例尺不可靠"}，长度只显示为躯干长的倍数。`
        : "长度以身体比例（躯干长、肩宽的倍数）显示；新建分析时填写本人身高后，可另外显示估算的厘米值。";
    notes.unshift(txt);
    box.replaceChildren(...notes.map((n) => el("p", {}, n)));
    badges.append(el("span", { class: "badge", title: txt }, cmShown() ? `身高 ${scl.height_cm} cm` : "长度按身体比例"));
  }
  badges.append(el("button", { class: "badge", onclick: () => box.classList.toggle("hidden") }, `注意事项 ${notes.length} ▾`));
  box.classList.toggle("hidden", !sc.needs_review);
}
function renderFindings() {
  const rep = R.rep, box = $("#findings"); box.replaceChildren();
  const top = rep.summary.top_issues;
  if (!top.length) box.append(el("div", { class: "finding" }, el("div", { class: "what" }, "重点指标未发现超出允许范围的偏差"),
    el("div", { class: "how" }, "注意：不可评价、测量不确定的指标不在此列，请查看右侧计数。")));
  top.forEach((it, k) => {
    const f = featBy(it.feature), c = it.card;
    const go = () => { select(it.segment, it.feature); zoomTo(segById(it.segment)); };
    const look = () => { go(); if (c && c.user_time !== null && c.user_time !== undefined) { setMaster("user"); seekUser(c.user_time); } else seekRef(it.time); };
    const card = el("div", { class: `finding ${it.status}`, "data-k": `${it.segment}|${it.feature}` },
      el("div", { class: "rank" }, el("b", {}, String(k + 1)), `${it.segment_name.replace(/^第.式 /, "")} · ${rep.status_labels[it.status]}`),
      el("div", { class: "what" }, it.label),
      c ? el("div", { class: "how card-text" }, c.text)
        : el("div", { class: "num" }, fmtVal(f, it.value, false), el("small", {}, `${f.unit === "deg" ? "°" : f.name === "head_turn" ? "肩宽" : "躯干长"} · ${it.signed > 0 ? f.more : f.less}`)),
      it.spread_note ? el("div", { class: "hint" }, it.spread_note) : null,
      el("div", { class: "card-act" },
        el("button", { onclick: look }, "看对应画面"),
        el("button", { onclick: () => { go(); setDetail(true); seekRef(it.time); } }, "看其他重复"),
        el("button", { onclick: () => openCorrection(it.segment) }, "这里识别不对")));
    box.append(card);
  });
  const s = rep.summary, c = s.counts || {};
  const evaluated = (c.good || 0) + (c.warn || 0) + (c.bad || 0), notEval = (c.unmeasurable || 0) + (c.na || 0) + (c.uncertain || 0) + (c.imprecise || 0);
  $("#r-summary").replaceChildren(
    el("div", {}, el("span", { class: "big" }, `${c.good || 0}/${evaluated}`), " 个可评价重点指标达标"),
    el("div", { class: "row" }, "不可评价", el("b", {}, `${notEval} 项`)),
    el("div", { class: "row" }, "练习者用时", el("b", {}, `${rep.timing.tempo_ratio.toFixed(2)} 倍标准`)),
    el("div", { class: "row" }, "动作对应可信度", el("b", {}, `${confidence(rep.timing.align_cost)}（${rep.timing.align_cost.toFixed(2)}）`)),
    el("div", { class: "row", title: s.pass_rate_definition }, "说明", el("b", {}, "比例不是动作总分")));
}

// --- time maps (reference time <-> user time)
function buildMaps() {
  const t = R.T, u = R.rep.series.user_t, rt = [], ut = [];
  t.forEach((x, i) => { if (u[i] !== null) { rt.push(x); ut.push(u[i]); } });
  R.map = { rt, ut };
  R.pb = R.rep.playback || null;   // 0.5: loop / jump playback maps on the analysis grid
  R.hasReps = R.rep.segments.some((s) => s.reps);
}
function gridMap(t0, dt, ys, x) {   // ys sampled at t0 + k dt; ~1x outside the range
  const n = ys.length;
  if (!n) return x;
  const k = (x - t0) / dt;
  if (k <= 0) return ys[0] + (x - t0);
  if (k >= n - 1) return ys[n - 1] + (x - t0 - (n - 1) * dt);
  const i = Math.floor(k), w = k - i;
  return ys[i] + w * (ys[i + 1] - ys[i]);
}
function interp(xs, ys, x) {
  if (!xs.length) return x;
  if (x <= xs[0]) return ys[0] + (x - xs[0]);
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1] + (x - xs[xs.length - 1]);
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
  const w = xs[hi] === xs[lo] ? 0 : (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + w * (ys[hi] - ys[lo]);
}
// reference-led: reference time -> practitioner time; practitioner-led: practitioner time -> reference time.
// Between repetitions the shown video runs at ~1x and jumps (loop back to the template / skip extra repetitions).
const refToUser = (t) => Math.max(0, R.pb ? gridMap(R.pb.ref_t0, R.pb.dt, R.pb.user_t_for_ref, t) : interp(R.map.rt, R.map.ut, t));
const userToRef = (u) => Math.max(0, R.pb ? gridMap(R.pb.user_t0, R.pb.dt, R.pb.ref_t_for_user, u) : interp(R.map.ut, R.map.rt, u));
function repAtUser(u) {   // [segment, repetition index] of the practitioner repetition containing time u
  for (const s of R.rep.segments) {
    if (!s.reps || s.reps.repeated === false) continue;
    for (const [k, r] of (s.reps.user || []).entries()) if (u >= r.start && u <= r.end) return [s, k];
  }
  return [null, -1];
}
// the practitioner repetition on screen (practitioner-led): curves, values and the skeleton follow it
function currentRep() {
  if (R.master !== "user") return null;
  const [s, k] = repAtUser(vUser().currentTime);
  if (!s) return null;
  const u = s.reps.user[k];
  return u && u.paired && u.series ? { seg: s, k, u, refRep: s.reps.ref[u.ref] } : null;
}
function repLabel() {
  const [s, k] = repAtUser(vUser().currentTime);
  if (!s) return "";
  const u = s.reps.user[k], n = s.reps.user.length, nr = s.reps.ref.length;
  return `练习者第 ${k + 1}/${n} 次 ↔ 标准第 ${u.ref + 1}${nr > 1 ? `/${nr}` : ""} 次${u.mirror ? "（镜像侧）" : ""}`;
}
const _useries = {};
function userSeries(fname) {   // default-aligned series, with the playing repetition substituted on its reference interval
  const base = R.rep.series.user[fname], c = currentRep();
  if (!c) return base;
  const key = `${c.seg.id}|${c.k}|${fname}`;
  if (_useries.key === key) return _useries.arr;
  const arr = base.slice(), i0 = nearestIdx(R.T, c.refRep.start), vals = c.u.series[fname] || [];
  for (let j = 0; j < vals.length && i0 + j < arr.length; j++) arr[i0 + j] = vals[j];
  _useries.key = key; _useries.arr = arr;
  return arr;
}
function nearestIdx(arr, x) {
  let lo = 0, hi = arr.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (arr[m] <= x) lo = m; else hi = m; }
  return Math.abs(arr[hi] - x) < Math.abs(arr[lo] - x) ? hi : lo;
}
const inRange = (t) => t >= R.rep.scope.ref_range[0] - 0.05 && t <= R.rep.scope.ref_range[1] + 0.05;

// --- selection
function select(segId, feat, opt = {}) {
  R.seg = segId; R.feat = feat;
  $$(".finding").forEach((c) => c.classList.toggle("sel", c.dataset.k === `${segId}|${feat}`));
  renderChips(); renderFocus(); renderHeatSel();
  redrawAll();
}
function zoomTo(seg) {
  if (!seg) return;
  const pad = (seg.ref_end - seg.ref_start) * 0.06;
  setView(seg.ref_start - pad, seg.ref_end + pad);
}
function setView(a, b) {
  const [A, B] = R.rep.scope.ref_range;
  const minSpan = 3;
  let w = Math.max(minSpan, Math.min(B - A, b - a));
  a = clamp(a, A, B - w); b = a + w;
  R.view = [a, b];
  redrawAll();
}

// ================================================================== timeline
const TL = { G: 118, PAD: 10, rows: [] };
function tlLayout() {
  const rows = [["ruler", 22], ["ref", 34], ["link", 34], ["user", 34], ["feat", 18], ["pat", 14], ["heatHead", 20]];
  if (R.heatOpen) for (const f of R.rep.features) rows.push(["lane:" + f.name, 11]);
  let y = 0;
  TL.rows = rows.map(([k, h]) => { const r = { k, y, h }; y += h + (k === "heatHead" || k.startsWith("lane") ? 1 : 3); return r; });
  TL.H = y + 4;
}
const tlRow = (k) => TL.rows.find((r) => r.k === k);
const tx = (t) => TL.G + ((t - R.view[0]) / (R.view[1] - R.view[0])) * (TL.W - TL.G - TL.PAD);
const txInv = (x) => R.view[0] + ((x - TL.G) / (TL.W - TL.G - TL.PAD)) * (R.view[1] - R.view[0]);
function userView() { return [refToUser(R.view[0]), refToUser(R.view[1])]; }
const ux = (u, UV) => TL.G + ((u - UV[0]) / Math.max(1e-6, UV[1] - UV[0])) * (TL.W - TL.G - TL.PAD);
const uxInv = (x, UV) => UV[0] + ((x - TL.G) / (TL.W - TL.G - TL.PAD)) * (UV[1] - UV[0]);
function ctx2d(cv, height) {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth;
  if (height) cv.style.height = `${height}px`;
  const H = height || cv.clientHeight;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  return [g, W, H];
}
function niceStep(span, px) {
  const target = span / Math.max(1, px / 80);
  for (const s of [0.5, 1, 2, 5, 10, 15, 30, 60, 120]) if (s >= target) return s;
  return 300;
}
function segColor(seg) {
  const c = seg.features[R.feat], st = c ? c.status : "unmeasurable";
  return css(st === "good" ? "--good-soft" : STATUS_VAR[st] || "--unm");
}
function drawTimeline() {
  if (!R.rep || !R.feat) return;
  tlLayout();
  const cv = $("#tl");
  const [g, W] = ctx2d(cv, TL.H);
  TL.W = W;
  const UV = userView(), rows = TL.rows;
  g.font = "12px system-ui"; g.textBaseline = "middle";
  // gutter labels + track backgrounds
  const labels = { ruler: "标准时间", ref: "标准示范", link: "对应关系", user: "练习者", feat: `当前：${featBy(R.feat).label}`, pat: "差异模式", heatHead: `${R.heatOpen ? "▾" : "▸"} 全部指标（${R.rep.features.length}）` };
  for (const r of rows) {
    if (r.k !== "ruler" && r.k !== "link") { g.fillStyle = css("--track"); g.fillRect(TL.G, r.y, W - TL.G - TL.PAD, r.h); }
    let lab = labels[r.k];
    if (r.k.startsWith("lane:")) { const f = featBy(r.k.slice(5)); lab = f.label; }
    if (!lab) continue;
    const sel = r.k === "lane:" + R.feat || r.k === "feat";
    const ident = r.k === "ref" ? "--ref" : r.k === "user" ? "--user" : null;
    if (ident) { g.fillStyle = css(ident); g.fillRect(0, r.y + 4, 4, r.h - 8); }
    g.fillStyle = ident ? css(ident) : sel ? css("--ink") : css("--ink2");
    if (ident) { g.font = "700 12px system-ui"; g.fillText(lab, 9, r.y + r.h / 2, TL.G - 14); continue; }
    g.font = `${sel ? "700 " : ""}${r.k.startsWith("lane") ? 10 : 12}px system-ui`;
    g.fillText(lab, 6, r.y + r.h / 2, TL.G - 10);
  }
  g.font = "12px system-ui";
  // ruler
  const ru = tlRow("ruler"), span = R.view[1] - R.view[0], step = niceStep(span, W - TL.G);
  g.strokeStyle = css("--axis"); g.fillStyle = css("--ink2"); g.lineWidth = 1;
  for (let t = Math.ceil(R.view[0] / step) * step; t <= R.view[1]; t += step) {
    const x = tx(t); g.beginPath(); g.moveTo(x, ru.y + ru.h - 6); g.lineTo(x, ru.y + ru.h); g.stroke();
    g.textAlign = "center"; g.fillText(`${t.toFixed(step < 1 ? 1 : 0)}s`, x, ru.y + 8); g.textAlign = "left";
  }
  const rr = tlRow("ref"), lr = tlRow("link"), urw = tlRow("user");
  const clipX = (x) => clamp(x, TL.G, W - TL.PAD);
  // segments on both tracks + correspondence bands
  R.rep.segments.forEach((s, si) => {
    const col = segColor(s), sel = s.id === R.seg;
    const x0 = clipX(tx(s.ref_start)), x1 = clipX(tx(s.ref_end));
    const u0 = clipX(ux(s.user_start, UV)), u1 = clipX(ux(s.user_end, UV));
    const RP = s.reps && s.reps.user.length && s.reps.repeated !== false ? s.reps : null;
    const band = (a, b, c, d) => { g.beginPath(); g.moveTo(a, lr.y); g.lineTo(b, lr.y); g.lineTo(d, lr.y + lr.h); g.lineTo(c, lr.y + lr.h); g.closePath(); g.fill(); };
    g.globalAlpha = RP ? 0.07 : sel ? 0.30 : 0.16; g.fillStyle = col; band(x0, x1, u0, u1);
    if (RP) for (const r of RP.user) {   // one band per paired repetition: reference repetition <-> practitioner repetition
      if (!r.paired) continue;
      const q = RP.ref[r.ref];
      g.globalAlpha = r.default ? (sel ? 0.7 : 0.4) : (sel ? 0.28 : 0.14);
      band(clipX(tx(q.start)), clipX(tx(q.end)), clipX(ux(r.start, UV)), clipX(ux(r.end, UV)));
    }
    g.globalAlpha = 1;
    for (const [a, b, r] of [[x0, x1, rr], [u0, u1, urw]]) {
      if (b - a < 1) continue;
      g.fillStyle = css("--card"); g.fillRect(a + 1, r.y + 2, b - a - 2, r.h - 4);
      g.fillStyle = col; g.globalAlpha = sel ? 0.55 : 0.35; g.fillRect(a + 1, r.y + 2, b - a - 2, r.h - 4); g.globalAlpha = 1;
      g.fillStyle = css(r === rr ? "--ref" : "--user"); g.fillRect(a + 1, r.y + r.h - 5, b - a - 2, 3);   // identity base line
      if (sel) { g.strokeStyle = css("--ink"); g.lineWidth = 2; g.strokeRect(a + 1, r.y + 2, b - a - 2, r.h - 4); }
      g.fillStyle = css("--ink"); g.font = `${sel ? "700 " : ""}12px system-ui`;
      const name = s.name.replace(/^第(.)式 /, "$1 ");
      if (b - a > 30) g.fillText(name, a + 6, r.y + (r === urw ? 11 : r.h / 2), b - a - 10);
    }
    // repetition intervals (active cores): solid = paired with the reference, hollow = no counterpart
    if (RP) {
      const reps = [...RP.ref.map((q) => [q.start, q.end, true, rr, tx]), ...RP.user.map((q) => [q.start, q.end, q.paired, urw, (v) => ux(v, UV)])];
      for (const [a0, b0, ok, row, X] of reps) {
        const a = clipX(X(a0)), b = clipX(X(b0)); if (b - a < 2) continue;
        const y = row === rr ? row.y + row.h - 11 : row.y + 14;   // practitioner: above its own time labels
        g.fillStyle = css("--ink"); g.strokeStyle = css("--ink"); g.lineWidth = 1; g.globalAlpha = sel ? 0.75 : 0.4;
        if (ok) g.fillRect(a + 1, y, b - a - 2, 4); else g.strokeRect(a + 1.5, y + 0.5, b - a - 3, 3);
        g.globalAlpha = 1;
      }
      const cu = RP.count_user, cr = RP.count_ref, ub = u1 - u0;
      const lbl = `${cu} 次 · 标准 ${cr} 次`;
      g.font = "12px system-ui"; const need = g.measureText(s.name.replace(/^第(.)式 /, "$1 ")).width + 30;
      g.font = "11px system-ui";
      if ((cu > 1 || cr > 1) && ub > need + g.measureText(lbl).width) {   // only where it does not cover the name
        g.fillStyle = css("--ink"); g.textAlign = "right"; g.fillText(lbl, u1 - 6, urw.y + 11); g.textAlign = "left";
      }
    }
    // key moments (ticks) on the reference track
    g.strokeStyle = css("--ink"); g.lineWidth = sel ? 1.5 : 1; g.globalAlpha = sel ? 0.9 : 0.35;
    for (const h of s.hold_times) { const x = tx(h); if (x < TL.G || x > W - TL.PAD) continue; g.beginPath(); g.moveTo(x, rr.y + rr.h - (sel ? 9 : 5)); g.lineTo(x, rr.y + rr.h - 2); g.stroke(); }
    g.globalAlpha = 1;
  });
  // correspondence lines (older reports without repetition bands)
  g.strokeStyle = css("--axis"); g.lineWidth = 1;
  const lstep = niceStep(span, (W - TL.G) / 2.5);
  if (!R.hasReps) for (let t = Math.ceil(R.view[0] / lstep) * lstep; t <= R.view[1]; t += lstep) {
    g.beginPath(); g.moveTo(tx(t), lr.y + 1); g.lineTo(ux(refToUser(t), UV), lr.y + lr.h - 1); g.stroke();
  }
  // practitioner ruler ticks (its own time)
  g.fillStyle = css("--muted"); g.font = "10px system-ui"; g.textAlign = "right";
  const ustep = niceStep(UV[1] - UV[0], W - TL.G);
  for (let u = Math.ceil(UV[0] / ustep) * ustep; u <= UV[1]; u += ustep) {
    const x = ux(u, UV); g.fillRect(x, urw.y + urw.h - 5, 1, 5); g.fillText(`${u.toFixed(0)}s`, x - 3, urw.y + urw.h - 7);
  }
  g.textAlign = "left";
  // status strips: current feature, patterns, all-feature lanes
  const i0 = Math.max(0, nearestIdx(R.T, R.view[0]) - 1), i1 = Math.min(R.T.length - 1, nearestIdx(R.T, R.view[1]) + 1);
  const strip = (row, colorAt) => {
    for (let i = i0; i <= i1; i++) {
      const c = colorAt(i); if (!c) continue;
      const a = clipX(tx(R.T[i] - R.dt / 2)), b = clipX(tx(R.T[i] + R.dt / 2));
      g.fillStyle = c; g.fillRect(a, row.y + 1, Math.max(1, b - a + 0.5), row.h - 2);
    }
  };
  const statusColor = (st) => (st === "out" ? null : st === "na" ? css("--line") : st === "good" ? css("--good-soft") : css(STATUS_VAR[st]));
  strip(tlRow("feat"), (i) => statusColor(statusAt(i, R.feat)));
  const lab = R.rep.patterns.labels;
  strip(tlRow("pat"), (i) => (R.segOf[i] < 0 || lab[i] < 0 ? null : lab[i] === 0 ? css("--axis") : css(SERIES[(lab[i] - 1) % SERIES.length])));
  if (R.heatOpen) for (const f of R.rep.features) {
    const row = tlRow("lane:" + f.name);
    strip(row, (i) => statusColor(statusAt(i, f.name)));
    if (f.name === R.feat) { g.strokeStyle = css("--accent"); g.lineWidth = 1.5; g.strokeRect(TL.G, row.y, W - TL.G - TL.PAD, row.h); }
  }
  // issue markers
  TL.markers = [];
  R.rep.summary.top_issues.forEach((it, k) => {
    const x = tx(it.time); if (x < TL.G || x > W - TL.PAD) return;
    const y = rr.y + 1;
    g.fillStyle = css(STATUS_VAR[it.status]); g.strokeStyle = css("--card"); g.lineWidth = 2;
    g.beginPath(); g.arc(x, y + 6, 8, 0, 7); g.fill(); g.stroke();
    g.fillStyle = it.status === "warn" ? "#0b0b0b" : "#fff"; g.font = "700 11px system-ui"; g.textAlign = "center";
    g.fillText(String(k + 1), x, y + 6.5); g.textAlign = "left";
    TL.markers.push({ x, y: y + 6, it });
  });
  // playhead (reference) and its mapped position on the practitioner track
  const t = currentRefTime(), xp = tx(t), xu = ux(R.master === "user" ? vUser().currentTime : refToUser(t), UV);
  g.strokeStyle = css("--ink"); g.fillStyle = css("--ink"); g.lineWidth = 2;
  if (xp >= TL.G && xp <= W - TL.PAD) {
    g.beginPath(); g.moveTo(xp, 0); g.lineTo(xp, lr.y); g.lineTo(xu, lr.y + lr.h); g.lineTo(xu, urw.y + urw.h); g.stroke();
    g.beginPath(); g.moveTo(xp, tlRow("feat").y); g.lineTo(xp, TL.H); g.stroke();
    g.beginPath(); g.moveTo(xp - 6, 0); g.lineTo(xp + 6, 0); g.lineTo(xp, 8); g.closePath(); g.fill();
    for (const [x, y, v] of [[xp, rr.y + rr.h / 2, "--ref"], [xu, urw.y + urw.h / 2, "--user"]]) {   // which track is which
      g.fillStyle = css(v); g.strokeStyle = css("--card"); g.lineWidth = 2; g.beginPath(); g.arc(x, y, 5, 0, 7); g.fill(); g.stroke();
    }
  }
}
function toggleHeat() {   // the expanded 14-lane view is tall: stop sticking so it never covers the layers below
  R.heatOpen = !R.heatOpen;
  $("#tl-panel").classList.toggle("tall", R.heatOpen);
  redrawAll();
}
// timeline interaction
function tlHit(y) { return TL.rows.find((r) => y >= r.y && y < r.y + r.h + 3); }
function segAtX(x, row) {
  const UV = userView();
  return R.rep.segments.find((s) => row === "user" ? (x >= ux(s.user_start, UV) && x <= ux(s.user_end, UV)) : (x >= tx(s.ref_start) && x <= tx(s.ref_end)));
}
let tlDrag = null;
function tlSeekAt(x, mode) {
  if (mode === "user") { setMaster("user"); seekUser(uxInv(x, userView())); }
  else seekRef(txInv(x));
}
function setupTimeline() {
  const cv = $("#tl");
  cv.addEventListener("pointerdown", (e) => {
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    if (x < TL.G) return;
    const row = tlHit(y);
    tlDrag = { mode: row && row.k === "user" ? "user" : "ref", x0: x, moved: false };
    cv.setPointerCapture(e.pointerId);
    if (master().paused) tlSeekAt(x, tlDrag.mode);
  });
  cv.addEventListener("pointermove", (e) => {
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    if (tlDrag) {
      if (Math.abs(x - tlDrag.x0) > 3) tlDrag.moved = true;
      if (tlDrag.moved) tlSeekAt(clamp(x, TL.G, TL.W - TL.PAD), tlDrag.mode);
      return;
    }
    tlTooltip(x, y);
  });
  cv.addEventListener("pointerleave", () => $("#tl-tip").classList.add("hidden"));
  cv.addEventListener("pointerup", (e) => {
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    const d = tlDrag; tlDrag = null;
    if (d && d.moved) return;
    const row = tlHit(y);
    if (x < TL.G) {   // gutter: toggle heat group / pick lane
      if (row && row.k === "heatHead") toggleHeat();
      else if (row && row.k.startsWith("lane:")) select(R.seg, row.k.slice(5));
      return;
    }
    const mk = (TL.markers || []).find((m) => Math.hypot(m.x - x, m.y - y) < 10);
    if (mk) { select(mk.it.segment, mk.it.feature); seekRef(mk.it.time); return; }
    if (!row) return;
    if (row.k === "heatHead") { toggleHeat(); return; }
    if (row.k.startsWith("lane:")) select(R.seg, row.k.slice(5));
    if (row.k === "ref" || row.k === "user") { const s = segAtX(x, row.k); if (s && s.id !== R.seg) select(s.id, R.feat); }
    tlSeekAt(x, row.k === "user" ? "user" : "ref");
  });
  cv.addEventListener("dblclick", (e) => {
    const r = cv.getBoundingClientRect(), row = tlHit(e.clientY - r.top);
    const s = row && (row.k === "ref" || row.k === "user") ? segAtX(e.clientX - r.left, row.k) : null;
    if (s) { select(s.id, R.feat); zoomTo(s); }
  });
  cv.addEventListener("wheel", (e) => {
    if (!e.ctrlKey && !e.shiftKey) return;
    e.preventDefault();
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, span = R.view[1] - R.view[0];
    if (e.shiftKey) { const d = (e.deltaY || e.deltaX) / 400 * span; setView(R.view[0] + d, R.view[1] + d); return; }
    const c = txInv(x), k = e.deltaY > 0 ? 1.25 : 0.8;
    setView(c - (c - R.view[0]) * k, c + (R.view[1] - c) * k);
  }, { passive: false });
  $("#z-all").onclick = () => setView(...R.rep.scope.ref_range);
  $("#z-seg").onclick = () => zoomTo(segById(R.seg));
  $("#z-in").onclick = () => { const c = currentRefTime(), s = (R.view[1] - R.view[0]) * 0.35; setView(c - s, c + s); };
  $("#z-out").onclick = () => { const c = (R.view[0] + R.view[1]) / 2, s = (R.view[1] - R.view[0]) * 0.8; setView(c - s, c + s); };
}
function tlTooltip(x, y) {
  const tip = $("#tl-tip"), row = tlHit(y);
  if (x < TL.G || !row || row.k === "ruler") { tip.classList.add("hidden"); return; }
  const lines = [];
  const mk = (TL.markers || []).find((m) => Math.hypot(m.x - x, m.y - y) < 10);
  if (mk) lines.push(el("b", {}, `重点差异：${mk.it.segment_name}`), mk.it.text);
  else if (row.k === "user") {
    const u = uxInv(x, userView()), s = segAtX(x, "user");
    lines.push(el("b", {}, `练习者 ${u.toFixed(1)} 秒 ↔ 标准 ${userToRef(u).toFixed(1)} 秒`), s ? `${s.name}（用时 ${s.duration_ratio ? s.duration_ratio.toFixed(2) + " 倍" : "—"}）` : "");
  } else {
    const t = txInv(x), i = nearestIdx(R.T, t), s = R.segOf[i] >= 0 ? R.rep.segments[R.segOf[i]] : null;
    lines.push(el("b", {}, `标准 ${t.toFixed(1)} 秒 ↔ 练习者 ${refToUser(t).toFixed(1)} 秒`));
    if (s) lines.push(s.name);
    const fname = row.k.startsWith("lane:") ? row.k.slice(5) : (row.k === "feat" || row.k === "ref" || row.k === "link") ? R.feat : null;
    if (fname && s) {
      const f = featBy(fname), st = statusAt(i, fname), a = R.rep.series.ref[fname][i], b = userSeries(fname)[i];
      lines.push(`${f.label}：${R.rep.status_labels[st] || { missing: "缺失" }[st] || st}  标准 ${fmtVal(f, a, true)} · 练习者 ${fmtVal(f, b, true)}`);
    }
    if (row.k === "pat" && R.rep.patterns.labels[i] >= 0) {
      const p = R.rep.patterns.items.find((p) => p.id === R.rep.patterns.labels[i]);
      if (p) lines.push(p.id === 0 ? p.label : `偏差模式 ${p.id}：${p.label}`);
    }
  }
  tip.replaceChildren(...lines.filter(Boolean).map((l) => (l instanceof Node ? l : el("div", {}, l))));
  tip.classList.remove("hidden");
  tip.style.left = `${Math.min(TL.W - 280, x + 12)}px`; tip.style.top = `${y + 14}px`;
}

// ================================================================== player
const vRef = () => $("#v-ref"), vUser = () => $("#v-user");
const master = () => (R.master === "ref" ? vRef() : vUser());
const slave = () => (R.master === "ref" ? vUser() : vRef());
function currentRefTime() { return R.master === "ref" ? vRef().currentTime : userToRef(vUser().currentTime); }
function seekRef(t) {
  const [A, B] = R.rep.scope.ref_range;
  t = clamp(t, A, B - 0.01);
  if (R.master === "ref") vRef().currentTime = t; else vUser().currentTime = refToUser(t);
  syncSlave(true);
  followView(t);
  tick(true);
}
function seekUser(u) {   // practitioner-master seek: keeps the exact repetition clicked
  const [A, B] = R.rep.scope.ref_range;
  vUser().currentTime = clamp(u, refToUser(A), refToUser(B - 0.01));
  syncSlave(true); followView(currentRefTime()); tick(true);
}
function setMaster(m) {
  if (R.master === m) return;
  const t = currentRefTime(); master().pause(); R.master = m; $("#master").value = m; slave().pause();
  vRef().muted = m !== "ref"; vUser().muted = m === "ref"; master().playbackRate = +$("#rate").value; seekRef(t);
}
function followView(t) {
  const [a, b] = R.view, span = b - a;
  if (t < a || t > b) setView(t - span * 0.2, t + span * 0.8);
}
let playerInit = false;
function setupPlayer() {
  vRef().src = R.rep.reference.video_url; vUser().src = R.rep.user.video_url;
  vRef().muted = true; vUser().muted = false;
  if (playerInit) return;
  playerInit = true;
  setupTimeline();
  $("#play").onclick = togglePlay;
  $("#step-back").onclick = () => seekRef(currentRefTime() - 0.5);
  $("#step-fwd").onclick = () => seekRef(currentRefTime() + 0.5);
  $("#rate").onchange = () => { master().playbackRate = +$("#rate").value; };
  $("#master").onchange = () => setMaster($("#master").value);
  for (const id of ["#ly-skel", "#ly-joint", "#ly-arrow"]) $(id).onchange = () => { drawOverlays(); drawNorm(); };
  $$("#norm-mode button").forEach((b) => b.addEventListener("click", () => {
    R.normMode = b.dataset.m; $$("#norm-mode button").forEach((x) => x.classList.toggle("on", x === b)); drawNorm();
  }));
  for (const v of [vRef(), vUser()]) {
    v.addEventListener("play", () => { if (v === master()) { $("#play").textContent = "⏸ 暂停"; slave().play().catch(() => {}); loop(); } });
    v.addEventListener("pause", () => { if (v === master()) { $("#play").textContent = "▶ 播放"; slave().pause(); syncSlave(true); } });
    v.addEventListener("seeked", () => tick(true));
    v.addEventListener("loadeddata", () => tick(true));
  }
  $$(".layer-toggle").forEach((b) => b.addEventListener("click", () => {
    const layer = b.closest(".layer"), c = layer.classList.toggle("collapsed");
    b.textContent = c ? "▸" : "▾"; b.setAttribute("aria-expanded", String(!c)); redrawAll();
  }));
  window.addEventListener("resize", redrawAll);
  document.addEventListener("keydown", (e) => {
    if ($("#view-report").classList.contains("hidden") || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    else if (e.code === "ArrowLeft") { e.preventDefault(); seekRef(currentRefTime() - 0.5); }
    else if (e.code === "ArrowRight") { e.preventDefault(); seekRef(currentRefTime() + 0.5); }
  });
  const ro = new ResizeObserver(() => redrawAll());
  ro.observe($("#tl-panel"));
}
function togglePlay() {
  const m = master();
  if (m.paused) {
    const [A, B] = R.rep.scope.ref_range;
    if (currentRefTime() >= B - 0.1) seekRef(A);   // restart at the beginning of the analysed range
    m.play();
  } else m.pause();
}
function syncSlave(hard) {
  const m = master(), s = slave();
  const target = R.master === "ref" ? refToUser(m.currentTime) : userToRef(m.currentTime);
  if (!isFinite(target)) return;
  const err = s.currentTime - target;
  if (hard || Math.abs(err) > 0.35) { s.currentTime = Math.max(0, target); s.playbackRate = m.playbackRate; return; }
  const h = 1.0, f = R.master === "ref" ? refToUser : userToRef;
  const slope = (f(m.currentTime + h) - f(m.currentTime - h)) / (2 * h);
  const rate = clamp(m.playbackRate * slope - err * 0.8, 0.25, 4);
  if (Math.abs(s.playbackRate - rate) > 0.02) s.playbackRate = rate;
}
function loop() {
  cancelAnimationFrame(R.raf);
  const step = () => {
    if (master().paused) return;
    const t = currentRefTime(), [, B] = R.rep.scope.ref_range;
    if (t >= B - 0.02) { master().pause(); return; }   // playback stays inside the analysed range
    syncSlave(false);
    const [a, b] = R.view;
    if (t > b - (b - a) * 0.05 || t < a) setView(t - (b - a) * 0.1, t + (b - a) * 0.9);   // loops jump back
    tick(false);
    R.raf = requestAnimationFrame(step);
  };
  R.raf = requestAnimationFrame(step);
}
let lastHeavy = -1;
function tick(force) {
  if (!R.rep || !R.feat) return;
  const t = currentRefTime();
  const lab = repLabel();
  $("#clock").textContent = lab ? `${lab} · 练习者 ${vUser().currentTime.toFixed(1)} 秒 ↔ 标准 ${t.toFixed(1)} 秒`
    : `练习者 ${vUser().currentTime.toFixed(1)} 秒 ↔ 标准 ${t.toFixed(1)} 秒`;
  $("#cap-ref").textContent = `${vRef().currentTime.toFixed(1)} s`;
  $("#cap-user").textContent = `${vUser().currentTime.toFixed(1)} s${lab ? " · " + lab.split(" ↔ ")[0].replace("练习者", "") : ""}`;
  const i = nearestIdx(R.T, t), si = R.segOf[i];
  if (si >= 0 && R.rep.segments[si].id !== R.seg) { R.seg = R.rep.segments[si].id; renderChips(); renderFocus(); renderHeatSel(); }
  drawTimeline(); drawOverlays();
  if (force || Math.abs(t - lastHeavy) > 0.09) { lastHeavy = t; drawNorm(); drawCurve(); renderFocusValues(); renderNow(); }
}
function redrawAll() { if (!R.rep || !R.feat) return; drawTimeline(); drawOverlays(); drawNorm(); drawCurve(); renderFocusValues(); renderNow(); }

// ================================================================== picture layer
function featureJointNames(side) {
  const f = featBy(R.feat);
  let js = f.joints;
  if (side === "user" && R.rep.left_right.applied)
    js = js.map((j) => j.startsWith("left_") ? "right_" + j.slice(5) : j.startsWith("right_") ? "left_" + j.slice(6) : j);
  return js;
}
function drawOverlay(side, video, canvas) {
  if (!canvas.clientWidth) return;
  const [g, W, H] = ctx2d(canvas);
  const t = currentRefTime(), outside = !inRange(t);
  $(side === "ref" ? "#out-ref" : "#out-user").classList.toggle("hidden", !outside);
  if (outside || !video.videoWidth || !$("#ly-skel").checked) return;
  const P = R.poses[side];
  if (!P.t.length) return;
  const i = nearestIdx(P.t, video.currentTime);
  if (Math.abs(P.t[i] - video.currentTime) > 0.25) return;
  const [fw, fh] = P.frame_size, sc = Math.min(W / fw, H / fh), ox = (W - fw * sc) / 2, oy = (H - fh * sc) / 2;
  const xy = P.xy[i], cf = P.conf[i], thr = P.conf_thr ?? 50;
  const idx = Object.fromEntries(P.joints.map((j, k) => [j, k]));
  const filled = filledAt(P, i);                    // filled across a short gap: drawn dashed / hollow, not as observed
  const pt = (k) => (k === undefined || xy[2 * k] < 0 || (cf[k] < thr && !filled.has(k)) ? null : [ox + xy[2 * k] * sc, oy + xy[2 * k + 1] * sc]);
  const idc = css(side === "ref" ? "--ref" : "--user");
  g.lineCap = "round";
  for (const [w, c] of [[4.5, "rgba(0,0,0,0.55)"], [2.2, idc]]) {   // dark halo keeps the colour readable on any frame
    g.strokeStyle = c; g.lineWidth = w;
    for (const [p, q] of P.edges) {
      const A = pt(p), B = pt(q);
      if (!A || !B) continue;
      g.setLineDash(filled.has(p) || filled.has(q) ? [5, 4] : []);
      g.beginPath(); g.moveTo(...A); g.lineTo(...B); g.stroke();
    }
  }
  g.setLineDash([]);
  g.fillStyle = idc; g.strokeStyle = idc; g.lineWidth = 1.5;
  P.joints.forEach((j, k) => {
    if (/eye|ear|mouth|face|hand|thumb|pinky|index$/.test(j) && !/foot_index/.test(j)) return;
    const A = pt(k);
    if (!A) return;
    g.beginPath(); g.arc(A[0], A[1], filled.has(k) ? 3.2 : 2.6, 0, 7);
    filled.has(k) ? g.stroke() : g.fill();
  });
  if (!$("#ly-joint").checked) return;
  // highlight the joints used by the selected indicator; angle features get an arc + value
  const f = featBy(R.feat), names = featureJointNames(side), pts = names.map((n) => pt(idx[n]));
  const ti = nearestIdx(R.T, t), st = statusAt(ti, R.feat);
  const col = idc;
  g.strokeStyle = "#ffffff"; g.lineWidth = 6.5;
  if (f.unit === "deg" && f.joints.length === 3 && pts.every(Boolean)) { g.beginPath(); g.moveTo(...pts[0]); g.lineTo(...pts[1]); g.lineTo(...pts[2]); g.stroke(); }
  g.strokeStyle = col; g.lineWidth = 4;
  if (f.unit === "deg" && f.joints.length === 3 && pts.every(Boolean)) {
    g.beginPath(); g.moveTo(...pts[0]); g.lineTo(...pts[1]); g.lineTo(...pts[2]); g.stroke();
    const a1 = Math.atan2(pts[0][1] - pts[1][1], pts[0][0] - pts[1][0]), a2 = Math.atan2(pts[2][1] - pts[1][1], pts[2][0] - pts[1][0]);
    let d = a2 - a1; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
    g.lineWidth = 2; g.beginPath(); g.arc(pts[1][0], pts[1][1], 18, a1, a1 + d, d < 0); g.stroke();
    const v = R.rep.series[side === "ref" ? "ref" : "user"][R.feat][ti];
    if (v !== null) {
      const lab = side === "ref" ? "标准" : "练习者";
      const stTxt = side === "user" && st !== "out" ? ` · ${R.rep.status_labels[st] || { missing: "缺失" }[st] || ""}` : "";
      g.font = "700 13px system-ui"; const txt = `${lab} ${Math.round(v)}°${stTxt}`, tw = g.measureText(txt).width;
      const lx = clamp(pts[1][0] + 22, 4, W - tw - 10), ly = clamp(pts[1][1] - 10, 18, H - 6);
      g.fillStyle = col; g.beginPath(); g.roundRect(lx - 5, ly - 14, tw + 10, 20, 5); g.fill();
      g.fillStyle = "#ffffff"; g.fillText(txt, lx, ly);
    }
  }
  for (const p of pts) if (p) { g.strokeStyle = "#ffffff"; g.lineWidth = 4.5; g.beginPath(); g.arc(p[0], p[1], 7, 0, 7); g.stroke(); g.strokeStyle = col; g.lineWidth = 2.5; g.stroke(); }
}
function drawOverlays() { if (!R.rep) return; drawOverlay("ref", vRef(), $("#c-ref")); drawOverlay("user", vUser(), $("#c-user")); }

const NJ = () => R.rep.normalized.joints;
const pose2 = (flat) => (flat ? NJ().map((_, j) => (flat[2 * j] === null ? null : [flat[2 * j], flat[2 * j + 1]])) : null);
function drawNorm() {
  if (!R.rep) return;
  const cv = $("#norm"); if (!cv.clientWidth) return;
  const [g, W, H] = ctx2d(cv);
  const nz = R.rep.normalized, t = currentRefTime(), outside = !inRange(t);
  $("#out-norm").classList.toggle("hidden", !outside);
  if (outside) return;
  let ref, usr, ell = null, note;
  if (R.normMode === "seg") {
    const sf = nz.segments.find((s) => s.segment === R.seg);
    if (!sf) return;
    ref = pose2(sf.ref_pose); usr = pose2(sf.user_pose); ell = sf.ellipses;
    note = `本段 ${sf.samples} 个定势时刻的平均姿态；虚线椭圆 = 这些时刻偏差的分布（1 倍标准差），不是测量精度。`;
  } else {
    const i = nearestIdx(R.T, t); ref = pose2(nz.ref[i]);
    if (R.master === "user" && nz.user_own) {   // the practitioner frame on screen vs the reference frame it maps to
      const j = Math.round((vUser().currentTime - nz.user_own_t0) / nz.user_own_dt);
      usr = pose2(nz.user_own[clamp(j, 0, nz.user_own.length - 1)]);
    } else usr = pose2(nz.user[i]);
    note = "已减少体型与拍摄距离的影响，仍受二维视角与识别误差影响。灰色箭头 = 从标准示范（蓝）指向练习者（品红）的位置偏移，不是要求移动的方向。";
  }
  $("#norm-note").textContent = note;
  const box = { x0: -1.9, x1: 1.9, y0: -2.7, y1: 2.4 };
  const sc = Math.min(W / (box.x1 - box.x0), H / (box.y1 - box.y0));
  const cx = W / 2, cy = (-box.y0) * sc + (H - (box.y1 - box.y0) * sc) / 2;
  const P = (p) => [cx + p[0] * sc, cy + p[1] * sc];
  g.strokeStyle = css("--line"); g.lineWidth = 1;
  for (let v = -1.5; v <= 1.5; v += 0.5) { const x = cx + v * sc; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); }
  for (let v = -2.5; v <= 2; v += 0.5) { const y = cy + v * sc; g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke(); }
  g.strokeStyle = css("--ink2"); g.lineWidth = 2; g.beginPath(); g.moveTo(10, H - 12); g.lineTo(10 + sc, H - 12); g.stroke();
  g.fillStyle = css("--ink2"); g.font = "11px system-ui"; g.fillText("1 个躯干长", 10, H - 18);
  const skel = (pose, color, w) => {
    if (!pose) return;
    g.strokeStyle = color; g.lineWidth = w; g.lineCap = "round";
    for (const [a, b] of nz.edges) if (pose[a] && pose[b]) { g.beginPath(); g.moveTo(...P(pose[a])); g.lineTo(...P(pose[b])); g.stroke(); }
    if (pose[0] && pose[1] && pose[2]) { g.beginPath(); g.moveTo(...P([(pose[1][0] + pose[2][0]) / 2, (pose[1][1] + pose[2][1]) / 2])); g.lineTo(...P(pose[0])); g.stroke(); }
    g.fillStyle = color; pose.forEach((p) => { if (p) { g.beginPath(); g.arc(...P(p), 3, 0, 7); g.fill(); } });
  };
  skel(ref, css("--ref"), 3.5); skel(usr, css("--user"), 3);
  g.font = "700 12px system-ui";
  [["标准示范", "--ref"], ["练习者", "--user"]].forEach(([lab, v], k) => {
    g.fillStyle = css(v); g.fillRect(10, 12 + k * 18, 16, 4); g.fillText(lab, 32, 17 + k * 18);
  });
  // rings on joints used by the selected indicator
  if ($("#ly-joint").checked) {
    const ring = new Set(featBy(R.feat).joints);
    g.strokeStyle = css("--ink"); g.lineWidth = 2;
    NJ().forEach((j, k) => { if (ring.has(j) && usr && usr[k]) { g.beginPath(); g.arc(...P(usr[k]), 8, 0, 7); g.stroke(); } });
  }
  if (ell) {
    g.strokeStyle = css("--user"); g.globalAlpha = 0.7; g.lineWidth = 1.2; g.setLineDash([3, 3]);
    ell.forEach((e, j) => { if (!e || !ref || !ref[j]) return; const c = P([ref[j][0] + e.cx, ref[j][1] + e.cy]);
      g.beginPath(); g.ellipse(c[0], c[1], Math.max(1, e.a * sc), Math.max(1, e.b * sc), (e.angle * Math.PI) / 180, 0, 7); g.stroke(); });
    g.setLineDash([]); g.globalAlpha = 1;
  }
  if ($("#ly-arrow").checked && ref && usr) {
    g.strokeStyle = css("--arrow"); g.fillStyle = css("--arrow"); g.lineWidth = 2;
    const mags = [];
    ref.forEach((p, j) => {
      const q = usr[j]; if (!p || !q) return;
      const A = P(p), B = P(q), m = Math.hypot(q[0] - p[0], q[1] - p[1]); mags.push([j, m, B]);
      if (m < 0.03) return;
      g.beginPath(); g.moveTo(...A); g.lineTo(...B); g.stroke();
      const ang = Math.atan2(B[1] - A[1], B[0] - A[0]);
      g.beginPath(); g.moveTo(...B); g.lineTo(B[0] - 7 * Math.cos(ang - 0.45), B[1] - 7 * Math.sin(ang - 0.45)); g.lineTo(B[0] - 7 * Math.cos(ang + 0.45), B[1] - 7 * Math.sin(ang + 0.45)); g.closePath(); g.fill();
    });
    g.font = "12px system-ui"; g.fillStyle = css("--ink");
    mags.sort((a, b) => b[1] - a[1]).slice(0, 2).forEach(([j, m, B]) => {
      if (m < 0.08) return;
      const txt = `${JOINT_CN[NJ()[j]]} ${m.toFixed(2)}`, tw = g.measureText(txt).width;
      g.fillText(txt, clamp(B[0] + 6, 4, W - tw - 4), clamp(B[1] - 6, 14, H - 4));
    });
  }
}

// ================================================================== curve layer
function renderChips() {
  const box = $("#feat-chips"); box.replaceChildren();
  const seg = segById(R.seg); if (!seg) return;
  const feats = [...R.rep.features].sort((a, b) => (seg.features[b.name].focus - seg.features[a.name].focus));
  for (const f of feats) {
    const c = seg.features[f.name];
    box.append(el("button", { class: `chip${c.focus ? " key" : ""}${f.name === R.feat ? " on" : ""}`, title: `${R.rep.status_labels[c.status]}${c.reason ? "：" + c.reason : ""}`,
      onclick: () => select(R.seg, f.name) }, el("i", { style: `background:${c.status === "na" ? css("--axis") : css(STATUS_VAR[c.status])}` }), f.label));
  }
}
function drawCurve() {
  const cv = $("#curve"); if (!R.rep || !cv.clientWidth) return;
  const [g, W, H] = ctx2d(cv);
  TL.W = TL.W || W;
  const X = (t) => TL.G + ((t - R.view[0]) / (R.view[1] - R.view[0])) * (W - TL.G - TL.PAD);
  const f = featBy(R.feat), t = R.T, ref = R.rep.series.ref[R.feat], usr = userSeries(R.feat);
  const i0 = Math.max(0, nearestIdx(t, R.view[0]) - 1), i1 = Math.min(t.length - 1, nearestIdx(t, R.view[1]) + 1);
  let lo = Infinity, hi = -Infinity;
  for (let i = i0; i <= i1; i++) for (const v of [ref[i], usr[i]]) if (v !== null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!isFinite(lo)) { g.fillStyle = css("--muted"); g.fillText("此范围内无有效数据", TL.G + 10, 20); return; }
  const tol = f.tolerance.good; lo -= tol; hi += tol;
  const P = { t: 8, b: 20 };
  const Y = (y) => P.t + (1 - (y - lo) / (hi - lo)) * (H - P.t - P.b);
  R.curveGeom = { X, W };
  g.fillStyle = css("--track"); g.fillRect(TL.G, P.t, W - TL.G - TL.PAD, H - P.t - P.b);
  // n/a (not observable) and segment boundaries
  R.rep.segments.forEach((s) => {
    const a = clamp(X(s.ref_start), TL.G, W - TL.PAD), b = clamp(X(s.ref_end), TL.G, W - TL.PAD);
    if (s.features[R.feat].status === "na" && b > a) {
      g.save(); g.beginPath(); g.rect(a, P.t, b - a, H - P.t - P.b); g.clip();
      g.strokeStyle = css("--axis"); g.lineWidth = 1;
      for (let x = a - H; x < b; x += 8) { g.beginPath(); g.moveTo(x, H - P.b); g.lineTo(x + H, P.t); g.stroke(); }
      g.restore();
    }
    g.strokeStyle = css("--axis"); g.setLineDash([3, 3]); g.beginPath(); g.moveTo(a, P.t); g.lineTo(a, H - P.b); g.stroke(); g.setLineDash([]);
    if (b - a > 40) { g.fillStyle = s.id === R.seg ? css("--ink") : css("--muted"); g.font = `${s.id === R.seg ? "700 " : ""}11px system-ui`; g.fillText(s.name.replace(/^第(.)式 /, "$1 "), a + 4, P.t + 10, b - a - 8); }
  });
  // tolerance band around the reference
  g.fillStyle = css("--line"); g.globalAlpha = 0.9;
  let run = [];
  const flush = () => { if (run.length > 1) { g.beginPath(); run.forEach(([x, y]) => g.lineTo(x, Y(y + tol))); [...run].reverse().forEach(([x, y]) => g.lineTo(x, Y(y - tol))); g.closePath(); g.fill(); } run = []; };
  for (let i = i0; i <= i1; i++) { if (ref[i] === null || statusAt(i, R.feat) === "na") flush(); else run.push([X(t[i]), ref[i]]); }
  flush(); g.globalAlpha = 1;
  const line = (arr, color, w) => {
    g.strokeStyle = color; g.lineWidth = w; g.beginPath(); let open = false;
    for (let i = i0; i <= i1; i++) { const v = arr[i]; if (v === null || statusAt(i, R.feat) === "na") { open = false; continue; } open ? g.lineTo(X(t[i]), Y(v)) : g.moveTo(X(t[i]), Y(v)); open = true; }
    g.stroke();
  };
  line(ref, css("--ref"), 2); line(usr, css("--user"), 2.2);
  const seg = segById(R.seg);
  if (seg) { g.fillStyle = css("--ink"); for (const h of seg.hold_times) { const x = X(h); if (x < TL.G || x > W - TL.PAD) continue; g.beginPath(); g.moveTo(x, H - P.b + 2); g.lineTo(x - 4, H - P.b + 9); g.lineTo(x + 4, H - P.b + 9); g.fill(); } }
  // y labels in the gutter (aligned with the timeline gutter)
  g.fillStyle = css("--ink2"); g.font = "11px system-ui"; g.textAlign = "right";
  const fv = (v) => (f.unit === "deg" ? `${Math.round(v)}°` : v.toFixed(2));
  g.fillText(fv(hi), TL.G - 6, P.t + 8); g.fillText(fv(lo), TL.G - 6, H - P.b);
  g.textAlign = "left"; g.font = "700 12px system-ui"; g.fillStyle = css("--ink"); g.fillText(f.label, 6, (H - P.b) / 2 + 4, TL.G - 50);
  g.font = "10px system-ui"; g.fillStyle = css("--muted"); g.fillText("▲ 定势时刻", W - TL.PAD - 60, H - 4);
  // playhead + hover
  const xp = X(currentRefTime());
  if (xp >= TL.G && xp <= W - TL.PAD) { g.strokeStyle = css("--ink"); g.lineWidth = 2; g.beginPath(); g.moveTo(xp, P.t); g.lineTo(xp, H - P.b); g.stroke(); }
  const tip = $("#curve-tip");
  if (R.hoverX === null || R.hoverX < TL.G) { tip.classList.add("hidden"); return; }
  const th = R.view[0] + ((R.hoverX - TL.G) / (W - TL.G - TL.PAD)) * (R.view[1] - R.view[0]), i = nearestIdx(t, th);
  g.strokeStyle = css("--axis"); g.lineWidth = 1; g.beginPath(); g.moveTo(X(t[i]), P.t); g.lineTo(X(t[i]), H - P.b); g.stroke();
  for (const [arr, col] of [[ref, "--ref"], [usr, "--user"]]) if (arr[i] !== null) { g.fillStyle = css(col); g.strokeStyle = css("--card"); g.lineWidth = 2; g.beginPath(); g.arc(X(t[i]), Y(arr[i]), 4, 0, 7); g.fill(); g.stroke(); }
  const st = statusAt(i, R.feat), d = ref[i] !== null && usr[i] !== null ? usr[i] - ref[i] : null;
  tip.replaceChildren(el("b", {}, `标准时间 ${t[i].toFixed(1)} 秒`),
    el("div", {}, el("span", { class: "ref" }, "● 标准 "), fmtVal(f, ref[i], true), "　", el("span", { class: "user" }, "● 练习者 "), fmtVal(f, usr[i], true)),
    el("div", {}, `差值 ${fmtSigned(f, d)} · ${R.rep.status_labels[st] || { missing: "缺失", out: "区间外" }[st]}`));
  tip.classList.remove("hidden");
  tip.style.left = `${Math.min(W - 200, X(t[i]) + 10)}px`; tip.style.top = "6px";
}
$("#curve").addEventListener("mousemove", (e) => { R.hoverX = e.clientX - e.target.getBoundingClientRect().left; drawCurve(); });
$("#curve").addEventListener("mouseleave", () => { R.hoverX = null; drawCurve(); });
$("#curve").addEventListener("click", (e) => {
  if (!R.curveGeom) return;
  const x = e.clientX - e.target.getBoundingClientRect().left, W = R.curveGeom.W;
  if (x < TL.G) return;
  seekRef(R.view[0] + ((x - TL.G) / (W - TL.G - TL.PAD)) * (R.view[1] - R.view[0]));
});

// ================================================================== focus + details
function renderFocus() {
  const seg = segById(R.seg), f = featBy(R.feat); if (!seg || !f) return;
  const c = seg.features[R.feat];
  $("#focus-sub").textContent = `${seg.name} · ${f.label}${c.focus ? " · 本段重点" : ""}`;
  $("#view-sub").textContent = `当前指标：${f.label}（${seg.name}）`;
  $("#focus-status").replaceChildren(el("span", { class: `sym ${c.status}` }, STATUS_GLYPH[c.status] || "?"),
    `${R.rep.status_labels[c.status]} · ${f.label}`);
  $("#focus-text").textContent = c.text || c.reason || "";
  const tolCm = toCm(f, f.tolerance.good);
  const def = `${f.desc}。允许偏差 ±${f.unit === "deg" ? f.tolerance.good + "°" : f.tolerance.good + (tolCm !== null ? `（约 ±${Math.round(tolCm)} cm）` : "")}；`
    + (c.hold_n ? (seg.reps ? `本段典型偏差 = ${c.hold_valid}/${c.hold_n} 次对应上的重复中，与标准定势同一动作相位区间的差值中位数；有效覆盖 ${pct(c.coverage)}。`
      : `本段典型偏差按 ${c.hold_valid}/${c.hold_n} 个有效定势时刻计算，整段有效覆盖 ${pct(c.coverage)}。`) : "")
    + scaleNote(f) + (seg.flags.length ? " ⚠ " + seg.flags.join("；") : "") + (seg.note ? " 注：" + seg.note : "");
  $("#focus-def").textContent = def;
  renderFocusValues();
}
function renderFocusValues() {
  const seg = segById(R.seg), f = featBy(R.feat); if (!seg || !f) return;
  const i = nearestIdx(R.T, currentRefTime()), a = R.rep.series.ref[R.feat][i], b = userSeries(R.feat)[i];
  const c = seg.features[R.feat];
  const box = $("#focus-values");
  box.replaceChildren(
    el("div", { class: "ref" }, el("dt", {}, "此刻 · 标准示范"), el("dd", {}, fmtVal(f, a, true))),
    el("div", { class: "user" }, el("dt", {}, "此刻 · 练习者"), el("dd", {}, fmtVal(f, b, true))),
    el("div", {}, el("dt", {}, "此刻 · 差值"), el("dd", {}, statusAt(i, R.feat) === "na" ? "不可测" : fmtSigned(f, a !== null && b !== null ? b - a : null))),
    el("div", {}, el("dt", {}, "本段典型偏差"), el("dd", {}, c.value === undefined ? "—" : fmtVal(f, c.value, true))),
    el("div", {}, el("dt", {}, "本段用时"), el("dd", {}, seg.duration_ratio ? `${seg.duration_ratio.toFixed(2)} 倍` : "—")),
    ...(seg.reps ? [el("div", { title: "练习者的每一次重复都与标准示范的对应重复比较；标准示范了左右两侧时按同侧对应" }, el("dt", {}, "重复次数"),
      el("dd", {}, seg.reps.repeated === false ? "非重复动作（整段对应）"
        : `标准 ${seg.reps.count_ref} · 练习者 ${seg.reps.count_user}${seg.reps.alternating ? "（左右交替）" : ""}`))] : []),
    el("div", {}, el("dt", {}, "对应可信度"), el("dd", {}, confidence(seg.align_cost))));
}
function renderNow() {
  const tb = $("#now"); if (!R.rep || !tb.closest("details").open) return;
  const i = nearestIdx(R.T, currentRefTime());
  tb.replaceChildren(el("tr", {}, el("th", {}, "指标"), el("th", { class: "ref" }, "标准示范"), el("th", { class: "user" }, "练习者"), el("th", {}, "差值（练习者 − 标准）"), el("th", {}, "状态")));
  for (const f of R.rep.features) {
    const a = R.rep.series.ref[f.name][i], b = userSeries(f.name)[i], st = statusAt(i, f.name);
    tb.append(el("tr", { "data-f": f.name, class: f.name === R.feat ? "sel" : "", onclick: () => select(R.seg, f.name) },
      el("td", {}, f.label), el("td", { class: "ref" }, fmtVal(f, a, true)), el("td", { class: "user" }, fmtVal(f, b, true)),
      el("td", {}, st === "na" ? "不可测" : fmtSigned(f, a !== null && b !== null ? b - a : null)),
      el("td", {}, el("span", { class: `sym ${st === "missing" ? "unmeasurable" : st}`, style: "width:16px;height:16px;font-size:10px" }, STATUS_GLYPH[st] || ""))));
  }
}
function renderDetails() {
  const rep = R.rep;
  // matrix
  const tb = $("#heat"); tb.replaceChildren();
  const head = el("tr", {}, el("th", {}, ""));
  for (const s of rep.segments) head.append(el("th", { class: "col", "data-seg": s.id, title: s.name, onclick: () => { select(s.id, R.feat); zoomTo(s); } }, s.name.replace(/^第(.)式 /, "$1 ")));
  tb.append(head);
  for (const f of rep.features) {
    const tr = el("tr", {}, el("th", { style: "text-align:left", title: f.desc }, f.label));
    for (const s of rep.segments) {
      const c = s.features[f.name], glyph = c.status === "warn" ? "!" : c.status === "bad" ? "!!" : c.status === "uncertain" ? "≈" : c.status === "imprecise" ? "±" : "";
      tr.append(el("td", { class: `st-${c.status}${c.focus ? " focus" : ""}`, "data-k": `${s.id}|${f.name}`,
        title: `${s.name} · ${f.label}\n${rep.status_labels[c.status]}${c.reason ? "：" + c.reason : ""}${c.text ? "\n" + c.text : ""}`,
        onclick: () => { select(s.id, f.name); zoomTo(s); seekRef((s.hold_times[0] ?? s.ref_start) + 0.01); } },
        (c.value !== undefined ? fmtVal(f, c.value) : "") + glyph));
    }
    tb.append(tr);
  }
  const lg = $("#legend"); lg.replaceChildren();
  for (const [s, lab] of [["good", "达标"], ["warn", "需注意 !"], ["bad", "偏差明显 !!"], ["uncertain", "对应不可靠 ≈"], ["imprecise", "测量不确定 ±"], ["unmeasurable", "不可评价"], ["na", "此视角不可测"]])
    lg.append(el("span", {}, el("i", { class: `st-${s}` }), lab));
  lg.append(el("span", {}, "粗框 = 本段重点"));
  // patterns
  const pat = rep.patterns;
  $("#pat-meta").textContent = `${pat.share_definition}。未分类（可测指标不足）：${pct(pat.unclassified_share)}；离群时刻（可能为识别错误，未归入任何模式）：${pct(pat.outlier_share)}。` + (pat.k ? `轮廓系数 ${pat.silhouette.toFixed(2)}。` : "");
  const pc = $("#pat-cards"); pc.replaceChildren();
  for (const p of pat.items) {
    const color = p.id === 0 ? css("--axis") : css(SERIES[(p.id - 1) % SERIES.length]);
    pc.append(el("div", { class: "pat-card" },
      el("h3", {}, el("span", { class: "chipc", style: `background:${color}` }), p.id === 0 ? "已测指标在允许范围内" : `偏差模式 ${p.id}`),
      el("div", { class: "label" }, p.label),
      el("div", { class: "share" }, `标准进程中占 ${pct(p.share)} · 练习者时长占 ${pct(p.user_share)}`),
      el("div", { class: "share" }, "主要出现在：" + Object.entries(p.segments).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([id, v]) => `${segName(id).replace(/^第.式 /, "")} ${pct(v)}`).join("、")),
      el("p", {}, el("button", { onclick: () => { seekRef(p.example_t); window.scrollTo({ top: 0, behavior: "smooth" }); } }, `定位典型时刻（${p.example_t.toFixed(1)} 秒）`))));
  }
  // scope
  const sc = rep.scope;
  $("#scope-reason").textContent = sc.reason;
  const st = $("#scope-table"); st.replaceChildren();
  for (const c of sc.candidates || []) st.append(el("tr", { class: sc.segments.includes(c.segment) ? "sel" : "" },
    el("td", { class: c.eligible ? "ok" : "no" }, c.eligible ? "✓" : "✗"), el("td", {}, c.name),
    el("td", {}, `${c.user_range[0].toFixed(0)}–${c.user_range[1].toFixed(0)} 秒`),
    el("td", {}, c.eligible ? (sc.segments.includes(c.segment) ? "已分析" : "合格") : c.reasons.join("；"))));
  if (!(sc.candidates || []).length) st.append(el("tr", {}, el("td", {}, "手动指定区间，未做自动检查。")));
  // glossary + tech
  const gl = $("#glossary"); gl.replaceChildren();
  for (const f of rep.features) gl.append(el("dt", {}, f.label), el("dd", {}, `${f.desc}。允许偏差 ±${f.unit === "deg" ? f.tolerance.good + "°" : f.tolerance.good}。`));
  gl.append(el("dt", {}, "定势时刻"), el("dd", {}, "每段中速度较低、且姿势最偏离自然站立的时刻（近似“定势”）；段内判定只用这些时刻，有效时刻不足则不判定。正式版本应由教练标注。"));
  gl.append(el("dt", {}, "对应关系"), el("dd", {}, "两人节奏和重复次数不同，系统把练习者的每个时刻对应到标准示范的同一动作阶段后再比较；时间轴中间的色带显示这种对应（越宽越慢）。"));
  $("#tech").textContent = JSON.stringify({ model: rep.model, scope: { ...sc, candidates: undefined }, options: rep.options, timing: rep.timing,
    timings_s: rep.timings_s, left_right: rep.left_right, counts: rep.summary.counts, schema: rep.schema }, null, 1);
  $("#now").closest("details").addEventListener("toggle", renderNow);
}
function renderHeatSel() {
  $$("#heat td.sel, #heat th.sel").forEach((x) => x.classList.remove("sel"));
  const td = document.querySelector(`#heat td[data-k="${R.seg}|${R.feat}"]`); if (td) td.classList.add("sel");
  const th = document.querySelector(`#heat th[data-seg="${R.seg}"]`); if (th) th.classList.add("sel");
}

// ================================================================== bench (developer)
async function openBench(id) {
  show("view-bench");
  const list = await api("/api/bench").catch(() => []);
  const box = $("#bench-list"); box.replaceChildren();
  if (!list.length) box.append(el("p", { class: "msg" }, "还没有对比结果。"));
  for (const b of list) box.append(el("a", { href: `#/bench/${b.id}` }, el("span", {}, `${b.id} · ${b.backends.join(", ")}`), el("span", { class: "st" }, b.created)));
  const body = $("#bench-body"); body.replaceChildren();
  const pick = id || (list[0] && list[0].id);
  if (!pick) return;
  const b = await api(`/api/bench/${pick}`);
  const sc = b.scope || {}, fmt = (x, d = 0) => (x === null || x === undefined ? "—" : x.toFixed(d));
  const t1 = el("table", { class: "bench-table" }, el("tr", {}, ...["算法", "设备", "ms/帧（标准/练习者）", "检测率", "脚部可见", "抖动×1000", "对应可信度", "达标率", "与基线结论一致", "授权"].map((h) => el("th", {}, h))));
  for (const r of b.rows) {
    if (r.error) { t1.append(el("tr", {}, el("td", {}, r.backend), el("td", { colspan: 9 }, "失败：" + r.error))); continue; }
    const ag = b.agreement[r.backend];
    t1.append(el("tr", {}, el("td", { title: r.label }, r.backend), el("td", { title: r.device_note || "" }, r.device),
      el("td", {}, `${fmt(r.ms_per_frame.reference)} / ${fmt(r.ms_per_frame.user)}`), el("td", {}, `${pct(r.detection[0])} / ${pct(r.detection[1])}`),
      el("td", {}, `${pct(r.visible.reference.feet)} / ${pct(r.visible.user.feet)}`),
      el("td", {}, `${fmt(r.jitter[0] && r.jitter[0] * 1000, 1)} / ${fmt(r.jitter[1] && r.jitter[1] * 1000, 1)}`),
      el("td", {}, `${confidence(r.align_cost)}（${r.align_cost.toFixed(2)}）`), el("td", {}, pct(r.pass_rate)),
      el("td", {}, ag ? `${pct(ag.status_agreement)}（${ag.cells} 项）` : "基线"), el("td", {}, r.license)));
  }
  const same = Object.values(b.frame_check || {}).every((v) => v.ref_identical && v.user_identical);
  body.append(el("div", { class: "card" }, el("h2", {}, `结果 · ${b.user}`),
    el("p", { class: "hint" }, `区间：${(sc.segments || []).join("–")}；${b.sample_fps} fps；输入帧${same ? "完全一致" : "不一致（结论无效）"}。${(b.definitions || {}).jitter || ""}`),
    el("div", { class: "tbl-wrap" }, t1)));
  for (const [key, title] of [["disagreement", "标准视频"], ["disagreement_user", "练习者视频"]]) {
    const d = b[key] || {}, pairs = Object.keys(d);
    if (!pairs.length) continue;
    const t2 = el("table", { class: "bench-table" }, el("tr", {}, el("th", {}, "指标"), el("th", {}, "容差"), ...pairs.map((p) => el("th", {}, p.replace("|", " vs ")))));
    for (const f of b.features) {
      t2.append(el("tr", {}, el("td", {}, f.label), el("td", {}, f.tolerance),
        ...pairs.map((p) => { const v = d[p][f.name]; if (!v) return el("td", {}, "—");
          const cls = v.over_tol < 0.02 ? "dis-0" : v.over_tol < 0.1 ? "dis-1" : "dis-2";
          const nd = f.unit === "deg" ? 1 : 2;
          return el("td", { class: cls, title: `中位 ${v.median.toFixed(2)}，P90 ${v.p90.toFixed(2)}，P95 ${v.p95.toFixed(2)}，n=${v.n}` }, `${v.median.toFixed(nd)} / ${v.p95.toFixed(nd)} / ${pct(v.over_tol)}`); })));
    }
    body.append(el("div", { class: "card" }, el("h2", {}, `同一帧上的算法间测量差异（${title}）`),
      el("p", { class: "hint" }, "格内：中位差 / P95 / 超过允许偏差的帧比例。浅绿 < 2%，黄 < 10%，红 ≥ 10%。这是算法之间的差异，不代表对真实角度的误差。"),
      el("div", { class: "tbl-wrap" }, t2)));
  }
}

// ================================================================== boot
(async () => {
  Object.assign(CFG, await api("/api/config").catch(() => ({})));
  $$(".dev-only").forEach((e) => e.classList.toggle("hidden", !CFG.dev_mode));
  $("#tasks-chip").onclick = () => { TASKS.open = !TASKS.open; renderTasks(); };
  document.addEventListener("click", (e) => { if (TASKS.open && !e.target.closest(".tasks")) { TASKS.open = false; renderTasks(); } });
  route();
  refreshTasks();   // picks up analyses already running on the server (other testers, or before a reload)
})();
