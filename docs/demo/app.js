// Sentinel UI. One file, no build step. Works against the live API or a recorded run (static mode).
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (x, d = 3) => (x == null || Number.isNaN(x) ? "-" : Number(x).toFixed(d));
const pc = (x, d = 0) => (x == null ? "-" : (x * 100).toFixed(d) + "%");
const sci = (x) => (x == null ? "-" : x === 0 ? "0" : Number(x).toExponential(0));
const riskTxt = (x) => (x >= 0.995 ? ">99%" : x < 0.005 ? "<1%" : pc(x));

const NODES = [
  ["data_steward", "Data Steward", "profile + quality gate"],
  ["task_selector", "Task & Model Selector", "rule-based framing"],
  ["feature_engineer", "Feature Engineer", "deterministic transforms"],
  ["trainer", "Trainer", "seeded Optuna, CV-ranked"],
  ["explainer", "Explainer", "Tree-SHAP contributions"],
  ["deployer_monitor", "Deployer / Monitor", "registry, service, drift"],
];
const TABS = [["data", "Data"], ["quality", "Quality gate"], ["board", "Leaderboard"], ["explain", "Explain"],
  ["fleet", "Fleet"], ["deploy", "Deploy"], ["monitor", "Monitor"]];

const S = {
  mode: "live", info: null, datasets: [], selected: null, budget: "fast", tab: "data", runId: null, status: "idle",
  events: [], stationNote: {}, stationState: {}, partial: {}, result: null, fleetSel: null, drift: null, retrain: null,
  driftSensor: null, sigmaIdx: 0, staticDrift: null, staticDeploy: null, busy: false, error: null, uploadKey: null, uploadName: null,
  ledgerCount: 0,
};
window.__S = S;

async function j(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.text()).slice(0, 300) || r.status);
  return r.json();
}

// ------------------------------------------------------------------ boot
async function boot() {
  try { S.info = await j("api/mode"); S.mode = "live"; } catch { S.mode = "static"; }
  const b = $("#modeBadge");
  if (S.mode === "live") {
    b.textContent = "Live app"; b.className = "badge live";
    S.datasets = await j("api/datasets");
    S.pastRuns = (await j("api/runs")).filter((r) => r.status === "done").slice(0, 6);
    S.selected = (S.datasets.find((d) => d.key === "cmapss_fd001" && d.rows) || S.datasets.find((d) => d.rows) || {}).key;
  } else {
    b.textContent = "Recorded run"; b.className = "badge rec";
    b.title = "Real pipeline output exported from a run. Run the Docker stack for live training and uploads.";
    S.result = await j("data/run.json");
    S.events = await j("data/events.json");
    S.staticDrift = await j("data/drift.json").catch(() => null);
    S.staticDeploy = await j("data/deploy.json").catch(() => null);
    S.datasets = [S.result.dataset];
    S.selected = S.result.dataset.key;
    S.runId = S.result.run_id;
    S.status = "done";
    S.events.forEach((e) => applyEvent(e, true));
    setHash();
    S.driftSensor = driftSensors()[0];
    S.tab = "quality";
    S.fleetSel = S.result.fleet[0]?.id;
  }
  drawStations(); drawTabs(); drawLedger(); render();
  const certBtn = $("#btnExportCert");
  if (certBtn) certBtn.onclick = exportAuditCertificate;
}

// ------------------------------------------------------------------ pipeline events
const SEEN = new Set();
function applyEvent(e, silent) {
  if (e.seq != null) { if (SEEN.has(e.seq)) return; SEEN.add(e.seq); }
  S.events.includes(e) || S.events.push(e);
  const st = S.stationState;
  if (e.status === "start") st[e.node] = "running";
  else if (e.status === "done" || e.status === "halt") { st[e.node] = e.status; S.stationNote[e.node] = e.title; }
  else if (e.status === "progress" && e.payload?.type === "trial") S.stationNote[e.node] = `${famName(e.payload.family)}: trial ${e.payload.trial}/${e.payload.of}`;
  const p = e.payload || {};
  if (p.scorecard) S.partial.scorecard = p.scorecard;
  if (p.framing) S.partial.framing = p.framing;
  if (p.leaderboard) S.partial.leaderboard = p.leaderboard;
  if (p.global) S.partial.global = p.global;
  if (!silent) { drawStations(); drawTabs(); appendLedger(e); if (S.tab === "data") render(); }
}
const famName = (f) => ({ lightgbm: "LightGBM", xgboost: "XGBoost", random_forest: "Random Forest", linear: "Linear baseline" }[f] || f);
const nodeLabel = (id) => (NODES.find((n) => n[0] === id) || [0, id])[1];

function drawStations() {
  $("#stations").innerHTML = NODES.map(([id, label, sub], i) => {
    const s = S.stationState[id] || "idle";
    return `<li class="st ${s}"><span class="dot">${s === "done" ? "&#10003;" : s === "halt" ? "!" : i + 1}</span>
      <div><b>${esc(label)}</b><small>${esc(S.stationNote[id] || sub)}</small></div></li>`;
  }).join("");
}

function setHash() {
  const h = S.result?.run_hash;
  $("#hashBadge").textContent = h ? `run hash: ${h.slice(0, 12)}...` : "run hash: none yet";
  $("#hashBadge").title = h ? `Full hash ${h}. Same data + seed + thresholds reproduce it.` : "";
}

function exportAuditCertificate() {
  const r = S.result;
  if (!r) { alert("Load or complete a pipeline run before exporting the run report."); return; }
  const h = r.run_hash || "not computed";
  const ds = r.dataset || {};
  const champ = r.champion || {};
  const lb = r.leaderboard || [];
  const sc = r.scorecard || {};
  const fr = r.framing || {};
  const drop = (r.feature_spec && r.feature_spec.dropped) || [];
  const hold = (m) => (m.holdout && m.holdout.all_rows) || {};
  const w = window.open("", "_blank");
  if (!w) { alert("The browser blocked the pop-up. Allow pop-ups for this site to open the run report."); return; }
  w.document.write(`<!doctype html>
<html><head><meta charset="utf-8"><title>Sentinel run report ${esc(h.slice(0, 10))}</title>
<style>
  body { font-family: "Segoe UI", system-ui, sans-serif; color: #101418; background: #fff; padding: 40px; max-width: 860px; margin: 0 auto; line-height: 1.5; font-size: 13px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h2 { font-size: 14px; margin: 24px 0 8px; border-bottom: 2px solid #101418; padding-bottom: 4px; text-transform: uppercase; letter-spacing: .04em; }
  .header { display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 3px solid #d8001b; padding-bottom: 16px; margin-bottom: 20px; }
  .hash { background: #eceeef; border: 1px solid #cdd3d8; padding: 8px 12px; font-family: Consolas, monospace; font-size: 12px; word-break: break-all; margin: 8px 0; }
  table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 12px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #e1e5e8; }
  th { background: #eceeef; font-size: 11px; text-transform: uppercase; }
  .note { font-size: 11px; color: #5b6670; margin-top: 20px; }
  @media print { body { padding: 0; } button { display: none; } }
</style></head><body>
<div class="header"><div><h1>Sentinel run report</h1>
  <div style="color:#5b6670">Generated in the browser from the loaded run. Nothing here is signed or certified.</div></div>
  <button onclick="window.print()" style="padding:8px 16px;background:#101418;color:#fff;border:none;cursor:pointer;font-weight:600">Print / save as PDF</button></div>
<b>Run hash (SHA-256 of data fingerprint, options, decision log and leaderboard)</b>
<div class="hash">${esc(h)}</div>
<h2>Data and framing</h2>
<table>
  <tr><td width="30%"><b>Dataset</b></td><td>${esc(ds.name || ds.key || "")}</td></tr>
  <tr><td><b>Rows</b></td><td>${ds.rows != null ? ds.rows.toLocaleString() : "-"}</td></tr>
  <tr><td><b>Task</b></td><td>${esc(fr.task || "-")}, target ${esc(fr.target_col || "-")}${fr.horizon ? `, alarm horizon ${fr.horizon} ${esc(r.time_noun || "step")}s` : ""}</td></tr>
  <tr><td><b>Engineered features</b></td><td>${r.feature_spec && r.feature_spec.n_features != null ? r.feature_spec.n_features : "-"}</td></tr>
  <tr><td><b>Dropped columns</b></td><td>${drop.length ? esc(drop.map((d) => (typeof d === "string" ? d : d.col || JSON.stringify(d))).join(", ")) : "none"}</td></tr>
</table>
<h2>Data quality scorecard</h2>
<table><thead><tr><th>Check</th><th>Status</th><th>Value</th><th>Threshold</th></tr></thead><tbody>
  ${(sc.checks || []).map((c) => `<tr><td>${esc(c.label)}</td><td>${esc(String(c.status).toUpperCase())}</td><td>${esc(String(c.value))}</td><td>${esc(c.threshold)}</td></tr>`).join("")}
</tbody></table>
<h2>Leaderboard (grouped cross-validation, then holdout)</h2>
<table><thead><tr><th>Model</th><th>CV ${esc((fr.metric || "roc_auc").replace("_", " ").toUpperCase())}</th><th>Std dev</th><th>Holdout ROC-AUC</th><th></th></tr></thead><tbody>
  ${lb.map((m) => `<tr><td><b>${esc(m.name)}</b></td><td>${fmt(m.cv_mean, 4)}</td><td>&plusmn;${fmt(m.cv_std, 4)}</td><td>${fmt(hold(m).roc_auc, 4)}</td><td>${m.family === champ.family ? "Champion" : ""}</td></tr>`).join("")}
</tbody></table>
<p>Champion: <b>${esc(champ.name || "-")}</b>, alert threshold ${fmt(champ.threshold, 4)}.</p>
<p class="note">Sentinel makes no language-model calls. The hash above changes if any input, threshold, decision or score changes; re-running the same data with the same seed and pinned library versions reproduces it.</p>
</body></html>`);
  w.document.close();
}

// ------------------------------------------------------------------ ledger
function entryHtml(d) {
  const inputs = Object.entries(d.inputs || {}).map(([k, v]) => `${k}=${typeof v === "number" ? +v.toFixed(4) : v}`).join("  ");
  return `<div class="entry"><div class="tag">${esc(d.rule)} <i>${esc(nodeLabel(d.node))}</i></div>
    <div class="why">${esc(d.outcome)}</div>${inputs ? `<div class="why" style="color:var(--steel)">${esc(inputs)}</div>` : ""}</div>`;
}
function drawLedger() {
  const t = $("#tape");
  const items = [];
  for (const e of S.events) {
    if (e.status === "start") items.push(`<div class="entry node"><div class="tag">&#9656; ${esc(nodeLabel(e.node).toUpperCase())}</div><div class="why">${esc(e.title)}</div></div>`);
    (e.decisions || []).forEach((d) => items.push(entryHtml({ node: e.node, ...d })));
  }
  S.ledgerCount = items.length;
  t.innerHTML = items.length ? items.join("") : `<p class="empty">Every rule the pipeline applies is written here as it fires. Start a run to see it.</p>`;
  $("#ledgerCount").textContent = `${S.ledgerCount} entries`;
}
function appendLedger(e) {
  const t = $("#tape");
  if (t.querySelector(".empty")) t.innerHTML = "";
  if (e.status === "start") { t.insertAdjacentHTML("beforeend", `<div class="entry node"><div class="tag">&#9656; ${esc(nodeLabel(e.node).toUpperCase())}</div><div class="why">${esc(e.title)}</div></div>`); S.ledgerCount++; }
  (e.decisions || []).forEach((d) => { t.insertAdjacentHTML("beforeend", entryHtml({ node: e.node, ...d })); S.ledgerCount++; });
  $("#ledgerCount").textContent = `${S.ledgerCount} entries`;
  t.scrollTop = t.scrollHeight;
}

// ------------------------------------------------------------------ tabs
function avail(id) {
  const r = S.result, p = S.partial;
  return { data: true, quality: !!(p.scorecard || r?.scorecard), board: !!(p.leaderboard || r?.leaderboard),
    explain: !!(p.global || r?.explain), fleet: !!r?.fleet, deploy: !!r?.serving, monitor: !!r?.monitor }[id];
}
function drawTabs() {
  $("#tabs").innerHTML = TABS.map(([id, label]) =>
    `<button class="tab" role="tab" data-id="${id}" aria-selected="${S.tab === id}" ${avail(id) ? "" : "disabled"}>${label}</button>`).join("");
  $("#tabs").querySelectorAll(".tab").forEach((b) => (b.onclick = () => { S.tab = b.dataset.id; drawTabs(); render(); }));
}

function render() {
  const v = $("#view");
  ({ data: viewData, quality: viewQuality, board: viewBoard, explain: viewExplain, fleet: viewFleet, deploy: viewDeploy, monitor: viewMonitor }[S.tab])(v);
}

// ------------------------------------------------------------------ Data
function viewData(v) {
  const staticMode = S.mode === "static";
  const cards = S.datasets.map((d) => `
    <button class="dsc" data-key="${esc(d.key)}" aria-pressed="${S.selected === d.key}" ${staticMode || !d.rows ? "disabled" : ""}>
      <h3>${esc(d.name)}</h3><p>${esc(d.description)}</p>
      <div class="meta">${(d.rows || 0).toLocaleString()} rows${d.holdout_rows ? " + " + d.holdout_rows.toLocaleString() + " held out" : ""}</div></button>`).join("");
  const upload = S.mode === "live" ? `
    <div class="card"><h3>Bring your own data</h3>
      <p class="sub">Upload a CSV of sensor or maintenance data. Name the failure column, or leave it empty for run-to-failure logs (an entity column plus a counting time column): Sentinel derives the target itself.</p>
      <div class="row"><input type="file" id="file" accept=".csv"><input type="text" id="target" placeholder="target column (optional)" style="min-width:220px">
      <button class="btn ghost" id="upBtn">Upload</button><span id="upMsg" class="sub" style="margin:0"></span></div></div>` : "";
  const controls = staticMode
    ? `<div class="note"><b>You are viewing a recorded run.</b> Every number here was produced by the real Sentinel pipeline on NASA C-MAPSS FD001 and exported. Live training, uploads and container builds run in the Docker stack (<code>docker compose up</code>).</div>
       <div class="row"><button class="btn" id="replay">Replay the pipeline</button><span class="sub" style="margin:0">Watch the six nodes and the decision ledger fire in order.</span></div>`
    : `<div class="row" style="margin:6px 0 4px"><button class="btn" id="run" ${S.status === "running" ? "disabled" : ""}>${S.status === "running" ? "Running..." : "Run pipeline"}</button>
       <label class="sub" style="margin:0">Search budget <select id="budget"><option value="fast" ${S.budget === "fast" ? "selected" : ""}>Fast (about 2 min)</option><option value="full" ${S.budget === "full" ? "selected" : ""}>Full (about 6 min)</option></select></label></div>
       ${S.error ? `<div class="note" style="border-color:var(--red)">${esc(S.error)}</div>` : ""}`;
  v.innerHTML = `<div class="card"><h3>Pick a dataset, press run</h3>
      <p class="sub">Six nodes run in order. Each one applies fixed rules and writes them to the ledger on the right. The same input and seed always produce the same run hash.</p>
      <div class="ds">${cards}</div>${controls}</div>${upload}
    ${S.mode === "live" && S.pastRuns?.length ? `<div class="card"><h3>Earlier runs</h3><p class="sub">Open a finished run with its ledger. Runs with the same hash are reproductions of each other.</p>
      <div class="row">${S.pastRuns.map((r) => `<button class="btn ghost open" data-id="${esc(r.id)}">${esc(r.dataset)} &middot; ${esc(r.run_hash.slice(0, 8))}</button>`).join("")}</div></div>` : ""}
    <div class="card"><h3>What makes this different</h3><div class="grid2">
      <div><b class="mono">No model in the decision path</b><p class="sub">Profiling, task framing, model choice, feature engineering, explanation and drift response are closed-form rules, statistical tests and cross-validated metrics. Machine learning is used where it is the right tool: the failure-prediction models themselves.</p></div>
      <div><b class="mono">Auditable by construction</b><p class="sub">Every decision names the rule, the numbers it read and the threshold it compared against. The whole run reduces to one hash that anyone can reproduce.</p></div></div></div>`;
  v.querySelectorAll(".dsc").forEach((b) => (b.onclick = () => { S.selected = b.dataset.key; render(); }));
  $("#budget", v) && ($("#budget", v).onchange = (e) => (S.budget = e.target.value));
  $("#run", v) && ($("#run", v).onclick = startRun);
  $("#replay", v) && ($("#replay", v).onclick = replay);
  $("#upBtn", v) && ($("#upBtn", v).onclick = doUpload);
  v.querySelectorAll(".open").forEach((b) => (b.onclick = () => openRun(b.dataset.id)));
}

async function openRun(id) {
  resetRun();
  const r = await j(`api/runs/${id}`);
  S.result = r.result; S.runId = id; S.status = "done";
  (await j(`api/runs/${id}/trace`)).forEach((e) => applyEvent(e, true));
  setHash(); S.driftSensor = driftSensors()[0]; S.fleetSel = S.result.fleet?.[0]?.id; S.tab = "board";
  drawStations(); drawTabs(); drawLedger(); render();
}

async function doUpload() {
  const f = $("#file").files[0];
  if (!f) { $("#upMsg").textContent = "Choose a CSV file first."; return; }
  const fd = new FormData(); fd.append("file", f); fd.append("target", $("#target").value || "");
  try {
    const r = await j("api/upload", { method: "POST", body: fd });
    S.datasets = S.datasets.filter((d) => !d.key.startsWith("upload:"));
    S.datasets.push({ key: r.key, name: f.name, description: `Uploaded: ${r.columns.length} columns (${r.columns.slice(0, 5).join(", ")}...)`, rows: 1, uploadTarget: $("#target").value || null });
    S.selected = r.key; render();
  } catch (e) { $("#upMsg").textContent = "Upload failed: " + e.message; }
}

function resetRun() {
  S.events = []; S.stationNote = {}; S.stationState = {}; S.partial = {}; S.result = null; S.error = null;
  S.drift = null; S.retrain = null; S.fleetSel = null; SEEN.clear(); setHash(); drawStations(); drawLedger();
}

async function startRun() {
  resetRun(); S.status = "running"; drawTabs(); render();
  const ds = S.datasets.find((d) => d.key === S.selected) || {};
  try {
    const { run_id } = await j("api/runs", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ dataset: S.selected, budget: S.budget, target: ds.uploadTarget || null }) });
    S.runId = run_id;
    let finished = false;
    const finish = async () => {
      if (finished) return;
      const r = await j(`api/runs/${run_id}`);
      if (r.status === "running") return;
      finished = true; clearInterval(poll); es.close();
      S.status = r.status;
      if (r.status === "error") S.error = "Pipeline error: " + r.error;
      if (r.result) { S.result = r.result; S.partial.scorecard = r.result.scorecard || S.partial.scorecard; }
      if (S.result && !S.result.halted) { setHash(); S.driftSensor = driftSensors()[0]; S.fleetSel = S.result.fleet[0]?.id; S.tab = "board"; }
      else if (S.result?.halted) S.tab = "quality";
      drawStations(); drawTabs(); render();
    };
    const es = new EventSource(`api/runs/${run_id}/events`);
    es.onmessage = (m) => applyEvent(JSON.parse(m.data));
    es.addEventListener("end", finish);
    es.onerror = () => { /* reconnects replay from the start; SEEN de-duplicates */ };
    const poll = setInterval(() => finish().catch(() => {}), 3000);
  } catch (e) { S.status = "error"; S.error = e.message; drawTabs(); render(); }
}

async function replay() {
  const evs = S.events.slice();
  resetRunKeepResult(); S.status = "running"; drawTabs(); render();
  let prev = evs[0]?.t || 0;
  for (const e of evs) {
    const dt = Math.min(Math.max((e.t - prev) * 1000 * 0.12, 0), 500); prev = e.t;
    if (e.status === "progress") { await new Promise((r) => setTimeout(r, 25)); }
    else await new Promise((r) => setTimeout(r, Math.max(dt, 90)));
    applyEventReplay(e);
  }
  S.status = "done"; drawStations(); drawTabs(); render();
}
function resetRunKeepResult() { S.stationNote = {}; S.stationState = {}; S.partial = {}; S.events = []; SEEN.clear(); S.ledgerCount = 0; drawStations(); drawLedger(); }
function applyEventReplay(e) { applyEvent(e); }

// ------------------------------------------------------------------ Quality
function viewQuality(v) {
  const sc = S.partial.scorecard || S.result?.scorecard;
  if (!sc) { v.innerHTML = `<p class="empty">The scorecard appears once the Data Steward finishes.</p>`; return; }
  const dot = sc.passed ? (sc.counts.warn ? "warn" : "pass") : "fail";
  v.innerHTML = `
    <div class="kpis">
      <div class="kpi"><label>Quality score</label><b>${sc.score}</b><small>pass = 1, warn = 0.5, fail = 0</small></div>
      <div class="kpi"><label>Checks</label><b>${sc.counts.pass}<span style="color:var(--steel)"> / </span>${sc.counts.warn}<span style="color:var(--steel)"> / </span>${sc.counts.fail}</b><small>pass / warn / fail</small></div>
      <div class="kpi"><label>Gate</label><b style="font-size:17px;line-height:1.5"><span class="pill ${dot}">${esc(sc.verdict)}</span></b><small>any fail halts the run</small></div>
    </div>
    ${sc.passed ? "" : `<div class="note" style="border-color:var(--red)"><b>Run halted before any model was trained.</b> A failed check means the data cannot support a trustworthy model. Fix the source data and run again. Nothing was guessed or imputed on your behalf.</div>`}
    <div class="card"><h3>Data quality scorecard</h3><p class="sub">Fixed thresholds, no judgement calls. Warnings trigger an automatic action (a dropped column, a changed metric) that is logged in the ledger.</p>
    <table><thead><tr><th>Check</th><th>Status</th><th>Value</th><th>Threshold</th><th>What was found</th></tr></thead><tbody>
    ${sc.checks.map((c) => `<tr><td><b>${esc(c.label)}</b></td><td><span class="pill ${c.status}">${c.status}</span></td>
      <td class="num">${esc(c.value)}</td><td class="mono" style="font-size:12px;color:var(--steel)">${esc(c.threshold)}</td><td>${esc(c.detail)}</td></tr>`).join("")}
    </tbody></table></div>`;
}

// ------------------------------------------------------------------ Leaderboard
function foldStrip(folds, lo, hi) {
  const w = 96, h = 14;
  const dots = folds.map((f, i) => `<circle cx="${4 + (i * (w - 8)) / (folds.length - 1 || 1)}" cy="${h / 2}" r="3" fill="#101418"/>`).join("");
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-label="per-fold scores"><line x1="0" y1="${h / 2}" x2="${w}" y2="${h / 2}" stroke="#cdd3d8"/>${dots}</svg>`;
}
function viewBoard(v) {
  const rows = S.result?.leaderboard || S.partial.leaderboard;
  if (!rows) { v.innerHTML = `<p class="empty">The leaderboard appears when the Trainer finishes.</p>`; return; }
  const metric = rows[0].metric, hi = metric !== "rmse";
  const vals = rows.map((r) => r.cv_mean), lo = Math.min(...vals), mx = Math.max(...vals);
  const mName = { roc_auc: "ROC-AUC", average_precision: "average precision", rmse: "RMSE" }[metric];
  const cls = S.result?.framing?.task === "regression" ? false : true;
  const champ = rows[0];
  const finKey = metric === "rmse" ? "rmse" : metric;
  v.innerHTML = `
    <div class="kpis">
      <div class="kpi"><label>Champion</label><b style="font-size:20px">${esc(champ.name)}</b><small>rank 1 by cross-validation</small></div>
      <div class="kpi"><label>CV ${esc(mName)}</label><b>${fmt(champ.cv_mean, 4)}</b><small>&plusmn; ${fmt(champ.cv_std, 4)} over 5 grouped folds</small></div>
      <div class="kpi"><label>Holdout ${esc(mName)}</label><b>${fmt(champ.holdout?.all_rows?.[finKey], 4)}</b><small>unseen ${esc(S.result?.entity_noun?.toLowerCase() || "unit")}s, all rows</small></div>
      ${cls ? `<div class="kpi"><label>Holdout F1</label><b>${fmt(champ.holdout?.all_rows?.f1, 3)}</b><small>at the out-of-fold alert threshold</small></div>` : ""}
    </div>
    <div class="card"><h3>Cross-validated leaderboard</h3>
      <p class="sub">Ranked by cross-validated ${esc(mName)} (${hi ? "higher" : "lower"} is better). Folds are grouped so no ${esc((S.result?.entity_noun || "unit").toLowerCase())} appears in both training and validation. The holdout is scored for every model but never used to choose between them: the numbers pick the winner.</p>
      <table><thead><tr><th>#</th><th>Model</th><th>CV ${esc(mName)}</th><th>Folds</th><th>Holdout (all rows)</th><th>Holdout (last obs.)</th><th>Trials</th><th>Time</th></tr></thead><tbody>
      ${rows.map((r) => {
        const w = 15 + 85 * (mx === lo ? 1 : hi ? (r.cv_mean - lo) / (mx - lo) : (mx - r.cv_mean) / (mx - lo));
        const fo = r.holdout?.final_obs || {};
        return `<tr><td class="num">${r.rank}</td><td><b>${esc(r.name)}</b> ${r.rank === 1 ? '<span class="pill pass">champion</span>' : ""}
          <details><summary class="sub" style="cursor:pointer;margin:2px 0 0">parameters</summary><div class="code" style="display:block;margin-top:4px">${esc(JSON.stringify(r.params))}</div></details></td>
          <td style="min-width:170px"><div class="num">${fmt(r.cv_mean, 4)} <span style="color:var(--steel)">&plusmn;${fmt(r.cv_std, 4)}</span></div><div class="bar"><i style="width:${w}%;${r.rank === 1 ? "" : "background:#8a949c"}"></i></div></td>
          <td>${foldStrip(r.cv_folds)}</td>
          <td class="num">${fmt(r.holdout?.all_rows?.[finKey], 4)}</td>
          <td class="num">${fmt(fo[finKey], 4)} <span style="color:var(--steel)">n=${fo.n ?? "-"}</span></td>
          <td class="num">${r.n_trials}</td><td class="num">${r.fit_seconds}s</td></tr>`;
      }).join("")}</tbody></table>
      <p class="sub" style="margin-top:10px">Bars span the observed range of the four models, so small gaps look large. Read the numbers: on this data the gradient-boosted models are separated from the linear baseline by only a few thousandths, which is itself a finding.</p></div>
    ${S.result?.explain?.parity?.checked ? `<div class="note">Native Tree-SHAP used by the service equals <code>shap.TreeExplainer</code> to within <b>${S.result.explain.parity.max_abs_diff.toExponential(1)}</b> on ${S.result.explain.parity.rows} rows.</div>` : ""}`;
}

// ------------------------------------------------------------------ Explain
function hbars(items, key, labelKey, max, color = "#101418") {
  return items.map((i) => `<div style="display:grid;grid-template-columns:minmax(150px,1.1fr) 1.6fr 48px;gap:10px;align-items:center;margin:6px 0">
    <span style="font-size:12.5px">${esc(i[labelKey])}</span><div class="bar"><i style="width:${(100 * i[key]) / max}%;background:${color}"></i></div><span class="num" style="font-size:12px">${pc(i[key], 1)}</span></div>`).join("");
}
function viewExplain(v) {
  const g = S.result?.explain?.global || S.partial.global;
  if (!g) { v.innerHTML = `<p class="empty">Explanations appear when the Explainer finishes.</p>`; return; }
  const rel = S.result?.explain?.reliability || [];
  const sMax = g.sensors[0].share, fMax = g.features[0].share;
  const relBars = rel.length ? `<div class="card"><h3>How much to trust a score</h3>
    <p class="sub">For each score band, the share of out-of-fold examples that really did fail within the alarm horizon. This is the "confidence" attached to every alert: a measured hit rate, not a self-reported probability.</p>
    ${relChart(rel)}</div>` : "";
  v.innerHTML = `<div class="grid2">
    <div class="card"><h3>Which sensors drive risk</h3><p class="sub">Mean absolute Tree-SHAP contribution, summed over each sensor's engineered features, as a share of the total.</p>${hbars(g.sensors.slice(0, 9), "share", "label", sMax)}</div>
    <div class="card"><h3>Top engineered features</h3><p class="sub">The single features the champion leans on most across the training data.</p>${hbars(g.features.slice(0, 9), "share", "label", fMax, "#5b6670")}</div></div>
    ${relBars}
    <div class="note">Explanations are computed, not written. A sentence such as <i>"Ps30 static pressure well above baseline (+2.4 sd), 19% of the signal"</i> is a template filled with a SHAP value and a z-score. Open the Fleet tab to read one per engine.</div>`;
}
function relChart(rel) {
  const W = 560, H = 190, pad = 34, bw = (W - pad) / rel.length;
  const bars = rel.map((b, i) => {
    const x = pad + i * bw, hgt = b.positive_rate == null ? 0 : (H - 34) * b.positive_rate;
    const mid = (H - 34) * ((b.lo + b.hi) / 2);
    return `<rect x="${x + 4}" y="${H - 20 - hgt}" width="${bw - 8}" height="${hgt}" fill="#101418"/>
      <line x1="${x + 2}" x2="${x + bw - 2}" y1="${H - 20 - mid}" y2="${H - 20 - mid}" stroke="#d8001b" stroke-width="2"/>
      <text x="${x + bw / 2}" y="${H - 6}" text-anchor="middle">${(b.lo * 100).toFixed(0)}</text>
      <text x="${x + bw / 2}" y="${H - 24 - hgt}" text-anchor="middle">${b.n ? pc(b.positive_rate) : ""}</text>`;
  }).join("");
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:640px" role="img" aria-label="Observed failure rate by score band">
    <line x1="${pad}" x2="${W}" y1="${H - 20}" y2="${H - 20}" stroke="#101418"/>${bars}
    <text x="0" y="${H - 20}">0%</text><text x="0" y="14">100%</text></svg>
    <div class="legend">Dark bars: observed hit rate per score band (score in %). Red ticks: where a perfectly calibrated score would sit.</div>`;
}

// ------------------------------------------------------------------ Fleet
function riskCurve(f, thr) {
  const t = f.curve.t, r = f.curve.risk, W = 620, H = 190, L = 34, B = 22;
  const x = (i) => L + ((W - L - 6) * (t[i] - t[0])) / Math.max(1, t[t.length - 1] - t[0]);
  const y = (p) => 8 + (H - B - 8) * (1 - p);
  const d = r.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(p).toFixed(1)}`).join(" ");
  const area = `${d} L${x(r.length - 1).toFixed(1)} ${y(0)} L${x(0).toFixed(1)} ${y(0)} Z`;
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Failure risk over time">
    <path d="${area}" fill="rgba(216,0,27,.10)"/><path d="${d}" fill="none" stroke="#d8001b" stroke-width="2"/>
    ${thr != null ? `<line x1="${L}" x2="${W - 6}" y1="${y(thr)}" y2="${y(thr)}" stroke="#101418" stroke-dasharray="4 3"/><text x="${W - 8}" y="${y(thr) - 4}" text-anchor="end">alert threshold ${pc(thr)}</text>` : ""}
    <line x1="${L}" x2="${W - 6}" y1="${y(0)}" y2="${y(0)}" stroke="#101418"/><text x="2" y="${y(1) + 8}">100%</text><text x="8" y="${y(0)}">0</text>
    <text x="${L}" y="${H - 6}">${t[0]}</text><text x="${W - 8}" y="${H - 6}" text-anchor="end">${S.result.time_noun} ${t[t.length - 1]}</text></svg>`;
}
function spark(vals) {
  const W = 190, H = 38, mn = Math.min(...vals), mx = Math.max(...vals), sp = mx - mn || 1;
  const pts = vals.map((v, i) => `${((W * i) / (vals.length - 1 || 1)).toFixed(1)},${(H - 4 - ((H - 8) * (v - mn)) / sp).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}"><polyline points="${pts}" fill="none" stroke="#101418" stroke-width="1.4"/></svg>`;
}
function waterfall(drivers) {
  const mx = Math.max(...drivers.map((d) => Math.abs(d.contribution))) || 1;
  return `<div class="wf">${drivers.map((d) => {
    const w = (Math.abs(d.contribution) / mx) * 50;
    return `<span>${esc(d.label)}</span><div class="axis"><i class="${d.contribution > 0 ? "up" : "dn"}" style="width:${w}%"></i></div><span class="num">${pc(d.share)}</span>`;
  }).join("")}</div><div class="legend" style="margin-top:6px"><span style="color:var(--red)">&#9632;</span> pushes toward failure &nbsp; <span style="color:var(--teal)">&#9632;</span> pushes away &nbsp; percent = share of the explained signal</div>`;
}
function viewFleet(v) {
  const fl = S.result.fleet, thr = S.result.champion.threshold, noun = S.result.entity_noun;
  const sel = fl.find((f) => f.id === S.fleetSel) || fl[0];
  const nAlert = fl.filter((f) => f.level === "ALERT").length, nWatch = fl.filter((f) => f.level === "WATCH").length;
  const reg = S.result.framing.task === "regression";
  v.innerHTML = `
    <div class="kpis"><div class="kpi"><label>${esc(noun)}s scored</label><b>${fl.length}</b><small>holdout, never used to train</small></div>
      <div class="kpi"><label>Alerts</label><b style="color:var(--red)">${nAlert}</b><small>score at or above ${thr != null ? pc(thr) : "-"}</small></div>
      <div class="kpi"><label>Watch</label><b style="color:var(--amber)">${nWatch}</b><small>at least half the threshold</small></div></div>
    <div class="split"><div class="card" style="padding:0;max-height:720px;overflow:auto" role="region" aria-label="Fleet machine list"><table class="mid"><thead><tr><th>${esc(noun)}</th><th>Risk</th><th>Status</th><th>Actual RUL</th></tr></thead><tbody>
      ${fl.map((f) => `<tr class="clickable ${f.id === sel.id ? "sel" : ""}" data-id="${esc(f.id)}" tabindex="0" role="button" aria-selected="${f.id === sel.id}" aria-label="${esc(noun)} ${esc(f.id)} risk ${pc(f.risk)} status ${f.level}"><td class="num">${esc(f.id)}</td>
        <td style="min-width:110px"><div class="bar" style="height:8px"><i style="width:${Math.min(100, f.risk * 100)}%;background:${f.level === "ALERT" ? "var(--red)" : f.level === "WATCH" ? "var(--amber)" : "var(--teal)"}"></i></div></td>
        <td><span class="pill ${f.level}">${f.level}</span></td><td class="num">${f.actual_rul != null ? f.actual_rul : f.actual ? "failed" : "ok"}</td></tr>`).join("")}</tbody></table></div>
    <div class="card"><h3>${esc(noun)} ${esc(sel.id)} <span class="pill ${sel.level}">${sel.level}</span></h3>
      <div class="sentence ${sel.level === "OK" ? "ok" : ""}">${esc(sel.sentence)}</div>
      ${waterfall(sel.drivers)}
      <div class="card" style="margin-top:16px;background:var(--paper);border:1px solid var(--ink);padding:14px">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
          <div>
            <h3 style="font-size:14px;margin:0;letter-spacing:.02em">WHAT-IF SKETCH</h3>
            <p class="sub" style="margin:2px 0 0;font-size:12px">Rough sketch: drag a top driver up or down and see how the risk would move. A linear estimate from this reading&rsquo;s Tree-SHAP contributions; it does not re-run the model.</p>
          </div>
          <button class="btn ghost" id="btnResetSim" style="font-size:11px;padding:3px 8px">Reset to actual</button>
        </div>
        <div style="margin-top:12px;display:grid;gap:8px">
          ${(sel.drivers || []).slice(0, 3).map((d, i) => `
            <div style="display:grid;grid-template-columns:minmax(140px, 1.4fr) 1.5fr 84px;gap:8px;align-items:center;font-size:12px">
              <span title="${esc(d.label)}">${esc(d.label.split(',')[0])}</span>
              <input type="range" class="sim-slider" data-idx="${i}" min="-2.5" max="2.5" step="0.1" value="${Math.min(2.5, Math.max(-2.5, d.z || 0)).toFixed(1)}" aria-label="${esc(d.label)} deviation">
              <span class="num sim-val" id="simVal_${i}">${(d.z >= 0 ? "+" : "") + Number(d.z || 0).toFixed(1)} sd</span>
            </div>
          `).join("")}
        </div>
        <div style="margin-top:12px;padding-top:10px;border-top:1px dashed var(--rule);display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:12px;color:var(--steel)">Simulated failure risk: <b class="num" id="simRisk" style="font-size:15px;color:var(--ink)">${riskTxt(sel.risk)}</b></span>
          <span class="pill ${sel.level}" id="simPill">${sel.level}</span>
        </div>
      </div>
      ${sel.curve ? `<h3 style="margin-top:18px;font-size:15px">Risk over the ${esc(S.result.time_noun)}s observed</h3>${riskCurve(sel, thr)}` : ""}
      ${sel.sensors && Object.keys(sel.sensors).length ? `<h3 style="margin-top:14px;font-size:15px">Top sensors</h3><div class="spark-grid">${Object.entries(sel.sensors).map(([k, vals]) => `<div class="spark"><b>${esc(SENSOR_LABEL(k))}</b>${spark(vals)}</div>`).join("")}</div>` : ""}
      <p class="sub" style="margin-top:12px">Ground truth (revealed after the fact): ${sel.actual_rul != null ? `this ${esc(noun.toLowerCase())} failed <b>${sel.actual_rul}</b> ${esc(S.result.time_noun)}s after the last observation.` : sel.actual ? "this machine did fail." : "no failure occurred."}</p></div></div>`;
  
  v.querySelectorAll("tr.clickable").forEach((tr) => {
    const pick = () => { S.fleetSel = isNaN(+tr.dataset.id) ? tr.dataset.id : +tr.dataset.id; render(); };
    tr.onclick = pick;
    tr.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } };
  });

  const sliders = v.querySelectorAll(".sim-slider");
  const simValEls = [0, 1, 2].map((i) => v.querySelector(`#simVal_${i}`));
  const simRiskEl = v.querySelector("#simRisk");
  const simPillEl = v.querySelector("#simPill");
  const btnReset = v.querySelector("#btnResetSim");

  if (sliders.length && simRiskEl) {
    const origRisk = sel.risk;
    const clamped = Math.max(1e-5, Math.min(1 - 1e-5, origRisk));
    const baseMargin = Math.log(clamped / (1 - clamped));
    const drivers = (sel.drivers || []).slice(0, 3);
    
    function updateSim() {
      let deltaMargin = 0;
      sliders.forEach((sl, i) => {
        const d = drivers[i];
        if (!d) return;
        const curZ = parseFloat(sl.value);
        if (simValEls[i]) simValEls[i].textContent = (curZ >= 0 ? "+" : "") + curZ.toFixed(1) + " sd";
        const origZ = Math.min(2.5, Math.max(-2.5, d.z || 0));
        const sensitivity = Math.abs(origZ) >= 0.3 ? d.contribution / origZ : 0;  // too near baseline to estimate a slope
        deltaMargin += (curZ - origZ) * sensitivity;
      });
      const newMargin = baseMargin + deltaMargin;
      const newRisk = 1 / (1 + Math.exp(-newMargin));
      simRiskEl.textContent = riskTxt(newRisk);
      const lvl = thr != null ? (newRisk >= thr ? "ALERT" : (newRisk >= thr * 0.5 ? "WATCH" : "OK")) : (newRisk >= 0.5 ? "ALERT" : "OK");
      simPillEl.className = "pill " + lvl;
      simPillEl.textContent = lvl;
      if (lvl === "ALERT") simRiskEl.style.color = "var(--red)";
      else if (lvl === "WATCH") simRiskEl.style.color = "var(--amber)";
      else simRiskEl.style.color = "var(--teal)";
    }

    sliders.forEach((sl) => (sl.oninput = updateSim));
    if (btnReset) {
      btnReset.onclick = () => {
        sliders.forEach((sl, i) => {
          if (drivers[i]) sl.value = Math.min(2.5, Math.max(-2.5, drivers[i].z || 0)).toFixed(1);
        });
        updateSim();
      };
    }
  }
}
function SENSOR_LABEL(k) { const s = (S.result?.explain?.global?.sensors || []).find((x) => x.sensor === k); return s ? s.label : k; }

// ------------------------------------------------------------------ Deploy
function viewDeploy(v) {
  const r = S.result, reg = r.registry, par = r.serving.parity;
  const dep = S.deployResult || S.staticDeploy;
  const folder = dep?.folder || r.serving.folder;
  const files = dep?.files || ["Dockerfile", "README.md", "feature_spec.json", "metadata.json", "model.*", "reference.json", "requirements.txt", "sentinel/features.py", "sentinel/explainer.py", "serve.py"];
  const dk = dep?.docker;
  v.innerHTML = `
    <div class="kpis"><div class="kpi"><label>Registered model</label><b style="font-size:16px">${esc(reg.model_name)}</b><small>version ${reg.version}, alias champion</small></div>
      <div class="kpi"><label>MLflow run</label><b class="mono" style="font-size:16px">${esc(reg.mlflow_run_id.slice(0, 10))}</b><small>params, metrics, artifacts</small></div>
      <div class="kpi"><label>Serving parity</label><b style="font-size:16px"><span class="pill ${par.ok ? "pass" : "fail"}">${par.ok ? "identical" : "mismatch"}</span></b><small>max |diff| ${sci(par.max_abs_diff)} over ${par.entities} entities</small></div></div>
    <div class="grid2"><div class="card"><h3>Generated model service</h3>
      <p class="sub">One versioned folder per model. It carries the exact <code>features.py</code> and Tree-SHAP code used in training, so training and serving cannot disagree.</p>
      <pre class="code">${esc(files.map((f) => "  " + f).join("\n"))}</pre></div>
    <div class="card"><h3>Run it anywhere</h3>
      <pre class="code">docker build -t sentinel-model .
docker run -p 8080:8080 sentinel-model
curl -X POST localhost:8080/predict \\
  -H 'content-type: application/json' \\
  -d '{"rows": [ ...raw sensor rows, oldest first... ]}'</pre>
      <p class="sub">Returns risk, alert level, a measured confidence and the ranked drivers for every ${esc(r.entity_noun.toLowerCase())}.</p></div></div>
    <div class="card"><h3>Container smoke test</h3>${S.mode === "live"
      ? `<p class="sub">Builds the image, starts the container, calls <code>/health</code> and <code>/predict</code>, and checks the container's scores against the trained model.</p>
        <div class="row"><button class="btn" id="dk" ${S.busy ? "disabled" : ""}>${S.busy ? "Building image..." : "Build and test container"}</button></div>`
      : `<p class="sub">Recorded result from the Docker stack:</p>`}
      ${dk ? (dk.ran ? `<div class="note"><span class="pill ${dk.ok ? "pass" : "fail"}">${dk.ok ? "container matches model" : "mismatch"}</span> &nbsp; image <code>${esc(dk.image)}</code>, ${dk.build_seconds < 5 ? "layers cached" : "built in " + dk.build_seconds + " s"}, max |diff| ${sci(dk.max_abs_diff)}</div>`
        : `<div class="note">Docker is not reachable from this server (${esc(dk.reason)}). The in-process parity test above still passed.</div>`) : ""}
    </div>`;
  $("#dk", v) && ($("#dk", v).onclick = async () => {
    S.busy = true; render();
    try { S.deployResult = await j(`api/runs/${S.runId}/deploy`, { method: "POST" }); } catch (e) { S.deployResult = { docker: { ran: false, reason: e.message }, folder, files }; }
    S.busy = false; render();
  });
}

// ------------------------------------------------------------------ Monitor
function driftSensors() {
  const s = (S.result?.explain?.global?.sensors || []).map((x) => x.sensor).filter((x) => x !== S.result?.framing?.time_col);
  return s.slice(0, 6);
}
function sigmaList() { return S.mode === "static" ? (S.staticDrift?.sigmas || [0]) : [0, 0.25, 0.5, 1, 1.5, 2, 3]; }
async function getDrift() {
  const sig = sigmaList()[S.sigmaIdx], sen = S.driftSensor;
  if (S.mode === "static") { S.drift = S.staticDrift?.grid?.[sen]?.[String(sig)] || null; return; }
  S.drift = await j(`api/runs/${S.runId}/drift`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ sensor: sen, sigma: sig }) });
}
function psiRows(rows) {
  const lab = Object.fromEntries((S.result.explain.global.features || []).map((f) => [f.feature, f.label]));
  return rows.map((r) => {
    const w = Math.min(100, (r.psi / 1.0) * 100);
    const col = r.psi >= 0.25 ? "var(--red)" : r.psi >= 0.1 ? "var(--amber)" : "var(--teal)";
    return `<div style="display:grid;grid-template-columns:minmax(170px,1.2fr) 1.6fr 60px;gap:10px;align-items:center;margin:6px 0">
      <span style="font-size:12.5px">${esc(lab[r.feature] || r.feature)}</span>
      <div class="bar" style="height:12px"><i style="width:${w}%;background:${col}"></i>
        <span style="position:absolute;left:10%;top:-3px;bottom:-3px;border-left:1px dashed var(--steel)"></span><span style="position:absolute;left:25%;top:-3px;bottom:-3px;border-left:1px solid var(--ink)"></span></div>
      <span class="num" style="font-size:12px">${r.psi >= 10 ? ">10" : r.psi.toFixed(2)}</span></div>`;
  }).join("");
}
async function viewMonitor(v) {
  const r = S.result, sensors = driftSensors(), sigmas = sigmaList();
  if (!S.drift) v.innerHTML = `<p class="empty">Computing drift against the training distributions...</p>`;
  if (!S.drift) { try { await getDrift(); } catch (e) { S.error = e.message; } }
  if (S.tab !== "monitor") return;
  const d = S.drift, sig = sigmas[S.sigmaIdx], metric = r.framing.metric;
  const mn = { roc_auc: "ROC-AUC", average_precision: "avg. precision", rmse: "RMSE" }[metric];
  v.innerHTML = `
    <div class="card"><h3>Simulate sensor drift</h3><p class="sub">Shift one sensor by a number of training standard deviations, as a miscalibration or a replaced probe would. The monitor compares live feature distributions with training (Population Stability Index). The retrain rule is fixed: any top-10 feature at PSI 0.25 or above.</p>
      <div class="row"><label>Sensor <select id="sen">${sensors.map((s) => `<option value="${esc(s)}" ${s === S.driftSensor ? "selected" : ""}>${esc(SENSOR_LABEL(s))}</option>`).join("")}</select></label>
      <label>Shift <input type="range" id="sig" min="0" max="${sigmas.length - 1}" step="1" value="${S.sigmaIdx}"> <b class="num" id="sigv">${sig} sd</b></label></div>
      ${S.mode === "static" ? `<p class="sub" style="margin:8px 0 0">Recorded scenarios: the slider steps through shifts computed by the real monitor.</p>` : ""}</div>
    ${d ? `<div class="kpis"><div class="kpi"><label>Monitor status</label><b style="font-size:17px;line-height:1.6"><span class="pill ${d.report.status}">${d.report.status}</span></b><small>${esc(d.report.reason)}</small></div>
      <div class="kpi"><label>Champion ${esc(mn)}</label><b>${fmt(d.champion_before[metric], 4)} &rarr; ${fmt(d.champion_after[metric], 4)}</b><small>before / after the shift</small></div>
      ${d.champion_before.f1 != null ? `<div class="kpi"><label>Champion F1</label><b>${fmt(d.champion_before.f1, 3)} &rarr; ${fmt(d.champion_after.f1, 3)}</b><small>at the alert threshold</small></div>` : ""}</div>
      <div class="card"><h3>PSI on the top-10 features</h3><p class="sub">Bars clamp at 1.0. Dashed line: watch (0.10). Solid line: retrain (0.25). Measured on the healthy early-life window so degradation itself does not register as drift.</p>${psiRows(d.report.top10)}</div>
      <div class="card"><h3>Champion / challenger</h3>
        <p class="sub">When the rule trips, Sentinel refits the champion's model family on the training data plus half the drifted ${esc(r.entity_noun.toLowerCase())}s, and scores both models on the other half, which the challenger never saw. It is promoted only if it wins.</p>
        <div class="row"><button class="btn" id="rt" ${d.report.status !== "RETRAIN" || S.busy ? "disabled" : ""}>${S.busy ? "Retraining..." : "Retrain and compare"}</button></div>
        ${S.retrain ? retrainHtml(S.retrain, mn, metric) : ""}</div>` : `<p class="empty">No recorded result for this scenario.</p>`}`;
  $("#sen", v).onchange = async (e) => { S.driftSensor = e.target.value; S.drift = null; S.retrain = null; render(); };
  $("#sig", v).oninput = (e) => { S.sigmaIdx = +e.target.value; $("#sigv", v).textContent = sigmas[S.sigmaIdx] + " sd"; };
  $("#sig", v).onchange = async () => { S.drift = null; S.retrain = null; render(); };
  $("#rt", v) && ($("#rt", v).onclick = async () => {
    S.busy = true; render();
    try {
      if (S.mode === "static") S.retrain = S.staticDrift?.retrain?.[S.driftSensor]?.[String(sig)] || { triggered: false, reason: "no recorded retrain for this scenario" };
      else S.retrain = await j(`api/runs/${S.runId}/retrain`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ sensor: S.driftSensor, sigma: sig }) });
    } catch (e) { S.retrain = { triggered: false, reason: e.message }; }
    S.busy = false; render();
  });
}
function retrainHtml(t, mn, metric) {
  if (!t.triggered) return `<div class="note">${esc(t.reason || "not triggered")}</div>`;
  const fam = t.metric === "rmse";
  return `<table style="margin-top:12px"><thead><tr><th>Model</th><th>${esc(mn)}</th><th>F1</th><th>Evaluated on</th></tr></thead><tbody>
    <tr><td><b>Incumbent</b> (deployed)</td><td class="num">${fmt(t.incumbent[metric], 4)}</td><td class="num">${fmt(t.incumbent.f1, 3)}</td><td rowspan="2" class="sub">${t.eval_rows.toLocaleString()} rows from units the challenger never saw</td></tr>
    <tr><td><b>Challenger</b> (retrained on ${t.adaptation_rows.toLocaleString()} drifted rows)</td><td class="num">${fmt(t.challenger[metric], 4)}</td><td class="num">${fmt(t.challenger.f1, 3)}</td></tr></tbody></table>
    <div class="note"><span class="pill ${t.promoted ? "pass" : "warn"}">${t.promoted ? "promoted" : "kept incumbent"}</span> &nbsp; ${t.promoted
      ? (t.registry ? `Challenger registered as <b>${esc(t.registry.model_name)} v${t.registry.version}</b> and packaged as a new service folder.` : "Challenger would be registered as the next model version and packaged as a new service folder (the live app does this).")
      : "The challenger did not beat the incumbent on unseen units, so nothing changed."} Rule: ${esc(t.rule)}.</div>`;
}

boot();
