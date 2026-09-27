/* Panchayat agromet dashboard. Every number comes from the API (no hard-coded metrics). */
"use strict";

const S = {
  meta: null, issues: [], issue: null, data: null, lead: 0, v: null, lang: null, view: "split",
  gp: null, gpLayer: null, blkLayer: null, geoGp: null, geoBlk: null, byCode: {}, sel: null,
};
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};
async function api(path, opts = {}) {
  const tok = (() => { try { return sessionStorage.getItem("adminToken") || ""; } catch { return ""; } })();
  const headers = Object.assign({}, opts.headers || {}, tok ? { "X-Admin-Token": tok } : {});
  const r = await fetch(path, Object.assign({}, opts, { headers }));
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* not json */ }
    throw new Error(`${r.status}: ${msg}`);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
}
function banner(msg) { const b = $("banner"); b.textContent = msg; b.hidden = !msg; }
const pct = (x, d = 1) => (x == null || !isFinite(x) ? "–" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(d)}%`);
const num = (x, d = 1) => (x == null || !isFinite(x) ? "–" : Number(x).toFixed(d));

/* ---------------------------------------------------------------- colour scales
   Layers, bins, ramps, colours and the basemap all come from config/dashboard.yaml and
   config/advisory_rules.yaml via /api/meta - nothing is defined here. */
const D = () => S.meta.dashboard;
const LAYER = (v) => D().layers[v];
const SEV = (s) => S.meta.severity_colors[s];

function binColor(v, bins, ramp) {
  if (v == null || !isFinite(v)) return D().colors.no_data;
  let i = 0; while (i < bins.length && v >= bins[i]) i++;
  return ramp[Math.min(i, ramp.length - 1)];
}
function layerColor(v, x) { const L_ = LAYER(v); return binColor(x, L_.bins, D().ramps[L_.ramp]); }

function gpValue(rec, v, k) {
  if (!rec) return null;
  const L_ = LAYER(v), a = (key) => (rec[key] ? rec[key][k] : null);
  if (L_.derived) {
    const [x, op, y] = L_.derived, xa = a(x), ya = a(y);
    if (xa == null || ya == null) return null;
    return op === "-" ? xa - ya : op === "+" ? xa + ya : null;
  }
  const x = a(v);
  return x == null ? null : x * (L_.scale ?? 1);
}
function blockValue(lgd, field, k) {
  const b = S.data?.blocks?.[String(lgd)];
  return b && b[field] ? b[field][k] : null;
}
/* the block map shows the layer's own block field, or rain when the layer has none (e.g. probabilities) */
const blockLayer = (v) => (LAYER(v).block_field ? v : Object.keys(D().layers).find((k) => D().layers[k].block_field === "r"));

/* ---------------------------------------------------------------- maps */
let mapG, mapB, syncing = false;
function makeMap(id) {
  const bm = D().basemap;
  const m = L.map(id, { zoomControl: true, attributionControl: true, preferCanvas: true, minZoom: bm.min_zoom });
  L.tileLayer(bm.url, { maxZoom: bm.max_zoom, opacity: bm.opacity, attribution: bm.attribution }).addTo(m);
  return m;
}
function syncMaps(a, b) {
  a.on("move", () => { if (syncing) return; syncing = true; b.setView(a.getCenter(), a.getZoom(), { animate: false }); syncing = false; });
}

function urbanFill(blockLgd) {
  const L_ = LAYER(S.v);
  if (!L_.block_field) return D().colors.no_gram_panchayat;
  return layerColor(S.v, blockValue(blockLgd, L_.block_field, S.lead));
}
function styleUrban(f) {
  return { fillColor: urbanFill(f.properties.block_lgd), fillOpacity: 0.7, color: "#fff", weight: 0.8, dashArray: "3 3" };
}
function urbanTipHtml(p) {
  const k = S.lead;
  return `<b>Area without a Gram Panchayat</b> <span class="muted">${esc(p.block_name)} block</span><br>
    City / town / cantonment, forest or reservoir · ${num(p.area_km2)} km²<br>
    Block forecast: rain <b>${num(blockValue(p.block_lgd, "r", k))}</b> mm · Tmax ${num(blockValue(p.block_lgd, "tx", k))} °C ·
    Tmin ${num(blockValue(p.block_lgd, "tn", k))} °C · RH ${num(blockValue(p.block_lgd, "rh", k), 0)}%<br>
    <span class="muted">Not downscaled: panchayat-level forecasts apply to Gram Panchayats only.</span>`;
}
function styleGp(f) {
  if (!f.properties.modelled) return styleUrban(f);
  const rec = S.data?.gp?.[f.properties.gp_code];
  const fill = LAYER(S.v).categorical ? (rec ? SEV(rec.sev).hex : D().colors.no_data) : layerColor(S.v, gpValue(rec, S.v, S.lead));
  const sel = S.sel === f.properties.gp_code;
  return { fillColor: fill, fillOpacity: 0.88, color: sel ? "#000" : "rgba(255,255,255,.7)", weight: sel ? 2.5 : 0.4 };
}
function styleBlk(f) {
  const bl = blockLayer(S.v);
  return { fillColor: layerColor(bl, blockValue(f.properties.block_lgd, LAYER(bl).block_field, S.lead)),
           fillOpacity: 0.88, color: "#fff", weight: 1.2 };
}

function tip(e, html) {
  const t = $("tip"); t.innerHTML = html; t.hidden = false;
  const x = e.originalEvent.clientX, y = e.originalEvent.clientY;
  t.style.left = `${Math.min(x + 14, window.innerWidth - 260)}px`; t.style.top = `${y + 12}px`;
}
function hideTip() { $("tip").hidden = true; }

function gpTipHtml(p) {
  const rec = S.data?.gp?.[p.gp_code];
  if (!rec) return urbanTipHtml(p);
  const k = S.lead;
  const date = S.data.valid_dates[k];
  return `<b>${esc(p.gp_name)}</b> <span class="muted">${esc(p.block_name)}</span><br>${date}<br>
    Rain <b>${num(rec.r?.[k])}</b> mm (block ${num(rec.fr?.[k])}) · P10–P90 ${num(rec.r10?.[k])}–${num(rec.r90?.[k])}<br>
    Tmax ${num(rec.tx?.[k])} °C · Tmin ${num(rec.tn?.[k])} °C · RH ${num(rec.rh?.[k], 0)}%<br>
    <span class="sev ${rec.sev}">${SEV(rec.sev).icon} ${esc(sevLabel(rec.sev))}</span>`;
}
const sevLabel = (s) => SEV(s)[`label_${S.lang}`];

function drawLayers() {
  if (S.gpLayer) S.gpLayer.remove();
  if (S.blkLayer) S.blkLayer.remove();
  if (S.urbanLayer) S.urbanLayer.remove();
  if (S.geoUrban) {
    S.urbanLayer = L.geoJSON(S.geoUrban, {
      style: styleUrban,
      onEachFeature: (f, l) => { l.on("mousemove", (e) => tip(e, urbanTipHtml(f.properties))); l.on("mouseout", hideTip); },
    }).addTo(mapG);
  }
  S.gpLayer = L.geoJSON(S.geoGp, {
    style: styleGp,
    onEachFeature: (f, l) => {
      S.byCode[f.properties.gp_code] = l;
      l.on("mousemove", (e) => tip(e, gpTipHtml(f.properties)));
      l.on("mouseout", hideTip);
      l.on("click", () => selectGp(f.properties.gp_code));
    },
  }).addTo(mapG);
  S.blkLayer = L.geoJSON(S.geoBlk, {
    style: styleBlk,
    onEachFeature: (f, l) => {
      l.on("mousemove", (e) => {
        const bl = blockLayer(S.v);
        tip(e, `<b>${esc(f.properties.block_name)}</b> block<br>${esc(LAYER(bl).legend)}: <b>${num(blockValue(f.properties.block_lgd, LAYER(bl).block_field, S.lead))}</b><br>
          <span class="muted">One value for all ${f.properties.n_gps} panchayats</span>`);
      });
      l.on("mouseout", hideTip);
    },
  }).addTo(mapB);
  L.geoJSON(S.geoBlk, { style: { fill: false, color: "#333", weight: 1.1 }, interactive: false }).addTo(mapG);
}
function restyle() {
  S.urbanLayer?.setStyle(styleUrban); S.gpLayer?.setStyle(styleGp); S.blkLayer?.setStyle(styleBlk); renderLegend();
}
function renderLegend() {
  const el = $("legend");
  const L_ = LAYER(S.v);
  if (L_.categorical) {
    el.innerHTML = `<b>${esc(L_.legend)}</b>` + Object.keys(S.meta.severity_colors).map((s) =>
      `<div class="cat"><span class="sw" style="background:${SEV(s).hex}"></span>${SEV(s).icon} ${esc(sevLabel(s))}</div>`).join("");
    return;
  }
  const ramp = D().ramps[L_.ramp];
  const ticks = [L_.end_labels?.[0] ?? "", ...L_.bins.map(String)];
  ticks[ticks.length - 1] += L_.end_labels ? "" : "+";
  if (L_.end_labels) ticks.push(L_.end_labels[1]);
  el.innerHTML = `<b>${esc(L_.legend)}</b><div class="ramp">${ramp.map((c) => `<span style="background:${c}"></span>`).join("")}</div>
    <div class="ticks">${ticks.filter((_, i) => i % 2 === 0 || i === ticks.length - 1).map((t) => `<span>${esc(t)}</span>`).join("")}</div>
    <div class="cat"><span class="sw" style="background:transparent;border:1.5px dashed var(--ink-2)"></span>${L_.block_field ? "no Gram Panchayat: block value" : "no Gram Panchayat"}</div>`;
}

/* ---------------------------------------------------------------- issue loading */
async function loadIssue(issue) {
  S.issue = issue;
  banner("");
  try { S.data = await api(`/api/issues/${issue}`); }
  catch (e) { banner(`Could not load forecast ${issue}: ${e.message}`); return; }
  const r = $("leadRange"); r.max = S.data.leads.length;
  if (S.issue !== S.lastIssue) { S.lead = 0; S.lastIssue = S.issue; }
  r.value = Math.min(S.lead + 1, S.data.leads.length);
  S.lead = Number(r.value) - 1;
  updateLeadLabel();
  const gaps = Object.entries(S.data.meta.input_gaps || {});
  banner(gaps.length ? `Today's forecast was made with incomplete inputs: ${gaps.map(([k, g]) => `${D().input_labels[k]} ${Math.round(g.available * 100)}% available (normally ${Math.round(g.seasonal_norm * 100)}%)`).join("; ")}. It will be complete once that data arrives.` : "");
  $("csvBtn").href = `/api/export/${issue}.csv`;
  restyle();
  if (S.sel) selectGp(S.sel, false);
}
const leadName = (k) => (k === 0 ? "Today" : `Day ${k}`);
const validated = (k) => S.meta.lead_days.includes(k);
const rangeText = (a) => (a.length ? `${Math.min(...a)}–${Math.max(...a)}` : "");
const unvalidatedNames = (leads) => leads.filter((k) => !validated(k)).map(leadName).join(", ");
function updateLeadLabel() {
  const d = S.data?.valid_dates?.[S.lead], k = S.data?.leads?.[S.lead];
  $("leadLabel").textContent = d ? `${leadName(k)} · ${d}${validated(k) ? "" : " *"}` : "";
  $("leadLabel").title = validated(k) ? "" : `Outside the validated ${rangeText(S.meta.lead_days)} day range: indicative, advisories not issued for this day`;
}

/* ---------------------------------------------------------------- GP detail */
async function selectGp(code, pan = true) {
  S.sel = code; restyle();
  activateTab("t-gp");
  const layer = S.byCode[code];
  if (pan && layer) mapG.fitBounds(layer.getBounds(), { maxZoom: 11 });
  $("gpEmpty").hidden = true;
  const box = $("gpDetail"); box.hidden = false; box.innerHTML = `<div class="muted">Loading…</div>`;
  let fc, b;
  try {
    [fc, b] = await Promise.all([api(`/api/forecast?gp_code=${code}&issue_date=${S.issue}`),
                                 api(`/api/advisory/${code}?issue_date=${S.issue}&lang=${S.lang}`)]);
  } catch (e) { box.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
  const days = fc.days;
  const rows = days.map((d, i) => `<tr${i === S.lead ? ' style="font-weight:700"' : ""}>
    <td>${esc(leadName(d.lead_day))}${validated(d.lead_day) ? "" : " *"}<br><span class="muted">${d.valid_date.slice(8, 10)}/${d.valid_date.slice(5, 7)}</span></td>
    <td>${num(d.fc_rain)}</td><td><b>${num(d.rain_pred)}</b></td>
    <td>${num(d.rain_q10)}–${num(d.rain_q90)}</td><td>${d.rain_p_ge_2p5 == null ? "–" : Math.round(d.rain_p_ge_2p5 * 100) + "%"}</td>
    <td>${num(d.tmax_pred)}/${num(d.tmin_pred)}</td><td>${num(d.rh_pred, 0)}</td><td>${num(d.wind_pred, 0)}</td></tr>`).join("");
  const advs = [...b.general, ...b.crop_advisories, ...b.disease_alerts];
  const L_ = b.labels;
  box.innerHTML = `
    <div class="card">
      <div class="row" style="justify-content:space-between">
        <div><h3 style="margin:0">${esc(b.gp_name)}</h3><div class="muted">${esc(L_.block)}: ${esc(b.block_name)} · LGD ${esc(code)} · ${esc(L_.issued)} ${esc(b.issue_date)}</div></div>
        <span class="sev ${b.overall_severity}" aria-label="Warning level">${SEV(b.overall_severity).icon} ${esc(b.overall_label)}</span>
      </div>
      <p class="muted" style="margin:8px 0 0">${esc(b.summary)}</p>
    </div>
    <div class="card"><h3>${days.length}-day forecast: block vs panchayat</h3>
      <div class="chart" id="gpChart" aria-label="Rain by day, block vs panchayat"></div>
      <table class="data"><thead><tr><th>Day</th><th>Block rain</th><th>GP rain</th><th>Likely range</th><th>Chance</th><th>Tmax/Tmin</th><th>RH</th><th>Wind</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="muted">Rain in mm. Range = P10–P90 (80% interval, conformally calibrated).${days.some((d) => !validated(d.lead_day)) ? ` * ${unvalidatedNames(days.map((d) => d.lead_day))}: outside the validated day ${rangeText(S.meta.lead_days)} range (indicative); advisories cover days ${rangeText(S.meta.lead_days)}.` : ""}</p></div>
    <div class="card"><h3>Why this panchayat differs from its block (${leadName(S.data.leads[S.lead]).toLowerCase()})</h3><div id="shap"></div>
      <p class="muted">Contribution of each factor group to the panchayat's rain correction (mm, exact TreeSHAP). Positive = wetter than the block value.</p></div>
    <div class="card"><h3>${esc(L_.general_advisory)} / ${esc(L_.crop_advisory)}</h3>
      ${b.crops?.length ? `<p class="muted">${b.crops.map(esc).join(" · ")}</p>` : ""}
      ${advs.map((a) => `<div class="adv ${a.severity}"><b>${SEV(a.severity).icon} ${esc(a.title)}</b>${esc(a.text)} ${esc(a.action)}</div>`).join("")}
      <div class="muted">${esc(b.review?.status === "approved" ? `${L_.reviewed_by}: ${b.review.officer}` : b.review?.note || "")}</div></div>
    <div class="card"><h3>Bulletin & farmer messages</h3>
      <div class="row">
        <a class="btn primary" href="/api/bulletin/${code}.pdf?issue_date=${S.issue}&lang=${S.lang}">⬇ PDF bulletin</a>
        <a class="btn" href="/api/bulletin/${code}.txt?issue_date=${S.issue}&lang=${S.lang}" target="_blank" rel="noopener">Text</a>
        <a class="btn" href="/api/ivr/${code}?issue_date=${S.issue}&lang=${S.lang}" target="_blank" rel="noopener">IVR script</a>
      </div>
      <p class="muted" style="margin-bottom:4px">SMS (${[...b.sms].length} characters):</p><div class="sms">${esc(b.sms)}</div>
      <details style="margin-top:8px"><summary>Register for SMS / WhatsApp (demo, nothing is sent)</summary>
        <form id="subForm" class="row" style="margin-top:6px">
          <input type="text" name="name" placeholder="Name" aria-label="Name" style="flex:1 1 120px">
          <input type="tel" name="phone" placeholder="Mobile number" aria-label="Mobile number" required style="flex:1 1 140px">
          <select name="channel" aria-label="Channel"><option value="sms">SMS</option><option value="whatsapp">WhatsApp</option><option value="ivr">IVR call</option></select>
          <button class="btn" type="submit">Register</button><span id="subMsg" class="muted"></span>
        </form></details></div>`;
  drawGpChart(days);
  drawShap(days[S.lead]);
  $("subForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    try {
      const r = await api("/api/subscribe", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: f.get("name"), phone: f.get("phone"), gp_code: code, lang: S.lang, channel: f.get("channel") }) });
      $("subMsg").textContent = `Registered (…${r.phone_last2}), ${r.channel}, ${r.lang}.`;
      e.target.reset();
    } catch (err) { $("subMsg").textContent = err.message; }
  });
}

function drawGpChart(days) {
  const el = $("gpChart"); const W = el.clientWidth || 380, H = 170, P = { l: 34, r: 8, t: 10, b: 24 };
  const maxV = Math.max(1, ...days.flatMap((d) => [d.rain_q90 ?? 0, d.fc_rain ?? 0, d.rain_pred ?? 0])) * 1.1;
  const x = (i) => P.l + (i + 0.5) * (W - P.l - P.r) / days.length;
  const y = (v) => H - P.b - (v / maxV) * (H - P.t - P.b);
  const bw = Math.min(18, (W - P.l - P.r) / days.length / 3);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Block vs panchayat rain">`;
  for (let g = 0; g <= 4; g++) { const v = maxV * g / 4;
    s += `<line x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${P.l - 4}" y="${y(v) + 3}" font-size="10" text-anchor="end" fill="var(--ink-2)">${v.toFixed(0)}</text>`; }
  days.forEach((d, i) => {
    const cx = x(i);
    if (i === S.lead) s += `<rect x="${cx - (W - P.l - P.r) / days.length / 2 + 2}" y="${P.t}" width="${(W - P.l - P.r) / days.length - 4}" height="${H - P.t - P.b}" rx="4" fill="var(--focus)" opacity=".15"/>`;
    const blk = d.fc_rain ?? 0, gp = d.rain_pred ?? 0;
    s += `<rect x="${cx - bw - 1}" y="${y(blk)}" width="${bw}" height="${Math.max(0, y(0) - y(blk))}" rx="3" fill="var(--block)" data-i="${i}" data-k="b"/>`;
    s += `<rect x="${cx + 1}" y="${y(gp)}" width="${bw}" height="${Math.max(0, y(0) - y(gp))}" rx="3" fill="var(--model)" data-i="${i}" data-k="g"/>`;
    if (d.rain_q10 != null) s += `<line x1="${cx + 1 + bw / 2}" x2="${cx + 1 + bw / 2}" y1="${y(d.rain_q10)}" y2="${y(d.rain_q90)}" stroke="var(--ink)" stroke-width="2"/>`;
    s += `<text x="${cx}" y="${H - 8}" font-size="10" text-anchor="middle" fill="var(--ink-2)"${i === S.lead ? ' font-weight="700"' : ""}>${d.lead_day === 0 ? "Today" : "D" + d.lead_day}</text>`;
    s += `<rect x="${cx - bw - 4}" y="${P.t}" width="${2 * bw + 8}" height="${H - P.t - P.b}" fill="transparent" data-hit="${i}"/>`;
  });
  s += `<g font-size="10" fill="var(--ink)"><rect x="${P.l}" y="0" width="9" height="9" rx="2" fill="var(--block)"/><text x="${P.l + 12}" y="8">Block</text>
        <rect x="${P.l + 52}" y="0" width="9" height="9" rx="2" fill="var(--model)"/><text x="${P.l + 64}" y="8">Panchayat (P10–P90)</text></g></svg>`;
  el.innerHTML = s;
  el.querySelectorAll("[data-hit]").forEach((r) => {
    const i = Number(r.dataset.hit), d = days[i];
    r.addEventListener("mousemove", (e) => tip({ originalEvent: e }, `<b>${leadName(d.lead_day)}</b> ${d.valid_date.slice(0, 10)}<br>Block ${num(d.fc_rain)} mm<br>Panchayat <b>${num(d.rain_pred)}</b> mm (${num(d.rain_q10)}–${num(d.rain_q90)})`));
    r.addEventListener("mouseleave", hideTip);
  });
}

const FAMILY_LABEL = { forecast: "Block forecast", season: "Season", terrain: "Terrain / Ghats", landcover: "Land cover",
  hydro: "Rivers / reservoirs", soil: "Soil", vegetation: "Vegetation (NDVI)", climatology: "Rain climatology",
  history: "Past local bias", antecedent: "Recent rain", other: "Other" };
function drawShap(day) {
  const el = $("shap");
  const items = Object.entries(day).filter(([k]) => k.startsWith("shap_rain_") && k !== "shap_rain_baseline")
    .map(([k, v]) => [k.replace("shap_rain_", ""), v]).filter(([, v]) => v != null)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 7);
  if (!items.length) { el.innerHTML = `<div class="muted">No explanation available.</div>`; return; }
  const m = Math.max(0.1, ...items.map(([, v]) => Math.abs(v)));
  el.innerHTML = items.map(([k, v]) => {
    const w = (Math.abs(v) / m) * 50;
    return `<div class="bar-row"><span>${esc(FAMILY_LABEL[k] || k)}</span><div class="bar-track"><span class="zero"></span>
      <span class="fill" style="background:${v >= 0 ? "var(--model)" : "var(--block)"};${v >= 0 ? `left:50%` : `left:${50 - w}%`};width:${w}%"></span></div>
      <span style="text-align:right">${v >= 0 ? "+" : ""}${v.toFixed(2)}</span></div>`;
  }).join("");
}

/* ---------------------------------------------------------------- scorecard */
const VAR_INFO = { rain: ["Rain", "mm"], tmax: ["Max temperature", "°C"], tmin: ["Min temperature", "°C"],
  rh: ["Humidity", "%"], wind: ["Wind", "km/h"] };
const better = (x) => (x == null || !isFinite(x) ? "–" : x >= 0 ? `${(x * 100).toFixed(0)}% better` : `${(-x * 100).toFixed(0)}% worse`);

async function renderScorecard() {
  const el = $("scorecard");
  let s;
  try { s = await api("/api/evaluation/summary"); } catch (e) { el.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
  const pv = s.headline?.per_variable || {};
  const kpi = (v) => { const d = pv[v], [name, u] = VAR_INFO[v];
    return `<div class="kpi"><div class="v">${better(d.skill_vs_block_copy)}</div><div class="l"><b>${name}</b><br>
      typical error ${num(d.rmse_model, 1)} ${u} (block value alone: ${num(d.rmse_block_copy, 1)} ${u})</div></div>`; };
  const byLead = (v) => s.variables?.[v]?.by_lead || {};
  const leads = S.meta.live_lead_days;
  const [t0, t1] = s.test_period.map((d) => new Date(d).toLocaleDateString("en-GB", { month: "short", year: "numeric" }));
  const leadRow = (k) => {
    const r = byLead("rain")[k], t = byLead("tmax")[k];
    if (!r) return `<tr><td>${leadName(k)}</td><td colspan="3" class="muted">not tested (indicative only)</td></tr>`;
    return `<tr><td>${leadName(k)}</td><td>${num(r.pred?.rmse, 1)} mm</td><td>${num(r.bl_block_copy?.rmse, 1)} mm</td><td>${better(r.pred?.skill_vs_block_copy)}${t ? ` · Tmax ${better(t.pred?.skill_vs_block_copy)}` : ""}</td></tr>`;
  };
  el.innerHTML = `
    <div class="card"><h3>How accurate is the panchayat forecast?</h3>
      <p class="muted">Tested on real ${esc(S.meta.forecast_label)} forecasts (days ${rangeText(s.tested_leads)}) from ${t0} to ${t1}, on panchayats the model had never seen,
        against ${s.truth_labels.map(esc).join(" and ")}. "Better" = smaller error than using the block value for every panchayat.</p>
      <div class="kpis">${Object.keys(VAR_INFO).filter((v) => pv[v]).map(kpi).join("")}</div></div>
    <div class="card"><h3>Rain accuracy by forecast day</h3>
      <table class="data"><thead><tr><th>Day</th><th>Panchayat error</th><th>Block error</th><th>Result</th></tr></thead>
      <tbody>${leads.map(leadRow).join("")}</tbody></table>
      <p class="muted">Error = typical (root-mean-square) difference from what was observed. ${unvalidatedNames(leads) ? `${unvalidatedNames(leads)} are shown on the map but were not part of testing.` : ""}</p></div>
    <p><a class="btn" href="/api/report" target="_blank" rel="noopener">Full technical validation report</a></p>`;
}

/* ---------------------------------------------------------------- officer */
async function renderOfficer() {
  const el = $("officer");
  let tok = ""; try { tok = sessionStorage.getItem("adminToken") || ""; } catch { /* no storage */ }
  let rv = { issue: { status: "draft" }, reviews: [] }, h = {};
  try { [rv, h] = await Promise.all([api(`/api/review/${S.issue}`), api("/api/health")]); } catch { /* shown below */ }
  el.innerHTML = `
    <div class="card"><h3>Access</h3><p class="muted">Write actions need the officer token (${esc(h.auth || "")}).</p>
      <div class="row"><input type="text" id="tok" placeholder="X-Admin-Token" value="${esc(tok)}" aria-label="Officer token" style="flex:1">
      <button class="btn" id="tokSave">Save</button></div></div>
    <div class="card"><h3>Issue ${esc(S.issue)}: <span class="sev ${rv.issue.status === "published" ? "green" : "yellow"}">${esc(rv.issue.status)}</span></h3>
      <p class="muted">${rv.reviews.length} panchayat bulletins reviewed.</p>
      <div class="row"><input type="text" id="officerName" placeholder="Officer name" aria-label="Officer name" style="flex:1">
        <button class="btn" id="approveBtn" ${S.sel ? "" : "disabled"}>Approve selected GP</button>
        <button class="btn" id="rejectBtn" ${S.sel ? "" : "disabled"}>Reject</button></div>
      <textarea id="note" rows="2" style="width:100%;margin-top:6px" placeholder="Review note / corrected SMS text" aria-label="Review note"></textarea>
      <div class="row" style="margin-top:6px"><button class="btn primary" id="publishBtn">Publish issue</button>
        <button class="btn" id="dissBtn">Disseminate (simulated)</button><span id="offMsg" class="muted"></span></div></div>
    <div class="card"><h3>Run the downscaler</h3>
      <div class="row"><button class="btn" id="runLive">Run today's live forecast</button></div>
      <form id="upForm" style="margin-top:8px"><p class="muted">Or upload the official block forecast (CSV: block_lgd/block_name, lead_day, rain, tmax, tmin, rh, wind). <a href="/api/template/block_forecast.csv">Template</a></p>
        <div class="row"><input type="date" name="issue_date" required aria-label="Issue date"><input type="file" name="file" accept=".csv" required aria-label="Block forecast CSV">
        <button class="btn" type="submit">Downscale</button></div></form><div id="runMsg" class="muted"></div></div>
    <div class="card"><h3>Short-range rain (NWP, next hours)</h3><div id="nowcast" class="muted">Loading…</div></div>
    <div class="card"><h3>Outbox (simulated)</h3><div id="outbox" class="muted">Loading…</div></div>
    <div class="card"><h3>Audit log</h3><div id="audit" class="muted">Loading…</div></div>`;
  $("tokSave").onclick = () => { try { sessionStorage.setItem("adminToken", $("tok").value.trim()); } catch { /* ignore */ } $("offMsg").textContent = "Token saved for this tab."; };
  const officer = () => {
    const n = $("officerName").value.trim();
    if (n.length < 2) throw new Error("Enter the reviewing officer's name first.");
    return n;
  };
  const review = async (status) => {
    try {
      const edits = $("note").value.trim() && status === "approved" && $("note").value.trim().length <= 160 ? { sms: $("note").value.trim() } : {};
      await api(`/api/review/${S.issue}/${S.sel}`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status, officer: officer(), note: $("note").value.trim(), edits }) });
      $("offMsg").textContent = `${status} ${S.sel}`; renderOfficer();
    } catch (e) { $("offMsg").textContent = e.message; }
  };
  $("approveBtn").onclick = () => review("approved");
  $("rejectBtn").onclick = () => review("rejected");
  $("publishBtn").onclick = async () => { try { await api(`/api/review/${S.issue}/publish`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ officer: officer() }) }); renderOfficer(); } catch (e) { $("offMsg").textContent = e.message; } };
  $("dissBtn").onclick = async () => { try { const r = await api(`/api/disseminate/${S.issue}`, { method: "POST" }); $("offMsg").textContent = `${r.subscribers} subscribers: ${JSON.stringify(r.queued)} (simulated)`; renderOfficer(); } catch (e) { $("offMsg").textContent = e.message; } };
  $("runLive").onclick = async () => { $("runMsg").textContent = "Running… (about a minute)"; try { const m = await api("/api/run?source=live", { method: "POST" }); $("runMsg").textContent = `Issued ${m.issue_date}: ${m.n_gps} panchayats in ${m.seconds}s.`; await init(true); } catch (e) { $("runMsg").textContent = e.message; } };
  $("upForm").onsubmit = async (e) => { e.preventDefault(); $("runMsg").textContent = "Downscaling…"; try { const m = await api("/api/downscale", { method: "POST", body: new FormData(e.target) }); $("runMsg").textContent = `Issued ${m.issue_date}: ${m.n_gps} panchayats.`; await init(true); } catch (err) { $("runMsg").textContent = err.message; } };
  api("/api/nowcast").then((n) => {
    const names = Object.fromEntries((S.geoBlk?.features || []).map((f) => [String(f.properties.block_lgd), f.properties.block_name]));
    $("nowcast").innerHTML = `<p>${esc(n.source)}</p><table class="data"><thead><tr><th>Block</th><th>Next ${num(Object.values(n.blocks)[0]?.hours, 0)} h (mm)</th><th>Max 15-min</th></tr></thead><tbody>
      ${Object.entries(n.blocks || {}).map(([k, b]) => `<tr><td>${esc(names[k] || k)}</td><td>${num(b.mm_total)}${b.missing_steps ? " *" : ""}</td><td>${num(b.max_15min_mm)}</td></tr>`).join("")}</tbody></table>`;
  }).catch((e) => { $("nowcast").textContent = e.message; });
  api("/api/outbox?limit=15").then((o) => { $("outbox").innerHTML = o.length ? `<table class="data"><tbody>${o.map((m) => `<tr><td>${esc(m.channel)}</td><td>${esc(m.gp_code)}</td><td style="text-align:left">${esc(m.message.slice(0, 90))}${m.message.length > 90 ? "…" : ""}</td><td>${m.segments}</td></tr>`).join("")}</tbody></table>` : "Empty."; }).catch((e) => { $("outbox").textContent = e.message; });
  api("/api/audit?limit=12").then((a) => { $("audit").innerHTML = a.length ? a.map((x) => `<div>${esc(x.ts_utc)} · ${esc(x.actor)} · ${esc(x.action)}</div>`).join("") : "Empty."; }).catch((e) => { $("audit").textContent = e.message; });
}

/* ---------------------------------------------------------------- tabs, search, locate */
function activateTab(id) {
  document.querySelectorAll('[role="tab"]').forEach((t) => {
    const on = t.id === id; t.setAttribute("aria-selected", on);
    $(t.getAttribute("aria-controls")).hidden = !on;
  });
  if (id === "t-score") renderScorecard();
  if (id === "t-officer") renderOfficer();
}
function setupSearch() {
  const inp = $("search"), ul = $("searchResults"); let items = [], idx = -1, timer;
  const close = () => { ul.hidden = true; inp.setAttribute("aria-expanded", "false"); };
  const pick = (it) => { close(); inp.value = it.gp_name; selectGp(it.gp_code); };
  inp.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      if (inp.value.trim().length < 2) { close(); return; }
      try { items = await api(`/api/search?q=${encodeURIComponent(inp.value.trim())}`); } catch { items = []; }
      idx = -1;
      ul.innerHTML = items.map((it, i) => `<li role="option" id="sr${i}" aria-selected="false">${esc(it.gp_name)} <span class="muted">${esc(it.block_name)}</span></li>`).join("") || `<li class="muted">No match</li>`;
      ul.hidden = false; inp.setAttribute("aria-expanded", "true");
      ul.querySelectorAll("li[role=option]").forEach((li, i) => li.addEventListener("mousedown", () => pick(items[i])));
    }, 180);
  });
  inp.addEventListener("keydown", (e) => {
    if (ul.hidden || !items.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault(); idx = (idx + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      ul.querySelectorAll("li").forEach((li, i) => li.setAttribute("aria-selected", i === idx));
      inp.setAttribute("aria-activedescendant", `sr${idx}`);
    } else if (e.key === "Enter" && idx >= 0) { e.preventDefault(); pick(items[idx]); }
    else if (e.key === "Escape") close();
  });
  inp.addEventListener("blur", () => setTimeout(close, 150));
}
function setupLocate() {
  $("locateBtn").onclick = () => {
    if (!navigator.geolocation) { banner("Location is not available in this browser."); return; }
    navigator.geolocation.getCurrentPosition(async (p) => {
      try { const r = await api(`/api/locate?lat=${p.coords.latitude}&lon=${p.coords.longitude}`); selectGp(r.gp_code); }
      catch (e) { banner(e.message); }
    }, () => banner("Location permission denied."), { timeout: 10000 });
  };
}

/* ---------------------------------------------------------------- controls built from /api/meta */
function buildControls() {
  // map layers
  const layers = D().layers;
  const saved = store.get("layer", "");
  S.v = layers[saved] ? saved : Object.keys(layers)[0];
  $("varSel").innerHTML = Object.entries(layers).map(([k, l]) => `<option value="${k}"${k === S.v ? " selected" : ""}>${esc(l.label)}</option>`).join("");
  // bulletin languages
  const langs = S.meta.languages, savedLang = store.get("lang", "");
  S.lang = langs.includes(savedLang) ? savedLang : S.meta.default_language;
  document.documentElement.lang = S.lang;
  $("langSeg").innerHTML = langs.map((l) => `<button data-lang="${l}" aria-pressed="${l === S.lang}">${esc(D().language_names[l])}</button>`).join("");
  $("langSeg").querySelectorAll("[data-lang]").forEach((b) => {
    b.onclick = () => {
      S.lang = b.dataset.lang; store.set("lang", S.lang);
      $("langSeg").querySelectorAll("[data-lang]").forEach((x) => x.setAttribute("aria-pressed", x === b));
      document.documentElement.lang = S.lang; renderLegend();
      if (S.sel) selectGp(S.sel, false);
    };
  });
  // forecast horizon
  const h = S.meta.live_lead_days;
  $("leadRange").max = h.length;
}

/* ---------------------------------------------------------------- init */
async function init(reloadIssues = false) {
  if (!reloadIssues) {
    try { S.meta = await api("/api/meta"); } catch (e) { banner(`API unavailable: ${e.message}`); return; }
    const bq = S.meta.boundaries;
    document.title = D().app_name; $("appTitle").textContent = D().app_name;
    $("appSub").textContent = `${S.meta.district} district` + (bq ? ` · ${bq.modelled_gram_panchayats} Gram Panchayats · ${bq.blocks} blocks` : "");
    buildControls();
    [S.geoGp, S.geoBlk, S.geoUrban] = await Promise.all([api("/api/geo/panchayats"), api("/api/geo/blocks"),
      api("/api/geo/urban_areas")]);
    mapG = makeMap("mapG"); mapB = makeMap("mapB");
    const b = L.geoJSON(S.geoBlk).getBounds(); mapG.fitBounds(b); mapB.fitBounds(b);
    syncMaps(mapG, mapB); syncMaps(mapB, mapG);
    drawLayers();
  }
  const iss = await api("/api/issues");
  S.issues = iss.issues;
  if (!S.issues.length) { banner("No forecast issues yet. Run: python -m src.pipeline.run --only forecast"); return; }
  $("issueSel").textContent = S.issues[0];
  await loadIssue(S.issues[0]);
}

document.addEventListener("DOMContentLoaded", () => {
  const theme = store.get("theme", "");
  if (theme) document.documentElement.dataset.theme = theme;
  $("themeBtn").onclick = () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const nxt = cur === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = nxt; store.set("theme", nxt);
  };
  $("leadRange").oninput = (e) => { S.lead = Number(e.target.value) - 1; updateLeadLabel(); restyle(); if (S.sel) selectGp(S.sel, false); };
  $("varSel").onchange = (e) => { S.v = e.target.value; store.set("layer", S.v); restyle(); };
  const setView = (v) => {
    S.view = v; store.set("view", v);
    $("maps").classList.toggle("single", v === "single");
    $("viewSplit").setAttribute("aria-pressed", v === "split"); $("viewSingle").setAttribute("aria-pressed", v === "single");
    setTimeout(() => { mapG?.invalidateSize(); mapB?.invalidateSize(); }, 50);
  };
  $("viewSplit").onclick = () => setView("split"); $("viewSingle").onclick = () => setView("single");
  document.querySelectorAll('[role="tab"]').forEach((t) => (t.onclick = () => activateTab(t.id)));
  setupSearch(); setupLocate();
  setView(store.get("view", "split"));
  init();
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
});
