/* terminator local demo: controls + Plotly wiring. */
"use strict";

const $ = (id) => document.getElementById(id);
const XTICKS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7];

let targets = {};
let seed = 0;
let inflight = null;

function pct(arr) { return arr.map((v) => 100 * v); }

function fillScenarios(planet) {
  const sel = $("scenario");
  const prev = sel.value;
  sel.innerHTML = "";
  for (const s of targets[planet]) {
    const opt = document.createElement("option");
    opt.value = opt.textContent = s;
    sel.appendChild(opt);
  }
  sel.value = targets[planet].includes(prev) ? prev : targets[planet][0];
}

function renderDetections(det) {
  const rows = Object.entries(det)
    .sort((a, b) => (a[0] === "atmosphere" ? -1 : b[0] === "atmosphere" ? 1
                     : b[1] - a[1]))
    .map(([k, v]) => `<tr><td>${k}</td><td>${v.toFixed(1)}</td></tr>`);
  $("detections").querySelector("tbody").innerHTML = rows.join("");
}

function renderPlot(j) {
  const model = {
    x: j.model_wavelength, y: pct(j.model_depth), mode: "lines",
    name: `model (${j.scenario})`,
    line: { width: 1, color: "#0b7285" }, opacity: 0.7,
    visible: $("show_model").checked ? true : "legendonly",
    hovertemplate: "%{x:.3f} µm — %{y:.4f}%<extra>model</extra>",
  };
  const points = {
    x: j.wavelength, y: pct(j.points), mode: "markers",
    name: `simulated (R=${j.R})`,
    marker: { size: 5, color: "#c92a2a" },
    error_y: { type: "data", array: pct(j.depth_err), thickness: 1, width: 2 },
    hovertemplate: "%{x:.3f} µm — %{y:.4f}% ± %{error_y.array:.4f}%" +
      "<extra>simulated</extra>",
  };
  const n = j.n_transits;
  const layout = {
    title: { text: `${j.planet} — ${j.scenario}, ${n} transit${n > 1 ? "s" : ""}`,
             font: { size: 15 } },
    xaxis: { type: "log", tickvals: XTICKS, ticktext: XTICKS.map(String),
             title: { text: "Wavelength (µm)" } },
    yaxis: { title: { text: "Transit depth (%)" } },
    legend: { x: 0.02, y: 0.98 },
    margin: { t: 40, r: 20, b: 45, l: 60 },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff",
  };
  Plotly.react("plot", [model, points], layout, { responsive: true });
}

function bandColor(i, n) {
  const f = n > 1 ? i / (n - 1) : 0;      // 0 = bluest, 1 = reddest
  return `hsl(${Math.round(225 * (1 - f))}, 70%, 45%)`;
}

function renderLightcurve(j) {
  const n = j.groups.length;
  // stack curves with a shared offset step; bluest band on top
  const step = 1.6 * Math.max(...j.groups.map(
    (g) => g.depth + 6 * g.sigma_point));
  const traces = [];
  j.groups.forEach((g, i) => {
    const off = -i * step;
    const color = bandColor(i, n);
    traces.push({
      x: j.time_hr, y: g.flux.map((v) => v + off), mode: "markers",
      name: g.label, marker: { size: 3, color, opacity: 0.55 },
      hovertemplate: "%{x:.2f} hr — %{y:.5f}<extra>" + g.label + "</extra>",
    });
    traces.push({
      x: j.time_hr, y: g.flux_model.map((v) => v + off), mode: "lines",
      line: { width: 1.5, color }, showlegend: false, hoverinfo: "skip",
    });
  });
  const nt = j.n_transits;
  const layout = {
    title: { text: `Transit light curve${n > 1 ? "s" : ""} — ` +
            `${nt > 1 ? `phase-folded, ${nt} transits, ` : ""}` +
            `${(j.cadence_s).toFixed(0)} s sampling`, font: { size: 14 } },
    xaxis: { title: { text: "Time from mid-transit (hr)" } },
    yaxis: { title: { text: n > 1 ? "Relative flux + offset"
                                  : "Relative flux" } },
    legend: { x: 0.02, y: 0.02, yanchor: "bottom" },
    margin: { t: 40, r: 20, b: 45, l: 60 },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff",
  };
  Plotly.react("lcplot", traces, layout, { responsive: true });
}

async function fetchJSON(url, signal) {
  const r = await fetch(url, { signal });
  let j;
  try {
    j = await r.json();
  } catch {
    throw new Error(`Server error (HTTP ${r.status})`);
  }
  if (!r.ok) throw new Error(j.error);
  return j;
}

async function refresh() {
  const params = {
    planet: $("planet").value, scenario: $("scenario").value,
    n_transits: $("n_transits").value, R: $("R").value, seed: seed,
  };
  const qObs = new URLSearchParams(params);
  const qLc = new URLSearchParams({ ...params, groups: $("lc_groups").value });
  const ctrl = new AbortController();
  if (inflight) inflight.abort();
  inflight = ctrl;
  $("seedlabel").textContent = `seed = ${seed}`;
  try {
    const [obs, lc] = await Promise.all([
      fetchJSON(`/api/observe?${qObs}`, ctrl.signal),
      fetchJSON(`/api/lightcurve?${qLc}`, ctrl.signal),
    ]);
    $("error").textContent = "";
    renderPlot(obs);
    renderDetections(obs.detections);
    renderLightcurve(lc);
  } catch (e) {
    if (e.name !== "AbortError") $("error").textContent = e.message;
  }
}

async function init() {
  targets = await (await fetch("/api/targets")).json();
  const psel = $("planet");
  for (const p of Object.keys(targets)) {
    const opt = document.createElement("option");
    opt.value = opt.textContent = p;
    psel.appendChild(opt);
  }
  fillScenarios(psel.value);
  psel.addEventListener("change", () => { fillScenarios(psel.value); refresh(); });
  for (const id of ["scenario", "n_transits", "R", "lc_groups"])
    $(id).addEventListener("change", refresh);
  $("reroll").addEventListener("click", () => { seed += 1; refresh(); });
  $("show_model").addEventListener("change", () =>
    Plotly.restyle("plot", { visible: $("show_model").checked ? true : "legendonly" }, [0]));
  refresh();
}

init();
