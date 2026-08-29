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

async function refresh() {
  const q = new URLSearchParams({
    planet: $("planet").value, scenario: $("scenario").value,
    n_transits: $("n_transits").value, R: $("R").value, seed: seed,
  });
  const ctrl = new AbortController();
  if (inflight) inflight.abort();
  inflight = ctrl;
  $("seedlabel").textContent = `seed = ${seed}`;
  try {
    const r = await fetch(`/api/observe?${q}`, { signal: ctrl.signal });
    let j;
    try {
      j = await r.json();
    } catch {
      $("error").textContent = `Server error (HTTP ${r.status})`;
      return;
    }
    if (!r.ok) { $("error").textContent = j.error; return; }
    $("error").textContent = "";
    renderPlot(j);
    renderDetections(j.detections);
  } catch (e) {
    if (e.name !== "AbortError") $("error").textContent = String(e);
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
  for (const id of ["scenario", "n_transits", "R"])
    $(id).addEventListener("change", refresh);
  $("reroll").addEventListener("click", () => { seed += 1; refresh(); });
  $("show_model").addEventListener("change", () =>
    Plotly.restyle("plot", { visible: $("show_model").checked ? true : "legendonly" }, [0]));
  refresh();
}

init();
