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

let lastFeatureSnr = null;

function renderFeatureSnr(f) {
  lastFeatureSnr = f;
  const entries = [["atmosphere (template)",
                    f.template_snr_diagonal.toFixed(1)]];
  if ($("show_bands").checked) {
    Object.entries(f.band_snr).sort((a, b) => b[1] - a[1])
      .forEach(([k, v]) => entries.push([`${k} band`, v.toFixed(1)]));
  }
  $("featuresnr").querySelector("tbody").innerHTML = entries
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

function renderStatus(j) {
  $("status").innerHTML = (j.caveats || []).map((c) => {
    const low = c.toLowerCase();
    const bad = low.includes("saturat");
    const short = bad ? "unvalidated noise (saturation)"
      : low.includes("toy") ? "H₂/He toy scenarios"
      : "statistical noise only";
    return `<span class="chip ${bad ? "bad" : "warn"}" title="${c}">` +
      `${short}</span>`;
  }).join("");
}

function renderPlot(j) {
  const model = {
    x: j.model_wavelength, y: pct(j.model_depth), mode: "lines",
    name: `model (${j.scenario})`,
    line: { width: 1, color: "#168ea0" }, opacity: 0.7,
    visible: $("show_model").checked ? true : "legendonly",
    hovertemplate: "%{x:.3f} µm — %{y:.4f}%<extra>model</extra>",
  };
  const points = {
    x: j.wavelength, y: pct(j.points), mode: "markers",
    name: `simulated (R=${j.R})`,
    marker: { size: 5, color: "#b3261e" },
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
    font: { family: 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", ' +
            "Inter, sans-serif", color: "#485a60" },
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
    font: { family: 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", ' +
            "Inter, sans-serif", color: "#485a60" },
  };
  Plotly.react("lcplot", traces, layout, { responsive: true });

  if (j.shape === "quadratic-ld") {
    const uSpan = j.groups.length > 1
      ? `u₁ = ${j.groups[0].u1.toFixed(2)}→` +
        `${j.groups[j.groups.length - 1].u1.toFixed(2)} (blue→red)`
      : `u₁ = ${j.groups[0].u1.toFixed(2)}, ` +
        `u₂ = ${j.groups[0].u2.toFixed(2)}`;
    $("sysparams").textContent =
      `Quadratic limb darkening (PHOENIX via exotic-ld), ${uSpan}; ` +
      `b = ${j.b.toFixed(2)}, a/R★ = ${j.a_over_rs.toFixed(1)} ` +
      `(NEA PSCompPars).`;
  } else {
    $("sysparams").textContent =
      "Trapezoid fallback: central transit (b = 0), no limb darkening " +
      "(no registry entry or LD table for this planet).";
  }
}

function rows(tbodyId, entries) {
  $(tbodyId).querySelector("tbody").innerHTML = entries
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

function fmt(v, digits = 2) {
  return (v === null || v === undefined) ? null : Number(v).toFixed(digits);
}

let lastTarget = null;

function renderTarget(j) {
  lastTarget = j;
  const t = j.target, d = j.derived, o = j.observing;
  rows("targetparams", [
    ["Rₚ (Rⱼ)", fmt(t.R_p_RJ, 3)],
    ["Mₚ (Mⱼ)", fmt(t.M_p_MJ, 3)],
    ["Tₑq (K)", fmt(t.T_eq_K, 0)],
    ["gₚ (m/s²)", fmt(d.g_p_ms2, 1)],
    ["H (km)", fmt(d.H_km, 0)],
    ["P (d)", fmt(t.period_d, 4)],
    ["t₁₄ (h)", fmt(t.t14_s / 3600, 2)],
    ["a/R★", fmt(t.a_over_rs, 2)],
    ["i (°)", fmt(t.inc_deg, 2)],
    ["e", fmt(t.ecc, 3)],
    ["b", fmt(d.b, 3)],
    ["depth (%)", fmt(t.depth * 100, 3)],
    ["Tₑff (K)", fmt(t.T_eff_K, 0)],
    ["log g★", fmt(t.log_g_star, 2)],
    ["[M/H]★", fmt(t.Met_star, 2)],
    ["R★ (R☉)", fmt(t.R_s_RSun, 3)],
    ["J (mag)", fmt(t.J_mag, 2)],
  ]);
  return o;
}

function hrs(seconds, digits = 2) {
  return (seconds === null || seconds === undefined) ? null
    : (seconds / 3600).toFixed(digits) + " h";
}

function renderDiagnostics(obs) {
  const w = obs.depth_err.map((s) => 1 / (s * s));
  const sigWhite = 1 / Math.sqrt(w.reduce((a, v) => a + v, 0));
  const errs = [...obs.depth_err].sort((a, b) => a - b);
  const medErr = errs[Math.floor(errs.length / 2)];
  const d = lastTarget ? lastTarget.derived : {};
  const o = lastTarget ? lastTarget.observing : {};
  const nt = obs.n_transits;

  rows("diag_spec", [
    ["Bins @ R=" + obs.R, obs.wavelength.length],
    ["Median σ/bin (ppm)", fmt(medErr * 1e6, 0)],
    ["White-light σ (ppm)", fmt(sigWhite * 1e6, 1)],
    ["Feature amp. (ppm)", fmt(d.feature_amp_ppm, 0)],
    ["Amp / σ per bin", d.feature_amp_ppm
      ? fmt(d.feature_amp_ppm / (medErr * 1e6), 1) : null],
    ["Template S/N", fmt(obs.feature_snr.template_snr_diagonal, 1)],
  ]);

  const retuned = o.macc && o.default_macc && o.macc !== o.default_macc;
  rows("diag_readout", [
    ["Instrument config", o.instrument],
    ["Readout mode", o.readout_mode],
    ["Frame time (s)", fmt(o.tframe_s, 2)],
    ["Ramp MACC n:m:d", o.macc],
    ["Frames per ramp", o.nframes_per_ramp],
    ["Ramp duration (s)", fmt(o.ramp_s, 1)],
    ["slicersim default", retuned
      ? `${o.default_macc} (${fmt(o.default_ramp_s, 0)} s, retuned)`
      : (o.default_macc ? `${o.default_macc} (kept)` : null)],
    ["Ramps simulated", o.nramps],
    ["Ramps per transit (√t)", fmt(o.ramps_per_transit, 0)],
    ["Saturation", o.fully_desaturated === undefined ? null
      : (o.fully_desaturated ? '<span class="chip ok">clean</span>'
         : `<span class="chip bad">${o.nsaturated_detpx} px residual</span>`)],
  ]);

  rows("diag_time", [
    ["In-transit / transit", hrs(o.t_in_s)],
    ["Baseline / transit", o.t_out_s !== undefined
      ? hrs(o.t_out_s) + " (equal)" : null],
    ["Transits", nt],
    ["Total in-transit", hrs(o.t_in_s * nt)],
    ["Total on target", (o.t_in_s && o.t_out_s)
      ? hrs((o.t_in_s + o.t_out_s) * nt) : null],
  ]);
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
    const needTarget = !lastTarget
      || lastTarget.target.name !== $("planet").value;
    const [obs, lc, target] = await Promise.all([
      fetchJSON(`/api/observe?${qObs}`, ctrl.signal),
      fetchJSON(`/api/lightcurve?${qLc}`, ctrl.signal),
      needTarget
        ? fetchJSON(`/api/target?planet=` +
                    encodeURIComponent($("planet").value), ctrl.signal)
        : Promise.resolve(null),
    ]);
    $("error").textContent = "";
    if (target) renderTarget(target);
    renderPlot(obs);
    renderStatus(obs);
    renderFeatureSnr(obs.feature_snr);
    renderLightcurve(lc);
    renderDiagnostics(obs);
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
  $("show_bands").addEventListener("change", () => {
    if (lastFeatureSnr) renderFeatureSnr(lastFeatureSnr);
  });
  $("show_model").addEventListener("change", () =>
    Plotly.restyle("plot", { visible: $("show_model").checked ? true : "legendonly" }, [0]));
  refresh();
}

init();
