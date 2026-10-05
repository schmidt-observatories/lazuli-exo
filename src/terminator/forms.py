"""Web-form strings -> validated terminator.simulate() parameters.

Shared by every UI (local Flask demo, the Django simulators site) so that
defaults, ranges, clamping and error messages live in one place. A UI
passes its raw form/query mapping of strings to parse(); it gets typed
values, fatal `errors` (the run cannot proceed) and non-fatal `warnings`
(a value was replaced or clamped and the run proceeds).

Blank optional fields carry meaning: mu blank -> equilibrium-chemistry mu;
log_p_cloud blank -> no deck; log_a_haze blank -> no haze.
"""
from collections import namedtuple

from .targets import get_target

ParseResult = namedtuple("ParseResult", "values errors warnings")

# name -> (default, kind, extra). kind: float | int | choice | str | optfloat
# extra: (lo, hi) for numbers, tuple of choices for choice.
FIELDS = {
    "planet_name": ("", "str", None),
    "teff": ("5500", "float", (1000.0, 8000.0)),
    "logg": ("4.5", "float", (2.0, 6.0)),
    "feh": ("0.0", "float", (-2.0, 1.0)),
    "rstar": ("1.0", "float", (0.05, 20.0)),
    "mag": ("", "float", (-2.0, 25.0)),             # required (blank default)
    "band": ("J", "choice", ("J", "H", "Ks", "V")),
    "mag_system": ("vega", "choice", ("vega", "ab")),
    "rp": ("1.0", "float", (0.01, 3.0)),
    "mp": ("1.0", "float", (1e-4, 30.0)),
    "teq": ("1200", "float", (100.0, 4000.0)),
    "t14_h": ("3.0", "float", (0.05, 48.0)),
    "log_met": ("0.0", "float", (-1.0, 3.0)),
    "c_to_o": ("0.55", "float", (0.3, 1.0)),
    "mu": ("", "optfloat", (1.0, 100.0)),
    "log_p_cloud": ("", "optfloat", (-8.0, 1.0)),
    "log_a_haze": ("", "optfloat", (-2.0, 6.0)),
    "haze_slope": ("-4.0", "float", (-10.0, 0.0)),
    "n_transits": ("1", "int", (1, 1000)),
    "resolution": ("100", "int", (10, 500)),
    "seed": ("0", "int", (0, 10 ** 9)),
    "instrument_config": ("cbe", "choice", ("cbe", "eol")),
}
FIELD_DEFAULTS = {k: v[0] for k, v in FIELDS.items()}
REQUIRED = ("mag",)
POSITIVE = ("rstar", "rp", "mp", "teq", "t14_h")
INSTRUMENTS = {"cbe": "lazuli_cbe", "eol": "lazuli_eol"}


def _number(name, raw, kind, lo, hi, errors, warnings):
    cast = int if kind == "int" else float
    try:
        v = cast(float(raw)) if kind == "int" else float(raw)
    except (TypeError, ValueError):
        default = FIELD_DEFAULTS[name]
        if name in REQUIRED or default == "":
            errors.append(f"{name}: {raw!r} is not a number")
            return None
        warnings.append(f"{name}: {raw!r} is not a number, using {default}")
        return cast(default)
    if name in POSITIVE and v <= 0:
        errors.append(f"{name} must be > 0, got {v:g}")   # a typo, not something to clamp
        return v
    if not (lo <= v <= hi):
        clamped = min(max(v, lo), hi)
        warnings.append(f"{name}={v:g} outside [{lo:g}, {hi:g}], clamped to {clamped:g}")
        v = clamped
    return v


def parse(data):
    """Parse a mapping of form strings. Missing keys take their defaults."""
    values, errors, warnings = {}, [], []
    for name, (default, kind, extra) in FIELDS.items():
        raw = data.get(name)
        raw = default if raw is None else str(raw).strip()
        if kind == "str":
            values[name] = raw
        elif kind == "choice":
            if raw not in extra:
                if raw != "":
                    warnings.append(f"{name}: {raw!r} is not one of {list(extra)}, using {default!r}")
                raw = default
            values[name] = raw
        elif raw == "":
            if name in REQUIRED:
                errors.append(f"{name} is required")
            values[name] = None
        else:
            values[name] = _number(name, raw, kind, *extra, errors, warnings)
    return ParseResult(values, errors, warnings)


def to_params(v):
    """Typed form values -> the terminator.simulate() parameter dict."""
    return dict(name=v["planet_name"] or "custom", scenario="custom",
                T_eq_K=v["teq"], R_p_RJ=v["rp"], M_p_MJ=v["mp"], R_s_RSun=v["rstar"],
                teff_K=v["teff"], logg=v["logg"], feh=v["feh"], mag=v["mag"],
                band=v["band"], mag_system=v["mag_system"], t14_s=v["t14_h"] * 3600.0,
                log_met=v["log_met"], c_to_o=v["c_to_o"], mu=v["mu"],
                log_P_cloud_bar=v["log_p_cloud"], log_a_haze=v["log_a_haze"],
                haze_slope=v["haze_slope"])


def from_preset(planet, scenario="clear"):
    """Typed form values prefilled from a showcase planet + scenario."""
    from .presets import SCENARIO_PARAMS
    t = get_target(planet)
    s = SCENARIO_PARAMS[scenario]
    base = parse({}).values
    base.update(planet_name=t["name"], teff=t["T_eff_K"], logg=t["log_g_star"], feh=t["Met_star"],
                rstar=t["R_s_RSun"], mag=t["J_mag"], band="J", mag_system="vega",
                rp=t["R_p_RJ"], mp=t["M_p_MJ"], teq=t["T_eq_K"], t14_h=t["t14_s"] / 3600.0,
                log_met=s["log_met"], c_to_o=s["c_to_o"], mu=None,
                log_p_cloud=s["log_P_cloud_bar"], log_a_haze=s["log_a_haze"],
                haze_slope=s["haze_slope"])
    return base
