# One-off POSEIDON diagnostics (poseidon env, ~1-3 min each)

Direct POSEIDON runs used on 2026-10-04 to validate the parametric model's
scaling assumptions; their numbers are quoted in spec amendments 8-10.

- `zeta_diag.py` — a K2-18 b-like planet's own zeta(lambda) vs the shifted
  1 R_J fiducial (median +0.06, peaks +0.14; fiducial self-check exact).
- `split_diag.py` — does zeta at fixed slant column x depend on the (R_p, g)
  split? (slope dzeta/dx ~1 at 300 K, ~0.6 at 1000 K; small H/R_p term.)
- `deck_diag2.py` / `deck_diag_w12.py` — clear + 1 mbar deck for K2-18 b-like
  and WASP-12 b-like planets vs the model (deck realized at 1.376 mbar; Ein
  deck operator; hot-end extrapolation).

Run: `POSEIDON_input_data=$HOME/inputs/ ~/anaconda3/envs/poseidon/bin/python precompute/diag/<script>.py`
