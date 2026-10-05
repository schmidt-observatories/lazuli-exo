import json
from pathlib import Path

import pytest

REPORT = Path(__file__).resolve().parents[1] / "notebooks" / "figs" / "06_validation.json"


@pytest.mark.skipif(not REPORT.exists(), reason="run precompute/validate_zeta_grid.py")
def test_validation_report_passes():
    rows = json.loads(REPORT.read_text())
    assert sum(not r["like_for_like"] for r in rows) == 14 * 4
    gated = [r for r in rows if r["like_for_like"] or not r["T_clamped"]]
    failed = [(r["planet"], r["scenario"], round(r["amp_ratio"], 2))
              for r in gated if not r["passed"]]
    assert failed == []
