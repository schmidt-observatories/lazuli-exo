"""Write the analytic 'flat' scenario tables: constant depth = (Rp/Rs)^2.

Run: ~/anaconda3/envs/py313/bin/python precompute/make_flat_tables.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from scenarios import make_wl_grid

from terminator import tables, targets

TABLES = tables.resolve_tables_dir(None)


def main():
    wl = make_wl_grid()
    for name in targets.PLANET_NAMES:
        t = targets.get_target(name)
        tables.write_model_table(
            TABLES, name, "flat", wl, np.full_like(wl, t["depth"]),
            meta={"generator": "make_flat_tables.py",
                  "depth": t["depth"], "provenance": t["provenance"]})
        print(f"flat: {name}  depth={t['depth']:.5f}")
    tables.rebuild_index(TABLES)


if __name__ == "__main__":
    main()
