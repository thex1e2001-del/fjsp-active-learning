"""Create a canonical, auditable summary from the saved FJSP experiment output.

This script deliberately treats ``results/exp1_main.json`` as historical data. It
does not manufacture missing repetitions or fair-baseline rows. It checks the
reported assignment-level results, applies one tie tolerance, and writes a
machine-readable summary for the manuscript.
"""
from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TIE_TOL = 1e-3


def load_main():
    path = RESULTS / "exp1_main.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    required = {"instance", "balanced", "rf_best", "best_known"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{row.get('instance')}: missing {sorted(missing)}")
        for key in ("balanced", "rf_best", "best_known"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"{row['instance']}: non-finite {key}")
    return rows


def main():
    rows = load_main()
    rows = sorted(rows, key=lambda x: x["instance"])
    balanced = np.array([round(float(r["balanced"])) for r in rows])
    rf = np.array([round(float(r["rf_best"])) for r in rows])
    delta = balanced - rf
    wins = delta > TIE_TOL
    draws = np.abs(delta) <= TIE_TOL
    losses = delta < -TIE_TOL
    imp_all = (balanced - rf) / balanced * 100.0
    imp_win = imp_all[wins]
    ranks = np.array([stats.rankdata([r[k] for k in ("random", "spt", "lpt", "balanced", "rf_best")
                                      if k in r], method="average")
                      for r in rows]) if all("random" in r for r in rows) else None

    # Do not call the historical RF-vs-Balanced test significant. The test is
    # retained for transparency and is based on rounded, saved makespans.
    wilcoxon = stats.wilcoxon(rf, balanced, zero_method="pratt",
                              alternative="two-sided", method="auto")
    summary = {
        "source": "results/exp1_main.json",
        "n_instances": len(rows),
        "tie_tolerance": TIE_TOL,
        "win_draw_loss_rf_vs_balanced": [int(wins.sum()), int(draws.sum()), int(losses.sum())],
        "avg_improvement_all_pct": round(float(np.mean(imp_all)), 4),
        "avg_improvement_winning_instances_pct": round(float(np.mean(imp_win)), 4) if len(imp_win) else None,
        "wilcoxon": {
            "rounded_makespans": True,
            "zero_method": "pratt",
            "alternative": "two-sided",
            "statistic": float(wilcoxon.statistic),
            "p_value": float(wilcoxon.pvalue),
        },
        "instances": [
            {"instance": r["instance"], "balanced": int(round(r["balanced"])),
             "rf": int(round(r["rf_best"])), "bks": int(round(r["best_known"])),
             "delta_pct": round(float((r["balanced"] - r["rf_best"]) / r["balanced"] * 100), 4)}
            for r in rows
        ],
        "environment": {"python": sys.version, "platform": platform.platform(),
                         "numpy": np.__version__, "scipy": scipy.__version__},
    }
    out = RESULTS / "canonical_exp1_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
