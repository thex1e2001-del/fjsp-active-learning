"""Summarize calibrated_screening.py results into auditable tables.

Usage:
  python summarize_calibrated.py results/calibrated_screening_formal_small.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def finite(values):
    return [float(v) for v in values if v is not None]


def stats(values):
    xs = finite(values)
    if not xs:
        return {"n": 0, "mean": None, "std": None, "best": None}
    return {
        "n": len(xs),
        "mean": mean(xs),
        "std": pstdev(xs) if len(xs) > 1 else 0.0,
        "best": min(xs),
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        for item in row["scorers"]:
            groups[(item["instance"], item["scorer"])].append(item)
    out = []
    for (instance, scorer), items in sorted(groups.items()):
        makespan = stats([x["best_selected_makespan"] for x in items])
        out.append({
            "instance": instance,
            "scorer": scorer,
            "seeds": makespan["n"],
            "attempted_seeds": len(items),
            "mean_best_makespan": makespan["mean"],
            "std_best_makespan": makespan["std"],
            "best_makespan": makespan["best"],
            "mean_feasible_ratio": mean(x["feasible_ratio"] for x in items),
            "mean_low_gap_ratio": mean(x["low_gap_ratio"] for x in items),
            "mean_gap_feasible": mean(x["mean_gap_feasible"] for x in items
                                       if x["mean_gap_feasible"] is not None)
            if any(x["mean_gap_feasible"] is not None for x in items) else None,
        })
    # Balanced is an independent reference assignment, so include it in the
    # same table while keeping its provenance explicit in the JSON source.
    balanced_groups = defaultdict(list)
    for row in rows:
        balanced_groups[row["instance"]].append(row["balanced"])
    for instance, items in sorted(balanced_groups.items()):
        makespan = stats([x.get("makespan") for x in items if x.get("has_solution")])
        feasible = [x for x in items if x.get("has_solution")]
        gaps = [x.get("gap") for x in feasible if x.get("gap") is not None]
        out.append({
            "instance": instance,
            "scorer": "Balanced-ref",
            "seeds": makespan["n"],
            "attempted_seeds": len(items),
            "mean_best_makespan": makespan["mean"],
            "std_best_makespan": makespan["std"],
            "best_makespan": makespan["best"],
            "mean_feasible_ratio": len(feasible) / len(items) if items else 0.0,
            "mean_low_gap_ratio": sum(g <= 0.05 for g in gaps) / len(items) if items else 0.0,
            "mean_gap_feasible": mean(gaps) if gaps else None,
        })
    out.sort(key=lambda x: (x["instance"], x["scorer"]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()
    source = Path(args.input)
    rows = json.loads(source.read_text(encoding="utf-8"))
    summary = summarize(rows)
    prefix = Path(args.output_prefix) if args.output_prefix else source.with_suffix("")
    csv_path = prefix.with_suffix(".summary.csv")
    md_path = prefix.with_suffix(".summary.md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(summary[0]) if summary else ["instance", "scorer"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    lines = ["# Calibrated screening summary", "", f"Source: `{source.name}`", "",
             "Means exclude runs without a feasible selected incumbent. Seeds shows successful/attempted runs; conditional means with different success counts are not directly comparable.", "",
             "| Instance | Scorer | Seeds (success/attempted) | Mean best | Std | Best | Feasible | Low-gap | Mean gap |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in summary:
        def fmt(v): return "-" if v is None else f"{v:.3f}"
        lines.append("| {instance} | {scorer} | {seeds}/{attempted} | {mean_best} | {std} | "
                     "{best} | {feasible:.3f} | {low_gap:.3f} | {gap} |".format(
                         instance=x["instance"], scorer=x["scorer"], seeds=x["seeds"],
                         attempted=x["attempted_seeds"],
                         mean_best=fmt(x["mean_best_makespan"]),
                         std=fmt(x["std_best_makespan"]), best=fmt(x["best_makespan"]),
                         feasible=x["mean_feasible_ratio"], low_gap=x["mean_low_gap_ratio"],
                         gap=fmt(x["mean_gap_feasible"])))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
