"""Draw the original two-seed comparison at a readable single-column size."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAPER = Path(__file__).resolve().parent
DATA = PAPER.parent / "results"
INSTANCES = [f"mk{i:02d}" for i in range(1, 11)]
# BKS values from Table III of the development manuscript.
BKS = [40, 26, 204, 60, 172, 58, 139, 523, 307, 198]
METHODS = ["AL-RF", "RF-H", "LB", "HeurSeq", "Random"]
STYLES = [
    ("#12639a", "-", "o"),
    ("#c35d15", "--", "s"),
    ("#16896f", "-.", "^"),
    ("#aa467b", ":", "D"),
    ("#555555", "--", "x"),
]


def main():
    runs = []
    for name in (
        "active_screening_dev.json",
        "active_screening_dev_rest.json",
        "active_screening_dev_seed43_rest.json",
    ):
        runs.extend(json.loads((DATA / name).read_text(encoding="utf-8")))
    by_key = {(run["instance"], run["seed"]): run for run in runs}
    assert len(runs) == len(by_key) == 20

    values = defaultdict(list)
    for instance in INSTANCES:
        for seed in (42, 43):
            for row in by_key[instance, seed]["rows"]:
                values[instance, row["scorer"]].append(row["best"])
    fig, ax = plt.subplots(figsize=(3.48, 2.20), dpi=350)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.18, top=0.75)
    x = np.arange(len(INSTANCES))
    for method, (color, linestyle, marker) in zip(METHODS, STYLES):
        y = [np.mean(values[name, method]) / bks * 100 for name, bks in zip(INSTANCES, BKS)]
        ax.plot(x, y, color=color, linestyle=linestyle, marker=marker,
                linewidth=1.25, markersize=3.5, label=method)
    ax.set_xlim(-0.35, 9.55)
    ax.set_ylim(98, 205)
    ax.set_yticks([100, 125, 150, 175, 200])
    ax.set_xticks(x, INSTANCES, fontsize=7)
    ax.set_ylabel("Makespan / BKS (%)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7, length=2)
    ax.tick_params(axis="x", length=2)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 3, 1, 4, 2]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, fontsize=7, columnspacing=0.75, handlelength=1.6)
    fig.savefig(PAPER / "active_method_comparison.png", dpi=350)
    plt.close(fig)


if __name__ == "__main__":
    main()
