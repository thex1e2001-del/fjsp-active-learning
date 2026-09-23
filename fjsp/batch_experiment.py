"""
Batch experiment: run RF-embedded MIP on all 20 instances × 3 λ × 3 runs.
Incrementally saves results to results/summary.csv.
Also runs baselines: exact MIP, balanced+seq.
"""
import csv, json, os, sys, time, traceback
from pathlib import Path
from typing import List, Dict

import numpy as np

from fjsp_baseline import FJSPInstance, PureFJSPMakespanModel, FixedAssignmentSequencingModel
from rf_embedded_mip import train_small_rf_for_mip, RFEmbeddedMIP
from assignment_generators import AssignmentGenerator
from heuristic_sequencer import HeuristicSequencer

# ============================================================
# Config
# ============================================================

INSTANCES = [
    # Brandimarte
    "data/instances/brandimarte/mk01.json",
    "data/instances/brandimarte/mk02.json",
    "data/instances/brandimarte/mk03.json",
    "data/instances/brandimarte/mk04.json",
    "data/instances/brandimarte/mk05.json",
    "data/instances/brandimarte/mk06.json",
    "data/instances/brandimarte/mk07.json",
    "data/instances/brandimarte/mk08.json",
    "data/instances/brandimarte/mk09.json",
    "data/instances/brandimarte/mk10.json",
    # Hurink vdata (10 selected)
    "data/instances/hurink/vdata/la16.json",
    "data/instances/hurink/vdata/la21.json",
    "data/instances/hurink/vdata/la26.json",
    "data/instances/hurink/vdata/la31.json",
    "data/instances/hurink/vdata/la36.json",
    "data/instances/hurink/vdata/la40.json",
    "data/instances/hurink/vdata/mt10.json",
    "data/instances/hurink/vdata/mt20.json",
    "data/instances/hurink/vdata/abz5.json",
    "data/instances/hurink/vdata/abz6.json",
]

LAMBDAS = [0.0, 0.001, 0.01]
N_RUNS = 3
TIME_LIMIT = 120  # seconds per solve
RF_TREES = 10
RF_DEPTH = 5

RESULTS_DIR = "results"
MODEL_DIR = "data/rf_models/batch"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def make_summary_csv(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "instance", "n_jobs", "n_machines", "n_ops", "best_known",
                "lambda", "run",
                "makespan", "gap_pct", "wall_time", "num_vars", "num_constrs",
                "rf_surrogate", "status", "error"
            ])


def append_result(path: str, row: Dict):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            row.get("instance", ""),
            row.get("n_jobs", ""),
            row.get("n_machines", ""),
            row.get("n_ops", ""),
            row.get("best_known", ""),
            row.get("lambda", ""),
            row.get("run", ""),
            row.get("makespan", ""),
            row.get("gap_pct", ""),
            row.get("wall_time", ""),
            row.get("num_vars", ""),
            row.get("num_constrs", ""),
            row.get("rf_surrogate", ""),
            row.get("status", ""),
            row.get("error", ""),
        ])


def run_one(inst: FJSPInstance, tree_rules, feature_names,
            lam: float, run_id: int) -> Dict:
    """Run RF-embedded MIP once, return result dict."""
    t0 = time.time()
    try:
        model = RFEmbeddedMIP(inst, tree_rules, feature_names,
                              lambda_=lam, time_limit=TIME_LIMIT, mip_gap=0.0)
        r = model.solve()
        elapsed = time.time() - t0

        mk = r.get("makespan")
        best = inst.best_known
        gap = ((mk - best) / best * 100) if (mk and best) else None

        return {
            "makespan": mk,
            "gap_pct": round(gap, 2) if gap else None,
            "wall_time": round(elapsed, 1),
            "num_vars": r.get("num_vars", 0),
            "num_constrs": r.get("num_constrs", 0),
            "rf_surrogate": round(r.get("rf_surrogate", 0), 1) if r.get("rf_surrogate") else None,
            "status": r.get("status", -1),
            "error": None,
        }
    except Exception as e:
        return {
            "makespan": None, "gap_pct": None,
            "wall_time": round(time.time() - t0, 1),
            "num_vars": 0, "num_constrs": 0,
            "rf_surrogate": None, "status": -1,
            "error": str(e)[:200],
        }


# ============================================================
# Main
# ============================================================

def main():
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    make_summary_csv(summary_path)
    baseline_path = os.path.join(RESULTS_DIR, "baselines.csv")
    if not os.path.exists(baseline_path):
        with open(baseline_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["instance", "exact_makespan", "exact_time", "exact_gap",
                        "balanced_makespan", "balanced_time",
                        "spt_makespan", "spt_time"])

    total_solves = len(INSTANCES) * len(LAMBDAS) * N_RUNS
    idx = 0

    for inst_path in INSTANCES:
        inst_name = Path(inst_path).stem
        print(f"\n{'='*60}")
        print(f"INSTANCE: {inst_name}")
        print(f"{'='*60}")

        # Load instance
        inst = FJSPInstance.from_json_file(inst_path)
        print(f"  {inst.num_jobs}x{inst.num_machines}, {inst.total_ops} ops, "
              f"best_known={inst.best_known}")

        # ---- Train RF (once per instance) ----
        print(f"  Training RF ({RF_TREES} trees, depth {RF_DEPTH})...")
        t0_rf = time.time()
        try:
            rf, tree_rules, feats = train_small_rf_for_mip(
                inst, n_trees=RF_TREES, max_depth=RF_DEPTH,
                n_samples=min(2000, inst.total_ops * 10), seed=42
            )
            n_leaves = sum(len(rules) for rules in tree_rules)
            print(f"  RF done in {time.time()-t0_rf:.0f}s: {n_leaves} leaves")
        except Exception as e:
            print(f"  RF ERROR: {e}")
            traceback.print_exc()
            continue

        # ---- Baselines (once per instance) ----
        print(f"  Running baselines...")
        try:
            gen = AssignmentGenerator(inst, seed=42)
            baselines = {}

            # Exact MIP
            t0 = time.time()
            exact = PureFJSPMakespanModel(inst, time_limit=TIME_LIMIT).solve()
            baselines["exact_makespan"] = exact.get("makespan")
            baselines["exact_time"] = round(time.time() - t0, 1)
            baselines["exact_gap"] = exact.get("gap")

            # Balanced + seq
            t0 = time.time()
            bal_assign = gen.balanced_assignment()
            bal = FixedAssignmentSequencingModel(inst, bal_assign, time_limit=30).solve()
            baselines["balanced_makespan"] = bal.get("makespan")
            baselines["balanced_time"] = round(time.time() - t0, 1)

            # SPT + seq
            t0 = time.time()
            spt_assign = gen.spt_assignment()
            spt = FixedAssignmentSequencingModel(inst, spt_assign, time_limit=30).solve()
            baselines["spt_makespan"] = spt.get("makespan")
            baselines["spt_time"] = round(time.time() - t0, 1)

            with open(baseline_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([inst_name] + [baselines.get(k) for k in [
                    "exact_makespan", "exact_time", "exact_gap",
                    "balanced_makespan", "balanced_time",
                    "spt_makespan", "spt_time"
                ]])
            print(f"  Exact={baselines['exact_makespan']}, "
                  f"Balanced={baselines['balanced_makespan']}, "
                  f"SPT={baselines['spt_makespan']}")
        except Exception as e:
            print(f"  Baseline ERROR: {e}")

        # ---- RF-embedded MIP λ sweep × runs ----
        for lam in LAMBDAS:
            for run_id in range(1, N_RUNS + 1):
                idx += 1
                label = f"λ={lam}" if lam > 0 else "λ=0(exact)"
                print(f"  [{idx}/{total_solves}] {label} run {run_id}...", end=" ", flush=True)

                result = run_one(inst, tree_rules, feats, lam, run_id)

                row = {
                    "instance": inst_name,
                    "n_jobs": inst.num_jobs,
                    "n_machines": inst.num_machines,
                    "n_ops": inst.total_ops,
                    "best_known": inst.best_known,
                    "lambda": lam,
                    "run": run_id,
                    **result,
                }
                append_result(summary_path, row)

                if result["error"]:
                    print(f"ERROR: {result['error'][:80]}")
                else:
                    print(f"mk={result['makespan']}, "
                          f"gap={result['gap_pct']}%, "
                          f"t={result['wall_time']}s")

    print(f"\n{'='*60}")
    print(f"DONE: {idx} solves completed")
    print(f"Results: {summary_path}")
    print(f"Baselines: {baseline_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
