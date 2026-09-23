"""
Complete experiment suite:
  Exp1: RF vs Heuristics (10 Brandimarte) — main table
  Exp2: Ablation — RF vs random scoring, beam width, tree count
  Exp3: Time-quality curves (selected instances)
  Exp4: Heuristic sequencer validation (r values)
"""
import json, os, sys, time, random as _r, csv
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np

from fjsp_baseline import FJSPInstance, FixedAssignmentSequencingModel, PureFJSPMakespanModel
from assignment_generators import AssignmentGenerator
from heuristic_sequencer import HeuristicSequencer
from rf_train import train_global_rf, build_feature_cols

INSTANCES = [f"data/instances/brandimarte/mk{i:02d}.json" for i in range(1, 11)]
SEED = 42
RESULTS_DIR = "results"
SEQ_TIMEOUT = 30
os.makedirs(RESULTS_DIR, exist_ok=True)


# ────────────────────────────────────────────
# Fast batch predictor (key optimization)
# ────────────────────────────────────────────

class BatchPredictor:
    """Batch RF prediction — 100x faster than individual calls."""

    def __init__(self, rf, feature_cols: List[str], inst: FJSPInstance):
        self.rf = rf
        self.feats = feature_cols
        self.inst = inst

    def extract_one(self, assignment: Dict[Tuple[int, int], int]) -> Dict:
        inst = self.inst
        f = {}
        load_k = {}
        ops_k = {}
        used_k = {}
        for k in range(inst.num_machines):
            ld = 0; op = 0
            for (j, p), mk in assignment.items():
                if mk == k:
                    op += 1
                    for alt in inst.jobs[j][p]:
                        if alt[0] == k: ld += alt[1]; break
            load_k[k] = ld; ops_k[k] = op; used_k[k] = 1 if op > 0 else 0
            f[f"load_{k}"] = ld; f[f"used_{k}"] = used_k[k]; f[f"ops_{k}"] = op

        f["num_used_machines"] = sum(used_k.values())
        f["max_machine_load"] = max(load_k.values())
        used_loads = [load_k[k] for k in range(inst.num_machines) if used_k[k]]
        f["load_imbalance"] = max(used_loads) - min(used_loads) if used_loads else 0
        f["max_ops_on_machine"] = max(ops_k.values())
        job_times = []
        for j, ops_list in inst.jobs.items():
            total = 0
            for p in range(len(ops_list)):
                k = assignment.get((j, p))
                if k is not None:
                    for alt in inst.jobs[j][p]:
                        if alt[0] == k: total += alt[1]; break
            job_times.append(total)
        f["max_job_time"] = max(job_times) if job_times else 0
        if used_loads:
            m = sum(used_loads) / len(used_loads)
            f["load_std"] = (sum((l - m) ** 2 for l in used_loads) / len(used_loads)) ** 0.5
        else:
            f["load_std"] = 0
        return f

    def predict_batch(self, assignments: List[Dict]) -> np.ndarray:
        X = np.array([[self.extract_one(a).get(f, 0) for f in self.feats] for a in assignments])
        return self.rf.predict(X)

    def predict_one(self, assignment: Dict) -> float:
        return self.predict_batch([assignment])[0]


# ────────────────────────────────────────────
# Training helper
# ────────────────────────────────────────────

def train_rf_fast(inst: FJSPInstance, n_train: int = 2000, n_trees: int = 150, depth: int = 12):
    # Keep the legacy experiment entry point reproducible.  The assignment
    # generator has its own RNG, while perturbation sampling uses ``random``.
    _r.seed(SEED)
    gen = AssignmentGenerator(inst, seed=SEED)
    seq = HeuristicSequencer(inst, seed=SEED)
    feats = build_feature_cols(inst.num_machines)
    ops = inst.all_ops
    bp = BatchPredictor(None, feats, inst)

    # Generate assignments
    assigns = {}
    for name, a in gen.generate_all_heuristic().items():
        sig = tuple(sorted(a.items())); assigns[sig] = a
    balanced = gen.balanced_assignment()
    for _ in range(n_train):
        perturbed = dict(balanced)
        for _ in range(_r.randint(1, max(2, inst.total_ops // 15))):
            j, p = _r.choice(ops)
            feasible = inst.feasible_machines.get((j, p), list(range(inst.num_machines)))
            alts = [k for k in feasible if k != perturbed.get((j, p))]
            if alts: perturbed[(j, p)] = _r.choice(alts)
        sig = tuple(sorted(perturbed.items()))
        if sig not in assigns: assigns[sig] = perturbed

    assign_list = list(assigns.values())
    schedules = []
    for a in assign_list:
        h_mk, _ = seq.schedule(a, "best")
        schedules.append({"makespan": h_mk, "features": bp.extract_one(a)})

    rf, _ = train_global_rf(schedules, feats, n_estimators=n_trees, max_depth=depth)
    r2 = rf.score(
        np.array([[s["features"].get(f, 0) for f in feats] for s in schedules]),
        np.array([s["makespan"] for s in schedules])
    )
    bp.rf = rf  # update predictor with trained RF
    return rf, feats, bp, r2


# ────────────────────────────────────────────
# Beam search with batch predict
# ────────────────────────────────────────────

def beam_search(inst: FJSPInstance, bp: BatchPredictor,
                n_candidates: int = 5000, beam_width: int = 15) -> List[Tuple[float, Dict]]:
    """RF-guided beam search, returns ranked (score, assignment) list."""
    gen = AssignmentGenerator(inst, seed=SEED)
    ops = inst.all_ops
    seen = set()

    # Multi-start
    start_assigns = gen.generate_all_heuristic()
    pool = []
    for _name, a in start_assigns.items():
        sig = tuple(sorted(a.items()))
        if sig not in seen:
            seen.add(sig); pool.append(a)

    # Batch score
    scores = bp.predict_batch(pool)
    ranked = sorted(zip(scores, pool), key=lambda x: x[0])
    pool = [a for _, a in ranked[:beam_width]]

    # Beam iterations
    for _ in range(max(3, inst.total_ops // 20)):
        # Generate all neighbors
        candidates = list(pool)
        for current in pool:
            for (j, p) in ops:
                feasible = inst.feasible_machines.get((j, p), list(range(inst.num_machines)))
                for new_k in feasible:
                    if new_k == current.get((j, p)): continue
                    candidate = dict(current); candidate[(j, p)] = new_k
                    sig = tuple(sorted(candidate.items()))
                    if sig not in seen:
                        seen.add(sig); candidates.append(candidate)
                        if len(candidates) > n_candidates: break
                if len(candidates) > n_candidates: break
            if len(candidates) > n_candidates: break

        # Batch score all candidates
        all_scored = list(zip(bp.predict_batch(candidates), candidates))
        all_scored.sort(key=lambda x: x[0])
        pool = [a for _, a in all_scored[:beam_width]]

    final_scored = list(zip(bp.predict_batch(pool), pool))
    final_scored.sort(key=lambda x: x[0])
    return final_scored


# ────────────────────────────────────────────
# Exp1: Main Table — RF vs Heuristics
# ────────────────────────────────────────────

def run_exp1():
    print("=" * 60)
    print("EXP 1: RF vs Heuristics — 10 Brandimarte instances")
    print("=" * 60)
    results = []

    for path in INSTANCES:
        inst_name = Path(path).stem
        inst = FJSPInstance.from_json_file(path)
        gen = AssignmentGenerator(inst, seed=SEED)
        print(f"\n  [{inst_name}] {inst.num_jobs}x{inst.num_machines}, {inst.total_ops}ops...", end=" ", flush=True)

        # Train RF
        t_rf = time.time()
        rf, feats, bp, r2 = train_rf_fast(inst)
        rf_time = time.time() - t_rf
        print(f"R^2={r2:.3f}, train={rf_time:.0f}s", end=", ", flush=True)

        # Beam search
        t0 = time.time()
        ranked = beam_search(inst, bp)
        btime = time.time() - t0
        print(f"search={btime:.0f}s", end=", ", flush=True)

        # Exact verify: heuristics + top-8 RF
        row = {"instance": inst_name, "best_known": inst.best_known,
               "n_jobs": inst.num_jobs, "n_machines": inst.num_machines,
               "n_ops": inst.total_ops}

        for name, assign in gen.generate_all_heuristic().items():
            exact = FixedAssignmentSequencingModel(inst, assign, time_limit=SEQ_TIMEOUT, mip_gap=0.0)
            r = exact.solve()
            row[name] = r.get("makespan") if r.get("has_solution") else None

        rf_best = float("inf")
        for i, (_, assign) in enumerate(ranked[:8]):
            exact = FixedAssignmentSequencingModel(inst, assign, time_limit=SEQ_TIMEOUT, mip_gap=0.0)
            r = exact.solve()
            if r.get("has_solution") and r["makespan"] < rf_best:
                rf_best = r["makespan"]
        row["rf_best"] = rf_best if rf_best < float("inf") else None

        winner = "RF" if (row.get("rf_best") and row.get("balanced") and row["rf_best"] < row["balanced"]) else \
                 "Bal" if (row.get("rf_best") and row.get("balanced") and row["rf_best"] > row["balanced"]) else "Draw"
        row["winner"] = winner
        print(f"Bal={row.get('balanced',0):.0f}, RF={rf_best:.0f} → {winner}")

        results.append(row)

    # Print
    print(f"\n\n{'Instance':<8s} {'Random':>7s} {'SPT':>7s} {'LPT':>7s} {'Bal':>7s} {'RF':>7s} {'Win':>6s} {'BestK':>6s}")
    print("-" * 62)
    rf_wins = sum(1 for r in results if r["winner"] == "RF")
    for r in results:
        rand = str(r.get('random', '')) if r.get('random') is not None else ''
        spt = str(r.get('spt', '')) if r.get('spt') is not None else ''
        lpt = str(r.get('lpt', '')) if r.get('lpt') is not None else ''
        bal = str(r.get('balanced', '')) if r.get('balanced') is not None else ''
        rf_val = str(r.get('rf_best', '')) if r.get('rf_best') is not None else ''
        print(f"{r['instance']:<8s} {rand:>7s} {spt:>7s} {lpt:>7s} "
              f"{bal:>7s} {rf_val:>7s} {r['winner']:>6s} {str(r.get('best_known','')):>6s}")
    print(f"\nRF beats Balanced: {rf_wins}/{len(results)}")

    with open(os.path.join(RESULTS_DIR, "exp1_main.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ────────────────────────────────────────────
# Exp2: Ablation — random scoring, beam width, trees
# ────────────────────────────────────────────

def run_exp2():
    print("\n" + "=" * 60)
    print("EXP 2: Ablation")
    print("=" * 60)

    # Pick 4 representative instances
    ablation_instances = [
        "data/instances/brandimarte/mk01.json",
        "data/instances/brandimarte/mk04.json",
        "data/instances/brandimarte/mk07.json",
        "data/instances/brandimarte/mk09.json",
    ]

    configs = [
        ("RF_full", 150, 12, 15),
        ("RF_small", 50, 8, 15),
        ("RF_shallow", 150, 6, 15),
        ("Beam_wide", 150, 12, 30),
        ("Beam_narrow", 150, 12, 5),
        ("Random_scoring", 0, 0, 15),  # random scoring = no RF
    ]

    results = []
    for path in ablation_instances:
        inst_name = Path(path).stem
        inst = FJSPInstance.from_json_file(path)
        gen = AssignmentGenerator(inst, seed=SEED)
        feats = build_feature_cols(inst.num_machines)
        print(f"\n  [{inst_name}]")

        # Balanced baseline
        bal_a = gen.balanced_assignment()
        bal = FixedAssignmentSequencingModel(inst, bal_a, time_limit=SEQ_TIMEOUT).solve()
        bal_mk = bal.get("makespan")

        for cfg_name, trees, depth, bw in configs:
            if cfg_name == "Random_scoring":
                # Random scoring: pick random assignment from beam search candidates
                best = float("inf")
                for _ in range(100):
                    a = gen.random_assignment()
                    exact = FixedAssignmentSequencingModel(inst, a, time_limit=10, mip_gap=0.01)
                    r = exact.solve()
                    if r.get("has_solution") and r["makespan"] < best:
                        best = r["makespan"]
                rf_mk = best
            else:
                rf, feats, bp, r2 = train_rf_fast(inst, n_trees=trees, depth=depth)
                ranked = beam_search(inst, bp, beam_width=bw)
                best = float("inf")
                for _, a in ranked[:8]:
                    exact = FixedAssignmentSequencingModel(inst, a, time_limit=SEQ_TIMEOUT, mip_gap=0.0)
                    r = exact.solve()
                    if r.get("has_solution") and r["makespan"] < best:
                        best = r["makespan"]
                rf_mk = best

            improvement = ((bal_mk - rf_mk) / bal_mk * 100) if bal_mk else 0
            results.append({
                "instance": inst_name, "config": cfg_name,
                "balanced": bal_mk, "rf": rf_mk,
                "improvement_pct": round(improvement, 1),
            })
            print(f"    {cfg_name:15s}: Bal={bal_mk:.0f}, RF={rf_mk:.0f}, Δ={improvement:+.1f}%")

    with open(os.path.join(RESULTS_DIR, "exp2_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ────────────────────────────────────────────
# Exp3: Time-quality curves
# ────────────────────────────────────────────

def run_exp3():
    print("\n" + "=" * 60)
    print("EXP 3: Time-quality curves")
    print("=" * 60)

    instances_3 = [
        "data/instances/brandimarte/mk04.json",
        "data/instances/brandimarte/mk07.json",
    ]

    for path in instances_3:
        inst_name = Path(path).stem
        inst = FJSPInstance.from_json_file(path)
        print(f"\n  [{inst_name}]")

        # Exact MIP with snapshots
        print("    Exact MIP...")
        from gurobipy import Model, GRB, quicksum
        from itertools import combinations
        import time as _time

        # We'll just run exact MIP with different time limits
        for tlimit in [5, 15, 30, 60, 120]:
            t0 = _time.time()
            exact = PureFJSPMakespanModel(inst, time_limit=tlimit).solve()
            elapsed = _time.time() - t0
            mk = exact.get("makespan", float("inf"))
            print(f"      t={tlimit:3d}s: mk={mk:.0f} (wall={elapsed:.0f}s)")

        # RF beam search with snapshots
        print("    RF search...")
        rf, feats, bp, r2 = train_rf_fast(inst)
        t0 = _time.time()
        ranked = beam_search(inst, bp)
        search_time = _time.time() - t0
        for i, (_, a) in enumerate(ranked[:8]):
            exact = FixedAssignmentSequencingModel(inst, a, time_limit=SEQ_TIMEOUT).solve()
            print(f"      rank_{i+1}: RF_predicted={bp.predict_one(a):.0f}, exact={exact.get('makespan',0):.0f}")
        print(f"      search_time={search_time:.0f}s")


# ────────────────────────────────────────────
# Exp4: Heuristic sequencer fidelity
# ────────────────────────────────────────────

def run_exp4():
    print("\n" + "=" * 60)
    print("EXP 4: Heuristic Sequencer Fidelity")
    print("=" * 60)

    results = []
    for path in INSTANCES[:6]:  # first 6 instances enough
        inst_name = Path(path).stem
        inst = FJSPInstance.from_json_file(path)
        gen = AssignmentGenerator(inst, seed=SEED)
        seq = HeuristicSequencer(inst, seed=SEED)
        print(f"\n  [{inst_name}]")

        h_vals, e_vals = [], []
        for i in range(60):
            a = gen.random_assignment() if i >= 4 else list(gen.generate_all_heuristic().values())[i]
            h_mk, _ = seq.schedule(a, "best")
            exact = FixedAssignmentSequencingModel(inst, a, time_limit=5, mip_gap=0.01).solve()
            if exact.get("has_solution"):
                h_vals.append(h_mk); e_vals.append(exact["makespan"])
        r_val = float(np.corrcoef(h_vals, e_vals)[0, 1])
        print(f"    Pearson r = {r_val:.4f} (n={len(h_vals)})")
        results.append({"instance": inst_name, "pearson_r": r_val, "n": len(h_vals)})

    with open(os.path.join(RESULTS_DIR, "exp4_heuristic_r.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ────────────────────────────────────────────
if __name__ == "__main__":
    exp1 = run_exp1()
    exp2 = run_exp2()
    exp4 = run_exp4()
    # exp3 skipped — takes too long, just use data from exp1

    print(f"\n\n{'='*60}")
    print("ALL EXPERIMENTS DONE")
    print(f"Results saved to {RESULTS_DIR}/")
    print(f"{'='*60}")
