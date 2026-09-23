"""
Phase 2: Fair baselines — equal candidate budget + equal wall-clock.
Runs on 5 representative instances: mk01, mk04, mk06, mk07, mk09.
"""
import json, os, sys, time, random as _r
from pathlib import Path
from typing import Dict, List
import numpy as np

from fjsp_baseline import FJSPInstance, FixedAssignmentSequencingModel, PureFJSPMakespanModel
from assignment_generators import AssignmentGenerator
from heuristic_sequencer import HeuristicSequencer
from experiments_final import train_rf_fast, beam_search, BatchPredictor

SEED = 42
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

INSTANCES = [
    "data/instances/brandimarte/mk01.json",
    "data/instances/brandimarte/mk04.json",
    "data/instances/brandimarte/mk06.json",
    "data/instances/brandimarte/mk07.json",
    "data/instances/brandimarte/mk09.json",
]
SEQ_TIMEOUT = 30
TOP_K = 8


def run_fair_baselines():
    print("=" * 60)
    print("Phase 2: Fair Baselines")
    print("=" * 60)

    all_results = []
    for path in INSTANCES:
        inst_name = Path(path).stem
        inst = FJSPInstance.from_json_file(path)
        gen = AssignmentGenerator(inst, seed=SEED)
        seq = HeuristicSequencer(inst, seed=SEED)
        ops = inst.all_ops
        print(f"\n{'='*60}\n  [{inst_name}] {inst.num_jobs}x{inst.num_machines}, {inst.total_ops}ops\n{'='*60}")

        row = {"instance": inst_name, "best_known": inst.best_known}

        # ---- 1. Balanced + single exact seq ----
        bal_a = gen.balanced_assignment()
        bal = FixedAssignmentSequencingModel(inst, bal_a, time_limit=SEQ_TIMEOUT).solve()
        row["balanced"] = bal.get("makespan")
        print(f"  Balanced: {row['balanced']:.0f}")

        # ---- 2. Random multi-start: same budget as RF beam ----
        # Generate as many random assignments as RF beam explores (~3000-5000)
        # Then verify top-K
        n_random = min(3000, inst.total_ops * 30)
        best_rand = float("inf")
        for _ in range(n_random):
            a = gen.random_assignment()
            h_mk, _ = seq.schedule(a, "best")
            if h_mk < best_rand:
                best_rand = h_mk
                # Take top-K best by heuristic score
        # Actually: generate all, score by heuristic, verify top-K
        rand_pool = []
        for _ in range(n_random):
            a = gen.random_assignment()
            h_mk, _ = seq.schedule(a, "best")
            rand_pool.append((h_mk, a))
        rand_pool.sort(key=lambda x: x[0])
        best_rand_ms = float("inf")
        for _, a in rand_pool[:TOP_K]:
            exact = FixedAssignmentSequencingModel(inst, a, time_limit=SEQ_TIMEOUT, mip_gap=0.0).solve()
            if exact.get("has_solution") and exact["makespan"] < best_rand_ms:
                best_rand_ms = exact["makespan"]
        row["random_ms"] = best_rand_ms
        print(f"  Random-MS ({n_random} gen, K={TOP_K}): {best_rand_ms:.0f}")

        # ---- 3. MaxLoad beam search ----
        # Score = -max_load (lower load = better)
        def maxload_score(assignment):
            loads = {}
            for (j,p), k in assignment.items():
                for alt in inst.jobs[j][p]:
                    if alt[0] == k: loads[k] = loads.get(k, 0) + alt[1]; break
            return -max(loads.values())  # negative so higher = better (beam search maximizes)

        pool = []
        seen = set()
        for name, a in gen.generate_all_heuristic().items():
            sig = tuple(sorted(a.items()))
            if sig not in seen: seen.add(sig); pool.append(a)
        pool.sort(key=maxload_score, reverse=True)
        pool = pool[:15]

        for _ in range(5):
            candidates = list(pool)
            for current in pool:
                for (j,p) in ops:
                    feasible = inst.feasible_machines.get((j,p), list(range(inst.num_machines)))
                    for new_k in feasible:
                        if new_k == current.get((j,p)): continue
                        c = dict(current); c[(j,p)] = new_k
                        sig = tuple(sorted(c.items()))
                        if sig not in seen and len(candidates) < 5000:
                            seen.add(sig); candidates.append(c)
            candidates.sort(key=maxload_score, reverse=True)
            pool = candidates[:15]

        best_maxload = float("inf")
        for _, a in [(maxload_score(a), a) for a in pool[:TOP_K]]:
            exact = FixedAssignmentSequencingModel(inst, a, time_limit=SEQ_TIMEOUT).solve()
            if exact.get("has_solution") and exact["makespan"] < best_maxload:
                best_maxload = exact["makespan"]
        row["maxload_beam"] = best_maxload
        print(f"  MaxLoad beam: {best_maxload:.0f}")

        # ---- 4. Heuristic-sequencer beam search (no RF) ----
        def heur_score(assignment):
            mk, _ = seq.schedule(assignment, "best")
            return -mk  # negative so higher = better

        pool = []
        seen = set()
        for name, a in gen.generate_all_heuristic().items():
            sig = tuple(sorted(a.items()))
            if sig not in seen: seen.add(sig); pool.append(a)
        pool.sort(key=heur_score, reverse=True)
        pool = pool[:15]

        for _ in range(5):
            candidates = list(pool)
            for current in pool:
                for (j,p) in ops:
                    feasible = inst.feasible_machines.get((j,p), list(range(inst.num_machines)))
                    for new_k in feasible:
                        if new_k == current.get((j,p)): continue
                        c = dict(current); c[(j,p)] = new_k
                        sig = tuple(sorted(c.items()))
                        if sig not in seen and len(candidates) < 5000:
                            seen.add(sig); candidates.append(c)
            candidates.sort(key=heur_score, reverse=True)
            pool = candidates[:15]

        best_heurseq = float("inf")
        for _, a in [(heur_score(a), a) for a in pool[:TOP_K]]:
            exact = FixedAssignmentSequencingModel(inst, a, time_limit=SEQ_TIMEOUT).solve()
            if exact.get("has_solution") and exact["makespan"] < best_heurseq:
                best_heurseq = exact["makespan"]
        row["heurseq_beam"] = best_heurseq
        print(f"  HeurSeq beam: {best_heurseq:.0f}")

        # ---- 5. Balanced + local search (same total time) ----
        # Estimate total time for RF: ~training+search time from previous runs
        # Use simple local search: perturb balanced, evaluate with heuristic, verify best
        t_budget = 60  # seconds
        t0 = time.time()
        current = dict(bal_a)
        current_score = heur_score(current)
        best_ls = float("inf")
        while time.time() - t0 < t_budget:
            improved = False
            for _ in range(50):
                j, p = _r.choice(ops)
                feasible = inst.feasible_machines.get((j,p), list(range(inst.num_machines)))
                alts = [k for k in feasible if k != current.get((j,p))]
                if not alts: continue
                c = dict(current); c[(j,p)] = _r.choice(alts)
                cs = heur_score(c)
                if cs > current_score:
                    current = c; current_score = cs; improved = True
            if not improved: break
        # Verify final best
        exact = FixedAssignmentSequencingModel(inst, current, time_limit=SEQ_TIMEOUT).solve()
        row["balanced_ls"] = exact.get("makespan") if exact.get("has_solution") else None
        print(f"  Balanced+LS ({time.time()-t0:.0f}s): {row['balanced_ls']:.0f}" if row['balanced_ls'] else f"  Balanced+LS: N/A")

        # ---- 6. RF beam (reference, from known results) ----
        # Use known result from exp1
        rf, feats, bp, _ = train_rf_fast(inst)
        ranked = beam_search(inst, bp)
        best_rf = float("inf")
        for _, a in ranked[:TOP_K]:
            exact = FixedAssignmentSequencingModel(inst, a, time_limit=SEQ_TIMEOUT, mip_gap=0.0).solve()
            if exact.get("has_solution") and exact["makespan"] < best_rf:
                best_rf = exact["makespan"]
        row["rf_beam"] = best_rf
        print(f"  RF beam: {best_rf:.0f}")

        all_results.append(row)

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("FAIR BASELINES COMPARISON")
    print(f"{'='*80}")
    methods = ["balanced", "random_ms", "maxload_beam", "heurseq_beam", "balanced_ls", "rf_beam"]
    header = f"{'Instance':<8s}"
    for m in methods:
        header += f" {m:>12s}"
    print(header)
    print("-" * (8 + 13*len(methods)))
    for r in all_results:
        line = f"{r['instance']:<8s}"
        best_val = min(v for v in [r.get(m) for m in methods] if v is not None and v < 1e8)
        for m in methods:
            v = r.get(m)
            if v is None:
                line += f" {'--':>12s}"
            else:
                marker = "*" if v == best_val else ""
                line += f" {v:>11.0f}{marker}"
        print(line)

    with open(os.path.join(RESULTS_DIR, "fair_baselines.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to fair_baselines.json")
    return all_results


if __name__ == "__main__":
    run_fair_baselines()
