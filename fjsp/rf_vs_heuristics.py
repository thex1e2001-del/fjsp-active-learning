"""
Head-to-head: RF-guided assignment search vs heuristic rules.
All methods use exact sequencing for final evaluation.
Goal: show RF consistently beats SPT/LPT/Balanced across 10 Brandimarte instances.
"""
import json, os, sys, time, random as _r
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from fjsp_baseline import FJSPInstance, FixedAssignmentSequencingModel
from assignment_generators import AssignmentGenerator
from heuristic_sequencer import HeuristicSequencer
from rf_train import train_global_rf, build_feature_cols, save_rf_models


INSTANCES = [
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
]

SEED = 42
N_TRAIN_ASSIGN = 3000   # assignments to label with heuristic
RF_TREES = 150           # more trees = better
RF_DEPTH = 12            # deeper trees
BEAM_WIDTH = 15
N_CANDIDATES = 5000     # explore more
SEQ_TIMEOUT = 30         # seconds for exact seq verification
TOP_K_VERIFY = 8

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_instance(inst_path: str) -> Dict:
    inst_name = Path(inst_path).stem
    print(f"\n{'='*60}")
    print(f"  {inst_name}")
    print(f"{'='*60}")

    inst = FJSPInstance.from_json_file(inst_path)
    gen = AssignmentGenerator(inst, seed=SEED)
    seq = HeuristicSequencer(inst, seed=SEED)
    feats = build_feature_cols(inst.num_machines)
    ops = inst.all_ops

    # ---- 1. Generate labeled training data (heuristic sequencer, fast) ----
    print(f"  [1/3] Labeling {N_TRAIN_ASSIGN} assignments (heuristic seq)...")
    t0 = time.time()
    train_assigns = {}

    # Heuristic baselines
    for name, a in gen.generate_all_heuristic().items():
        sig = tuple(sorted(a.items()))
        train_assigns[sig] = a

    # Random
    for _ in range(500):
        a = gen.random_assignment()
        sig = tuple(sorted(a.items()))
        if sig not in train_assigns:
            train_assigns[sig] = a

    # Perturbations from balanced (most promising region)
    balanced = gen.balanced_assignment()
    for _ in range(N_TRAIN_ASSIGN):
        perturbed = dict(balanced)
        n_mutate = _r.randint(1, max(2, inst.total_ops // 15))
        for _ in range(n_mutate):
            j, p = _r.choice(ops)
            feasible = inst.feasible_machines.get((j, p), list(range(inst.num_machines)))
            alts = [k for k in feasible if k != perturbed.get((j, p))]
            if alts:
                perturbed[(j, p)] = _r.choice(alts)
        sig = tuple(sorted(perturbed.items()))
        if sig not in train_assigns:
            train_assigns[sig] = perturbed

    n_unique = len(train_assigns)
    print(f"    {n_unique} unique assignments")

    # Heuristic labeling
    schedules = []
    for assign in train_assigns.values():
        h_mk, _ = seq.schedule(assign, "best")
        features = gen.extract_features(assign)
        schedules.append({"makespan": h_mk, "features": features})

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.0f}s ({n_unique/elapsed:.0f}/s)")

    # ---- 2. Train RF ----
    print(f"  [2/3] Training RF ({RF_TREES} trees, depth {RF_DEPTH})...")
    t0 = time.time()
    rf, _ = train_global_rf(schedules, feats, n_estimators=RF_TREES, max_depth=RF_DEPTH)
    rf_models = {"global": (rf, feats)}
    gen.set_rf_models(rf_models, feats)
    r2 = rf.score(
        np.array([[s["features"].get(f, 0) for f in feats] for s in schedules]),
        np.array([s["makespan"] for s in schedules])
    )
    print(f"    R^2={r2:.4f}, time={time.time()-t0:.0f}s")

    # ---- 3. Beam search + top-k exact verification ----
    print(f"  [3/3] Beam search ({N_CANDIDATES} candidates, beam={BEAM_WIDTH})...")
    t0 = time.time()

    # Multi-start beam search
    start_assigns = gen.generate_all_heuristic()
    pool = []
    seen = set()
    for _name, a in start_assigns.items():
        sig = tuple(sorted(a.items()))
        if sig not in seen:
            seen.add(sig)
            pool.append((gen._predict_assignment(a, "global"), a))

    pool.sort(key=lambda x: x[0])
    pool = pool[:BEAM_WIDTH]

    # Beam iterations: for each operation, try ALL feasible machines
    n_explored = 0
    for _ in range(max(3, inst.total_ops // 20)):
        candidates = list(pool)
        for _, current in pool:
            for (j, p) in ops:
                feasible = inst.feasible_machines.get((j, p), list(range(inst.num_machines)))
                for new_k in feasible:
                    if new_k == current.get((j, p)):
                        continue
                    candidate = dict(current)
                    candidate[(j, p)] = new_k
                    sig = tuple(sorted(candidate.items()))
                    if sig not in seen:
                        seen.add(sig)
                        score = gen._predict_assignment(candidate, "global")
                        candidates.append((score, candidate))
                        n_explored += 1
                        if len(candidates) > N_CANDIDATES:
                            break
                if len(candidates) > N_CANDIDATES:
                    break
            if len(candidates) > N_CANDIDATES:
                break
        candidates.sort(key=lambda x: x[0])
        pool = candidates[:BEAM_WIDTH]

    search_time = time.time() - t0
    print(f"    Explored {n_explored} candidates in {search_time:.0f}s")

    # ---- Exact sequencing on top-k + heuristics ----
    print(f"    Verifying top-{TOP_K_VERIFY} and heuristics with exact sequencing...")
    results = {}

    # Heuristic baselines (exact)
    for name, assign in gen.generate_all_heuristic().items():
        exact = FixedAssignmentSequencingModel(inst, assign, time_limit=SEQ_TIMEOUT, mip_gap=0.0)
        r = exact.solve()
        results[name] = r.get("makespan", float("inf"))
        print(f"      {name:10s}: {results[name]:.0f}")

    # RF top-K
    rf_best = float("inf")
    for i, (score, assign) in enumerate(pool[:TOP_K_VERIFY]):
        exact = FixedAssignmentSequencingModel(inst, assign, time_limit=SEQ_TIMEOUT, mip_gap=0.0)
        r = exact.solve()
        mk = r.get("makespan", float("inf"))
        if mk < rf_best:
            rf_best = mk
        if i < 5:
            print(f"      rf_top{i+1:2d}: {mk:.0f}")

    results["rf_best"] = rf_best

    return {
        "instance": inst_name,
        **results,
        "best_known": inst.best_known,
        "n_ops": inst.total_ops,
        "n_jobs": inst.num_jobs,
        "n_machines": inst.num_machines,
        "rf_r2": r2,
        "rf_candidates_explored": n_explored,
    }


def main():
    print("=" * 60)
    print("RF Assignment Search vs Heuristics — Brandimarte Benchmark")
    print("=" * 60)

    all_results = []
    for path in INSTANCES:
        try:
            r = run_instance(path)
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # ---- Summary Table ----
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS: RF vs Heuristics — Brandimarte")
    print(f"{'='*80}")
    print(f"{'Instance':<8s} {'Random':>7s} {'SPT':>7s} {'LPT':>7s} "
          f"{'Balanced':>9s} {'RF_best':>8s} {'Winner':>9s} {'BestK':>7s} {'R^2':>7s}")
    print("-" * 80)

    rf_wins = 0
    bal_wins = 0
    for r in all_results:
        balanced = r.get("balanced", float("inf"))
        rf = r.get("rf_best", float("inf"))
        winner = "RF" if rf < balanced else "Bal"
        if rf < balanced:
            rf_wins += 1
        else:
            bal_wins += 1
        print(f"{r['instance']:<8s} {r.get('random',0):7.0f} {r.get('spt',0):7.0f} "
              f"{r.get('lpt',0):7.0f} {balanced:9.0f} {rf:8.0f} "
              f"{winner:>9s} {r.get('best_known',0):7.0f} {r.get('rf_r2',0):6.3f}")

    print("-" * 80)
    print(f"RF beats Balanced: {rf_wins}/{len(all_results)} instances")

    # Save
    out_path = os.path.join(RESULTS_DIR, "rf_vs_heuristics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
