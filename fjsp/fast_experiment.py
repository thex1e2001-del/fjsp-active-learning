"""
L1 + L3 experiments using FAST heuristic sequencing for assignment evaluation.
Evaluates 500-2000 assignments in seconds instead of minutes.
"""
import json, os, time, random as _r
from typing import Dict, List, Tuple

import numpy as np

from fjsp_baseline import FJSPInstance, FixedAssignmentSequencingModel
from assignment_generators import AssignmentGenerator
from heuristic_sequencer import HeuristicSequencer
from rf_train import train_global_rf, build_feature_cols, save_rf_models


def run_fast_experiment(instance: FJSPInstance, seed: int = 42):
    """
    Fast experiment:
    1. Generate 2000 diverse assignments
    2. Evaluate all with heuristic sequencer (fast)
    3. Train RF on heuristic labels
    4. Beam search with RF → top-20
    5. Verify top-20 with exact sequencing
    6. Compare: balanced vs RF vs exact MIP
    """
    gen = AssignmentGenerator(instance, seed=seed)
    seq = HeuristicSequencer(instance, seed=seed)
    feats = build_feature_cols(instance.num_machines)
    ops = instance.all_ops

    print(f"\n{'='*60}")
    print(f"FAST Experiment: {instance.name} ({instance.num_jobs}x{instance.num_machines})")
    print(f"Best known: {instance.best_known}")
    print(f"{'='*60}")

    # ---- Step 1: Generate 2000 diverse assignments ----
    print("\n[1/4] Generating 2000 diverse assignments...")
    t0 = time.time()
    train_assigns = {}
    # Heuristic-based
    for name, a in gen.generate_all_heuristic().items():
        sig = tuple(sorted(a.items()))
        train_assigns[sig] = a
    # Random
    for _ in range(500):
        a = gen.random_assignment()
        sig = tuple(sorted(a.items()))
        if sig not in train_assigns:
            train_assigns[sig] = a
    # Perturbations from balanced
    balanced = gen.balanced_assignment()
    for _ in range(2000):
        perturbed = dict(balanced)
        # Mutate 1-3 operations
        n_mutate = _r.randint(1, 3)
        for _ in range(n_mutate):
            j, p = _r.choice(ops)
            feasible = instance.feasible_machines.get((j, p), list(range(instance.num_machines)))
            alts = [k for k in feasible if k != perturbed.get((j, p))]
            if alts:
                perturbed[(j, p)] = _r.choice(alts)
        sig = tuple(sorted(perturbed.items()))
        if sig not in train_assigns:
            train_assigns[sig] = perturbed
    print(f"  {len(train_assigns)} unique assignments generated in {time.time()-t0:.1f}s")

    # ---- Step 2: Heuristic evaluation ----
    print("\n[2/4] Evaluating with heuristic sequencer...")
    t0 = time.time()
    schedules = []
    for assign in train_assigns.values():
        h_mk, h_rule = seq.schedule(assign, "best")
        features = gen.extract_features(assign)
        schedules.append({
            "makespan": h_mk,
            "features": features,
            "assignment": {f"({j},{p})": k for (j, p), k in assign.items()},
        })
    elapsed = time.time() - t0
    print(f"  {len(schedules)} evaluated in {elapsed:.1f}s ({len(schedules)/elapsed:.0f}/s)")

    best_heuristic = min(s["makespan"] for s in schedules)
    print(f"  Best heuristic makespan: {best_heuristic:.1f}")

    # ---- Step 3: Train RF ----
    print("\n[3/4] Training RF on heuristic labels...")
    rf, _ = train_global_rf(schedules, feats, n_estimators=100, max_depth=12)
    rf_models = {"global": (rf, feats)}
    gen.set_rf_models(rf_models, feats)

    # ---- Step 4: Beam search + top-k exact verification ----
    print("\n[4/4] Beam search with RF → exact verification...")

    # Beam search over assignment space (fast with RF, <1s)
    t0 = time.time()
    start_assigns = gen.generate_all_heuristic()
    start_assigns["rf_best"] = gen.rf_search_assignment(rf_key="global", n_iter=500, n_neighbors=3)

    pool = []
    seen = set()
    for _name, a in start_assigns.items():
        sig = tuple(sorted(a.items()))
        if sig not in seen:
            seen.add(sig)
            pool.append((gen._predict_assignment(a, "global"), a))

    pool.sort(key=lambda x: x[0])
    pool = pool[:15]

    # Beam iterations
    for _ in range(5):
        candidates = list(pool)
        for _, current in pool:
            for (j, p) in ops:
                feasible = instance.feasible_machines.get((j, p), list(range(instance.num_machines)))
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
                        if len(candidates) > 3000:
                            break
                if len(candidates) > 3000:
                    break
            if len(candidates) > 3000:
                break
        candidates.sort(key=lambda x: x[0])
        pool = candidates[:20]

    beam_time = time.time() - t0
    print(f"  Beam search: {len(seen)} candidates explored in {beam_time:.1f}s")

    # --- Exact sequencing for top-k ---
    print(f"\n  Verifying top-8 with exact sequencing...")
    results = {}
    for i, (score, assign) in enumerate(pool[:8]):
        exact = FixedAssignmentSequencingModel(instance, assign, time_limit=30, mip_gap=0.0)
        r = exact.solve()
        mk = r.get("makespan", float("inf"))
        print(f"    #{i+1}: RF={score:.1f}, exact_makespan={mk:.1f}, "
              f"n_machines={len(set(assign.values()))}")
        results[f"rf_top{i+1}"] = mk

    # --- Also evaluate heuristics with exact ----
    print(f"\n  Heuristic baselines (exact):")
    for name, assign in gen.generate_all_heuristic().items():
        exact = FixedAssignmentSequencingModel(instance, assign, time_limit=30, mip_gap=0.0)
        r = exact.solve()
        results[name] = r.get("makespan", float("inf"))
        print(f"    {name}: makespan={results[name]:.1f}")

    # --- Exact MIP baseline ---
    from fjsp_baseline import PureFJSPMakespanModel
    print(f"\n  Exact FJSP MIP baseline...")
    exact_mip = PureFJSPMakespanModel(instance, time_limit=60).solve()
    results["exact_baseline"] = exact_mip.get("makespan", float("inf"))
    print(f"    exact: makespan={results['exact_baseline']:.1f}, "
          f"gap={exact_mip.get('gap', 0):.4f}, time={exact_mip['wall_time']:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    best_rf = min([v for k, v in results.items() if "rf_top" in k])
    print(f"  Best heuristic (balanced):  {results.get('balanced', 'N/A')}")
    print(f"  Best RF search:             {best_rf}")
    print(f"  Exact MIP baseline:         {results['exact_baseline']}")
    print(f"  Best known:                 {instance.best_known}")

    return results


if __name__ == "__main__":
    inst = FJSPInstance.from_json_file("data/instances/brandimarte/mk01.json")
    run_fast_experiment(inst)
