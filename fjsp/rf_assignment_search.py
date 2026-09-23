"""
L2: RF-driven assignment space search.

Given a trained RF, search the assignment space by:
1. Generate candidate assignments (random perturbations)
2. Score with RF (fast)
3. Select top-k
4. Solve exact sequencing for top-k
5. Return best makespan found

Compares with exact MIP at the same time budget.
"""
import time
from typing import Dict, List, Tuple, Optional

import numpy as np

from fjsp_baseline import FJSPInstance, PureFJSPMakespanModel, FixedAssignmentSequencingModel
from assignment_generators import AssignmentGenerator


def rf_guided_search(
    instance: FJSPInstance,
    rf_models: Dict,
    feature_cols: List[str],
    time_budget: float = 60.0,
    n_candidates: int = 500,
    top_k: int = 10,
    seed: int = 42,
) -> Dict:
    """
    RF-guided assignment search with time budget.

    1. Generate random candidate assignments
    2. Score with RF
    3. Filter top-k
    4. Solve exact sequencing for each
    5. Return best result
    """
    t0 = time.time()
    gen = AssignmentGenerator(instance, seed=seed)
    gen.set_rf_models(rf_models, feature_cols)

    # Step 1-2: Generate and score candidates
    candidates = []
    for _ in range(n_candidates):
        assign = gen.random_assignment()
        score = gen._predict_assignment(assign, "global")
        candidates.append((score, assign))

    # Step 3: Filter top-k
    candidates.sort(key=lambda x: x[0])
    top_assignments = [a for _, a in candidates[:top_k]]

    # Also include heuristic assignments
    for name, assign in gen.generate_all_heuristic().items():
        top_assignments.append(assign)

    # Deduplicate
    seen = set()
    unique = []
    for a in top_assignments:
        sig = tuple(sorted(a.items()))
        if sig not in seen:
            seen.add(sig)
            unique.append(a)
    top_assignments = unique

    # Step 4: Solve exact sequencing for each, with remaining time budget
    remaining_time = time_budget - (time.time() - t0)
    per_instance_time = max(1.0, remaining_time / len(top_assignments))

    best_result = {"makespan": float("inf"), "has_solution": False}
    results = []

    for assign in top_assignments:
        if time.time() - t0 >= time_budget:
            break

        sub_time_limit = min(per_instance_time, time_budget - (time.time() - t0))
        if sub_time_limit < 0.1:
            break

        seq_solver = FixedAssignmentSequencingModel(
            instance, assign, time_limit=sub_time_limit, mip_gap=0.0
        )
        result = seq_solver.solve()

        results.append({
            "assignment_n_machines": len(set(assign.values())),
            "makespan": result.get("makespan", float("inf")),
            "seq_time": result["wall_time"],
            "has_solution": result["has_solution"],
        })

        if result.get("has_solution") and result["makespan"] < best_result["makespan"]:
            best_result = result

    total_time = time.time() - t0

    return {
        "method": "rf_guided_search",
        "best_makespan": best_result.get("makespan"),
        "has_solution": best_result.get("has_solution", False),
        "total_time": total_time,
        "n_evaluated": len(results),
        "top_k_results": results[:5],
    }


def compare_search_vs_exact(
    instance: FJSPInstance,
    rf_models: Dict,
    feature_cols: List[str],
    time_budgets: List[float] = [5, 10, 30, 60, 120],
    seed: int = 42,
) -> List[Dict]:
    """
    Compare RF-guided search vs exact MIP at multiple time budgets.
    Returns a table of results.
    """
    comparison = []

    for budget in time_budgets:
        print(f"\n--- Time budget: {budget}s ---")

        # Exact MIP
        t0 = time.time()
        exact = PureFJSPMakespanModel(instance, time_limit=budget).solve()
        exact_time = time.time() - t0

        # RF search
        rf_result = rf_guided_search(
            instance, rf_models, feature_cols,
            time_budget=budget, n_candidates=300, top_k=10,
            seed=seed,
        )

        row = {
            "time_budget": budget,
            "exact_makespan": exact.get("makespan"),
            "exact_gap": exact.get("gap"),
            "exact_time": exact_time,
            "exact_has_sol": exact.get("has_solution"),
            "rf_makespan": rf_result["best_makespan"],
            "rf_time": rf_result["total_time"],
            "rf_has_sol": rf_result["has_solution"],
            "rf_n_evaluated": rf_result["n_evaluated"],
        }

        if row["exact_makespan"] and row["rf_makespan"]:
            row["rf_vs_exact"] = row["rf_makespan"] - row["exact_makespan"]
        else:
            row["rf_vs_exact"] = None

        comparison.append(row)
        print(f"  Exact: makespan={row['exact_makespan']}, gap={row['exact_gap']}")
        print(f"  RF:    makespan={row['rf_makespan']}, evaluated {row['rf_n_evaluated']} assignments")

    return comparison


if __name__ == "__main__":
    from fjsp_baseline import FJSPInstance
    from rf_train import load_rf_models, build_feature_cols

    inst = FJSPInstance.from_json_file("data/instances/brandimarte/mk01.json")

    # Load pre-trained RF models (or train if not available)
    models = load_rf_models("data/rf_models/mk01")
    feats = build_feature_cols(inst.num_machines)

    print(f"Instance: {inst.name} ({inst.num_jobs}x{inst.num_machines}, {inst.total_ops} ops)")
    print(f"Best known: {inst.best_known}")

    results = compare_search_vs_exact(inst, models, feats,
                                       time_budgets=[5, 10, 30, 60])
    print("\n=== Summary ===")
    for r in results:
        print(f"  Time {r['time_budget']:4.0f}s: Exact={r['exact_makespan']}, "
              f"RF={r['rf_makespan']}, Diff={r['rf_vs_exact']}")
