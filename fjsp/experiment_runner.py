"""
Unified experiment runner for L1, L2, L3.

Usage:
  python experiment_runner.py --layer L1 --instance mk01
  python experiment_runner.py --layer L2 --instance mk01 --time_budgets 5,30,60
  python experiment_runner.py --layer L3 --instance mk01 --rounds 5
  python experiment_runner.py --layer all --dataset brandimarte
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List

from fjsp_baseline import FJSPInstance, PureFJSPMakespanModel, FixedAssignmentSequencingModel
from schedule_generator import generate_dataset
from assignment_generators import AssignmentGenerator
from rf_train import train_global_rf, save_rf_models, load_rf_models, build_feature_cols
from rf_assignment_search import rf_guided_search, compare_search_vs_exact
from iterative_learning import IterativeLearner


# ============================================================
# L1: Assignment comparison
# ============================================================

def run_l1(instance: FJSPInstance, output_dir: str = None,
           n_train_samples: int = 5000, seed: int = 42) -> Dict:
    """Compare assignment methods: random, SPT, LPT, balanced, RF, exact."""
    print(f"\n{'='*60}")
    print(f"L1: Assignment Comparison — {instance.name}")
    print(f"{'='*60}")

    gen = AssignmentGenerator(instance, seed=seed)
    results = {}
    feats = build_feature_cols(instance.num_machines)

    # Step 1: Generate training data WITH TRUE LABELS
    # Generate diverse assignments → exact sequencing → get true makespan
    print(f"\n[1/3] Generating training data with TRUE makespan labels...")
    print(f"  Generating {min(300, n_train_samples // 10)} diverse assignments...")

    # Collect diverse assignments (fewer random, more heuristic-based)
    train_assignments = {}
    # Only a few random ones
    for _ in range(min(20, n_train_samples // 80)):
        assign = gen.random_assignment()
        sig = tuple(sorted(assign.items()))
        if sig not in train_assignments:
            train_assignments[sig] = assign

    # Heuristic assignments
    for name, assign in gen.generate_all_heuristic().items():
        sig = tuple(sorted(assign.items()))
        if sig not in train_assignments:
            train_assignments[sig] = assign

    # Perturbed versions of balanced (most likely to produce good labels)
    balanced = gen.balanced_assignment()
    for _ in range(30):
        perturbed = dict(balanced)
        j, p = gen.rng.choice(instance.all_ops)
        feasible = instance.feasible_machines.get((j, p), list(range(instance.num_machines)))
        alternatives = [k for k in feasible if k != perturbed.get((j, p))]
        if alternatives:
            perturbed[(j, p)] = gen.rng.choice(alternatives)
        sig = tuple(sorted(perturbed.items()))
        if sig not in train_assignments:
            train_assignments[sig] = perturbed

    assignments_list = list(train_assignments.values())
    print(f"  {len(assignments_list)} unique assignments to evaluate")

    # Exact sequencing for each (short time limit, allow gap)
    true_labeled_schedules = []
    for i, assign in enumerate(assignments_list):
        seq = FixedAssignmentSequencingModel(instance, assign, time_limit=5, mip_gap=0.01)
        result = seq.solve()
        if result.get("has_solution"):
            features = gen.extract_features(assign)
            true_labeled_schedules.append({
                "makespan": result["makespan"],
                "features": features,
                "assignment": {f"({j},{p})": k for (j, p), k in assign.items()},
            })
        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{len(assignments_list)}...")

    print(f"  Got {len(true_labeled_schedules)} valid labeled schedules")
    best_label = min(s["makespan"] for s in true_labeled_schedules)
    print(f"  Best makespan in training: {best_label:.1f}")

    # Train RF with true labels
    rf, _ = train_global_rf(true_labeled_schedules, feats, n_estimators=100, max_depth=12)
    rf_models = {"global": (rf, feats)}
    gen.set_rf_models(rf_models, feats)

    # Step 2: Generate test assignments from all methods
    print("\n[2/3] Generating test assignments...")
    assignments = gen.generate_all_heuristic()
    assignments["rf_global"] = gen.rf_global_assignment(n_candidates=500)
    assignments["rf_search"] = gen.rf_search_assignment(n_iter=1000, n_neighbors=5)

    # Step 3: Solve exact sequencing for each
    print("\n[3/3] Solving exact sequencing for each assignment...")
    for method_name, assign in assignments.items():
        t0 = time.time()
        seq = FixedAssignmentSequencingModel(
            instance, assign, time_limit=60, mip_gap=0.0
        )
        seq_result = seq.solve()
        elapsed = time.time() - t0

        n_machines = len(set(assign.values()))
        results[method_name] = {
            "makespan": seq_result.get("makespan"),
            "has_solution": seq_result.get("has_solution", False),
            "n_machines_used": n_machines,
            "seq_time": elapsed,
        }

        print(f"  {method_name:12s}: makespan={results[method_name]['makespan']}, "
              f"machines={n_machines}, time={elapsed:.1f}s")

    # Exact baseline
    print("\n--- Exact FJSP Baseline ---")
    exact = PureFJSPMakespanModel(instance, time_limit=60).solve()
    results["exact_baseline"] = {
        "makespan": exact.get("makespan"),
        "has_solution": exact.get("has_solution", False),
        "gap": exact.get("gap"),
        "time": exact["wall_time"],
    }
    print(f"  Exact baseline: makespan={exact.get('makespan')}, "
          f"gap={exact.get('gap')}, time={exact['wall_time']:.1f}s")

    # Compute gaps
    best_known = instance.best_known
    exact_mk = exact.get("makespan")

    print(f"\n{'='*60}")
    print(f"L1 SUMMARY — {instance.name}")
    print(f"{'='*60}")
    print(f"{'Method':>15s} {'Makespan':>9s} {'vs Exact':>9s} "
          f"{'vs BestKnown':>12s} {'Machines':>9s}")
    print("-" * 60)

    for method_name in ["random", "spt", "lpt", "balanced", "rf_global", "rf_search"]:
        r = results.get(method_name, {})
        mk = r.get("makespan", 0)
        gap_exact = (mk - exact_mk) / exact_mk * 100 if exact_mk else 0
        gap_known = (mk - best_known) / best_known * 100 if best_known else 0
        print(f"{method_name:>15s} {mk:9.1f} {gap_exact:+8.1f}% "
              f"{gap_known:+11.1f}% {r.get('n_machines_used', 0):9d}")

    print(f"{'exact_baseline':>15s} {exact_mk:9.1f} {'--':>9s} "
          f"{(exact_mk - best_known) / best_known * 100:+11.1f}%  --")

    return results


# ============================================================
# L2: RF-guided assignment search vs exact MIP
# ============================================================

def run_l2(instance: FJSPInstance, time_budgets: List[float] = None,
           n_train_samples: int = 5000, seed: int = 42) -> List[Dict]:
    """Compare RF search vs exact MIP at multiple time budgets."""
    if time_budgets is None:
        time_budgets = [5, 10, 30, 60, 120]

    print(f"\n{'='*60}")
    print(f"L2: RF Search vs Exact MIP — {instance.name}")
    print(f"{'='*60}")

    # Train RF
    print("\n[1/2] Training RF...")
    schedules = generate_dataset(instance, n_samples=n_train_samples, seed=seed)
    feats = build_feature_cols(instance.num_machines)
    rf, _ = train_global_rf(schedules, feats, n_estimators=100, max_depth=12)
    rf_models = {"global": (rf, feats)}

    print("\n[2/2] Running time-budget comparison...")
    results = compare_search_vs_exact(instance, rf_models, feats,
                                       time_budgets=time_budgets, seed=seed)

    # Print summary
    print(f"\n{'='*60}")
    print(f"L2 SUMMARY — {instance.name}")
    print(f"{'='*60}")
    print(f"{'Budget':>7s} {'Exact Mk':>9s} {'Exact Gap':>9s} "
          f"{'RF Mk':>9s} {'Delta':>9s} {'RF Evals':>9s}")
    print("-" * 60)
    for r in results:
        print(f"{r['time_budget']:6.0f}s {r.get('exact_makespan', 0) or 0:9.1f} "
              f"{r.get('exact_gap', 0) or 0:9.4f} "
              f"{r.get('rf_makespan', 0) or 0:9.1f} "
              f"{r.get('rf_vs_exact', 0) or 0:+9.1f} "
              f"{r['rf_n_evaluated']:9d}")

    return results


# ============================================================
# L3: Iterative learning
# ============================================================

def run_l3(instance: FJSPInstance, n_rounds: int = 5,
           initial_samples: int = 2000, output_dir: str = None,
           seed: int = 42) -> List[Dict]:
    """Run iterative self-improvement loop."""
    if output_dir is None:
        output_dir = f"data/iterative/{instance.name.lower()}"

    print(f"\n{'='*60}")
    print(f"L3: Iterative Self-Improvement — {instance.name}")
    print(f"{'='*60}")

    learner = IterativeLearner(instance, output_dir=output_dir, seed=seed)
    history = learner.run(
        n_rounds=n_rounds,
        initial_samples=initial_samples,
        n_candidates_per_round=500,
        top_k_per_round=10,
    )
    learner.print_summary()

    return history


# ============================================================
# Main
# ============================================================

def get_instances(dataset: str, data_dir: str = "data/instances") -> List[str]:
    """Get list of instance paths for a dataset."""
    paths = []

    if dataset == "brandimarte":
        for i in range(1, 11):
            p = os.path.join(data_dir, "brandimarte", f"mk{i:02d}.json")
            if os.path.exists(p):
                paths.append(p)
    elif dataset == "fattahi":
        import glob
        paths = sorted(glob.glob(os.path.join(data_dir, "fattahi", "*.json")))
    elif dataset == "hurink_vdata":
        import glob
        paths = sorted(glob.glob(os.path.join(data_dir, "hurink", "vdata", "*.json")))
    else:
        # Assume it's a direct path
        if os.path.exists(dataset):
            paths.append(dataset)
        else:
            paths.append(os.path.join(data_dir, dataset))

    return paths


def main():
    parser = argparse.ArgumentParser(description="FJSP Experiment Runner")
    parser.add_argument("--layer", choices=["L1", "L2", "L3", "all"], default="L1",
                        help="Which experiment layer to run")
    parser.add_argument("--instance", type=str, default="brandimarte/mk01.json",
                        help="Instance path or dataset name")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run on entire dataset (brandimarte, fattahi, hurink_vdata)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of learning rounds (L3 only)")
    parser.add_argument("--time_budgets", type=str, default="5,30,60,120",
                        help="Time budgets in seconds (L2 only)")
    parser.add_argument("--train_samples", type=int, default=5000,
                        help="Training samples for initial RF")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/instances")

    args = parser.parse_args()

    # Resolve instances
    if args.dataset:
        instance_paths = get_instances(args.dataset, args.data_dir)
    else:
        instance_paths = get_instances(args.instance, args.data_dir)

    if not instance_paths:
        print(f"No instances found!")
        sys.exit(1)

    print(f"Running {args.layer} on {len(instance_paths)} instance(s)")

    all_results = {}

    for path in instance_paths:
        inst = FJSPInstance.from_json_file(path)
        print(f"\n{'#'*60}")
        print(f"# {inst.name} ({inst.num_jobs}x{inst.num_machines}, "
              f"{inst.total_ops} ops, best_known={inst.best_known})")
        print(f"{'#'*60}")

        if args.layer in ("L1", "all"):
            l1_result = run_l1(inst, n_train_samples=args.train_samples, seed=args.seed)
            all_results[f"{inst.name}_L1"] = l1_result

        if args.layer in ("L2", "all"):
            budgets = [float(b) for b in args.time_budgets.split(",")]
            l2_result = run_l2(inst, time_budgets=budgets,
                               n_train_samples=args.train_samples, seed=args.seed)
            all_results[f"{inst.name}_L2"] = l2_result

        if args.layer in ("L3", "all"):
            l3_result = run_l3(inst, n_rounds=args.rounds,
                               initial_samples=min(2000, args.train_samples),
                               seed=args.seed)
            all_results[f"{inst.name}_L3"] = l3_result

    # Save overall results
    os.makedirs("results", exist_ok=True)
    out_path = f"results/experiment_{args.layer}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    # Convert non-serializable values
    serializable = {}
    for k, v in all_results.items():
        if isinstance(v, list):
            serializable[k] = v
        elif isinstance(v, dict):
            serializable[k] = {kk: str(vv) if not isinstance(vv, (int, float, bool, str, list, dict, type(None))) else vv
                               for kk, vv in v.items()}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
