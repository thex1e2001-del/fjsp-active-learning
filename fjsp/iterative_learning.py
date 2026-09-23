"""
L3: Iterative Self-Improvement.

Closed-loop learning:
  Round 0: Generate training data via heuristic SGS → train RF₀
  Round 1: Use RF₀ to search assignments → exact sequencing → get true makespan
           Add (features, true_makespan) to training set → retrain RF₁
  Round 2: Use RF₁ to search → add data → retrain RF₂
  ...

This is the core narrative: "RF learns from its own mistakes and improves with experience."
"""
import json
import os
import time
from typing import Dict, List

import numpy as np

from fjsp_baseline import FJSPInstance, FixedAssignmentSequencingModel
from assignment_generators import AssignmentGenerator
from schedule_generator import generate_dataset
from rf_train import (
    build_feature_cols,
    prepare_training_data,
    train_global_rf,
    save_rf_models,
    load_rf_models,
)


class IterativeLearner:
    """Closed-loop iterative learning of assignment quality."""

    def __init__(self, instance: FJSPInstance, output_dir: str = "data/iterative",
                 seed: int = 42):
        self.inst = instance
        self.output_dir = output_dir
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.feat_cols = build_feature_cols(instance.num_machines)

        os.makedirs(output_dir, exist_ok=True)
        self.history: List[Dict] = []

    def run(self, n_rounds: int = 5,
            initial_assignments: int = 50,
            n_candidates_per_round: int = 2000,
            top_k_per_round: int = 10,
            beam_width: int = 20,
            verbose: bool = True) -> List[Dict]:
        """
        Iterative learning with TRUE labels from the start.

        Round 0: Generate diverse assignments → exact seq (true labels) → train RF
        Round 1-N: Beam search over assignment space, guided by RF
                    → exact seq top-k → add to training → retrain

        Args:
            n_rounds: number of learning rounds (round 0 + n_rounds retrain)
            initial_assignments: number of diverse assignments for Round 0
            n_candidates_per_round: beam search pool size
            top_k_per_round: how many to exact-sequence per round
            beam_width: beam search width (keep top-N at each step)
        """
        gen = AssignmentGenerator(self.inst, seed=self.seed)

        # ---- Round 0: initial training data WITH TRUE LABELS ----
        if verbose:
            print(f"\n{'='*60}")
            print(f"Round 0: Generating true-label training data")
            print(f"{'='*60}")

        t0 = time.time()

        # Generate diverse assignments
        train_assigns = {}
        for _ in range(min(30, initial_assignments // 2)):
            assign = gen.random_assignment()
            sig = tuple(sorted(assign.items()))
            if sig not in train_assigns:
                train_assigns[sig] = assign

        for name, assign in gen.generate_all_heuristic().items():
            sig = tuple(sorted(assign.items()))
            if sig not in train_assigns:
                train_assigns[sig] = assign

        balanced = gen.balanced_assignment()
        for _ in range(initial_assignments - len(train_assigns)):
            perturbed = dict(balanced)
            j, p = gen.rng.choice(self.inst.all_ops)
            feasible = self.inst.feasible_machines.get((j, p), list(range(self.inst.num_machines)))
            alternatives = [k for k in feasible if k != perturbed.get((j, p))]
            if alternatives:
                perturbed[(j, p)] = gen.rng.choice(alternatives)
            sig = tuple(sorted(perturbed.items()))
            if sig not in train_assigns:
                train_assigns[sig] = perturbed

        assignments_list = list(train_assigns.values())
        if verbose:
            print(f"  {len(assignments_list)} unique assignments to label")

        # Exact sequencing for each
        true_labeled = []
        for i, assign in enumerate(assignments_list):
            seq = FixedAssignmentSequencingModel(self.inst, assign, time_limit=5, mip_gap=0.01)
            result = seq.solve()
            if result.get("has_solution"):
                features = gen.extract_features(assign)
                true_labeled.append({
                    "makespan": result["makespan"],
                    "assignment": {f"({j},{p})": k for (j, p), k in assign.items()},
                    "features": features,
                })
            if verbose and (i + 1) % 20 == 0:
                print(f"  Labeled {i + 1}/{len(assignments_list)}...")

        if verbose:
            best = min(s["makespan"] for s in true_labeled)
            print(f"  {len(true_labeled)} labeled, best makespan: {best:.1f}")

        # Train RF
        rf, _ = train_global_rf(true_labeled, self.feat_cols,
                                 n_estimators=100, max_depth=12)
        save_rf_models({"global": (rf, self.feat_cols)}, self.feat_cols,
                       os.path.join(self.output_dir, "round_0"))

        best_mk = min(s["makespan"] for s in true_labeled)
        self.history.append({
            "round": 0,
            "n_training_samples": len(true_labeled),
            "rf_r2": rf.score(*prepare_training_data(true_labeled, self.feat_cols)),
            "best_assignment_makespan": best_mk,
            "wall_time": time.time() - t0,
        })
        if verbose:
            print(f"  R^2: {self.history[-1]['rf_r2']:.4f}")
            print(f"  Best makespan: {best_mk:.1f}")

        # ---- Rounds 1..N: beam search + retrain ----
        all_schedules = list(true_labeled)
        best_overall_mk = best_mk
        best_overall_assign = None

        for round_idx in range(1, n_rounds + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Round {round_idx}: Beam search + retraining")
                print(f"{'='*60}")

            t0 = time.time()
            gen.set_rf_models({"global": (rf, self.feat_cols)}, self.feat_cols)

            # Beam search over assignments
            # Start from multiple starting points
            pool = []
            start_assigns = gen.generate_all_heuristic()
            start_assigns["rf_best"] = gen.rf_search_assignment(
                rf_key="global", n_iter=200, n_neighbors=5
            )

            seen_sig = set()
            for name, assign in start_assigns.items():
                sig = tuple(sorted(assign.items()))
                if sig not in seen_sig:
                    seen_sig.add(sig)
                    pool.append((gen._predict_assignment(assign, "global"), assign))

            pool.sort(key=lambda x: x[0])
            pool = pool[:beam_width]

            if verbose:
                print(f"  Beam search from {len(pool)} starting points...")

            # Beam search iterations
            for b_iter in range(10):
                candidates = list(pool)
                for _, current in pool:
                    # For each operation, try all feasible alternatives
                    for (j, p) in self.inst.all_ops:
                        feasible = self.inst.feasible_machines.get(
                            (j, p), list(range(self.inst.num_machines)))
                        for new_k in feasible:
                            if new_k == current.get((j, p)):
                                continue
                            candidate = dict(current)
                            candidate[(j, p)] = new_k
                            sig = tuple(sorted(candidate.items()))
                            if sig not in seen_sig:
                                seen_sig.add(sig)
                                score = gen._predict_assignment(candidate, "global")
                                candidates.append((score, candidate))
                                if len(candidates) > 10000:
                                    break
                        if len(candidates) > 10000:
                            break
                    if len(candidates) > 10000:
                        break

                # Sort by RF score, keep top beam_width
                candidates.sort(key=lambda x: x[0])
                pool = candidates[:beam_width]

            if verbose:
                rf_scores = [s for s, _ in pool[:5]]
                print(f"  Top-5 RF scores after beam search: {[f'{s:.1f}' for s in rf_scores]}")

            # Step 2: Exact sequencing for top-k
            new_data = []
            for i, (score, assign) in enumerate(pool[:top_k_per_round]):
                seq = FixedAssignmentSequencingModel(
                    self.inst, assign, time_limit=10, mip_gap=0.01)
                result = seq.solve()

                if result.get("has_solution"):
                    mk = result["makespan"]
                    features = gen.extract_features(assign)
                    new_data.append({
                        "makespan": mk,
                        "assignment": {f"({j},{p})": k for (j, p), k in assign.items()},
                        "features": features,
                    })
                    if mk < best_overall_mk:
                        best_overall_mk = mk
                        best_overall_assign = assign

                    if verbose:
                        print(f"  Top-{i+1}: RF={score:.1f}, true_makespan={mk:.1f}")

            # Step 3: Add to training and retrain
            all_schedules.extend(new_data)
            rf, _ = train_global_rf(all_schedules, self.feat_cols,
                                     n_estimators=100, max_depth=12)
            save_rf_models({"global": (rf, self.feat_cols)}, self.feat_cols,
                           os.path.join(self.output_dir, f"round_{round_idx}"))

            self.history.append({
                "round": round_idx,
                "n_training_samples": len(all_schedules),
                "n_new_samples": len(new_data),
                "rf_r2": rf.score(*prepare_training_data(all_schedules, self.feat_cols)),
                "best_assignment_makespan": best_overall_mk,
                "wall_time": time.time() - t0,
            })

            if verbose:
                print(f"  New data: {len(new_data)}, total: {len(all_schedules)}")
                print(f"  Best makespan: {best_overall_mk:.1f}, R^2: {self.history[-1]['rf_r2']:.4f}")

        hist_path = os.path.join(self.output_dir, "learning_history.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def print_summary(self):
        """Print the learning curve."""
        print(f"\n{'='*60}")
        print("LEARNING CURVE")
        print(f"{'='*60}")
        print(f"{'Round':>6s} {'Train N':>8s} {'RF R^2':>8s} "
              f"{'Best Mk':>9s} {'vs Opt':>8s} {'Time':>8s}")
        print("-" * 58)

        known = self.inst.best_known or 0
        for h in self.history:
            mk = h.get("best_assignment_makespan", 0) or 0
            gap = (mk - known) / known * 100 if known else 0
            print(f"{h['round']:6d} {h['n_training_samples']:8d} "
                  f"{h['rf_r2']:8.4f} {mk:9.1f} {gap:+8.1f}% "
                  f"{h['wall_time']:7.1f}s")


if __name__ == "__main__":
    from fjsp_baseline import FJSPInstance

    inst = FJSPInstance.from_json_file("data/instances/brandimarte/mk01.json")
    print(f"Instance: {inst.name} ({inst.num_jobs}x{inst.num_machines})")
    print(f"Best known makespan: {inst.best_known}")

    learner = IterativeLearner(inst, output_dir="data/iterative/mk01")
    learner.run(n_rounds=3, initial_assignments=50,
                n_candidates_per_round=2000, top_k_per_round=10)
    learner.print_summary()
