"""
Assignment generators for FJSP (routing only, no sequencing).
Each generator produces a mapping (j,p) -> machine_k.

Methods:
  - random: randomly pick feasible machine
  - spt: pick machine with shortest processing time
  - lpt: pick machine with longest processing time
  - balanced: pick machine with minimum current load
  - rf_global: use global RF to predict assignment quality
  - rf_per_machine: use per-machine RFs to evaluate assignments
  - rf_search: sample and evaluate assignments with RF
"""
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np

from fjsp_baseline import FJSPInstance


class AssignmentGenerator:
    """Generate machine assignments for FJSP instances."""

    def __init__(self, instance: FJSPInstance, seed: int = None):
        self.inst = instance
        self.rng = random.Random(seed)

    # ---------------------------------------------------------------
    # Rule-based generators
    # ---------------------------------------------------------------

    def random_assignment(self) -> Dict[Tuple[int, int], int]:
        """Randomly assign each operation to a feasible machine."""
        assign = {}
        for (j, p) in self.inst.all_ops:
            feasible = self.inst.feasible_machines.get((j, p), list(range(self.inst.num_machines)))
            assign[(j, p)] = self.rng.choice(feasible)
        return assign

    def spt_assignment(self) -> Dict[Tuple[int, int], int]:
        """Pick machine with Shortest Processing Time for each operation."""
        assign = {}
        for (j, p) in self.inst.all_ops:
            feasible = self.inst.feasible_machines.get((j, p), list(range(self.inst.num_machines)))
            best_k = min(feasible, key=lambda k: next(
                (alt[1] for alt in self.inst.jobs[j][p] if alt[0] == k), 99999
            ))
            assign[(j, p)] = best_k
        return assign

    def lpt_assignment(self) -> Dict[Tuple[int, int], int]:
        """Pick machine with Longest Processing Time for each operation."""
        assign = {}
        for (j, p) in self.inst.all_ops:
            feasible = self.inst.feasible_machines.get((j, p), list(range(self.inst.num_machines)))
            best_k = max(feasible, key=lambda k: next(
                (alt[1] for alt in self.inst.jobs[j][p] if alt[0] == k), 0
            ))
            assign[(j, p)] = best_k
        return assign

    def balanced_assignment(self) -> Dict[Tuple[int, int], int]:
        """Pick machine with minimum current load (greedy load balancing)."""
        assign = {}
        loads = defaultdict(float)

        # Process jobs in random order
        ops = list(self.inst.all_ops)
        self.rng.shuffle(ops)

        for (j, p) in ops:
            feasible = self.inst.feasible_machines.get((j, p), list(range(self.inst.num_machines)))
            best_k = min(feasible, key=lambda k: loads[k])
            assign[(j, p)] = best_k
            for alt in self.inst.jobs[j][p]:
                if alt[0] == best_k:
                    loads[best_k] += alt[1]
                    break

        return assign

    # ---------------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------------

    def extract_features(self, assignment: Dict[Tuple[int, int], int]) -> Dict:
        """Extract assignment + sequencing-proxy features."""
        features = {}
        inst = self.inst

        # Per-machine stats
        load_k = {}
        ops_k = {}
        used_k = {}
        for k in range(inst.num_machines):
            ld = 0
            op = 0
            for (j, p), mk in assignment.items():
                if mk == k:
                    op += 1
                    for alt in inst.jobs[j][p]:
                        if alt[0] == k:
                            ld += alt[1]
                            break
            load_k[k] = ld
            ops_k[k] = op
            used_k[k] = 1 if op > 0 else 0
            features[f"load_{k}"] = ld
            features[f"used_{k}"] = used_k[k]
            features[f"ops_{k}"] = op

        # --- Aggregate features ---
        features["num_used_machines"] = sum(used_k.values())
        features["max_machine_load"] = max(load_k.values())

        used_loads = [load_k[k] for k in range(inst.num_machines) if used_k[k]]
        features["load_imbalance"] = max(used_loads) - min(used_loads) if used_loads else 0
        features["max_ops_on_machine"] = max(ops_k.values())

        job_times = []
        for j, ops_list in inst.jobs.items():
            total = 0
            for p in range(len(ops_list)):
                k = assignment.get((j, p))
                if k is not None:
                    for alt in inst.jobs[j][p]:
                        if alt[0] == k:
                            total += alt[1]
                            break
            job_times.append(total)
        features["max_job_time"] = max(job_times) if job_times else 0

        if used_loads:
            mean_load = sum(used_loads) / len(used_loads)
            var = sum((l - mean_load) ** 2 for l in used_loads) / len(used_loads)
            features["load_std"] = var ** 0.5
        else:
            features["load_std"] = 0

        return features

    # ---------------------------------------------------------------
    # RF-based generators
    # ---------------------------------------------------------------

    def set_rf_models(self, rf_models: Dict, feature_cols: List[str]):
        """Set RF models for prediction.
        rf_models: {key: (rf_object, feature_cols)} where key can be 'global', 0, 1, ...
        """
        self.rf_models = rf_models
        self.feature_cols = feature_cols

    def _predict_assignment(self, assignment: Dict[Tuple[int, int], int],
                            rf_key: str = "global") -> float:
        """Predict makespan for an assignment using a specific RF."""
        if not hasattr(self, 'rf_models') or rf_key not in self.rf_models:
            return float("inf")

        rf, feats = self.rf_models[rf_key]
        features = self.extract_features(assignment)
        feat_vec = np.array([[features.get(f, 0) for f in feats]])
        return rf.predict(feat_vec)[0]

    def rf_global_assignment(self, n_candidates: int = 100) -> Dict[Tuple[int, int], int]:
        """Generate N random assignments, evaluate with global RF, return best."""
        best_assign = None
        best_score = float("inf")

        for _ in range(n_candidates):
            assign = self.random_assignment()
            score = self._predict_assignment(assign, "global")
            if score < best_score:
                best_score = score
                best_assign = assign

        return best_assign

    def rf_search_assignment(self, initial: Dict[Tuple[int, int], int] = None,
                             n_iter: int = 500, n_neighbors: int = 10,
                             rf_key: str = "global") -> Dict[Tuple[int, int], int]:
        """Local search in assignment space: perturb and evaluate with RF.

        Args:
            initial: starting assignment (use balanced if None)
            n_iter: number of search iterations
            n_neighbors: neighbors to evaluate per iteration
            rf_key: which RF model to use for scoring

        Returns:
            Best assignment found
        """
        current = initial if initial is not None else self.balanced_assignment()
        current_score = self._predict_assignment(current, rf_key)
        ops = self.inst.all_ops

        for iteration in range(n_iter):
            improved = False

            for _ in range(n_neighbors):
                # Perturb: change machine for one random operation
                j, p = self.rng.choice(ops)
                feasible = self.inst.feasible_machines.get((j, p), list(range(self.inst.num_machines)))
                alternatives = [k for k in feasible if k != current.get((j, p))]
                if not alternatives:
                    continue
                new_k = self.rng.choice(alternatives)

                candidate = dict(current)
                candidate[(j, p)] = new_k
                score = self._predict_assignment(candidate, rf_key)

                if score < current_score:
                    current = candidate
                    current_score = score
                    improved = True

            if not improved:
                break  # local optimum reached

        return current

    def generate_all_heuristic(self) -> Dict[str, Dict[Tuple[int, int], int]]:
        """Generate assignments from all rule-based methods."""
        return {
            "random": self.random_assignment(),
            "spt": self.spt_assignment(),
            "lpt": self.lpt_assignment(),
            "balanced": self.balanced_assignment(),
        }


if __name__ == "__main__":
    from fjsp_baseline import FJSPInstance

    inst = FJSPInstance.from_json_file("data/instances/brandimarte/mk01.json")
    gen = AssignmentGenerator(inst, seed=42)

    print("Assignment methods:")
    for name in ["random", "spt", "lpt", "balanced"]:
        assign = getattr(gen, f"{name}_assignment")()
        n_mach = len(set(assign.values()))
        print(f"  {name:10s}: uses {n_mach}/{inst.num_machines} machines")
