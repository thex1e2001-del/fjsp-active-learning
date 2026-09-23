"""Budget-matched calibrated surrogate screening for FJSP.

RF-H is trained only on heuristic labels from a training pool. CRF adds a
residual model fitted to a small set of time-limited fixed-assignment MIP
labels from the training pool. The test pool is never used for model fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from assignment_generators import AssignmentGenerator
from experiments_final import BatchPredictor
from fjsp_baseline import FJSPInstance, FixedAssignmentSequencingModel
from heuristic_sequencer import HeuristicSequencer
from rf_train import build_feature_cols

ROOT = Path(__file__).resolve().parent
DEFAULT_INSTANCES = [f"mk{i:02d}" for i in range(1, 11)]


def assignment_key(assignment: Dict[Tuple[int, int], int]) -> str:
    raw = ";".join(f"{j}:{p}:{k}" for (j, p), k in sorted(assignment.items()))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def check_assignment(inst: FJSPInstance, assignment: Dict[Tuple[int, int], int]) -> None:
    if set(assignment) != set(inst.all_ops):
        raise ValueError("assignment does not contain exactly all operations")
    for op, machine in assignment.items():
        if machine not in inst.feasible_machines.get(op, []):
            raise ValueError(f"infeasible machine {machine} for operation {op}")


def perturb_assignment(inst: FJSPInstance, anchor: Dict[Tuple[int, int], int],
                       rng: random.Random) -> Dict[Tuple[int, int], int]:
    candidate = dict(anchor)
    for _ in range(rng.randint(1, max(1, inst.total_ops // 15))):
        op = rng.choice(inst.all_ops)
        feasible = inst.feasible_machines.get(op, list(range(inst.num_machines)))
        alternatives = [k for k in feasible if k != candidate[op]]
        if alternatives:
            candidate[op] = rng.choice(alternatives)
    return candidate


def make_pool(inst: FJSPInstance, seed: int, size: int) -> List[Dict[Tuple[int, int], int]]:
    """Create a deterministic random plus balanced-neighborhood candidate pool."""
    gen = AssignmentGenerator(inst, seed=seed)
    rng = random.Random(seed + 100_003)
    anchor = gen.balanced_assignment()
    pool, seen = [], set()
    while len(pool) < size:
        candidate = gen.random_assignment() if rng.random() < 0.6 else perturb_assignment(inst, anchor, rng)
        key = assignment_key(candidate)
        if key in seen:
            continue
        check_assignment(inst, candidate)
        seen.add(key)
        pool.append(candidate)
    return pool


def feature_matrix(bp: BatchPredictor, assignments: List[Dict], features: List[str]) -> np.ndarray:
    return np.asarray([[bp.extract_one(a).get(f, 0.0) for f in features]
                       for a in assignments], dtype=float)


def train_rf(X: np.ndarray, y: np.ndarray, seed: int, trees: int = 100,
             depth: int = 12, leaf: int = 5) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
        random_state=seed, n_jobs=-1,
    )
    model.fit(X, y)
    return model


def status_name(status: int) -> str:
    return {
        1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
        5: "UNBOUNDED", 9: "TIME_LIMIT", 11: "INTERRUPTED",
    }.get(int(status), str(status))


def solve_assignment(inst: FJSPInstance, assignment: Dict, verify_time: float,
                     cache: Dict[str, Dict]) -> Dict:
    key = assignment_key(assignment)
    if key in cache:
        return cache[key]
    result = FixedAssignmentSequencingModel(
        inst, assignment, time_limit=verify_time, mip_gap=0.0
    ).solve()
    row = {
        "assignment_id": key,
        "has_solution": bool(result.get("has_solution")),
        "status_code": int(result.get("status_code", result.get("status", -1))),
        "sol_count": int(result.get("sol_count", 0)),
        "status": status_name(result.get("status", -1)),
        "makespan": float(result["makespan"]) if result.get("makespan") is not None else None,
        "gap": float(result["gap"]) if result.get("gap") is not None and math.isfinite(result["gap"]) else None,
        "best_bound": float(result["best_bound"]) if result.get("best_bound") is not None and math.isfinite(result["best_bound"]) else None,
        "wall_time": float(result.get("wall_time", 0.0)),
    }
    cache[key] = row
    return row


def fit_and_score(inst: FJSPInstance, train_pool: List[Dict], test_pool: List[Dict],
                  seed: int, calibration: int, verify_time: float, top_k: int):
    features = build_feature_cols(inst.num_machines)
    bp = BatchPredictor(None, features, inst)
    sequencer = HeuristicSequencer(inst, seed=seed)
    train_X = feature_matrix(bp, train_pool, features)
    test_X = feature_matrix(bp, test_pool, features)
    train_h = np.asarray([sequencer.schedule(a, "best")[0] for a in train_pool], dtype=float)
    test_h = np.asarray([sequencer.schedule(a, "best")[0] for a in test_pool], dtype=float)
    max_load = test_X[:, features.index("max_machine_load")]
    max_job = test_X[:, features.index("max_job_time")]
    lb_test = np.maximum(max_load, max_job)

    rf_h = train_rf(train_X, train_h, seed=seed)
    rf_h_test = rf_h.predict(test_X)
    order = np.argsort(train_h)
    idxs = np.unique(np.linspace(0, len(order) - 1, min(calibration, len(order)), dtype=int))
    cache: Dict[str, Dict] = {}
    calibration_rows = []
    for pos in order[idxs]:
        result = solve_assignment(inst, train_pool[int(pos)], verify_time, cache)
        if result["has_solution"]:
            calibration_rows.append({"idx": int(pos), "heuristic": float(train_h[pos]), **result})
    accepted = [r for r in calibration_rows if r["gap"] is None or r["gap"] <= 0.05]

    residual_constant = 0.0
    fallback_reason = None
    if len(accepted) >= 3:
        residual_X = train_X[[r["idx"] for r in accepted]]
        residual_y = np.asarray([r["makespan"] - r["heuristic"] for r in accepted], dtype=float)
        residual_constant = float(np.median(residual_y))
        residual_model = train_rf(residual_X, residual_y, seed=seed + 17,
                                  trees=50, depth=5, leaf=2)
        residual_test = residual_model.predict(test_X)
    else:
        residual_test = np.full(len(test_pool), residual_constant, dtype=float)
        fallback_reason = "fewer_than_3_low_gap_calibration_labels"

    scores = {
        "LB": lb_test,
        "RF-H": rf_h_test,
        "CRF": rf_h_test + residual_test,
        "HeurSeq": test_h,
        # A seeded permutation makes Random a genuine random baseline while
        # preserving the exact same candidate pool and verification budget.
        "Random": np.asarray(random.Random(seed + 70_001).sample(
            range(len(test_pool)), len(test_pool)), dtype=float),
    }
    selections = {}
    for scorer, values in scores.items():
        chosen = np.argsort(values)[:top_k]
        selections[scorer] = {
            "indices": [int(i) for i in chosen],
            "predictions": [float(values[i]) for i in chosen],
        }
    meta = {
        "features": features,
        "n_train": len(train_pool),
        "n_test": len(test_pool),
        "calibration_requested": int(calibration),
        "calibration_solved": len(calibration_rows),
        "calibration_accepted": len(accepted),
        "calibration_acceptance_rate": (len(accepted) / len(calibration_rows)
                                         if calibration_rows else 0.0),
        "crf_fallback": fallback_reason,
        "calibration_rows": calibration_rows,
        "rf_h_train_r2": float(rf_h.score(train_X, train_h)),
        "cache": cache,
    }
    return selections, test_pool, meta, cache


def run_one(instance_name: str, seed: int, n_train: int, n_test: int,
            calibration: int, verify_time: float, top_k: int) -> Dict:
    path = ROOT / "data" / "instances" / "brandimarte" / f"{instance_name}.json"
    inst = FJSPInstance.from_json_file(str(path))
    train_pool = make_pool(inst, seed=seed, size=n_train)
    test_pool = make_pool(inst, seed=seed + 10_000, size=n_test)
    t0 = time.perf_counter()
    selections, test_pool, meta, cache = fit_and_score(
        inst, train_pool, test_pool, seed, calibration, verify_time, top_k
    )
    balanced = AssignmentGenerator(inst, seed=seed).balanced_assignment()
    balanced_row = solve_assignment(inst, balanced, verify_time, cache)
    scorer_rows = []
    for scorer, selection in selections.items():
        selected = []
        for idx, prediction in zip(selection["indices"], selection["predictions"]):
            result = solve_assignment(inst, test_pool[idx], verify_time, cache)
            selected.append({"prediction": prediction, **result})
        feasible = [r for r in selected if r["has_solution"]]
        low_gap = [r for r in feasible if r["gap"] is not None and r["gap"] <= 0.05]
        scorer_rows.append({
            "instance": instance_name, "seed": seed, "scorer": scorer,
            "top_k": top_k,
            "best_selected_makespan": min((r["makespan"] for r in feasible), default=None),
            "feasible_count": len(feasible),
            "feasible_ratio": len(feasible) / len(selected) if selected else 0.0,
            "low_gap_count": len(low_gap),
            "low_gap_ratio": len(low_gap) / len(selected) if selected else 0.0,
            "mean_gap_feasible": (float(np.mean([r["gap"] for r in feasible if r["gap"] is not None]))
                                   if any(r["gap"] is not None for r in feasible) else None),
            "selected": selected,
        })
    return {
        "instance": instance_name, "seed": seed, "top_k": top_k,
        "verify_time": verify_time, "wall_time": time.perf_counter() - t0,
        "balanced": balanced_row, "scorers": scorer_rows, "calibration": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--n-train", type=int, default=600)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--calibration", type=int, default=30)
    parser.add_argument("--verify-time", type=float, default=5.0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", default="results/calibrated_screening.json")
    args = parser.parse_args()
    all_rows = []
    for instance in args.instances:
        for seed in args.seeds:
            print(f"[{instance} seed={seed}]", flush=True)
            row = run_one(instance, seed, args.n_train, args.n_test,
                          args.calibration, args.verify_time, args.k)
            all_rows.append(row)
            print(f"  done in {row['wall_time']:.1f}s; "
                  f"calibration={row['calibration']['calibration_accepted']}", flush=True)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"saved {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
