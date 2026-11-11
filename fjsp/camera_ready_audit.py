"""Traceable paired replay of the accepted paper's fixed GitHub implementation.

No source modules or historical results are modified. Each unique solve is
persisted immediately; method accounting charges reused solves independently.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def best(rows):
    return min((r["makespan"] for r in rows if r["has_solution"]), default=None)


def validated_incumbent(inst, schedule, makespan, assignment):
    """Reject an invalid solver incumbent without relaxing schedule tolerances."""
    from schedule_validation import validate_schedule
    try:
        validate_schedule(inst, schedule, makespan, assignment, tolerance=1e-4)
    except ValueError as exc:
        return False, str(exc)
    return True, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="protocols/camera_ready_mk06_audit_20261111.json")
    ap.add_argument("--output", default="results/camera_ready_mk06_audit_20261111")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    protocol_path = ROOT / args.protocol
    cfg = json.loads(protocol_path.read_text(encoding="utf-8"))
    source = ROOT / cfg["source_dir"]
    sys.path.insert(0, str(source))
    import numpy as np
    import sklearn
    import gurobipy as gp
    import calibrated_screening as cs
    import active_screening_dev as original
    from heuristic_sequencer import HeuristicSequencer
    from sklearn.ensemble import RandomForestRegressor
    from schedule_validation import validate_schedule

    assert Path(cs.__file__).resolve().parent == source.resolve()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {p.name: sha(p) for p in sorted(source.glob("*.py"))}
    # Authenticate extracted Python and input files against the archived ZIP.
    import zipfile
    archive = ROOT / "archives/camera_ready_source_check_20261110/github_fjsp_active_learning_8606fae.zip"
    with zipfile.ZipFile(archive) as z:
        for p in list(source.glob("*.py")) + [source / "data/instances/brandimarte" / f"{n}.json" for n in cfg["instances"]]:
            suffix = "/fjsp/" + p.relative_to(source).as_posix()
            members = [n for n in z.namelist() if n.endswith(suffix)]
            assert len(members) == 1 and hashlib.sha256(z.read(members[0])).hexdigest() == sha(p), p
    identity = {"protocol_sha256": sha(protocol_path), "runner_sha256": sha(__file__),
                "source_hashes": source_hashes, "archive_sha256": sha(archive),
                "validator_sha256": sha(ROOT / "schedule_validation.py"),
                "numpy": np.__version__, "sklearn": sklearn.__version__,
                "gurobi": list(gp.gurobi.version()), "python": platform.python_version()}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())["identity"] == identity, "Existing run identity mismatch; use a new output directory."
    else:
        cpu = subprocess.check_output(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"], text=True).strip()
        dump(manifest_path, {"identity": identity, "protocol": cfg, "cpu": cpu,
                             "platform": platform.platform(), "created_utc": datetime.now(timezone.utc).isoformat()})
    gp.setParam("OutputFlag", 0)

    for name in cfg["instances"]:
        old_path = source / "data/instances/brandimarte" / f"{name}.json"
        verified_path = ROOT / "data/verified/brandimarte" / f"{name}.json"
        old_data = json.loads(old_path.read_text())
        verified_data = json.loads(verified_path.read_text())
        old = cs.FJSPInstance(old_data)
        verified = cs.FJSPInstance(verified_data)
        assert old.jobs == verified.jobs and old.feasible_machines == verified.feasible_machines
        fixed_data = copy.deepcopy(old_data)
        fixed_data["num_machines"] = verified.num_machines
        fixed = cs.FJSPInstance(fixed_data)
        for seed in cfg["seeds"]:
            run_dir = out / f"{name}_seed{seed}"
            run_dir.mkdir(exist_ok=True)
            if (run_dir / "result.json").exists():
                print(f"{name} seed={seed}: complete; retained", flush=True)
                continue
            started = time.perf_counter()
            train = cs.make_pool(old, seed, cfg["n_train"])
            test = cs.make_pool(old, seed + 10000, cfg["n_test"])
            assert train == cs.make_pool(fixed, seed, cfg["n_train"])
            assert test == cs.make_pool(fixed, seed + 10000, cfg["n_test"])
            keys_train = [cs.assignment_key(a) for a in train]
            keys_test = [cs.assignment_key(a) for a in test]
            assert not set(keys_train) & set(keys_test), "Train/test overlap; stop rather than silently change pool."
            seq = HeuristicSequencer(old, seed)
            yh = np.array([seq.schedule(a, "best")[0] for a in train], float)
            yt = np.array([seq.schedule(a, "best")[0] for a in test], float)
            seq_fixed = HeuristicSequencer(fixed, seed)
            assert np.array_equal(yh, [seq_fixed.schedule(a, "best")[0] for a in train])
            assert np.array_equal(yt, [seq_fixed.schedule(a, "best")[0] for a in test])
            order = np.argsort(yh)
            positions = np.unique(np.linspace(0, len(order)-1, min(cfg["calibration"], len(order)), dtype=int))
            calibration_indices = [int(i) for i in order[positions]]
            features = {}
            for variant, inst in [("legacy", old), ("corrected", fixed)]:
                cols = cs.build_feature_cols(inst.num_machines)
                bp = cs.BatchPredictor(None, cols, inst)
                # Original feature routine retained for protocol parity.
                X = cs.feature_matrix(bp, train, cols)
                Xt = cs.feature_matrix(bp, test, cols)
                features[variant] = (cols, X, Xt)
            prepared = {"instance": name, "seed": seed, "instance_hashes": {"legacy": sha(old_path), "verified": sha(verified_path)},
                        "old_machines": old.num_machines, "corrected_machines": fixed.num_machines,
                        "held_fixed_horizon": old.horizon, "verified_source_horizon": verified.horizon,
                        "calibration_indices": calibration_indices, "calibration_rank_positions": positions.tolist(),
                        "train_test_overlap": [], "heuristic_train": yh.tolist(), "heuristic_test": yt.tolist(),
                        "features": {v: {"columns": f, "train": X.tolist(), "test": Xt.tolist()} for v, (f, X, Xt) in features.items()},
                        "pools": {pool_name: [{"id": cs.assignment_key(a), "assignment": [[j,p,k] for (j,p),k in a.items()]} for a in pool] for pool_name, pool in [("train", train), ("test", test)]},
                        "preparation_seconds": time.perf_counter()-started}
            if (run_dir / "prepared.json").exists():
                previous = json.loads((run_dir / "prepared.json").read_text())
                assert previous["pools"] == prepared["pools"] and previous["instance_hashes"] == prepared["instance_hashes"]
            else:
                dump(run_dir / "prepared.json", prepared)
            if args.prepare_only:
                print(f"{name} seed={seed}: pools, features, disjointness and heuristic parity verified", flush=True)
                continue

            bank = {}
            bank_path = run_dir / "solves.jsonl"
            if bank_path.exists():
                for line in bank_path.read_text().splitlines():
                    row = json.loads(line)
                    assert row["assignment_id"] not in bank
                    bank[row["assignment_id"]] = row
            physical_initial = len(bank)
            requests = []

            def evaluate(pool_name, idx, method):
                assignment = (train if pool_name == "train" else test)[idx]
                key = cs.assignment_key(assignment)
                hit = key in bank
                if not hit:
                    t0 = time.perf_counter()
                    model, s, c, cmax = cs.FixedAssignmentSequencingModel(fixed, assignment, cfg["verify_seconds"], 0.0).build()
                    model.update()
                    old_model, _, _, _ = cs.FixedAssignmentSequencingModel(old, assignment, cfg["verify_seconds"], 0.0).build()
                    old_model.update()
                    assert model.Fingerprint == old_model.Fingerprint, "Model changed; cannot share outcomes across variants."
                    fingerprint = int(model.Fingerprint)
                    old_model.dispose()
                    check_build_seconds = time.perf_counter()-t0
                    model.Params.OutputFlag = 0
                    model.Params.TimeLimit = cfg["verify_seconds"]
                    model.Params.MIPGap = cfg["mip_gap"]
                    model.Params.Seed = cfg["solver_seed"]
                    model.Params.Threads = cfg["solver_threads"]
                    t_opt = time.perf_counter()
                    model.optimize()
                    optimize_seconds = time.perf_counter()-t_opt
                    raw_has = model.SolCount > 0
                    raw_makespan = finite(cmax.X) if raw_has else None
                    schedule = [{"job_id": j, "job_pos": p, "machine": assignment[j,p], "start": s[j,p].X, "completion": c[j,p].X} for j,p in fixed.all_ops] if raw_has else []
                    has, validation_error = validated_incumbent(fixed, schedule, cmax.X, assignment) if raw_has else (False, None)
                    row = {"assignment_id": key, "pool": pool_name, "pool_index": idx,
                           "has_solution": has, "makespan": raw_makespan if has else None,
                           "solver_has_incumbent": raw_has, "solver_reported_makespan": raw_makespan,
                           "validation_error": validation_error,
                           "rejected_invalid_incumbent": raw_has and not has,
                           "status_code": int(model.Status), "status": cs.status_name(model.Status),
                           "sol_count": int(model.SolCount), "gap": finite(model.MIPGap) if has else None,
                           "best_bound": finite(model.ObjBound), "nodes": float(model.NodeCount),
                           "solver_runtime": float(model.Runtime), "optimize_seconds": optimize_seconds,
                           "build_and_equivalence_seconds": check_build_seconds,
                           "physical_total_seconds": time.perf_counter()-t0,
                           "model_fingerprint": fingerprint, "old_corrected_fingerprints_equal": True,
                           "schedule_validated": has, "schedule": schedule,
                           "completed_utc": datetime.now(timezone.utc).isoformat()}
                    model.dispose()
                    with bank_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                    bank[key] = row
                    print(f"{name} seed={seed} solve={len(bank)} {method} {pool_name}[{idx}] best={row['makespan']} status={row['status']} validation_error={validation_error}", flush=True)
                requests.append({"method": method, "pool": pool_name, "index": idx, "assignment_id": key, "physical_cache_hit": hit})
                return bank[key]

            calibration = [evaluate("train", i, "calibration-shared") for i in calibration_indices]
            labels = [(i,r) for i,r in zip(calibration_indices,calibration) if r["has_solution"] and r["makespan"] is not None]
            summaries = []
            selections = {}
            for variant, (cols, X, Xt) in features.items():
                tf = time.perf_counter()
                if len(labels) >= 5:
                    direct = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=2, random_state=seed, n_jobs=1).fit(X[[i for i,r in labels]], [r["makespan"] for i,r in labels])
                    pred = direct.predict(Xt)
                else:
                    pred = np.full(len(test), np.mean([r["makespan"] for i,r in labels]) if labels else np.inf)
                al_fit_seconds = time.perf_counter()-tf
                tf = time.perf_counter()
                rfh = cs.train_rf(X, yh, seed).predict(Xt)
                rfh_fit_seconds = time.perf_counter()-tf
                scores = {"AL-RF": pred, "RF-H": rfh,
                          "LB": np.maximum(Xt[:,cols.index("max_machine_load")], Xt[:,cols.index("max_job_time")]),
                          "HeurSeq": yt, "Random": np.asarray(random.Random(seed+70001).sample(range(len(test)),len(test)),float),
                          "HeurSeq-35": yt}
                for method, values in scores.items():
                    # Feature-independent controls need one result, shared by both variants.
                    if variant == "corrected" and method not in ("AL-RF", "RF-H"):
                        continue
                    tag = variant + "/" + method
                    k = cfg["baseline_k"] if method == "HeurSeq-35" else cfg["top_k"]
                    indices = [int(i) for i in np.argsort(values)[:k]]
                    selections[tag] = {"indices": indices, "all_predictions": [finite(v) for v in values]}
                    selected = [evaluate("test", i, tag) for i in indices]
                    owned = calibration + selected if method == "AL-RF" else selected
                    unique = {r["assignment_id"]: r for r in owned}
                    summaries.append({"variant": variant if method in ("AL-RF", "RF-H") else "shared", "method": method,
                                      "test_best": best(selected), "calibration_best": best(calibration) if method=="AL-RF" else None,
                                      "combined_mip_best": best(owned), "final_feasible": sum(r["has_solution"] for r in selected),
                                      "calibration_feasible": len(labels) if method=="AL-RF" else 0,
                                      "logical_requests": len(owned), "charged_unique_solves": len(unique),
                                      "nominal_solver_cap_seconds": len(unique)*cfg["verify_seconds"],
                                      "charged_optimize_seconds": sum(r["optimize_seconds"] for r in unique.values()),
                                      "fit_predict_seconds": al_fit_seconds if method=="AL-RF" else rfh_fit_seconds if method=="RF-H" else 0.0,
                                      "assignment_ids": list(unique), "test_indices": indices})
            # Replay the unmodified original run_one using this bank. Checks indexing,
            # calibration acceptance, all five rankings and result aggregation.
            observed = []
            saved_solve = original.solve_assignment
            def replay_solve(inst, assignment, verify_time, cache):
                key = cs.assignment_key(assignment)
                observed.append(key)
                return bank[key]
            try:
                original.solve_assignment = replay_solve
                parity_result = original.run_one(name, seed, cfg["n_train"], cfg["n_test"], cfg["calibration"], cfg["verify_seconds"], cfg["top_k"])
            finally:
                original.solve_assignment = saved_solve
            expected = [keys_train[i] for i in calibration_indices]
            for method in ("AL-RF", "RF-H", "LB", "HeurSeq", "Random"):
                expected.extend(keys_test[i] for i in selections["legacy/"+method]["indices"])
                record = next(r for r in summaries if r["method"]==method and r["variant"] in ("legacy", "shared"))
                orig = next(r for r in parity_result["rows"] if r["scorer"]==method)
                assert orig["best"]==record["test_best"] and orig["feasible"]==record["final_feasible"] and orig["labels"]==len(labels)
            assert expected == observed, "Original call-order/selection parity failed"
            dump(run_dir / "result.json", {"instance": name, "seed": seed, "summaries": summaries,
                 "selections": selections, "calibration_best": best(calibration), "calibration_feasible": len(labels),
                 "heuristic_best_train": float(min(yh)), "heuristic_best_test": float(min(yt)),
                 "physical_unique_solves": len(bank), "physical_new_solves_this_invocation": len(bank)-physical_initial,
                 "physical_cache_hits_this_invocation": sum(r["physical_cache_hit"] for r in requests),
                 "requests": requests, "original_replay_parity": True,
                 "current_invocation_seconds": time.perf_counter()-started,
                 "note": "Paired evaluation-bank replay. Charged solver time excludes preprocessing and duplicate model equivalence checks; not independent end-to-end runtime."})
            print(f"{name} seed={seed}: DONE; original parity and every schedule validated", flush=True)


if __name__ == "__main__":
    main()
