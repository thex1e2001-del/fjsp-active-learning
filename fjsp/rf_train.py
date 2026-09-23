"""
Train Random Forest models to predict schedule makespan from assignment features.
Two variants:
  - global_rf: single RF trained on all data, predicts total makespan
  - per_machine_rf: one RF per machine, trained only on samples where k is used
"""
import json
import os
import pickle
import time
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def prepare_training_data(schedules: List[Dict], feature_cols: List[str],
                          target_col: str = "makespan") -> tuple:
    """Convert schedule list to X, y arrays."""
    X = []
    y = []
    for s in schedules:
        feat_vec = [s["features"].get(f, 0) for f in feature_cols]
        X.append(feat_vec)
        y.append(s[target_col])
    return np.array(X), np.array(y)


def train_global_rf(schedules: List[Dict], feature_cols: List[str],
                    n_estimators: int = 100, max_depth: int = 15,
                    random_state: int = 42) -> RandomForestRegressor:
    """Train a single global RF predicting makespan from assignment features."""
    X, y = prepare_training_data(schedules, feature_cols)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)

    # R^2 score
    score = rf.score(X, y)
    print(f"  Global RF: R^2={score:.4f}, trees={n_estimators}, depth={max_depth}")

    return rf, feature_cols


def train_per_machine_rf(schedules: List[Dict], feature_cols: List[str],
                         num_machines: int,
                         n_estimators: int = 100, max_depth: int = 15,
                         random_state: int = 42) -> Dict[int, tuple]:
    """Train one RF per machine, each on samples where that machine is used."""
    models = {}
    for k in range(num_machines):
        # Filter samples where machine k is used
        k_schedules = [s for s in schedules if s["features"].get(f"used_{k}", 0) == 1]

        if len(k_schedules) < 50:
            print(f"  Machine {k}: only {len(k_schedules)} samples, skipping")
            continue

        X, y = prepare_training_data(k_schedules, feature_cols)

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        models[k] = (rf, feature_cols)
        print(f"  Machine {k} RF: {len(k_schedules)} samples, R^2={rf.score(X, y):.4f}")

    return models


def build_feature_cols(num_machines: int) -> List[str]:
    """Build the standard feature columns for assignment-level prediction."""
    # Aggregate features (new, richer)
    cols = [
        "num_used_machines",
        "max_machine_load",
        "load_imbalance",
        "max_ops_on_machine",
        "max_job_time",
        "load_std",
    ]
    # Per-machine features
    for k in range(num_machines):
        cols.append(f"load_{k}")
        cols.append(f"used_{k}")
        cols.append(f"ops_{k}")
    return cols


def save_rf_models(rf_models: dict, feature_cols: List[str], output_dir: str):
    """Save RF models and feature info."""
    os.makedirs(output_dir, exist_ok=True)

    for key, (rf, feats) in rf_models.items():
        path = os.path.join(output_dir, f"rf_{key}.joblib")
        with open(path, "wb") as f:
            pickle.dump(rf, f)

    # Save feature columns
    with open(os.path.join(output_dir, "features.json"), "w") as f:
        json.dump({"features": feature_cols, "num_features": len(feature_cols)}, f)

    print(f"Saved {len(rf_models)} models to {output_dir}")


def load_rf_models(model_dir: str) -> Dict:
    """Load RF models from directory."""
    models = {}
    feat_path = os.path.join(model_dir, "features.json")
    feats = []
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            feats = json.load(f)["features"]

    import glob as _glob
    for path in _glob.glob(os.path.join(model_dir, "rf_*.joblib")):
        key = os.path.splitext(os.path.basename(path))[0].replace("rf_", "")
        with open(path, "rb") as f:
            rf = pickle.load(f)
        models[key] = (rf, feats)

    return models


if __name__ == "__main__":
    from schedule_generator import generate_dataset, SerialSGS
    from fjsp_baseline import FJSPInstance

    inst = FJSPInstance.from_json_file("data/instances/brandimarte/mk01.json")
    print(f"Instance: {inst.name}")

    print("\n--- Generating training data ---")
    schedules = generate_dataset(inst, n_samples=2000)

    feat_cols = build_feature_cols(inst.num_machines)
    print(f"Features ({len(feat_cols)}): {feat_cols}")

    print("\n--- Training global RF ---")
    global_rf, _ = train_global_rf(schedules, feat_cols, n_estimators=50, max_depth=10)

    print("\n--- Training per-machine RFs ---")
    per_machine = train_per_machine_rf(schedules, feat_cols, inst.num_machines,
                                       n_estimators=50, max_depth=10)

    # Save
    out = "data/rf_models/mk01"
    save_rf_models({"global": (global_rf, feat_cols), **per_machine}, feat_cols, out)
