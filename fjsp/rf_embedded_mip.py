r"""
Simplified RF-embedded MIP: single global RF, aggregate feature variables.

Key differences from original per-machine approach:
  - 1 global RF (not 6 per-machine RFs)
  - ~12 feature variables (not per-operation q variables)
  - Leaf activation: y[t,l]  (not q[i,j,k,s,t,r])
  - Lexicographic or weighted objective: Cmax + \lambda * Q_global

Variable count: O(n_trees * n_leaves) ~ 4000, NOT O(n_ops * n_machines * n_trees * n_leaves)
"""
import json, os, time, pickle
from typing import Dict, List, Tuple, Optional
from itertools import combinations

import numpy as np
from gurobipy import Model, GRB, quicksum

from fjsp_baseline import FJSPInstance
from schedule_generator import generate_dataset
from assignment_generators import AssignmentGenerator
from rf_train import train_global_rf, build_feature_cols


# ============================================================
# MIP-friendly feature set (only linearly-expressible features)
# ============================================================

def mip_feature_cols(num_machines: int) -> List[str]:
    """Features easily expressible as linear functions of x variables."""
    cols = []
    for k in range(num_machines):
        cols.append(f"load_{k}")
    for k in range(num_machines):
        cols.append(f"ops_{k}")
    # used_k is binary, derived from ops_k > 0
    return cols


# ============================================================
# Train small RF and extract tree structure
# ============================================================

def train_small_rf_for_mip(instance: FJSPInstance, n_trees: int = 20,
                           max_depth: int = 6, n_samples: int = 2000,
                           seed: int = 42) -> Tuple:
    """Train a compact global RF and return (rf, tree_rules, feature_cols)."""
    from sklearn.ensemble import RandomForestRegressor

    feats = mip_feature_cols(instance.num_machines)

    print(f"Training RF: {n_trees} trees, depth {max_depth}, {len(feats)} features")

    # Generate labeled training data
    gen = AssignmentGenerator(instance, seed=seed)
    from heuristic_sequencer import HeuristicSequencer
    seq = HeuristicSequencer(instance)

    print("  Generating assignments...")
    train_assignments = {}
    for name, a in gen.generate_all_heuristic().items():
        sig = tuple(sorted(a.items()))
        train_assignments[sig] = a
    for _ in range(500):
        a = gen.random_assignment()
        sig = tuple(sorted(a.items()))
        if sig not in train_assignments:
            train_assignments[sig] = a
    balanced = gen.balanced_assignment()
    import random as _r
    ops_list = instance.all_ops
    for _ in range(1000):
        perturbed = dict(balanced)
        for _ in range(_r.randint(1, 3)):
            j, p = _r.choice(ops_list)
            feasible = instance.feasible_machines.get((j, p), list(range(instance.num_machines)))
            alts = [k for k in feasible if k != perturbed.get((j, p))]
            if alts:
                perturbed[(j, p)] = _r.choice(alts)
        sig = tuple(sorted(perturbed.items()))
        if sig not in train_assignments:
            train_assignments[sig] = a

    # Extract MIP-compatible features only
    X, y = [], []
    all_feats = build_feature_cols(instance.num_machines)  # full feature set
    full_feat_idx = {f: i for i, f in enumerate(all_feats)}

    t0 = time.time()
    for assign in train_assignments.values():
        full_f = gen.extract_features(assign)
        # Map to MIP-compatible feature vector
        mip_vec = [full_f.get(f, 0) for f in feats]
        X.append(mip_vec)
        h_mk, _ = seq.schedule(assign, "best")
        y.append(h_mk)

    X = np.array(X)
    y = np.array(y)

    print(f"  {len(X)} samples, labeling took {time.time()-t0:.1f}s")

    rf = RandomForestRegressor(
        n_estimators=n_trees, max_depth=max_depth,
        min_samples_leaf=max(5, len(X) // 100),
        random_state=seed, n_jobs=-1,
    )
    rf.fit(X, y)
    print(f"  R^2 = {rf.score(X, y):.4f}")
    print(f"  Total leaves: {sum(t.tree_.n_node_samples[t.tree_.children_left == -1].shape[0] for t in rf.estimators_)}")

    # Extract tree rules
    all_trees = []
    for t_idx, tree in enumerate(rf.estimators_):
        rules = extract_tree_rules(tree, feats)
        all_trees.append(rules)

    return rf, all_trees, feats


def extract_tree_rules(tree, feature_names: List[str]) -> List[Dict]:
    """Extract all root-to-leaf paths from a decision tree.

    Returns list of {leaf_id, conditions: [(feature_name, op, threshold), ...], value: float}
    """
    rules = []
    t = tree.tree_
    left = t.children_left
    right = t.children_right
    threshold = t.threshold
    feature = t.feature
    value = t.value

    def traverse(node_id, path_conditions):
        if left[node_id] == -1:  # leaf
            leaf_val = float(value[node_id][0][0])
            rules.append({
                "leaf_id": node_id,
                "conditions": list(path_conditions),
                "value": leaf_val,
            })
        else:
            feat_name = feature_names[feature[node_id]]
            thresh = threshold[node_id]
            # Left branch: feature <= threshold
            traverse(left[node_id], path_conditions + [(feat_name, "<=", thresh)])
            # Right branch: feature > threshold
            traverse(right[node_id], path_conditions + [(feat_name, ">", thresh)])

    traverse(0, [])
    return rules


# ============================================================
# RF-Embedded FJSP MIP
# ============================================================

class RFEmbeddedMIP:
    """FJSP MIP with embedded global RF surrogate.

    Variables:
      x[j,p,k]     - assignment (binary)
      delta[...]   - sequencing (binary)
      S[j,p], C[j,p], Cmax - timing
      z_f          - RF feature variables (continuous/binary)
      y[t,l]       - leaf activation (binary)

    Constraints:
      (1)-(5)  Standard FJSP
      (6)      Feature linking: z_f = f(x)
      (7)      Leaf uniqueness: Σ_l y[t,l] = 1
      (8)      Path constraints (big-M): y[t,l]=1 ⇒ feature conditions

    Objective:
      min Cmax + lambda * Q_global    (weighted, configurable)
    """

    def __init__(self, instance: FJSPInstance, tree_rules: List[List[Dict]],
                 feature_names: List[str],
                 lambda_: float = 0.0, time_limit: float = None,
                 mip_gap: float = 0.0):
        self.inst = instance
        self.tree_rules = tree_rules  # per-tree list of leaf rules
        self.feature_names = feature_names
        self.lambda_ = lambda_
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.M = instance.horizon
        self.big_M_feat = instance.horizon * 2  # for feature constraints

    def build(self):
        m = Model("RF_Embedded_MIP")
        inst = self.inst
        ops = inst.all_ops
        M = self.M
        BigM_feat = self.big_M_feat
        eps = 0.001

        # ---- Standard FJSP variables ----
        x = {}
        s = {}
        c = {}
        delta = {}
        for (j, p) in ops:
            s[(j, p)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"s_{j}_{p}")
            c[(j, p)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"c_{j}_{p}")
            for k in inst.feasible_machines.get((j, p), range(inst.num_machines)):
                x[(j, p, k)] = m.addVar(vtype=GRB.BINARY, name=f"x_{j}_{p}_{k}")

        for k in range(inst.num_machines):
            ops_on_k = [(j, p) for (j, p) in ops
                        if k in inst.feasible_machines.get((j, p), [])]
            for a, b in combinations(ops_on_k, 2):
                delta[(a[0], a[1], b[0], b[1], k)] = m.addVar(vtype=GRB.BINARY)

        Cmax = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Cmax")

        # ---- RF feature variables ----
        z = {}  # feature_name → variable
        for fname in self.feature_names:
            if fname.startswith("load_"):
                z[fname] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"z_{fname}")
            elif fname.startswith("ops_"):
                z[fname] = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"z_{fname}")
            else:
                z[fname] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"z_{fname}")

        # ---- RF leaf activation variables ----
        y = {}  # (t_idx, leaf_id) → binary
        for t_idx, rules in enumerate(self.tree_rules):
            for rule in rules:
                y[(t_idx, rule["leaf_id"])] = m.addVar(vtype=GRB.BINARY,
                    name=f"y_{t_idx}_{rule['leaf_id']}")

        # ---- Surrogate value ----
        T = len(self.tree_rules)
        leaf_exprs = []
        for t_idx, rules in enumerate(self.tree_rules):
            for rule in rules:
                leaf_exprs.append(y[(t_idx, rule["leaf_id"])] * rule["value"])
        Q_global = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_global")
        m.addConstr(Q_global == quicksum(leaf_exprs) / T)

        m.update()

        # ============================================================
        # Constraints
        # ============================================================

        # ---- C1: Assignment uniqueness ----
        for (j, p) in ops:
            feasible = inst.feasible_machines.get((j, p), range(inst.num_machines))
            m.addConstr(quicksum(x[(j, p, k)] for k in feasible) == 1)

        # ---- C2: Completion = start + processing ----
        for (j, p) in ops:
            feasible = inst.feasible_machines.get((j, p), list(range(inst.num_machines)))
            dur_expr = quicksum(
                next(alt[1] for alt in inst.jobs[j][p] if alt[0] == k) * x[(j, p, k)]
                for k in feasible
            )
            m.addConstr(c[(j, p)] == s[(j, p)] + dur_expr)

        # ---- C3: Job precedence ----
        for j_idx, ops_list in inst.jobs.items():
            for p in range(len(ops_list) - 1):
                m.addConstr(s[(j_idx, p + 1)] >= c[(j_idx, p)])

        # ---- C4: Machine non-overlap ----
        for k in range(inst.num_machines):
            ops_on_k = [(j, p) for (j, p) in ops
                        if k in inst.feasible_machines.get((j, p), [])]
            for a, b in combinations(ops_on_k, 2):
                j1, p1 = a
                j2, p2 = b
                d_ab = delta[(j1, p1, j2, p2, k)]
                m.addConstr(
                    s[(j2, p2)] >= c[(j1, p1)] - M * (1 - d_ab)
                    - M * (2 - x[(j1, p1, k)] - x[(j2, p2, k)])
                )
                m.addConstr(
                    s[(j1, p1)] >= c[(j2, p2)] - M * d_ab
                    - M * (2 - x[(j1, p1, k)] - x[(j2, p2, k)])
                )

        # ---- C5: Makespan ----
        for (j, p) in ops:
            m.addConstr(Cmax >= c[(j, p)])

        # ---- C6: Feature linking ----
        for k in range(inst.num_machines):
            # Load_k
            load_expr = quicksum(
                next(alt[1] for alt in inst.jobs[j][p] if alt[0] == k) * x[(j, p, k)]
                for (j, p) in ops
                if k in inst.feasible_machines.get((j, p), [])
            )
            m.addConstr(z[f"load_{k}"] == load_expr)

            # Ops_k
            ops_expr = quicksum(
                x[(j, p, k)]
                for (j, p) in ops
                if k in inst.feasible_machines.get((j, p), [])
            )
            m.addConstr(z[f"ops_{k}"] == ops_expr)

        # ---- C7: Leaf uniqueness ----
        for t_idx in range(len(self.tree_rules)):
            leaf_ids = [rule["leaf_id"] for rule in self.tree_rules[t_idx]]
            m.addConstr(quicksum(y[(t_idx, lid)] for lid in leaf_ids) == 1)

        # ---- C8: Path constraints (big-M for each leaf condition) ----
        for t_idx, rules in enumerate(self.tree_rules):
            for rule in rules:
                y_var = y[(t_idx, rule["leaf_id"])]
                for feat_name, op, threshold in rule["conditions"]:
                    z_feat = z[feat_name]
                    if op == "<=":
                        m.addConstr(z_feat <= threshold + BigM_feat * (1 - y_var))
                    elif op == ">":
                        m.addConstr(z_feat >= threshold + eps - BigM_feat * (1 - y_var))

        # ============================================================
        # Objective
        # ============================================================
        if self.lambda_ == 0:
            m.setObjective(Cmax, GRB.MINIMIZE)
        else:
            m.setObjective(Cmax + self.lambda_ * Q_global, GRB.MINIMIZE)

        return m, x, s, c, delta, Cmax, Q_global, y

    def solve(self):
        m, x, s, c, delta, Cmax, Q_global, y = self.build()

        if self.time_limit is not None:
            m.setParam("TimeLimit", self.time_limit)
        m.setParam("MIPGap", self.mip_gap)
        m.setParam("OutputFlag", 0)

        t0 = time.time()
        m.optimize()
        wall_time = time.time() - t0

        result = {
            "has_solution": m.SolCount > 0,
            "status": m.Status,
            "wall_time": wall_time,
            "node_count": int(m.NodeCount),
        }

        if m.SolCount > 0:
            result["makespan"] = Cmax.X
            result["gap"] = m.MIPGap
            result["rf_surrogate"] = Q_global.X
            result["num_vars"] = m.NumVars
            result["num_constrs"] = m.NumConstrs

            result["assignments"] = []
            for (j, p) in self.inst.all_ops:
                feasible = self.inst.feasible_machines.get((j, p), range(self.inst.num_machines))
                for k in feasible:
                    if x[(j, p, k)].X > 0.5:
                        result["assignments"].append({
                            "job_id": j, "job_pos": p, "machine": k,
                        })
                        break

        return result


# ============================================================
# Compare: λ sweep
# ============================================================

def sweep_lambda(instance: FJSPInstance, tree_rules, feature_names,
                 lambdas: List[float] = None, time_limit: float = 60) -> List[Dict]:
    """Run RF-embedded MIP with different λ values."""
    if lambdas is None:
        lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]

    results = []
    for lam in lambdas:
        label = f"λ={lam}" if lam > 0 else "exact (λ=0)"
        print(f"\n  [{label}]")
        model = RFEmbeddedMIP(instance, tree_rules, feature_names,
                              lambda_=lam, time_limit=time_limit)
        r = model.solve()
        r["lambda"] = lam
        r["label"] = label
        results.append(r)

        if r["has_solution"]:
            print(f"    makespan={r['makespan']:.1f}, rf_surrogate={r.get('rf_surrogate', 0):.1f}, "
                  f"time={r['wall_time']:.1f}s, vars={r['num_vars']}, constrs={r['num_constrs']}")
        else:
            print(f"    NO SOLUTION, time={r['wall_time']:.1f}s")

    return results


if __name__ == "__main__":
    import sys

    inst = FJSPInstance.from_json_file("data/instances/brandimarte/mk01.json")
    print(f"Instance: {inst.name} ({inst.num_jobs}x{inst.num_machines}, "
          f"{inst.total_ops} ops, best_known={inst.best_known})")

    # Train compact RF
    rf, tree_rules, feats = train_small_rf_for_mip(inst, n_trees=10, max_depth=5)

    print(f"\nRF: {len(tree_rules)} trees, {sum(len(rules) for rules in tree_rules)} total leaves")

    # λ sweep
    print(f"\n{'='*60}")
    print("LAMBDA SWEEP")
    print(f"{'='*60}")

    results = sweep_lambda(inst, tree_rules, feats,
                           lambdas=[0.0, 0.001, 0.01, 0.1, 1.0],
                           time_limit=60)

    # Also run exact MIP baseline for reference
    from fjsp_baseline import PureFJSPMakespanModel
    print(f"\n  [Exact MIP baseline]")
    baseline = PureFJSPMakespanModel(inst, time_limit=60).solve()
    print(f"    makespan={baseline.get('makespan')}, time={baseline['wall_time']:.1f}s")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'λ':>8s} {'Makespan':>10s} {'RF Surr':>10s} {'Time':>8s} {'Vars':>8s} {'Constrs':>8s}")
    print("-" * 62)
    for r in results:
        mk = r.get('makespan', 'N/A')
        rf_val = r.get('rf_surrogate', 'N/A')
        print(f"{r['lambda']:8.3f} {str(mk):>10s} {str(rf_val):>10s} "
              f"{r['wall_time']:7.1f}s {r['num_vars']:8d} {r['num_constrs']:8d}")
    print(f"{'exact':>8s} {baseline.get('makespan', 'N/A'):>10s} {'--':>10s} "
          f"{baseline['wall_time']:7.1f}s")
