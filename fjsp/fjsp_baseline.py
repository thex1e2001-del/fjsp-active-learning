"""
Exact FJSP MIP baseline + fixed-assignment sequencing model.
Uses Gurobi to solve:
  - Full FJSP (min makespan, routing + sequencing)  → baseline
  - Fixed-assignment sequencing (given routing, min makespan) → diagnostic
"""
import json
import time
from itertools import combinations
from typing import Dict, Tuple, List, Optional

from gurobipy import Model, GRB, quicksum


class FJSPInstance:
    """Load and represent an FJSP instance from JSON."""

    def __init__(self, data: dict):
        self.name = data.get("name", "unknown")
        self.num_jobs = data["num_jobs"]
        self.num_machines = data["num_machines"]
        self.horizon = data["horizon"]
        self.best_known = data.get("best_known_makespan")

        self.jobs: Dict[int, List[List[List[int]]]] = {}
        for j_str, ops in data["jobs"].items():
            self.jobs[int(j_str)] = [
                [[m, t] for (m, t) in alt] for alt in ops
            ]

        self.feasible_machines: Dict[Tuple[int, int], List[int]] = {}
        for key, machines in data.get("feasible_machines", {}).items():
            # key is "(j,p)"
            self.feasible_machines[eval(key)] = machines

    @classmethod
    def from_json_file(cls, path: str) -> "FJSPInstance":
        with open(path) as f:
            return cls(json.load(f))

    @property
    def all_ops(self) -> List[Tuple[int, int]]:
        ops = []
        for j, ops_list in self.jobs.items():
            for p in range(len(ops_list)):
                ops.append((j, p))
        return ops

    @property
    def total_ops(self) -> int:
        return sum(len(ops) for ops in self.jobs.values())

    @property
    def num_ops_per_job(self) -> Dict[int, int]:
        return {j: len(ops) for j, ops in self.jobs.items()}


# ============================================================
# Pure FJSP Makespan Model (Baseline)
# ============================================================

class PureFJSPMakespanModel:
    """Standard exact FJSP MIP: min Cmax, simultaneous routing + sequencing."""

    def __init__(self, instance: FJSPInstance, time_limit: float = None, mip_gap: float = 0.0):
        self.inst = instance
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.M = instance.horizon

    def build(self):
        m = Model("PureFJSP")
        inst = self.inst
        ops = inst.all_ops
        M = self.M

        # Variables
        x = {}   # x[j,p,k] : assignment
        s = {}   # start time
        c = {}   # completion time
        delta = {}  # sequencing on same machine

        for (j, p) in ops:
            s[(j, p)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"s_{j}_{p}")
            c[(j, p)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"c_{j}_{p}")
            for k in inst.feasible_machines.get((j, p), range(inst.num_machines)):
                x[(j, p, k)] = m.addVar(vtype=GRB.BINARY, name=f"x_{j}_{p}_{k}")

        # Sequencing variables: delta[(j1,p1,j2,p2,k)] = 1 if (j1,p1) before (j2,p2) on machine k
        for k in range(inst.num_machines):
            ops_on_k = [(j, p) for (j, p) in ops if k in inst.feasible_machines.get((j, p), [])]
            for a, b in combinations(ops_on_k, 2):
                delta[(a[0], a[1], b[0], b[1], k)] = m.addVar(vtype=GRB.BINARY)

        Cmax = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Cmax")
        m.update()

        # Constraints
        # C1: Each operation assigned to exactly one feasible machine
        for (j, p) in ops:
            feasible = inst.feasible_machines.get((j, p), range(inst.num_machines))
            m.addConstr(quicksum(x[(j, p, k)] for k in feasible) == 1, f"assign_{j}_{p}")

        # C2: Completion time = start + processing time
        for (j, p) in ops:
            feasible = inst.feasible_machines.get((j, p), range(inst.num_machines))
            dur_expr = quicksum(inst.jobs[j][p][idx][1] * x[(j, p, k)]
                                for idx, k in enumerate(feasible))
            m.addConstr(c[(j, p)] == s[(j, p)] + dur_expr, f"dur_{j}_{p}")

        # C3: Job precedence
        for j, ops_list in inst.jobs.items():
            for p in range(len(ops_list) - 1):
                m.addConstr(s[(j, p + 1)] >= c[(j, p)], f"prec_{j}_{p}")

        # C4: Machine non-overlap
        for k in range(inst.num_machines):
            ops_on_k = [(j, p) for (j, p) in ops if k in inst.feasible_machines.get((j, p), [])]
            for a, b in combinations(ops_on_k, 2):
                j1, p1 = a
                j2, p2 = b
                d = delta[(j1, p1, j2, p2, k)]
                m.addConstr(s[(j2, p2)] >= c[(j1, p1)] - M * (1 - d) - M * (2 - x[(j1, p1, k)] - x[(j2, p2, k)]),
                            f"seq1_{j1}_{p1}_{j2}_{p2}_{k}")
                m.addConstr(s[(j1, p1)] >= c[(j2, p2)] - M * d - M * (2 - x[(j1, p1, k)] - x[(j2, p2, k)]),
                            f"seq2_{j1}_{p1}_{j2}_{p2}_{k}")

        # C5: Makespan
        for (j, p) in ops:
            m.addConstr(Cmax >= c[(j, p)], f"mk_{j}_{p}")

        m.setObjective(Cmax, GRB.MINIMIZE)

        return m, x, s, c, delta, Cmax

    def solve(self):
        m, x, s, c, delta, Cmax = self.build()

        m.setParam("OutputFlag", 0)
        if self.time_limit is not None:
            m.setParam("TimeLimit", self.time_limit)
        m.setParam("MIPGap", self.mip_gap)

        t0 = time.time()
        m.optimize()
        wall_time = time.time() - t0

        result = {
            "has_solution": m.SolCount > 0,
            "status": m.Status,
            "status_code": int(m.Status),
            "sol_count": int(m.SolCount),
            "wall_time": wall_time,
        }

        if m.SolCount > 0:
            result["makespan"] = Cmax.X
            result["gap"] = m.MIPGap
        # ObjBound is available for feasible time-limited runs and is useful
        # for reporting what the solver actually certified.
        if m.SolCount > 0 or m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            result["best_bound"] = float(m.ObjBound)
            result["node_count"] = int(m.NodeCount)
            result["assignments"] = []
            for (j, p) in self.inst.all_ops:
                feasible = self.inst.feasible_machines.get((j, p), range(self.inst.num_machines))
                for k in feasible:
                    if x[(j, p, k)].X > 0.5:
                        result["assignments"].append({
                            "job_id": j, "job_pos": p, "machine": k,
                            "start": s[(j, p)].X, "completion": c[(j, p)].X,
                        })
                        break

        return result


# ============================================================
# Fixed-Assignment Sequencing Model
# ============================================================

class FixedAssignmentSequencingModel:
    """Given a fixed machine assignment, solve the exact sequencing subproblem."""

    def __init__(self, instance: FJSPInstance, assignment: Dict[Tuple[int, int], int],
                 time_limit: float = None, mip_gap: float = 0.0):
        self.inst = instance
        self.assignment = assignment  # {(j,p): k}
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.M = instance.horizon

    def _pairs_same_machine(self):
        pairs = {}
        for k in range(self.inst.num_machines):
            ops_k = [(j, p) for (j, p), kk in self.assignment.items() if kk == k]
            pairs[k] = list(combinations(ops_k, 2))
        return pairs

    def build(self):
        m = Model("FixedAssignmentSeq")
        inst = self.inst
        ops = inst.all_ops
        M = self.M
        pairs_by_machine = self._pairs_same_machine()

        s = {(j, p): m.addVar(lb=0, vtype=GRB.CONTINUOUS) for (j, p) in ops}
        c = {(j, p): m.addVar(lb=0, vtype=GRB.CONTINUOUS) for (j, p) in ops}

        delta = {}
        for k, prs in pairs_by_machine.items():
            for (a, b) in prs:
                j1, p1 = a
                j2, p2 = b
                delta[(j1, p1, j2, p2, k)] = m.addVar(vtype=GRB.BINARY)

        Cmax = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
        m.update()

        # Completion = start + processing time
        for (j, p) in ops:
            k = self.assignment[(j, p)]
            dur = inst.jobs[j][p][0][1]  # fallback
            for alt in inst.jobs[j][p]:
                if alt[0] == k:
                    dur = alt[1]
                    break
            m.addConstr(c[(j, p)] == s[(j, p)] + dur)

        # Job precedence
        for j, ops_list in inst.jobs.items():
            for p in range(len(ops_list) - 1):
                m.addConstr(s[(j, p + 1)] >= c[(j, p)])

        # Machine non-overlap
        for k, prs in pairs_by_machine.items():
            for (a, b) in prs:
                j1, p1 = a
                j2, p2 = b
                d = delta[(j1, p1, j2, p2, k)]
                m.addConstr(s[(j2, p2)] >= c[(j1, p1)] - M * (1 - d))
                m.addConstr(s[(j1, p1)] >= c[(j2, p2)] - M * d)

        # Makespan
        for (j, p) in ops:
            m.addConstr(Cmax >= c[(j, p)])

        m.setObjective(Cmax, GRB.MINIMIZE)

        return m, s, c, Cmax

    def solve(self):
        m, s, c, Cmax = self.build()

        m.setParam("OutputFlag", 0)
        if self.time_limit is not None:
            m.setParam("TimeLimit", self.time_limit)
        m.setParam("MIPGap", self.mip_gap)

        t0 = time.time()
        m.optimize()
        wall_time = time.time() - t0

        result = {
            "has_solution": m.SolCount > 0,
            "status": m.Status,
            "status_code": int(m.Status),
            "sol_count": int(m.SolCount),
            "wall_time": wall_time,
        }

        if m.SolCount > 0:
            result["makespan"] = Cmax.X
            result["gap"] = m.MIPGap
        if m.SolCount > 0 or m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            result["best_bound"] = float(m.ObjBound)

        return result


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/instances/brandimarte/mk01.json"
    inst = FJSPInstance.from_json_file(path)
    print(f"Instance: {inst.name}, {inst.num_jobs}x{inst.num_machines}, ops={inst.total_ops}")
    print(f"Best known: {inst.best_known}")

    print("\n--- Baseline (exact FJSP) ---")
    result = PureFJSPMakespanModel(inst, time_limit=60).solve()
    print(f"  makespan={result.get('makespan')}, time={result['wall_time']:.2f}s, gap={result.get('gap', 0):.4f}")
