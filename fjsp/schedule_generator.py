"""
Generate diverse feasible FJSP schedules for RF training data.
Uses Serial Schedule Generation Scheme (SGS) with randomized dispatching rules.
"""
import json
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from fjsp_baseline import FJSPInstance


class SerialSGS:
    """Serial Schedule Generation Scheme for FJSP.

    For each step:
      1. Select a ready operation (all predecessors completed)
      2. Select a machine from feasible set
      3. Schedule it as early as possible on that machine
    """

    RULES = ["random", "spt", "lpt", "mwr", "lwr", "fifo"]
    MACHINE_RULES = ["random", "shortest", "longest", "least_load", "mixed"]

    def __init__(self, instance: FJSPInstance, seed: int = None):
        self.inst = instance
        self.rng = random.Random(seed)

    def generate_one(self, op_rule: str = None, mach_rule: str = None) -> Dict:
        """Generate one complete schedule, return all data."""
        if op_rule is None:
            op_rule = self.rng.choice(self.RULES)
        if mach_rule is None:
            mach_rule = self.rng.choice(self.MACHINE_RULES)

        inst = self.inst

        # State
        machine_available = [0.0] * inst.num_machines  # when machine becomes free
        job_next_op = [0] * inst.num_jobs  # next operation index to process
        job_ready_time = [0.0] * inst.num_jobs  # when job becomes ready

        assignment = {}  # (j,p) -> k
        start_times = {}
        completion_times = {}
        op_order = []

        # Ready operations: (j, p) where p == job_next_op[j] and all earlier ops done
        pending = set()
        for j in range(inst.num_jobs):
            if inst.num_ops_per_job[j] > 0:
                pending.add((j, 0))

        while pending:
            # --- Step 1: Select operation ---
            ready_list = list(pending)
            selected_op = self._select_operation(ready_list, op_rule, job_ready_time)

            j, p = selected_op
            pending.discard(selected_op)

            # --- Step 2: Select machine ---
            feasible = inst.feasible_machines.get((j, p), list(range(inst.num_machines)))
            selected_mach = self._select_machine(feasible, inst.jobs[j][p], mach_rule, machine_available)

            # --- Step 3: Schedule ---
            proc_time = None
            for alt in inst.jobs[j][p]:
                if alt[0] == selected_mach:
                    proc_time = alt[1]
                    break

            # Earliest start = max(machine_free, job_ready)
            est = max(machine_available[selected_mach], job_ready_time[j])
            start_times[(j, p)] = est
            completion_times[(j, p)] = est + proc_time

            # Update state
            machine_available[selected_mach] = est + proc_time
            job_ready_time[j] = est + proc_time
            assignment[(j, p)] = selected_mach
            op_order.append(((j, p), selected_mach, est, est + proc_time))

            # --- Step 4: Add next operations ---
            next_p = p + 1
            if next_p < inst.num_ops_per_job[j]:
                pending.add((j, next_p))

        makespan = max(completion_times.values()) if completion_times else 0

        # Compute assignment-level features (not sequencing)
        features = self._extract_features(assignment, makespan)

        return {
            "makespan": makespan,
            "assignment": {f"({j},{p})": k for (j, p), k in assignment.items()},
            "features": features,
            "num_used_machines": len(set(assignment.values())),
            "op_rule": op_rule,
            "mach_rule": mach_rule,
        }

    def _select_operation(self, ready, rule, job_ready):
        if rule == "random":
            return self.rng.choice(ready)
        elif rule == "fifo":
            return ready[0]
        elif rule == "spt":
            # Operation with shortest min processing time
            def min_pt(op):
                j, p = op
                return min(alt[1] for alt in self.inst.jobs[j][p])
            return min(ready, key=min_pt)
        elif rule == "lpt":
            def max_pt(op):
                j, p = op
                return max(alt[1] for alt in self.inst.jobs[j][p])
            return max(ready, key=max_pt)
        elif rule == "mwr":
            # Most work remaining
            def work_rem(op):
                j, p = op
                return sum(
                    sum(alt[1] for alt in self.inst.jobs[j][pp])
                    for pp in range(p, self.inst.num_ops_per_job[j])
                )
            return max(ready, key=work_rem)
        elif rule == "lwr":
            def work_rem(op):
                j, p = op
                return sum(
                    sum(alt[1] for alt in self.inst.jobs[j][pp])
                    for pp in range(p, self.inst.num_ops_per_job[j])
                )
            return min(ready, key=work_rem)
        return self.rng.choice(ready)

    def _select_machine(self, feasible, alts, rule, machine_avail):
        """alts = [[m, t], [m, t], ...]"""
        alt_dict = {m: t for m, t in alts if m in feasible}

        if rule == "random":
            return self.rng.choice(feasible)
        elif rule == "shortest":
            return min(feasible, key=lambda k: alt_dict.get(k, 99999))
        elif rule == "longest":
            return max(feasible, key=lambda k: alt_dict.get(k, 0))
        elif rule == "least_load":
            return min(feasible, key=lambda k: machine_avail[k])
        elif rule == "mixed":
            return self.rng.choice([
                min(feasible, key=lambda k: alt_dict.get(k, 99999)),
                min(feasible, key=lambda k: machine_avail[k]),
            ])
        return self.rng.choice(feasible)

    def _extract_features(self, assignment, makespan):
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

        # --- New aggregate features ---
        features["num_used_machines"] = sum(used_k.values())

        # Bottleneck machine load (= strongest simple makespan lower bound)
        features["max_machine_load"] = max(load_k.values())

        # Load imbalance: max - min across USED machines
        used_loads = [load_k[k] for k in range(inst.num_machines) if used_k[k]]
        features["load_imbalance"] = max(used_loads) - min(used_loads) if used_loads else 0

        # Max operations on any machine (sequencing bottleneck)
        features["max_ops_on_machine"] = max(ops_k.values())

        # Max job total time (= another makespan lower bound)
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

        # Machine load std dev
        if used_loads:
            mean_load = sum(used_loads) / len(used_loads)
            var = sum((l - mean_load) ** 2 for l in used_loads) / len(used_loads)
            features["load_std"] = var ** 0.5
        else:
            features["load_std"] = 0

        return features


def generate_dataset(instance: FJSPInstance, n_samples: int = 5000,
                     seed: int = 42, verbose: bool = True) -> List[Dict]:
    """Generate N diverse feasible schedules for RF training."""
    rng = random.Random(seed)
    sgs = SerialSGS(instance, seed=seed)
    seen_assignments = set()
    data = []

    attempts = 0
    max_attempts = n_samples * 10

    while len(data) < n_samples and attempts < max_attempts:
        op_rule = rng.choice(SerialSGS.RULES)
        mach_rule = rng.choice(SerialSGS.MACHINE_RULES)
        schedule = sgs.generate_one(op_rule=op_rule, mach_rule=mach_rule)

        # Deduplicate by assignment signature
        sig = tuple(sorted(schedule["assignment"].items()))
        if sig not in seen_assignments:
            seen_assignments.add(sig)
            data.append(schedule)

        attempts += 1

        if verbose and len(data) % 1000 == 0:
            print(f"  Generated {len(data)} unique schedules...")

    print(f"Generated {len(data)} unique schedules from {attempts} attempts")
    return data


if __name__ == "__main__":
    inst = FJSPInstance.from_json_file("data/instances/brandimarte/mk01.json")
    print(f"Generating training data for {inst.name}...")
    t0 = time.time()
    data = generate_dataset(inst, n_samples=500)
    print(f"Done in {time.time()-t0:.1f}s, {len(data)} samples")
    print(f"Makespan range: {min(d['makespan'] for d in data):.0f} - {max(d['makespan'] for d in data):.0f}")
