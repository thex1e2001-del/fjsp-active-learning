"""
Fast heuristic sequencer for fixed-assignment FJSP scheduling.
Evaluates an assignment's makespan in milliseconds using dispatching rules,
enabling RF to explore thousands of candidate assignments.
"""
from typing import Dict, List, Tuple
import random


class HeuristicSequencer:
    """Constructive heuristic scheduler for a fixed machine assignment.

    Given assignment {(j,p): k}, schedule all operations using dispatching rules.
    Does NOT use MIP — runs in O(n_ops * n_machines) time.
    """

    RULES = ["est", "spt", "lpt", "mwr", "fifo"]

    def __init__(self, instance, seed: int = 42):
        self.inst = instance
        self.rng = random.Random(seed)

    def schedule(self, assignment: Dict[Tuple[int, int], int],
                 rule: str = "best") -> Tuple[float, str]:
        """
        Schedule all operations given fixed assignment.

        Args:
            assignment: {(j,p): machine_k}
            rule: dispatching rule, or "best" to try all and return best

        Returns:
            (makespan, rule_used)
        """
        if rule == "best":
            best_mk = float("inf")
            best_rule = "est"
            for r in self.RULES:
                mk = self._schedule_one(assignment, r)
                if mk < best_mk:
                    best_mk = mk
                    best_rule = r
            return best_mk, best_rule
        else:
            return self._schedule_one(assignment, rule), rule

    def _schedule_one(self, assignment: Dict[Tuple[int, int], int],
                      rule: str) -> float:
        """Execute one pass of constructive scheduling."""
        inst = self.inst

        # State
        machine_avail = {k: 0.0 for k in range(inst.num_machines)}
        job_ready = {j: 0.0 for j in range(inst.num_jobs)}
        job_next_op = {j: 0 for j in range(inst.num_jobs)}

        # Group operations by machine
        machine_ops = {k: [] for k in range(inst.num_machines)}
        for (j, p), k in assignment.items():
            dur = self._get_duration(j, p, k)
            machine_ops[k].append((j, p, dur))

        # Track completed operations per job
        completed = set()

        while True:
            # Find ready operations
            ready = []
            for k, ops in machine_ops.items():
                for (j, p, dur) in ops:
                    if (j, p) in completed:
                        continue
                    # Check precedence: all earlier ops of this job must be done
                    all_prev_done = True
                    for pp in range(p):
                        if (j, pp) not in completed:
                            all_prev_done = False
                            break
                    if all_prev_done:
                        ready.append((j, p, k, dur))

            if not ready:
                break

            # Pick operations using rule
            if rule == "est":
                # Earliest start time: pick op that can start earliest
                # = one with min max(machine_avail, job_ready)
                best_op = min(ready, key=lambda x: max(machine_avail[x[2]], job_ready[x[0]]))
            elif rule == "spt":
                best_op = min(ready, key=lambda x: x[3])
            elif rule == "lpt":
                best_op = max(ready, key=lambda x: x[3])
            elif rule == "mwr":
                # Most work remaining for the job
                def work_rem(op):
                    j, p, k, dur = op
                    return sum(
                        self._get_duration(j, pp, assignment.get((j, pp), 0))
                        for pp in range(p, inst.num_ops_per_job.get(j, 0))
                    )
                best_op = max(ready, key=work_rem)
            elif rule == "fifo":
                best_op = ready[0]
            else:
                best_op = self.rng.choice(ready)

            j, p, k, dur = best_op

            # Schedule
            start = max(machine_avail[k], job_ready[j])
            finish = start + dur

            machine_avail[k] = finish
            job_ready[j] = finish
            completed.add((j, p))

        makespan = max(machine_avail.values()) if machine_avail else 0
        return makespan

    def _get_duration(self, j: int, p: int, k: int) -> float:
        """Get processing time of operation (j,p) on machine k."""
        for alt in self.inst.jobs[j][p]:
            if alt[0] == k:
                return alt[1]
        return 0


def evaluate_assignment_fast(instance, assignment, sequencer=None, rule="best"):
    """Fast assignment evaluation using heuristic sequencing."""
    if sequencer is None:
        sequencer = HeuristicSequencer(instance)
    return sequencer.schedule(assignment, rule)


if __name__ == "__main__":
    import time
    from fjsp_baseline import FJSPInstance, FixedAssignmentSequencingModel
    from assignment_generators import AssignmentGenerator

    inst = FJSPInstance.from_json_file("data/instances/brandimarte/mk01.json")
    gen = AssignmentGenerator(inst, seed=42)

    # Test balanced assignment
    assign = gen.balanced_assignment()
    seq = HeuristicSequencer(inst)

    # Heuristic
    t0 = time.time()
    h_mk, h_rule = seq.schedule(assign, "best")
    h_time = time.time() - t0
    print(f"Heuristic: makespan={h_mk:.1f} (rule={h_rule}), time={h_time*1000:.1f}ms")

    for rule in HeuristicSequencer.RULES:
        mk, _ = seq.schedule(assign, rule)
        print(f"  {rule}: {mk:.1f}")

    # Exact
    print("\nExact sequencing...")
    exact = FixedAssignmentSequencingModel(inst, assign, time_limit=30)
    result = exact.solve()
    print(f"Exact: makespan={result.get('makespan', 'N/A')}, time={result['wall_time']:.2f}s")
