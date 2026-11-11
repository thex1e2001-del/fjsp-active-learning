"""Independent checks of explicit schedules; no solver/heuristic dependencies."""
import math


def validate_assignment(instance, assignment):
    expected = {(j, p) for j, ops in instance.jobs.items() for p in range(len(ops))}
    if set(assignment) != expected:
        raise ValueError('assignment must contain exactly every operation')
    for (j, p), k in assignment.items():
        if k not in dict(instance.jobs[j][p]):
            raise ValueError(f'ineligible machine {k} for {(j, p)}')


def validate_schedule(instance, rows, makespan=None, assignment=None, tolerance=1e-5):
    """Raise on missing/duplicate ops, bad durations, overlap or precedence.

    Rows use job_id, job_pos, machine, start, completion. Objective must equal
    the maximum completion (within an absolute numerical tolerance).
    """
    expected = {(j, p) for j, ops in instance.jobs.items() for p in range(len(ops))}
    by_op = {}
    machines = {k: [] for k in range(instance.num_machines)}
    if assignment is not None:
        validate_assignment(instance, assignment)
    for row in rows:
        op = (row['job_id'], row['job_pos'])
        k, start, end = row['machine'], float(row['start']), float(row['completion'])
        if op not in expected or op in by_op:
            raise ValueError(f'unknown/duplicate operation {op}')
        durations = dict(instance.jobs[op[0]][op[1]])
        if k not in durations or (assignment is not None and assignment[op] != k):
            raise ValueError(f'invalid machine for {op}')
        if not math.isfinite(start) or not math.isfinite(end) or start < -tolerance:
            raise ValueError(f'invalid time for {op}')
        if abs(end - start - durations[k]) > tolerance:
            raise ValueError(f'wrong duration for {op}')
        by_op[op] = (start, end)
        machines[k].append((start, end, op))
    if set(by_op) != expected:
        raise ValueError('missing operations')
    for j, ops in instance.jobs.items():
        for p in range(1, len(ops)):
            if by_op[j, p][0] < by_op[j, p-1][1] - tolerance:
                raise ValueError(f'job precedence violation {(j, p)}')
    for k, operations in machines.items():
        operations.sort()
        for a, b in zip(operations, operations[1:]):
            if b[0] < a[1] - tolerance:
                raise ValueError(f'machine overlap on {k}: {a[2]}, {b[2]}')
    actual = max(end for start, end in by_op.values())
    if makespan is not None and (not math.isfinite(makespan) or abs(actual-makespan) > tolerance):
        raise ValueError('objective does not equal schedule makespan')
    return actual
