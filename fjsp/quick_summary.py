"""Quick summary of batch results."""
import csv
from collections import defaultdict

rows = []
with open("results/summary.csv") as f:
    for r in csv.DictReader(f):
        if r.get("makespan") and r["makespan"] != "None":
            rows.append(r)

groups = defaultdict(list)
for r in rows:
    groups[(r["instance"], r["lambda"])].append(float(r["makespan"]))

baselines = {}
with open("results/baselines.csv") as f:
    for r in csv.DictReader(f):
        baselines[r["instance"]] = r

# Get best_known from summary
best_known = {}
for r in rows:
    if r["instance"] not in best_known:
        best_known[r["instance"]] = r.get("best_known", "")

def fmt(v):
    if v is None: return "--".rjust(9)
    return f"{v:9.0f}"

def fmt_s(v):
    if v is None: return "--".rjust(9)
    return f"{str(v):>9s}"

header = f"{'Instance':<8s} {'lambda=0':>9s} {'l=0.001':>9s} {'l=0.01':>9s} {'Balanced':>9s} {'ExactBL':>9s} {'BestK':>7s} {'Winner':>10s}"
print(header)
print("-" * len(header))

rf_wins = 0
exact_wins = 0
draws = 0
fail = 0
total_valid = 0

for inst in sorted(set(r["instance"] for r in rows)):
    vals = {}
    for lam in ["0.0", "0.001", "0.01"]:
        g = groups.get((inst, lam), [])
        vals[lam] = sum(g) / len(g) if g else None

    bl = baselines.get(inst, {})
    exact_bl = bl.get("exact_makespan", "--")
    balanced = bl.get("balanced_makespan", "--")
    bk = best_known.get(inst, "")

    best_rf = min(v for v in [vals["0.001"], vals["0.01"]] if v is not None) if (vals["0.001"] or vals["0.01"]) else None
    exact_emb = vals["0.0"]
    winner = "--"

    if best_rf is not None and exact_emb is not None:
        total_valid += 1
        diff = exact_emb - best_rf
        if diff > 1:
            winner = "RF"
            rf_wins += 1
        elif diff < -1:
            winner = "Exact"
            exact_wins += 1
        else:
            winner = "Draw"
            draws += 1
    elif best_rf is None and exact_emb is not None:
        fail += 1
        winner = "FAIL(RF)"

    print(f"{inst:<8s} {fmt(vals['0.0'])} {fmt(vals['0.001'])} {fmt(vals['0.01'])} "
          f"{fmt_s(balanced)} {fmt_s(exact_bl)} {str(bk):>7s} {winner:>10s}")

print()
print(f"RF wins: {rf_wins}, Exact wins: {exact_wins}, Draws: {draws}, RF no-sol: {fail} (on {len(rows)} total runs)")
