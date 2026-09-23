import json

with open("results/exp1_main.json") as f:
    data = json.load(f)

print(f"{'Inst':<8s} {'Random':>7s} {'SPT':>7s} {'LPT':>7s} {'Bal':>7s} {'RF':>7s} {'Win':>6s} {'Gap%':>7s}")
print("-" * 62)
rf_wins = bal_wins = draw = 0
for r in data:
    bal = r.get("balanced") or 0
    rf = r.get("rf_best") or 0
    gap = ((rf - bal) / bal * 100) if bal else 0
    w = "RF" if rf < bal else ("Bal" if rf > bal else "Draw")
    if rf < bal: rf_wins += 1
    elif rf > bal: bal_wins += 1
    else: draw += 1
    print(f"{r['instance']:<8s} {str(r.get('random','')):>7s} {str(r.get('spt','')):>7s} "
          f"{str(r.get('lpt','')):>7s} {str(bal):>7s} {str(rf):>7s} {w:>6s} {gap:+6.1f}%")
print(f"\nRF: {rf_wins}, Balanced: {bal_wins}, Draw: {draw}")
print(f"RF beats Balanced on {rf_wins}/{len(data)} instances")
