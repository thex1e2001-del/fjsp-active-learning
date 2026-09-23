"""
Phase 2-4 consolidation: Finish ALL remaining experiments efficiently.
"""
import json, os, sys, time, random as _r
from pathlib import Path
from typing import Dict, List
import numpy as np
from scipy import stats as scipy_stats

from fjsp_baseline import FJSPInstance, FixedAssignmentSequencingModel
from assignment_generators import AssignmentGenerator
from heuristic_sequencer import HeuristicSequencer
from experiments_final import train_rf_fast, beam_search

SEED = 42
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
ALL_INSTANCES = [f"data/instances/brandimarte/mk{i:02d}.json" for i in range(1, 11)]
SEQ_TIMEOUT = 30
TOP_K = 8


# ═══════════════════════════════════════════════════
# Phase 2: Complete mk09 fair baselines
# ═══════════════════════════════════════════════════

def complete_mk09():
    print("=" * 50)
    print("Phase 2: Complete mk09 fair baselines")
    print("=" * 50)
    path = "data/instances/brandimarte/mk09.json"
    inst = FJSPInstance.from_json_file(path)
    gen = AssignmentGenerator(inst, seed=SEED)
    seq = HeuristicSequencer(inst, seed=SEED)
    ops = inst.all_ops
    row = {"instance": "mk09", "best_known": inst.best_known}

    # Balanced
    bal_a = gen.balanced_assignment()
    bal = FixedAssignmentSequencingModel(inst, bal_a, time_limit=SEQ_TIMEOUT).solve()
    row["balanced"] = bal.get("makespan")
    print(f"  Balanced: {row['balanced']:.0f}")

    # Random-MS
    n_random = min(3000, inst.total_ops * 20)
    rand_pool = []
    for _ in range(n_random):
        a = gen.random_assignment(); h_mk, _ = seq.schedule(a, "best"); rand_pool.append((h_mk, a))
    rand_pool.sort(key=lambda x: x[0])
    best = float("inf")
    for _, a in rand_pool[:TOP_K]:
        exact = FixedAssignmentSequencingModel(inst, a, time_limit=SEQ_TIMEOUT, mip_gap=0.0).solve()
        if exact.get("has_solution") and exact["makespan"] < best: best = exact["makespan"]
    row["random_ms"] = best
    print(f"  Random-MS: {best:.0f}")

    # MaxLoad beam
    pool = []
    seen = set()
    for name, a in gen.generate_all_heuristic().items():
        sig = tuple(sorted(a.items()))
        if sig not in seen: seen.add(sig); pool.append(a)
    for _ in range(5):
        candidates = list(pool)
        for current in pool:
            for (j,p) in ops:
                feasible = inst.feasible_machines.get((j,p), list(range(inst.num_machines)))
                for new_k in feasible:
                    if new_k == current.get((j,p)): continue
                    c = dict(current); c[(j,p)] = new_k
                    sig = tuple(sorted(c.items()))
                    if sig not in seen and len(candidates) < 3000: seen.add(sig); candidates.append(c)
        # Score by max_load (lower better)
        def maxload(a):
            loads = {}
            for (jp_j, jp_p), mk in a.items():
                for alt in inst.jobs[jp_j][jp_p]:
                    if alt[0] == mk: loads[mk] = loads.get(mk,0) + alt[1]; break
            return -max(loads.values()) if loads else 0
        candidates.sort(key=maxload, reverse=True)
        pool = candidates[:15]
    best = float("inf")
    for a in pool[:TOP_K]:
        exact = FixedAssignmentSequencingModel(inst, a, time_limit=SEQ_TIMEOUT).solve()
        if exact.get("has_solution") and exact["makespan"] < best: best = exact["makespan"]
    row["maxload_beam"] = best
    print(f"  MaxLoad beam: {best:.0f}")

    # RF beam
    rf, feats, bp, _ = train_rf_fast(inst)
    ranked = beam_search(inst, bp)
    best = float("inf")
    for _, a in ranked[:TOP_K]:
        exact = FixedAssignmentSequencingModel(inst, a, time_limit=SEQ_TIMEOUT, mip_gap=0.0).solve()
        if exact.get("has_solution") and exact["makespan"] < best: best = exact["makespan"]
    row["rf_beam"] = best
    print(f"  RF beam: {best:.0f}")

    # Load existing results and merge
    fair_path = os.path.join(RESULTS_DIR, "fair_baselines.json")
    if os.path.exists(fair_path):
        with open(fair_path) as f: existing = json.load(f)
        existing = [r for r in existing if r["instance"] != "mk09"]
        existing.append(row)
    else:
        existing = [row]
    with open(fair_path, "w") as f: json.dump(existing, f, indent=2)
    print(f"  Merged into fair_baselines.json")
    return row


# ═══════════════════════════════════════════════════
# Phase 3: Feature ablation + surrogate comparison
# ═══════════════════════════════════════════════════

SIMPLE_FEATURES = ["load", "ops", "used"]  # per-machine only
FULL_FEATURES = ["load", "ops", "used", "agg"]  # + max_load, imbalance, etc.

def run_ablation():
    print("\n" + "=" * 50)
    print("Phase 3: Feature & Surrogate Ablation")
    print("=" * 50)

    instances = [
        "data/instances/brandimarte/mk01.json",
        "data/instances/brandimarte/mk04.json",
        "data/instances/brandimarte/mk07.json",
    ]

    surrogates = ["MaxLoad", "LinearReg", "RF"]
    results = []

    for path in instances:
        inst_name = Path(path).stem
        inst = FJSPInstance.from_json_file(path)
        gen = AssignmentGenerator(inst, seed=SEED)
        seq = HeuristicSequencer(inst, seed=SEED)
        ops = inst.all_ops
        feats = [f"load_{k}" for k in range(inst.num_machines)] + [f"ops_{k}" for k in range(inst.num_machines)]
        print(f"\n  [{inst_name}]")

        # Generate training data
        balanced = gen.balanced_assignment()
        train_assigns = [balanced]
        for _ in range(800):
            perturbed = dict(balanced)
            for _ in range(_r.randint(1, max(2, inst.total_ops // 15))):
                j, p = _r.choice(ops)
                feasible = inst.feasible_machines.get((j, p), list(range(inst.num_machines)))
                alts = [k for k in feasible if k != perturbed.get((j, p))]
                if alts: perturbed[(j, p)] = _r.choice(alts)
            train_assigns.append(perturbed)

        h_labels = [seq.schedule(a, "best")[0] for a in train_assigns]
        from rf_train import train_global_rf

        for surr_name in surrogates:
            if surr_name == "MaxLoad":
                # Score = -max_load
                def scorer(a):
                    loads = {}
                    for (jp_j, jp_p), mk in a.items():
                        for alt in inst.jobs[jp_j][jp_p]:
                            if alt[0] == mk: loads[mk] = loads.get(mk,0) + alt[1]; break
                    return -max(loads.values()) if loads else 0
            elif surr_name == "LinearReg":
                from sklearn.linear_model import LinearRegression
                X = np.array([[a.get((j,p),0) for (j,p) in ops] for a in train_assigns])
                lr = LinearRegression().fit(X, h_labels)
                def scorer(a): return -lr.predict([[a.get((j,p),0) for (j,p) in ops]])[0]
            elif surr_name == "RF":
                schedules = [{"makespan": h_labels[i], "features": {}} for i in range(len(train_assigns))]
                rf, _ = train_global_rf(schedules, feats, n_estimators=50, max_depth=8)
                def scorer(a):
                    f = {}; loads = {}
                    for (jp_j, jp_p), mk in a.items():
                        for alt in inst.jobs[jp_j][jp_p]:
                            if alt[0] == mk: loads[mk] = loads.get(mk,0) + alt[1]; break
                    for k in range(inst.num_machines): f[f"load_{k}"] = loads.get(k,0); f[f"ops_{k}"] = 0
                    for (jp_j, jp_p), mk in a.items(): f[f"ops_{mk}"] = f.get(f"ops_{mk}",0) + 1
                    X = np.array([[f.get(ff,0) for ff in feats]])
                    return -rf.predict(X)[0]

            # Beam search with this scorer
            pool = []
            seen = set()
            for name, a in gen.generate_all_heuristic().items():
                sig = tuple(sorted(a.items()))
                if sig not in seen: seen.add(sig); pool.append(a)
            for _ in range(4):
                candidates = list(pool)
                for current in pool:
                    for (j,p) in ops:
                        feasible = inst.feasible_machines.get((j,p), list(range(inst.num_machines)))
                        for new_k in feasible:
                            if new_k == current.get((j,p)): continue
                            c = dict(current); c[(j,p)] = new_k
                            sig = tuple(sorted(c.items()))
                            if sig not in seen and len(candidates) < 2000: seen.add(sig); candidates.append(c)
                candidates.sort(key=scorer, reverse=True)
                pool = candidates[:10]

            best = float("inf")
            for a in pool[:TOP_K]:
                exact = FixedAssignmentSequencingModel(inst, a, time_limit=SEQ_TIMEOUT).solve()
                if exact.get("has_solution") and exact["makespan"] < best: best = exact["makespan"]
            results.append({"instance": inst_name, "surrogate": surr_name, "makespan": best})
            winner = "*" if best <= min(r["makespan"] for r in results[-3:] if "makespan" in r) else ""
            print(f"    {surr_name:12s}: {best:.0f} {winner}")

    with open(os.path.join(RESULTS_DIR, "ablation_surrogates.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    print(f"\n  Ablation Summary:")
    for r in results:
        print(f"    {r['instance']} {r['surrogate']}: {r['makespan']}")
    return results


# ═══════════════════════════════════════════════════
# Phase 4: Quick BKS gap consolidation
# ═══════════════════════════════════════════════════

BKS = {
    "mk01": 40, "mk02": 26, "mk03": 204, "mk04": 60, "mk05": 172,
    "mk06": 57, "mk07": 139, "mk08": 523, "mk09": 307, "mk10": 193,
}

def final_bks_table():
    print("\n" + "=" * 50)
    print("Phase 4: Final BKS Gap Table")
    print("=" * 50)

    # Use known data from corrected statistics
    rf_vals = {"mk01":49,"mk02":42,"mk03":218,"mk04":78,"mk05":193,"mk06":100,"mk07":178,"mk08":523,"mk09":332,"mk10":277}
    bal_vals = {"mk01":49,"mk02":44,"mk03":230,"mk04":68,"mk05":210,"mk06":108,"mk07":206,"mk08":523,"mk09":354,"mk10":285}

    print(f"{'Inst':<6s} {'RF':>6s} {'Gap%':>7s} {'Bal':>6s} {'Gap%':>7s} {'BKS':>6s}")
    print("-" * 44)
    for inst in sorted(BKS.keys()):
        bks = BKS[inst]
        rf = rf_vals[inst]; bal = bal_vals[inst]
        rf_gap = (rf - bks) / bks * 100; bal_gap = (bal - bks) / bks * 100
        print(f"{inst:<6s} {rf:6.0f} {rf_gap:+6.1f}% {bal:6.0f} {bal_gap:+6.1f}% {bks:6d}")

    rows = []
    for inst in sorted(BKS.keys()):
        rows.append({"instance":inst, "rf":rf_vals[inst], "rf_gap":round((rf_vals[inst]-BKS[inst])/BKS[inst]*100,1),
                     "bal":bal_vals[inst], "bal_gap":round((bal_vals[inst]-BKS[inst])/BKS[inst]*100,1), "bks":BKS[inst]})
    with open(os.path.join(RESULTS_DIR, "final_bks.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


# ═══════════════════════════════════════════════════
# Phase 4: Generate all LaTeX tables
# ═══════════════════════════════════════════════════

def generate_latex_tables():
    print("\n" + "=" * 50)
    print("Phase 4: Generate LaTeX Tables")
    print("=" * 50)

    # Table 1: Instance characteristics + BKS
    with open(os.path.join(RESULTS_DIR, "final_bks.json")) as f:
        bks_data = json.load(f)

    tex = []
    tex.append("% Table: Instance characteristics and best-known bounds")
    tex.append("\\begin{table}[htbp]")
    tex.append("\\centering")
    tex.append("\\caption{Instance characteristics and gap to best-known solutions.}")
    tex.append("\\begin{tabular}{lrrrrrr}")
    tex.append("\\toprule")
    tex.append("Instance & $n \\times m$ & RF Search & RF Gap\\% & Balanced & Bal Gap\\% & BKS \\\\")
    tex.append("\\midrule")
    for r in bks_data:
        inst = r["instance"]
        nm = f"$\\times$"  # placeholder
        tex.append(f"{inst} & -- & {r['rf']:.0f} & {r['rf_gap']:+.1f} & {r['bal']:.0f} & {r['bal_gap']:+.1f} & {r['bks']} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")

    # Table 2: Fair baselines
    fair_path = os.path.join(RESULTS_DIR, "fair_baselines.json")
    if os.path.exists(fair_path):
        with open(fair_path) as f: fair_data = json.load(f)
        tex.append("")
        tex.append("% Table: Fair Baselines")
        tex.append("\\begin{table}[htbp]")
        tex.append("\\centering")
        tex.append("\\caption{Fair comparison under equal candidate-verification budget.}")
        tex.append("\\begin{tabular}{lrrrrrr}")
        tex.append("\\toprule")
        tex.append("Instance & Balanced & Random-MS & MaxLoad & HeurSeq & Bal+LS & RF \\\\")
        tex.append("\\midrule")
        for r in fair_data:
            line = r["instance"]
            for m in ["balanced", "random_ms", "maxload_beam", "heurseq_beam", "balanced_ls", "rf_beam"]:
                v = r.get(m)
                line += f" & {v:.0f}" if v is not None else " & --"
            line += " \\\\"
            tex.append(line)
        tex.append("\\bottomrule")
        tex.append("\\end{tabular}")
        tex.append("\\end{table}")

    # Table 3: Ablation
    abl_path = os.path.join(RESULTS_DIR, "ablation_surrogates.json")
    if os.path.exists(abl_path):
        with open(abl_path) as f: abl_data = json.load(f)
        tex.append("")
        tex.append("% Table: Ablation")
        tex.append("\\begin{table}[htbp]")
        tex.append("\\centering")
        tex.append("\\caption{Surrogate model ablation.}")
        tex.append("\\begin{tabular}{llr}")
        tex.append("\\toprule")
        tex.append("Instance & Surrogate & Makespan \\\\")
        tex.append("\\midrule")
        for r in abl_data:
            tex.append(f"{r['instance']} & {r['surrogate']} & {r['makespan']:.0f} \\\\")
        tex.append("\\bottomrule")
        tex.append("\\end{tabular}")
        tex.append("\\end{table}")

    with open(os.path.join(RESULTS_DIR, "latex_tables.tex"), "w") as f:
        f.write("\n".join(tex))
    print(f"  Wrote {len(tex)} lines to latex_tables.tex")


if __name__ == "__main__":
    complete_mk09()
    run_ablation()
    final_bks_table()
    generate_latex_tables()
    print("\n" + "=" * 50)
    print("ALL PHASES COMPLETE")
    print("=" * 50)
    for f in ["fair_baselines.json", "ablation_surrogates.json", "final_bks.json", "latex_tables.tex"]:
        print(f"  results/{f}")
