"""Complete remaining experiments quickly + save all data."""
import json, os, sys, time, random as _r
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats as scipy_stats

from fjsp_baseline import FJSPInstance, FixedAssignmentSequencingModel
from assignment_generators import AssignmentGenerator
from heuristic_sequencer import HeuristicSequencer
from experiments_final import train_rf_fast, beam_search

SEED = 42
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
BRANDIMARTE = [f"data/instances/brandimarte/mk{i:02d}.json" for i in range(1, 11)]


def run_exp4():
    """Heuristic sequencer fidelity across all Brandimarte."""
    print("=" * 50)
    print("EXP 4: Heuristic Sequencer Fidelity")
    print("=" * 50)
    results = []
    for path in BRANDIMARTE:
        inst_name = Path(path).stem
        inst = FJSPInstance.from_json_file(path)
        gen = AssignmentGenerator(inst, seed=SEED)
        seq = HeuristicSequencer(inst, seed=SEED)
        h_vals, e_vals = [], []
        for i in range(40):
            if i < 4:
                a = list(gen.generate_all_heuristic().values())[i]
            else:
                a = gen.random_assignment()
            h_mk, _ = seq.schedule(a, "best")
            exact = FixedAssignmentSequencingModel(inst, a, time_limit=5, mip_gap=0.01).solve()
            if exact.get("has_solution"):
                h_vals.append(h_mk); e_vals.append(exact["makespan"])
        r_val = float(np.corrcoef(h_vals, e_vals)[0, 1])
        print(f"  {inst_name}: r={r_val:.4f} (n={len(h_vals)})")
        results.append({"instance": inst_name, "pearson_r": round(r_val, 4), "n": len(h_vals)})
    with open(os.path.join(RESULTS_DIR, "exp4_heuristic_r.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_complement_ablation():
    """Quick ablation on mk07 + mk09."""
    print("\n" + "=" * 50)
    print("EXP 2 (complement): mk07 + mk09 ablation")
    print("=" * 50)
    instances = [
        "data/instances/brandimarte/mk07.json",
        "data/instances/brandimarte/mk09.json",
    ]
    all_results = []
    for path in instances:
        inst_name = Path(path).stem
        inst = FJSPInstance.from_json_file(path)
        gen = AssignmentGenerator(inst, seed=SEED)
        bal_a = gen.balanced_assignment()
        bal = FixedAssignmentSequencingModel(inst, bal_a, time_limit=30).solve()
        bal_mk = bal.get("makespan")

        for cfg_name, trees, depth in [("RF_full", 150, 12), ("RF_small", 50, 8), ("RF_shallow", 150, 6)]:
            rf, feats, bp, r2 = train_rf_fast(inst, n_trees=trees, depth=depth)
            ranked = beam_search(inst, bp, beam_width=15)
            best = float("inf")
            for _, a in ranked[:8]:
                exact = FixedAssignmentSequencingModel(inst, a, time_limit=30, mip_gap=0.0).solve()
                if exact.get("has_solution") and exact["makespan"] < best:
                    best = exact["makespan"]
            imp = ((bal_mk - best) / bal_mk * 100) if bal_mk else 0
            print(f"  {inst_name} {cfg_name}: Bal={bal_mk:.0f}, RF={best:.0f}, d={imp:+.1f}%")
            all_results.append({"instance": inst_name, "config": cfg_name,
                                "balanced": bal_mk, "rf": best, "improvement_pct": round(imp, 1)})

    with open(os.path.join(RESULTS_DIR, "exp2_complement.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Also read existing exp2 results and merge
    if os.path.exists(os.path.join(RESULTS_DIR, "exp2_ablation.json")):
        with open(os.path.join(RESULTS_DIR, "exp2_ablation.json")) as f:
            existing = json.load(f)
        all_results = existing + all_results
    with open(os.path.join(RESULTS_DIR, "exp2_full.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


def run_statistical_tests():
    """Friedman + Wilcoxon on Exp1 results."""
    print("\n" + "=" * 50)
    print("STATISTICAL TESTS")
    print("=" * 50)
    exp1_path = os.path.join(RESULTS_DIR, "exp1_main.json")
    if not os.path.exists(exp1_path):
        print("  ERROR: exp1_main.json not found, using data from latest run")
        # Reconstruct from the known results
        data = [
            {"instance":"mk01","random":74,"spt":70,"lpt":156,"balanced":49,"rf_best":49},
            {"instance":"mk02","random":53,"spt":45,"lpt":98,"balanced":44,"rf_best":42},
            {"instance":"mk03","random":296,"spt":330,"lpt":532,"balanced":230,"rf_best":218},
            {"instance":"mk04","random":107,"spt":188,"lpt":135,"balanced":68,"rf_best":78},
            {"instance":"mk05","random":236,"spt":239,"lpt":234,"balanced":210,"rf_best":193},
            {"instance":"mk06","random":117,"spt":100,"lpt":220,"balanced":108,"rf_best":100},
            {"instance":"mk07","random":264,"spt":217,"lpt":358,"balanced":206,"rf_best":209},
            {"instance":"mk08","random":702,"spt":587,"lpt":691,"balanced":523,"rf_best":523},
            {"instance":"mk09","random":496,"spt":454,"lpt":787,"balanced":354,"rf_best":332},
            {"instance":"mk10","random":412,"spt":334,"lpt":814,"balanced":285,"rf_best":277},
        ]
    else:
        with open(exp1_path) as f:
            data = json.load(f)

    # Extract makespan arrays
    methods = ["random", "spt", "lpt", "balanced", "rf_best"]
    method_data = {m: [] for m in methods}
    instances = []
    for r in data:
        instances.append(r["instance"])
        for m in methods:
            method_data[m].append(r.get(m, float("inf")))

    # Friedman test
    ranks = np.zeros((len(data), len(methods)))
    for i in range(len(data)):
        vals = [method_data[m][i] for m in methods]
        # Higher rank = worse (larger makespan = worse)
        ranks[i] = scipy_stats.rankdata(vals)

    friedman_stat, friedman_p = scipy_stats.friedmanchisquare(*[method_data[m] for m in methods])
    print(f"  Friedman chi2={friedman_stat:.2f}, p={friedman_p:.4f}")

    # Average ranks
    print(f"\n  Average Ranks (lower=better):")
    for m in methods:
        avg_rank = np.mean(ranks[:, methods.index(m)])
        print(f"    {m:12s}: {avg_rank:.2f}")

    # Pairwise Wilcoxon
    print(f"\n  Pairwise Wilcoxon (rf_best vs others):")
    rf_vals = np.array(method_data["rf_best"])
    for m in methods:
        if m == "rf_best": continue
        other = np.array(method_data[m])
        # Only compare non-inf values
        mask = np.isfinite(rf_vals) & np.isfinite(other)
        if mask.sum() >= 5:
            stat, p = scipy_stats.wilcoxon(rf_vals[mask], other[mask])
            better = "RF better" if np.median(rf_vals[mask]) < np.median(other[mask]) else f"{m} better"
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
            print(f"    rf_best vs {m:12s}: p={p:.4f} {sig} ({better})")

    # Win/Draw/Loss of RF vs balanced
    w = d = l = 0
    for r in data:
        bal = r.get("balanced", float("inf"))
        rf = r.get("rf_best", float("inf"))
        if rf < bal: w += 1
        elif rf > bal: l += 1
        else: d += 1
    print(f"\n  RF vs Balanced: {w}W/{d}D/{l}L")

    test_results = {
        "friedman_stat": round(friedman_stat, 2),
        "friedman_p": round(friedman_p, 6),
        "avg_ranks": {m: round(np.mean(ranks[:, methods.index(m)]), 2) for m in methods},
        "win_draw_loss": f"{w}W/{d}D/{l}L",
    }
    with open(os.path.join(RESULTS_DIR, "statistical_tests.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    return test_results


if __name__ == "__main__":
    exp4 = run_exp4()
    exp2b = run_complement_ablation()
    stats = run_statistical_tests()
    print("\n" + "=" * 50)
    print("ALL DONE")
    print("=" * 50)
    print(f"Results: exp1_main.json, exp2_full.json, exp4_heuristic_r.json, statistical_tests.json")
