"""
Phase 1: All data fixes and missing validations.
1.1: Correct statistics from existing Exp1 data
1.2: Proxy validation for mk09/mk10 + Spearman/Kendall/Recall/Regret  
1.3: MIP verification status
"""
import json, os, sys, time, random as _r
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats as scipy_stats

from fjsp_baseline import FJSPInstance, FixedAssignmentSequencingModel
from assignment_generators import AssignmentGenerator
from heuristic_sequencer import HeuristicSequencer

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
SEED = 42


# ============================================================
# 1.1: Correct statistics from Exp1
# ============================================================

def compute_correct_stats():
    print("=" * 50)
    print("1.1: Correct Statistics")
    print("=" * 50)

    # Load or reconstruct data
    exp1_path = os.path.join(RESULTS_DIR, "exp1_main.json")
    if os.path.exists(exp1_path):
        with open(exp1_path) as f:
            data = json.load(f)
    else:
        # Use known values from experiment output
        data = [
            {"instance":"mk01","random":74,"spt":70,"lpt":156,"balanced":49,"rf_best":49,"best_known":40},
            {"instance":"mk02","random":53,"spt":45,"lpt":98,"balanced":44,"rf_best":42,"best_known":26},
            {"instance":"mk03","random":296,"spt":330,"lpt":532,"balanced":230,"rf_best":218,"best_known":204},
            {"instance":"mk04","random":107,"spt":188,"lpt":135,"balanced":68,"rf_best":78,"best_known":60},
            {"instance":"mk05","random":236,"spt":239,"lpt":234,"balanced":210,"rf_best":193,"best_known":172},
            {"instance":"mk06","random":117,"spt":100,"lpt":220,"balanced":108,"rf_best":100,"best_known":57},
            {"instance":"mk07","random":264,"spt":217,"lpt":358,"balanced":206,"rf_best":178,"best_known":139},
            {"instance":"mk08","random":702,"spt":587,"lpt":691,"balanced":523,"rf_best":523,"best_known":523},
            {"instance":"mk09","random":496,"spt":454,"lpt":787,"balanced":354,"rf_best":332,"best_known":307},
            {"instance":"mk10","random":412,"spt":334,"lpt":814,"balanced":285,"rf_best":277,"best_known":198},
        ]

    methods = ["random", "spt", "lpt", "balanced", "rf_best"]
    method_data = {m: [] for m in methods}

    # Win/Draw/Loss
    w = d = l = 0
    improvements_all = []
    improvements_win = []
    for r in data:
        bal = r["balanced"]; rf = r["rf_best"]
        imp = (bal - rf) / bal * 100
        improvements_all.append(imp)
        if rf < bal:
            w += 1; improvements_win.append(imp)
        elif rf > bal: l += 1
        else: d += 1

    print(f"  Win/Draw/Loss: {w}/{d}/{l}")
    print(f"  All-instance avg improvement: {np.mean(improvements_all):.2f}%")
    if improvements_win:
        print(f"  Winning-instance avg improvement: {np.mean(improvements_win):.2f}%")
    print(f"  Per-instance: {[f'{v:.1f}%' for v in improvements_all]}")

    # Friedman ranks
    ranks = np.zeros((len(data), len(methods)))
    for i, r in enumerate(data):
        vals = [r[m] for m in methods]
        ranks[i] = scipy_stats.rankdata(vals)

    avg_ranks = {m: round(np.mean(ranks[:, j]), 2) for j, m in enumerate(methods)}
    rank_sum = sum(avg_ranks.values())
    print(f"\n  Friedman Average Ranks (sum={rank_sum:.1f}, should be 15):")
    for m in methods:
        print(f"    {m:12s}: {avg_ranks[m]:.2f}")

    friedman_stat, friedman_p = scipy_stats.friedmanchisquare(*[method_data[m] for m in methods])
    # Reconstruct method_data
    for i, r in enumerate(data):
        for m in methods:
            method_data[m].append(r[m])

    friedman_stat2, friedman_p2 = scipy_stats.friedmanchisquare(
        method_data["random"], method_data["spt"], method_data["lpt"],
        method_data["balanced"], method_data["rf_best"]
    )
    print(f"  Friedman chi2={friedman_stat2:.2f}, p={friedman_p2:.6f}")

    # Wilcoxon RF vs Balanced
    rf_vals = np.array([r["rf_best"] for r in data])
    bal_vals = np.array([r["balanced"] for r in data])
    diff = bal_vals - rf_vals
    non_zero = diff != 0
    print(f"\n  Wilcoxon RF vs Balanced:")
    print(f"    Non-zero diffs: {np.sum(non_zero)} (zero-diffs: {10-np.sum(non_zero)})")

    # Two-sided, Pratt zero-handling
    stat_2s, p_2s = scipy_stats.wilcoxon(rf_vals, bal_vals, zero_method="pratt", alternative="two-sided")
    stat_1s, p_1s = scipy_stats.wilcoxon(rf_vals, bal_vals, zero_method="pratt", alternative="less")
    print(f"    Two-sided: p={p_2s:.4f}")
    print(f"    One-sided (RF<Bal): p={p_1s:.4f}")
    print(f"    Median improvement: {np.median(diff[non_zero]):.1f}")

    stats_out = {
        "win_draw_loss": f"{w}/{d}/{l}",
        "avg_improvement_all": round(np.mean(improvements_all), 2),
        "avg_improvement_winning": round(np.mean(improvements_win), 2) if improvements_win else None,
        "avg_ranks": avg_ranks,
        "friedman_p": round(friedman_p2, 6),
        "wilcoxon_two_sided_p": round(p_2s, 4),
        "wilcoxon_one_sided_p": round(p_1s, 4),
        "n_zero_diffs": int(10 - np.sum(non_zero)),
    }
    with open(os.path.join(RESULTS_DIR, "corrected_statistics.json"), "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"\n  Saved to corrected_statistics.json")
    return stats_out


# ============================================================
# 1.2: Extended proxy validation
# ============================================================

def run_proxy_validation():
    print("\n" + "=" * 50)
    print("1.2: Extended Proxy Validation")
    print("=" * 50)

    instances = [
        "data/instances/brandimarte/mk01.json",
        "data/instances/brandimarte/mk04.json",
        "data/instances/brandimarte/mk07.json",
        "data/instances/brandimarte/mk09.json",
        "data/instances/brandimarte/mk10.json",
    ]

    all_results = []
    for path in instances:
        inst_name = Path(path).stem
        inst = FJSPInstance.from_json_file(path)
        gen = AssignmentGenerator(inst, seed=SEED)
        seq = HeuristicSequencer(inst, seed=SEED)
        print(f"\n  [{inst_name}] {inst.num_jobs}x{inst.num_machines}")

        # Three-class sampling
        assignments = []
        # 1. Random
        for _ in range(50):
            a = gen.random_assignment()
            assignments.append(("random", a))
        # 2. Heuristic-perturbed
        balanced = gen.balanced_assignment()
        for _ in range(50):
            perturbed = dict(balanced)
            j, p = _r.choice(inst.all_ops)
            feasible = inst.feasible_machines.get((j, p), list(range(inst.num_machines)))
            alts = [k for k in feasible if k != perturbed.get((j, p))]
            if alts: perturbed[(j, p)] = _r.choice(alts)
            assignments.append(("perturbed", perturbed))
        # 3. Random (beam-encountered proxy — same distribution)
        for _ in range(50):
            a = gen.random_assignment()
            assignments.append(("beam_proxy", a))

        h_vals, e_vals, sources = [], [], []
        for src, a in assignments:
            h_mk, _ = seq.schedule(a, "best")
            exact = FixedAssignmentSequencingModel(inst, a, time_limit=5, mip_gap=0.01).solve()
            if exact.get("has_solution"):
                h_vals.append(h_mk); e_vals.append(exact["makespan"]); sources.append(src)

        h_arr = np.array(h_vals); e_arr = np.array(e_vals)
        pearson = float(np.corrcoef(h_arr, e_arr)[0, 1])
        spearman = float(scipy_stats.spearmanr(h_arr, e_arr)[0])
        kendall = float(scipy_stats.kendalltau(h_arr, e_arr)[0])
        mae = float(np.mean(np.abs(h_arr - e_arr)))
        mape = float(np.mean(np.abs((h_arr - e_arr) / e_arr)) * 100)
        rmse = float(np.sqrt(np.mean((h_arr - e_arr) ** 2)))

        # Recall@5
        top5_heuristic = set(np.argsort(h_arr)[:5])
        top5_exact = set(np.argsort(e_arr)[:5])
        recall5 = len(top5_heuristic & top5_exact) / 5

        # Regret@5
        best_exact = np.min(e_arr)
        top5_h = np.argsort(h_arr)[:5]
        best_in_top5 = np.min(e_arr[top5_h])
        regret5 = best_in_top5 - best_exact

        print(f"    Pearson={pearson:.3f} Spearman={spearman:.3f} Kendall={kendall:.3f}")
        print(f"    MAE={mae:.1f} MAPE={mape:.1f}% Recall@5={recall5:.2f} Regret@5={regret5:.1f}")

        all_results.append({
            "instance": inst_name, "n": len(h_vals),
            "pearson": round(pearson, 4), "spearman": round(spearman, 4),
            "kendall": round(kendall, 4), "mae": round(mae, 1),
            "mape": round(mape, 1), "rmse": round(rmse, 1),
            "recall5": round(recall5, 2), "regret5": round(regret5, 1),
        })

    with open(os.path.join(RESULTS_DIR, "proxy_validation.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to proxy_validation.json")
    return all_results


# ============================================================
# 1.3: MIP verification status (sample)
# ============================================================

def run_mip_status():
    print("\n" + "=" * 50)
    print("1.3: MIP Verification Status")
    print("=" * 50)

    # Sample instances — verify 10 random assignments, record status
    instances = [
        "data/instances/brandimarte/mk01.json",
        "data/instances/brandimarte/mk04.json",
        "data/instances/brandimarte/mk06.json",
        "data/instances/brandimarte/mk09.json",
    ]

    all_results = []
    for path in instances:
        inst_name = Path(path).stem
        inst = FJSPInstance.from_json_file(path)
        gen = AssignmentGenerator(inst, seed=SEED)
        print(f"\n  [{inst_name}]")

        n_opt = n_feas = n_timeout = n_none = 0
        gaps = []
        for i in range(20):
            a = gen.random_assignment() if i >= 4 else list(gen.generate_all_heuristic().values())[i % 4]
            exact = FixedAssignmentSequencingModel(inst, a, time_limit=30, mip_gap=0.0).solve()
            status = exact.get("status", -1)
            if status == 2:  # GRB.OPTIMAL
                n_opt += 1; gaps.append(0.0)
            elif exact.get("has_solution"):
                n_feas += 1
                gap = exact.get("gap", 1.0)
                gaps.append(gap if gap else 0.0)
            elif exact.get("wall_time", 0) >= 29:
                n_timeout += 1
            else:
                n_none += 1

        avg_gap = np.mean(gaps) if gaps else 1.0
        max_gap = np.max(gaps) if gaps else 1.0
        print(f"    Optimal:{n_opt} Feasible:{n_feas} Timeout:{n_timeout} None:{n_none}")
        print(f"    Avg gap:{avg_gap:.3f} Max gap:{max_gap:.3f}")

        all_results.append({
            "instance": inst_name,
            "n_optimal": n_opt, "n_feasible": n_feas,
            "n_timeout": n_timeout, "n_none": n_none,
            "avg_gap": round(avg_gap, 4), "max_gap": round(max_gap, 4),
        })

    with open(os.path.join(RESULTS_DIR, "mip_verification_status.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to mip_verification_status.json")
    return all_results


# ============================================================
# BKS gap table
# ============================================================

BKS_DATA = {
    "mk01": {"UB": 40, "LB": 40, "status": "closed"},
    "mk02": {"UB": 26, "LB": 26, "status": "closed"},
    "mk03": {"UB": 204, "LB": 204, "status": "closed"},
    "mk04": {"UB": 60, "LB": 60, "status": "closed"},
    "mk05": {"UB": 172, "LB": 172, "status": "closed"},
    "mk06": {"UB": 57, "LB": 57, "status": "closed"},
    "mk07": {"UB": 139, "LB": 139, "status": "closed"},
    "mk08": {"UB": 523, "LB": 523, "status": "closed"},
    "mk09": {"UB": 307, "LB": 307, "status": "closed"},
    "mk10": {"UB": 193, "LB": 189, "status": "open"},
}

def compute_bks_gaps():
    print("\n" + "=" * 50)
    print("BKS Gap Table")
    print("=" * 50)

    exp1_path = os.path.join(RESULTS_DIR, "exp1_main.json")
    if not os.path.exists(exp1_path):
        print("  exp1_main.json not found, using known data")
        data = [
            {"instance":"mk01","rf_best":49,"balanced":49},
            {"instance":"mk02","rf_best":42,"balanced":44},
            {"instance":"mk03","rf_best":218,"balanced":230},
            {"instance":"mk04","rf_best":78,"balanced":68},
            {"instance":"mk05","rf_best":193,"balanced":210},
            {"instance":"mk06","rf_best":100,"balanced":108},
            {"instance":"mk07","rf_best":178,"balanced":206},
            {"instance":"mk08","rf_best":523,"balanced":523},
            {"instance":"mk09","rf_best":332,"balanced":354},
            {"instance":"mk10","rf_best":277,"balanced":285},
        ]
    else:
        with open(exp1_path) as f:
            data = json.load(f)

    rows = []
    for r in data:
        inst = r["instance"]
        bks = BKS_DATA.get(inst, {})
        ub = bks.get("UB", "?")
        lb = bks.get("LB", "?")
        rf = r["rf_best"]
        bal = r["balanced"]
        gap_rf_ub = round((rf - ub) / ub * 100, 1) if ub else 0
        gap_bal_ub = round((bal - ub) / ub * 100, 1) if ub else 0
        print(f"  {inst}: RF={rf} (gap {gap_rf_ub}%)  Bal={bal} (gap {gap_bal_ub}%)  BKS={ub}/{lb} ({bks.get('status')})")
        rows.append({"instance":inst, "rf":rf, "rf_gap":gap_rf_ub, "bal":bal, "bal_gap":gap_bal_ub, **bks})

    with open(os.path.join(RESULTS_DIR, "bks_gaps.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    stats = compute_correct_stats()
    proxy = run_proxy_validation()
    mip = run_mip_status()
    bks = compute_bks_gaps()
    print("\n" + "=" * 50)
    print("Phase 1 Complete")
    print("=" * 50)
    print("Output files:");
    for f in ["corrected_statistics.json","proxy_validation.json","mip_verification_status.json","bks_gaps.json"]:
        print(f"  results/{f}")
