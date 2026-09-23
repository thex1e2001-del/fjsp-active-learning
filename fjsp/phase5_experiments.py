"""
Phase 5: Complete all remaining experiments + generate final tables.
Experiments: Top-K sensitivity, beam width sensitivity, 7-surrogate comparison,
             feature group ablation, quantitative failure analysis.
"""
import json, os, sys, time, random as _r
from pathlib import Path
from typing import Dict, List
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from fjsp_baseline import FJSPInstance, FixedAssignmentSequencingModel
from assignment_generators import AssignmentGenerator
from heuristic_sequencer import HeuristicSequencer
from experiments_final import train_rf_fast, beam_search, BatchPredictor
from rf_train import train_global_rf, build_feature_cols

SEED = 42
SEQ_TIMEOUT = 30
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
INSTS = [
    "data/instances/brandimarte/mk01.json",
    "data/instances/brandimarte/mk04.json",
    "data/instances/brandimarte/mk07.json",
]

def load_inst(path): return FJSPInstance.from_json_file(path)

# ═══════════════════════════════════════════
# Exp 1: Top-K sensitivity
# ═══════════════════════════════════════════
def exp_topk():
    print("="*50,"\nExp 1: Top-K sensitivity\n","="*50)
    Ks = [1,3,5,8,10,15,20]
    results = []
    for path in INSTS:
        inst = load_inst(path); name = Path(path).stem
        print(f"\n  [{name}]")
        rf, feats, bp, r2 = train_rf_fast(inst)
        ranked = beam_search(inst, bp)
        for K in Ks:
            best = float("inf")
            for _, a in ranked[:K]:
                e = FixedAssignmentSequencingModel(inst, a, time_limit=SEQ_TIMEOUT).solve()
                if e.get("has_solution") and e["makespan"] < best: best = e["makespan"]
            results.append({"instance":name,"K":K,"makespan":best})
            print(f"    K={K:2d}: {best:.0f}")
    with open(os.path.join(RESULTS_DIR,"topk_sensitivity.json"),"w") as f: json.dump(results,f,indent=2)
    return results

# ═══════════════════════════════════════════
# Exp 2: Beam width sensitivity
# ═══════════════════════════════════════════
def exp_beamwidth():
    print("\n"+"="*50+"\nExp 2: Beam width sensitivity\n"+"="*50)
    Bs = [5,10,15,30,50]
    results = []
    for path in INSTS:
        inst = load_inst(path); name = Path(path).stem
        print(f"\n  [{name}]")
        rf, feats, bp, r2 = train_rf_fast(inst)
        for B in Bs:
            ranked = beam_search(inst, bp, beam_width=B)
            best = float("inf")
            for _, a in ranked[:8]:
                e = FixedAssignmentSequencingModel(inst, a, time_limit=SEQ_TIMEOUT).solve()
                if e.get("has_solution") and e["makespan"] < best: best = e["makespan"]
            results.append({"instance":name,"beam_width":B,"makespan":best})
            print(f"    B={B:2d}: {best:.0f}")
    with open(os.path.join(RESULTS_DIR,"beamwidth_sensitivity.json"),"w") as f: json.dump(results,f,indent=2)
    return results

# ═══════════════════════════════════════════
# Exp 3: 7-surrogate scorer comparison
# ═══════════════════════════════════════════
def exp_surrogates():
    print("\n"+"="*50+"\nExp 3: 7-surrogate comparison\n"+"="*50)
    all_results = []
    for path in INSTS:
        inst = load_inst(path); name = Path(path).stem
        gen = AssignmentGenerator(inst, seed=SEED); seq = HeuristicSequencer(inst, seed=SEED)
        feats = build_feature_cols(inst.num_machines); ops = inst.all_ops
        print(f"\n  [{name}]")

        # Generate training data
        balanced = gen.balanced_assignment()
        train_a = [balanced]
        for _ in range(800):
            p = dict(balanced)
            for _ in range(_r.randint(1,max(2,inst.total_ops//15))):
                j,pp = _r.choice(ops)
                f = inst.feasible_machines.get((j,pp),list(range(inst.num_machines)))
                alts = [k for k in f if k!=p.get((j,pp))]
                if alts: p[(j,pp)] = _r.choice(alts)
            train_a.append(p)
        h_labels = [seq.schedule(a,"best")[0] for a in train_a]
        X = np.array([[a.get((j,pp),0) for (j,pp) in ops] for a in train_a])

        # Wrapper for beam search
        def run_beam(scorer_fn):
            pool = []
            seen = set()
            for nm, a in gen.generate_all_heuristic().items():
                sig = tuple(sorted(a.items()))
                if sig not in seen: seen.add(sig); pool.append(a)
            for _ in range(4):
                cands = list(pool)
                for cur in pool:
                    for (j,pp) in ops:
                        feasible = inst.feasible_machines.get((j,pp),list(range(inst.num_machines)))
                        for nk in feasible:
                            if nk==cur.get((j,pp)): continue
                            c = dict(cur); c[(j,pp)]=nk
                            sig = tuple(sorted(c.items()))
                            if sig not in seen and len(cands)<2000: seen.add(sig); cands.append(c)
                cands.sort(key=scorer_fn, reverse=True)
                pool = cands[:10]
            best = float("inf")
            for a in pool[:8]:
                e = FixedAssignmentSequencingModel(inst,a,time_limit=SEQ_TIMEOUT).solve()
                if e.get("has_solution") and e["makespan"]<best: best=e["makespan"]
            return best

        def get_loads(a):
            loads={}
            for (jj,pp),mk in a.items():
                for alt in inst.jobs[jj][pp]:
                    if alt[0]==mk: loads[mk]=loads.get(mk,0)+alt[1]; break
            return loads

        scorers = {
            "MaxLoad": lambda a: -max(get_loads(a).values()) if get_loads(a) else 0,
            "LB": lambda a: -max(max(get_loads(a).values()) if get_loads(a) else 0,
                                  max(sum(alt[1] for alt in inst.jobs[j][pp] if alt[0]==a.get((j,pp),0))
                                      for j in range(inst.num_jobs) for pp in range(len(inst.jobs.get(j,[]))))),
            "Imbalance": lambda a: -(max(get_loads(a).values())-min(get_loads(a).values())) if len(get_loads(a))>1 else 0,
            "HeurSeq": lambda a: -seq.schedule(a,"best")[0],
        }

        # Linear regression
        lr = LinearRegression().fit(X, h_labels)
        scorers["LinReg"] = lambda a: -lr.predict([[a.get((j,pp),0) for (j,pp) in ops]])[0]
        # CART
        cart = DecisionTreeRegressor(max_depth=6, random_state=SEED).fit(X, h_labels)
        scorers["CART"] = lambda a: -cart.predict([[a.get((j,pp),0) for (j,pp) in ops]])[0]

        # RF (use train_rf_fast)
        rf, _, bp, _ = train_rf_fast(inst, n_trees=50, depth=8)
        scorers["RF"] = lambda a: -bp.predict_one(a)

        for sname, sfn in scorers.items():
            mk = run_beam(sfn)
            all_results.append({"instance":name,"surrogate":sname,"makespan":mk})
            print(f"    {sname:12s}: {mk:.0f}")

    with open(os.path.join(RESULTS_DIR,"surrogate_comparison.json"),"w") as f: json.dump(all_results,f,indent=2)

    # Summary table
    print(f"\n  Surrogate Summary:")
    for r in all_results:
        print(f"    {r['instance']} {r['surrogate']}: {r['makespan']:.0f}")
    return all_results

# ═══════════════════════════════════════════
# Exp 4: Feature group ablation
# ═══════════════════════════════════════════
def exp_feature_ablation():
    print("\n"+"="*50+"\nExp 4: Feature group ablation\n"+"="*50)
    all_results = []
    for path in INSTS[:2]:  # mk01, mk04 (faster)
        inst = load_inst(path); name = Path(path).stem
        gen = AssignmentGenerator(inst, seed=SEED); seq = HeuristicSequencer(inst, seed=SEED)
        ops = inst.all_ops; nm = inst.num_machines
        print(f"\n  [{name}]")

        # Generate training data
        balanced = gen.balanced_assignment()
        train_a = [balanced]
        for _ in range(500):
            p = dict(balanced)
            for _ in range(_r.randint(1,max(2,inst.total_ops//15))):
                j,pp = _r.choice(ops)
                f = inst.feasible_machines.get((j,pp),list(range(nm)))
                alts = [k for k in f if k!=p.get((j,pp))]
                if alts: p[(j,pp)] = _r.choice(alts)
            train_a.append(p)
        h_labels = [seq.schedule(a,"best")[0] for a in train_a]

        # Feature groups
        from sklearn.ensemble import RandomForestRegressor
        def make_features(assigns, groups):
            X = []
            for a in assigns:
                row = []
                loads = {}
                for (jj,pp),mk in a.items():
                    for alt in inst.jobs[jj][pp]:
                        if alt[0]==mk: loads[mk]=loads.get(mk,0)+alt[1]; break
                if "load" in groups:
                    row.extend([loads.get(k,0) for k in range(nm)])
                if "ops" in groups:
                    oc = {k:0 for k in range(nm)}
                    for (jj,pp),mk in a.items(): oc[mk]+=1
                    row.extend([oc[k] for k in range(nm)])
                if "balance" in groups:
                    used = [loads.get(k,0) for k in range(nm) if loads.get(k,0)>0]
                    row.append(max(used)-min(used) if len(used)>1 else 0)
                    row.append(max(used) if used else 0)
                if "job" in groups:
                    jt = []
                    for j in range(inst.num_jobs):
                        total=0
                        for pp in range(len(inst.jobs.get(j,[]))):
                            mk=a.get((j,pp))
                            if mk is not None:
                                for alt in inst.jobs[j][pp]:
                                    if alt[0]==mk: total+=alt[1]; break
                        jt.append(total)
                    row.append(max(jt) if jt else 0)
                X.append(row)
            return np.array(X)

        group_sets = {
            "Load only": ["load"],
            "+Ops count": ["load","ops"],
            "+Balance": ["load","ops","balance"],
            "+Job structure": ["load","ops","balance","job"],
        }

        def rf_beam(groups):
            X = make_features(train_a, groups)
            rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=SEED, n_jobs=-1)
            rf.fit(X, h_labels)

            def scorer(a):
                feats = make_features([a], groups)[0]
                return -rf.predict([feats])[0]

            pool = []
            seen = set()
            for nm_g, a in gen.generate_all_heuristic().items():
                sig = tuple(sorted(a.items()))
                if sig not in seen: seen.add(sig); pool.append(a)
            for _ in range(4):
                cands = list(pool)
                for cur in pool:
                    for (j,pp) in ops:
                        feasible = inst.feasible_machines.get((j,pp),list(range(nm)))
                        for nk in feasible:
                            if nk==cur.get((j,pp)): continue
                            c = dict(cur); c[(j,pp)]=nk
                            sig = tuple(sorted(c.items()))
                            if sig not in seen and len(cands)<1500: seen.add(sig); cands.append(c)
                cands.sort(key=scorer, reverse=True)
                pool = cands[:10]
            best = float("inf")
            for a in pool[:8]:
                e = FixedAssignmentSequencingModel(inst,a,time_limit=SEQ_TIMEOUT).solve()
                if e.get("has_solution") and e["makespan"]<best: best=e["makespan"]
            return best

        for gname, groups in group_sets.items():
            mk = rf_beam(groups)
            all_results.append({"instance":name,"feature_group":gname,"makespan":mk,"n_features":len(groups)})
            print(f"    {gname:20s}: {mk:.0f}")

    with open(os.path.join(RESULTS_DIR,"feature_ablation.json"),"w") as f: json.dump(all_results,f,indent=2)
    return all_results

# ═══════════════════════════════════════════
# Exp 5: Quantitative failure analysis
# ═══════════════════════════════════════════
def exp_failure():
    print("\n"+"="*50+"\nExp 5: Quantitative failure analysis\n"+"="*50)
    failures = [
        ("data/instances/brandimarte/mk04.json", "mk04", 68),
        ("data/instances/brandimarte/mk06.json", "mk06", 57),
        ("data/instances/brandimarte/mk10.json", "mk10", 193),
    ]
    results = []
    for path, name, bks in failures:
        inst = load_inst(path); gen = AssignmentGenerator(inst, seed=SEED); seq = HeuristicSequencer(inst, seed=SEED)
        print(f"\n  [{name}] BKS={bks}")

        row = {"instance": name, "bks": bks}
        # Balanced
        ba = gen.balanced_assignment()
        bal = FixedAssignmentSequencingModel(inst, ba, time_limit=SEQ_TIMEOUT).solve()
        row["balanced"] = bal.get("makespan")
        # RF
        rf, feats, bp, r2 = train_rf_fast(inst)
        ranked = beam_search(inst, bp)
        best = float("inf")
        for _, a in ranked[:8]:
            e = FixedAssignmentSequencingModel(inst, a, time_limit=SEQ_TIMEOUT).solve()
            if e.get("has_solution") and e["makespan"] < best: best = e["makespan"]
        row["rf"] = best
        row["rf_r2"] = round(r2, 3)

        # Analyze RF top assignment vs balanced
        # Load stats
        def get_stats(a):
            loads = {}
            for (jj,pp),mk in a.items():
                for alt in inst.jobs[jj][pp]:
                    if alt[0]==mk: loads[mk]=loads.get(mk,0)+alt[1]; break
            used = [v for v in loads.values()]
            return {
                "max_load": max(used) if used else 0,
                "min_load": min(used) if used else 0,
                "imbalance": max(used)-min(used) if len(used)>1 else 0,
                "n_used": len(used),
                "total_load": sum(used),
            }
        bs = get_stats(ba)
        row.update({f"bal_{k}":v for k,v in bs.items()})
        if ranked:
            rs = get_stats(ranked[0][1])
            row.update({f"rf_{k}":v for k,v in rs.items()})

        print(f"    Balanced: {row['balanced']:.0f}, RF: {row['rf']:.0f}, R^2={row['rf_r2']:.3f}")
        print(f"    Bal loads: max={bs['max_load']:.0f} imb={bs['imbalance']:.0f}")
        if ranked: print(f"    RF loads:  max={rs['max_load']:.0f} imb={rs['imbalance']:.0f}")
        results.append(row)

    with open(os.path.join(RESULTS_DIR,"failure_analysis.json"),"w") as f: json.dump(results,f,indent=2)
    return results


if __name__ == "__main__":
    r1 = exp_topk()
    r2 = exp_beamwidth()
    r3 = exp_surrogates()
    r4 = exp_feature_ablation()
    r5 = exp_failure()
    print("\n"+"="*50)
    print("ALL 5 EXPERIMENTS COMPLETE")
    print("="*50)
    for f in ["topk_sensitivity.json","beamwidth_sensitivity.json","surrogate_comparison.json","feature_ablation.json","failure_analysis.json"]:
        print(f"  results/{f}")
