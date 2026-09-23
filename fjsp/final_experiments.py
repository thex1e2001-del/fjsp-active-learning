"""
Final complete experiments: statistics, proxy validation, fair baselines, scorer ablation, failure analysis.
All 10 Brandimarte instances, unified protocol.
"""
import json, os, sys, time, random as _r, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from fjsp_baseline import FJSPInstance, FixedAssignmentSequencingModel
from assignment_generators import AssignmentGenerator
from heuristic_sequencer import HeuristicSequencer
from experiments_final import train_rf_fast, beam_search
from rf_train import build_feature_cols

SEED = 42; SEQ_TO = 30; RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
INSTS = [f"data/instances/brandimarte/mk{i:02d}.json" for i in range(1,11)]
LAM = FJSPInstance.from_json_file

# ═══════════════════════════════════════════
# 1. CORRECT STATISTICS
# ═══════════════════════════════════════════
def correct_stats():
    print("="*50,"\n1. Correct Statistics\n","="*50)
    data = [
        {"i":"mk01","r":74,"s":70,"l":156,"b":49,"f":49,"bk":40},
        {"i":"mk02","r":53,"s":45,"l":98,"b":44,"f":42,"bk":26},
        {"i":"mk03","r":296,"s":330,"l":532,"b":230,"f":218,"bk":204},
        {"i":"mk04","r":107,"s":188,"l":135,"b":68,"f":78,"bk":60},
        {"i":"mk05","r":236,"s":239,"l":234,"b":210,"f":193,"bk":172},
        {"i":"mk06","r":117,"s":100,"l":220,"b":108,"f":100,"bk":58},
        {"i":"mk07","r":264,"s":217,"l":358,"b":206,"f":209,"bk":139},
        {"i":"mk08","r":702,"s":587,"l":691,"b":523,"f":523,"bk":523},
        {"i":"mk09","r":496,"s":454,"l":787,"b":354,"f":332,"bk":307},
        {"i":"mk10","r":412,"s":334,"l":814,"b":285,"f":277,"bk":198},
    ]
    methods = ["r","s","l","b","f"]
    imp_all=[(d["b"]-d["f"])/d["b"]*100 for d in data]
    imp_win=[v for v in imp_all if v>0]
    w=sum(1 for v in imp_all if v>0); d_=sum(1 for v in imp_all if v==0); l_=sum(1 for v in imp_all if v<0)
    print(f"  W/D/L: {w}/{d_}/{l_}")
    print(f"  All-instance avg improvement: {np.mean(imp_all):.2f}%")
    print(f"  Winning-instance avg improvement: {np.mean(imp_win):.2f}%")
    # Friedman
    ranks=[]
    for d in data:
        vals=[d[m] for m in methods]
        ranks.append(scipy_stats.rankdata(vals,method='average'))
    ranks=np.array(ranks)
    avg_ranks={methods[i]:round(np.mean(ranks[:,i]),2) for i in range(5)}
    print(f"  Friedman avg ranks: {avg_ranks}, sum={sum(avg_ranks.values()):.2f}")
    fv=[data[i][m] for m in methods for i in range(10)]
    fried_stat,fried_p=scipy_stats.friedmanchisquare(
        [d["r"] for d in data],[d["s"] for d in data],[d["l"] for d in data],
        [d["b"] for d in data],[d["f"] for d in data])
    print(f"  Friedman chi2={fried_stat:.2f}, p={fried_p:.6f}")
    # Wilcoxon
    rf_v=np.array([d["f"] for d in data]); bal_v=np.array([d["b"] for d in data])
    w2, p2 = scipy_stats.wilcoxon(rf_v, bal_v, zero_method='pratt', alternative='two-sided')
    w1, p1 = scipy_stats.wilcoxon(rf_v, bal_v, zero_method='pratt', alternative='less')
    print(f"  Wilcoxon two-sided: p={p2:.4f}")
    print(f"  Wilcoxon one-sided (RF<Bal): p={p1:.4f}")
    print(f"  Non-zero diffs: {sum(rf_v!=bal_v)}, median diff: {np.median(bal_v[rf_v!=bal_v]-rf_v[rf_v!=bal_v]):.1f}")
    with open(os.path.join(RESULTS_DIR,"final_statistics.json"),"w") as f: json.dump({
        "wdl":f"{w}/{d_}/{l_}","avg_imp_all":round(np.mean(imp_all),2),
        "avg_imp_win":round(np.mean(imp_win),2),"avg_ranks":avg_ranks,
        "friedman_p":round(fried_p,6),"wilcoxon_two_sided_p":round(p2,4),
        "wilcoxon_one_sided_p":round(p1,4)},f,indent=2)


# ═══════════════════════════════════════════
# 2. COMPLETE PROXY VALIDATION (all 10, 3 sample types)
# ═══════════════════════════════════════════
def proxy_validation_all():
    print("\n"+"="*50+"\n2. Complete Proxy Validation (10 instances)\n"+"="*50)
    results=[]
    for path in INSTS:
        inst=LAM(path); name=Path(path).stem
        gen=AssignmentGenerator(inst,seed=SEED); seq=HeuristicSequencer(inst,seed=SEED)
        h_all=[]; e_all=[]
        # Uniform random (40)
        for _ in range(40):
            a=gen.random_assignment(); h,_=seq.schedule(a,"best")
            e=FixedAssignmentSequencingModel(inst,a,time_limit=5,mip_gap=0.01).solve()
            if e.get("has_solution"): h_all.append(h); e_all.append(e["makespan"])
        if len(h_all)<20: continue
        ha=np.array(h_all); ea=np.array(e_all)
        pear=float(np.corrcoef(ha,ea)[0,1])
        spear=float(scipy_stats.spearmanr(ha,ea)[0])
        kend=float(scipy_stats.kendalltau(ha,ea)[0])
        mae=float(np.mean(np.abs(ha-ea))); mape=float(np.mean(np.abs((ha-ea)/ea))*100)
        top5h=set(np.argsort(ha)[:5]); top5e=set(np.argsort(ea)[:5])
        rec5=len(top5h&top5e)/5
        best_e=np.min(ea); best_in_top5=np.min(ea[np.argsort(ha)[:5]])
        reg5=best_in_top5-best_e
        results.append({"instance":name,"n":len(h_all),"pearson":round(pear,4),
            "spearman":round(spear,4),"kendall":round(kend,4),"mae":round(mae,1),
            "mape":round(mape,1),"recall5":round(rec5,2),"regret5":round(reg5,1)})
        print(f"  {name}: Pear={pear:.3f} Spear={spear:.3f} Kendall={kend:.3f} Recall@5={rec5:.2f} Regret@5={reg5:.1f}")
    with open(os.path.join(RESULTS_DIR,"proxy_validation_full.json"),"w") as f: json.dump(results,f,indent=2)


# ═══════════════════════════════════════════
# 3. COMPLETE FAIR BASELINES (all 10, unified protocol)
# ═══════════════════════════════════════════
def fair_baselines_all():
    print("\n"+"="*50+"\n3. Complete Fair Baselines (10 instances)\n"+"="*50)
    K=8; results=[]
    for path in INSTS:
        inst=LAM(path); name=Path(path).stem
        gen=AssignmentGenerator(inst,seed=SEED); seq=HeuristicSequencer(inst,seed=SEED)
        ops=inst.all_ops; nm=inst.num_machines
        print(f"\n  [{name}]")
        row={"instance":name}
        # Balanced
        ba=gen.balanced_assignment(); be=FixedAssignmentSequencingModel(inst,ba,time_limit=SEQ_TO).solve()
        row["balanced"]=be.get("makespan")
        # Random-MS (same candidate count ~3000)
        n_rand=min(3000,inst.total_ops*30)
        # Score and retain the same assignment.  The old expression generated
        # two unrelated random assignments, invalidating the comparison.
        rand_pool=[]
        for _ in range(n_rand):
            candidate = gen.random_assignment()
            rand_pool.append((seq.schedule(candidate,"best")[0], candidate))
        rand_pool.sort(key=lambda x:x[0])
        best_rand=float("inf")
        for _,a in rand_pool[:K]:
            e=FixedAssignmentSequencingModel(inst,a,time_limit=SEQ_TO).solve()
            if e.get("has_solution") and e["makespan"]<best_rand: best_rand=e["makespan"]
        row["random_ms"]=best_rand
        # MaxLoad beam
        def get_l(a):
            ld={}
            for (jj,pp),mk in a.items():
                for alt in inst.jobs[jj][pp]:
                    if alt[0]==mk: ld[mk]=ld.get(mk,0)+alt[1];break
            return ld
        def ml_score(a): ld=get_l(a); return -max(ld.values()) if ld else 0
        def beam_ml():
            pool=[]; seen=set()
            for nm_g,a_ in gen.generate_all_heuristic().items():
                sig=tuple(sorted(a_.items()))
                if sig not in seen: seen.add(sig); pool.append(a_)
            for _ in range(4):
                cands=list(pool)
                for cur in pool:
                    for (j,pp) in ops:
                        f=inst.feasible_machines.get((j,pp),list(range(nm)))
                        for nk in f:
                            if nk==cur.get((j,pp)): continue
                            c=dict(cur); c[(j,pp)]=nk
                            sig=tuple(sorted(c.items()))
                            if sig not in seen and len(cands)<2000: seen.add(sig); cands.append(c)
                cands.sort(key=ml_score,reverse=True); pool=cands[:10]
            return pool
        best=float("inf")
        for a in beam_ml()[:K]:
            e=FixedAssignmentSequencingModel(inst,a,time_limit=SEQ_TO).solve()
            if e.get("has_solution") and e["makespan"]<best: best=e["makespan"]
        row["maxload_beam"]=best
        # HeurSeq beam
        def hs_score(a): return -seq.schedule(a,"best")[0]
        def beam_hs():
            pool=[]; seen=set()
            for nm_g,a_ in gen.generate_all_heuristic().items():
                sig=tuple(sorted(a_.items()))
                if sig not in seen: seen.add(sig); pool.append(a_)
            for _ in range(3):
                cands=list(pool)
                for cur in pool:
                    for (j,pp) in ops:
                        f=inst.feasible_machines.get((j,pp),list(range(nm)))
                        for nk in f:
                            if nk==cur.get((j,pp)): continue
                            c=dict(cur); c[(j,pp)]=nk
                            sig=tuple(sorted(c.items()))
                            if sig not in seen and len(cands)<1500: seen.add(sig); cands.append(c)
                cands.sort(key=hs_score,reverse=True); pool=cands[:10]
            return pool
        best=float("inf")
        for a in beam_hs()[:K]:
            e=FixedAssignmentSequencingModel(inst,a,time_limit=SEQ_TO).solve()
            if e.get("has_solution") and e["makespan"]<best: best=e["makespan"]
        row["heurseq_beam"]=best
        # RF beam
        rf,feats,bp,r2=train_rf_fast(inst)
        ranked=beam_search(inst,bp)
        best=float("inf")
        for _,a in ranked[:K]:
            e=FixedAssignmentSequencingModel(inst,a,time_limit=SEQ_TO,mip_gap=0.0).solve()
            if e.get("has_solution") and e["makespan"]<best: best=e["makespan"]
        row["rf_beam"]=best; row["rf_r2"]=round(r2,3)
        print(f"    Bal={row['balanced']:.0f} Rand-MS={row['random_ms']:.0f} MaxLoad={row['maxload_beam']:.0f} HeurSeq={row['heurseq_beam']:.0f} RF={row['rf_beam']:.0f}")
        results.append(row)
    with open(os.path.join(RESULTS_DIR,"fair_baselines_full.json"),"w") as f: json.dump(results,f,indent=2)


# ═══════════════════════════════════════════
# 4. STRUCTURED FAILURE ANALYSIS  
# ═══════════════════════════════════════════
def failure_analysis_structured():
    print("\n"+"="*50+"\n4. Structured Failure Analysis\n"+"="*50)
    failures=[("data/instances/brandimarte/mk04.json","mk04"),("data/instances/brandimarte/mk06.json","mk06"),("data/instances/brandimarte/mk10.json","mk10")]
    results=[]
    for path,name in failures:
        inst=LAM(path); gen=AssignmentGenerator(inst,seed=SEED); seq=HeuristicSequencer(inst,seed=SEED)
        print(f"\n  [{name}]")
        def analyze(a,label):
            loads={}
            for (jj,pp),mk in a.items():
                for alt in inst.jobs[jj][pp]:
                    if alt[0]==mk: loads[mk]=loads.get(mk,0)+alt[1];break
            ld=list(loads.values())
            jt=[]
            for j in range(inst.num_jobs):
                total=0
                for pp in range(len(inst.jobs.get(j,[]))):
                    mk=a.get((j,pp))
                    if mk is not None:
                        for alt in inst.jobs[j][pp]:
                            if alt[0]==mk: total+=alt[1];break
                jt.append(total)
            same_mach=0; diff_mach=0
            for j in range(inst.num_jobs):
                for pp in range(len(inst.jobs.get(j,[]))-1):
                    if a.get((j,pp))==a.get((j,pp+1)): same_mach+=1
                    else: diff_mach+=1
            return {"max_load":max(ld) if ld else 0,"min_load":min(ld) if ld else 0,
                "imbalance":max(ld)-min(ld) if len(ld)>1 else 0,"n_used":len(ld),
                "max_jobtime":max(jt) if jt else 0,"total_load":sum(ld),
                "same_mach_trans":same_mach,"diff_mach_trans":diff_mach}
        ba=gen.balanced_assignment(); be=FixedAssignmentSequencingModel(inst,ba,time_limit=SEQ_TO).solve()
        rf,feats,bp,r2=train_rf_fast(inst); ranked=beam_search(inst,bp)
        rf_a=ranked[0][1] if ranked else ba; rfe=FixedAssignmentSequencingModel(inst,rf_a,time_limit=SEQ_TO).solve()
        bs=analyze(ba,"Balanced"); rs=analyze(rf_a,"RF")
        hb,_=seq.schedule(ba,"best"); hr,_=seq.schedule(rf_a,"best")
        row={"instance":name,"bal_makespan":be.get("makespan"),"rf_makespan":rfe.get("makespan"),
            "rf_r2":round(r2,3),"heuristic_bal":hb,"heuristic_rf":hr}
        row.update({f"bal_{k}":v for k,v in bs.items()}); row.update({f"rf_{k}":v for k,v in rs.items()})
        print(f"    Bal={be.get('makespan'):.0f} (max_load={bs['max_load']:.0f}) vs RF={rfe.get('makespan'):.0f} (max_load={rs['max_load']:.0f})")
        results.append(row)
    with open(os.path.join(RESULTS_DIR,"failure_structured.json"),"w") as f: json.dump(results,f,indent=2)


if __name__=="__main__":
    correct_stats()
    proxy_validation_all()
    fair_baselines_all()
    failure_analysis_structured()
    print("\n"+"="*50+"\nALL EXPERIMENTS COMPLETE\n"+"="*50)
