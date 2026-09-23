# Workflow — FJSP Paper Revision

## Phase 1: P0 Hard Fixes (Day 1)

### 1.1 Collect correct numbers
- [ ] Re-extract Exp1 data from results/exp1_main.json
- [ ] Recompute Friedman ranks correctly (sum=15 per instance)
- [ ] Recompute Wilcoxon with exact settings (one-sided, Pratt zero-handling)
- [ ] Compute all-instance average improvement (not just winning instances)
- [ ] Count Win/Draw/Loss correctly (7/2/1)

### 1.2 Collect missing proxy validation data
- [ ] Run `complete_experiments.py --exp4-only` for mk09, mk10
- [ ] Or write `proxy_validation_extended.py`:
  - 100 random + 100 heuristic-perturbed + 100 beam-encountered assignments per instance
  - Compute: Pearson, Spearman, Kendall, Recall@5, Regret@5, MAE, MAPE

### 1.3 Collect MIP verification status
- [ ] Write `mip_gap_report.py`:
  - For each instance's top-8 verified candidates, record: status (optimal/feasible/timeout), MIP gap
  - Output: results/mip_verification_status.csv

### 1.4 Fix paper text
- [ ] Search-replace all inflated claims in manuscript.tex:
  - `sufficient` → `informative`
  - `exact MIP guarantees` → `MIP-based verification`
  - `complete Brandimarte` → `Brandimarte MK01–MK10 subset`
  - `strongest rule-based` → `selected rule-based`
  - `8/1/1` → `7/2/1`
  - `5.7%` → `3.32% (all); 5.98% (winning)`

---

## Phase 2: P1 Fairness Experiments (Day 2)

### 2.1 Equal-candidate comparison
- [ ] Write `fair_baselines.py`:
  - Random multi-start: generate N assignments (same count as RF beam explored), verify top-K
  - MaxLoad beam: same beam search, score = max_load (lower is better)
  - Heuristic-sequencer beam: same beam search, score = heuristic makespan directly
  - Compare all + RF beam on same instances
- [ ] Run on MK01–MK10
- [ ] Output: results/fair_comparison.csv

### 2.2 BKS gap table
- [ ] Write `bks_gap.py`:
  - Hardcode BKS/UB/LB values from FJSPLib
  - Compute gap for each method
  - Output LaTeX table

### 2.3 Same wall-clock baseline
- [ ] Write `wall_clock_baselines.py`:
  - For each instance, record RF total time T
  - Run Full FJSP MIP with time limit = T
  - Run Balanced + local search with time limit = T
  - Optional: CP-SAT with time limit = T
  - Output: results/wall_clock_comparison.csv

---

## Phase 3: P2 Mechanism Evidence (Day 2-3)

### 3.1 Feature group ablation
- [ ] Write `feature_ablation.py`:
  - Define feature groups: machine-load, +ops, +balance, +job, +routing, +precedence, full
  - For each group: train RF → beam search → exact verify
  - Output: results/feature_ablation.csv

### 3.2 Surrogate scorer comparison
- [ ] Write `surrogate_comparison.py`:
  - Scorers: MaxLoad, LB, Heuristic-sequencer, LinearRegression, CART, RF, GBDT
  - For each: use as beam search scorer → exact verify
  - Output: results/surrogate_comparison.csv

### 3.3 New features implementation
- [ ] Add to `assignment_generators.py` / `experiments_final.py`:
  - job-level: jobTime max/mean/std/gini, bottleneck_ratio
  - routing-quality: speedLoss mean/max, fastest_machine_count
  - precedence-aware: same_machine_consecutive, stage_load
  - LB features: LB_M, LB_J, LB
- [ ] Re-train RF with new features, re-run beam search

### 3.4 Failure case diagnosis
- [ ] Write `failure_analysis.py`:
  - mk04: compare Balanced vs RF assignment feature-by-feature
  - mk06: analyze top-K diversity, heuristic label accuracy, SEQ MIP gaps
  - mk10: chart R² vs instance size, feature importance degradation

### 3.5 Fix Table 2/3 inconsistency
- [ ] Identify which run produced mk01=43 vs mk01=49
- [ ] Re-run with unified parameters and fixed seed
- [ ] Ensure all tables pull from the same experiment output

---

## Phase 4: P3 Bonuses (Day 3-4, optional)

### 4.1 Beam diversity
- [ ] Add Hamming distance filter to `beam_search()` in experiments_final.py
- [ ] Compare diversity-aware vs vanilla beam on 3 instances

### 4.2 Learning curve
- [ ] Run beam search with 100/300/500/1000/2000/3000/5000 training samples
- [ ] Plot x=samples vs y=final_makespan/Spearman

### 4.3 Top-K sensitivity
- [ ] Run beam search with K=1,3,5,8,10,15,20
- [ ] Plot x=K vs y=final_makespan/verification_time

### 4.4 Beam width × iterations grid
- [ ] Run B=5,10,15,30 × iter=N/40,N/20,N/10

### 4.5 RF-embedded MIP → appendix
- [ ] Move §4.4 content to Appendix A
- [ ] Main text: one sentence mention

### 4.6 Management insights
- [ ] Add §6.7 or §7 subsection: three managerial takeaways

### 4.7 Cross-benchmark (optional)
- [ ] Run on Kacem instances (ka01–ka05)

---

## Phase 5: Paper Rewrite (Day 4-5)

### 5.1 Structural changes
- [ ] Reorder sections per PAPER_OUTLINE.md
- [ ] Add Propositions 1–2 in §3
- [ ] Move RF-embedded MIP to Appendix
- [ ] Add Management Insights in §7

### 5.2 Tables & Figures
- [ ] Generate all 8 tables (LaTeX-ready)
- [ ] Generate all 5 figures (PDF/PNG)
- [ ] Ensure all numbers match across abstract/body/tables

### 5.3 Final checks
- [ ] All claims match data
- [ ] No "sufficient", "guarantees", "complete" language
- [ ] Statistical tests transparently reported
- [ ] Code and data availability statement
- [ ] Author info filled in
