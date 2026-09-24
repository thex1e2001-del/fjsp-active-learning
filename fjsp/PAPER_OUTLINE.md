> 历史阶段计划（保留追溯）。当前执行优先级、实验参数和迁移状态以 [EXPERIMENT_ROADMAP.md](../EXPERIMENT_ROADMAP.md) 与 [MIGRATION_HANDOFF.md](../MIGRATION_HANDOFF.md) 为准；本文中的未完成设想不代表已实现。

# Paper Outline — FJSP RF-Guided Decomposition Heuristic

## Title
> **An Interpretable Surrogate-Guided Decomposition Heuristic for Flexible Job Shop Scheduling**

Alternative:
> **Random-Forest-Guided Assignment Screening for Flexible Job Shop Scheduling: A Decomposition-Based Heuristic Approach**

## One-sentence Contribution
We propose an interpretable RF-guided assignment screening framework for FJSP that uses aggregate workload and routing features trained on low-cost heuristic labels to efficiently filter machine assignments before MIP-based sequencing verification.

## Positioning
- **NOT** claiming SOTA makespan performance
- **IS** claiming: lightweight, interpretable, lower-cost-than-exact-MIP screening heuristic
- **Honest about gaps**: reports BKS gap, MIP optimality gap, failure cases

---

## §1 Introduction (~2 pages)

1. **FJSP motivation**: routing+sequencing coupling → MIP scale explosion
2. **Existing approaches**: exact MIP (intractable on large), meta-heuristics (hand-crafted, no learning), ML end-to-end (opaque, training cost)
3. **Gap**: lack of lightweight, interpretable learned heuristics for screening assignments before expensive MIP
4. **Our approach**: aggregate features → RF surrogate trained on heuristic labels → beam search → MIP-based verification
5. **Contributions** (low-risk phrasing):
   - An interpretable surrogate-guided decomposition framework
   - A low-cost heuristic labeling procedure validated for ranking accuracy (not just Pearson r)
   - Controlled experiments with budget-matched baselines, BKS gap reporting, and ablation

## §2 Literature Review (~1.5 pages)

| Line | Key Works | Our Position |
|---|---|---|
| FJSP exact/CP/metaheuristic | Mastrolilli (2000), Brandimarte (1993), FJSPLib, CP-SAT | Not competing on makespan; our baseline is same-budget comparison |
| Decomposition & routing-sequencing separation | Shifting Bottleneck (Adams 1988), Xia & Wu (2005) | Learned surrogate replaces hand-crafted routing rules |
| Surrogate-assisted optimization / learning-guided search | Jin (2011), Maragno et al. (2023), Bengio et al. (2021) | Lightweight, interpretable surrogate for discrete assignment space |
| Tree-in-MIP (brief) | Verwer et al. (2017), OptiCL (Maragno 2023) | Motivate why per-operation encoding fails; our aggregate approach avoids this |

## §3 Problem and Lower-Bound Motivation (~1.5 pages)

### 3.1 FJSP Standard Formulation (compact)

### 3.2 Decomposition Insight
- Fixed assignment → JSP subproblem SEQ(x)
- Search space: O(m^N × N!) → decomposed to assignment search + sequencing

### 3.3 Lower-Bound Propositions
- **Proposition 1**: Cmax(x) ≥ max_k load_k(x)
- **Proposition 2**: Cmax(x) ≥ max_j jobTime_j(x)
- **Corollary**: LB(x) = max{LB_M, LB_J}

> These lower bounds motivate aggregate workload and job-route features as surrogate inputs.
> (Replaces the overclaimed "aggregate features are sufficient")

## §4 Methodology (~4 pages)

### 4.1 Assignment Representation
- Feature vector φ(x): per-machine (load_k, ops_k, used_k) + aggregate (max_load, imbalance, max_ops, max_job_time)
- **New**: job-level features (jobTime distribution, bottleneck ratio)
- **New**: routing-quality features (speedLoss, fastest-machine count)
- **New**: precedence-aware features (same-job consecutive ops, stage-level load)
- **New**: LB features (LB_M, LB_J, LB)

### 4.2 Heuristic Sequencing Label
- 5 dispatching rules → best makespan as proxy label
- **Validation**: Pearson, Spearman, Kendall, Recall@K, Regret@K
- **Three-class sampling**: random / heuristic-perturbed / beam-encountered assignments
- **Bias reporting**: systematic overestimation, but ranking preserved

### 4.3 Surrogate Training
- RF (150 trees, depth 12) trained on 2000–3000 heuristic-labeled assignments
- Learning curve: 100→5000 samples, performance plateau at ~1000–2000

### 4.4 Diversity-Aware RF Beam Search
- Multi-start: balanced, SPT, LPT, random assignments
- Neighborhood: single-operation machine changes
- **New**: Hamming-distance diversity filter: score(x) = RF(x) − α·d(x,pool)
- Batch RF prediction for efficiency

### 4.5 MIP-Based Sequencing Verification
- Top-K assignments verified via SEQ(x) MIP
- **30s time limit per candidate — reported with optimality gaps**
- Table: #verified, #optimal, #feasible-only, avg gap, max gap

### 4.6 Active Calibration (Optional Enhancement)
- Heuristic-label pretraining + exact-label residual correction
- r(x) = Cmax^MIP(x) − Cmax^heur(x)
- Final surrogate: Ĉmax = RF_heur(x) + RF_residual(x)

### Appendix (removed from main body)
- RF-Embedded MIP formulation (conceptual only, no experiments)

## §5 Experimental Design (~1.5 pages)

### 5.1 Instances
- Brandimarte MK01–MK10 (10 instances, 55–240 ops)
- **New**: Kacem small instances for cross-benchmark validation (optional)
- Instance characteristics table with BKS/UB/LB

### 5.2 Baselines (budget-matched)
| Baseline | Controls for |
|---|---|
| Random multi-start + K verification | "Is improvement just from evaluating more candidates?" |
| MaxLoad beam search | "Is RF just learning max load?" |
| Heuristic-sequencer beam search | "Is RF just learning the heuristic itself?" |
| Balanced + local search | "Can local search from Balanced match RF?" |
| Full FJSP MIP (same wall-clock) | "Is decomposition better than direct MIP under time limit?" |
| CP-SAT (same wall-clock) | "Constraint programming baseline" |

### 5.3 Surrogate Ablation Scorers
| Scorer | Purpose |
|---|---|
| MaxLoad | Lower-bound baseline |
| LB = max(LB_M, LB_J) | Full lower-bound baseline |
| Heuristic sequencer (no RF) | Does RF add value over direct heuristic? |
| Linear Regression | Is non-linearity needed? |
| CART (single tree) | Is ensemble needed? |
| GBDT/XGBoost | Is RF the best tree model? |

### 5.4 Metrics
- Makespan, gap to BKS/UB, wall-clock time
- Friedman + Nemenyi/Holm post-hoc, Wilcoxon with exact settings
- Win/Draw/Loss counts, median relative improvement
- Feature importance (SHAP)

### 5.5 Reproducibility
- 10 random seeds, report mean ± std
- Code and data availability statement

## §6 Results (~5 pages)

### 6.1 Proxy Label Validation (Table 2)
- Pearson, Spearman, Kendall, Recall@5, Regret@5, MAE, MAPE
- Per-instance, across 3 assignment sampling types
- Scatter plot: heuristic vs MIP makespan (Figure 1)

### 6.2 Main Comparison Under Equal Verification Budget (Table 3)
- Balanced | Random-MS | MaxLoad-Beam | HeurSeq-Beam | RF-Beam | RF+Calib
- Per instance: makespan, gap to BKS, total time, #verified

### 6.3 Main Comparison Under Equal Wall-Clock Budget (Table 4)
- Full MIP | CP-SAT | Balanced-LS | RF Search
- Per instance: makespan, total time

### 6.4 Gap to Best-Known Solutions (Table 5)
- RF Search vs BKS/UB vs LB, per instance
- Gap analysis: when does RF approach BKS, when does it fail

### 6.5 Ablation Study (Table 6)
- Feature groups: machine-load | +ops | +balance | +job | +routing | +precedence | full
- Surrogate models: MaxLoad | LB | LinReg | CART | RF | GBDT
- Beam parameters: B×iterations grid

### 6.6 Failure Case Analysis
- mk04: RF=78 vs Balanced=68 — why?
- mk06: RF=100 vs BKS=57 — why so far?
- mk10: RF=277 vs UB=193, R²≈0.5 — feature boundary

### 6.7 Computational Cost (Table 7)
- Per-instance: training, beam search, verification time breakdown
- Training sample sensitivity curve (Figure)

### 6.8 MIP Verification Status (Table 8)
- #verified, #optimal, #feasible-only, avg gap, max gap

## §7 Discussion (~1 page)

- **When RF helps**: medium instances with moderate flexibility, routing dominates sequencing difficulty
- **When RF fails**: large flexible instances where aggregate features lose discriminative power
- **Boundary conditions**: low-dimensional features insufficient when machine interactions and precedence dominate
- **Managerial insights**:
  1. Interpretable assignment screening: RF uses workload/routing/bottleneck features → managers understand why
  2. Low-cost candidate filtering: screen many routing alternatives, send few to exact solver
  3. Deployment scope: suitable for rolling-horizon where full MIP is too slow

## §8 Conclusion (~0.5 page)
- Summary: lightweight, interpretable surrogate-guided decomposition
- Honest about limitations: not SOTA, but practical and transparent
- Future: cross-instance transfer, iterative label correction, extension to other variants

---

## Figures (5)

| # | Content |
|---|---|
| Fig 1 | Heuristic vs MIP scatter (3 instances, with Pearson/Spearman annotation) |
| Fig 2 | Beam search convergence curve (RF score, actual makespan, diversity over iterations) |
| Fig 3 | Feature importance / SHAP (top-15 features) |
| Fig 4 | Training sample learning curve (x=samples, y=Spearman/final_makespan) |
| Fig 5 | Gap-to-BKS bar chart (all instances, multi-method) |

## Tables (8)

| # | Content | Rows |
|---|---|---|
| Table 1 | Instance characteristics + BKS/UB/LB | 10 |
| Table 2 | Proxy label validation (Pearson/Spearman/Kendall/Recall/Regret) | 10 |
| Table 3 | Main results: equal verification budget | 10 × 6 methods |
| Table 4 | Main results: equal wall-clock budget | 10 × 4 methods |
| Table 5 | Gap to best-known solutions | 10 |
| Table 6 | Ablation: feature groups + surrogate models | ~20 configs |
| Table 7 | Computational cost breakdown | 10 |
| Table 8 | MIP verification status | 10 |
