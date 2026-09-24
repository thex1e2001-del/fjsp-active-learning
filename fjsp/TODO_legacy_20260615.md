> 历史存档，不是当前执行依据。旧数字和要求可能已被审计否定；请勿按本文件恢复结论。当前任务见 [TODO.md](TODO.md) 和 [实验路线](../EXPERIMENT_ROADMAP.md)。

# TODO — Final Revision (Based on 审阅意见建议 2026-06-15)

## 🔴 P0: 硬伤 — 数据和结论矛盾（不修直接拒）

### 1. 统一所有实验表 — 同一 protocol/seed/settings
- [ ] Table 3/4/5/6/7 必须来自同一实验设置
- [ ] 所有方法：相同候选生成上限、相同 top-K verification、相同 30s/60s/120s、相同 seeds、相同 solver
- [ ] 表中不允许有缺失值，不能只有部分实例

### 2. 重算所有 summary 数字
- [ ] 按 Table 3 重算全样本平均改善 → 应为 ~3.32%（当前写 1.81% 不对）
- [ ] 按 Table 3 重算 winning-instance 改善 → 应为 ~6.84%（当前写 4.29% 不对）
- [ ] Friedman rank 重算 → RF≈1.25, Balanced≈1.90（当前写 RF=1.20 需确认 tie-handling）
- [ ] Wilcoxon 透明化：报告 two-sided/one-sided/zero-handling (wilcox/Pratt/zsplit)/exact or normal/effect size
- [ ] 删掉 "p=0.053" 单一句式

### 3. 补完整 Proxy Ranking Validation — 全 10 实例
- [ ] 每实例三类采样：random / heuristic-neighborhood / beam-generated
- [ ] 每实例报告：Pearson, Spearman, Kendall, Pairwise accuracy, MAE, MAPE, Recall@5, Recall@10, Regret@5, Regret@10
- [ ] 正面讨论 mk01 Recall@5=0.20：说明 top-tail ranking 在某些实例有问题
- [ ] 核心指标改为 Regret@K

### 4. 补完整 Fair Baselines — 全 10 实例
- [ ] Random-MS, MaxLoad, LB, Imbalance, HeurSeq, Balanced-LS, RF — 全 10 实例
- [ ] 统一 candidate budget 和 verification budget
- [ ] 诚实报告：简单 workload scorer 在多数实例上 ≈ RF

### 5. 补 MIP Verification Status 表
- [ ] Per instance: #verified, #optimal, #feasible-only, avg gap, max gap, best candidate status
- [ ] 30s verification 到底求到了什么必须说清楚

### 6. 改结论
- [ ] 从 "RF consistently improves" → "RF is competitive; simple workload scorers remain strong; RF helps only under certain conditions"
- [ ] Abstract 删除任何 SOTA 暗示，加入 "simple workload-based scores remain strong"

### 7. 补定量 Failure Analysis
- [ ] mk04/mk06/mk10 完整诊断表：Balanced/MaxLoad/HeurSeq/RF/RF+precedence features 各列
- [ ] 指标：makespan, max_load, max_jobtime, load_std, speedloss_mean, same-job transitions, critical machine load, MIP gap, heuristic label, RF score
- [ ] mk04: 证明 RF 选了负载更好但 sequencing 兼容性更差的 assignment

---

## 🟠 P1: 方法补强

### 8. 特征扩展 — 从 3m+4 扩到四组
- [ ] A. Machine workload: load_k, ops_k, used_k, max_load, load_std, load_Gini
- [ ] B. Job-path: jobTime_j, max/mean/std/top-3/Gini, bottleneck_ratio
- [ ] C. Routing-efficiency: speedLoss mean/max/std, fastest-machine ratio, total detour cost
- [ ] D. Precedence-aware: same-job consecutive same/different machine count, stage-level load, critical-job concentration
- [ ] 加 Feature table

### 9. Beam search 加 diversity mechanism
- [ ] Hamming distance: d(x,x') = (1/|O|)·Σ 1{k(x)≠k(x')}
- [ ] score(x) = RF(x) - α·min_{x'∈S} d(x,x')
- [ ] 或 top-3B 后 diversity filtering

### 10. 加 Exact-label calibration（可选 but 加分）
- [ ] r(x) = C_MIP(x) - C̃_heur(x)
- [ ] Ĉ_final(x) = Ĉ_heur(x) + r̂(x)
- [ ] heuristic-label pretraining + exact-label residual calibration

### 11. 可复现性细节
- [ ] Heuristic sequencer: ready set definition, tie-breaking, insertion logic, complexity
- [ ] Random assignment generation method
- [ ] Beam search deduplication
- [ ] Train/test split
- [ ] Training assignment proportions (40% random, 40% neighborhood, 20% beam)
- [ ] RF hyperparameters tuning or not
- [ ] Gurobi: gap tolerance, threads, time limit
- [ ] 10 random seeds 覆盖哪些随机过程

---

## 🟡 P2: 正文扩写

### 12. Introduction → 2–2.5 页
- [ ] P1: Manufacturing motivation
- [ ] P2: Routing-sequencing coupling
- [ ] P3: Traditional methods
- [ ] P4: Learning-based scheduling opportunities/limitations
- [ ] P5: Our approach — interpretable screening, not end-to-end
- [ ] P6: RQ1–RQ4 + contributions

### 13. Related Work → 4–5 页, 7 小节
- [ ] 2.1 FJSP problem, benchmarks, BKS reporting
- [ ] 2.2 Exact / MIP / CP formulations
- [ ] 2.3 Metaheuristics and hybrid heuristics
- [ ] 2.4 Decomposition in scheduling
- [ ] 2.5 Learning-based scheduling (DRL/GNN)
- [ ] 2.6 Surrogate-assisted combinatorial optimization
- [ ] 2.7 Tree-in-MIP / constraint learning + position of this work
- [ ] 加 Literature taxonomy table

### 14. Problem & Lower Bounds → 2–3 页
- [ ] Notation table
- [ ] Full MIP formulation (完整约束)
- [ ] Proof of Proposition 1 & 2
- [ ] Example showing same max_load → different sequencing makespan
- [ ] Feature motivation paragraph

### 15. Methodology → 4–5 页, 7 小节
- [ ] 4.1 Assignment representation and feature groups (加 feature table)
- [ ] 4.2 Heuristic sequencer (加 pseudocode)
- [ ] 4.3 Label generation and sampling design
- [ ] 4.4 Surrogate model (why RF, cite Breiman 2001)
- [ ] 4.5 Diversity-aware RF beam search (加 pseudocode)
- [ ] 4.6 MIP/CP-based sequencing verification
- [ ] 4.7 Optional exact-label calibration

### 16. Experiments → 完整 8 主实验
- [ ] Exp 1: Benchmark and instance characterization (加 Kacem small, Hurink, synthetic)
- [ ] Exp 2: Proxy label validation (全实例, Recall/Regret)
- [ ] Exp 3: Main results under equal verification budget
- [ ] Exp 4: Main results under equal wall-clock budget (加 CP-SAT, Full MIP)
- [ ] Exp 5: Scorer ablation (全10实例, 10 scorers)
- [ ] Exp 6: Feature group ablation (F1→F4)
- [ ] Exp 7: Search parameter diagnostics (B×iter grid, K sensitivity)
- [ ] Exp 8: Failure case analysis (结构化诊断表)
- [ ] Appendix A: Training sample learning curve
- [ ] Appendix B: Exact-label calibration experiment
- [ ] Appendix C: Scale/flexibility synthetic experiments
- [ ] Appendix D: Rolling-horizon managerial experiment

### 17. Discussion → 4 段机制讨论
- [ ] What did the surrogate actually learn? (Feature importance, SHAP)
- [ ] When is learned guidance better than workload scoring? (by size/flexibility/R²/regret)
- [ ] Why does it fail? (mk04: precedence; mk06/mk10: sparsity/coverage/diversity)
- [ ] Managerial implications (绑定 rolling-horizon experiment)

### 18. Conclusion
- [ ] 不重复摘要
- [ ] 提出了什么 | 发现了什么 (RF useful but not uniformly dominant) | 边界是什么

---

## 🟢 P3: 引用和格式

### 19. 引用扩到 35–50 篇
- [ ] A. FJSP benchmarks (10 篇): Brucker, Brandimarte, Hurink, Dauzère, Kacem, Fattahi, Behnke, Birgin, Pinedo
- [ ] B. Exact/MIP/CP (5 篇): Ozgüven, Meng, OR-Tools CP-SAT, IBM CP Optimizer, Gurobi
- [ ] C. Metaheuristics (7 篇): Mastrolilli, Pezzella, Gao, Xia&Wu, Li&Gao, Zheng&Wang, Xie
- [ ] D. Decomposition (3 篇): Adams, Fattahi, Brandimarte
- [ ] E. ML4CO (6 篇): Bengio, Khalil, Gasse, Hutter, He, Kotary
- [ ] F. Learning-based scheduling (5 篇): Zhang, Song, Wang, Moon, Smit
- [ ] G. Surrogate & RF (4 篇): Jin, Breiman, Volpato, Kotsiopoulos
- [ ] H. Tree-in-MIP (5 篇): Verwer, Maragno, Bertsimas&Dunn, Elmachtoub, Biggs
- [ ] I. Statistical testing (4 篇): Demšar, García, Benavoli, Wilcoxon
- [ ] 删除未使用的 arXiv placeholder (Keegan2025, Abdellaoui2026, Xefteris2026 如果正文未讨论)

### 20. 格式修整
- [ ] Author/University → 填真实信息
- [ ] Data Availability → 真实 GitHub URL
- [ ] FJSPLib BKS 来源正式引用 + 访问日期
- [ ] 表格编号和正文引用统一
- [ ] PDF 交叉引用去掉红框
- [ ] seed 口径统一 ("fixed seed 42" vs "10 random seeds")
- [ ] Gurobi/CP-SAT 参数完整报告

### 21. RF-embedded MIP Appendix 降调
- [ ] 写明 "conceptual extension only, not part of computational contribution"
- [ ] 写明 weighted objective may distort true makespan

### 22. Abstract 重写
- [ ] 删除 "consistent improvement over simpler scoring methods"
- [ ] 加入 "simple workload-based scores remain strong"
- [ ] 加入 "not intended as high-performance FJSP solver"
- [ ] 加入 "screening under limited verification budget"

---

## ✅ Done from previous rounds
- [x] Basic structure and LaTeX formatting
- [x] Main results table (Balanced vs RF) on 10 instances
- [x] BKS gap table
- [x] Partial fair baselines (5 instances)
- [x] Partial surrogate ablation (3 instances)
- [x] Top-K sensitivity (3 instances)
- [x] Beam width sensitivity (3 instances)
- [x] Wilcoxon/Friedman code (needs recalculation)
- [x] RF-embedded MIP appendix
