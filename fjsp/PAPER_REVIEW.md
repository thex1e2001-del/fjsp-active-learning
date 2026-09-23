# 论文与实验系统性审阅

审阅日期：2026-09-24  
审阅对象：`fjsp/paper/manuscript.tex`、`fjsp/` 实验代码、`fjsp/results/` 保存结果  
目标：评估 EI 会议/期刊投稿可行性，并给出两个月内可执行的最小改进路线。

## 总体判断

当前稿件**不建议直接投稿**。问题集中在证据链和定位，而不是缺少更复杂的模型：论文把一个有实用直觉的轻量筛选流程写成了方法贡献，但现有实验还不足以支持“RF 有效”“公平比较”“可解释泛化”等更强结论。

当前投稿准备度约为 **3/10**：代码和数据基础较完整，稿件结构也已形成；但公平预算、独立测试、重复实验、候选级求解状态和实现—论文一致性仍有阻断问题。完成下面的 P0 修订后，可以作为一篇**范围克制的应用/比较型 EI 论文**，准备度可提高到约 **6/10**。它仍不适合包装成新的 FJSP 求解理论或 SOTA 算法。

最适合的核心叙事是：

> 在有限排序验证预算下，轻量级分配特征和 RF 是否能帮助筛选 FJSP 机器分配？在哪些实例结构上有用，何时不如简单负载规则？

这比“提出一个优越的 RF 调度算法”更符合现有证据，也更容易回答审稿人的质疑。

## 现有工作的优点

1. 问题拆分清楚：把 routing 和 fixed-assignment sequencing 分开，工程流程容易理解。
2. 代码结构相对完整，包含数据读取、规则分配、启发式排序器、固定分配 MIP、RF 搜索和结果目录。
3. 论文已经承认方法不是 SOTA，并保留了 mk04 等失败现象；这种诚实定位适合应用型投稿。
4. Brandimarte MK01–MK10 是熟悉的公开基准，审稿人容易复核。
5. 当前保存结果至少可以支持一个保守的描述性结论：单一历史协议下，RF 相对 Balanced 为 6 胜、2 平、2 负，平均改善 1.8125%，不能宣称显著优越。

## P0：不修就容易被拒的阻断问题

### 1. 论文描述的特征没有被当前主实验实现

论文表格和方法段写了四组特征：machine workload、job path、routing efficiency、precedence-aware。但主实验的 `BatchPredictor.extract_one()` 只计算机器负载、机器工序数、使用标记、最大负载、负载不均衡、最大作业时间和负载标准差，见 `fjsp/experiments_final.py`；`rf_train.py:81-98` 也只构建这些列。代码中没有 speed loss、Gini、stage load 或 same-job transition 特征。

这会让审稿人认为方法部分夸大了实现。必须二选一：

- 最快方案：把论文特征表改成当前真正使用的特征，并删除 routing/precedence 的贡献声称；
- 较强方案：实现这些特征，做独立 feature-group ablation，并把所有结果重新生成。

对两个月 EI 目标，建议采用第一种，保留“低维 workload-based representation”定位。

### 2. RF 与 Balanced 的主比较不公平

主实验中 Balanced 只验证一个分配，而 RF 训练模型、进行束搜索，再验证 Top-8 候选。两者消耗的候选生成、搜索和 MIP 验证预算完全不同。RF 的较低 makespan可能来自评估了更多候选，不能归因于 RF。

当前 `fair_baselines.json` 只有 mk09 一行，不能支撑正文中的全基准公平比较。并且历史公平脚本直到最近才修复了“用于评分的随机候选”和“实际保留的随机候选”不一致问题。

必须补一个统一协议：所有 scorer 使用同一候选池、相同 Top-K、相同 MIP 时间上限；同时报告总墙钟时间。至少比较 Balanced、Random-MS、MaxLoad/LB、HeurSeq、RF。没有这张全实例表，RF 的主要贡献无法成立。

### 3. 没有真正验证 RF 的泛化或筛选能力

`train_global_rf()` 输出的 R² 是在训练数据上调用 `rf.score(X, y)`，属于 in-sample fit，不能称作预测能力。`final_experiments.py:84-105` 的 proxy validation 测的是启发式排序标签与限时 MIP 结果的相关性，而不是 RF 预测与独立 MIP 结果的相关性。

论文必须区分三件事：

- heuristic label 与 MIP 结果是否相关；
- RF 在未见分配上的预测误差和排名能力；
- RF 选出的 Top-K 经 MIP 验证后是否优于基线。

最低要求是按实例划分训练/测试分配，或使用固定的 5-fold split；测试分配不得参与 RF 训练、beam search 调参和最终报告。报告 test Spearman、MAE、Recall@K、Regret@K。

### 4. 只有单一历史种子，统计结论不够可靠

当前主表是单一 seed 的 10 个实例，不是 10 次独立重复。按现有保存结果复算，RF–Balanced 的双侧 Pratt Wilcoxon 为 `p=0.1236`。因此论文必须使用描述性语言，不能写“统计显著”“consistent improvement”或暗示稳定泛化。

建议每实例至少 5 个预先固定的 seed；若预算允许用 10 个。把 seed 视为同一实例内的重复，最终统计仍以实例为配对单位，报告均值、标准差、最好值和可行解比例。不要把 10 个 seed 伪装成 50 个独立数据集。

### 5. MIP 验证状态没有进入结果表

固定分配 MIP 设置了 30 秒和 `MIPGap=0.0`，但大实例可能只得到 feasible incumbent。当前结果文件主要保存 makespan，没有逐候选 status、bound、gap、wall time 和 candidate ID。论文却多次使用“exact sequencing”“exact verified”一类容易被理解为最优证明的词。

统一使用“time-limited fixed-assignment MIP verification”，并保存：`status`、`SolCount`、incumbent makespan、best bound、MIP gap、wall time。若比较 BKS，明确说明这是可行解与 BKS 的差距，不能当作算法最优性差距。

### 6. 历史结果与当前代码不是同一实验版本

`exp1_main.json` 是旧协议结果；后来修复了 RF 扰动的全局随机种子和公平基线候选错配。当前代码重新运行未必得到同样的主表。仓库 README 已将其标为历史结果，这是正确的；但正式稿不能同时声称“代码可直接复现主表”和“主表来自旧版本”。

正式投稿前要么用当前代码重跑并覆盖正式结果，要么在结果文件中记录 commit、依赖版本、参数和生成日期，把旧表明确命名为 exploratory archive。

## P1：强烈建议补充的实验与方法修订

### 7. 加强 baseline，但不要追求大而全

Random、SPT、LPT、Balanced 是合理 sanity checks，却不能代表 FJSP 领域的强基线。最低可行组合是：

- Balanced；
- Random multi-start；
- MaxLoad 或 `max{machine load, job time}` lower bound；
- HeurSeq scorer；
- RF scorer；
- 一个公开实现或可靠复现的 tabu/GA/CP-SAT 参照，作为外部质量参照。

如果无法公平复现强元启发式，就明确说本文研究的是“screening under limited verification budget”，不要把结果和 SOTA 直接排名。

### 8. 重新定义代理验证实验

现在的 40 个随机分配、每个 5 秒 MIP、`mip_gap=0.01` 可以作为 pilot，但不够做核心证据。建议每实例固定生成一个候选池，例如 500 或 1,000 个分配，所有 scorer 共享该池；先用廉价 heuristic 给出排序，再用同一限时 MIP 对统一 Top-K 验证。报告：

- Spearman/Kendall；
- Recall@1、Recall@5、Recall@10；
- Regret@K；
- scorer 时间；
- MIP 可行比例和平均 gap。

这样能直接回答“筛选器是否减少昂贵验证”，比只报告 Pearson 更有说服力。

### 9. 预算必须同时按候选数和总时间控制

只比较 Top-K 数量仍可能偏袒 RF，因为 RF 训练和 beam 候选生成有额外成本。正文应提供两种协议：

1. equal-candidate/equal-verification：相同候选数、相同 K、相同单候选时间；
2. equal-wall-clock：RF 训练、候选生成和 MIP 验证全部计入总时限。

主结论以第二种为主，第一种作为机制分析。

### 10. 修正复杂度表述

启发式排序器在每步扫描 ready operations 和前序操作，当前实现不能简单写成每规则 `O(N·m)`；最坏情况下更接近二次级别，具体还取决于作业长度和实现细节。建议删除过于精确的线性复杂度声称，改为“避免 MIP，适合批量候选预筛选”，并实测每实例每千候选耗时。

### 11. “可解释性”需要具体证据

当前 RF 只有可读输入特征，没有独立 feature importance、局部解释或决策案例。可选方案：

- 最快方案：将“interpretable”改为“feature-transparent”，只解释特征定义和筛选规则；
- 稍强方案：在独立测试集上报告 permutation importance，并展示一个 Balanced 与 RF 候选的特征—makespan 对照。

不要把 RF 的默认 impurity importance 当成因果解释。

## 论文结构与文字建议

### 建议保留

- FJSP routing–sequencing 分解动机；
- workload/job-time lower bounds 作为特征动机；
- heuristic label 的局限和 MIP-based verification；
- mk04 失败和高柔性实例的边界条件；
- 公开基准和可复现代码。

### 建议删除或降级

- RF-embedded MIP appendix：当前只是概念公式，没有计算实验，容易引出新的建模和目标错配质疑；
- RQ4 “managerial value”：没有生产现场、滚动窗口或用户实验，不能凭 benchmark 推导管理价值；
- “four feature groups”“diversity-aware beam search”“10 independent repetitions”：若没有对应代码和结果，应全部删除；
- 将简单 Balanced 对比写成 RF 优越性；
- 用 Pearson 相关性作为主要贡献指标。

### 建议标题

首选：

> **Random-Forest-Assisted Candidate Screening for Flexible Job Shop Scheduling: An Exploratory Budget-Matched Study**

如果补齐公平实验，可以用：

> **Interpretable Surrogate Screening of Machine Assignments for Flexible Job Shop Scheduling under Limited Verification Budgets**

现标题 “A Surrogate-Assisted Decomposition Framework for Interpretable Assignment Screening” 偏方法论文，容易让审稿人按创新算法标准审查，当前证据不够匹配。

## 最小可投稿实验方案

### 阶段 A：代码和记录修复

1. 把真实特征列表集中定义在一个模块，论文表格自动从该列表生成。
2. 给 `train_rf_fast`、`beam_search` 和所有 generator 显式传入 seed，不使用隐含的全局随机状态。
3. 每个 candidate 保存 assignment hash、特征、scorer、预测值、MIP status、bound、gap、makespan 和耗时。
4. 增加独立可行性检查：机器资格、每个工序恰好一次、作业内先后、同机不重叠和 makespan 下界。

### 阶段 B：统一候选池实验

对 MK01–MK10，每实例 5 个 seed；每个 seed 生成同一候选池。比较 Balanced、Random-MS、MaxLoad/LB、HeurSeq 和 RF。建议先用 5 秒或 10 秒固定验证预算，试跑后冻结。每方法每实例每 seed 验证相同 K，报告均值、标准差、最佳值、总时间、gap 和可行比例。

### 阶段 C：独立测试与小消融

- 70/30 或 5-fold assignment split，测试 RF 的独立 ranking；
- K∈{1,3,5,8}，从同一已排序候选记录导出；
- 只做一个 feature ablation：machine-load-only vs load+job-time；
- 选 mk04 和 mk06 做失败诊断，使用同一 candidate ID 关联特征和结果。

不建议在当前投稿周期再加入 DRL、跨基准迁移、RF-in-MIP、新的复杂多样性算法或管理现场实验。

## 可能的审稿意见与预先回答

| 可能意见 | 当前风险 | 修订后的回答 |
|---|---|---|
| 方法只是 RF 加启发式规则，创新不足 | 高 | 将贡献定位为有限验证预算下的可复现实证框架，并以统一 scorer 对比和失败边界作为贡献 |
| RF 是否真的优于简单负载规则 | 无法回答 | 同一候选池、同一 K、同一墙钟时间，完整报告 MaxLoad/LB/HeurSeq |
| 结果是否过拟合 | 高 | 独立 assignment split、多个 seed、固定调参协议 |
| MIP 结果是否最优 | 高 | 报 status/bound/gap，统一使用 time-limited verification |
| 为什么只用 Brandimarte | 中 | 说明这是第一阶段公开基准研究；若篇幅允许加少量 Hurink/Kacem，只报告预先冻结的协议 |
| 所谓可解释性在哪里 | 中 | 改为 feature-transparent 或补 permutation importance 与案例 |
| 与 SOTA 相比如何 | 高 | 不宣称 SOTA；加入一个可信外部参照并说明研究目标是 screening |

## 最终建议

当前不应再扩展模型，而应先修复证据链。投稿顺序建议是：

1. 完成统一候选池和多 seed 实验；
2. 根据结果决定 RF 是否保留为主方法。如果 RF 不优于 MaxLoad，也可以把论文写成“学习型筛选与简单规则的比较及适用边界”；
3. 按保守标题重写摘要、贡献和结论；
4. 清理作者、单位、仓库链接、FJSPLib 来源和许可证；
5. 再选择审稿周期明确的 EI 会议或期刊。

如果只修文字、不补公平实验和独立测试，我不建议投稿；最可能收到的意见会集中在“比较不公平、结果不可复现、RF 泛化未验证、贡献不足”。
