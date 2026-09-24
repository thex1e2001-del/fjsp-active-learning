# 改进方法与投稿目标计划

制定日期：2026-09-24。目标：在 2026-11-23 前完成可提交稿件，并争取获得首轮正式回复。录用、见刊、EI 检索和首轮回复分别核验，不作时间或录用保证。

## 一、投稿定位

不再把当前工作包装成新的 FJSP 求解理论或 SOTA 算法。主线改为：

> **在有限 fixed-assignment MIP 验证预算下，lower-bound 和轻量代理模型能否有效筛选机器分配？少量精确标签校准是否能修正启发式标签偏差？**

这个问题有三个好处：它保留现有代码和数据，能自然解释 RF 不稳定的结果，也允许诚实报告“简单 MaxLoad 规则有时更好”。论文贡献由算法优越性转为预算控制、代理校准和适用边界的实证证据。

建议题目：

> **Calibrated Surrogate Screening of Machine Assignments for Flexible Job Shop Scheduling under Limited Verification Budgets**

根据当前 1 秒、10 实例 × 3 seed 结果，校准实验暂未显示稳定收益，因此当前稿件应使用保守题目：

> **Random-Forest-Assisted Candidate Screening for Flexible Job Shop Scheduling: A Budget-Matched Empirical Study**

## 二、拟改进方法：Calibrated RF Screening

### 1. 候选分配与特征

每个实例和 seed 独立生成训练池与测试池，候选是合法的机器分配。第一版只使用代码已经实现、容易复核的特征：

- 每台机器的负载、工序数、使用标记；
- 最大机器负载、负载不均衡、负载标准差；
- 最大作业路径时间；
- lower bound `LB=max(max_machine_load,max_job_time)`。

论文删掉当前没有实现的 speed loss、Gini、stage load 和 precedence feature。若后续实现扩展特征，必须单独做 ablation，不能和第一版结果混用。

### 2. 三层评分器

对同一候选池运行以下方法：

1. **LB**：直接以 `LB` 排序；
2. **RF-H**：用廉价 heuristic sequencer 的 makespan 标签训练 RF；
3. **CRF**：先训练 RF-H，再用少量 fixed-assignment MIP 标签学习残差。

对候选 $x$：

\[
H(x)=\text{heuristic makespan},\qquad
r(x)=C^{MIP}_{max}(x)-H(x),
\]
\[
\widehat C^{CRF}(x)=\widehat C^{RF-H}(x)+\widehat r(x).
\]

残差模型只使用训练池中通过预设 gap 门槛的 MIP 标签。若有效标签不足，自动回退到 RF-H，并在结果中记录回退原因。这样可以避免把不可靠的 timeout incumbent 当成精确真值。

### 3. 预算与选择

- 训练池：600 个 assignment；测试池：400 个 assignment；两者完全去重；
- exact calibration：训练池中固定抽取 30 个候选，单候选 5 秒，记录 status/bound/gap；
- 选择阶段：每个 scorer 在测试池上选 Top-5；所有方法使用相同的 5 次 MIP 验证；
- end-to-end 版本再报告总墙钟时间，包含 heuristic labeling、RF 训练、残差训练、候选评分和 MIP 验证；
- 每实例使用 5 个预先固定 seed，调参不查看最终测试池。

### 4. 必须的消融

只保留三项，控制论文篇幅：

- LB vs RF-H vs CRF；
- calibration labels 为 0、15、30 的敏感性；
- Top-K 为 1、3、5、8 的成本—收益曲线。

指标报告均值、标准差、最好值、相对 Balanced 的改善、总时间、可行解比例和 MIP gap。只有在额外评估完整测试池、得到可靠池内 oracle 时才报告 Regret@K；当前小规模协议使用“selected Top-K 的最佳可行 incumbent”作为主要质量指标，避免把未验证候选误当作真值。

### 5. Go / No-Go 规则

- 如果 CRF 在至少 6/10 实例上降低 Top-K 最佳可行 makespan，且平均总时间没有明显恶化：以“少量精确标签校准代理筛选”为方法贡献；
- 如果 CRF 只改善少数实例：以“识别校准有效的结构条件”为贡献；
- 如果 CRF 没有收益：删除校准方法的主张，改投 budget-matched empirical study，不反复调参追求有利结果。

## 三、实验实施阶段

### 第 1 周：方法和记录重构

- 集中定义真实 feature list；
- 新增 seed、candidate hash、train/test split 和统一 scorer 接口；
- `FixedAssignmentSequencingModel.solve()` 保存 status、objective、bound、gap、wall time；
- 新增 assignment 可行性检查和自动结果表生成；
- 只在 MK01、MK04、MK06 试跑，测量 5 秒和 10 秒预算的实际耗时。

### 第 2 周：试验与冻结协议

- 完成 LB/RF-H/CRF 的小规模试跑；
- 观察 calibration 标签是否足够、MIP gap 是否可接受；
- 冻结 5 秒或 10 秒预算、30 个 calibration labels、Top-K 和 RF 超参数；
- 任何最终测试结果产生后不再逐实例调参。

### 第 3–4 周：主实验

- MK01–MK10 × 5 seeds；
- 同一候选池、同一 K、同一验证时间；
- 完成 scorer 对比、calibration 敏感性、Top-K 曲线；
- 对 mk04、mk06 做候选级失败分析。

### 第 5 周：稳健性与第二基准

- 若主实验稳定，增加少量 Hurink 或 Kacem 实例，不追求覆盖所有数据集；
- 增加一个可信外部参照：公开 tabu/GA 实现或 CP-SAT，明确它是质量参照，不混入 scorer 公平池；
- 检查不同求解器线程、license 和硬件说明。

### 第 6 周：论文重写

- 6–8 页 EI 会议版或 8–12 页应用期刊版；
- 删除 RF-in-MIP 概念附录、未经实验支持的管理价值和未实现特征；
- 只保留真实贡献、失败边界和完整实验协议；
- 生成投稿版 PDF、补充材料、代码说明和 response-to-reviewer 模板。

### 第 7–8 周：投稿和回复准备

- 先投主目标期刊；
- 若官方会议仍有明确 CFP、全文截止日期和 EI 收录说明，则并行准备会议版；
- 按目标期刊模板提交；
- 维护一份逐条审稿回复表，首轮意见到达后优先修复实验可复现性和表格问题。

## 四、目标期刊与会议

以下是**候选池**，不是未经核验的“容易录用”名单。投稿前必须打开官方页面，确认当期 EI Compendex/Engineering Village 或单位认可目录、首轮决定时间、APC、稿件类型和撤稿/特刊信息。

### 主投期刊

| 优先级 | 候选 | 匹配理由 | 主要风险 | 计划动作 |
|---|---|---|---|---|
| P1 | [Algorithms](https://www.mdpi.com/journal/algorithms) | surrogate、组合优化、启发式搜索和实验研究匹配 | 需要核验当前 EI 收录和 APC；方法创新不足时仍会被质疑 | 首先核验收录、近期 FJSP 文章和首次决定时间 |
| P2 | [Machines](https://www.mdpi.com/journal/machines) | 制造系统和车间调度背景更自然 | 需要补制造应用解释，不能只写算法表格 | 若实验稳定且制造叙事清楚，作为主投或备投 |
| P3 | [Applied Sciences](https://www.mdpi.com/journal/applsci) | 应用型优化和智能制造范围宽 | 学科范围宽，EI 状态和栏目匹配需单独确认 | 作为预算和篇幅允许时的备用渠道 |

### 条件性会议路线

优先检索 IEEE IEEM、IEEE ICIEA、IEOM 及智能制造/工业工程方向的 2026 或 2027 官方 CFP。只有同时满足以下条件才列入主投：

1. 官方 CFP 明确全文投稿截止日期在计划窗口内；
2. 官方或出版社页面明确说明 proceedings 的 EI Compendex/Engineering Village 送检安排；
3. 论文类型、页数和主题匹配；
4. 单位认可该会议，而不是只认可“主办方声称将检索”；
5. 能在投稿前完成完整实验和版权/注册费用确认。

如果当前年度的 IEEE IEEM/ICIEA/CASE 截止日期已经过去，不再为了赶会议压缩实验质量，转为期刊路线。会议录用快不等于 EI 检索快，二者要分开记录。

### 不适合本轮目标的渠道

暂不把 C&IE、EJOR、IJPR、Journal of Manufacturing Systems 等作为两个月首轮回复目标。它们与主题匹配，但审稿周期和创新要求更高，适合作为后续强化版或第二篇工作。

## 五、最终选刊表

在正式投稿前建立一张可审计表，至少包含：官方投稿链接、核验日期、EI/Scopus 状态、最近相关论文、首次决定时间定义、APC/注册费、页数、是否开放获取、单位认可情况、主投理由和备投理由。任何“快速录用”“保证 EI”的第三方宣传都不能作为依据。

## 六、当前决策

截至 2026-09-24 的正式小规模结果（MK01–MK10、3 seeds、Top-5、1 秒验证）中，CRF 与 RF-H 的最佳可行 makespan 相同（不能据此推出候选集合相同）；CRF 在 30 个组合中有 26 个因低 gap 标签不足而回退。MK06 的两 seed、5 秒复核仍得到相同结论：CRF 与 RF-H 持平，LB 和 HeurSeq 在均值上更低。该结果支持“预算匹配实证研究”的定位，不支持声称 CRF 已经稳定改善 RF-H。

## 七、决策门槛

第 2 周结束时做一次硬决策：

- 若 CRF 或 RF 在统一预算下有稳定 Regret@5 优势：投算法/优化方向期刊；
- 若方法和 MaxLoad 接近但失败边界有价值：投制造应用/工业工程方向期刊，主打比较研究；
- 若 RF、CRF 和简单规则都没有稳定差别：停止堆方法，改写成“代理筛选在何种结构下可靠”的短文；
- 若 MIP 许可或运行时间无法支持统一实验：暂缓投稿，不用不完整的历史表替代正式证据。

## 五 seed 扩展状态

已完成 seeds 45/46，累计 50 组；CRF 回退 43/50，候选集合相同 48/50，双方均可行的 46 组最佳 makespan 全部持平（容差 1e-3）。生成表见 `results/calibrated_screening_five_seeds.summary.md`。仍需严格训练/测试去重检查、等总成本核算、同池更长预算对照和最终 PDF 校验，不能把增加 seeds 等同于达到投稿条件。
