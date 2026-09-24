# FJSP：随机森林辅助的柔性作业车间调度

本仓库包含一个面向柔性作业车间调度问题（Flexible Job Shop Scheduling Problem, FJSP）的实验项目。项目将机器分配（routing）与固定分配后的工序排序（sequencing）分开处理：先用轻量级特征和随机森林（RF）筛选候选机器分配，再用限时 Gurobi MIP 对少量候选进行排序验证。

当前仓库用于整理投稿材料和继续实验。项目定位是可解释的候选分配筛选工具，不声称替代成熟的 FJSP 元启发式算法，也不声称已经达到最优或 state of the art。

## 当前进展

- 已实现 FJSP JSON 数据读取、随机/SPT/LPT/Balanced 分配、启发式排序器、固定分配排序 MIP、完整 FJSP MIP 和 RF 引导束搜索。
- 已整理 Brandimarte MK01--MK10 的历史实验结果、代理标签验证、部分敏感性分析和英文论文初稿。
- 已新增 `fjsp/reproduce_audit.py`，从保存的主实验 JSON 生成统一统计摘要。
- 已修复 RF 扰动采样的随机种子，以及公平基线中“被评分候选”和“被保留候选”不一致的问题。
- OR 环境中的 Gurobi 10.0.1 可用；mk01、mk04、mk06 已通过小规模真实求解和机器分配可行性检查。
- 投稿目标：优先选择正规 EI 会议或 EI 期刊，计划在 2026-11-23 前完成投稿并争取获得首轮回复。这个时间目标不等于录用、见刊或 EI 检索承诺。
- 正式小规模实验已完成：`LB`、`RF-H`、`CRF`、`HeurSeq` 和 `Random` 共享候选池、Top-K 和限时固定分配 MIP 预算；结果同时保存可行解、solver status、best bound、gap 和耗时。
- 10 个 Brandimarte 实例 × 3 seeds 的 1 秒验证协议已完成：CRF 与 RF-H 的 Top-5 选择完全一致，26/30 个组合因低 gap 校准标签不足回退，因此暂不声称 CRF 有稳定收益。MK06 的两 seed、5 秒复核也已完成，结论不变。
- 英文稿件已改为预算匹配实证研究定位，删除未实现特征、RF-in-MIP 概念附录和过强的优越性表述。

### 当前历史结果的口径

`fjsp/results/exp1_main.json` 是单一固定种子（42）的历史探索结果，不是最终多种子实验。按统一 `1e-3` 平局容差复算，RF 相对 Balanced 为 **6 胜 / 2 平 / 2 负**，平均 makespan 改善 **1.8125%**；四舍五入 makespan 后的双侧 Pratt Wilcoxon 检验为 `p=0.1236`。这些数字用于追踪现状，不能写成 RF 已显著优于基线。

## 目录

```text
.
├── README.md
├── SUBMISSION_PLAN.md              # 投稿目标、审计发现和执行计划
├── fjsp/PAPER_REVIEW.md            # 论文与实验的系统性审阅意见
├── fjsp/POSTING_STRATEGY.md        # 改进方法、实验协议和目标刊物/会议
├── paper_analysis.md               # 项目分析记录
└── fjsp/
    ├── data/instances/             # FJSP JSON 基准实例
    ├── paper/manuscript.tex        # 英文论文草稿
    ├── results/                    # 历史结果和 canonical_exp1_summary.json
    ├── fjsp_baseline.py            # FJSP 与固定分配排序 MIP
    ├── assignment_generators.py   # 机器分配生成器
    ├── heuristic_sequencer.py     # 低成本启发式排序器
    ├── experiments_final.py       # RF 训练、束搜索和主实验入口
    ├── final_experiments.py       # 统计、代理验证和基线实验脚本
    ├── reproduce_audit.py          # 主结果审计与统计复算
    ├── calibrated_screening.py     # 训练/测试分离的 CRF 公平筛选协议
    ├── summarize_calibrated.py     # 自动生成汇总 CSV/Markdown
    ├── PAPER_OUTLINE.md
    ├── TODO.md
    └── WORKFLOW.md
```

`refer/` 是工作目录中的另一个配送分单项目，包含大量仿真数据，与本 FJSP 仓库无关，因此没有纳入 Git 提交。

## 环境与运行

建议使用已经配置 Gurobi 的 OR Conda 环境：

```powershell
conda activate OR
cd fjsp
python -c "import gurobipy; print(gurobipy.gurobi.version())"
python reproduce_audit.py
```

直接运行历史主实验入口前，请先阅读 `SUBMISSION_PLAN.md` 并设置统一的求解时间、随机种子和结果输出目录。主实验包含大量 MIP 求解，运行时间可能从数十分钟到数小时不等；需要有效的 Gurobi license。

普通 Python 依赖见 `fjsp/requirements.txt`。`ortools` 不是当前核心路径的必要依赖。

## 投稿前 TODO

1. 用统一协议完成多随机种子实验，并保存每个候选的 assignment、solver status、incumbent、bound、MIP gap 和耗时。
2. 补齐全 10 个 Brandimarte 实例的公平基线比较；当前公平基线文件只有部分记录。
3. 修复并重新生成 candidate-level failure analysis，确保特征和 makespan 属于同一个候选。
4. 从同一份结果自动生成论文表格，核对 BKS 来源、数据版本、参考文献和作者信息。
5. 核验当期 EI 会议/期刊的官方投稿窗口、费用、首轮决定时间、收录承诺和单位认可要求。

CRF 小规模实验可复现命令：

```powershell
cd fjsp
python calibrated_screening.py --instances mk01 mk04 mk06 --seeds 42 43 44 `
  --n-train 300 --n-test 150 --calibration 15 --verify-time 1 --k 5 `
  --output results/calibrated_screening_formal_small.json
python summarize_calibrated.py results/calibrated_screening_formal_small.json
```

详细安排见 [`SUBMISSION_PLAN.md`](SUBMISSION_PLAN.md)，论文草稿见 [`fjsp/paper/manuscript.tex`](fjsp/paper/manuscript.tex)。

## 数据与复现说明

Brandimarte 等实例应按其原始文献和 FJSPLib 版本引用。仓库中的结果是研究过程记录，正式投稿前会重新生成并标注软件版本、硬件、随机种子、求解器状态和最佳已知解来源。除非另有说明，实验中的“验证”指限时固定分配排序 MIP 验证，不指已证明最优。

## License

暂未指定开源许可证。正式公开仓库前，需要确认代码、基准实例和引用数据的许可证及再分发条件。
