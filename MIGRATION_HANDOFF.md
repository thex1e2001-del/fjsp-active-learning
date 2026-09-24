# 项目迁移与接手续作说明

更新：2026-09-24。当前研究路线基线提交：`e10d236`；本交接包另作后续提交。接手时以 `git log -1` 和远端 `main` 为准，不把文档中基线hash误认为最新提交。

## 1. 先读这三份

1. [EXPERIMENT_ROADMAP.md](EXPERIMENT_ROADMAP.md)：详细补实验计划、参数、优先级、验收标准和不应声称的结论。
2. [fjsp/TODO.md](fjsp/TODO.md)：当前执行清单。
3. [README.md](README.md)：结构、现有结果和运行入口。

本次追加：[METHOD_DIRECTION.md](METHOD_DIRECTION.md) 的 Gate A/B/C 规定先检验价值再长跑；[PAPER_VISUALS_PLAN.md](PAPER_VISUALS_PLAN.md) 是新增图表TODO。接手时优先按这两个补充文件理解路线更新，避免只补seed/图表而没有解决方法价值问题。

旧 `SUBMISSION_PLAN.md`、`fjsp/POSTING_STRATEGY.md`、`fjsp/PAPER_OUTLINE.md`、`fjsp/WORKFLOW.md` 和旧TODO包含已过时的数字/设想。它们用于理解历史，不优先于新路线。**不能恢复旧文件中的3.32%改善、显著性、四组特征、10次重复等缺证据表述。**

## 2. 当前准确状态

- 仓库：[Gitee fjsp](https://gitee.com/xie-xianyi/fjsp)，分支 `main`。
- 10个Brandimarte实例×5seeds（42–46）的一秒实验完成，共50组；原始两批、合并文件、汇总和审计均已版本化。
- CRF回退43/50，Top-5集合与RF-H相同48/50；46个双方可行组合最佳makespan都在1e-3内持平。不能声称CRF改善RF。
- MK06只有两个seed的5秒探索，300/150与200/100池设置不同，不能当成控制变量时间敏感性。
- 已重建全部50组候选池：训练/测试交集为0，选中ID及已有校准ID归属正确，见 `fjsp/results/calibrated_screening_five_seeds.pool_audit.json`。程序尚未强制跨池去重。
- 论文可继续写，但未满足立即投稿条件；缺完整等总成本对照、可靠的预算敏感性、失败机制证据和最终PDF检查。五seeds完成不等于实验全部完成。
- 交接准备时没有运行中的Python实验；迁移后不用等待旧进程，也没有自动安排继续运行的任务。

## 3. 应保留的目录和关键文件

| 内容 | 路径 | 注意 |
|---|---|---|
| 当前实验代码 | `fjsp/calibrated_screening.py`、`fjsp/fjsp_baseline.py`、生成器/排序器/RF模块 | 现有runner不是v2计划实现 |
| 原始当前结果 | `fjsp/results/calibrated_screening_formal_small.json`、`calibrated_screening_seeds45_46.json` | 不覆盖，合并来源 |
| 合并/审计/汇总 | `fjsp/results/calibrated_screening_five_seeds*` | 含50组和来源审计 |
| 审计工具 | `fjsp/merge_screening_results.py`、`summarize_calibrated.py`、`audit_screening_pools.py` | 不调用MIP优化；部分import仍要求gurobipy安装 |
| 独立5秒探索 | `fjsp/results/calibrated_screening_verify5_mk06*` | 2组，不能拼主表 |
| 实例 | `fjsp/data/instances/` | 基准来源、BKS、许可证仍待核验 |
| 论文 | `fjsp/paper/manuscript.tex`、`five_seed_tables.tex` | Overleaf中放同一目录 |
| 环境 | `fjsp/requirements-reproduction.txt`、`environment_snapshot.json` | 迁移时的核心依赖快照，不包含license |
| 后续路线 | `EXPERIMENT_ROADMAP.md`、`fjsp/TODO.md` | 新的当前执行依据 |

Git明确忽略了 `refer/`（另一个配送项目）、Conda环境、IDE配置、日志和编译中间文件。Git推送**不是整个D盘工作目录的备份**。如果还需要 `refer/`、未跟踪文件或旧本地PDF，请另行手动迁移。检查命令：

```powershell
git status --short --untracked-files=all
git ls-files
git status --short --ignored
```

不要把license、token、SSH私钥或本机凭据放进仓库。该要求不妨碍迁移代码/实验；新机独立配置认证和求解许可。

## 4. 新机器获取项目

安装完整Git后，在任意目标父目录：

```powershell
git clone https://gitee.com/xie-xianyi/fjsp.git
cd fjsp
git log -5 --oneline
git status -sb
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

这里clone后的外层目录也叫 `fjsp`，其下仍有内层 `fjsp/` 代码目录。后续命令中的“仓库根目录”指含 `README.md`、`EXPERIMENT_ROADMAP.md`、`.git/` 的外层目录。

不需要复制 `.git` 凭据配置来取得代码；有写权限的用户在新机按Gitee正常流程登录。若选择离线备份，旧机可在仓库外创建：

```powershell
git bundle create ../fjsp-migration.bundle --all
git bundle verify ../fjsp-migration.bundle
```

新机可 `git clone /path/to/fjsp-migration.bundle fjsp`，再将origin URL设为Gitee。bundle只是可选迁移步骤，本次未默认生成。

### 旧机Git问题的实际解决办法

旧系统Git位于 `D:/Application/vpn/Git/cmd/git.exe`，缺失 `git-remote-https`。后来从Git for Windows官方release下载独立MinGit，已成功推送并核对远端hash；不是仓库权限丢失，也不需要创建新仓库。

旧机可用工具路径：

```powershell
& "$env:LOCALAPPDATA/fjsp-tools/MinGit-2.55.0.5-64-bit/cmd/git.exe" push origin main
```

不要在新机照抄旧机绝对路径。新机优先安装完整Git，正确配置HTTPS凭据或经过主机密钥核验的SSH。仓库URL保持原样，不将token写入URL。

## 5. Python与求解器环境

旧机解释器：`D:/Anaconda/envs/OR/python.exe`，Python **3.10.16**。当前核心版本：numpy 2.0.2、scipy 1.15.1、scikit-learn 1.6.1、gurobipy 10.0.1、joblib 1.4.2、threadpoolctl 3.5.0。

推荐新建独立环境，不覆盖现有环境：

```powershell
conda create -n fjsp-repro python=3.10.16 pip
conda activate fjsp-repro
python -m pip install -r fjsp/requirements-reproduction.txt
python -c "import numpy, scipy, sklearn, gurobipy; print(numpy.__version__, scipy.__version__, sklearn.__version__, gurobipy.gurobi.version())"
```

`requirements-reproduction.txt`固定核心依赖，不是整个Conda环境导出。在目标OS/CPU或旧Gurobi wheel不可用时，应记录替代版本并做新协议试跑，不静默更换后宣称逐数值复现。

Gurobi许可需在新机独立安装/激活；装了gurobipy不代表有可用license。受限license可能只能解小模型，未通过大实例测试前不开始长跑。先在仓库根目录做小模型许可检查（此项确实启动一次优化）：

```powershell
python -c "import gurobipy as gp; m=gp.Model(); m.Params.OutputFlag=0; x=m.addVar(lb=1); m.setObjective(x); m.optimize(); print('status=',m.Status,'solutions=',m.SolCount)"
```

这个小测试不能证明完整FJSP规模的license也可用。`ortools`当前未被主实验使用，不需要为读取/汇总结果安装；未来E5使用时另行锁版本。

当前硬件快照见 `fjsp/environment_snapshot.json`，是**迁移时机器**，不是旧实验硬件证明。稿件中的Ultra 5 125H和当前i5-14600KF不一致，需要核验，尤其不能跨机器比较1秒/5秒性能。

## 6. 迁移后先验收，不必重跑50组MIP

从仓库根目录运行：

```powershell
python fjsp/merge_screening_results.py
python fjsp/summarize_calibrated.py fjsp/results/calibrated_screening_five_seeds.json
python fjsp/audit_screening_pools.py
```

预期审计输出：runs=50、crf_fallback=43、identical sets=48、both feasible=46、tied best=46；池审计为50组、overlapping runs=0、total overlap=0。任一不符先检查数据、版本与候选生成器，不继续生成正式表。

汇总器只用标准库，可以单独运行而不装优化环境。池审计/合并入口因imports需要核心依赖，但不会求解MIP。文件字节hash会受跨平台换行影响；提交身份、字段内容、assignment ID和语义计数一起检查，不能仅靠字节差异判断实验改变。

可选的新机求解烟雾检查（输出新文件，不覆盖历史）：

```powershell
cd fjsp
python calibrated_screening.py --instances mk01 --seeds 42 --n-train 30 --n-test 20 --calibration 3 --verify-time 1 --k 2 --output results/migration_smoke_new_machine.json
cd ..
```

它只检查现有流程能运行，不是新正式实验；求解时限受硬件/负载影响，不要求目标值逐位相同。原runner没有resume，同输出路径会覆盖前次记录；长跑必须先落实E0。

## 7. Overleaf交接

- 上传/同步 `fjsp/paper/` 下两个TeX文件，把 `manuscript.tex` 设为主文件；`five_seed_tables.tex`由脚本生成，不能手工改数值。
- 当前使用article文档类，英文正文，标准LaTeX包，参考文献暂内嵌 `thebibliography`，没有必须迁移的单独bib文件。
- 当前仅做过TeX环境配对等静态检查，**未完成最终PDF编译和视觉检查**；新机/Overleaf需检查错误、引用、表格宽度、页数及占位作者。
- 编译器可先用pdfLaTeX，依目标模板再调整。保留原稿后替换目标刊会模板，不让模板改动影响结果数字。
- 当前无需Word。目标明确接受LaTeX/PDF就继续；若强制DOCX再转换并检查公式/引用/表格。
- Overleaf云端可能有本地仓库没有的修改，迁移前请导出云端zip/PDF作备份并比较差异。GitHub/Gitee同步不会自动包含未同步的Overleaf修改。

## 8. 第一轮接手应该做什么

1. 验证clone、依赖、50组汇总和零交集审计，保存新机环境元数据。
2. 阅读E0并实现日志/断点/排程校验/成本记录/强制去重；修正完整MIP无解时访问 `.X` 的边界问题。
3. 新机统一参数后先E1A同池预算复核与E2误差诊断，不直接重复所有旧实验。
4. 按METHOD_DIRECTION先Gate A测成本与瓶颈；如有依据再Gate B一次轻量修改；Gate C冻结后用新确认种子（默认142–146）做E1B/E3。42–46已看过，用作开发。无优势时不默认认为改成实证短文就足够。
5. 并行官方核验EI目标渠道、费用和时间窗口；还未正式选刊，也没有“保证录用”的依据。

## 9. 给下一位协作者的可复制任务

> 先读MIGRATION_HANDOFF.md、EXPERIMENT_ROADMAP.md和fjsp/TODO.md。当前已有MK01–MK10×5seeds的一秒探索结果，CRF没有最佳makespan收益，不能声称论文已能投稿。50组训练/测试分配重建审计无交集，但程序还未强制去重。保留历史JSON，优先实现E0的日志、协议指纹、断点恢复、逐方法计时及独立排程检查，然后按E1A固定池/固定选择在同一新机器做1/5/10秒复核。不要以更长时限但改变候选规模的运行冒充控制变量实验。主稿为Overleaf TeX，最终PDF未验收，暂无Word需要。Git远端为gitee.com/xie-xianyi/fjsp.git，正常提交推送，不携带任何license/token。
