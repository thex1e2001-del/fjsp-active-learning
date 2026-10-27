# Active-Learning Candidate Screening for Flexible Job Shop Scheduling

This repository contains the code, benchmark data, experiment outputs, and paper source for a budget-matched study of candidate screening in the flexible job shop scheduling problem (FJSP).

The workflow separates machine assignment from fixed-assignment sequencing. It uses assignment features and a random-forest surrogate to rank candidates, then verifies the same number of candidates for every method with a time-limited mixed-integer programming model.

## Repository layout

- `fjsp/` — Python implementation, benchmark instances, results, and paper files.
- `fjsp/data/instances/` — JSON benchmark instances, including Brandimarte, Fattahi, and Hurink sets.
- `fjsp/results/` — Saved candidate-level outputs and summary tables.
- `fjsp/paper/manuscript_icima.tex` — Main ICIMA 2026 manuscript.
- `fjsp/paper/active_tables.tex` — Results table included by the manuscript.
- `fjsp/paper/active_method_comparison.png` — Normalized method comparison figure.

## Main experiment

The manuscript reports Brandimarte MK01–MK10 with two deterministic seeds per instance. The compared screeners are:

- `LB` — a workload and job-time lower-bound score;
- `RF-H` — a random forest trained on heuristic labels;
- `AL-RF` — a random forest calibrated with time-limited MIP labels;
- `HeurSeq` — direct heuristic ranking;
- `Random` — seeded random ranking.

All methods use the same independent test pool, select five candidates, and receive the same five-second fixed-assignment MIP verification budget. Calibration calls are reported separately.

## Reproduction

Create an environment with Python 3.10 or later and install the dependencies:

```powershell
cd fjsp
python -m pip install -r requirements-reproduction.txt
```

Gurobi is required for the MIP verification experiments and must be installed and licensed separately. To regenerate the active-learning development results:

```powershell
cd fjsp
python active_screening_dev.py --help
```

The saved JSON files in `fjsp/results/` are the exact inputs used to generate the manuscript table. The summary utilities can be used to inspect or regenerate CSV and Markdown summaries without rerunning the solver.

## Paper compilation

Upload these files to Overleaf and set `manuscript_icima.tex` as the main file:

```text
fjsp/paper/manuscript_icima.tex
fjsp/paper/active_tables.tex
fjsp/paper/active_method_comparison.png
```

The paper uses the IEEEtran conference class and compiles with pdfLaTeX.

## Data and code availability

The source code is released for research and educational use. The benchmark instances originate from public FJSP benchmark collections; users should follow the terms and citation requirements of the corresponding source collections. The reported JSON results and generation scripts are included for auditability.

## Citation

If you use this repository, please cite the accompanying paper:

```text
Xianyi Xie. Active-Learning-Assisted Candidate Screening for Flexible Job Shop Scheduling. ICIMA 2026.
```

## Contact

Xianyi Xie — 2431062085@st.btbu.edu.cn
Beijing Technology and Business University
