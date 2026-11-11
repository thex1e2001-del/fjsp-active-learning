# Five-seed equal-call audit (2026-11-11)

This supplement replays the paper's candidate-screening protocol on Brandimarte
MK01--MK10 with predetermined seeds 42--46. The original two-seed submission
results remain separate. The main question is whether AL-RF's best feasible
incumbent from **30 calibration plus five final MIP calls** improves on the best
incumbent from **HeurSeq's top 35 test candidates**, with five seconds requested
for each solve. Each method is charged for its own calls even when the physical
replay reused an identical assignment result.

The audit completed all 50 instance-seed cases. On the ten instance-level
five-seed means, AL-RF wins five and loses five. Mean makespans are 208.70 and
208.34, respectively; an exact two-sided signed-rank test gives p=0.943359.
This does **not** establish an equal-total-call advantage for AL-RF. It also
does not establish equal end-to-end wall-clock cost: candidate generation,
model fitting, hardware threads, and early solver termination differ.

## Files and checks

- `raw_cases.zip` contains all 50 `mkXX_seedYY` folders. Each has
  `prepared.json` (ordered candidate pools, features, labels), `solves.jsonl`
  (every unique solver call, including rejected or missing incumbents), and
  `result.json` (method selections and call accounting). `raw_cases_manifest.json`
  gives the archive and per-file SHA-256 hashes.
- `per_seed_summary.csv`, `instance_summary.csv`, `analysis.json`, and
  `REPORT.md` are directly readable without unpacking the archive.
- `manifest.json`, `recovery_amendment.json`, `reuse_provenance.json`, and
  `completion.json` document source identities, reused MK06 cases, a validation
  rule amendment after a stopped attempt, and final status.
- `../../camera_ready_audit.py` is the hash-matched runner used for the batch;
  `../../summarize_camera_ready_full.py` rebuilds and checks the summaries.
  The protocol is in `../../protocols/camera_ready_full_five_seed_20261111.json`.
- `../../data/verified/brandimarte/` has the independently checked benchmark
  files. Only MK06's machine count differs from the original input (10 vs 15).

Of 3484 physical unique solves, 3468 schedules passed an independent feasibility
check, four calls returned no incumbent, and 12 solver incumbents were rejected
by that check. Rejected calls still count toward each method's budget. The last
case was resumed from its saved calls after the process ended unexpectedly;
there is no evidence of a data or solver fault. See `recovery_amendment.json`.

## Recompute the analysis

From `fjsp/` in a checkout of the source commit and with NumPy and SciPy
installed, unpack the ZIP into this directory and run:

```powershell
python summarize_camera_ready_full.py --require-complete
```

This verifies all 50 case files, call counts, candidate provenance, accepted
schedules, cost sums, and instance-level tests. It rewrites the summary files;
run it in a disposable checkout if you want to preserve the committed bytes.

The archived runner records the original machine-local paths and an archived
source ZIP used for source authentication. A fresh solver rerun requires mapping
those paths to a local checkout and a working Gurobi 10 license. The CSV/JSON
analysis can be reproduced directly from the published raw cases without Gurobi.

The five seeds overlap development work. This is a supplementary evaluation,
not an independent held-out confirmation or a new method claim.
