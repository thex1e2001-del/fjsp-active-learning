# Gate A development report

Date: 2026-10-27
Protocol: development-only, fixed small pools (`n_train=60`, `n_test=30`, `K=3`), seeds 42 and 43, instances MK01/MK04/MK06/MK08. This report must not be merged with the formal 50-run result.

## Result

| instance | seed | budget | LB | RF-H | HeurSeq | Random | RF-H calibration labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| MK01 | 42 | 1 s | 48 | 54 | 48 | 55 | 2 |
| MK01 | 43 | 1 s | 42 | 42 | 42 | 42 | 2 |
| MK04 | 42 | 1 s | 67 | 67 | 67 | 67 | 0 |
| MK04 | 43 | 1 s | 75 | 81 | 81 | 75 | 0 |
| MK06 | 42 | 1 s | 115 | 115 | 115 | 110 | 0 |
| MK06 | 43 | 1 s | 119 | 123 | 123 | 123 | 0 |
| MK08 | 42 | 1 s | 581 | 571 | 568 | 699 | 0 |
| MK08 | 43 | 1 s | 615 | 604 | no feasible | 669 | 0 |
| MK01 | 42 | 5 s | 48 | 54 | 48 | 55 | 2 |
| MK01 | 43 | 5 s | 42 | 42 | 42 | 42 | 2 |
| MK04 | 42 | 5 s | 67 | 67 | 67 | 67 | 0 |
| MK04 | 43 | 5 s | 75 | 81 | 81 | 75 | 0 |
| MK06 | 42 | 5 s | 110 | 110 | 110 | 106 | 0 |
| MK06 | 43 | 5 s | 110 | 112 | 115 | 118 | 0 |
| MK08 | 42 | 5 s | 535 | 536 | 547 | 556 | 0 |
| MK08 | 43 | 5 s | 562 | 562 | 579 | 579 | 0 |

## Decision

This development run does not support a claim that RF-H is a generally better method. RF-H is better than HeurSeq in some MK08 cases, tied in several cases, and worse than LB/HeurSeq in MK01/MK04/MK06 cases. CRF produced no additional accepted calibration labels on MK04/MK06/MK08. The 5-second budget improves feasibility on MK08, so solver time is a material factor.

The fast submission path is therefore an empirical budget-and-boundary paper. The main claim should be limited to whether learning-based screening pays for its training cost under a frozen budget. CRF should remain exploratory unless a larger, pre-registered run supplies reliable bounded labels.

## Files

- `gateA_dev_4x2.json`: 1-second development run.
- `gateA_dev_4x2_5s.json`: 5-second development run.
- `gateA_mk01_seed42.json`, `gateA_mk01_seed43.json`: initial smoke runs.

The results were generated after pinning the active runtime to NumPy 1.26.4, SciPy 1.17.1 and scikit-learn 1.9.1. The existing migration snapshot describes a different historical environment and must not be overwritten.

## New-seed confirmation pilot

A separate pilot used seeds 142 and 143, instances MK01/MK04/MK06/MK08, `n_train=100`, `n_test=50`, `K=5`, and a 5-second verification limit. It is a confirmation pilot rather than the full 10-instance confirmation because the pool size differs from the historical formal run.

| instance | seed | LB | RF-H | HeurSeq | Random | accepted CRF labels |
|---|---:|---:|---:|---:|---:|---:|
| MK01 | 142 | 45 | 45 | 45 | 51 | 3 |
| MK01 | 143 | 42 | 42 | 42 | 42 | 4 |
| MK04 | 142 | 68 | 68 | 68 | 95 | 1 |
| MK04 | 143 | 67 | 67 | 67 | 73 | 1 |
| MK06 | 142 | 111 | 111 | 111 | 126 | 0 |
| MK06 | 143 | 100 | 101 | 100 | 108 | 0 |
| MK08 | 142 | 549 | 563 (4/5 feasible) | 585 | 578 | 0 |
| MK08 | 143 | 540 | 537 | 537 | 537 | 0 |

The pilot reproduces the development conclusion on unseen seeds: RF-H is sometimes competitive but has no stable advantage over LB or HeurSeq, and CRF rarely obtains enough bounded labels to operate as a distinct method. This supports freezing the paper around budget sensitivity and method boundaries rather than adding another RF/CRF tuning round.
