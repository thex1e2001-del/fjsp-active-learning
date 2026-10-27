# Method variant screening report

Date: 2026-10-27. Development-only; these variants are not part of the formal 50-run result.

Three changes were tested on MK01/MK04/MK06/MK08, seeds 42/43, small pools (60 train, 30 test), K=3, 1-second verification:

1. **RF-LB residual**: train RF on heuristic makespan minus the workload lower bound. It tied LB on MK01/MK04, was worse on MK06 in one run, and did not improve MK08.
2. **RankBlend**: fixed rank sum of RF-H and LB. It did not consistently improve either scorer.
3. **Anchor-RF / Portfolio**: reserve one verification slot for Balanced, or reserve slots for Balanced/LB/HeurSeq. These variants sometimes improved MK01/MK04/MK06, but also worsened cases and did not solve MK08 infeasibility.

Decision: none is stable enough to freeze as the main method. The clean baseline remains RF-H, LB, HeurSeq, Random and exploratory CRF. The negative variant results support a bounded empirical paper and prevent post-hoc method selection.
