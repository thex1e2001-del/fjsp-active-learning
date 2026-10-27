# Active-learning screening development summary

Protocol: 200 training assignments, 100 independent test assignments, 30 heuristic-ranked calibration candidates, 5-second fixed-assignment MIP labels/verification, K=5; two seeds per instance.

| Instance | AL-RF | RF-H | LB | HeurSeq | Random |
|---|---:|---:|---:|---:|---:|
| mk01 | 45.0 | 45.0 | 45.0 | 45.0 | 48.0 |
| mk02 | 39.5 | 44.5 | 40.5 | 39.5 | 40.5 |
| mk03 | 240.0 | 242.0 | 241.5 | 243.0 | 247.5 |
| mk04 | 71.0 | 71.0 | 71.0 | 71.0 | 71.0 |
| mk05 | 185.0 | 185.0 | 184.5 | 186.0 | 192.5 |
| mk06 | 110.0 | 113.0 | 111.0 | 114.0 | 116.5 |
| mk07 | 206.5 | 208.5 | 204.0 | 208.5 | 209.0 |
| mk08 | 532.5 | 540.0 | 537.0 | 531.5 | 558.5 |
| mk09 | 377.0 | 380.5 | 382.0 | 376.5 | 380.5 |
| mk10 | 310.5 | 312.0 | 329.0 | 313.0 | 314.5 |

ActiveRF: mean=211.70, wins at instance-seed level=10

RF-H: mean=214.15, wins at instance-seed level=4

LB: mean=214.55, wins at instance-seed level=10

HeurSeq: mean=212.80, wins at instance-seed level=8

Random: mean=217.85, wins at instance-seed level=2
