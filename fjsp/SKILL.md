---
name: rfsj-rf-mip
description: Use when conducting research on FJSP (Flexible Job Shop Scheduling) using RF-embedded MIP — covering problem formulation, RF model design, MIP constraint encoding (tree rules, big-M, sequence/assignment), diagnostic experiments, multi-objective strategies (raw weight vs. utopia point normalization), and UTD-targeting paper writing pipeline. Also use when migrating the RF+MIP framework from the rider stress scheduling domain to public FJSP benchmarks.
---

# RFSJ: RF-Embedded MIP for Flexible Job Shop Scheduling

## Overview

This skill covers the full **RF→MIP→Diagnose→Paper** pipeline for using Random Forest surrogate models embedded in Mixed-Integer Programming to solve Flexible Job Shop Scheduling Problems (FJSP). The methodology originates from rider operational stress scheduling (`refer/` project) and is being generalized to public FJSP benchmarks (Brandimarte, Hurink, Kacem, etc.).

**Core framework:** Train RFs to predict schedule-level quality → encode tree rules as MIP constraints → solve RF-guided MIP → diagnose objective misalignment → iterate.

**Key insight from diagnostics:** The RF surrogate objective (min ΣQ_k) does NOT align with the true scheduling objective (min makespan). The current method is best positioned as an **assignment generator / warm-start** for exact MIP, not as an end-to-end surrogate optimizer.

---

## When to Use This Skill

- Designing an RF+MIP pipeline for FJSP or similar combinatorial scheduling problems
- Encoding decision tree rules as MIP constraints (big-M, leaf activation, dynamic features)
- Debugging surrogate objective misalignment with diagnostic experiments
- Choosing between raw-weight and utopia-point multi-objective strategies
- Migrating the RF+MIP framework to a new dataset or problem domain
- Writing a paper targeting MS, OR, EJOR, C&IE, or IJPR on learning-augmented optimization

---

## 1. Problem Formulation: FJSP

### 1.1 Standard FJSP Definition

| Element | Notation | Description |
|---|---|---|
| Jobs | $J = \{1, ..., n\}$ | Each job $j$ has a sequence of $o_j$ operations |
| Machines | $M = \{1, ..., m\}$ | Each machine can process certain operations |
| Operation | $(j, p)$ | $p$-th operation of job $j$ |
| Processing time | $p_{j,p,k}$ | Time to process $(j,p)$ on machine $k$ |
| Feasible machines | $M_{j,p} \subseteq M$ | Machines capable of processing $(j,p)$ |

### 1.2 Decision Variables

**Assignment (Routing):**
$$x_{j,p,k} \in \{0,1\}$$

> Job $j$ operation $p$ is assigned to machine $k$

**Sequencing:**
$$\delta_{a,b,k} \in \{0,1\}$$

> Operation $a$ precedes operation $b$ on machine $k$

**Timing:**
$$S_{j,p} \geq 0, \quad C_{j,p} \geq 0, \quad C_{\max} \geq 0$$

### 1.3 Standard FJSP MIP Constraints

**Assignment uniqueness:**
$$\sum_{k \in M_{j,p}} x_{j,p,k} = 1 \quad \forall j,p$$

**Completion time:**
$$C_{j,p} = S_{j,p} + \sum_{k} p_{j,p,k} \cdot x_{j,p,k} \quad \forall j,p$$

**Job precedence:**
$$S_{j,p+1} \geq C_{j,p} \quad \forall j, p < o_j$$

**Machine non-overlap (big-M):**
$$S_{b} \geq C_{a} - M(1 - \delta_{a,b,k}) - M(2 - x_{a,k} - x_{b,k})$$
$$S_{a} \geq C_{b} - M \cdot \delta_{a,b,k} - M(2 - x_{a,k} - x_{b,k})$$

**Makespan:**
$$C_{\max} \geq C_{j,p} \quad \forall j,p$$

**Objective:**
$$\min C_{\max}$$

---

## 2. RF-Embedded MIP: Full Architecture

### 2.1 RF Model Specification

| Item | Rider Stress Domain (`refer/`) | FJSP Domain (current) |
|---|---|---|
| **Number of RFs** | 6 (one per courier type) | 6 (one per machine, `rf_machine_0..5.joblib`) |
| **Output** | 2D `[delay_interval, workload_mean_std]` (or scalar `delay_interval`) | scalar `total_duration` / makespan |
| **Training label** | Per-order delay + stress | Per-schedule makespan |
| **Training data** | ~780K delivery records | 5000 heuristic-generated FJSP schedules |
| **Features** | 15 features (static + 2 dynamic) | 19 features (all aggregate, no sequencing) |

### 2.2 FJSP RF Features (Current — Assignment-Only)

```python
features = [
    'num_used_machines',     # Number of machines used in schedule
    # Per-machine (×6):
    'load_0', 'used_0', 'ops_0',   # Machine 0: total load, used flag, operation count
    'load_1', 'used_1', 'ops_1',
    'load_2', 'used_2', 'ops_2',
    'load_3', 'used_3', 'ops_3',
    'load_4', 'used_4', 'ops_4',
    'load_5', 'used_5', 'ops_5',
]
```

**Critical weakness:** No sequencing-sensitive features (critical path length, job tail time, machine idle/wait, precedence blocking). The surrogate can only learn coarse assignment-level patterns.

### 2.3 Tree Rule Encoding → MIP

Each RF tree is traversed to extract all root-to-leaf paths. Each path is a conjunction of `(feature, op, threshold)` conditions:

```python
def traverse(node, path):
    feature_index = tree.feature[node]
    if feature_index == -2:  # leaf node
        rules.append({"path": path, "leaf_values": value[node].flatten()})
    else:
        traverse(tree.children_left[node],  path + [(feature_name, '<=', threshold)])
        traverse(tree.children_right[node], path + [(feature_name, '>',  threshold)])
```

### 2.4 MIP Variables

| Variable | Type | Description |
|---|---|---|
| $x_{j,p,k} \in \{0,1\}$ | Binary | Assignment: operation $(j,p)$ → machine $k$ |
| $\delta_{a,b,k} \in \{0,1\}$ | Binary | Sequencing: $a$ before $b$ on machine $k$ |
| $S_{j,p}, C_{j,p}$ | Continuous | Start / completion time |
| $C_{\max}$ | Continuous | Makespan |
| $z_{m,f}$ | Continuous/Integer | RF feature variables (e.g., `load_k`, `num_used_machines`) |
| $q_{k,t,r} \in \{0,1\}$ | Binary | Leaf activation: machine $k$, tree $t$, leaf $r$ |

### 2.5 Surrogate Objective (Current — Problematic)

$$\min \sum_{k \in M} Q_k$$

where $Q_k = \sum_t \sum_r q_{k,t,r} \cdot \text{leaf\_value}(k, t, r)$

> **Why this fails**: Each $Q_k$ is trained on schedules where machine $k$ is used, predicting the **global** makespan. Summing them double-counts the same schedule-level cost and biases toward using fewer machines.

### 2.6 Key MIP Constraints (Beyond Standard FJSP)

**Leaf uniqueness (one leaf per tree per machine):**
$$\sum_{r} q_{k,t,r} = \text{used}_k \quad \forall k,t$$

**Feature linking (example — `load_k`):**
$$\text{load}_k = \sum_{j,p} p_{j,p,k} \cdot x_{j,p,k} \quad \forall k$$

**Dynamic path constraints (big-M):**
For a leaf path containing `load_k <= threshold`:
$$\text{load}_k \leq \text{threshold} + M \cdot (1 - q_{k,t,r})$$

For `load_k > threshold`:
$$\text{load}_k \geq \text{threshold} + \epsilon - M \cdot (1 - q_{k,t,r})$$

**Static path constraints:**
If feature value is known pre-optimization and violates the path condition:
$$q_{k,t,r} = 0 \quad \text{(pre-fix incompatible leaves)}$$

---

## 3. Diagnostic Pipeline (Critical — Do Not Skip)

### 3.1 The Core Diagnostic Experiment

The diagnostic isolates whether poor makespan comes from **bad assignment** or **bad sequencing**:

```
Original RF+MIP schedule → makespan = 90
        │
        ├── Extract assignment map (machine per operation)
        │
        └── Fix assignment, re-solve exact sequencing
                │
                └── Fixed assignment makespan = X
```

### 3.2 Interpreting Diagnostic Results

| Pattern | RF+MIP Cmax | Fixed-Assign Cmax | Baseline Cmax | Conclusion |
|---|---|---|---|---|
| **A** | 90 | 45 | 40 | Sequencing is the main problem; assignment is decent |
| **B** | 90 | 85 | 40 | Assignment is bad → RF surrogate learned wrong |
| **C** | 90 | 90 | 40 | Model only did routing, sequencing untouched |

**Your actual result:** `90 → 52 → 40` → Pattern A confirmed. Sequencing loss ≈ 38, assignment loss ≈ 12.

### 3.3 Diagnostic Code Template

```python
# 1. Extract assignment from RF+MIP solution
assign = {(j,p): k  for (j,p,k) in zip(jobs, ops, machines) if x[j,p,k].X > 0.5}

# 2. Build fixed-assignment MIP: keep x fixed, re-optimize only δ, S, C
model = Model("fixed_assignment_sequencing")
# ... standard precedence + non-overlap constraints ...
model.setObjective(Cmax, GRB.MINIMIZE)
model.optimize()

# 3. Compare
print(f"Baseline: {baseline_makespan}, RF+MIP: {rfmip_makespan}, Fixed: {fixed_makespan}")
```

---

## 4. Multi-Objective Strategy: Two Variants

Based on the rider stress scheduling project, there are two proven approaches for combining RF surrogate with the true objective.

### 4.1 Strategy A — Raw Weighted Sum (Current)

```python
# In optimize_2obj.py / rf_mip_model_v2.py
objective = Cmax + λ * sum(Q_k for k in machines)
model.setObjective(objective, GRB.MINIMIZE)
```

- λ = 0.001 recommended as starting point
- Simple, one MIP solve per instance
- Risk: if Cmax dominates, RF becomes irrelevant; if λ too large, scheduling degrades

### 4.2 Strategy B — Utopia Point Normalization

Three-stage approach (from `dce9102f-*.py` variant):

**Stage 1 — Utopia $T^\circ$ (pure makespan):**
```python
result_T = create_and_solve_model(w1=1, w2=0)
T_utopia = result_T['makespan']
```

**Stage 2 — Utopia $W^\circ$ (pure RF surrogate):**
```python
result_W = create_and_solve_model(w1=0, w2=1)
W_utopia = result_W['surrogate_value']
```

**Stage 3 — Normalized combination:**
$$\min \frac{w_1}{|T^\circ|} \cdot \text{makespan} + \frac{w_2}{|W^\circ|} \cdot \text{RF\_surrogate}$$

```python
coeff_T = w1 / max(abs(T_utopia), 1e-6)
coeff_W = w2 / max(abs(W_utopia), 1e-6)
```

| Strategy | Solves per instance | Pros | Cons |
|---|---|---|---|
| A (raw) | 1 | Fast | Dimension-sensitive; λ hard to tune |
| B (utopia) | 3 | Dimension-free; commensurable | 3× solve time |

### 4.3 Lexicographic Alternative (Recommended for FJSP)

```python
# Gurobi multi-objective — makespan always preferred
model.setObjectiveN(Cmax, index=0, priority=2, name="makespan")
model.setObjectiveN(surrogate_obj, index=1, priority=1, name="rf_surrogate")
```

> Makespan is primary; RF surrogate only breaks ties. This gives the most predictable behavior.

---

## 5. Public FJSP Benchmarks

### 5.1 Dataset Summary

| Dataset | Instances | Job × Machine Range | Best-Known | Difficulty |
|---|---|---|---|---|
| **Brandimarte** | mk01–mk10 | 10×6 to 20×15 | Known | Easy–Medium |
| **Kacem** | ka01–ka05 | 4×5 to 15×10 | Known | Very Easy |
| **Fattahi** | fa01–fa20 | 2×2 to 12×8 | Known | Easy |
| **Hurink edata** | 66 instances | 6×5 to 30×15 | Known | Medium |
| **Hurink rdata** | 66 instances | 6×5 to 30×15 | Known | Medium |
| **Hurink vdata** | 66 instances | 6×5 to 30×15 | Known | Medium–Hard |
| **BCdata** | 21 instances | 10×11 to 15×18 | Known | Hard |

### 5.2 Instance Selection Strategy for Paper

| Experiment Stage | Use | Why |
|---|---|---|
| **Development** | mk01 | Known optimum=40, fast to solve, verify correctness |
| **Parameter tuning** | mk01–mk05 | Small enough for multiple runs |
| **Main results** | mk06–mk10 + Hurink vdata | Larger, exact MIP struggles → RF advantage emerges |
| **Robustness** | Hurink edata + rdata | Different flexibility levels |

### 5.3 Standard Instance Format (JSON)

```json
{
  "num_jobs": 10,
  "num_machines": 6,
  "horizon": 300,
  "jobs": {
    "0": [[[0, 3], [1, 6], [2, 1]], [[3, 5], [0, 2]], ...],
    "1": [[[1, 4], [4, 7]], [[0, 3], [5, 2]], ...]
  },
  "feasible_machines": {
    "(0,0)": [0, 1, 2],
    "(0,1)": [3, 0]
  }
}
```

---

## 6. Recommended Research Pipeline (5 Stages)

### Stage 1: Setup & Baseline

**Goal:** Establish the exact MIP baseline on the chosen benchmark suite.

**Actions:**
1. Implement standard FJSP MIP (`fjsp_baseline.py`)
2. Run on mk01–mk10, record: makespan, gap, solve time, first feasible time
3. Verify against known optimal values
4. Identify instances where baseline struggles (gap > 5% after 300s)

**Output:** Baseline results table for all target instances

---

### Stage 2: RF Training & Feature Engineering

**Goal:** Train RF surrogate models with features that capture both assignment and sequencing quality.

**Actions:**
1. Generate training data: run heuristic schedule generation schemes (serial SGS with randomized rules) to produce 5000+ diverse feasible schedules per instance
2. Extract features per schedule:
   - **Assignment features:** `num_used_machines`, per-machine `load_k`, `ops_k`
   - **Sequencing features (CRITICAL — add these):** critical path length, max job tail time, total idle time, machine utilization variance, precedence chain depth
3. Train per-machine RFs or a single global RF predicting makespan
4. Export models as joblib + tree structure txt

**Key improvement over current:** Add sequencing-sensitive features. Without them, the surrogate can only learn assignment-level patterns.

**Output:** `rf_models/` with joblib files + tree structure exports

---

### Stage 3: RF-Embedded MIP Construction

**Goal:** Build the full RF+MIP model and solve.

**Actions:**
1. Parse RF trees → `forest_rules` (leaf paths with thresholds)
2. Build Gurobi model with:
   - Standard FJSP constraints (assignment, precedence, non-overlap, makespan)
   - Feature-linking constraints (e.g., `load_k = sum(p * x)`)
   - RF leaf activation constraints (uniqueness, big-M paths)
3. **Crucial:** Use lexicographic or weighted objective with Cmax as primary
4. Solve with configurable time limit and gap tolerance

**Gurobi parameters:**
```python
model.setParam("TimeLimit", 300)
model.setParam("MIPGap", 0.01)
model.setParam("Threads", 16)
model.setParam("Heuristics", 0.2)
big_m_constant = 1e4  # Tightened from 1e5
```

**Output:** Optimization results per instance (assignments, makespan, solve time)

---

### Stage 4: Diagnostic Experiments

**Goal:** Identify and fix objective misalignment before writing the paper.

**Actions:**
1. **Fixed-assignment diagnostic:** Extract RF+MIP assignment → re-solve exact sequencing → compare makespans
2. **Objective surgery:**
   - Run with `min Cmax` only (baseline)
   - Run with `min Cmax + λ·RF_surrogate` (weighted)
   - Run with lexicographic `Cmax → RF` (recommended)
3. **Warm-start experiment:** Feed RF-generated schedule as MIP start to baseline, compare convergence
4. **Ablation:** Remove RF entirely (random assignment + exact sequencing) as lower bound

**Expected findings (based on your current results):**
- RF+MIP raw: poor makespan (sequencing unoptimized)
- RF assignment + exact sequencing: moderate improvement
- Lexicographic Cmax→RF: near-baseline quality with RF tie-breaking
- RF warm-start: faster convergence on large instances

**Output:** Diagnostic table + convergence curves

---

### Stage 5: Paper Writing

**Goal:** Produce a complete draft targeting an MS&E journal.

**Recommended journal:** C&IE, EJOR, or IJPR for algorithm-heavy work; MS or OR if contribution is methodological enough.

**Paper structure:**
```
§1 Introduction
    - Motivating example: FJSP in semiconductor / automotive manufacturing
    - Gap: exact MIP slow on large instances; existing heuristics don't learn from data
    - Contribution: (1) RF-to-MIP encoding for FJSP, (2) diagnostic framework, (3) two-stage approach

§2 Literature Review
    - FJSP exact methods (MIP, CP, B&B)
    - ML for scheduling (RL, surrogate models, learning-to-branch)
    - Predict-then-optimize in OR

§3 Problem Formulation
    - Standard FJSP MIP (notation, variables, constraints)
    - RF surrogate motivation

§4 RF-Embedded MIP Model
    - RF architecture (features, training, tree extraction)
    - MIP encoding (leaf activation, feature linking, big-M constraints)
    - Objective design (lexicographic Cmax→RF)

§5 Diagnostic Analysis
    - Fixed-assignment experiment design
    - Results: decomposition into assignment vs. sequencing error
    - Ablation: random assignment baseline

§6 Computational Experiments
    - Setup: instances, hardware, solver config
    - Main results: benchmark comparison table
    - Convergence analysis: MIP gap over time
    - Warm-start effectiveness
    - Sensitivity: λ parameter, time limits

§7 Managerial Insights / Discussion
    - When to use RF+MIP vs. pure MIP
    - Interpretability of RF routing decisions
    - Limitations and future work

§8 Conclusion
```

**Writing patterns to use (from management-science-writing skill):**
- Pattern 1: Motivating example (real manufacturing numbers)
- Pattern 2: Gap table (8-12 references, explicit differentiation)
- Pattern 4: SAR Managerial Insight (Setting → Finding → Recommendation)

---

## 7. Quick Reference: Code Architecture

```
project/
├── fjsp_baseline.py              # Exact FJSP MIP (baseline)
├── schedule_generator.py         # Heuristic schedule generation for RF training
├── rf_train.py                   # Train per-machine + global RFs
├── rf_mip_model.py               # Core RF+MIP model (build + solve)
├── diagnose_fixed_assignment.py  # Fixed-assignment diagnostic
├── experiment_runner.py          # Batch run all instances
├── stats_analysis.py             # Statistical tests + tables
│
├── data/
│   ├── instances/                # mk01.json ... mk10.json, hurink/
│   └── rf_models/
│       ├── per_machine/          # rf_machine_0..5.joblib
│       └── global/               # global_rf.joblib
│
├── results/                      # Optimization outputs by instance
├── figures/                      # Convergence curves, gap plots
│
└── paper/
    ├── manuscript.tex            # INFORMS-style LaTeX
    └── refs.bib
```

---

## 8. Known Pitfalls & Prevention

| Pitfall | Symptom | Prevention |
|---|---|---|
| **Surrogate–true objective misalignment** | RF+MIP gap = 1% but makespan terrible | Diagnostic Stage 4; always include Cmax in objective |
| **Per-machine RF double-counting** | $\sum Q_k$ favours fewer machines | Use global RF or epigraph form: $\min T \text{ s.t. } T \geq Q_k$ |
| **Sequencing features missing** | Fixed-assignment makespan ≈ RF+MIP makespan | Add critical path, tail time, idle time features to RF |
| **Training data distribution shift** | Good on training, bad on optimization | Generate data from diverse heuristics; consider iterative retraining |
| **big-M too loose** | Slow solve, weak bounds | Tighten per-feature M based on actual value ranges |
| **Comparing different gap tolerances** | Unfair baseline vs. RF+MIP comparison | Set same `MIPGap` for all models in experiments |
| **Only testing on mk01** | Overfit to easy instance | Include mk05–mk10 and Hurink vdata in main results |

---

## 9. Transition from Rider Stress Domain

If porting the `refer/` methodology to FJSP, note these key differences:

| Aspect | Rider Stress (`refer/`) | FJSP |
|---|---|---|
| **RF output** | 2D `[delay, stress]` | scalar `makespan` |
| **Dynamic features** | `n_order`, `avg_distance` (varies per courier) | `load_k`, `num_used_machines` (varies per assignment) |
| **Sequence model** | Courier trip sequence (1,2,...,cap) | Machine operation sequence (δ variables) |
| **Multi-objective** | delay vs. stress trade-off | makespan vs. surrogate alignment |
| **Simulation** | 30-day closed-loop replay | Static instance evaluation |
| **Instance scale** | ~10 orders per 5-min slot | 55–225 operations per instance |

**Reusable components from `refer/`:**
- Tree traversal + rule extraction (`traverse()` function)
- Leaf activation constraints (C8–C9, C10–C12)
- big-M dynamic feature constraints
- Multi-objective utopia point normalization (Strategy B)
- Gurobi parameter tuning (TimeLimit, MIPGap, Threads)

**Must rewrite for FJSP:**
- Feature set (remove GPS/shopping/courier features, add manufacturing features)
- Assignment variable indexing (courier→machine, order→operation)
- Sequence constraints (courier capacity→machine non-overlap δ)
- Training data generation (driver simulation→heuristic schedule SGS)

---

## 10. Quick-Start Prompts

- "Setup FJSP baseline on mk01–mk10 and verify against known optima" → Stage 1
- "Design RF features for FJSP with sequencing information" → Stage 2
- "Build the RF+MIP model with lexicographic objective" → Stage 3
- "Run the fixed-assignment diagnostic experiment" → Stage 4
- "Write the computational experiment section for C&IE submission" → Stage 5
- "Compare raw weight vs. utopia point normalization strategies" → Section 4
- "Port the rider stress RF+MIP code to FJSP benchmarks" → Section 9

---

> Built from `refer/` rider stress scheduling project + FJSP diagnostic experiments.  
> Reference skill format: [management-science-writing](https://github.com/liyuanbo1024/management-science-writing) (agentskills.io specification)
