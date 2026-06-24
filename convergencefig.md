# Figure X: Judge Convergence Curve (Retrieval-Induced Agreement)

## Motivation

Most of the paper's findings can be summarized as:

Baseline PRM
→ high uncertainty

Retrieval-aware PRM
→ lower uncertainty

Lower uncertainty
→ less need for disagreement-aware routing

Currently this is spread across Exp 7, Exp 10, and Exp 14 tables.

This figure visualizes the entire story in one chart.

---

# Research Question

Does retrieval context cause independent judges (VersaPRM and DG-PRM) to converge over the course of a trajectory?

If yes:

* Retrieval reduces uncertainty
* Disagreement becomes less informative
* Explains why Retrieval-Aware + UA-3 does not outperform Retrieval-Aware alone

---

# Data Required

For every step:

```python
trajectory_id
step_idx

versa_score_baseline
dgprm_score_baseline

versa_score_retrieval
dgprm_score_retrieval

versa_score_retrieval_compressed
dgprm_score_retrieval_compressed
```

Compute:

```python
baseline_disagreement =
abs(versa_score_baseline - dgprm_score_baseline)

retrieval_disagreement =
abs(versa_score_retrieval - dgprm_score_retrieval)

compressed_disagreement =
abs(
    versa_score_retrieval_compressed
    -
    dgprm_score_retrieval_compressed
)
```

---

# Normalize Trajectories

Do NOT use raw step numbers.

Different trajectories have different lengths.

Normalize each trajectory:

```python
progress =
step_idx / total_steps
```

Map to:

```text
0%
10%
20%
...
100%
```

Bin into 10 buckets.

---

# Curves To Plot

## Curve 1 — Baseline PRM

Question + Step

Expected:

Highest disagreement.

This is the original system.

---

## Curve 2 — Retrieval-Aware PRM

Question + Retrieval + Step

Expected:

Lower disagreement.

Represents Exp 10.

---

## Curve 3 — Compressed Retrieval

Question + Compressed Retrieval + Step

Expected:

Very close to Retrieval-Aware.

Supports the token-saving result from Exp 10b.

---

## Curve 4 — UA-3 Triggered Steps Only

Subset of steps where:

```python
disagreement > tau
```

Expected:

Higher disagreement region.

Shows where uncertainty-adaptive routing activates.

---

## Curve 5 — Oracle Reference

Reference line:

```python
oracle_disagreement = 0
```

Dashed line.

Represents perfect agreement.

---

# Figure Layout

X-axis:

Trajectory Progress (%)

```text
0
10
20
...
100
```

Y-axis:

Mean Disagreement

```python
mean(abs(versa - dgprm))
```

---

# Example Appearance

```text
Disagreement

0.30 ┤ Baseline
     │ ╲╱╲╱╲╱╲╱╲╱

0.25 ┤

0.20 ┤ Retrieval-Aware
     │ ╲╱╲╱╲╱╲

0.15 ┤

0.10 ┤ Compressed Retrieval
     │ ╲╱╲╱╲

0.05 ┤

0.00 ┤ Oracle
     └──────────────────────────
       0 20 40 60 80 100
          Trajectory Progress
```

Use smoothing:

```python
rolling_window = 3
```

or LOWESS.

Goal:

Look like a signal/convergence plot.

---

# Secondary Variant (Recommended)

Create a second version with confidence bands.

For each bucket:

```python
mean_disagreement
std_disagreement
```

Plot:

```text
mean
± std
```

This follows common uncertainty visualization practice and makes convergence easier to interpret visually.

---

# Statistics To Report

For each curve:

```python
mean_disagreement
max_disagreement
area_under_curve
```

Most important:

```python
AUC_disagreement
```

Lower is better.

Example:

Baseline AUC = 0.214

Retrieval AUC = 0.137

Reduction = 36%

```

This gives a single quantitative summary.

---

# Hypotheses

H1:

Retrieval-Aware disagreement AUC
<
Baseline disagreement AUC

---

H2:

Compressed Retrieval disagreement AUC
≈
Retrieval-Aware disagreement AUC

within 10%.

---

H3:

UA-3-triggered steps concentrate in
high-disagreement regions.

---

# Expected Paper Narrative

Baseline:

Judges disagree frequently.

↓

Retrieval provides missing evidence.

↓

Judges converge.

↓

Uncertainty decreases.

↓

Disagreement routing becomes less necessary.

↓

Explains why:

Retrieval-Aware + UA-3

≈

Retrieval-Aware alone.

---

# Figure Caption Draft

Retrieval-aware scoring reduces disagreement between VersaPRM and DG-PRM throughout agent trajectories. Compressed retrieval preserves most of this convergence behavior. The reduction in disagreement explains why uncertainty-adaptive routing provides limited additional gains once retrieval context is available.
```
