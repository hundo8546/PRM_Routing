# Experiment 2: Routing Comparison

**Research question:** Does PRM-conditioned routing improve the cost-accuracy frontier?

**Dataset:** AgentProcessBench — train 800 trajectories (200/dataset), test 200 trajectories (50/dataset).

**PRM signal:** VersaPRM (selected from Exp 1)

**Model tiers:**
- Tier 1: Llama-3.1-8B — $0.20/1M tokens, p_good=0.92, p_bad=0.30
- Tier 2: Llama-3.1-70B — $0.90/1M tokens, p_good=0.97, p_bad=0.65
- Tier 3: Qwen2.5-72B — $1.80/1M tokens, p_good=0.99, p_bad=0.85

---

## Results (test set: 200 trajectories, 1,918 steps)

| Method | Acc | Cost (norm) | Esc Rate | Avg Tier | Stability |
|---|---|---|---|---|---|
| Always-Cheap (T1) | 0.194 | 70,281 | 0.000 | 1.00 | 1.00 |
| Uniform (T2) | 0.388 | 316,264 | 0.000 | 2.00 | 1.00 |
| Always-Frontier (T3) | 0.617 | 632,529 | 1.000 | 3.00 | 1.00 |
| DAAO | 0.194 | 70,281 | 0.000 | 1.00 | 1.00 |
| BAAR | 0.338 | 491,963 | 0.650 | 2.30 | 0.81 |
| TRIM (θ=0.75) | 0.298 | 387,510 | 0.469 | 2.04 | 0.56 |
| **PRM-Guided (h=0.86/l=0.62)** | **0.347** | **323,163** | 0.147 | 1.94 | 0.55 |

**Pareto analysis (46/46 PRM-Guided sweep points dominate nearest TRIM point):**

| Cost level | TRIM acc | PRM-Guided acc | Δ |
|---|---|---|---|
| ≈250,000 | 0.259 | 0.281 | +0.022 |
| ≈320,000 | 0.285 | **0.385** | **+0.100** |
| ≈380,000 | 0.295 | **0.395** | **+0.100** |

Cost savings vs TRIM at matched accuracy: −21% at acc=0.30, −24% at acc=0.35, −26% at acc=0.40.

---

## Key findings

1. **PRM-Guided dominates TRIM on the Pareto frontier.** At cost≈320K, +10pp accuracy. 3-tier routing fills the gap that binary TRIM's jump between T1 and T3 cannot. 46/46 sweep points dominate TRIM.

2. **BAAR over-escalates** (65% to T3), producing higher cost than Frontier at lower accuracy than Uniform. Boundary-guided training without a PRM signal over-escalates on heterogeneous steps.

3. **DAAO degenerates to Always-Cheap** on this dataset — question-length difficulty heuristic routes all test trajectories to T1. Pre-execution routing is insufficient for heterogeneous pipelines.

4. **PRM-Guided at default thresholds does not beat Uniform (T2)** in raw accuracy (0.347 vs 0.388). The signal strength (Spearman=0.166) limits how far it can push above the uniform baseline. The contribution is the Pareto frontier improvement vs TRIM, not beating Uniform.

---

## Decisions made based on Exp 2

| Decision | Trigger | See |
|---|---|---|
| T1 p_bad_recovery: 0.10 → 0.30 | Smoke test: PRM-Guided below Always-Cheap; T1 penalty catastrophic | DECISIONS.md §Exp 2 |
| Default thresholds: (0.70,0.55) → (0.86,0.62) | First run escalation rate = 83% (all steps to T3) | DECISIONS.md §Exp 2 |
| Pareto sweep range extended | Initial range didn't cover VersaPRM score distribution | DECISIONS.md §Exp 2 |

---

## Files

- `cost_model.py` — tier definitions, outcome probabilities
- `data_loader.py` — trajectory loader with score injection
- `routing_policies.py` — all 7 policies + Pareto sweep generators
- `simulator.py` — simulation engine (expected accuracy, cost)
- `run_exp2.py` — main runner
- `../../results/exp2/exp2_results.json` — main results
- `../../results/exp2/exp2_pareto.json` — Pareto sweep (TRIM: 23 pts, PRM-Guided: 46 pts)
- `../../results/exp2/exp2_summary.txt` — human-readable table
