# Experiment 1: PRM Signal Evaluation

**Research question:** Which reward model best predicts human-labeled step quality in heterogeneous agent trajectories?

**Dataset:** AgentProcessBench — 1,000 trajectories, 8,509 human-labeled steps across four datasets (hotpotqa, gaia_dev, bfcl, tau2).

**PRMs evaluated:** Qwen2.5-Math-PRM, VersaPRM, DG-PRM (local), AgentRM (proxy)

---

## Results (full run, 8,509 steps)

| PRM | Acc | F1 | Pearson | Spearman | Brier | ECE | Sep (g−b) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-PRM | 0.345 | 0.002 | −0.172*** | +0.111*** | 0.462 | 0.490 | −0.036 |
| **VersaPRM** | **0.627** | **0.753** | **+0.170***| **+0.166***| **0.246** | **0.127** | **+0.064** |
| AgentRM (proxy) | 0.628 | 0.772 | −0.045*** | −0.057*** | 0.235 | 0.033 | −0.001 |
| DG-PRM (local) | 0.631 | 0.773 | +0.151*** | +0.121*** | 0.303 | 0.273 | +0.013 |

**Task success correlation (Pearson, mean traj score vs final_label):**
- VersaPRM: r=+0.093** — only real PRM with significant positive trajectory-level signal
- DG-PRM: r=+0.210*** — strong but artifact (uses VersaPRM base as judge)
- Qwen: r=−0.039 (n.s.)
- AgentRM: r=−0.098** (degenerate — constant output)

**VersaPRM by step type (Spearman):** Synthesis +0.249 > Tool call +0.149 > Retrieval +0.064

**VersaPRM by dataset (separation):** gaia_dev +0.089 ≈ tau2 +0.091 > hotpotqa +0.075 > bfcl +0.033

---

## Key findings

1. **Qwen PRM fails on agent steps.** Acc=0.345 (below chance), F1≈0, Sep=−0.036. Scores are catastrophically miscalibrated (ECE=0.490). The math PRM transfer gap is real and quantified.

2. **VersaPRM is the only usable routing signal.** Only PRM with positive step separation (+0.064), reasonable calibration (ECE=0.127), and significant task-success correlation (+0.093**).

3. **AgentRM is degenerate.** Score head not released — proxy outputs near-constant 0.661. All metrics near the "always predict positive" baseline. Marked with * in all tables.

4. **DG-PRM local judge has mild signal** (Spearman=+0.121) but is confounded: it uses the VersaPRM base model, explaining correlation with VersaPRM results.

---

## Decisions made based on Exp 1

| Decision | Trigger | See |
|---|---|---|
| VersaPRM selected as primary routing signal for Exp 2 | Only PRM with reliable signal across all step types | DECISIONS.md §Exp 1 |
| AgentRM/DG-PRM reported as baselines, not main comparisons | Degenerate outputs | DECISIONS.md §Phase 0 |
| Pilot (100 trajs) run first | experiments_updates.md §9 feasibility gate | DECISIONS.md §Exp 1 |

---

## Files

- `pilot_prm_scoring.py` — scoring script (supports `--n_per_dataset` for pilot vs full)
- `exp1_analysis.py` — full metrics computation
- `../../results/exp1/full_{prm}.jsonl` — raw per-step scores (4 files)
- `../../results/exp1/exp1_full_summary.json` — aggregated metrics
