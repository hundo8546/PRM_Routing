# Research Improvements Needed

Compiled from code audit. Ordered by severity — fix top-down.

---

## 1. CRITICAL: Replace Simulated Routing Results with Real Ones

**Problem:** Every TSR, accuracy, and cost number in Exp 2–8 and 13–15 comes from a
counterfactual simulator (`simulator.py`) that uses six hardcoded assumed probabilities:

```
T1 (Llama-3.1-8B):   p_good_step_success=0.92, p_bad_step_recovery=0.30
T2 (Llama-3.1-70B):  p_good_step_success=0.97, p_bad_step_recovery=0.65
T3 (Qwen2.5-72B):    p_good_step_success=0.99, p_bad_step_recovery=0.85
```

These were never measured. No model was ever run on AgentProcessBench for routing evaluation.

**What Exp 12 found when it tried to calibrate:** GPT-5.5 p_good=0.46, p_bad=0.48
(essentially random), Claude-Opus-4.8 p_good=0.26, p_bad=0.37. The assumed values
are likely wildly off.

**Fix:** Run each of the three tier models as actual agents on the AgentProcessBench
test set (200 trajectories, 50 per dataset). Record which tasks succeed and which fail.
This replaces the simulator entirely with real outcomes.

**Affects:** Exp 2, 3, 4, 5, 6, 7, 8, 13, 14, 15 — every headline number in the paper.

---

## 2. CRITICAL: Replace Baseline Proxies with Real Implementations

The DAAO, BAAR, and TRIM implementations in `routing_policies.py` are simplified proxies,
not the actual methods from the papers. Beating a weaker baseline is not a valid comparison.

### 2a. TRIM — wrong PRM, wrong routing mechanism

**Current implementation (`TRIMStyle`):**

- Uses VersaPRM (not Qwen2.5-Math-PRM which TRIM actually uses)
- Lag-1 routing: score step T → route step T+1
- Binary: T1 or T3 only

**Real TRIM (arXiv:2601.10245):**

- Uses Qwen2.5-Math-PRM to score the cheap model's output
- If score < threshold, **re-generates the current step** with the strong model
- Same step re-generation, not next-step routing

**Fix:** Implement real TRIM mechanism:

1. Run cheap model (T1) on each step
2. Score with Qwen2.5-Math-PRM (scores already in `results/exp1/full_qwen.jsonl`)
3. If score < θ, re-run same step with T3
4. Compare cost + outcome to PRM-Guided

Note: Qwen PRM gives ~constant scores (~0.14) on agent steps — this is a real
finding that will likely still favor our approach, but it must be measured, not assumed.

### 2b. BAAR — wrong feature set, wrong model class

**Current implementation (`BAARStyle`):**

- Logistic regression on 5 structural features only: step type (one-hot), normalized
  position, normalized trajectory length
- No access to step content or interaction history

**Real BAAR (arXiv:2602.21227):**

- Router sees full interaction history: task input + past actions + tool outputs
- Trained via Boundary-Guided SFT (BoSFT) using always-cheap / always-expensive
  boundary policies to generate training labels
- Policy selects between cheap and expensive model at each step

**Fix:**

1. Extract step content from AgentProcessBench trajectories as features
2. Train a proper learned routing policy using boundary policy labels:
   - Label a step "use cheap" if cheap model succeeds on it
   - Label a step "use expensive" if cheap model fails but expensive succeeds
3. At minimum, add step content embeddings to the logistic regression features
   even if full SFT is not feasible

### 2c. DAAO — wrong difficulty scorer

**Current implementation (`DAAOStyle`):**

- Heuristic formula: `d = 0.5×(len/500) + 0.3×(commas/10) + 0.2×(WH-word)`
- Routes entire trajectory to fixed tier

**Real DAAO (arXiv:2509.11079):**

- VAE-based difficulty estimation with self-adjusting policy
- Adapts workflow depth and operator selection, not just model tier
- Three interdependent modules: VAE difficulty estimator, operator allocator, LLM router

**Fix:** DAAO is the least critical to get exactly right because the core limitation
(route once pre-execution, no step-level adaptation) is preserved by our proxy.
But the difficulty scorer should at minimum use a trained classifier, not a heuristic.
Option: fine-tune a small classifier on AgentProcessBench question difficulty
using final task success as the label.

---

## 3. HIGH: Empirically Calibrate Tier Probabilities (if simulation is kept)

If full re-running of all experiments is not feasible, the minimum fix is to
measure the actual per-tier step success rates empirically and replace the
hardcoded values in `cost_model.py`.

**Method:**

1. Sample ~100 steps with human_label==1 (good) and ~100 with human_label==-1 (bad)
2. Run each step through T1, T2, T3 models
3. Score output against ground truth
4. Measure empirical p_good_step_success and p_bad_step_recovery per tier
5. Update `cost_model.py` with real values
6. Re-run all simulation experiments

This is cheaper than re-running all routing experiments but still requires real inference.

---

## 4. HIGH: Convergence Figure — Missing DG-PRM Retrieval Scores

**Problem:** The planned convergence figure (Figure X: Judge Convergence Curve) requires
per-step disagreement between VersaPRM and DG-PRM under retrieval-aware inputs:
`|versa_retrieval - dgprm_retrieval|`

**What we have:**

- VersaPRM baseline scores: `results/exp1/full_versa.jsonl` ✓
- DG-PRM baseline scores: `results/exp1/full_dgprm.jsonl` ✓
- VersaPRM retrieval-aware scores: `results/exp10/exp10_retr_scores.jsonl` ✓ (1918 steps)
- DG-PRM retrieval-aware scores: **NOT COLLECTED**

The proxy `|versa_baseline - versa_retrieval|` was tested and found to have
Pearson r = -0.22 with actual judge disagreement — it would flip the story.

**Fix:** Re-run DG-PRM scoring with retrieval context using the same setup as
`experiments/exp910_compression/run_exp910.py` but for DG-PRM instead of VersaPRM.
Requires access to DG-PRM model and AgentProcessBench benchmark data on the cluster.

---

## 5. MEDIUM: Fix TRIM Comparison Using Available Qwen Scores

**Quick win available now (no new model runs needed):**

Qwen2.5-Math-PRM scores already exist for all 8,509 steps in
`results/exp1/full_qwen.jsonl`. Real TRIM uses this PRM.

Update `TRIMStyle` to use Qwen scores instead of VersaPRM scores and re-run Exp 2
routing comparison. This makes the TRIM baseline more faithful immediately.

Expected outcome: TRIM performs worse (Qwen mean score ≈ 0.14, near-constant on
agent steps, effectively random routing) — this strengthens our argument.

---

## 6. MEDIUM: Fix Threshold Calibration Leakage

**Problem:** `theta_low=0.62` and `theta_high=0.86` were computed from the full dataset
(all 8,509 steps including test set). Train-only p25=0.609, full-dataset p25=0.621.

**Fix:** Recompute thresholds using training set only (first 200 trajectories per dataset).dddsadfffsddsdssgdsassdd
Re-run affected experiments with corrected thresholds. The change is small (~0.01)
but should be reported correctly.

---

## 7. MEDIUM: Paper — Disclose Baseline Adaptations Explicitly

Add to experimental setup section:

> "Since DAAO, BAAR, and TRIM were not designed for heterogeneous agentic pipelines,
> we adapt each method's routing strategy to AgentProcessBench. TRIM is implemented
> using [Qwen2.5-Math-PRM / VersaPRM] as the scoring signal. BAAR is implemented as
> a logistic boundary classifier; the original SFT policy is not portable without
> retraining on AgentProcessBench trajectories. DAAO's query-level routing structure
> is preserved; the VAE difficulty estimator is replaced with [X]."

---

## 8. LOW: Fix BAAR Docstring

`routing_policies.py` line 161: docstring claims `versa_score` is a feature,
but `_featurise()` does not include it. Misleading but does not affect results.

```python
# Fix: either add versa_score to _featurise() or remove it from the docstring
```

---

## 9. LOW: Update Approach Comparison Diagrams

`paper/diagrams/approach_comparison.txt` currently reflects our proxy implementations,
not the real paper architectures. Update after fixes above are in place so diagrams
match what was actually run.

---

## 10. MEDIUM: Fix Caveman Compression Citation and Implementation

**Problem — two separate issues:**

**10a. The citation is wrong.**
The paper says:
```latex
Caveman-style compression~\cite{jiang2023} and LLMLingua~\cite{llmlingua2023}
```
`\cite{jiang2023}` points to the LLMLingua paper. "Caveman-style compression" and
LLMLingua are being conflated. LLMLingua is token-importance scoring via a small
language model — a completely different technique from stopword removal.

**10b. Neither method was actually used.**
The `caveman_compress()` function in `experiments/exp910_compression/run_exp910.py`
is a custom stopword-removal function written from scratch. It was NOT imported from:
- The JuliusBrussee/caveman GitHub repo (a Claude Code skill with no Python API,
  designed to compress model output responses, not arbitrary input text)
- LLMLingua (a learned compression method requiring a separate small LM)

The function name "caveman_compress" is borrowed from the branding concept only.

**Fix:**
1. Remove the incorrect `\cite{jiang2023}` attribution for caveman-style compression
2. Describe the compression approach accurately in the paper:
   > "We apply a lightweight rule-based compression: stopwords and filler tokens are
   > removed while numbers, proper nouns, and domain terms are preserved."
3. Cite LLMLingua separately as related work only, not as the basis for our method
4. If LLMLingua is actually needed as a stronger compression baseline, implement it:
   `pip install llmlingua` and apply `PromptCompressor` to queries and retrieval passages

---

## Summary Table

| #   | Issue                           | Affects Results                   | Requires Compute                          | Priority |
| --- | ------------------------------- | --------------------------------- | ----------------------------------------- | -------- |
| 1   | Simulated routing outcomes      | All Exp 2–8, 13–15 TSR numbers    | Yes — run 3 models on benchmark           | CRITICAL |
| 2a  | TRIM proxy                      | TRIM comparison validity          | Partial — Qwen scores available           | CRITICAL |
| 2b  | BAAR proxy                      | BAAR comparison validity          | Yes — need trajectory content + training  | CRITICAL |
| 2c  | DAAO proxy                      | DAAO comparison validity          | Small — train difficulty classifier       | HIGH     |
| 3   | Assumed probabilities           | All simulation numbers            | Yes — 200 sample steps per tier           | HIGH     |
| 4   | Missing DG-PRM retrieval scores | Convergence figure                | Yes — re-run DG-PRM with retrieval        | HIGH     |
| 5   | TRIM uses wrong PRM             | TRIM Pareto curve                 | No — Qwen scores already on disk          | MEDIUM   |
| 6   | Threshold leakage               | Thresholds in Exp 2+              | No — recompute from train set             | MEDIUM   |
| 7   | Undisclosed baseline adaptations| Paper framing                     | No — writing change                       | MEDIUM   |
| 10  | Wrong caveman citation          | Paper credibility                 | No — writing change                       | MEDIUM   |
| 8   | BAAR docstring wrong            | Nothing                           | No                                        | LOW      |
| 9   | Diagrams reflect proxies        | Paper figures                     | No — after #2 fixed                       | LOW      |

---

## What Is Already Valid (Do Not Change)

- **Exp 1**: VersaPRM/DG-PRM/Qwen signal quality (Spearman correlations, precision).
  These are real empirical measurements on real scores vs real human labels.
- **Exp 9/10**: Compression ratios, token savings, signal quality under different input
  contexts. Real VersaPRM runs.
- **Exp 11**: GPT-5.5 as routing judge. Real API calls, real scores.
- **The core claim of Exp 1**: VersaPRM has actionable signal on heterogeneous agent
  steps; Qwen-Math-PRM does not. This is real and stands regardless of routing results.
