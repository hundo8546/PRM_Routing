# Experiments 6, 7, 8: Disagreement-Aware Routing

**Status:** Complete (no model loading — used existing Exp 1 score files)
**Priority order per additional_experiments.md:** Exp 7 > Exp 8 > Exp 6

---

## Exp 7: Multi-Judge Disagreement (Versa vs DG-PRM) — Highest Priority

**Signal:** `disagreement = abs(versa_score - dgprm_score)`
DG-PRM scores cluster near 0.9. High disagreement means Versa is LOW while DG is HIGH
→ VersaPRM detects a problem that DG misses → escalate to T3.

**Routing:** disagreement > τ → T3; else versa_score > θ_high → T1; else → T2

| Policy | Acc | Cost (norm) | Esc Rate | vs VersaPRM | vs Uniform |
|---|---|---|---|---|---|
| MultiJudge τ=0.15 | 0.378 | 438,942 | 0.459 | +0.032 | −0.010 |
| MultiJudge τ=0.20 | 0.367 | 418,714 | 0.400 | +0.020 | −0.021 |
| MultiJudge τ=0.30 | 0.342 | 314,076 | 0.121 | −0.005 | −0.047 |

**Pareto sweep (30 points):** 17/30 beat VersaPRM baseline.
Best point: τ=0.08, acc=**0.473** (above Uniform 0.388), cost=514,636, prec=0.509.

**Key finding:** Multi-judge disagreement improves over VersaPRM at the cost of higher escalation rates. At τ=0.08, achieves accuracy above Uniform (0.473 vs 0.388) — the first method to beat Uniform — but at 63% higher cost. The Pareto curve improvement over plain VersaPRM is real but modest.

---

## Exp 8: Temporal Disagreement — Second Priority

**Signal:** `drop = score_{t-2} - score_{t-1}` (score deterioration over last two steps)
Large drops signal deteriorating trajectory quality → escalate next step.

**Routing:** drop > τ_drop → T3; else standard 3-tier PRMGuided

| Policy | Acc | Cost (norm) | Esc Rate | vs VersaPRM |
|---|---|---|---|---|
| Temporal τ=0.10 | 0.360 | 363,447 | 0.260 | +0.013 |
| Temporal τ=0.15 | 0.352 | 343,726 | 0.213 | +0.006 |
| Temporal τ=0.25 | 0.347 | 325,155 | 0.153 | +0.000 |

Best sweep: τ=0.05, acc=0.374, cost=395,790.

**Key finding:** Temporal drops add marginal but consistent improvement over VersaPRM alone (+0.001 to +0.013). Novel signal — consecutive step quality autocorrelation is low (as shown by Oracle ceiling in Exp 3), but sharp drops are still informative. Low implementation overhead since all scores are already computed.

---

## Exp 6: PRM Disagreement (Versa vs Qwen) — Third Priority

**Signal:** `disagreement = abs(versa_score - qwen_score)`

**Critical limitation:** Qwen scores are near-constant (~0.14–0.16). Therefore:
`disagreement ≈ versa_score − 0.14`

High disagreement → Versa thinks step is GOOD (but Qwen always says bad). Forcing T3
when disagreement is high means routing GOOD steps to T3 — counterintuitive but
accidentally creates a high-escalation regime (like a shifted Always-Frontier).

| Policy | Acc | Cost (norm) | Esc Rate | vs VersaPRM |
|---|---|---|---|---|
| Dis(VQ) τ=0.40 | **0.573** | 544,825 | 0.746 | +0.226 |
| Dis(VQ) τ=0.55 | 0.525 | 471,236 | 0.532 | +0.179 |
| Dis(VQ) τ=0.65 | 0.475 | 411,804 | 0.372 | +0.128 |

**Key finding:** High numbers are an artifact — not a useful uncertainty signal. The high accuracy comes from routing most steps to T3 (74.6% at τ=0.40), equivalent to an aggressive Always-Frontier. Reported for completeness; not included in paper claims.

---

## Combined Signal (Exp 7 + 8)

`Combined(Δ=0.10, D=0.15)`: acc=**0.385**, prec=0.518, cost=451,264 → +0.038 vs VersaPRM, essentially tied with Uniform (0.388) at 43% higher cost.

Most promising direction (per additional_experiments.md) partially confirmed: combining temporal + multi-judge closes most of the gap with Uniform but doesn't dominate it at matched cost.

---

## Decisions made

| Decision | Trigger | See |
|---|---|---|
| Exp 6 flagged as artifact | Qwen scores ~0.14 constant → disagreement = versa_score offset | DECISIONS.md §Exp 6 |
| Focus Pareto analysis on Exp 7 | Only method to exceed Uniform accuracy (acc=0.473 at τ=0.08) | DECISIONS.md §Exp 7 |

---

## Files

- `disagreement_loader.py` — multi-PRM score merger + augmented trajectory builder
- `disagreement_policies.py` — PRMDisagreement, MultiJudge, Temporal, Combined + sweep generators
- `run_exp678.py` — main runner
- `../../results/exp6/exp6_results.json`
- `../../results/exp7/exp7_results.json`
- `../../results/exp8/exp8_results.json`
- `../../results/exp7/exp_combined_results.json`
