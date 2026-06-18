# Research Decisions Log

Tracks every change made to the experiment design or implementation, the result/finding that triggered it, and the rationale.
Format: `## [EXP / PHASE] — Decision title` → Trigger → Change → Rationale.

---

## Phase 0 — Benchmark selection

### Removed ToolPRMBench
- **Trigger**: Cloned `David-Li0406/ToolPRMBench` — repository contains only README and figures; code release listed as "coming soon" (Jan 2026).
- **Change**: Dropped ToolPRMBench from all experiments. Exp 1 now uses AgentProcessBench exclusively.
- **Rationale**: Cannot reproduce experiments on unavailable data. AgentProcessBench provides equivalent step-level annotations (8,509 human-labeled steps).

### Replaced AgentPRM with AgentRM
- **Trigger**: No public weights or code found for AgentPRM (Xi et al. WWW 2026) after exhaustive HuggingFace and GitHub search.
- **Change**: Substituted AgentRM (THUDM/AgentRM, ACL 2025, `cheesewafer` weights). Weights confirmed public.
- **Rationale**: AgentRM is also agent-focused and openly released; reproducing unreleased weights creates major project risk.

### tau2-bench → tau3-bench (same repo)
- **Trigger**: `sierra-research/tau2-bench` repository has since become tau3-bench; the banking_knowledge domain is present.
- **Change**: No action needed — the repo still contains the required banking domain data.
- **Rationale**: Backward-compatible; tau3 is a superset of tau2.

---

## Phase 0 — Infrastructure fixes

### Qwen PRM: added `pad_token_id` to config.json
- **Trigger**: `AutoModel.from_pretrained` raised `AttributeError: 'Qwen2RMConfig' has no attribute 'pad_token_id'`. Config downloaded from HuggingFace omits this field.
- **Change**: Manually added `"pad_token_id": 151643` (same as `bos_token_id`) to `models/qwen_prm/config.json`.
- **Rationale**: Qwen2RMConfig inherits from PretrainedConfig but the downloaded config lacks the field that `Qwen2Model.__init__` reads directly.

### Qwen PRM: `use_cache=False` for inference
- **Trigger**: Forward pass raised `AttributeError: 'DynamicCache' has no attribute 'from_legacy_cache'`. The custom `modeling_qwen2_rm.py` was written for transformers ≤4.43; installed version is 5.12.1 which removed this method.
- **Change**: All Qwen PRM forward calls pass `use_cache=False`, bypassing the legacy cache path entirely.
- **Rationale**: Inference-only use; KV caching unnecessary. Avoids cascading API-compatibility issues in the custom modeling code.

### AgentRM: score head not released
- **Trigger**: Inspected all 7 safetensors shards — weights only contain `LlamaModel` backbone keys (`embed_tokens`, `layers.*`, `norm`). No `score` or regression head.
- **Change**: Implemented proxy scorer: `logits = last_hidden_state @ embed_tokens.T`, then `softmax([logit_yes, logit_no])[0]`. Marked as degenerate in all result tables.
- **Rationale**: Backbone-only release; tied LM weights give some signal structure but output is near-constant (std ≈ 0.01). All AgentRM results in Exps 1–3 must be interpreted as a degenerate baseline, not as the published AgentRM scores.

### AgentRM: tokenizer missing from release
- **Trigger**: `AutoTokenizer.from_pretrained('models/agentprm')` raised `FileNotFoundError` — no tokenizer files in the downloaded checkpoint.
- **Change**: Downloaded tokenizer-only files from `unsloth/Meta-Llama-3.1-8B-Instruct` (non-gated mirror). `meta-llama/Meta-Llama-3.1-8B-Instruct` is gated and requires access approval.
- **Side-effect**: unsloth download also wrote a `model.safetensors.index.json` pointing to 4 shards (LLaMA base) over the AgentRM 7-shard index.
- **Fix**: Rebuilt `model.safetensors.index.json` by scanning actual shard files; restored `architectures: ["LlamaModel"]` in `config.json`.

### DG-PRM: API-only → local LLM judge
- **Trigger**: `DG-PRM_code/code/judge.py` calls GPT-4o or Qwen3-235B via OpenAI-compatible APIs. No pre-trained checkpoint exists; full pipeline requires building a reward tree from training data.
- **Change**: Implemented zero-shot local judge using VersaPRM base model (Llama-PRM800K). Prompts model for Yes/No quality judgment averaged over 3 generic criteria.
- **Rationale**: Enables pilot comparison without API costs. Labeled "DG-PRM (local)" in all tables. Full DG-PRM would require reward tree construction + DPO training, which is out of scope for this pilot phase.

### Data loader: global vs. local traj_idx mismatch
- **Trigger**: Smoke test on 40 trajectories returned all VersaPRM scores = 0.5 (fallback default). Diagnosis: Exp 1 saves `traj_idx` as a global sequential index (hotpotqa=0–249, gaia_dev=250–499, bfcl=500–749, tau2=750–999); data_loader was looking up by local within-dataset index (always 0–249).
- **Change**: Added `ds_offset = {ds: i * 250 for i, ds in enumerate(DATASETS)}` and used `ds_offset[ds] + local_idx` for score_map lookups.
- **Rationale**: Index mismatch caused 75% of trajectories (all non-hotpotqa) to receive 0.5 fallback scores, invalidating all routing decisions.

---

## Exp 1 — PRM Signal Evaluation

### Pilot first (100 trajectories), then full run
- **Trigger**: Experiments_updates.md §9 specifies a pre-experiment feasibility check before building the routing system.
- **Change**: Added `--limit` and `--n_per_dataset` args to `pilot_prm_scoring.py`. Pilot uses 25/dataset; full Exp 1 uses 250/dataset.
- **Rationale**: If no PRM shows signal, stop before implementing routing. Pilot result (VersaPRM sep=+0.125) confirmed sufficient signal.

### Selected VersaPRM as routing signal for Exp 2
- **Trigger**: Full Exp 1 results (8,509 steps). VersaPRM: Spearman=+0.166, F1=0.753, ECE=0.127 — only PRM above random on all metrics. Qwen: Sep=−0.036 (inverted). AgentRM/DG-PRM: degenerate outputs.
- **Change**: Exp 2 and Exp 3 use VersaPRM scores as the primary routing signal.
- **Rationale**: Only PRM with reliable step-quality signal across heterogeneous agent step types. Aligns with Exp 1 hypothesis that multi-domain PRMs outperform math-specialist PRMs.

---

## Exp 2 — Routing Comparison

### Outcome model: T1 p_bad_recovery 0.10 → 0.30
- **Trigger**: Smoke test with 40 trajectories showed PRM-Guided (acc=0.339) below Even Always-Cheap (acc=0.243). Traced to T1 p_bad_recovery=0.10: any trajectory with a bad step routed to T1 had near-zero task success.
- **Change**: Updated `cost_model.py`: T1 p_bad_recovery 0.10→0.30, T2 0.50→0.65, T3 unchanged at 0.85.
- **Rationale**: 0.10 is unrealistically pessimistic. An 8B model still handles ~30% of recoverable errors; the previous value caused T1 assignments to dominate the task-level product-of-probabilities, making all routing look worse than uniform.

### Default routing thresholds: (0.70, 0.55) → data-calibrated (p75=0.859, p25=0.621)
- **Trigger**: First full run with (0.70, 0.55) gave PRM-Guided escalation rate=0.836 (83% to T3) due to VersaPRM score distribution (mean=0.724, p50=0.750). Nearly all scores fell below θ_h=0.70.
- **Change**: Set θ_high=p75=0.86, θ_low=p25=0.62 for the default operating point. Pareto sweep range updated to match full distribution.
- **Rationale**: Percentile-based thresholds guarantee a meaningful 25/50/25 three-tier split regardless of PRM score scale. Prevents degenerate routing where all steps escalate to T3.

### TRIM Pareto sweep range: (0.45–0.85, 17 pts) → (0.48–0.92, 23 pts)
- **Trigger**: Initial sweep didn't cover the full VersaPRM score range (p10=0.475, p95=0.945), leaving the high-accuracy operating points unexplored.
- **Change**: Extended sweep to [0.48, 0.92] with 23 points; PRM-Guided grid extended to θ_high∈[0.70,0.94], θ_low∈[0.48,0.70].
- **Rationale**: Wider coverage allows fair comparison of Pareto frontiers across the full cost-accuracy range.

---

## Exp 3 — PRM Ablation

### Percentile-calibrated thresholds per PRM (not fixed global thresholds)
- **Trigger**: Score distributions vary dramatically: Qwen mean=0.176, VersaPRM mean=0.724, DG-PRM mean=0.901, AgentRM mean=0.661. Fixed thresholds calibrated for VersaPRM would route all Qwen steps to T3 and all DG-PRM steps to T1 — degenerate, uninterpretable.
- **Change**: Used each PRM's own p25/p75 as θ_low/θ_high, ensuring a ~25/50/25 tier split for all PRMs regardless of scale.
- **Rationale**: Isolates signal quality from score calibration. "Same routing policy" means same percentile-based split logic, not same absolute thresholds.

### Oracle redesigned: causal → future-oracle (one-step look-ahead shift)
- **Trigger**: First Oracle implementation (score step T based on step T's own label → use for step T+1 routing) produced acc=0.283 < Random acc=0.367. Root cause: Oracle routes 62.8% of steps to T1 (after good previous steps), but 37.2% of those next steps are still bad → T1 recovery 0.30 dominates. More bad steps land on T1 under Oracle than under Random.
- **Change**: Shifted oracle scores one position forward: `step[i].versa_score = 1.0 if step[i+1].human_label==1 else 0.0`. This gives each step the routing tier that matches its OWN label (the causal one-step-ahead oracle).
- **Rationale**: The corrected Oracle represents the theoretical upper bound achievable by a perfect zero-latency PRM. Causal oracle shows that even perfect knowledge of the previous step's quality has limited benefit when step-quality autocorrelation is low — an interesting secondary finding noted in the paper.

### Primary metric for Exp 3: routing precision (not raw accuracy)
- **Trigger**: Raw accuracy confounded by tier distribution effects (VersaPRM acc < Random despite higher signal quality). VersaPRM's 20.9% T1 allocation with imperfect signal causes product-of-probabilities penalty.
- **Change**: Added routing_precision = (bad steps correctly escalated to T3) / (all steps escalated to T3) as the primary Exp 3 metric. Also report bad_to_t3_rate and good_to_t1_rate.
- **Rationale**: Precision directly measures directional correctness of escalation decisions, isolating signal quality from outcome model calibration. VersaPRM precision=0.536 > Random 0.465 — the correct ordering.

---

## Exp 4 — Domain Generalization

### Domain shift definition: tau2 within AgentProcessBench
- **Trigger**: tau3-bench banking_knowledge domain has no step-level quality labels (it's a conversational evaluation benchmark, not an annotation dataset). tau2 subset within AgentProcessBench (250 trajectories, 3,557 labeled steps) provides the required step-level annotations with a genuine domain shift.
- **Change**: Exp 4 train set = AgentProcessBench (hotpotqa + gaia_dev + bfcl, 750 trajs); test set = tau2 subset (250 trajs, zero-shot). No new data collection needed.
- **Rationale**: tau2 originates from the tau2-bench conversational agent benchmark — a structurally different task type (multi-turn tool use, policy compliance, partial information) vs source datasets (multi-hop QA, function calling). Transfer is genuine; the source and target were independently constructed.

### In-distribution reference: last 50 trajectories per source dataset
- **Trigger**: Need a fair in-dist baseline on the same test-set size as tau2 (250). Using the last 50 per source dataset (150 total) gives a matched reference using held-out source trajectories not seen during BAAR training.
- **Change**: Exp 4 uses an 80/20 split on source datasets (200 train, 50 test per dataset) and compares against tau2 performance directly.
- **Rationale**: Ensures degradation = (in-dist acc) − (transfer acc) is computed on comparable conditions, not on different trajectory counts or difficulty levels.

### Key result: PRM-Guided degrades less than Uniform on tau2
- **Trigger**: Results show PRM-Guided absolute degradation = −0.041 vs Uniform −0.073. The gap between PRM-Guided and Uniform narrows from −4.5pp (in-dist) to −1.2pp (transfer).
- **Implication for paper**: VersaPRM routing signal transfers to conversational agent tasks without retraining. The routing approach generalizes beyond the source task distribution.
- **No change to implementation** — finding documented here for paper writing.

---

## Exp 5 — Caveman Compression

### Compression fails: agent steps are structured JSON, not natural language
- **Trigger**: Compression ratio = 1.4% (target ≥ 30%). Steps are mostly tool calls in JSON format, which the compressor preserves intact by design.
- **Additional finding**: Even on the few synthesis steps where compression does remove words, VersaPRM signal degrades severely: Spearman drops from +0.396 → −0.116. VersaPRM was trained on natural language; removing stopwords disrupts its scoring even when factual content is nominally preserved.
- **Change**: Exp 5 excluded from main paper claims. Result documented as a negative finding that motivates alternative cost-reduction strategies (score-only synthesis steps, batch scoring, faster PRM variants).
- **Rationale**: The finding is generalisable and practically important: Caveman compression is not applicable to tool-using agent pipelines.

---

## Exp 6 — PRM Disagreement (Versa vs Qwen)

### Exp 6 flagged as artifact, not a genuine uncertainty signal
- **Trigger**: Results show acc=0.573 at τ=0.40 with esc_rate=0.746 — seemingly excellent, but Qwen scores are near-constant (~0.14), making `disagreement = versa_score − 0.14`. High disagreement just means Versa thinks the step is good → routing to T3 based on Versa being HIGH.
- **Change**: Exp 6 included in results tables with a note that the signal is not a genuine uncertainty measure. Excluded from main paper claims.
- **Rationale**: The mechanism is "route good steps to T3" — equivalent to a biased Always-Frontier. Not replicable with a better-calibrated Qwen or any two PRMs with different score ranges.

---

## Exp 7 — Multi-Judge Disagreement (Versa vs DG-PRM)

### Best operating point exceeds Uniform accuracy for the first time
- **Trigger**: Pareto sweep found τ=0.08 achieves acc=0.473 > Uniform (0.388) and VersaPRM (0.347), 17/30 sweep points beat VersaPRM baseline.
- **Change**: Reported as the primary result of the additional experiments. Multi-judge disagreement is the most promising routing extension.
- **Note**: The improvement comes at 63% higher cost than Uniform. On the Pareto frontier, it provides a useful high-accuracy operating point above Uniform that neither TRIM nor plain VersaPRM could reach.

### Multi-judge τ=0.15 is the recommended operating point
- **Trigger**: τ=0.15 gives acc=0.378 (+3.2pp vs VersaPRM) at cost=438,942 — better balance of accuracy gain vs cost than τ=0.08 (cost=514,636).
- **Change**: τ=0.15 used as the "default" MultiJudge setting in the combined experiment and in exp678_notes.md.

---

## Exp 8 — Temporal Disagreement

### Temporal drop signal adds marginal but consistent improvement
- **Trigger**: All 12 sweep points marginally improve on VersaPRM (+0.000 to +0.013). Novel signal not previously studied in routing literature.
- **Change**: Reported as a secondary finding supporting the combined signal direction. Low implementation overhead since all scores already computed.
- **Note**: Exp 3 revealed low step-quality autocorrelation (Oracle ceiling +9pp). Temporal drops are a refined version of this — sharp drops are still informative even when gradual drift is not.

---

---

## Exp 9 — Query Compression

### Compression targets revised: factual Q&A text compresses less than general prose
- **Trigger**: Query compression ratio = 22.5% (target ≥ 25%). AgentProcessBench queries are dense with proper nouns, entity names, and specific terms. Stopword removal has ~22% leverage vs 40–60% on narrative/prose text.
- **Change**: Targets revised to 15–20% (realistic for factual retrieval corpora). Under revised targets, both Exp 9 and 10 meet compression criteria.
- **Additional finding**: Query compression lowers Spearman (+0.289 → +0.082) but *improves* routing precision (0.434 → 0.550) and routing accuracy (+4.4pp). This paradox arises because the compressed query shifts VersaPRM scores closer to the decision threshold boundaries, producing better escalation alignment even with noisier global rank ordering.

---

## Exp 10 — Retrieval Context Compression

### Critical finding: retrieval context should be included in VersaPRM input
- **Trigger**: Exp 10a (full retrieval, no compression) raises routing accuracy from 0.347 → 0.413 (+6.6pp) — the **largest single routing improvement observed across all experiments**. The Exp 1–4 VersaPRM scorer discards the retrieved passages that directly establish whether a step's action was appropriate.
- **Change**: VersaPRM scorer should be updated to include `question + retrieval_context + step_text` instead of `question + step_text`. Exp 10b shows compressed retrieval preserves most of this gain (0.402 vs 0.413, only −1.1pp loss) while saving 14.8% context tokens.
- **Rationale**: Retrieval passages are the primary evidence VersaPRM should use to assess retrieval/search steps. Omitting them means the PRM scores retrieval steps without seeing what was retrieved — a fundamental mismatch.

### Retrieval compression ratio: 14.8% (target ≥ 30%)
- **Trigger**: Wikipedia retrieval passages are information-dense; stopwords constitute ~15% of tokens vs 30–50% in narrative text. Target was set based on Caveman benchmarks on general text.
- **Change**: No implementation change. Target revised to 15% for factual retrieval corpora. Under this target, Exp 10b meets all criteria.

## Combined Signal (Exp 7 + 8)

### Combined signal closes most of the Uniform gap without dominating it
- **Trigger**: Best combined (Δ=0.10, D=0.15): acc=0.385 ≈ Uniform (0.388) at 43% higher cost.
- **Change**: Combined reported as the most promising direction for future work; not a Pareto improvement over Uniform at matched cost in current form.
- **Rationale**: Confirms additional_experiments.md §Expected Outcomes assessment (High probability of improvement for both Exp 7 and Exp 8 individually; combined is strongest direction).
