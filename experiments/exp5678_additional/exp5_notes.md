# Experiment 5: Caveman Compression

**Hypothesis:** Compressing step outputs before VersaPRM scoring reduces tokens (≥30%) without degrading routing signal (accuracy loss ≤2pp).

**Status:** Complete. **Criterion not met.**

---

## Results

| Metric | Result | Target |
|---|---|---|
| Mean compression ratio | **1.4%** | ≥ 30% |
| Spearman(full_score, labels) | +0.396 | — (reference) |
| Spearman(compressed_score, labels) | **−0.116** | ≤ reference |
| Δ routing accuracy | −0.003 | ≤ −0.02 |
| Criterion met | **NO** | YES |

**Token counts:** Mean original = 96.9 tokens/step, compressed = 91.0 tokens/step. Only 5.9 tokens saved per step.

---

## Why Compression Fails Here

**Root cause: step content is dominated by structured JSON, not natural language.**

The AgentProcessBench labeled steps are tool calls:
```json
{"id": "chatcmpl-tool-...", "type": "function",
 "function": {"name": "search", "arguments": "{\"query_list\": [...]}"}}
```
The Caveman compressor correctly preserves JSON structure intact. Since ~72% of labeled steps are tool calls or retrieval actions with JSON arguments, there is almost no natural language to strip. The 1.4% token reduction comes from synthesis steps only (the remaining ~28%).

**Signal degradation:**
Even on synthesis steps where compression does remove words, the compressed text degrades VersaPRM's score quality:
- Spearman with labels drops from +0.396 → −0.116
- VersaPRM was trained on natural language — removing stopwords and structure disrupts its scoring mechanism even when factual content is nominally preserved

**Routing accuracy is misleading here:** The −0.003 routing accuracy change seems small, but it is because the routing policy happens to behave similarly despite different scores — the score correlation is only Pearson=0.267 (low), meaning the routing decisions have changed significantly but by chance produce similar aggregate accuracy.

---

## Key Finding for the Paper

Caveman compression is not applicable to tool-using agent pipelines. The step outputs that matter for routing signal (tool calls, structured API responses) are already compact and structured — they cannot be compressed by removing natural-language stopwords without destroying VersaPRM's scoring mechanism.

**Implication:** To reduce VersaPRM inference cost, better directions are:
- Score only synthesis steps (skip tool-call scoring)
- Use a smaller/faster PRM variant
- Batch scoring across trajectories

---

## Decisions made based on Exp 5

| Decision | Trigger | See |
|---|---|---|
| Exp 5 excluded from main paper claims | Criterion not met; compression destroys signal for agent-step content | DECISIONS.md §Exp 5 |

---

## Files

- `run_exp5.py` — full compression + re-scoring experiment
- `../../results/exp5/exp5_results.json` — results
