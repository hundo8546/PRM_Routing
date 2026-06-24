"""
Experiment 16: Real Compression Method Comparison.

Compares four compression methods for PRM input context reduction:
  A. None      — baseline, no compression
  B. Stopword  — custom rule-based removal (exp910 implementation)
  C. Caveman   — real LLM-based (JuliusBrussee/caveman via Claude Haiku)
  D. LLMLingua — learned token-importance (Jiang et al. 2023)

For each method, measures:
  - Token reduction ratio (query, retrieval, combined)
  - VersaPRM signal quality: Spearman correlation vs human labels
  - VersaPRM routing precision
  - Cost of compression itself (for Caveman only)

Applied to both query compression and retrieval context compression.

Requirements:
  pip install anthropic llmlingua   (on cluster)
  ANTHROPIC_API_KEY env var         (for Caveman)

Outputs:
  results/exp16/exp16_results.json
  results/exp16/exp16_summary.txt
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from scipy import stats

ROOT     = Path("/workspace/PRM_Routing")
DATA_DIR = ROOT / "benchmarks/AgentProcessBench/data/AgentProcessBench"
DATASETS = ["hotpotqa", "gaia_dev", "bfcl", "tau2"]
RESULTS  = ROOT / "results/exp16"
RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN_PER_DS = 200
TEST_PER_DS  = 50
TOTAL_PER_DS = 250
LLMLINGUA_RATE = 0.5   # keep 50% of tokens

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "exp910_compression"))

from compression_methods import (
    compress_none, compress_stopword,
    compress_caveman_batch, compress_llmlingua_batch,
    compression_ratio, count_tokens,
)

# ---------------------------------------------------------------------------
# Shared data loading (mirrors exp910 load_step_data)
# ---------------------------------------------------------------------------

def load_test_steps() -> list:
    """Load test-set steps with question, retrieval context, step text, human label."""
    score_map = {}
    ds_offset = {ds: i * 250 for i, ds in enumerate(DATASETS)}
    with open(ROOT / "results/exp1/full_versa.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["reward_score"] is not None:
                score_map[(r["dataset"], r["traj_idx"], r["msg_idx"])] = r["reward_score"]

    steps = []
    for ds in DATASETS:
        with open(DATA_DIR / f"{ds}.jsonl") as f:
            for local_idx, line in enumerate(f):
                if local_idx >= TOTAL_PER_DS:
                    break
                traj   = json.loads(line)
                msgs   = traj["messages"]
                q      = traj.get("question", "") or ""

                for idx_str, human_label in sorted(
                    traj.get("step_labels", {}).items(), key=lambda x: int(x[0])
                ):
                    msg_idx = int(idx_str)

                    # Collect retrieval text from preceding tool messages
                    retr_parts = []
                    for j in range(msg_idx):
                        m = msgs[j]
                        if m["role"] == "tool":
                            content = m.get("content", "") or ""
                            try:
                                parsed = json.loads(content)
                                for group in parsed.get("result", []):
                                    if isinstance(group, list):
                                        for item in group:
                                            doc = item.get("document", {}) if isinstance(item, dict) else {}
                                            txt = doc.get("contents", "")
                                            if txt:
                                                retr_parts.append(str(txt))
                            except Exception:
                                if content.strip():
                                    retr_parts.append(str(content)[:500])

                    retrieval = " ".join(retr_parts)[:2000]

                    msg     = msgs[msg_idx]
                    content = msg.get("content", "") or ""
                    tc      = msg.get("tool_calls", "") or ""
                    if isinstance(tc, list):
                        tc = json.dumps(tc)
                    step_text = (content + " " + str(tc)).strip()[:500]

                    global_idx  = ds_offset[ds] + local_idx
                    versa_score = score_map.get((ds, global_idx, msg_idx))

                    if local_idx >= TRAIN_PER_DS and versa_score is not None:
                        steps.append({
                            "dataset":     ds,
                            "traj_idx":    local_idx,
                            "global_idx":  global_idx,
                            "msg_idx":     msg_idx,
                            "question":    q,
                            "retrieval":   retrieval,
                            "step_text":   step_text,
                            "human_label": human_label,
                            "versa_score": versa_score,
                        })

    print(f"Loaded {len(steps)} test steps with VersaPRM scores.")
    return steps


# ---------------------------------------------------------------------------
# VersaPRM scorer
# ---------------------------------------------------------------------------

def load_versaprm_scorer():
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_dir    = str(ROOT / "models/versaprm/base")
    adapter_dir = str(ROOT / "models/versaprm/adapter")
    print("Loading VersaPRM ...")
    tok   = AutoTokenizer.from_pretrained(adapter_dir)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "left"
    base  = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    CAND  = [12, 10]
    SEP   = 23535

    def score(context: str, step: str) -> float:
        inp = context.strip() + " \n\n" + step.strip() + " \n\n\n\n"
        ids = torch.tensor(
            [tok.encode(inp, max_length=1024, truncation=True)]
        ).to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[:, :, CAND]
            scores = logits.softmax(dim=-1)[:, :, 1]
            mask   = (ids == SEP)
        return float(scores[mask].mean().cpu()) if mask.any() else float(scores[0, -1].cpu())

    return score


# ---------------------------------------------------------------------------
# Signal quality metrics
# ---------------------------------------------------------------------------

def signal_quality(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Spearman correlation and routing precision."""
    sp, _ = stats.spearmanr(scores, labels)
    lb    = (labels == 1).astype(int)
    threshold = np.median(scores)
    preds = (scores >= threshold).astype(int)
    tp  = ((preds == 1) & (lb == 0)).sum()
    esc = (preds == 1).sum()
    prec = float(tp / esc) if esc > 0 else 0.0
    return float(sp), prec


# ---------------------------------------------------------------------------
# Score all steps under one context variant
# ---------------------------------------------------------------------------

def score_variant(steps: list, scorer, contexts: list[str]) -> np.ndarray:
    """Score each step given its context string."""
    assert len(steps) == len(contexts)
    scores = []
    for i, (s, ctx) in enumerate(zip(steps, contexts)):
        scores.append(scorer(ctx, s["step_text"]))
        if (i + 1) % 100 == 0:
            print(f"    scored {i+1}/{len(steps)} ...")
    return np.array(scores)


# ---------------------------------------------------------------------------
# Compression ratio stats
# ---------------------------------------------------------------------------

def ratio_stats(originals: list[str], compressed: list[str]) -> dict:
    ratios = [compression_ratio(o, c) for o, c in zip(originals, compressed) if o.strip()]
    orig_tok = [count_tokens(t) for t in originals]
    comp_tok = [count_tokens(t) for t in compressed]
    return {
        "mean_reduction":  float(np.mean(ratios)),
        "orig_tokens_mean": float(np.mean(orig_tok)),
        "comp_tokens_mean": float(np.mean(comp_tok)),
        "n": len(ratios),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    run_caveman:   bool = True,
    run_llmlingua: bool = True,
    llmlingua_device: str = "auto",
):
    print("=" * 70)
    print("Experiment 16: Real Compression Method Comparison")
    print("=" * 70)

    # ── 1. Load data ────────────────────────────────────────────────────────
    steps = load_test_steps()
    n     = len(steps)
    questions  = [s["question"]  for s in steps]
    retrievals = [s["retrieval"] for s in steps]
    labels     = np.array([s["human_label"] for s in steps])
    baseline_scores = np.array([s["versa_score"] for s in steps])

    print(f"\nSteps: {n} | With retrieval: {sum(1 for r in retrievals if r.strip())}")

    # ── 2. Compress queries and retrievals under all methods ─────────────────
    print("\n[A] No compression (passthrough)")
    q_none  = [compress_none(q) for q in questions]
    r_none  = [compress_none(r) for r in retrievals]

    print("\n[B] Stopword removal (exp910 custom)")
    q_stop  = [compress_stopword(q) for q in questions]
    r_stop  = [compress_stopword(r) for r in retrievals]

    caveman_cost_usd = 0.0
    q_cave, r_cave = q_none[:], r_none[:]   # fallback = passthrough
    if run_caveman:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("\n[C] Caveman: ANTHROPIC_API_KEY not set — SKIPPING")
            run_caveman = False
        else:
            print("\n[C] Caveman LLM compression")
            q_cave, q_cost = compress_caveman_batch(questions)
            r_cave, r_cost = compress_caveman_batch(retrievals)
            caveman_cost_usd = q_cost + r_cost
            print(f"  Total caveman cost: ${caveman_cost_usd:.4f}")

    q_llml, r_llml = q_none[:], r_none[:]   # fallback = passthrough
    if run_llmlingua:
        print(f"\n[D] LLMLingua (rate={LLMLINGUA_RATE})")
        try:
            q_llml = compress_llmlingua_batch(questions,  rate=LLMLINGUA_RATE, device=llmlingua_device)
            r_llml = compress_llmlingua_batch(retrievals, rate=LLMLINGUA_RATE, device=llmlingua_device)
        except ImportError as e:
            print(f"  SKIPPING: {e}")
            run_llmlingua = False

    # ── 3. Load VersaPRM and score ───────────────────────────────────────────
    print("\nLoading VersaPRM ...")
    scorer = load_versaprm_scorer()

    def build_contexts(q_list, r_list):
        """Combine question + retrieval + step as VersaPRM context."""
        ctxs = []
        for q, r in zip(q_list, r_list):
            ctx = q.strip()
            if r.strip():
                ctx += " " + r.strip()
            ctxs.append(ctx)
        return ctxs

    results = {}

    variants = [
        ("A_none",     q_none,  r_none,  True),
        ("B_stopword", q_stop,  r_stop,  True),
        ("C_caveman",  q_cave,  r_cave,  run_caveman),
        ("D_llmlingua",q_llml,  r_llml,  run_llmlingua),
    ]

    for name, q_list, r_list, enabled in variants:
        if not enabled:
            results[name] = {"skipped": True}
            continue

        print(f"\nScoring variant {name} ...")
        contexts = build_contexts(q_list, r_list)
        scores   = score_variant(steps, scorer, contexts)
        sp, prec = signal_quality(scores, labels)

        q_ratio_stats = ratio_stats(questions,  q_list)
        r_ratio_stats = ratio_stats(retrievals, r_list)

        results[name] = {
            "spearman":           round(sp,   4),
            "routing_precision":  round(prec, 4),
            "query_compression":  q_ratio_stats,
            "retrieval_compression": r_ratio_stats,
            "n_steps":            n,
        }
        if name == "C_caveman":
            results[name]["caveman_cost_usd"] = round(caveman_cost_usd, 4)

        print(f"  Spearman={sp:+.4f}  Precision={prec:.4f}  "
              f"Q_reduction={q_ratio_stats['mean_reduction']:.1%}  "
              f"R_reduction={r_ratio_stats['mean_reduction']:.1%}")

    # ── 4. Print summary table ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    header = f"{'Method':<16} {'Spearman':>10} {'RoutPrec':>10} {'Q_Reduc':>9} {'R_Reduc':>9} {'Cost':>8}"
    print(header)
    print("-" * len(header))

    method_labels = {
        "A_none":      "A. None",
        "B_stopword":  "B. Stopword",
        "C_caveman":   "C. Caveman",
        "D_llmlingua": "D. LLMLingua",
    }
    for key, label in method_labels.items():
        r = results.get(key, {})
        if r.get("skipped"):
            print(f"{label:<16} {'SKIPPED':>10}")
            continue
        cost_str = f"${r.get('caveman_cost_usd', 0):.2f}" if key == "C_caveman" else "free"
        print(
            f"{label:<16} "
            f"{r['spearman']:>+10.4f} "
            f"{r['routing_precision']:>10.4f} "
            f"{r['query_compression']['mean_reduction']:>9.1%} "
            f"{r['retrieval_compression']['mean_reduction']:>9.1%} "
            f"{cost_str:>8}"
        )

    # Baseline (Exp 1 scores, no compression context)
    sp_base, prec_base = signal_quality(baseline_scores, labels)
    print(f"\n  Exp1 baseline (q+step only, no retrieval): "
          f"Spearman={sp_base:+.4f}  Precision={prec_base:.4f}")

    # ── 5. Save ──────────────────────────────────────────────────────────────
    output = {
        "config": {
            "n_steps":         n,
            "llmlingua_rate":  LLMLINGUA_RATE,
            "caveman_model":   "claude-haiku-4-5-20251001",
            "ran_caveman":     run_caveman,
            "ran_llmlingua":   run_llmlingua,
        },
        "baseline_exp1": {
            "spearman":          round(sp_base,   4),
            "routing_precision": round(prec_base, 4),
            "note":              "Exp1 VersaPRM scores, q+step context only",
        },
        "results": results,
    }
    out_path = RESULTS / "exp16_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Human-readable summary
    summary_lines = [
        "Experiment 16: Real Compression Method Comparison",
        f"N test steps: {n}  |  LLMLingua rate: {LLMLINGUA_RATE}",
        "",
        header,
        "-" * len(header),
    ]
    for key, label in method_labels.items():
        r = results.get(key, {})
        if r.get("skipped"):
            summary_lines.append(f"{label:<16} SKIPPED")
            continue
        cost_str = f"${r.get('caveman_cost_usd', 0):.2f}" if key == "C_caveman" else "free"
        summary_lines.append(
            f"{label:<16} "
            f"{r['spearman']:>+10.4f} "
            f"{r['routing_precision']:>10.4f} "
            f"{r['query_compression']['mean_reduction']:>9.1%} "
            f"{r['retrieval_compression']['mean_reduction']:>9.1%} "
            f"{cost_str:>8}"
        )
    with open(RESULTS / "exp16_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nResults saved to {RESULTS}/")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-caveman",   action="store_true",
                        help="Skip caveman (LLM-based) compression")
    parser.add_argument("--no-llmlingua", action="store_true",
                        help="Skip LLMLingua compression")
    parser.add_argument("--llmlingua-device", default="auto",
                        help="Device for LLMLingua model (auto/cuda/cpu)")
    args = parser.parse_args()

    run(
        run_caveman=not args.no_caveman,
        run_llmlingua=not args.no_llmlingua,
        llmlingua_device=args.llmlingua_device,
    )
