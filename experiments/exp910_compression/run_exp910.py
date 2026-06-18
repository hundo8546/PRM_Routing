"""
Experiments 9 & 10: Input Compression for PRM-Guided Routing.

Exp 9: Compress the user *query* before passing to VersaPRM.
Exp 10: Include *retrieval context* (tool results) in VersaPRM input; compress it.

Why these work where Exp 5 failed:
  Exp 5 compressed *step outputs* (tool calls) — JSON, no compression possible.
  Exps 9/10 compress *natural-language inputs*: queries (~114 tok) and
  retrieved passages (~506 tok/call), both highly compressible.

VersaPRM input variants tested:
  A. Baseline:   question + step_text                    (current Exp 1)
  B. Exp 9:      compressed_query + step_text
  C. Exp 10a:    question + full_retrieval + step_text   (enriched baseline)
  D. Exp 10b:    question + compressed_retrieval + step_text

Success criteria:
  Exp 9:  token reduction ≥ 25%, accuracy loss ≤ 2pp, precision loss ≤ 5pp
  Exp 10: context reduction ≥ 30%, accuracy loss ≤ 3pp, no sig. precision drop

Outputs:
  results/exp9/exp9_results.json
  results/exp10/exp10_results.json
"""

import json
import re
import sys
import copy
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

ROOT = Path("/workspace/PRM_Routing")
DATA_DIR = ROOT / "benchmarks/AgentProcessBench/data/AgentProcessBench"
DATASETS = ["hotpotqa", "gaia_dev", "bfcl", "tau2"]

sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_routing"))

from data_loader import load_trajectories
from routing_policies import PRMGuided
from simulator import evaluate_policy

RESULTS = {9: ROOT / "results/exp9", 10: ROOT / "results/exp10"}

# Test set: last 50/dataset (same as Exp 2/3)
TRAIN_PER_DS, TEST_PER_DS, TOTAL_PER_DS = 200, 50, 250
N_SCORE_SAMPLE = 400   # steps to re-score with VersaPRM

# ---------------------------------------------------------------------------
# Caveman compression (improved: preserves numbers, entities, key verbs)
# ---------------------------------------------------------------------------

STOPWORDS = frozenset({
    "a","an","the","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","must","can","need","in","on","at","by","for","with","from","to",
    "of","that","this","it","its","i","we","you","he","she","they","them",
    "their","our","my","your","his","her","which","who","what","as","if","so",
    "but","and","or","not","no","also","about","after","all","already","just",
    "more","than","then","there","these","those","up","out","into","onto",
    "upon","very","much","many","such","some","any","each","every","both",
    "between","through","during","before","after","above","below","while",
})

def caveman_compress(text: str, preserve_structure: bool = False) -> str:
    """
    Caveman compression for natural-language text.
    - Drops stopwords and filler words
    - Preserves numbers, proper nouns (Title Case), dates, percentages
    - Preserves punctuation that carries meaning (. ? !)
    - If preserve_structure: keeps sentence-ending periods and newlines
    """
    if not text or not text.strip():
        return text

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    compressed_sents = []
    for sent in sentences:
        tokens = re.findall(r'\b[\w\']+\b|\d+[%$]?|[.!?]', sent)
        kept = []
        for tok in tokens:
            if tok in '.!?':
                if kept:
                    kept[-1] = kept[-1].rstrip()
                    if preserve_structure:
                        kept.append('.')
                continue
            if tok.lower() in STOPWORDS and not tok[0].isupper() and not tok.isdigit():
                continue
            kept.append(tok)
        if kept:
            compressed_sents.append(' '.join(kept))

    sep = '. ' if preserve_structure else '. '
    return sep.join(compressed_sents)

# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _extract_retrieval_text(tool_content: str) -> str:
    """Extract natural-language passage text from tool result JSON."""
    if not tool_content:
        return ""
    try:
        parsed = json.loads(tool_content)
        passages = []
        # AgentProcessBench format: {"result": [[{"document": {"contents": "..."}}]]}
        result = parsed.get("result", [])
        for group in result:
            if isinstance(group, list):
                for item in group:
                    if isinstance(item, dict):
                        doc = item.get("document", {})
                        contents = doc.get("contents", "")
                        if contents:
                            passages.append(str(contents))
        return " ".join(passages)
    except Exception:
        # Fallback: return as-is if not parseable
        return str(tool_content)[:1000]


def load_step_data(n_per_dataset: int = TOTAL_PER_DS) -> list:
    """
    Load labeled steps with their full context:
      - question (original + compressed)
      - retrieval_context_full (all preceding tool results, joined)
      - retrieval_context_compressed
      - step_text
      - human_label
      - versa_score (from Exp 1)
    """
    # Load Exp 1 score map
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
                if local_idx >= n_per_dataset:
                    break
                traj = json.loads(line)
                msgs = traj["messages"]
                q = traj.get("question", "") or ""
                q_compressed = caveman_compress(q, preserve_structure=True)

                for idx_str, human_label in sorted(traj.get("step_labels", {}).items(),
                                                    key=lambda x: int(x[0])):
                    msg_idx = int(idx_str)
                    # Collect retrieval content from preceding tool messages
                    retrieval_parts_full = []
                    retrieval_parts_comp = []
                    for j in range(msg_idx):
                        m = msgs[j]
                        if m["role"] == "tool":
                            text = _extract_retrieval_text(m.get("content", "") or "")
                            if text.strip():
                                retrieval_parts_full.append(text)
                                retrieval_parts_comp.append(
                                    caveman_compress(text, preserve_structure=False)
                                )

                    retr_full = " ".join(retrieval_parts_full)[:2000]
                    retr_comp = " ".join(retrieval_parts_comp)[:2000]

                    # Step text (the agent's action/output)
                    msg = msgs[msg_idx]
                    content = msg.get("content", "") or ""
                    tc = msg.get("tool_calls", "") or ""
                    if isinstance(tc, list):
                        tc = json.dumps(tc)
                    step_text = (content + " " + str(tc)).strip()[:500]

                    global_idx = ds_offset[ds] + local_idx
                    versa_score = score_map.get((ds, global_idx, msg_idx), None)

                    steps.append({
                        "dataset": ds,
                        "traj_idx": local_idx,
                        "global_idx": global_idx,
                        "msg_idx": msg_idx,
                        "question_orig": q,
                        "question_comp": q_compressed,
                        "retrieval_full": retr_full,
                        "retrieval_comp": retr_comp,
                        "step_text": step_text,
                        "human_label": human_label,
                        "versa_score_exp1": versa_score,
                        "is_test": local_idx >= TRAIN_PER_DS,
                    })

    test_steps = [s for s in steps if s["is_test"] and s["versa_score_exp1"] is not None]
    print(f"Loaded {len(steps)} labeled steps total, {len(test_steps)} in test set with Exp1 scores")
    return test_steps


# ---------------------------------------------------------------------------
# Compression ratio analysis
# ---------------------------------------------------------------------------

def measure_compression_ratios(steps: list) -> dict:
    """Measure token savings for query and retrieval context compression."""
    q_orig_tok, q_comp_tok = [], []
    r_orig_tok, r_comp_tok = [], []
    n_with_retrieval = 0

    for s in steps:
        q_orig_tok.append(max(1, len(s["question_orig"]) // 4))
        q_comp_tok.append(max(1, len(s["question_comp"]) // 4))
        if s["retrieval_full"].strip():
            r_orig_tok.append(max(1, len(s["retrieval_full"]) // 4))
            r_comp_tok.append(max(1, len(s["retrieval_comp"]) // 4))
            n_with_retrieval += 1

    qo = np.array(q_orig_tok); qc = np.array(q_comp_tok)
    ro = np.array(r_orig_tok); rc = np.array(r_comp_tok)

    q_ratio = float((1 - qc / qo).mean())
    r_ratio = float((1 - rc / ro).mean()) if len(ro) > 0 else 0.0

    print(f"\n[Compression ratios]")
    print(f"  Query:    orig={qo.mean():.0f} tok → comp={qc.mean():.0f} tok  "
          f"reduction={q_ratio:.1%}  (n={len(qo)})")
    print(f"  Retrieval: orig={ro.mean():.0f} tok → comp={rc.mean():.0f} tok  "
          f"reduction={r_ratio:.1%}  (n={n_with_retrieval} steps with retrieval)")

    # Example
    ex = next((s for s in steps if s["question_orig"] and s["retrieval_full"]), None)
    if ex:
        print(f"\n  Query example ({len(ex['question_orig'])} → {len(ex['question_comp'])} chars):")
        print(f"    Orig: {ex['question_orig'][:120]}")
        print(f"    Comp: {ex['question_comp'][:120]}")
        print(f"\n  Retrieval example ({len(ex['retrieval_full'])} → {len(ex['retrieval_comp'])} chars):")
        print(f"    Orig: {ex['retrieval_full'][:120]}")
        print(f"    Comp: {ex['retrieval_comp'][:120]}")

    return {
        "query_orig_tokens": float(qo.mean()),
        "query_comp_tokens": float(qc.mean()),
        "query_compression_ratio": q_ratio,
        "retrieval_orig_tokens": float(ro.mean()) if len(ro) else 0,
        "retrieval_comp_tokens": float(rc.mean()) if len(rc) else 0,
        "retrieval_compression_ratio": r_ratio,
        "n_steps_with_retrieval": n_with_retrieval,
    }

# ---------------------------------------------------------------------------
# VersaPRM scorer (four variants)
# ---------------------------------------------------------------------------

def make_versaprm_scorer():
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_dir   = str(ROOT / "models/versaprm/base")
    adapter_dir = str(ROOT / "models/versaprm/adapter")
    print("\nLoading VersaPRM ...")
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    tok.pad_token = tok.eos_token; tok.padding_side = "left"
    base  = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    CAND  = [12, 10]
    SEP   = 23535

    def score(context_text: str, step_text: str) -> float:
        """Score a step given arbitrary context_text (prepended to step)."""
        inp = context_text.strip() + " \n\n" + step_text.strip() + " \n\n\n\n"
        ids = torch.tensor([tok.encode(inp, max_length=1024, truncation=True)]).to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[:, :, CAND]
            scores = logits.softmax(dim=-1)[:, :, 1]
            mask   = (ids == SEP)
        return float(scores[mask].mean().cpu()) if mask.any() else float(scores[0, -1].cpu())

    return score, tok, model


def run_scoring(steps, scorer, n_sample: int = N_SCORE_SAMPLE):
    """Score n_sample test steps under four context variants."""
    sample = steps[:n_sample]
    results = {
        "versa_exp1": [],   # A: original Exp1 score (question + step, no full context)
        "query_comp": [],   # B: compressed query + step
        "retr_full":  [],   # C: question + full retrieval + step
        "retr_comp":  [],   # D: question + compressed retrieval + step
        "labels":     [],
    }

    for i, s in enumerate(sample):
        q  = s["question_orig"]
        qc = s["question_comp"]
        rf = s["retrieval_full"]
        rc = s["retrieval_comp"]
        st = s["step_text"]

        results["versa_exp1"].append(s["versa_score_exp1"])
        results["query_comp"].append(scorer(qc, st))

        if rf.strip():
            results["retr_full"].append(scorer(q + " " + rf, st))
            results["retr_comp"].append(scorer(q + " " + rc, st))
        else:
            results["retr_full"].append(scorer(q, st))
            results["retr_comp"].append(scorer(q, st))

        results["labels"].append(s["human_label"])

        if (i + 1) % 50 == 0:
            print(f"  Scored {i+1}/{n_sample} steps ...")

    return {k: np.array(v) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Signal quality analysis
# ---------------------------------------------------------------------------

def signal_quality(scores, labels):
    sp, _ = stats.spearmanr(scores, labels)
    lb = (labels == 1).astype(int)
    threshold = np.median(scores)
    preds = (scores >= threshold).astype(int)
    tp = ((preds == 1) & (lb == 0)).sum()   # escalated and actually bad
    esc = (preds == 1).sum()
    prec = tp / esc if esc > 0 else 0.0
    return float(sp), float(prec)


# ---------------------------------------------------------------------------
# Routing simulation from re-scored steps
# ---------------------------------------------------------------------------

def simulate_routing_from_scores(test_trajs, new_scores_map, theta_high=0.86, theta_low=0.62):
    """Inject new scores into trajectories and evaluate PRMGuided routing."""
    import copy
    ds_offset = {ds: i * 250 for i, ds in enumerate(DATASETS)}
    aug = []
    for traj in test_trajs:
        t2 = copy.deepcopy(traj)
        for step in t2.steps:
            g = ds_offset[step.dataset] + step.traj_idx
            key = (step.dataset, g, step.msg_idx)
            if key in new_scores_map:
                step.versa_score = new_scores_map[key]
        aug.append(t2)
    return evaluate_policy(PRMGuided(theta_high, theta_low), aug)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 70)
    print("Experiments 9 & 10: Input Compression for PRM-Guided Routing")
    print("=" * 70)

    # Load all steps with context
    all_steps = load_step_data(TOTAL_PER_DS)
    test_steps = [s for s in all_steps if s["is_test"]]

    # Compression ratios
    comp_stats = measure_compression_ratios(test_steps)

    # Load VersaPRM and score all four variants
    scorer, tok, model = make_versaprm_scorer()
    print(f"\nScoring {N_SCORE_SAMPLE} test steps under 4 context variants ...")
    scored = run_scoring(test_steps, scorer, N_SCORE_SAMPLE)

    import torch
    del model; torch.cuda.empty_cache(); import gc; gc.collect()

    # Signal quality for each variant
    lbs = scored["labels"]
    variants = {
        "A. Baseline (q+step)":           scored["versa_exp1"],
        "B. Exp9  (q_comp+step)":          scored["query_comp"],
        "C. Exp10a (q+retr_full+step)":    scored["retr_full"],
        "D. Exp10b (q+retr_comp+step)":    scored["retr_comp"],
    }

    print(f"\n{'Variant':<36} {'Spearman':>10} {'RoutPrec':>10} {'vs Baseline':>12}")
    print("-" * 70)
    baseline_sp, baseline_prec = signal_quality(scored["versa_exp1"], lbs)
    print(f"{'A. Baseline (q+step)':<36} {baseline_sp:>+10.4f} {baseline_prec:>10.3f} {'—':>12}")
    for name, sc in list(variants.items())[1:]:
        sp, prec = signal_quality(sc, lbs)
        dsp = sp - baseline_sp; dprec = prec - baseline_prec
        print(f"{name:<36} {sp:>+10.4f} {prec:>10.3f} "
              f"  Δsp={dsp:+.4f} Δprec={dprec:+.4f}")

    # Correlation between variants and baseline
    print(f"\n{'Variant':<36} {'Pearson(var,A)':>15}")
    for name, sc in list(variants.items())[1:]:
        r, _ = stats.pearsonr(sc, scored["versa_exp1"])
        print(f"{name:<36} {r:>+15.4f}")

    # Load test trajectories for routing simulation
    all_trajs = load_trajectories("versa", TOTAL_PER_DS)
    ds_map = defaultdict(list)
    for t in all_trajs: ds_map[t.dataset].append(t)
    train_trajs, test_trajs = [], []
    for ts in ds_map.values():
        train_trajs.extend(ts[:TRAIN_PER_DS]); test_trajs.extend(ts[TRAIN_PER_DS:])

    # Build score maps for routing simulation
    ds_offset = {ds: i * 250 for i, ds in enumerate(DATASETS)}
    baseline_map = {(s["dataset"], s["global_idx"], s["msg_idx"]): float(s["versa_score_exp1"])
                    for s in test_steps[:N_SCORE_SAMPLE]}
    q_comp_map   = {(test_steps[i]["dataset"], test_steps[i]["global_idx"], test_steps[i]["msg_idx"]): float(sc)
                    for i, sc in enumerate(scored["query_comp"])}
    r_full_map   = {(test_steps[i]["dataset"], test_steps[i]["global_idx"], test_steps[i]["msg_idx"]): float(sc)
                    for i, sc in enumerate(scored["retr_full"])}
    r_comp_map   = {(test_steps[i]["dataset"], test_steps[i]["global_idx"], test_steps[i]["msg_idx"]): float(sc)
                    for i, sc in enumerate(scored["retr_comp"])}

    print(f"\n{'Routing results (PRMGuided h=0.86/l=0.62)'}")
    print(f"{'Variant':<36} {'Acc':>7} {'CostN':>9} {'EscRate':>9} {'ΔAcc':>8}")
    r_base = simulate_routing_from_scores(test_trajs, baseline_map)
    print(f"{'A. Baseline':<36} {r_base.mean_accuracy:>7.4f} "
          f"{r_base.mean_cost_norm_per_traj:>9.0f} {r_base.escalation_rate:>9.3f}  {'—':>6}")

    for name, score_map in [
        ("B. Exp9 (q_comp+step)", q_comp_map),
        ("C. Exp10a (q+retr_full+step)", r_full_map),
        ("D. Exp10b (q+retr_comp+step)", r_comp_map),
    ]:
        r = simulate_routing_from_scores(test_trajs, score_map)
        da = r.mean_accuracy - r_base.mean_accuracy
        print(f"{name:<36} {r.mean_accuracy:>7.4f} "
              f"{r.mean_cost_norm_per_traj:>9.0f} {r.escalation_rate:>9.3f} {da:>+8.4f}")

    # Criterion checks
    print("\n[Exp 9 criterion] query_reduction ≥ 25%, Δacc ≤ 2pp, Δprec ≤ 5pp")
    sp9, prec9 = signal_quality(scored["query_comp"], lbs)
    q_crit = (
        comp_stats["query_compression_ratio"] >= 0.25,
        abs(sp9 - baseline_sp) <= 0.05,
        abs(prec9 - baseline_prec) <= 0.05,
    )
    print(f"  query_reduction: {comp_stats['query_compression_ratio']:.1%}  {'✓' if q_crit[0] else '✗'}")
    print(f"  Δspearman:       {sp9-baseline_sp:+.4f}  {'✓' if q_crit[1] else '✗'}")
    print(f"  Δprec:           {prec9-baseline_prec:+.4f}  {'✓' if q_crit[2] else '✗'}")
    exp9_met = all(q_crit)
    print(f"  MET: {'YES ✓' if exp9_met else 'NO'}")

    print("\n[Exp 10 criterion] retr_reduction ≥ 30%, Δacc ≤ 3pp, prec not sig. worse")
    sp10d, prec10d = signal_quality(scored["retr_comp"], lbs)
    r_crit = (
        comp_stats["retrieval_compression_ratio"] >= 0.30,
        abs(sp10d - baseline_sp) <= 0.05,
        abs(prec10d - baseline_prec) <= 0.05,
    )
    print(f"  retr_reduction: {comp_stats['retrieval_compression_ratio']:.1%}  {'✓' if r_crit[0] else '✗'}")
    print(f"  Δspearman:      {sp10d-baseline_sp:+.4f}  {'✓' if r_crit[1] else '✗'}")
    print(f"  Δprec:          {prec10d-baseline_prec:+.4f}  {'✓' if r_crit[2] else '✗'}")
    exp10_met = all(r_crit)
    print(f"  MET: {'YES ✓' if exp10_met else 'NO'}")

    # Save
    result = {
        "compression_stats": comp_stats,
        "signal_quality": {
            "baseline_spearman": round(baseline_sp, 4),
            "baseline_prec": round(baseline_prec, 4),
            "query_comp_spearman": round(sp9, 4),
            "query_comp_prec": round(prec9, 4),
            "retr_full_spearman": round(signal_quality(scored["retr_full"], lbs)[0], 4),
            "retr_comp_spearman": round(sp10d, 4),
            "retr_comp_prec": round(prec10d, 4),
        },
        "exp9_criterion_met": exp9_met,
        "exp10_criterion_met": exp10_met,
        "n_scored": N_SCORE_SAMPLE,
    }
    for exp_num, path in RESULTS.items():
        with open(path / f"exp{exp_num}_results.json", "w") as f:
            json.dump(result, f, indent=2)
    print(f"\nResults saved to results/exp9/ and results/exp10/")
    return result


if __name__ == "__main__":
    run()
