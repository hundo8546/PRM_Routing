"""
Experiment 12: Modern Model Stack Evaluation.

Research question: Does PRM-guided routing continue to provide value when all
tiers are upgraded to modern 2026 models?

Compares the original stack (Exp 2) against two modern stacks:
  - Recommended:  T1=Qwen3-8B, T2=GPT-4o, T3=Claude-Opus-4
  - Open-weight:  T1=Gemma4-E4B, T2=Qwen3-8B, T3=Qwen3-30B-A3B

For each stack, runs the same Exp-2 policy comparison:
  Always-Cheap / Uniform / Always-Frontier / TRIM / PRMGuided / Multi-Judge

Tier calibration (live inference):
  If OPENAI_API_KEY is set:    calibrates GPT tier (T2 in Modern Recommended)
  If ANTHROPIC_API_KEY is set: calibrates Claude tier (T3 in Modern Recommended)
  Measures empirical p_good_step_success and p_bad_step_recovery on a sample
  of test steps, replacing the assumed placeholder values.
  Falls back to simulated values for any tier whose API key is absent.

Outputs:
  results/exp12/exp12_results.json
  results/exp12/exp12_summary.txt
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from dataclasses import replace
from collections import defaultdict

ROOT = Path("/workspace/PRM_Routing")
sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_routing"))

from data_loader import load_trajectories
from routing_policies import (
    UniformRouting, AlwaysFrontier, AlwaysCheap,
    TRIMStyle, PRMGuided,
)
from simulator import evaluate_policy, SUCCESS_THRESHOLD
from cost_model import TIERS, MODERN_TIERS, OPEN_WEIGHT_TIERS, Tier

RESULTS_DIR = ROOT / "results/exp12"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PER_DS = 200
TEST_PER_DS  = 50
TOTAL_PER_DS = 250

# Calibration sample size (steps per label class)
CALIBRATION_N = 50

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Tier calibration via live inference
# ---------------------------------------------------------------------------

CALIBRATION_PROMPT = (
    "You are evaluating whether an AI agent step is correct and helpful.\n\n"
    "Task: {question}\n"
    "Step type: {step_type}\n"
    "Step content: {step_content}\n\n"
    "Is this step correct and helpful for solving the task? "
    "Reply with only 'yes' or 'no'."
)


def _calibrate_openai(sample_steps: list, model: str = "gpt-4o") -> dict:
    """
    Call OpenAI to judge each step. Returns empirical p_good and p_bad.
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    good_correct = bad_correct = good_total = bad_total = 0
    total_cost = 0.0

    for step in sample_steps:
        prompt = CALIBRATION_PROMPT.format(
            question=step.question,
            step_type=step.step_type,
            step_content=step.step_content or "(no content)",
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1024,
            )
            answer = resp.choices[0].message.content.strip().lower()
            correct = answer.startswith("yes")
            if resp.usage:
                total_cost += (resp.usage.prompt_tokens * 3.75 +
                               resp.usage.completion_tokens * 15.0) / 1_000_000
            time.sleep(0.05)
        except Exception as e:
            print(f"    [OpenAI error] {e} — skipping step")
            continue

        if step.human_label == 1:
            good_total += 1
            good_correct += int(correct)
        else:
            bad_total += 1
            bad_correct += int(correct)

    p_good = good_correct / good_total if good_total else None
    p_bad  = bad_correct  / bad_total  if bad_total  else None
    print(f"    OpenAI ({model}): p_good={p_good:.3f} ({good_correct}/{good_total}), "
          f"p_bad={p_bad:.3f} ({bad_correct}/{bad_total}), cost=${total_cost:.4f}")
    return {"p_good": p_good, "p_bad": p_bad, "cost_usd": total_cost,
            "model": model, "n_good": good_total, "n_bad": bad_total}


def _calibrate_anthropic(sample_steps: list, model: str = "claude-opus-4-8") -> dict:
    """
    Call Anthropic to judge each step. Returns empirical p_good and p_bad.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    good_correct = bad_correct = good_total = bad_total = 0
    total_cost = 0.0
    # claude-opus-4-8 pricing: $15/1M input, $75/1M output
    INPUT_COST  = 15.0
    OUTPUT_COST = 75.0

    for step in sample_steps:
        prompt = CALIBRATION_PROMPT.format(
            question=step.question,
            step_type=step.step_type,
            step_content=step.step_content or "(no content)",
        )
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            if not resp.content:
                continue
            answer = resp.content[0].text.strip().lower()
            correct = answer.startswith("yes")
            total_cost += (resp.usage.input_tokens * INPUT_COST +
                           resp.usage.output_tokens * OUTPUT_COST) / 1_000_000
            time.sleep(0.05)
        except Exception as e:
            print(f"    [Anthropic error] {e} — skipping step")
            continue

        if step.human_label == 1:
            good_total += 1
            good_correct += int(correct)
        else:
            bad_total += 1
            bad_correct += int(correct)

    p_good = good_correct / good_total if good_total else None
    p_bad  = bad_correct  / bad_total  if bad_total  else None
    print(f"    Anthropic ({model}): p_good={p_good:.3f} ({good_correct}/{good_total}), "
          f"p_bad={p_bad:.3f} ({bad_correct}/{bad_total}), cost=${total_cost:.4f}")
    return {"p_good": p_good, "p_bad": p_bad, "cost_usd": total_cost,
            "model": model, "n_good": good_total, "n_bad": bad_total}


def calibrate_modern_tiers(test_trajs: list) -> tuple[dict, dict]:
    """
    Returns (calibrated_modern_tiers, calibration_log).

    Calibrates T2 (GPT tier) if OPENAI_API_KEY is set.
    Calibrates T3 (Claude tier) if ANTHROPIC_API_KEY is set.
    Falls back to simulated values for uncalibrated tiers.
    """
    # Sample steps from test trajectories
    all_steps = [s for t in test_trajs for s in t.steps]
    rng = np.random.default_rng(42)
    good_steps = [s for s in all_steps if s.human_label == 1]
    bad_steps  = [s for s in all_steps if s.human_label == -1]

    n = min(CALIBRATION_N, len(good_steps), len(bad_steps))
    idx_g = rng.choice(len(good_steps), n, replace=False)
    idx_b = rng.choice(len(bad_steps),  n, replace=False)
    sample = [good_steps[i] for i in idx_g] + [bad_steps[i] for i in idx_b]

    cal_log = {}
    tiers = dict(MODERN_TIERS)   # mutable copy

    if OPENAI_API_KEY:
        print(f"  Calibrating T2 (GPT tier) via OpenAI API ({n} good + {n} bad steps) ...")
        result = _calibrate_openai(sample, model="gpt-5.5")
        cal_log["t2_gpt"] = result
        if result["p_good"] is not None and result["p_bad"] is not None:
            tiers[2] = replace(
                tiers[2],
                p_good_step_success=result["p_good"],
                p_bad_step_recovery=result["p_bad"],
            )
    else:
        print("  T2 (GPT tier): no OPENAI_API_KEY — using simulated values")

    if ANTHROPIC_API_KEY:
        try:
            import anthropic
            print(f"  Calibrating T3 (Claude tier) via Anthropic API ({n} good + {n} bad steps) ...")
            result = _calibrate_anthropic(sample, model="claude-opus-4-8")
            cal_log["t3_claude"] = result
            if result["p_good"] is not None and result["p_bad"] is not None:
                tiers[3] = replace(
                    tiers[3],
                    p_good_step_success=result["p_good"],
                    p_bad_step_recovery=result["p_bad"],
                )
        except ImportError:
            print("  T3 (Claude tier): anthropic package not installed — pip install anthropic")
    else:
        print("  T3 (Claude tier): no ANTHROPIC_API_KEY — using simulated values")

    return tiers, cal_log


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def split(trajs, train_n=TRAIN_PER_DS):
    ds_map = defaultdict(list)
    for t in trajs:
        ds_map[t.dataset].append(t)
    train, test = [], []
    for ts in ds_map.values():
        train.extend(ts[:train_n]); test.extend(ts[train_n:])
    return train, test


def capability_gap(tiers: dict, test_trajs: list, train_trajs: list) -> dict:
    r_cheap    = evaluate_policy(AlwaysCheap(),    test_trajs, train_trajs, tiers=tiers)
    r_frontier = evaluate_policy(AlwaysFrontier(), test_trajs, train_trajs, tiers=tiers)
    return {
        "always_t1_tsr": round(r_cheap.task_success_rate,    4),
        "always_t1_acc": round(r_cheap.mean_accuracy,        4),
        "always_t3_tsr": round(r_frontier.task_success_rate, 4),
        "always_t3_acc": round(r_frontier.mean_accuracy,     4),
        "tsr_gap":       round(r_frontier.task_success_rate - r_cheap.task_success_rate, 4),
        "acc_gap":       round(r_frontier.mean_accuracy      - r_cheap.mean_accuracy,    4),
    }


FIXED_POLICIES = [
    AlwaysCheap(),
    UniformRouting(tier=2),
    AlwaysFrontier(),
    TRIMStyle(theta=0.75),
    PRMGuided(theta_high=0.86, theta_low=0.62),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print(f"\n{'='*70}")
    print("Experiment 12: Modern Model Stack Evaluation")
    print(f"{'='*70}\n")

    use_openai    = bool(OPENAI_API_KEY)
    use_anthropic = bool(ANTHROPIC_API_KEY)
    print(f"Live calibration: OpenAI={'yes' if use_openai else 'no (OPENAI_API_KEY not set)'}  "
          f"Anthropic={'yes' if use_anthropic else 'no (ANTHROPIC_API_KEY not set)'}\n")

    all_trajs = load_trajectories("versa", TOTAL_PER_DS)
    train_trajs, test_trajs = split(all_trajs)
    print(f"Train: {len(train_trajs)} | Test: {len(test_trajs)}\n")

    # ── Calibrate Modern Recommended tiers if any API key is present ──────────
    calibration_log = {}
    if use_openai or use_anthropic:
        print("Calibrating Modern Recommended tier parameters via live inference ...")
        calibrated_modern, calibration_log = calibrate_modern_tiers(test_trajs)
        print(f"  Calibration complete.\n")
    else:
        print("No API keys set — using simulated tier parameters for all stacks.\n")
        calibrated_modern = dict(MODERN_TIERS)

    STACKS = {
        "Original (Exp 2)":   TIERS,
        "Modern Recommended": calibrated_modern,
        "Modern Open-Weight": OPEN_WEIGHT_TIERS,
    }

    all_stack_results = {}

    for stack_name, tiers in STACKS.items():
        tier_names = {k: v.name for k, v in tiers.items()}
        print(f"\n{'='*65}")
        print(f"Stack: {stack_name}")
        print(f"  T1={tier_names[1]}  T2={tier_names[2]}  T3={tier_names[3]}")

        if stack_name == "Modern Recommended":
            t2 = tiers[2]; t3 = tiers[3]
            calibrated_t2 = "t2_gpt"    in calibration_log
            calibrated_t3 = "t3_claude" in calibration_log
            print(f"  T2 p_good={t2.p_good_step_success:.3f}  p_bad={t2.p_bad_step_recovery:.3f}"
                  f"  {'[calibrated]' if calibrated_t2 else '[simulated]'}")
            print(f"  T3 p_good={t3.p_good_step_success:.3f}  p_bad={t3.p_bad_step_recovery:.3f}"
                  f"  {'[calibrated]' if calibrated_t3 else '[simulated]'}")
        print(f"{'='*65}")

        gap = capability_gap(tiers, test_trajs, train_trajs)
        print(f"\nCapability gap (TSR):  Always-T1={gap['always_t1_tsr']:.4f}  "
              f"Always-T3={gap['always_t3_tsr']:.4f}  "
              f"Gap={gap['tsr_gap']:+.4f}")

        hdr = (f"{'Policy':<28} {'TSR':>7} {'Acc':>7} {'Cost$':>9} "
               f"{'CostN':>7} {'EscRate':>8} {'AvgTier':>8}")
        print(f"\n{hdr}")
        print("-" * len(hdr))

        stack_results = []
        for policy in FIXED_POLICIES:
            r = evaluate_policy(policy, test_trajs, train_trajectories=train_trajs, tiers=tiers)
            s = r.summary()
            print(f"{s['policy']:<28} {s['task_success_rate']:>7.4f} {s['accuracy']:>7.4f} "
                  f"{s['cost_usd_per_traj']:>9.5f} {s['cost_norm_per_traj']:>7.1f} "
                  f"{s['escalation_rate']:>8.3f} {s['avg_tier']:>8.3f}")
            stack_results.append(s)

        uniform_tsr = next(r["task_success_rate"] for r in stack_results if "Uniform"    in r["policy"])
        prm_tsr     = next(r["task_success_rate"] for r in stack_results if "PRM-Guided" in r["policy"])
        print(f"\n  PRM advantage over Uniform: TSR Δ={prm_tsr - uniform_tsr:+.4f}")

        all_stack_results[stack_name] = {
            "tier_names":        tier_names,
            "capability_gap":    gap,
            "results":           stack_results,
            "prm_advantage_tsr": round(prm_tsr - uniform_tsr, 4),
        }

    # ── Cross-stack comparison ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Cross-Stack Comparison (PRMGuided Task Success Rate)")
    print(f"{'='*70}")
    for stack_name, data in all_stack_results.items():
        prm = next(r for r in data["results"] if "PRM-Guided" in r["policy"])
        uni = next(r for r in data["results"] if "Uniform"    in r["policy"])
        gap = data["capability_gap"]
        print(f"  {stack_name:<24}  PRM TSR={prm['task_success_rate']:.4f}  "
              f"Uni TSR={uni['task_success_rate']:.4f}  "
              f"Δ={data['prm_advantage_tsr']:+.4f}  "
              f"T1-T3 gap={gap['tsr_gap']:+.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        "config": {
            "train_per_dataset": TRAIN_PER_DS,
            "test_per_dataset":  TEST_PER_DS,
            "prm_signal":        "versa",
            "success_threshold": SUCCESS_THRESHOLD,
            "live_calibration": {
                "openai":    use_openai,
                "anthropic": use_anthropic,
            },
        },
        "calibration": calibration_log,
        "stacks": all_stack_results,
    }
    with open(RESULTS_DIR / "exp12_results.json", "w") as f:
        json.dump(out, f, indent=2)

    lines = ["Experiment 12: Modern Model Stack Evaluation", ""]
    for stack_name, data in all_stack_results.items():
        lines.append(f"=== {stack_name} ===")
        gap = data["capability_gap"]
        lines.append(f"T1-T3 TSR gap: {gap['tsr_gap']:+.4f}  "
                     f"(T1={gap['always_t1_tsr']:.4f}  T3={gap['always_t3_tsr']:.4f})")
        hdr = f"{'Policy':<28} {'TSR':>7} {'Acc':>7} {'Cost$':>9} {'CostN':>7} {'EscRate':>8}"
        lines.append(hdr); lines.append("-" * len(hdr))
        for r in data["results"]:
            lines.append(
                f"{r['policy']:<28} {r['task_success_rate']:>7.4f} {r['accuracy']:>7.4f} "
                f"{r['cost_usd_per_traj']:>9.5f} {r['cost_norm_per_traj']:>7.1f} "
                f"{r['escalation_rate']:>8.3f}"
            )
        lines.append("")

    with open(RESULTS_DIR / "exp12_summary.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {RESULTS_DIR}")
    return all_stack_results


if __name__ == "__main__":
    run()
