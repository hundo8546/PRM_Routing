"""
Experiment 14: Uncertainty-Adaptive Threshold Routing.

Research question: Can disagreement between VersaPRM and DG-PRM be used to
dynamically adjust routing thresholds — rather than as a direct routing signal
— to improve decisions compared to static thresholds?

Motivation:
  Exp 7 showed that multi-judge disagreement is informative as an escalation
  trigger (disagreement > τ → T3).  This experiment tests an alternative:
  use the disagreement level to *tighten or loosen* the routing thresholds
  instead of hardwiring an escalation.

  Low uncertainty (PRMs agree)  → trust the signal → aggressive routing
  High uncertainty (PRMs clash) → hedge → conservative thresholds

Method:
  disagreement = abs(versa_score - dgprm_score)

  if   disagreement < τ_low:     use aggressive thresholds (θ_low=0.55, θ_high=0.90)
  elif disagreement < τ_high:    use standard  thresholds  (θ_low=0.62, θ_high=0.86)
  else:                          use cautious  thresholds  (θ_low=0.75, θ_high=0.80)

Variants (τ_low, τ_high):
  UA-1:  (0.10, 0.25)
  UA-2:  (0.05, 0.15)  — narrow bands, reacts more quickly
  UA-3:  (0.15, 0.30)  — wider bands, more stable

Comparisons:
  Fixed PRM-Guided (Exp 2 best)
  Multi-Judge disagreement (Exp 7 best)
  Temporal drop (Exp 8 best)
  Uniform (T2 baseline)

Success criterion:
  At least one UA variant improves task success over Fixed PRM-Guided while
  increasing cost less than Multi-Judge does.

Requires augmented trajectories with both versa and dgprm scores.

Outputs:
  results/exp14/exp14_results.json
  results/exp14/exp14_summary.txt
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path("/workspace/PRM_Routing")
sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_routing"))
sys.path.insert(0, str(Path(__file__).parent.parent / "exp5678_additional"))

from data_loader import load_trajectories
from routing_policies import PRMGuided, UniformRouting, RoutingPolicy
from simulator import evaluate_policy, SUCCESS_THRESHOLD
from disagreement_loader import load_augmented_trajectories
from disagreement_policies import (
    MultiJudgeDisagreement,
    TemporalDisagreement,
)

RESULTS_DIR = ROOT / "results/exp14"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PER_DS = 200
TEST_PER_DS  = 50
TOTAL_PER_DS = 250

# Routing threshold levels
AGGRESSIVE = (0.55, 0.90)   # (θ_low, θ_high) when uncertainty is low
STANDARD   = (0.62, 0.86)   # standard PRMGuided thresholds
CAUTIOUS   = (0.75, 0.80)   # narrow band → more T2/T3 when uncertain


# ---------------------------------------------------------------------------
# Routing policy
# ---------------------------------------------------------------------------

class UncertaintyAdaptiveRouting(RoutingPolicy):
    """
    Adjusts routing thresholds based on |versa_score - dgprm_score|.

    Low disagreement  (<τ_low)  → aggressive thresholds (trust the PRM, push to T1)
    Mid disagreement  (<τ_high) → standard thresholds
    High disagreement (≥τ_high) → cautious thresholds (narrower T2 band → more T3)
    """

    def __init__(
        self,
        tau_low:  float = 0.10,
        tau_high: float = 0.25,
        aggressive: tuple = AGGRESSIVE,
        standard:   tuple = STANDARD,
        cautious:   tuple = CAUTIOUS,
        default_tier: int = 2,
    ):
        self.tau_low  = tau_low
        self.tau_high = tau_high
        self.aggressive = aggressive
        self.standard   = standard
        self.cautious   = cautious
        self.default_tier = default_tier
        self.name = f"UA(τ={tau_low}/{tau_high})"

    def decide(self, traj, step_idx: int) -> int:
        if step_idx == 0:
            return self.default_tier

        prev = traj.steps[step_idx - 1]
        v = prev.versa_score
        d = getattr(prev, "extra_scores", {}).get("dgprm", v)   # fallback: no disagreement
        disagreement = abs(v - d)

        if disagreement < self.tau_low:
            theta_low, theta_high = self.aggressive
        elif disagreement < self.tau_high:
            theta_low, theta_high = self.standard
        else:
            theta_low, theta_high = self.cautious

        if v > theta_high:
            return 1
        elif v < theta_low:
            return 3
        else:
            return 2

    def fit(self, train_trajectories):
        pass


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def ua_sweep():
    tau_lows  = [0.05, 0.10, 0.15, 0.20]
    tau_highs = [0.15, 0.20, 0.25, 0.30, 0.40]
    return [
        UncertaintyAdaptiveRouting(tl, th)
        for tl in tau_lows for th in tau_highs
        if tl < th
    ]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def split(trajs, train_n=TRAIN_PER_DS):
    ds_map = defaultdict(list)
    for t in trajs:
        ds_map[t.dataset].append(t)
    train, test = [], []
    for ts in ds_map.values():
        train.extend(ts[:train_n]); test.extend(ts[train_n:])
    return train, test


def routing_quality(result) -> dict:
    all_steps = [s for r in result.traj_results for s in r.step_results]
    n_esc   = sum(1 for s in all_steps if s.tier_selected == 3)
    bad_t3  = sum(1 for s in all_steps if s.human_label == -1 and s.tier_selected == 3)
    good_t1 = sum(1 for s in all_steps if s.human_label == 1  and s.tier_selected == 1)
    n_bad   = sum(1 for s in all_steps if s.human_label == -1)
    n_good  = sum(1 for s in all_steps if s.human_label == 1)
    return {
        "routing_precision": bad_t3 / n_esc  if n_esc  else 0.0,
        "bad_to_t3_rate":    bad_t3 / n_bad  if n_bad  else 0.0,
        "good_to_t1_rate":   good_t1 / n_good if n_good else 0.0,
    }


def uncertainty_stats(test_trajs) -> dict:
    """Summarise disagreement distribution across test steps."""
    disagreements = []
    for traj in test_trajs:
        for step in traj.steps:
            v = step.versa_score
            d = getattr(step, "extra_scores", {}).get("dgprm", None)
            if d is not None:
                disagreements.append(abs(v - d))
    if not disagreements:
        return {}
    arr = np.array(disagreements)
    return {
        "mean":  float(arr.mean()),
        "p10":   float(np.percentile(arr, 10)),
        "p25":   float(np.percentile(arr, 25)),
        "p50":   float(np.percentile(arr, 50)),
        "p75":   float(np.percentile(arr, 75)),
        "p90":   float(np.percentile(arr, 90)),
        "n_steps": len(arr),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print(f"\n{'='*70}")
    print("Experiment 14: Uncertainty-Adaptive Threshold Routing")
    print(f"{'='*70}\n")

    # Load augmented trajectories (Versa + Qwen + DG-PRM scores)
    print("Loading augmented trajectories (Versa + DG-PRM) ...")
    all_trajs = load_augmented_trajectories(
        prm_names=["versa", "dgprm"], n_per_dataset=TOTAL_PER_DS
    )
    train_trajs, test_trajs = split(all_trajs)
    print(f"Train: {len(train_trajs)} | Test: {len(test_trajs)}\n")

    # Disagreement distribution
    unc = uncertainty_stats(test_trajs)
    if unc:
        print(f"Disagreement stats (|versa - dgprm|):")
        print(f"  mean={unc['mean']:.3f}  p10={unc['p10']:.3f}  p25={unc['p25']:.3f}  "
              f"p50={unc['p50']:.3f}  p75={unc['p75']:.3f}  p90={unc['p90']:.3f}\n")

    # ── Baselines ─────────────────────────────────────────────────────────────
    fixed_prm   = evaluate_policy(PRMGuided(0.86, 0.62),           test_trajs)
    uniform     = evaluate_policy(UniformRouting(2),               test_trajs)
    multi_judge = evaluate_policy(MultiJudgeDisagreement(0.20, 0.86), test_trajs)
    temporal    = evaluate_policy(TemporalDisagreement(0.15),       test_trajs)

    base_tsr  = fixed_prm.task_success_rate
    base_cost = fixed_prm.mean_cost_norm_per_traj
    mj_cost   = multi_judge.mean_cost_norm_per_traj

    print("Baselines:")
    for name, r in [("Uniform", uniform), ("Fixed PRM-Guided", fixed_prm),
                    ("Multi-Judge (Exp7)", multi_judge), ("Temporal (Exp8)", temporal)]:
        rq = routing_quality(r)
        print(f"  {name:<22} TSR={r.task_success_rate:.4f}  Acc={r.mean_accuracy:.4f}  "
              f"CostN={r.mean_cost_norm_per_traj:.0f}  Prec={rq['routing_precision']:.3f}")

    # ── UA variants ───────────────────────────────────────────────────────────
    UA_VARIANTS = [
        UncertaintyAdaptiveRouting(tau_low=0.10, tau_high=0.25),   # UA-1
        UncertaintyAdaptiveRouting(tau_low=0.05, tau_high=0.15),   # UA-2
        UncertaintyAdaptiveRouting(tau_low=0.15, tau_high=0.30),   # UA-3
    ]
    UA_LABELS = ["UA-1 (0.10/0.25)", "UA-2 (0.05/0.15)", "UA-3 (0.15/0.30)"]

    print(f"\n{'='*90}")
    print("EXPERIMENT 14 RESULTS")
    print(f"{'='*90}")
    hdr = (f"{'Policy':<26} {'TSR':>7} {'ΔvFixed':>8} {'Acc':>7} "
           f"{'CostN':>9} {'ΔCost%':>8} {'Prec':>6} {'EscRate':>8} {'AvgTier':>8}")
    print(hdr); print("-"*len(hdr))

    def print_row(name, r, rq):
        dtsr  = r.task_success_rate - base_tsr
        dcost = (r.mean_cost_norm_per_traj - base_cost) / base_cost if base_cost else 0
        print(f"{name:<26} {r.task_success_rate:>7.4f} {dtsr:>+8.4f} {r.mean_accuracy:>7.4f} "
              f"{r.mean_cost_norm_per_traj:>9.0f} {dcost:>+7.1%} "
              f"{rq['routing_precision']:>6.3f} {r.escalation_rate:>8.3f} {r.avg_tier:>8.3f}")

    # Print all comparators
    for name, r in [("Uniform", uniform), ("Fixed PRM-Guided", fixed_prm),
                    ("Multi-Judge (Exp7)", multi_judge), ("Temporal (Exp8)", temporal)]:
        print_row(name, r, routing_quality(r))

    print("-"*len(hdr))

    results = []
    for pol, label in zip(UA_VARIANTS, UA_LABELS):
        r  = evaluate_policy(pol, test_trajs)
        rq = routing_quality(r)
        print_row(label, r, rq)
        results.append({
            "policy": label,
            "tau_low":  pol.tau_low,
            "tau_high": pol.tau_high,
            **r.summary(),
            **{f"rq_{k}": round(v, 4) for k, v in rq.items()},
        })

    # ── Pareto sweep ──────────────────────────────────────────────────────────
    print(f"\nPareto sweep ({len(ua_sweep())} (τ_low, τ_high) combinations) ...")
    pareto = []
    for pol in ua_sweep():
        r  = evaluate_policy(pol, test_trajs)
        rq = routing_quality(r)
        pareto.append({
            "policy": pol.name,
            "tau_low": pol.tau_low, "tau_high": pol.tau_high,
            **r.summary(),
            "routing_precision": round(rq["routing_precision"], 4),
        })
    best = max(pareto, key=lambda x: x["task_success_rate"])
    print(f"  Best: {best['policy']}  TSR={best['task_success_rate']:.4f}  "
          f"CostN={best['cost_norm_per_traj']:.0f}  Prec={best['routing_precision']:.3f}")

    # ── Tier-adaptation breakdown ─────────────────────────────────────────────
    print(f"\nTier-level analysis (how often each uncertainty regime is triggered):")
    for pol, label in zip(UA_VARIANTS, UA_LABELS):
        counts = {"aggressive": 0, "standard": 0, "cautious": 0, "default": 0}
        for traj in test_trajs:
            for i, step in enumerate(traj.steps):
                if i == 0:
                    counts["default"] += 1
                    continue
                prev = traj.steps[i - 1]
                v = prev.versa_score
                d = getattr(prev, "extra_scores", {}).get("dgprm", v)
                dis = abs(v - d)
                if dis < pol.tau_low:
                    counts["aggressive"] += 1
                elif dis < pol.tau_high:
                    counts["standard"] += 1
                else:
                    counts["cautious"] += 1
        total = sum(counts.values())
        print(f"  {label:<24} "
              f"aggressive={counts['aggressive']/total:.1%}  "
              f"standard={counts['standard']/total:.1%}  "
              f"cautious={counts['cautious']/total:.1%}")

    # ── Success criterion ─────────────────────────────────────────────────────
    print(f"\n[Success Criterion]  TSR > fixed PRM-Guided AND cost increase < Multi-Judge cost increase")
    mj_dcost = mj_cost - base_cost
    for r_dict in results:
        dtsr  = r_dict["task_success_rate"] - base_tsr
        dcost = r_dict["cost_norm_per_traj"] - base_cost
        tsr_ok  = dtsr > 0
        cost_ok = dcost < mj_dcost
        print(f"  {r_dict['policy']:<22}  ΔTSR={dtsr:+.4f} {'✓' if tsr_ok else '✗'}  "
              f"ΔCostN={dcost:+.0f} (MJ={mj_dcost:+.0f}) {'✓' if cost_ok else '✗'}  "
              f"MET={'YES ✓' if (tsr_ok and cost_ok) else 'NO'}")

    # ── Save ──────────────────────────────────────────────────────────────────
    baselines = {
        "uniform":     uniform.summary(),
        "fixed_prm":   fixed_prm.summary(),
        "multi_judge": multi_judge.summary(),
        "temporal":    temporal.summary(),
    }
    out = {
        "config": {
            "aggressive_thresholds": list(AGGRESSIVE),
            "standard_thresholds":   list(STANDARD),
            "cautious_thresholds":   list(CAUTIOUS),
            "success_threshold": SUCCESS_THRESHOLD,
        },
        "uncertainty_stats": unc,
        "baselines": baselines,
        "results": results,
        "pareto": pareto,
    }
    with open(RESULTS_DIR / "exp14_results.json", "w") as f:
        json.dump(out, f, indent=2)

    lines = ["Experiment 14: Uncertainty-Adaptive Threshold Routing", "", hdr, "-"*len(hdr)]
    for r_dict in results:
        dtsr  = r_dict["task_success_rate"] - base_tsr
        dcost = (r_dict["cost_norm_per_traj"] - base_cost) / base_cost if base_cost else 0
        lines.append(
            f"{r_dict['policy']:<26} {r_dict['task_success_rate']:>7.4f} {dtsr:>+8.4f} "
            f"{r_dict['accuracy']:>7.4f} {r_dict['cost_norm_per_traj']:>9.0f} {dcost:>+7.1%} "
            f"{r_dict.get('rq_routing_precision', 0):>6.3f} "
            f"{r_dict['escalation_rate']:>8.3f} {r_dict['avg_tier']:>8.3f}"
        )
    with open(RESULTS_DIR / "exp14_summary.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {RESULTS_DIR}")
    return results, pareto


if __name__ == "__main__":
    run()
