"""
Experiment 13: Percentile-Based Dynamic Threshold Routing.

Research question: Can per-trajectory adaptive thresholds outperform the fixed
θ_high=0.86 / θ_low=0.62 thresholds while maintaining similar cost?

Motivation:
  Fixed thresholds are calibrated on the global VersaPRM score distribution.
  But individual trajectories have different score dynamics — a trajectory with
  consistently high scores will rarely escalate, while a low-scoring trajectory
  may over-escalate. Adaptive percentiles normalise routing to each trajectory's
  own score range.

Method:
  At each step i, compute running percentiles over scores[0..i-1].
  Use P_low and P_high as dynamic routing thresholds.
  Fall back to fixed thresholds when fewer than MIN_WINDOW steps have been seen.

Variants:
  Fixed (baseline):  θ_low=0.62, θ_high=0.86
  DynP10/90:         low=P10, high=P90 of scores so far
  DynP20/80:         low=P20, high=P80
  DynP25/75:         low=P25, high=P75

Success criterion:
  At least one percentile variant improves task success rate over fixed PRMGuided
  within ±10% of its cost.

Outputs:
  results/exp13/exp13_results.json
  results/exp13/exp13_summary.txt
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path("/workspace/PRM_Routing")
sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_routing"))

from data_loader import load_trajectories
from routing_policies import PRMGuided, UniformRouting, RoutingPolicy
from simulator import evaluate_policy, SUCCESS_THRESHOLD

RESULTS_DIR = ROOT / "results/exp13"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PER_DS = 200
TEST_PER_DS  = 50
TOTAL_PER_DS = 250

# Minimum number of prior scores before switching from fixed to adaptive thresholds.
MIN_WINDOW = 3


# ---------------------------------------------------------------------------
# Routing policy
# ---------------------------------------------------------------------------

class PercentileDynamicRouting(RoutingPolicy):
    """
    Routes each step based on where the previous step's score falls within
    the running percentile distribution of scores seen so far in this trajectory.

    When fewer than MIN_WINDOW prior scores are available the policy falls back
    to the supplied fixed thresholds so the first few steps are never mis-routed
    by an unrepresentative sample.
    """

    def __init__(
        self,
        pct_low: float  = 10.0,
        pct_high: float = 90.0,
        fallback_theta_low:  float = 0.62,
        fallback_theta_high: float = 0.86,
        min_window: int = MIN_WINDOW,
        default_tier: int = 2,
    ):
        self.pct_low  = pct_low
        self.pct_high = pct_high
        self.fallback_low  = fallback_theta_low
        self.fallback_high = fallback_theta_high
        self.min_window    = min_window
        self.default_tier  = default_tier
        self.name = f"DynP{int(pct_low)}/{int(pct_high)}"

    def decide(self, traj, step_idx: int) -> int:
        if step_idx == 0:
            return self.default_tier

        score = traj.steps[step_idx - 1].versa_score  # causal: prev step's score

        # Scores from all steps already scored before this decision
        prev_scores = [
            s.versa_score for s in traj.steps[:step_idx]
            if s.versa_score is not None
        ]

        if len(prev_scores) < self.min_window:
            # Not enough data — use fixed thresholds
            theta_low, theta_high = self.fallback_low, self.fallback_high
        else:
            theta_low  = float(np.percentile(prev_scores, self.pct_low))
            theta_high = float(np.percentile(prev_scores, self.pct_high))

        if score > theta_high:
            return 1
        elif score < theta_low:
            return 3
        else:
            return 2

    def fit(self, train_trajectories):
        pass   # no training needed


# ---------------------------------------------------------------------------
# Sweep: generate multiple percentile variants for Pareto comparison
# ---------------------------------------------------------------------------

def percentile_sweep():
    pct_pairs = [
        (5,  95),
        (10, 90),
        (15, 85),
        (20, 80),
        (25, 75),
        (30, 70),
    ]
    return [PercentileDynamicRouting(lo, hi) for lo, hi in pct_pairs]


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
    n_esc     = sum(1 for s in all_steps if s.tier_selected == 3)
    bad_t3    = sum(1 for s in all_steps if s.human_label == -1 and s.tier_selected == 3)
    good_t1   = sum(1 for s in all_steps if s.human_label == 1  and s.tier_selected == 1)
    n_bad     = sum(1 for s in all_steps if s.human_label == -1)
    n_good    = sum(1 for s in all_steps if s.human_label == 1)
    return {
        "routing_precision":  bad_t3 / n_esc  if n_esc   else 0.0,
        "bad_to_t3_rate":     bad_t3 / n_bad  if n_bad   else 0.0,
        "good_to_t1_rate":    good_t1 / n_good if n_good else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print(f"\n{'='*70}")
    print("Experiment 13: Percentile-Based Dynamic Threshold Routing")
    print(f"{'='*70}\n")

    all_trajs = load_trajectories("versa", TOTAL_PER_DS)
    train_trajs, test_trajs = split(all_trajs)
    print(f"Train: {len(train_trajs)} | Test: {len(test_trajs)}\n")

    # ── Fixed baselines ───────────────────────────────────────────────────────
    fixed_prm  = evaluate_policy(PRMGuided(0.86, 0.62), test_trajs)
    uniform    = evaluate_policy(UniformRouting(2),     test_trajs)
    fixed_rq   = routing_quality(fixed_prm)

    print(f"Fixed PRM-Guided:   TSR={fixed_prm.task_success_rate:.4f}  "
          f"Acc={fixed_prm.mean_accuracy:.4f}  "
          f"CostN={fixed_prm.mean_cost_norm_per_traj:.0f}  "
          f"Prec={fixed_rq['routing_precision']:.3f}")
    print(f"Uniform (T2):       TSR={uniform.task_success_rate:.4f}  "
          f"Acc={uniform.mean_accuracy:.4f}  "
          f"CostN={uniform.mean_cost_norm_per_traj:.0f}\n")

    # ── Fixed variants (different pct values) ────────────────────────────────
    FIXED_VARIANTS = [
        PercentileDynamicRouting(10, 90),
        PercentileDynamicRouting(20, 80),
        PercentileDynamicRouting(25, 75),
    ]

    print(f"{'='*80}")
    print("EXPERIMENT 13 RESULTS")
    print(f"{'='*80}")
    hdr = (f"{'Policy':<24} {'TSR':>7} {'ΔvFixed':>8} {'Acc':>7} "
           f"{'CostN':>9} {'ΔCost%':>8} {'Prec':>6} {'EscRate':>8} {'AvgTier':>8}")
    print(hdr); print("-"*len(hdr))

    base_tsr  = fixed_prm.task_success_rate
    base_cost = fixed_prm.mean_cost_norm_per_traj

    def print_row(name, r, rq):
        dtsr  = r.task_success_rate - base_tsr
        dcost = (r.mean_cost_norm_per_traj - base_cost) / base_cost if base_cost else 0
        print(f"{name:<24} {r.task_success_rate:>7.4f} {dtsr:>+8.4f} {r.mean_accuracy:>7.4f} "
              f"{r.mean_cost_norm_per_traj:>9.0f} {dcost:>+7.1%} "
              f"{rq['routing_precision']:>6.3f} {r.escalation_rate:>8.3f} {r.avg_tier:>8.3f}")

    # Print fixed baseline first for reference
    print_row("Fixed (0.62/0.86)", fixed_prm, fixed_rq)

    results = [{
        "policy": "Fixed (0.62/0.86)",
        "per_dataset": fixed_prm.per_dataset,
        **fixed_prm.summary(),
        **{f"rq_{k}": round(v, 4) for k, v in fixed_rq.items()},
    }]

    for pol in FIXED_VARIANTS:
        r  = evaluate_policy(pol, test_trajs)
        rq = routing_quality(r)
        print_row(pol.name, r, rq)
        results.append({
            "policy": pol.name,
            "per_dataset": r.per_dataset,
            **r.summary(),
            **{f"rq_{k}": round(v, 4) for k, v in rq.items()},
        })

    # ── Per-dataset breakdown ─────────────────────────────────────────────────
    print(f"\nPer-dataset TSR ({len(test_trajs)} test trajs):")
    datasets = ["hotpotqa", "gaia_dev", "bfcl", "tau2"]
    hdr2 = f"{'Policy':<24} " + "  ".join(f"{ds:<10}" for ds in datasets)
    print(hdr2); print("-"*len(hdr2))
    for r_dict in results:
        row = f"{r_dict['policy']:<24} "
        for ds in datasets:
            row += f"  {r_dict.get('per_dataset', {}).get(ds, {}).get('accuracy', float('nan')):<10.4f}"
        print(row)

    # ── Pareto sweep ─────────────────────────────────────────────────────────
    print(f"\nPareto sweep over all percentile pairs ...")
    pareto = []
    for pol in percentile_sweep():
        r  = evaluate_policy(pol, test_trajs)
        rq = routing_quality(r)
        pareto.append({
            "policy": pol.name,
            "pct_low": pol.pct_low, "pct_high": pol.pct_high,
            **r.summary(),
            "routing_precision": round(rq["routing_precision"], 4),
        })
    best = max(pareto, key=lambda x: x["task_success_rate"])
    print(f"  Best: {best['policy']}  TSR={best['task_success_rate']:.4f}  "
          f"CostN={best['cost_norm_per_traj']:.0f}  Prec={best['routing_precision']:.3f}")

    # ── Success criterion ─────────────────────────────────────────────────────
    print(f"\n[Success Criterion]  TSR > fixed AND cost within ±10%")
    for r_dict in results[1:]:
        dtsr  = r_dict["task_success_rate"] - base_tsr
        dcost = (r_dict["cost_norm_per_traj"] - base_cost) / base_cost if base_cost else 0
        tsr_ok  = dtsr > 0
        cost_ok = abs(dcost) <= 0.10
        print(f"  {r_dict['policy']:<20}  ΔTSR={dtsr:+.4f} {'✓' if tsr_ok else '✗'}  "
              f"ΔCost={dcost:+.1%} {'✓' if cost_ok else '✗'}  "
              f"MET={'YES ✓' if (tsr_ok and cost_ok) else 'NO'}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = {
        "config": {
            "min_window": MIN_WINDOW,
            "fallback_theta_low": 0.62,
            "fallback_theta_high": 0.86,
            "success_threshold": SUCCESS_THRESHOLD,
        },
        "results": results,
        "pareto": pareto,
        "fixed_baseline": {
            "task_success_rate": round(base_tsr, 4),
            "cost_norm_per_traj": round(base_cost, 1),
        },
    }
    with open(RESULTS_DIR / "exp13_results.json", "w") as f:
        json.dump(out, f, indent=2)

    lines = ["Experiment 13: Percentile-Based Dynamic Threshold Routing", "", hdr, "-"*len(hdr)]
    for r_dict in results:
        dtsr  = r_dict["task_success_rate"] - base_tsr
        dcost = (r_dict["cost_norm_per_traj"] - base_cost) / base_cost if base_cost else 0
        rq_prec = r_dict.get("rq_routing_precision", 0)
        lines.append(
            f"{r_dict['policy']:<24} {r_dict['task_success_rate']:>7.4f} {dtsr:>+8.4f} "
            f"{r_dict['accuracy']:>7.4f} {r_dict['cost_norm_per_traj']:>9.0f} {dcost:>+7.1%} "
            f"{rq_prec:>6.3f} {r_dict['escalation_rate']:>8.3f} {r_dict['avg_tier']:>8.3f}"
        )
    with open(RESULTS_DIR / "exp13_summary.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {RESULTS_DIR}")
    return results, pareto


if __name__ == "__main__":
    run()
