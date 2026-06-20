"""
Significance testing for Exp 14: Uncertainty-Adaptive Threshold Routing.

Tests whether UA-1, UA-2, UA-3 TSR improvements over Fixed PRM-Guided are
statistically significant.

Methods:
  1. Bootstrap 95% CI on TSR for each policy (10,000 resamples)
  2. Paired bootstrap test: p-value for H0: TSR(UA) - TSR(Fixed) <= 0
     Paired because same trajectories are used for all policies.

Run from /workspace/PRM_Routing/:
    python3 experiments/exp14_uncertainty_adaptive/significance_test.py
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
from routing_policies import PRMGuided, RoutingPolicy
from simulator import simulate_trajectory, SUCCESS_THRESHOLD
from cost_model import TIERS
from disagreement_loader import load_augmented_trajectories

TOTAL_PER_DS = 250
TRAIN_PER_DS = 200
N_BOOT       = 10_000
SEED         = 42
ALPHA        = 0.05

AGGRESSIVE = (0.55, 0.90)
STANDARD   = (0.62, 0.86)
CAUTIOUS   = (0.75, 0.80)


class UncertaintyAdaptiveRouting(RoutingPolicy):
    def __init__(self, tau_low=0.10, tau_high=0.25):
        self.tau_low  = tau_low
        self.tau_high = tau_high
        self.name = f"UA(tau={tau_low}/{tau_high})"

    def decide(self, traj, step_idx):
        if step_idx == 0:
            return 2
        prev = traj.steps[step_idx - 1]
        v = prev.versa_score
        d = getattr(prev, "extra_scores", {}).get("dgprm", v)
        disagreement = abs(v - d)
        if disagreement < self.tau_low:
            theta_low, theta_high = AGGRESSIVE
        elif disagreement < self.tau_high:
            theta_low, theta_high = STANDARD
        else:
            theta_low, theta_high = CAUTIOUS
        if v > theta_high: return 1
        elif v < theta_low: return 3
        else: return 2

    def fit(self, _): pass


def split(trajs, train_n=TRAIN_PER_DS):
    ds_map = defaultdict(list)
    for t in trajs:
        ds_map[t.dataset].append(t)
    train, test = [], []
    for ts in ds_map.values():
        train.extend(ts[:train_n]); test.extend(ts[train_n:])
    return train, test


def get_per_traj_successes(policy, trajectories):
    """Returns binary array: 1 if expected_task_accuracy >= SUCCESS_THRESHOLD."""
    results = [simulate_trajectory(t, policy) for t in trajectories]
    return np.array([1 if r.expected_task_accuracy >= SUCCESS_THRESHOLD else 0
                     for r in results])


def bootstrap_ci(successes, n_boot=N_BOOT, alpha=ALPHA, rng=None):
    """Bootstrap 95% CI on mean (TSR)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = len(successes)
    boot_means = np.array([
        rng.choice(successes, n, replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lo, hi, boot_means


def paired_bootstrap_pvalue(s_base, s_new, n_boot=N_BOOT, rng=None):
    """
    Paired bootstrap p-value for H0: TSR(new) - TSR(base) <= 0.
    One-sided test (we're testing for improvement).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = len(s_base)
    observed_diff = s_new.mean() - s_base.mean()
    # Center the differences under H0
    diffs = s_new - s_base
    centered = diffs - diffs.mean()   # center so E[diff]=0 under H0
    boot_diffs = np.array([
        rng.choice(centered, n, replace=True).mean()
        for _ in range(n_boot)
    ])
    p_value = (boot_diffs >= observed_diff).mean()
    return p_value, observed_diff


def effect_size_cohens_h(p1, p2):
    """Cohen's h for two proportions."""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def main():
    print(f"\n{'='*65}")
    print("Exp 14 Significance Tests (Bootstrap)")
    print(f"{'='*65}")
    print(f"  Resamples: {N_BOOT:,}   Seed: {SEED}   alpha: {ALPHA}")
    print()

    # Load augmented trajectories (need DG-PRM scores for UA policies)
    print("Loading augmented trajectories ...")
    all_trajs = load_augmented_trajectories(
        prm_names=["versa", "dgprm"], n_per_dataset=TOTAL_PER_DS
    )
    _, test_trajs = split(all_trajs)
    n = len(test_trajs)
    print(f"Test trajectories: {n}\n")

    rng = np.random.default_rng(SEED)

    # Per-trajectory success vectors
    policies = {
        "Fixed PRM-Guided":   PRMGuided(0.86, 0.62),
        "UA-1 (0.10/0.25)":  UncertaintyAdaptiveRouting(0.10, 0.25),
        "UA-2 (0.05/0.15)":  UncertaintyAdaptiveRouting(0.05, 0.15),
        "UA-3 (0.15/0.30)":  UncertaintyAdaptiveRouting(0.15, 0.30),
    }

    print("Running policies ...")
    successes = {}
    for name, pol in policies.items():
        s = get_per_traj_successes(pol, test_trajs)
        successes[name] = s
        print(f"  {name:<22}  TSR={s.mean():.4f}  ({s.sum()}/{n} successes)")
    print()

    # Bootstrap CIs
    print(f"{'─'*65}")
    print(f"{'Policy':<22} {'TSR':>6} {'95% CI':>18} {'CI Width':>9}")
    print(f"{'─'*65}")
    cis = {}
    for name, s in successes.items():
        lo, hi, _ = bootstrap_ci(s, rng=rng)
        cis[name] = (lo, hi)
        print(f"{name:<22} {s.mean():>6.4f}  [{lo:.4f}, {hi:.4f}]  {hi-lo:>9.4f}")

    # Paired bootstrap significance vs Fixed baseline
    base_name = "Fixed PRM-Guided"
    s_base = successes[base_name]

    print(f"\n{'─'*65}")
    print(f"Paired bootstrap tests vs {base_name} (H0: delta <= 0)")
    print(f"{'─'*65}")
    print(f"{'Comparison':<22} {'Obs.Delta':>10} {'p-value':>9} {'Sig?':>6} {'Cohen h':>9}")
    print(f"{'─'*65}")

    results_out = {}
    for name, s_new in successes.items():
        if name == base_name:
            continue
        p_val, obs_diff = paired_bootstrap_pvalue(s_base, s_new, rng=rng)
        h = effect_size_cohens_h(s_new.mean(), s_base.mean())
        sig = "YES *" if p_val < ALPHA else "no"
        print(f"{name:<22} {obs_diff:>+10.4f} {p_val:>9.4f} {sig:>6} {h:>9.4f}")
        results_out[name] = {
            "tsr":           round(float(s_new.mean()), 4),
            "tsr_base":      round(float(s_base.mean()), 4),
            "observed_delta": round(float(obs_diff), 4),
            "p_value":       round(float(p_val), 5),
            "significant":   bool(p_val < ALPHA),
            "cohens_h":      round(float(h), 4),
            "ci_95":         [round(cis[name][0], 4), round(cis[name][1], 4)],
            "ci_base_95":    [round(cis[base_name][0], 4), round(cis[base_name][1], 4)],
        }

    # Interpretation
    print(f"\n{'─'*65}")
    print("Interpretation")
    print(f"{'─'*65}")
    for name, r in results_out.items():
        overlap = r["ci_95"][0] < r["ci_base_95"][1]
        print(f"\n  {name}:")
        print(f"    TSR: {r['tsr_base']:.4f} -> {r['tsr']:.4f} (delta={r['observed_delta']:+.4f})")
        print(f"    p={r['p_value']:.4f}  {'SIGNIFICANT' if r['significant'] else 'not significant'} at alpha={ALPHA}")
        print(f"    Cohen's h={r['cohens_h']:.4f}  "
              f"({'small' if abs(r['cohens_h']) < 0.2 else 'medium' if abs(r['cohens_h']) < 0.5 else 'large'} effect)")
        print(f"    95% CIs {'overlap' if overlap else 'do NOT overlap'}: "
              f"base=[{r['ci_base_95'][0]:.4f},{r['ci_base_95'][1]:.4f}]  "
              f"new=[{r['ci_95'][0]:.4f},{r['ci_95'][1]:.4f}]")

    # Save
    out = {
        "n_trajectories": n,
        "n_bootstrap":    N_BOOT,
        "alpha":          ALPHA,
        "baseline":       base_name,
        "baseline_tsr":   round(float(s_base.mean()), 4),
        "baseline_ci_95": [round(cis[base_name][0], 4), round(cis[base_name][1], 4)],
        "comparisons":    results_out,
    }
    out_path = ROOT / "results/exp14/exp14_significance.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
