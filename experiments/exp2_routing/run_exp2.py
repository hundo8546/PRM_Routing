"""
Experiment 2: Routing Comparison on AgentProcessBench (1000 trajectories).

Evaluates all routing baselines and the proposed PRM-Guided policy.
Outputs:
  - results/exp2/exp2_results.json     — main results table
  - results/exp2/exp2_pareto.json      — Pareto curve data (threshold sweeps)
  - results/exp2/exp2_summary.txt      — human-readable table
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# Allow imports from sibling directory
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_trajectories
from routing_policies import (
    UniformRouting, AlwaysFrontier, AlwaysCheap,
    TRIMStyle, DAAOStyle, BAARStyle, PRMGuided,
    trim_sweep, prm_guided_sweep,
)
from simulator import evaluate_policy, PolicyResult

RESULTS_DIR = Path("/workspace/PRM_Routing/results/exp2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Train / test split: 800 train (200/dataset), 200 test (50/dataset)
# ---------------------------------------------------------------------------
TRAIN_PER_DS = 200
TEST_PER_DS = 50
TOTAL_PER_DS = 250   # full Exp 1 run


def split_trajectories(trajectories, train_per_ds=TRAIN_PER_DS):
    """Split by dataset, first train_per_ds = train, rest = test."""
    ds_trajs = defaultdict(list)
    for t in trajectories:
        ds_trajs[t.dataset].append(t)

    train, test = [], []
    for ds, trajs in ds_trajs.items():
        train.extend(trajs[:train_per_ds])
        test.extend(trajs[train_per_ds:])
    return train, test


# ---------------------------------------------------------------------------
# Main policies to evaluate
# ---------------------------------------------------------------------------
FIXED_POLICIES = [
    AlwaysCheap(),
    UniformRouting(tier=2),
    AlwaysFrontier(),
    DAAOStyle(),
    BAARStyle(escalation_threshold=0.4, low_tier=1, high_tier=3),
    TRIMStyle(theta=0.75),                          # p50 of VersaPRM score dist
    PRMGuided(theta_high=0.86, theta_low=0.62),     # p75/p25 of VersaPRM score dist
]


def print_table(results):
    header = (
        f"{'Policy':<28} {'Acc':>7} {'±':>5} {'Cost$':>9} {'CostN':>7} "
        f"{'EscRate':>8} {'AvgTier':>8} {'Stability':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        s = r.summary()
        print(
            f"{s['policy']:<28} {s['accuracy']:>7.4f} {s['accuracy_std']:>5.4f} "
            f"{s['cost_usd_per_traj']:>9.5f} {s['cost_norm_per_traj']:>7.1f} "
            f"{s['escalation_rate']:>8.3f} {s['avg_tier']:>8.3f} "
            f"{s['routing_stability']:>10.4f}"
        )


def run(prm_signal: str = "versa", save: bool = True, verbose: bool = True):
    print(f"\n{'='*70}")
    print(f"Experiment 2: Routing Comparison  (PRM signal: {prm_signal})")
    print(f"{'='*70}")

    # Load all 1000 trajectories with VersaPRM scores
    all_trajs = load_trajectories(prm_name=prm_signal, n_per_dataset=TOTAL_PER_DS)
    train_trajs, test_trajs = split_trajectories(all_trajs, TRAIN_PER_DS)

    if verbose:
        print(f"Train: {len(train_trajs)} trajectories | Test: {len(test_trajs)} trajectories")
        print(f"Test steps: {sum(len(t.steps) for t in test_trajs)}")
        print()

    # --- Evaluate fixed policies ---
    print("Evaluating fixed policies...")
    policy_results = []
    for policy in FIXED_POLICIES:
        r = evaluate_policy(policy, test_trajs, train_trajectories=train_trajs)
        policy_results.append(r)
        if verbose:
            s = r.summary()
            print(f"  {s['policy']:<28} acc={s['accuracy']:.4f}  cost_norm={s['cost_norm_per_traj']:.1f}")

    if verbose:
        print()
        print("[Main Results Table]")
        print_table(policy_results)

    # --- Pareto sweeps ---
    print("\nRunning Pareto threshold sweeps...")
    pareto_trim = []
    for pol in trim_sweep():
        r = evaluate_policy(pol, test_trajs)
        pareto_trim.append({"theta": pol.theta, **r.summary()})

    pareto_prm = []
    for pol in prm_guided_sweep():
        r = evaluate_policy(pol, test_trajs)
        pareto_prm.append({"theta_high": pol.theta_high, "theta_low": pol.theta_low, **r.summary()})

    if verbose:
        print(f"  TRIM sweep: {len(pareto_trim)} operating points")
        print(f"  PRM-Guided sweep: {len(pareto_prm)} operating points")

    # --- Per-dataset breakdown ---
    if verbose:
        print("\n[Per-Dataset Accuracy]")
        datasets = ["hotpotqa", "gaia_dev", "bfcl", "tau2"]
        header2 = f"{'Policy':<28} " + "  ".join(f"{ds:<12}" for ds in datasets)
        print(header2)
        print("-" * len(header2))
        for r in policy_results:
            row = f"{r.policy_name:<28} "
            for ds in datasets:
                acc = r.per_dataset.get(ds, {}).get("accuracy", float("nan"))
                row += f"  {acc:<12.4f}"
            print(row)

    # --- Save ---
    if save:
        main_out = {
            "config": {
                "prm_signal": prm_signal,
                "train_per_dataset": TRAIN_PER_DS,
                "test_per_dataset": TEST_PER_DS,
            },
            "results": [r.summary() for r in policy_results],
        }
        with open(RESULTS_DIR / "exp2_results.json", "w") as f:
            json.dump(main_out, f, indent=2)

        pareto_out = {"trim": pareto_trim, "prm_guided": pareto_prm}
        with open(RESULTS_DIR / "exp2_pareto.json", "w") as f:
            json.dump(pareto_out, f, indent=2)

        # Human-readable summary
        with open(RESULTS_DIR / "exp2_summary.txt", "w") as f:
            f.write("Experiment 2: Routing Comparison\n")
            f.write(f"PRM signal: {prm_signal} | Train: {TRAIN_PER_DS}/ds | Test: {TEST_PER_DS}/ds\n\n")
            lines = []
            header = (
                f"{'Policy':<28} {'Acc':>7} {'±':>5} {'Cost$':>9} {'CostN':>7} "
                f"{'EscRate':>8} {'AvgTier':>8} {'Stability':>10}"
            )
            lines.append(header)
            lines.append("-" * len(header))
            for r in policy_results:
                s = r.summary()
                lines.append(
                    f"{s['policy']:<28} {s['accuracy']:>7.4f} {s['accuracy_std']:>5.4f} "
                    f"{s['cost_usd_per_traj']:>9.5f} {s['cost_norm_per_traj']:>7.1f} "
                    f"{s['escalation_rate']:>8.3f} {s['avg_tier']:>8.3f} "
                    f"{s['routing_stability']:>10.4f}"
                )
            f.write("\n".join(lines))
        print(f"\nResults saved to {RESULTS_DIR}")

    return policy_results, pareto_trim, pareto_prm


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prm", default="versa", choices=["versa", "qwen", "agent", "dgprm"],
                        help="Which PRM's scores to use as routing signal")
    args = parser.parse_args()
    run(prm_signal=args.prm)
