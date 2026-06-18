"""
Experiment 4: Domain Generalization.

Research question: Does PRM-conditioned routing transfer to an unseen domain?

Setup:
  - Train/calibrate: AgentProcessBench (hotpotqa + gaia_dev + bfcl), 750 trajectories
  - Test:            AgentProcessBench (tau2 only), 250 trajectories
  - No retraining — zero-shot transfer of routing thresholds to banking/airline domain

tau2 is structurally different:
  - Source: tau2-bench conversational agent benchmark (airline, banking, retail tasks)
  - Multi-turn tool-using agent with 3,557 labeled steps (bad_rate=0.348)
  - vs in-distribution: hotpotqa (bad_rate=0.320), bfcl (0.260), gaia_dev (0.624)

Compares:
  1. In-distribution performance (same 3 datasets for train and test, via cross-dataset eval)
  2. Cross-domain performance (train on 3, test on tau2)
  3. Degradation = in-dist accuracy − cross-domain accuracy

Outputs:
  results/exp4/exp4_results.json
  results/exp4/exp4_summary.txt
  experiments/exp4_generalization/exp4_notes.md  (written at end)
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_routing"))

from data_loader import load_trajectories, DATASETS
from routing_policies import (
    UniformRouting, AlwaysFrontier, AlwaysCheap,
    TRIMStyle, BAARStyle, PRMGuided,
)
from simulator import evaluate_policy

RESULTS_DIR = Path("/workspace/PRM_Routing/results/exp4")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_DATASETS = ["hotpotqa", "gaia_dev", "bfcl"]   # train domains
TARGET_DATASET  = "tau2"                               # transfer domain

POLICIES = [
    AlwaysCheap(),
    UniformRouting(tier=2),
    AlwaysFrontier(),
    TRIMStyle(theta=0.75),
    BAARStyle(escalation_threshold=0.4, low_tier=1, high_tier=3),
    PRMGuided(theta_high=0.86, theta_low=0.62),
]


def split_by_domain(trajectories, source_ds, target_ds):
    """Split trajectories by dataset into source (train) and target (test)."""
    source = [t for t in trajectories if t.dataset in source_ds]
    target = [t for t in trajectories if t.dataset == target_ds]
    return source, target


def run():
    print(f"\n{'='*70}")
    print("Experiment 4: Domain Generalization")
    print(f"  Source (train): {SOURCE_DATASETS}")
    print(f"  Target (test):  {TARGET_DATASET}")
    print(f"{'='*70}\n")

    all_trajs = load_trajectories("versa", n_per_dataset=250)
    source_trajs, target_trajs = split_by_domain(all_trajs, SOURCE_DATASETS, TARGET_DATASET)

    print(f"Source: {len(source_trajs)} trajectories | Target: {len(target_trajs)} trajectories")
    print(f"Source steps: {sum(len(t.steps) for t in source_trajs)} | "
          f"Target steps: {sum(len(t.steps) for t in target_trajs)}\n")

    # ── In-distribution reference: evaluate each policy on source datasets ──
    # Use last 50 trajectories from each source dataset as the in-dist test set
    source_ds_map = defaultdict(list)
    for t in source_trajs:
        source_ds_map[t.dataset].append(t)
    indist_train = []
    indist_test  = []
    for ds, trajs in source_ds_map.items():
        indist_train.extend(trajs[:-50])   # first 200 → train
        indist_test.extend(trajs[-50:])    # last 50 → in-dist test

    print(f"In-dist train: {len(indist_train)} | In-dist test: {len(indist_test)}")
    print(f"Cross-domain test (tau2): {len(target_trajs)}\n")

    results = []

    print("Policy                        In-dist Acc  Transfer Acc  Degradation  EscRate(τ)  CostN(τ)")
    print("-" * 95)

    for policy in POLICIES:
        # In-distribution evaluation
        r_indist = evaluate_policy(policy, indist_test, train_trajectories=indist_train)

        # Cross-domain evaluation (zero-shot: same thresholds, fit on source if needed)
        r_transfer = evaluate_policy(policy, target_trajs, train_trajectories=source_trajs)

        degradation = r_indist.mean_accuracy - r_transfer.mean_accuracy
        rel_degradation = degradation / r_indist.mean_accuracy if r_indist.mean_accuracy > 0 else 0

        print(f"{policy.name:<30} {r_indist.mean_accuracy:>11.4f}  "
              f"{r_transfer.mean_accuracy:>12.4f}  "
              f"{degradation:>+11.4f}  "
              f"{r_transfer.escalation_rate:>10.3f}  "
              f"{r_transfer.mean_cost_norm_per_traj:>8.0f}")

        results.append({
            "policy": policy.name,
            "indist_acc":          round(r_indist.mean_accuracy, 4),
            "indist_acc_std":      round(r_indist.std_accuracy, 4),
            "indist_cost_norm":    round(r_indist.mean_cost_norm_per_traj, 1),
            "transfer_acc":        round(r_transfer.mean_accuracy, 4),
            "transfer_acc_std":    round(r_transfer.std_accuracy, 4),
            "transfer_cost_norm":  round(r_transfer.mean_cost_norm_per_traj, 1),
            "degradation":         round(degradation, 4),
            "relative_degradation": round(rel_degradation, 4),
            "transfer_esc_rate":   round(r_transfer.escalation_rate, 4),
            "transfer_avg_tier":   round(r_transfer.avg_tier, 3),
            "transfer_stability":  round(r_transfer.routing_stability, 4),
            "indist_per_dataset":  r_indist.per_dataset,
            "transfer_per_dataset": r_transfer.per_dataset,
        })

    # ── Summary analysis ──
    print()
    uniform_indist    = next(r for r in results if 'Uniform' in r['policy'])['indist_acc']
    uniform_transfer  = next(r for r in results if 'Uniform' in r['policy'])['transfer_acc']
    prm_indist        = next(r for r in results if 'PRM-Guided' in r['policy'])['indist_acc']
    prm_transfer      = next(r for r in results if 'PRM-Guided' in r['policy'])['transfer_acc']

    print(f"Uniform  (T2):   in-dist={uniform_indist:.4f}  transfer={uniform_transfer:.4f}  "
          f"Δ={uniform_transfer-uniform_indist:+.4f}")
    print(f"PRM-Guided:      in-dist={prm_indist:.4f}  transfer={prm_transfer:.4f}  "
          f"Δ={prm_transfer-prm_indist:+.4f}")
    print()

    # Compute PRM advantage (vs Uniform) in-dist vs on transfer
    prm_advantage_indist    = prm_indist - uniform_indist
    prm_advantage_transfer  = prm_transfer - uniform_transfer
    print(f"PRM-Guided advantage over Uniform:")
    print(f"  In-distribution: {prm_advantage_indist:+.4f}")
    print(f"  Transfer domain: {prm_advantage_transfer:+.4f}")
    print(f"  Retention of advantage: {prm_advantage_transfer/prm_advantage_indist:.1%}"
          if prm_advantage_indist != 0 else "  N/A")

    # ── Per-dataset breakdown for PRM-Guided ──
    print()
    print("PRM-Guided accuracy by dataset (in-dist vs target):")
    prm_row = next(r for r in results if 'PRM-Guided' in r['policy'])
    for ds in SOURCE_DATASETS:
        acc = prm_row['indist_per_dataset'].get(ds, {}).get('accuracy', float('nan'))
        print(f"  {ds:<12} {acc:.4f}  (in-dist)")
    tau2_acc = prm_row['transfer_per_dataset'].get('tau2', {}).get('accuracy', float('nan'))
    print(f"  {'tau2':<12} {tau2_acc:.4f}  (transfer)")

    # ── Save ──
    out = {
        "config": {
            "source_datasets": SOURCE_DATASETS,
            "target_dataset": TARGET_DATASET,
            "prm_signal": "versa",
        },
        "results": results,
    }
    with open(RESULTS_DIR / "exp4_results.json", "w") as f:
        json.dump(out, f, indent=2)

    lines = [
        "Experiment 4: Domain Generalization",
        f"Source: {SOURCE_DATASETS} | Target: {TARGET_DATASET}",
        "",
        f"{'Policy':<30} {'In-dist':>8} {'Transfer':>9} {'Degrad':>8} {'EscRate':>8} {'CostN':>8}",
        "-" * 75,
    ]
    for r in results:
        lines.append(
            f"{r['policy']:<30} {r['indist_acc']:>8.4f} {r['transfer_acc']:>9.4f} "
            f"{r['degradation']:>+8.4f} {r['transfer_esc_rate']:>8.3f} "
            f"{r['transfer_cost_norm']:>8.0f}"
        )
    with open(RESULTS_DIR / "exp4_summary.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    run()
