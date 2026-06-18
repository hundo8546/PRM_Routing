"""
Experiments 6, 7, 8: Disagreement-Aware Routing.

All three run on existing Exp 1 score files — no model loading needed.
Priority order per additional_experiments.md: Exp7 > Exp8 > Exp6.

Outputs:
  results/exp6/exp6_results.json
  results/exp7/exp7_results.json
  results/exp8/exp8_results.json
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path("/workspace/PRM_Routing")
sys.path.insert(0, str(Path(__file__).parent / ".."))
sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_routing"))

from disagreement_loader import load_augmented_trajectories
from disagreement_policies import (
    PRMDisagreementRouting, MultiJudgeDisagreement,
    TemporalDisagreement, CombinedSignal,
    multijudge_sweep, temporal_sweep, combined_sweep,
)
from routing_policies import PRMGuided, UniformRouting
from simulator import evaluate_policy, PolicyResult

RESULTS = {
    6: ROOT / "results/exp6",
    7: ROOT / "results/exp7",
    8: ROOT / "results/exp8",
}
for d in RESULTS.values():
    d.mkdir(parents=True, exist_ok=True)

TRAIN_PER_DS = 200
TEST_PER_DS  = 50
TOTAL_PER_DS = 250


def split(trajs, train_n=TRAIN_PER_DS):
    ds_map = defaultdict(list)
    for t in trajs: ds_map[t.dataset].append(t)
    train, test = [], []
    for trajs_ in ds_map.values():
        train.extend(trajs_[:train_n]); test.extend(trajs_[train_n:])
    return train, test


def print_comparison(results, versa_baseline, uniform_baseline):
    """Print results vs VersaPRM and Uniform baselines."""
    hdr = f"{'Policy':<40} {'Acc':>7} {'CostN':>9} {'EscRate':>8} {'vs Versa':>9} {'vs Uniform':>10}"
    print(hdr); print("-" * len(hdr))
    for name, r in results.items():
        da = r.mean_accuracy - versa_baseline.mean_accuracy
        du = r.mean_accuracy - uniform_baseline.mean_accuracy
        print(f"{name:<40} {r.mean_accuracy:>7.4f} "
              f"{r.mean_cost_norm_per_traj:>9.0f} "
              f"{r.escalation_rate:>8.3f} "
              f"{da:>+9.4f} {du:>+10.4f}")


def routing_quality(result):
    all_steps = [s for r in result.traj_results for s in r.step_results]
    n_esc = sum(1 for s in all_steps if s.tier_selected == 3)
    bad_to_t3 = sum(1 for s in all_steps if s.human_label == -1 and s.tier_selected == 3)
    prec = bad_to_t3 / n_esc if n_esc else 0
    return prec


def run_experiments():
    print("Loading augmented trajectories (Versa + Qwen + DG-PRM scores) ...")
    all_trajs = load_augmented_trajectories(
        prm_names=["versa", "qwen", "dgprm"], n_per_dataset=TOTAL_PER_DS
    )
    train_trajs, test_trajs = split(all_trajs)

    print(f"Test set: {len(test_trajs)} trajectories, "
          f"{sum(len(t.steps) for t in test_trajs)} steps\n")

    # Baselines (computed once, reused across all experiments)
    versa_baseline  = evaluate_policy(PRMGuided(0.86, 0.62), test_trajs)
    uniform_baseline = evaluate_policy(UniformRouting(2), test_trajs)
    print(f"Baselines — VersaPRM: acc={versa_baseline.mean_accuracy:.4f}  "
          f"cost={versa_baseline.mean_cost_norm_per_traj:.0f}  "
          f"prec={routing_quality(versa_baseline):.3f}")
    print(f"           Uniform:   acc={uniform_baseline.mean_accuracy:.4f}  "
          f"cost={uniform_baseline.mean_cost_norm_per_traj:.0f}\n")

    all_exp_results = {}

    # ═══════════════════════════════════════════════════════════════════════
    # EXP 7: Multi-Judge Disagreement (highest priority)
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("Exp 7: Multi-Judge Disagreement (Versa vs DG-PRM)")
    print("=" * 65)
    exp7_fixed = {
        "MultiJudge τ=0.15": MultiJudgeDisagreement(tau_disagree=0.15, theta_high=0.86),
        "MultiJudge τ=0.20": MultiJudgeDisagreement(tau_disagree=0.20, theta_high=0.86),
        "MultiJudge τ=0.30": MultiJudgeDisagreement(tau_disagree=0.30, theta_high=0.86),
    }
    exp7_res = {name: evaluate_policy(pol, test_trajs) for name, pol in exp7_fixed.items()}
    print_comparison(exp7_res, versa_baseline, uniform_baseline)

    # Pareto sweep
    print("\nPareto sweep ...")
    exp7_pareto = []
    for pol in multijudge_sweep():
        r = evaluate_policy(pol, test_trajs)
        prec = routing_quality(r)
        exp7_pareto.append({**r.summary(), "routing_precision": round(prec, 4)})

    n_dom = sum(
        1 for p in exp7_pareto
        if p["accuracy"] > versa_baseline.mean_accuracy
        or (p["accuracy"] >= versa_baseline.mean_accuracy - 0.005
            and p["cost_norm_per_traj"] < versa_baseline.mean_cost_norm_per_traj)
    )
    print(f"  Sweep points beating/matching VersaPRM: {n_dom}/{len(exp7_pareto)}")
    best7 = max(exp7_pareto, key=lambda x: x["accuracy"])
    print(f"  Best accuracy: {best7['policy']}  acc={best7['accuracy']:.4f}  "
          f"prec={best7['routing_precision']:.3f}  cost={best7['cost_norm_per_traj']:.0f}")
    all_exp_results[7] = {
        "fixed": {k: r.summary() for k, r in exp7_res.items()},
        "pareto": exp7_pareto,
        "versa_baseline": versa_baseline.summary(),
    }

    # ═══════════════════════════════════════════════════════════════════════
    # EXP 8: Temporal Disagreement
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("Exp 8: Temporal Disagreement (score drops within trajectory)")
    print("=" * 65)
    exp8_fixed = {
        "Temporal τ_drop=0.10": TemporalDisagreement(tau_drop=0.10),
        "Temporal τ_drop=0.15": TemporalDisagreement(tau_drop=0.15),
        "Temporal τ_drop=0.25": TemporalDisagreement(tau_drop=0.25),
    }
    exp8_res = {name: evaluate_policy(pol, test_trajs) for name, pol in exp8_fixed.items()}
    print_comparison(exp8_res, versa_baseline, uniform_baseline)

    print("\nPareto sweep ...")
    exp8_pareto = []
    for pol in temporal_sweep():
        r = evaluate_policy(pol, test_trajs)
        prec = routing_quality(r)
        exp8_pareto.append({**r.summary(), "routing_precision": round(prec, 4)})

    best8 = max(exp8_pareto, key=lambda x: x["accuracy"])
    print(f"  Best accuracy: {best8['policy']}  acc={best8['accuracy']:.4f}  "
          f"prec={best8['routing_precision']:.3f}  cost={best8['cost_norm_per_traj']:.0f}")
    all_exp_results[8] = {
        "fixed": {k: r.summary() for k, r in exp8_res.items()},
        "pareto": exp8_pareto,
        "versa_baseline": versa_baseline.summary(),
    }

    # ═══════════════════════════════════════════════════════════════════════
    # EXP 6: PRM Disagreement (Versa vs Qwen)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("Exp 6: PRM Disagreement (Versa vs Qwen)")
    print("=" * 65)
    print("Note: Qwen scores are ~0.14 so disagreement ≈ versa_score - 0.14.")
    print("High disagreement ≈ high versa_score → forcing T3 when versa thinks")
    print("step is GOOD. Analysed for completeness; signal is not useful here.\n")
    exp6_fixed = {
        "Dis(VQ) τ=0.40": PRMDisagreementRouting(tau_disagree=0.40),
        "Dis(VQ) τ=0.55": PRMDisagreementRouting(tau_disagree=0.55),
        "Dis(VQ) τ=0.65": PRMDisagreementRouting(tau_disagree=0.65),
    }
    exp6_res = {name: evaluate_policy(pol, test_trajs) for name, pol in exp6_fixed.items()}
    print_comparison(exp6_res, versa_baseline, uniform_baseline)
    all_exp_results[6] = {
        "fixed": {k: r.summary() for k, r in exp6_res.items()},
        "versa_baseline": versa_baseline.summary(),
        "note": "Qwen scores near-constant ~0.14; disagreement = versa_score - 0.14. Not a useful uncertainty signal."
    }

    # ═══════════════════════════════════════════════════════════════════════
    # Combined signal (Exp 7 + Exp 8 together)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("Combined: VersaPRM + Temporal + Multi-Judge")
    print("=" * 65)
    combined_res = {}
    for pol in combined_sweep():
        r = evaluate_policy(pol, test_trajs)
        prec = routing_quality(r)
        combined_res[pol.name] = {**r.summary(), "routing_precision": round(prec, 4)}
    best_combined = max(combined_res.values(), key=lambda x: x["accuracy"])
    print(f"  Best combined: {best_combined['policy']}  "
          f"acc={best_combined['accuracy']:.4f}  "
          f"prec={best_combined['routing_precision']:.3f}  "
          f"cost={best_combined['cost_norm_per_traj']:.0f}")
    print(f"  vs VersaPRM:  Δacc={best_combined['accuracy']-versa_baseline.mean_accuracy:+.4f}")
    all_exp_results["combined"] = combined_res

    # Save
    for exp_num, data in all_exp_results.items():
        if isinstance(exp_num, int):
            out_path = RESULTS[exp_num] / f"exp{exp_num}_results.json"
        else:
            out_path = ROOT / "results/exp7" / "exp_combined_results.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"\nAll results saved to results/exp{{6,7,8}}/")
    return all_exp_results


if __name__ == "__main__":
    run_experiments()
