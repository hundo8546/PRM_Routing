"""
Experiment 15: Retrieval-Aware PRM + Uncertainty-Adaptive Routing (UA-3).

Research question: Does combining retrieval-enriched VersaPRM scores (Exp 10a)
with UA-3 uncertainty-adaptive thresholds produce the strongest routing
configuration?

Hypothesis: Retrieval context improves the routing signal (Spearman 0.289 ->
0.307), and UA-3 threshold adaptation exploits disagreement between VersaPRM
and DG-PRM. The combination should outperform either alone.

Configurations compared:
  - Fixed PRM-Guided (Exp 2 baseline)          TSR=0.320
  - UA-3 only (Exp 14 best cheap variant)       TSR=0.385
  - Retrieval-Aware PRM (Exp 10a, routing acc)  acc=0.413 (TSR unknown)
  - Retrieval-Aware + UA-3 (THIS EXP)           TSR=?

Score source (in priority order):
  1. results/exp10/exp10_retr_scores.jsonl  -- per-step retr_full scores
     (produced by re-running exp910 with --save-scores flag once
      VersaPRM model weights are loaded)
  2. Fallback: standard exp1 VersaPRM scores
     (gives a lower bound; marks results as [PROXY] in output)

Requires augmented trajectories with DG-PRM scores for disagreement signal.

Outputs:
  results/exp15/exp15_results.json
  results/exp15/exp15_summary.txt
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path("/workspace/PRM_Routing")
sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_routing"))
sys.path.insert(0, str(Path(__file__).parent.parent / "exp5678_additional"))

from data_loader import load_trajectories, DATASETS
from routing_policies import PRMGuided, RoutingPolicy
from simulator import evaluate_policy, SUCCESS_THRESHOLD
from disagreement_loader import load_augmented_trajectories

import copy

RESULTS_DIR = ROOT / "results/exp15"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_PER_DS = 250
TRAIN_PER_DS = 200

AGGRESSIVE = (0.55, 0.90)
STANDARD   = (0.62, 0.86)
CAUTIOUS   = (0.75, 0.80)

RETR_SCORES_PATH = ROOT / "results/exp10/exp10_retr_scores.jsonl"


# ---------------------------------------------------------------------------
# UA-3 routing policy (same as Exp 14)
# ---------------------------------------------------------------------------

class UncertaintyAdaptiveRouting(RoutingPolicy):
    def __init__(self, tau_low=0.15, tau_high=0.30):
        self.tau_low  = tau_low
        self.tau_high = tau_high
        self.name = f"UA-3(tau={tau_low}/{tau_high})"

    def decide(self, traj, step_idx):
        if step_idx == 0:
            return 2
        prev = traj.steps[step_idx - 1]
        v = prev.versa_score
        d = getattr(prev, "extra_scores", {}).get("dgprm", v)
        disagreement = abs(v - d)
        if   disagreement < self.tau_low:  theta_low, theta_high = AGGRESSIVE
        elif disagreement < self.tau_high: theta_low, theta_high = STANDARD
        else:                              theta_low, theta_high = CAUTIOUS
        if   v > theta_high: return 1
        elif v < theta_low:  return 3
        else:                return 2

    def fit(self, _): pass


# ---------------------------------------------------------------------------
# Load retrieval-aware scores (Exp 10a output)
# ---------------------------------------------------------------------------

def load_retr_scores() -> tuple[dict, bool]:
    """
    Returns (score_map, is_real).
    score_map: (dataset, global_traj_idx, msg_idx) -> float
    is_real: True if real retr_full scores loaded, False if using exp1 fallback.
    """
    if RETR_SCORES_PATH.exists():
        score_map = {}
        with open(RETR_SCORES_PATH) as f:
            for line in f:
                r = json.loads(line)
                key = (r["dataset"], r["traj_idx"], r["msg_idx"])
                score_map[key] = float(r["retr_full_score"])
        print(f"  Loaded {len(score_map)} retrieval-aware scores from {RETR_SCORES_PATH.name}")
        return score_map, True
    else:
        print(f"  WARNING: {RETR_SCORES_PATH.name} not found.")
        print(f"  Running in PROXY mode using standard exp1 VersaPRM scores.")
        print(f"  To get real retrieval-aware scores, install transformers and run:")
        print(f"    python3 experiments/exp910_compression/run_exp910.py --save-scores")
        return {}, False


def inject_retr_scores(trajectories, score_map):
    """Replace versa_score with retrieval-aware score where available."""
    ds_offset = {ds: i * 250 for i, ds in enumerate(DATASETS)}
    out = []
    hit = miss = 0
    for traj in trajectories:
        t2 = copy.deepcopy(traj)
        for step in t2.steps:
            g   = ds_offset.get(step.dataset, 0) + step.traj_idx
            key = (step.dataset, g, step.msg_idx)
            if key in score_map:
                step.versa_score = score_map[key]
                hit += 1
            else:
                miss += 1
        out.append(t2)
    if score_map:
        print(f"  Score injection: {hit} hits, {miss} misses "
              f"({hit/(hit+miss):.1%} coverage)")
    return out


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


def routing_quality(result):
    all_steps = [s for r in result.traj_results for s in r.step_results]
    n_esc     = sum(1 for s in all_steps if s.tier_selected == 3)
    bad_t3    = sum(1 for s in all_steps if s.human_label == -1 and s.tier_selected == 3)
    n_bad     = sum(1 for s in all_steps if s.human_label == -1)
    return {
        "routing_precision": bad_t3 / n_esc if n_esc else 0.0,
        "bad_to_t3_rate":    bad_t3 / n_bad if n_bad else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print(f"\n{'='*70}")
    print("Experiment 15: Retrieval-Aware PRM + UA-3 Routing")
    print(f"{'='*70}\n")

    # Load retrieval-aware scores
    print("Loading retrieval-aware scores ...")
    retr_score_map, is_real_retr = load_retr_scores()
    mode_tag = "" if is_real_retr else " [PROXY]"
    print()

    # Load augmented trajectories (VersaPRM + DG-PRM for disagreement)
    print("Loading augmented trajectories (VersaPRM + DG-PRM) ...")
    all_trajs = load_augmented_trajectories(
        prm_names=["versa", "dgprm"], n_per_dataset=TOTAL_PER_DS
    )
    _, test_trajs = split(all_trajs)
    print(f"Test trajectories: {len(test_trajs)}\n")

    # Build retrieval-injected trajectories
    retr_test_trajs = inject_retr_scores(test_trajs, retr_score_map)

    # ── Policies ──────────────────────────────────────────────────────────────
    configs = [
        ("Fixed PRM-Guided",            PRMGuided(0.86, 0.62), test_trajs,      "Exp2 baseline"),
        ("UA-3 only",                   UncertaintyAdaptiveRouting(), test_trajs, "Exp14 best"),
        (f"Retr-Aware PRM{mode_tag}",   PRMGuided(0.86, 0.62), retr_test_trajs, "Exp10a routing"),
        (f"Retr-Aware + UA-3{mode_tag}",UncertaintyAdaptiveRouting(), retr_test_trajs, "THIS EXP"),
    ]

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 15 RESULTS{mode_tag}")
    print(f"{'='*80}")
    hdr = (f"{'Configuration':<30} {'TSR':>7} {'ΔvFixed':>8} {'Acc':>7} "
           f"{'CostN':>9} {'Prec':>6} {'EscRate':>8}")
    print(hdr); print("-"*len(hdr))

    base_tsr = base_cost = None
    results = []

    for name, pol, trajs, note in configs:
        r  = evaluate_policy(pol, trajs)
        rq = routing_quality(r)
        if base_tsr is None:
            base_tsr  = r.task_success_rate
            base_cost = r.mean_cost_norm_per_traj

        dtsr  = r.task_success_rate - base_tsr
        dcost = (r.mean_cost_norm_per_traj - base_cost) / base_cost if base_cost else 0
        print(f"{name:<30} {r.task_success_rate:>7.4f} {dtsr:>+8.4f} "
              f"{r.mean_accuracy:>7.4f} {r.mean_cost_norm_per_traj:>9.0f} "
              f"{rq['routing_precision']:>6.3f} {r.escalation_rate:>8.3f}  # {note}")
        results.append({
            "config": name,
            "note":   note,
            "proxy":  not is_real_retr,
            **r.summary(),
            "routing_precision": round(rq["routing_precision"], 4),
            "delta_tsr":  round(dtsr, 4),
            "delta_cost": round(dcost, 4),
        })

    # ── Success criterion ────────────────────────────────────────────────────
    retr_ua3 = next(r for r in results if "UA-3" in r["config"] and "Retr" in r["config"])
    ua3_only = next(r for r in results if r["config"] == "UA-3 only")
    retr_only = next(r for r in results if "Retr-Aware PRM" in r["config"] and "UA-3" not in r["config"])

    print(f"\n[Combination Effect]")
    print(f"  UA-3 alone:          TSR={ua3_only['task_success_rate']:.4f}")
    print(f"  Retrieval alone{mode_tag}:  TSR={retr_only['task_success_rate']:.4f}")
    print(f"  Retrieval + UA-3{mode_tag}: TSR={retr_ua3['task_success_rate']:.4f}  "
          f"(delta vs Fixed={retr_ua3['delta_tsr']:+.4f})")

    if not is_real_retr:
        print(f"\n  [NOTE] Running in PROXY mode — retrieval scores = standard exp1 VersaPRM.")
        print(f"  Retr-Aware rows show UA-3 applied to unmodified versa scores.")
        print(f"  Re-run after: pip install transformers torch && "
              f"python3 experiments/exp910_compression/run_exp910.py --save-scores")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = {
        "mode":           "real" if is_real_retr else "proxy",
        "retr_score_path": str(RETR_SCORES_PATH),
        "results":        results,
        "combination_effect": {
            "ua3_tsr":       ua3_only["task_success_rate"],
            "retr_tsr":      retr_only["task_success_rate"],
            "combined_tsr":  retr_ua3["task_success_rate"],
            "delta_vs_fixed": retr_ua3["delta_tsr"],
        },
    }
    with open(RESULTS_DIR / "exp15_results.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(RESULTS_DIR / "exp15_summary.txt", "w") as f:
        f.write(f"Experiment 15: Retrieval-Aware PRM + UA-3 [{out['mode'].upper()} mode]\n\n")
        f.write(hdr + "\n" + "-"*len(hdr) + "\n")
        for r in results:
            f.write(f"{r['config']:<30} {r['task_success_rate']:>7.4f} {r['delta_tsr']:>+8.4f} "
                    f"{r['accuracy']:>7.4f} {r['cost_norm_per_traj']:>9.0f} "
                    f"{r['routing_precision']:>6.3f} {r['escalation_rate']:>8.3f}\n")

    print(f"\nResults saved to {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    run()
