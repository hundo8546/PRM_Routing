"""
Experiment 3: PRM Variant Ablation.

Fixed routing policy (PRMGuided); variable: which PRM provides the routing signal.
Thresholds are percentile-calibrated per PRM so all produce comparable tier splits,
isolating signal *quality* from score *calibration*.

PRMs compared:
  versa   — VersaPRM          (multi-domain PRM, best Exp1 signal)
  qwen    — Qwen2.5-Math-PRM  (math specialist, baseline)
  dgprm   — DG-PRM local      (LLM-judge, moderate signal)
  agent   — AgentRM backbone  (constant output, degenerate signal)
  random  — random uniform [0,1] (zero-signal control)
  oracle  — human labels as signal (perfect-PRM upper bound)

Outputs:
  results/exp3/exp3_results.json
  results/exp3/exp3_summary.txt
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_routing"))

from data_loader import load_trajectories, DATASETS, EXP1_DIR, Trajectory, StepRecord
from routing_policies import PRMGuided
from simulator import evaluate_policy, PolicyResult
from cost_model import TIERS

RESULTS_DIR = Path("/workspace/PRM_Routing/results/exp3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PER_DS = 200
TEST_PER_DS  = 50
TOTAL_PER_DS = 250

PRM_DISPLAY = {
    "versa":  "VersaPRM",
    "qwen":   "Qwen2.5-Math-PRM",
    "dgprm":  "DG-PRM (local)",
    "agent":  "AgentRM (proxy)",
    "random": "Random (control)",
    "oracle": "Oracle (upper bound)",
}


# ---------------------------------------------------------------------------
# Score distribution helpers
# ---------------------------------------------------------------------------

def score_percentiles(prm_name: str, n_per_dataset: int = TOTAL_PER_DS):
    """Return (p25, p75) of the PRM score distribution on the full dataset."""
    scores = []
    path = EXP1_DIR / f"full_{prm_name}.jsonl"
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["reward_score"] is not None:
                scores.append(r["reward_score"])
    scores = np.array(scores)
    return float(np.percentile(scores, 25)), float(np.percentile(scores, 75))


def exp1_spearman(prm_name: str):
    """Spearman correlation between PRM scores and human labels (from Exp 1)."""
    scores, labels = [], []
    path = EXP1_DIR / f"full_{prm_name}.jsonl"
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["reward_score"] is not None:
                scores.append(r["reward_score"])
                labels.append(r["human_step_label"])
    r, _ = stats.spearmanr(scores, labels)
    return float(r)


# ---------------------------------------------------------------------------
# Special trajectory builders for Random and Oracle
# ---------------------------------------------------------------------------

def inject_scores(trajectories: list, score_fn) -> list:
    """Return copies of trajectories with versa_score replaced by score_fn(step)."""
    import copy
    out = []
    for traj in trajectories:
        t2 = copy.deepcopy(traj)
        for step in t2.steps:
            step.versa_score = score_fn(step)
        out.append(t2)
    return out


def make_random_trajectories(trajectories: list, seed: int = 42) -> list:
    """Inject random uniform [0,1] scores."""
    rng = np.random.default_rng(seed)
    scores_flat = rng.uniform(0, 1, sum(len(t.steps) for t in trajectories))
    idx = 0
    import copy
    out = []
    for traj in trajectories:
        t2 = copy.deepcopy(traj)
        for step in t2.steps:
            step.versa_score = float(scores_flat[idx])
            idx += 1
        out.append(t2)
    return out


def make_oracle_trajectories(trajectories: list) -> list:
    """
    Future-Oracle: perfect zero-latency signal — each step is scored based on
    its OWN label (not the previous step's). This gives the theoretical upper
    bound: a PRM that can perfectly assess the quality of the step it is about
    to execute before selecting the model tier.

    Implementation: shift oracle scores one position forward so that step i's
    routing decision uses step i's own label (by pretending the step before
    it already has that score). This is equivalent to a policy that directly
    maps label→tier without the one-step causal delay.
    """
    import copy
    out = []
    for traj in trajectories:
        t2 = copy.deepcopy(traj)
        # Shift: set step[i].versa_score = step[i+1].human_label so that
        # PRMGuided.decide uses the correct label for the upcoming step.
        for i in range(len(t2.steps) - 1):
            next_label = t2.steps[i + 1].human_label
            t2.steps[i].versa_score = 1.0 if next_label == 1 else 0.0
        # Last step: no look-ahead possible, use its own label
        if t2.steps:
            t2.steps[-1].versa_score = 1.0 if t2.steps[-1].human_label == 1 else 0.0
        out.append(t2)
    return out


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split(trajectories, train_per_ds=TRAIN_PER_DS):
    ds_map = defaultdict(list)
    for t in trajectories:
        ds_map[t.dataset].append(t)
    train, test = [], []
    for trajs in ds_map.values():
        train.extend(trajs[:train_per_ds])
        test.extend(trajs[train_per_ds:])
    return train, test


# ---------------------------------------------------------------------------
# Routing quality metric
# ---------------------------------------------------------------------------

def routing_quality(result: PolicyResult, trajectories: list) -> dict:
    """
    Measures how well the routing signal separates steps:
      - Fraction of bad steps correctly escalated to T3 (true escalation rate)
      - Fraction of good steps correctly routed to T1 (true T1 hit rate)
      - Net routing gain vs uniform (T2)
    """
    all_step_results = [s for r in result.traj_results for s in r.step_results]
    bad_to_t3 = sum(1 for s in all_step_results if s.human_label == -1 and s.tier_selected == 3)
    n_bad     = sum(1 for s in all_step_results if s.human_label == -1)
    good_to_t1 = sum(1 for s in all_step_results if s.human_label == 1 and s.tier_selected == 1)
    n_good    = sum(1 for s in all_step_results if s.human_label == 1)

    return {
        "bad_to_t3_rate":  bad_to_t3 / n_bad   if n_bad   else 0.0,
        "good_to_t1_rate": good_to_t1 / n_good if n_good else 0.0,
        "routing_precision": bad_to_t3 / max(1, sum(1 for s in all_step_results if s.tier_selected == 3)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print(f"\n{'='*70}")
    print("Experiment 3: PRM Variant Ablation")
    print(f"{'='*70}")
    print("Fixed policy: PRMGuided (percentile-calibrated thresholds per PRM)")
    print(f"Train: {TRAIN_PER_DS}/ds | Test: {TEST_PER_DS}/ds\n")

    # Load VersaPRM trajectories as the base (for Random/Oracle, scores are overwritten)
    all_trajs = load_trajectories("versa", TOTAL_PER_DS)
    train_trajs, test_trajs = split(all_trajs)

    results_table = []

    # --- Real PRMs ---
    for prm_name in ["versa", "qwen", "dgprm", "agent"]:
        print(f"  Running {PRM_DISPLAY[prm_name]} ...")
        p25, p75 = score_percentiles(prm_name)
        # Reload trajectories with this PRM's actual scores
        trajs = load_trajectories(prm_name, TOTAL_PER_DS)
        _, test = split(trajs)

        policy = PRMGuided(theta_high=p75, theta_low=p25)
        result = evaluate_policy(policy, test)
        rq = routing_quality(result, test)
        sp = exp1_spearman(prm_name)

        row = {
            "prm": prm_name,
            "display": PRM_DISPLAY[prm_name],
            "theta_high": round(p75, 3),
            "theta_low":  round(p25, 3),
            "exp1_spearman": round(sp, 4),
            **result.summary(),
            **{f"rq_{k}": round(v, 4) for k, v in rq.items()},
        }
        results_table.append(row)
        print(f"    θ=[{p25:.3f},{p75:.3f}]  acc={result.mean_accuracy:.4f}  "
              f"cost_norm={result.mean_cost_norm_per_traj:.0f}  "
              f"esc={result.escalation_rate:.3f}  spearman_r={sp:.4f}")

    # --- Random control ---
    print(f"  Running {PRM_DISPLAY['random']} ...")
    rand_test = make_random_trajectories(test_trajs)
    rand_policy = PRMGuided(theta_high=0.75, theta_low=0.25)  # p75/p25 of Uniform[0,1]
    rand_result = evaluate_policy(rand_policy, rand_test)
    rand_rq = routing_quality(rand_result, rand_test)
    results_table.append({
        "prm": "random",
        "display": PRM_DISPLAY["random"],
        "theta_high": 0.75, "theta_low": 0.25,
        "exp1_spearman": 0.0,
        **rand_result.summary(),
        **{f"rq_{k}": round(v, 4) for k, v in rand_rq.items()},
    })
    print(f"    acc={rand_result.mean_accuracy:.4f}  cost_norm={rand_result.mean_cost_norm_per_traj:.0f}  esc={rand_result.escalation_rate:.3f}")

    # --- Oracle upper bound ---
    print(f"  Running {PRM_DISPLAY['oracle']} ...")
    oracle_test = make_oracle_trajectories(test_trajs)
    oracle_policy = PRMGuided(theta_high=0.75, theta_low=0.25)
    oracle_result = evaluate_policy(oracle_policy, oracle_test)
    oracle_rq = routing_quality(oracle_result, oracle_test)
    results_table.append({
        "prm": "oracle",
        "display": PRM_DISPLAY["oracle"],
        "theta_high": 0.75, "theta_low": 0.25,
        "exp1_spearman": 1.0,
        **oracle_result.summary(),
        **{f"rq_{k}": round(v, 4) for k, v in oracle_rq.items()},
    })
    print(f"    acc={oracle_result.mean_accuracy:.4f}  cost_norm={oracle_result.mean_cost_norm_per_traj:.0f}  esc={oracle_result.escalation_rate:.3f}")

    # --- Print full table ---
    print(f"\n{'='*110}")
    print("EXPERIMENT 3 RESULTS")
    print(f"{'='*110}")
    hdr = (f"{'PRM':<24} {'θ_h':>5} {'θ_l':>5} {'Spearman':>9} "
           f"{'TSR':>7} {'Acc':>7} {'Cost(N)':>9} {'EscRate':>8} {'AvgTier':>8} "
           f"{'BadToT3':>8} {'GoodToT1':>9} {'Prec':>6}")
    print(hdr); print("-"*len(hdr))
    for r in results_table:
        print(
            f"{r['display']:<24} {r['theta_high']:>5.3f} {r['theta_low']:>5.3f} "
            f"{r['exp1_spearman']:>+9.4f} "
            f"{r.get('task_success_rate', 0):>7.4f} "
            f"{r['accuracy']:>7.4f} {r['cost_norm_per_traj']:>9.0f} "
            f"{r['escalation_rate']:>8.3f} {r['avg_tier']:>8.3f} "
            f"{r.get('rq_bad_to_t3_rate',0):>8.3f} {r.get('rq_good_to_t1_rate',0):>9.3f} "
            f"{r.get('rq_routing_precision',0):>6.3f}"
        )

    # --- Oracle gap analysis ---
    print()
    oracle_acc = next(r['accuracy'] for r in results_table if r['prm'] == 'oracle')
    random_acc = next(r['accuracy'] for r in results_table if r['prm'] == 'random')
    oracle_cost = next(r['cost_norm_per_traj'] for r in results_table if r['prm'] == 'oracle')
    random_cost = next(r['cost_norm_per_traj'] for r in results_table if r['prm'] == 'random')
    print(f"Oracle gap: acc [{random_acc:.4f} → {oracle_acc:.4f}], cost [{random_cost:.0f} → {oracle_cost:.0f}]")
    print()
    print("Fraction of Oracle gap closed (accuracy):")
    for r in results_table:
        if r['prm'] in ('random', 'oracle'):
            continue
        gap_closed = (r['accuracy'] - random_acc) / (oracle_acc - random_acc) if (oracle_acc - random_acc) != 0 else 0
        print(f"  {r['display']:<24}  acc={r['accuracy']:.4f}  gap_closed={gap_closed:+.1%}")

    # --- Save ---
    out = {"results": results_table}
    with open(RESULTS_DIR / "exp3_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Human-readable summary
    lines = ["Experiment 3: PRM Variant Ablation", "", hdr, "-"*len(hdr)]
    for r in results_table:
        lines.append(
            f"{r['display']:<24} {r['theta_high']:>5.3f} {r['theta_low']:>5.3f} "
            f"{r['exp1_spearman']:>+9.4f} "
            f"{r.get('task_success_rate', 0):>7.4f} "
            f"{r['accuracy']:>7.4f} {r['cost_norm_per_traj']:>9.0f} "
            f"{r['escalation_rate']:>8.3f} {r['avg_tier']:>8.3f} "
            f"{r.get('rq_bad_to_t3_rate',0):>8.3f} {r.get('rq_good_to_t1_rate',0):>9.3f} "
            f"{r.get('rq_routing_precision',0):>6.3f}"
        )
    with open(RESULTS_DIR / "exp3_summary.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {RESULTS_DIR}")
    return results_table


if __name__ == "__main__":
    run()
