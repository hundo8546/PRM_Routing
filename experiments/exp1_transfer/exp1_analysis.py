"""
Experiment 1 Analysis: PRM Signal Evaluation on AgentProcessBench (full 1000 trajectories).

Produces:
  - Main results table (Acc, F1, Pearson, Spearman, Brier, ECE, Sep)
  - Per-step-type breakdown table
  - Per-dataset breakdown table
  - Correlation with final task success
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.calibration import calibration_curve

RESULTS_DIR = Path("/workspace/PRM_Routing/results/exp1")
PRM_LABELS = {
    "qwen":  "Qwen2.5-Math-PRM",
    "versa": "VersaPRM",
    "agent": "AgentRM*",
    "dgprm": "DG-PRM (local)",
}
PRMS = list(PRM_LABELS.keys())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load(prm: str, prefix: str = "full"):
    path = RESULTS_DIR / f"{prefix}_{prm}.jsonl"
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if r["reward_score"] is not None]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def brier_score(scores, labels_bin):
    return float(np.mean((np.array(scores) - np.array(labels_bin)) ** 2))


def expected_calibration_error(scores, labels_bin, n_bins=10):
    scores = np.array(scores)
    labels = np.array(labels_bin)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = scores[mask].mean()
        avg_acc = labels[mask].mean()
        ece += mask.sum() * abs(avg_conf - avg_acc)
    return float(ece / len(scores))


def compute_metrics(rows, threshold: float = 0.5):
    scores = np.array([r["reward_score"] for r in rows])
    labels = np.array([r["human_step_label"] for r in rows])   # 1 or -1
    labels_bin = (labels == 1).astype(int)

    preds = (scores >= threshold).astype(int)

    acc = accuracy_score(labels_bin, preds)
    prec = precision_score(labels_bin, preds, zero_division=0)
    rec = recall_score(labels_bin, preds, zero_division=0)
    f1 = f1_score(labels_bin, preds, zero_division=0)
    pearson_r, pearson_p = stats.pearsonr(scores, labels)
    spearman_r, spearman_p = stats.spearmanr(scores, labels)
    brier = brier_score(scores, labels_bin)
    ece = expected_calibration_error(scores, labels_bin)
    sep = scores[labels == 1].mean() - scores[labels == -1].mean()

    return {
        "n": len(rows),
        "n_good": int((labels == 1).sum()),
        "n_bad": int((labels == -1).sum()),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pearson": pearson_r,
        "pearson_p": pearson_p,
        "spearman": spearman_r,
        "spearman_p": spearman_p,
        "brier": brier,
        "ece": ece,
        "separation": sep,
    }


def task_success_correlation(rows):
    """Pearson correlation between mean trajectory score and final_task_success."""
    # Group by trajectory
    traj_scores = defaultdict(list)
    traj_success = {}
    for r in rows:
        key = (r["dataset"], r["traj_idx"])
        traj_scores[key].append(r["reward_score"])
        traj_success[key] = r["final_task_success"]

    mean_scores = [np.mean(v) for k, v in traj_scores.items()]
    successes = [traj_success[k] for k in traj_scores]
    if len(set(successes)) < 2:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(mean_scores, successes)
    return float(r), float(p)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def print_main_table(all_metrics):
    header = f"{'PRM':<22} {'N':>5} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Pearson':>9} {'Spearman':>9} {'Brier':>7} {'ECE':>7} {'Sep':>8}"
    print(header)
    print("-" * len(header))
    for prm, m in all_metrics.items():
        stars_p = sig_stars(m["pearson_p"])
        stars_s = sig_stars(m["spearman_p"])
        print(
            f"{PRM_LABELS[prm]:<22} {m['n']:>5} "
            f"{m['accuracy']:>6.3f} {m['f1']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['pearson']:>+7.4f}{stars_p:<2} {m['spearman']:>+7.4f}{stars_s:<2} "
            f"{m['brier']:>7.4f} {m['ece']:>7.4f} {m['separation']:>+8.4f}"
        )
    print("* p<0.05, ** p<0.01, *** p<0.001")


def print_steptype_table(all_rows):
    step_types = sorted({r["step_type"] for rows in all_rows.values() for r in rows})
    header_cols = ["PRM"] + step_types
    col_w = 14
    print("  ".join(f"{c:<{col_w}}" for c in header_cols))
    print("-" * (col_w * len(header_cols) + 2 * (len(header_cols) - 1)))
    for prm, rows in all_rows.items():
        row_parts = [f"{PRM_LABELS[prm]:<{col_w}}"]
        by_type = defaultdict(list)
        for r in rows:
            by_type[r["step_type"]].append(r)
        for st in step_types:
            if st not in by_type or len(by_type[st]) < 5:
                row_parts.append(f"{'—':<{col_w}}")
                continue
            m = compute_metrics(by_type[st])
            row_parts.append(f"r={m['spearman']:+.3f}  ".ljust(col_w))
        print("  ".join(row_parts))


def print_dataset_table(all_rows):
    datasets = ["hotpotqa", "gaia_dev", "bfcl", "tau2"]
    col_w = 14
    header_cols = ["PRM"] + [f"Sep/{ds}" for ds in datasets]
    print("  ".join(f"{c:<{col_w}}" for c in header_cols))
    print("-" * (col_w * len(header_cols) + 2 * (len(header_cols) - 1)))
    for prm, rows in all_rows.items():
        row_parts = [f"{PRM_LABELS[prm]:<{col_w}}"]
        by_ds = defaultdict(list)
        for r in rows:
            by_ds[r["dataset"]].append(r)
        for ds in datasets:
            if ds not in by_ds:
                row_parts.append(f"{'—':<{col_w}}")
                continue
            scores = np.array([r["reward_score"] for r in by_ds[ds]])
            labels = np.array([r["human_step_label"] for r in by_ds[ds]])
            sep = scores[labels==1].mean() - scores[labels==-1].mean() if (labels==1).any() and (labels==-1).any() else float("nan")
            row_parts.append(f"{sep:+.4f}       ".ljust(col_w))
        print("  ".join(row_parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="full", choices=["full", "pilot"],
                        help="Which result files to load (full_ or pilot_)")
    args = parser.parse_args()

    all_rows = {}
    all_metrics = {}

    for prm in PRMS:
        try:
            rows = load(prm, args.prefix)
        except FileNotFoundError:
            print(f"[SKIP] {prm}: no {args.prefix}_ results yet")
            continue
        all_rows[prm] = rows
        all_metrics[prm] = compute_metrics(rows)

    if not all_metrics:
        print("No results found. Run pilot_prm_scoring.py first.")
        raise SystemExit(1)

    print("\n" + "=" * 80)
    print("EXPERIMENT 1: PRM SIGNAL EVALUATION — AgentProcessBench")
    print("=" * 80)

    print(f"\n[Main Table] — {args.prefix} run, threshold=0.5")
    print_main_table(all_metrics)

    print(f"\n[Per-Step-Type] Spearman correlation by step type")
    print_steptype_table(all_rows)

    print(f"\n[Per-Dataset] Score separation (good − bad) by dataset")
    print_dataset_table(all_rows)

    print(f"\n[Task Success Correlation] Pearson(mean_traj_score, task_success)")
    for prm, rows in all_rows.items():
        r, p = task_success_correlation(rows)
        stars = sig_stars(p) if not np.isnan(p) else ""
        print(f"  {PRM_LABELS[prm]:<22}  r={r:+.4f}{stars}  (p={p:.4f})" if not np.isnan(r) else f"  {PRM_LABELS[prm]:<22}  r=N/A")

    print()
    # Save summary JSON
    summary = {}
    for prm, m in all_metrics.items():
        summary[prm] = {k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()}
        r, p = task_success_correlation(all_rows[prm])
        summary[prm]["task_success_pearson"] = round(float(r), 6) if not np.isnan(r) else None

    out = RESULTS_DIR / f"exp1_{args.prefix}_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out}")
