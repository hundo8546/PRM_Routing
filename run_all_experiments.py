"""
Master runner: execute all experiments in dependency order.

Run from /workspace/PRM_Routing/:
    python3 run_all_experiments.py [--start N] [--only N [N ...]]

Exp dependencies:
  Exp 1  (PRM transfer)         — standalone, uses raw PRM score files
  Exp 2  (routing comparison)   — requires Exp 1 score files
  Exp 3  (PRM ablation)         — requires Exp 1 score files
  Exp 4  (domain transfer)      — requires Exp 1 score files
  Exp 5  (step compression)     — requires Exp 1 score files + VersaPRM model
  Exp 6-8 (disagreement)        — requires Exp 1 multi-PRM score files
  Exp 9-10 (input compression)  — requires Exp 1 score files + VersaPRM model
  Exp 11 (frontier judge)       — requires Exp 1 score files; Exp 10 scores optional
  Exp 12 (modern stack)         — requires Exp 1 score files
  Exp 13 (dynamic thresholds)   — requires Exp 1 score files
  Exp 14 (uncertainty-adaptive) — requires Exp 1 multi-PRM score files
"""

import sys
import time
import argparse
import traceback
from pathlib import Path

ROOT = Path(__file__).parent

EXPERIMENTS = {
    1:  ("Exp 1:  PRM Transfer",              "experiments/exp1_transfer/exp1_analysis.py"),
    2:  ("Exp 2:  Routing Comparison",         "experiments/exp2_routing/run_exp2.py"),
    3:  ("Exp 3:  PRM Ablation",               "experiments/exp3_ablation/run_exp3.py"),
    4:  ("Exp 4:  Domain Generalization",      "experiments/exp4_generalization/run_exp4.py"),
    5:  ("Exp 5:  Step Compression",           "experiments/exp5678_additional/run_exp5.py"),
    678:("Exp 6-8: Disagreement Routing",      "experiments/exp5678_additional/run_exp678.py"),
    910:("Exp 9-10: Input Compression",        "experiments/exp910_compression/run_exp910.py"),
    11: ("Exp 11: Frontier Judge",             "experiments/exp11_frontier_judge/run_exp11.py"),
    12: ("Exp 12: Modern Stack",               "experiments/exp12_modern_stack/run_exp12.py"),
    13: ("Exp 13: Dynamic Thresholds",         "experiments/exp13_dynamic_thresholds/run_exp13.py"),
    14: ("Exp 14: Uncertainty-Adaptive",       "experiments/exp14_uncertainty_adaptive/run_exp14.py"),
}

# Experiments that require the VersaPRM model weights to be loaded
MODEL_REQUIRED = {5, 910}


def run_experiment(exp_id: int, label: str, script: str) -> bool:
    """Import and run a single experiment script. Returns True on success."""
    import importlib.util

    path = ROOT / script
    if not path.exists():
        print(f"  [SKIP] {label} — script not found: {path}")
        return False

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    t0 = time.time()

    # Mirror what `python3 script.py` does: put the script's directory first
    script_dir = str(path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    spec = importlib.util.spec_from_file_location(f"exp_{exp_id}", path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        if hasattr(mod, "run"):
            mod.run()
        elif hasattr(mod, "run_experiments"):
            mod.run_experiments()
        elapsed = time.time() - t0
        print(f"\n  [DONE] {label}  ({elapsed:.1f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  [FAIL] {label}  ({elapsed:.1f}s)")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all PRM routing experiments.")
    parser.add_argument("--start", type=int, default=None,
                        help="Skip experiments with id < START")
    parser.add_argument("--only", type=int, nargs="+", default=None,
                        help="Run only these experiment ids")
    parser.add_argument("--skip-model", action="store_true",
                        help=f"Skip experiments requiring model weights: {MODEL_REQUIRED}")
    args = parser.parse_args()

    to_run = list(EXPERIMENTS.keys())
    if args.only:
        to_run = [k for k in to_run if k in args.only]
    elif args.start is not None:
        to_run = [k for k in to_run if k >= args.start]
    if args.skip_model:
        to_run = [k for k in to_run if k not in MODEL_REQUIRED]

    print(f"Running {len(to_run)} experiment(s): {to_run}")

    results = {}
    total_start = time.time()
    for exp_id in to_run:
        label, script = EXPERIMENTS[exp_id]
        ok = run_experiment(exp_id, label, script)
        results[exp_id] = "OK" if ok else "FAIL"

    elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"SUMMARY  ({elapsed:.1f}s total)")
    print(f"{'='*70}")
    for exp_id, status in results.items():
        label = EXPERIMENTS[exp_id][0]
        print(f"  {'✓' if status == 'OK' else '✗'}  {label}  [{status}]")


if __name__ == "__main__":
    main()
