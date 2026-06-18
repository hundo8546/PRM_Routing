"""
Multi-PRM data loader for disagreement experiments (Exps 6, 7, 8).

Merges VersaPRM, QwenPRM, and DG-PRM scores with trajectory data.
Returns augmented trajectories where each step carries all three scores.
"""

import json
import copy
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

ROOT = Path("/workspace/PRM_Routing")
EXP1_DIR = ROOT / "results/exp1"
DATASETS = ["hotpotqa", "gaia_dev", "bfcl", "tau2"]


def _build_offset():
    return {ds: i * 250 for i, ds in enumerate(DATASETS)}


DS_OFFSET = _build_offset()


def load_multi_scores(
    prm_names: List[str] = ("versa", "qwen", "dgprm"),
    n_per_dataset: int = 250,
) -> Dict[tuple, Dict[str, float]]:
    """
    Load scores for multiple PRMs and return a joint score map.

    key: (dataset, global_traj_idx, msg_idx)
    value: {prm_name: score}
    """
    joint: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    for prm in prm_names:
        path = EXP1_DIR / f"full_{prm}.jsonl"
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["reward_score"] is None:
                    continue
                key = (r["dataset"], r["traj_idx"], r["msg_idx"])
                joint[key][prm] = r["reward_score"]
    return dict(joint)


def load_augmented_trajectories(
    prm_names: List[str] = ("versa", "qwen", "dgprm"),
    n_per_dataset: int = 250,
    base_prm: str = "versa",
):
    """
    Load trajectories with multiple PRM scores attached to each step.

    Each step gets step.extra_scores = {prm: score} in addition to the
    standard step.versa_score field (which holds base_prm scores).

    Returns list of Trajectory objects (extended with extra_scores attribute).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_routing"))
    from data_loader import load_trajectories

    # Load base trajectories (with versa scores)
    trajs = load_trajectories(base_prm, n_per_dataset)

    # Load joint multi-PRM scores
    multi = load_multi_scores(prm_names, n_per_dataset)

    # Attach extra scores to each step
    for traj in trajs:
        for step in traj.steps:
            global_idx = DS_OFFSET[step.dataset] + step.traj_idx
            key = (step.dataset, global_idx, step.msg_idx)
            step.extra_scores = multi.get(key, {})

    n_with_all = sum(
        1 for t in trajs for s in t.steps
        if all(p in s.extra_scores for p in prm_names)
    )
    total = sum(len(t.steps) for t in trajs)
    print(f"Loaded {len(trajs)} trajectories, {total} steps "
          f"({n_with_all}/{total} have all {len(prm_names)} PRM scores)")
    return trajs
