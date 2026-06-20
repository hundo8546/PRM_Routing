"""
Data loader for Experiment 2.

Merges VersaPRM scores (from Exp 1 full run) with original AgentProcessBench
trajectory data to produce per-step records that include token estimates.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

ROOT = Path("/workspace/PRM_Routing")
DATA_DIR = ROOT / "benchmarks/AgentProcessBench/data/AgentProcessBench"
EXP1_DIR = ROOT / "results/exp1"

DATASETS = ["hotpotqa", "gaia_dev", "bfcl", "tau2"]


@dataclass
class StepRecord:
    traj_idx: int           # within-dataset index (0–249)
    global_traj_idx: int    # across all datasets
    dataset: str
    msg_idx: int            # message index in original trajectory
    step_position: int      # 0-based index among labeled steps in this traj
    n_steps_total: int      # total labeled steps in this trajectory
    step_type: str          # retrieval / tool_call / synthesis
    versa_score: float      # VersaPRM score (Exp 1)
    human_label: int        # 1 (good) or -1 (bad)
    final_success: int      # 1 = task success, 0 = failure
    input_tokens: int       # estimated context tokens at this step
    output_tokens: int      # estimated output tokens for this step
    question: str
    step_content: str = ""  # raw message content + tool call text (truncated to 600 chars)


@dataclass
class Trajectory:
    traj_idx: int
    global_traj_idx: int
    dataset: str
    steps: List[StepRecord] = field(default_factory=list)
    final_success: int = 0
    question: str = ""


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _step_tokens(msg: dict) -> int:
    """Estimate output tokens for a step (tool call args + content)."""
    content = msg.get("content", "") or ""
    tc = msg.get("tool_calls", "") or ""
    if isinstance(tc, list):
        tc_text = json.dumps(tc)
    else:
        tc_text = str(tc)
    return _estimate_tokens(content + tc_text)


def _context_tokens_at(msgs: list, up_to_idx: int) -> int:
    """Estimate cumulative context tokens at message index up_to_idx."""
    total = 0
    for i, msg in enumerate(msgs):
        if i >= up_to_idx:
            break
        content = msg.get("content", "") or ""
        tc = msg.get("tool_calls", "") or ""
        if isinstance(tc, list):
            tc = json.dumps(tc)
        total += _estimate_tokens(content + str(tc))
    return max(1, total)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_trajectories(prm_name: str = "versa", n_per_dataset: int = 250) -> List[Trajectory]:
    """
    Load trajectories with VersaPRM scores merged in.

    Args:
        prm_name: Which PRM's scores to use as the routing signal.
        n_per_dataset: How many trajectories per dataset to load.
    """
    # --- load Exp1 scores indexed by (dataset, global_traj_idx, msg_idx) ---
    # Exp1 stores traj_idx as a global sequential index across all datasets:
    #   hotpotqa 0-249 | gaia_dev 250-499 | bfcl 500-749 | tau2 750-999
    score_map: Dict[tuple, float] = {}
    score_file = EXP1_DIR / f"full_{prm_name}.jsonl"
    with open(score_file) as f:
        for line in f:
            r = json.loads(line)
            key = (r["dataset"], r["traj_idx"], r["msg_idx"])
            score_map[key] = r["reward_score"] if r["reward_score"] is not None else 0.5

    # Build dataset → global offset map (same order as Exp1 loader)
    ds_offset = {ds: i * 250 for i, ds in enumerate(DATASETS)}

    trajectories: List[Trajectory] = []
    global_idx = 0

    for ds in DATASETS:
        traj_file = DATA_DIR / f"{ds}.jsonl"
        with open(traj_file) as f:
            for local_idx, line in enumerate(f):
                if local_idx >= n_per_dataset:
                    break
                raw = json.loads(line)
                msgs = raw["messages"]
                labels = raw.get("step_labels", {})
                final_label = raw.get("final_label", None)
                final_success = 1 if final_label == 1 else 0
                question = raw.get("question", "")

                traj = Trajectory(
                    traj_idx=local_idx,
                    global_traj_idx=global_idx,
                    dataset=ds,
                    final_success=final_success,
                    question=question[:200],
                )

                # Sort labeled steps by message index
                sorted_steps = sorted(labels.items(), key=lambda x: int(x[0]))
                n_steps_total = len(sorted_steps)

                for pos, (idx_str, human_label) in enumerate(sorted_steps):
                    msg_idx = int(idx_str)
                    msg = msgs[msg_idx]

                    global_idx_for_lookup = ds_offset[ds] + local_idx
                    versa_score = score_map.get((ds, global_idx_for_lookup, msg_idx), 0.5)
                    input_tok = _context_tokens_at(msgs, msg_idx)
                    output_tok = _step_tokens(msg)

                    # Infer step type from message and build step_content for judge
                    tc = msg.get("tool_calls", "") or ""
                    content = msg.get("content", "") or ""
                    tc_text = ""
                    if tc and tc not in ("", "[]"):
                        try:
                            tc_parsed = json.loads(tc) if isinstance(tc, str) else tc
                            fn = tc_parsed[0].get("function", {}).get("name", "") if tc_parsed else ""
                            step_type = "retrieval" if ("search" in fn or "retriev" in fn) else "tool_call"
                            tc_text = json.dumps(tc_parsed)
                        except Exception:
                            step_type = "tool_call"
                            tc_text = str(tc)
                    elif content:
                        step_type = "synthesis"
                    else:
                        step_type = "unknown"
                    step_content = (content + " " + tc_text).strip()[:600]

                    step = StepRecord(
                        traj_idx=local_idx,
                        global_traj_idx=global_idx,
                        dataset=ds,
                        msg_idx=msg_idx,
                        step_position=pos,
                        n_steps_total=n_steps_total,
                        step_type=step_type,
                        versa_score=versa_score,
                        human_label=human_label,
                        final_success=final_success,
                        input_tokens=input_tok,
                        output_tokens=output_tok,
                        question=question[:100],
                        step_content=step_content,
                    )
                    traj.steps.append(step)

                trajectories.append(traj)
                global_idx += 1

    print(f"Loaded {len(trajectories)} trajectories, "
          f"{sum(len(t.steps) for t in trajectories)} labeled steps "
          f"(PRM signal: {prm_name})")
    return trajectories
