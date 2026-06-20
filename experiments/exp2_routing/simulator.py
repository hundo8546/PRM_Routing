"""
Simulation engine for Experiment 2.

Runs routing policies over trajectories and computes:
  - Expected task accuracy (counterfactual outcome model)
  - Total token cost (USD and normalised)
  - Routing statistics (escalation rate, avg tier, stability)

Outcome model (counterfactual simulation):
  Each step has a human label (1=good, -1=bad). Under tier T:
    - good step: succeeds with p_good_step_success[T]
    - bad  step: recovers with p_bad_step_recovery[T]
  Task success = all steps succeed (errors propagate per AgentProcessBench scheme).
  Expected task accuracy is computed analytically (product of per-step success probs).
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

from data_loader import Trajectory, StepRecord
from cost_model import TIERS
from routing_policies import RoutingPolicy

SUCCESS_THRESHOLD = 0.5   # expected_task_accuracy >= this → task counted as successful


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    traj_idx: int
    dataset: str
    step_position: int
    step_type: str
    tier_selected: int
    versa_score: float
    human_label: int
    p_step_success: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_norm: float


@dataclass
class TrajectoryResult:
    traj_idx: int
    dataset: str
    expected_task_accuracy: float   # product of per-step success probs
    total_cost_usd: float
    total_cost_norm: float
    step_results: List[StepResult] = field(default_factory=list)
    n_escalations: int = 0          # steps routed to Tier 3
    n_tier1: int = 0
    n_tier2: int = 0
    n_tier3: int = 0
    routing_changes: int = 0        # tier switches between consecutive steps


@dataclass
class PolicyResult:
    policy_name: str
    # Aggregate metrics
    mean_accuracy: float
    std_accuracy: float
    total_cost_usd: float
    mean_cost_usd_per_traj: float
    total_cost_norm: float
    mean_cost_norm_per_traj: float
    # Routing statistics
    escalation_rate: float          # fraction of steps routed to T3
    avg_tier: float
    routing_stability: float        # 1 - (tier changes / total steps)
    # Primary outcome metrics
    task_success_rate: float = 0.0          # fraction of trajs with expected_acc >= SUCCESS_THRESHOLD
    cost_per_successful_task: float = 0.0   # mean_cost_usd_per_traj / task_success_rate
    # Per-dataset breakdown
    per_dataset: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Raw trajectory results (for Pareto plotting)
    traj_results: List[TrajectoryResult] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "policy": self.policy_name,
            "task_success_rate": round(self.task_success_rate, 4),
            "cost_per_successful_task": round(self.cost_per_successful_task, 6),
            "accuracy": round(self.mean_accuracy, 4),
            "accuracy_std": round(self.std_accuracy, 4),
            "cost_usd_total": round(self.total_cost_usd, 6),
            "cost_usd_per_traj": round(self.mean_cost_usd_per_traj, 6),
            "cost_norm_per_traj": round(self.mean_cost_norm_per_traj, 2),
            "escalation_rate": round(self.escalation_rate, 4),
            "avg_tier": round(self.avg_tier, 3),
            "routing_stability": round(self.routing_stability, 4),
        }


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

def _p_step_success(tier_id: int, human_label: int, tiers: dict) -> float:
    """Expected probability that a step succeeds under this tier."""
    t = tiers[tier_id]
    return t.p_good_step_success if human_label == 1 else t.p_bad_step_recovery


def simulate_trajectory(
    traj: Trajectory,
    policy: RoutingPolicy,
    tiers: dict = None,
) -> TrajectoryResult:
    """Apply routing policy to a single trajectory and compute results."""
    if tiers is None:
        tiers = TIERS
    base_cost = tiers[1].input_cost_per_1m  # for relative cost normalisation

    step_results = []
    tiers_used = []

    for i, step in enumerate(traj.steps):
        tier_id = policy.decide(traj, i)
        p_success = _p_step_success(tier_id, step.human_label, tiers)
        t = tiers[tier_id]
        cost_usd = (step.input_tokens * t.input_cost_per_1m +
                    step.output_tokens * t.output_cost_per_1m) / 1_000_000
        cost_norm = (step.input_tokens + step.output_tokens) * (t.input_cost_per_1m / base_cost)

        step_results.append(StepResult(
            traj_idx=traj.traj_idx,
            dataset=traj.dataset,
            step_position=i,
            step_type=step.step_type,
            tier_selected=tier_id,
            versa_score=step.versa_score,
            human_label=step.human_label,
            p_step_success=p_success,
            input_tokens=step.input_tokens,
            output_tokens=step.output_tokens,
            cost_usd=cost_usd,
            cost_norm=cost_norm,
        ))
        tiers_used.append(tier_id)

    # Expected task accuracy = product of per-step success probs (error propagation)
    if not step_results:
        expected_acc = 0.0
    else:
        expected_acc = float(np.prod([s.p_step_success for s in step_results]))

    total_cost_usd = sum(s.cost_usd for s in step_results)
    total_cost_norm = sum(s.cost_norm for s in step_results)

    routing_changes = sum(
        1 for a, b in zip(tiers_used[:-1], tiers_used[1:]) if a != b
    ) if len(tiers_used) > 1 else 0

    return TrajectoryResult(
        traj_idx=traj.global_traj_idx,
        dataset=traj.dataset,
        expected_task_accuracy=expected_acc,
        total_cost_usd=total_cost_usd,
        total_cost_norm=total_cost_norm,
        step_results=step_results,
        n_escalations=sum(1 for t in tiers_used if t == 3),
        n_tier1=tiers_used.count(1),
        n_tier2=tiers_used.count(2),
        n_tier3=tiers_used.count(3),
        routing_changes=routing_changes,
    )


def evaluate_policy(
    policy: RoutingPolicy,
    trajectories: List[Trajectory],
    train_trajectories: List[Trajectory] = None,
    tiers: dict = None,
) -> PolicyResult:
    """
    Run a policy over all trajectories and aggregate metrics.

    Args:
        policy: Routing policy to evaluate.
        trajectories: Test trajectories.
        train_trajectories: If provided, fit learned policies first.
        tiers: Tier definitions to use (defaults to cost_model.TIERS).
    """
    if tiers is None:
        tiers = TIERS
    if train_trajectories is not None:
        policy.fit(train_trajectories)

    traj_results = [simulate_trajectory(t, policy, tiers=tiers) for t in trajectories]

    accs = np.array([r.expected_task_accuracy for r in traj_results])
    costs_usd = np.array([r.total_cost_usd for r in traj_results])
    costs_norm = np.array([r.total_cost_norm for r in traj_results])

    all_step_results = [s for r in traj_results for s in r.step_results]
    n_steps = len(all_step_results)
    tier_list = [s.tier_selected for s in all_step_results]
    n_escalations = sum(1 for t in tier_list if t == 3)
    total_routing_changes = sum(r.routing_changes for r in traj_results)
    total_step_transitions = sum(max(0, len(r.step_results) - 1) for r in traj_results)

    # Per-dataset breakdown
    from collections import defaultdict
    ds_accs = defaultdict(list)
    ds_costs = defaultdict(list)
    for r in traj_results:
        ds_accs[r.dataset].append(r.expected_task_accuracy)
        ds_costs[r.dataset].append(r.total_cost_norm)
    per_dataset = {
        ds: {
            "accuracy": float(np.mean(ds_accs[ds])),
            "cost_norm": float(np.mean(ds_costs[ds])),
        }
        for ds in ds_accs
    }

    n_successful = sum(1 for r in traj_results if r.expected_task_accuracy >= SUCCESS_THRESHOLD)
    task_success_rate = n_successful / len(traj_results) if traj_results else 0.0
    mean_cost_usd = float(costs_usd.mean())
    cost_per_success = mean_cost_usd / task_success_rate if task_success_rate > 0 else float("inf")

    return PolicyResult(
        policy_name=policy.name,
        mean_accuracy=float(accs.mean()),
        std_accuracy=float(accs.std()),
        total_cost_usd=float(costs_usd.sum()),
        mean_cost_usd_per_traj=mean_cost_usd,
        total_cost_norm=float(costs_norm.sum()),
        mean_cost_norm_per_traj=float(costs_norm.mean()),
        escalation_rate=n_escalations / n_steps if n_steps else 0.0,
        avg_tier=sum(tier_list) / len(tier_list) if tier_list else 0.0,
        routing_stability=1.0 - (total_routing_changes / total_step_transitions)
            if total_step_transitions > 0 else 1.0,
        task_success_rate=task_success_rate,
        cost_per_successful_task=cost_per_success,
        per_dataset=per_dataset,
        traj_results=traj_results,
    )
