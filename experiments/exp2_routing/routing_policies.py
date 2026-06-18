"""
Routing policies for Experiment 2.

Each policy implements decide(traj, step_idx) -> tier_id (1, 2, or 3).

Policies:
  1. UniformRouting       — always Tier 2 (lower bound on cost-awareness)
  2. AlwaysFrontier       — always Tier 3 (upper bound on accuracy)
  3. AlwaysCheap          — always Tier 1 (lower bound on accuracy)
  4. TRIMStyle            — VersaPRM binary threshold: high→T1, low→T3
  5. DAOOStyle            — pre-execution difficulty routing (query-level)
  6. BAARStyle            — learned boundary-guided routing (logistic regression)
  7. PRMGuided            — VersaPRM three-tier routing: high→T1, mid→T2, low→T3
"""

from __future__ import annotations

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from data_loader import Trajectory, StepRecord


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class RoutingPolicy(ABC):
    name: str

    def decide(self, traj: "Trajectory", step_idx: int) -> int:
        """Return tier id (1, 2, or 3) for step at step_idx."""
        raise NotImplementedError

    def fit(self, trajectories: List["Trajectory"]) -> None:
        """Optional: fit any learned parameters on training trajectories."""
        pass

    def __repr__(self):
        return self.name


# ---------------------------------------------------------------------------
# 1. Uniform routing (Tier 2 always)
# ---------------------------------------------------------------------------

class UniformRouting(RoutingPolicy):
    name = "Uniform (T2)"

    def __init__(self, tier: int = 2):
        self.tier = tier
        self.name = f"Uniform (T{tier})"

    def decide(self, traj, step_idx):
        return self.tier


# ---------------------------------------------------------------------------
# 2. Always-Frontier
# ---------------------------------------------------------------------------

class AlwaysFrontier(RoutingPolicy):
    name = "Always-Frontier (T3)"

    def decide(self, traj, step_idx):
        return 3


# ---------------------------------------------------------------------------
# 3. Always-Cheap
# ---------------------------------------------------------------------------

class AlwaysCheap(RoutingPolicy):
    name = "Always-Cheap (T1)"

    def decide(self, traj, step_idx):
        return 1


# ---------------------------------------------------------------------------
# 4. TRIM-style threshold (binary: T1 or T3 based on PRM score)
# ---------------------------------------------------------------------------

class TRIMStyle(RoutingPolicy):
    """
    Mimics TRIM (Kapoor et al. 2026): use PRM score threshold to route between
    cheap and frontier. Steps above θ → Tier 1; below → Tier 3.

    Because TRIM routes step T+1 based on score at step T, the first step
    always uses the default_tier.
    """
    name = "TRIM-style"

    def __init__(self, theta: float = 0.65, default_tier: int = 2):
        self.theta = theta
        self.default_tier = default_tier
        self.name = f"TRIM (θ={theta:.2f})"

    def decide(self, traj, step_idx):
        if step_idx == 0:
            return self.default_tier
        prev_score = traj.steps[step_idx - 1].versa_score
        return 1 if prev_score >= self.theta else 3


# ---------------------------------------------------------------------------
# 5. DAAO-style (pre-execution, query-level difficulty)
# ---------------------------------------------------------------------------

class DAAOStyle(RoutingPolicy):
    """
    Difficulty-Aware Agentic Orchestration (Su et al. 2025).
    Routes the entire trajectory to one tier based on pre-execution query
    difficulty estimation (query length + structural complexity proxy).
    """
    name = "DAAO"

    def __init__(self, low_thresh: float = 0.33, high_thresh: float = 0.66):
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self._difficulty_cache: dict = {}

    def _difficulty(self, question: str) -> float:
        """Normalised difficulty score in [0, 1] from question features."""
        q = question.strip()
        # Proxy: question length (longer = harder), question marks, sub-clauses
        length_score = min(1.0, len(q) / 500.0)
        clause_score = min(1.0, q.count(",") / 10.0)
        wh_score = 0.2 if any(w in q.lower() for w in ["which", "what", "why", "how", "who", "when"]) else 0.0
        return (length_score * 0.5 + clause_score * 0.3 + wh_score * 0.2)

    def _traj_tier(self, traj) -> int:
        if traj.global_traj_idx not in self._difficulty_cache:
            d = self._difficulty(traj.question)
            if d < self.low_thresh:
                tier = 1
            elif d < self.high_thresh:
                tier = 2
            else:
                tier = 3
            self._difficulty_cache[traj.global_traj_idx] = tier
        return self._difficulty_cache[traj.global_traj_idx]

    def decide(self, traj, step_idx):
        return self._traj_tier(traj)


# ---------------------------------------------------------------------------
# 6. BAAR-style (learned logistic boundary)
# ---------------------------------------------------------------------------

class BAARStyle(RoutingPolicy):
    """
    Budget-Aware Agentic Routing (Zhang et al. 2026).
    Trains a lightweight logistic-regression boundary on step features
    to predict whether a step needs escalation (human_label == -1).

    Features used: step_type (one-hot), step_position (normalised),
    n_steps_total (normalised), versa_score (from the selected PRM).
    """
    name = "BAAR"

    def __init__(self, escalation_threshold: float = 0.4, low_tier: int = 1, high_tier: int = 3):
        self.escalation_threshold = escalation_threshold
        self.low_tier = low_tier
        self.high_tier = high_tier
        self._clf = None
        self._fitted = False

    def _featurise(self, step) -> List[float]:
        type_enc = {
            "retrieval": [1, 0, 0],
            "tool_call": [0, 1, 0],
            "synthesis": [0, 0, 1],
        }.get(step.step_type, [0, 0, 0])
        pos_norm = step.step_position / max(1, step.n_steps_total - 1)
        n_total_norm = step.n_steps_total / 10.0
        return type_enc + [pos_norm, n_total_norm]

    def fit(self, trajectories):
        from sklearn.linear_model import LogisticRegression
        X, y = [], []
        for traj in trajectories:
            for step in traj.steps:
                X.append(self._featurise(step))
                y.append(1 if step.human_label == -1 else 0)  # 1 = needs escalation
        X, y = np.array(X), np.array(y)
        self._clf = LogisticRegression(max_iter=500, class_weight="balanced")
        self._clf.fit(X, y)
        self._fitted = True

    def decide(self, traj, step_idx):
        if not self._fitted:
            return 2  # fallback before fitting
        step = traj.steps[step_idx]
        feat = np.array(self._featurise(step)).reshape(1, -1)
        p_escalate = self._clf.predict_proba(feat)[0, 1]
        return self.high_tier if p_escalate >= self.escalation_threshold else self.low_tier


# ---------------------------------------------------------------------------
# 7. PRM-Guided Routing (proposed) — three-tier
# ---------------------------------------------------------------------------

class PRMGuided(RoutingPolicy):
    """
    Proposed method: use VersaPRM score at step T to route step T+1.

    Score > θ_high  → Tier 1 (confident, use cheap model)
    θ_low < score ≤ θ_high → Tier 2 (uncertain, use medium model)
    score ≤ θ_low   → Tier 3 (step failed, use frontier model to recover)
    """
    name = "PRM-Guided"

    def __init__(self, theta_high: float = 0.70, theta_low: float = 0.55, default_tier: int = 2):
        self.theta_high = theta_high
        self.theta_low = theta_low
        self.default_tier = default_tier
        self.name = f"PRM-Guided (h={theta_high:.2f}/l={theta_low:.2f})"

    def decide(self, traj, step_idx):
        if step_idx == 0:
            return self.default_tier
        prev_score = traj.steps[step_idx - 1].versa_score
        if prev_score > self.theta_high:
            return 1
        elif prev_score > self.theta_low:
            return 2
        else:
            return 3


# ---------------------------------------------------------------------------
# Threshold sweep helpers (for Pareto curves)
# ---------------------------------------------------------------------------

def trim_sweep(thetas=None) -> List[TRIMStyle]:
    """Generate a sweep of TRIM policies for Pareto curve.
    Range spans p10–p95 of VersaPRM score distribution (0.475–0.945).
    """
    if thetas is None:
        thetas = np.linspace(0.48, 0.92, 23)
    return [TRIMStyle(theta=float(t)) for t in thetas]


def prm_guided_sweep(theta_highs=None, theta_lows=None) -> List[PRMGuided]:
    """Generate a grid sweep of PRM-Guided policies for Pareto curve.
    theta_high: p60–p95 (0.70–0.94); theta_low: p10–p60 (0.48–0.70).
    """
    if theta_highs is None:
        theta_highs = np.linspace(0.70, 0.94, 7)
    if theta_lows is None:
        theta_lows = np.linspace(0.48, 0.70, 7)
    policies = []
    for th in theta_highs:
        for tl in theta_lows:
            if tl < th - 0.04:   # enforce meaningful gap between thresholds
                policies.append(PRMGuided(theta_high=float(th), theta_low=float(tl)))
    return policies
