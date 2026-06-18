"""
Disagreement-aware routing policies for Exps 6, 7, and 8.

All policies extend the base RoutingPolicy interface and access step.extra_scores.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_routing"))
from routing_policies import RoutingPolicy

if TYPE_CHECKING:
    from data_loader import Trajectory


# ---------------------------------------------------------------------------
# Exp 6: PRM Disagreement Routing (Versa vs Qwen)
# ---------------------------------------------------------------------------

class PRMDisagreementRouting(RoutingPolicy):
    """
    Exp 6: abs(versa_score - qwen_score) as uncertainty signal.

    If disagreement > τ_disagree → force T3 (uncertain → conservative).
    Otherwise → standard PRMGuided 3-tier routing on versa_score.

    Note: Qwen scores are all near 0.14–0.16, so disagreement ≈ versa_score - 0.14.
    High disagreement therefore correlates with HIGH Versa scores (Versa thinks good,
    Qwen thinks bad). Forcing T3 in that case is counterintuitive — analysed in paper.
    Sweep over τ_disagree to find if there is any useful regime.
    """
    name = "Disagreement (Versa-Qwen)"

    def __init__(
        self,
        tau_disagree: float = 0.55,
        theta_high: float = 0.86,
        theta_low: float = 0.62,
        default_tier: int = 2,
    ):
        self.tau_disagree = tau_disagree
        self.theta_high = theta_high
        self.theta_low = theta_low
        self.default_tier = default_tier
        self.name = f"Dis(VQ) τ={tau_disagree:.2f}"

    def decide(self, traj, step_idx):
        if step_idx == 0:
            return self.default_tier
        prev = traj.steps[step_idx - 1]
        v = prev.versa_score
        q = prev.extra_scores.get("qwen", v)   # fallback to versa if missing
        disagreement = abs(v - q)

        if disagreement > self.tau_disagree:
            return 3   # uncertain → frontier
        # Standard 3-tier on versa
        if v > self.theta_high:
            return 1
        elif v > self.theta_low:
            return 2
        else:
            return 3


# ---------------------------------------------------------------------------
# Exp 7: Multi-Judge Disagreement (Versa vs DG-PRM)
# ---------------------------------------------------------------------------

class MultiJudgeDisagreement(RoutingPolicy):
    """
    Exp 7: abs(versa_score - dg_score) as uncertainty signal.

    DG-PRM scores cluster near 0.9. High disagreement means Versa is LOW
    while DG is HIGH → Versa detects a potential problem that DG misses →
    escalate to T3.

    Routing:
      disagreement > τ_disagree → T3  (conflicting signal → cautious)
      versa_score > θ_high         → T1  (both agree it's good)
      else                         → T2
    """
    name = "Multi-Judge (Versa-DG)"

    def __init__(
        self,
        tau_disagree: float = 0.20,
        theta_high: float = 0.86,
        default_tier: int = 2,
    ):
        self.tau_disagree = tau_disagree
        self.theta_high = theta_high
        self.default_tier = default_tier
        self.name = f"MultiJudge τ={tau_disagree:.2f}"

    def decide(self, traj, step_idx):
        if step_idx == 0:
            return self.default_tier
        prev = traj.steps[step_idx - 1]
        v = prev.versa_score
        d = prev.extra_scores.get("dgprm", v)
        disagreement = abs(v - d)

        if disagreement > self.tau_disagree:
            return 3   # conflicting signal → conservative
        if v > self.theta_high:
            return 1
        return 2


# ---------------------------------------------------------------------------
# Exp 8: Temporal Disagreement (score drop within trajectory)
# ---------------------------------------------------------------------------

class TemporalDisagreement(RoutingPolicy):
    """
    Exp 8: Score drop Δ = score_{t-1} - score_{t-2} as a trajectory warning signal.

    A large drop in PRM scores between consecutive steps signals deteriorating
    trajectory quality → escalate the next step.

    Routing for step t+1 (using scores at t and t-1):
      delta(t) = score_{t} - score_{t-1}   (negative = drop)
      If -delta > τ_drop → T3   (significant quality drop → escalate)
      Elif score_t > θ_high → T1
      Elif score_t > θ_low  → T2
      Else                  → T3
    """
    name = "Temporal Drop"

    def __init__(
        self,
        tau_drop: float = 0.15,
        theta_high: float = 0.86,
        theta_low: float = 0.62,
        default_tier: int = 2,
    ):
        self.tau_drop = tau_drop
        self.theta_high = theta_high
        self.theta_low = theta_low
        self.default_tier = default_tier
        self.name = f"Temporal τ_drop={tau_drop:.2f}"

    def decide(self, traj, step_idx):
        if step_idx == 0:
            return self.default_tier

        prev_score = traj.steps[step_idx - 1].versa_score

        # Compute drop from two steps back (if available)
        if step_idx >= 2:
            two_back_score = traj.steps[step_idx - 2].versa_score
            delta = prev_score - two_back_score   # positive = improvement
            drop = -delta                          # positive = deterioration
            if drop > self.tau_drop:
                return 3   # significant quality drop → escalate
        # Fall through to standard 3-tier routing on prev_score
        if prev_score > self.theta_high:
            return 1
        elif prev_score > self.theta_low:
            return 2
        else:
            return 3


# ---------------------------------------------------------------------------
# Combined: Versa + Temporal + Multi-Judge (most promising direction per notes)
# ---------------------------------------------------------------------------

class CombinedSignal(RoutingPolicy):
    """
    Combined routing signal: VersaPRM + temporal drop + Versa-DG disagreement.

    Escalation triggers (any one → T3):
      1. Large Versa score drop (temporal)
      2. High Versa-DG disagreement (multi-judge)
    T1 trigger: prev versa_score > θ_high AND no escalation triggers.
    Otherwise: T2.
    """
    name = "Combined (Versa+Temporal+MultiJudge)"

    def __init__(
        self,
        tau_drop: float = 0.15,
        tau_disagree: float = 0.20,
        theta_high: float = 0.86,
        default_tier: int = 2,
    ):
        self.tau_drop = tau_drop
        self.tau_disagree = tau_disagree
        self.theta_high = theta_high
        self.default_tier = default_tier
        self.name = f"Combined(Δ={tau_drop:.2f},D={tau_disagree:.2f})"

    def decide(self, traj, step_idx):
        if step_idx == 0:
            return self.default_tier

        prev = traj.steps[step_idx - 1]
        v = prev.versa_score
        d = prev.extra_scores.get("dgprm", v)

        # Trigger 1: Versa-DG disagreement
        if abs(v - d) > self.tau_disagree:
            return 3

        # Trigger 2: temporal drop
        if step_idx >= 2:
            two_back = traj.steps[step_idx - 2].versa_score
            if (two_back - v) > self.tau_drop:
                return 3

        # No escalation triggers
        if v > self.theta_high:
            return 1
        return 2


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def multijudge_sweep(tau_vals=None, theta_highs=None):
    """Sweep of MultiJudgeDisagreement for Pareto curve."""
    if tau_vals is None:
        tau_vals = np.linspace(0.08, 0.45, 10)
    if theta_highs is None:
        theta_highs = [0.75, 0.86, 0.92]
    return [
        MultiJudgeDisagreement(tau_disagree=float(t), theta_high=float(th))
        for t in tau_vals for th in theta_highs
    ]


def temporal_sweep(tau_drop_vals=None):
    if tau_drop_vals is None:
        tau_drop_vals = np.linspace(0.05, 0.40, 12)
    return [TemporalDisagreement(tau_drop=float(t)) for t in tau_drop_vals]


def combined_sweep():
    tau_drops = [0.10, 0.15, 0.25]
    tau_dis   = [0.15, 0.20, 0.30]
    return [CombinedSignal(td, tdis) for td in tau_drops for tdis in tau_dis]
