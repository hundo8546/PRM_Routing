"""
Cost model for heterogeneous model tiers.

Costs are in USD per 1M tokens (approximate public API rates, June 2026).
Token counts per step are estimated from AgentProcessBench message lengths.
"""

from dataclasses import dataclass
from typing import Dict

# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tier:
    id: int
    name: str
    input_cost_per_1m: float   # USD per 1M input tokens
    output_cost_per_1m: float  # USD per 1M output tokens
    # Outcome probabilities under counterfactual routing
    p_good_step_success: float  # P(step succeeds | human_label==1, this tier)
    p_bad_step_recovery: float  # P(step succeeds | human_label==-1, this tier)


# Original stack (Exp 1–10)
TIERS: Dict[int, Tier] = {
    1: Tier(
        id=1,
        name="Llama-3.1-8B",
        input_cost_per_1m=0.20,
        output_cost_per_1m=0.20,
        p_good_step_success=0.92,
        p_bad_step_recovery=0.30,
    ),
    2: Tier(
        id=2,
        name="Llama-3.1-70B",
        input_cost_per_1m=0.90,
        output_cost_per_1m=0.90,
        p_good_step_success=0.97,
        p_bad_step_recovery=0.65,
    ),
    3: Tier(
        id=3,
        name="Qwen2.5-72B",
        input_cost_per_1m=1.80,
        output_cost_per_1m=1.80,
        p_good_step_success=0.99,
        p_bad_step_recovery=0.85,
    ),
}

# Modern recommended stack (Exp 12)
# NOTE: update input/output costs with actual API rates before running.
MODERN_TIERS: Dict[int, Tier] = {
    1: Tier(
        id=1,
        name="Qwen3-8B",
        input_cost_per_1m=0.22,
        output_cost_per_1m=0.22,
        p_good_step_success=0.93,
        p_bad_step_recovery=0.50,   # notably stronger than Llama 8B
    ),
    2: Tier(
        id=2,
        name="GPT-5.5",
        input_cost_per_1m=3.75,
        output_cost_per_1m=15.00,
        p_good_step_success=0.985,
        p_bad_step_recovery=0.82,   # strong agentic reasoning
    ),
    3: Tier(
        id=3,
        name="Claude-Opus-4.7",
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
        p_good_step_success=0.995,
        p_bad_step_recovery=0.88,
    ),
}

# Alternative open-weight modern stack (Exp 12)
OPEN_WEIGHT_TIERS: Dict[int, Tier] = {
    1: Tier(
        id=1,
        name="Gemma4-E4B",
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.10,
        p_good_step_success=0.91,
        p_bad_step_recovery=0.38,
    ),
    2: Tier(
        id=2,
        name="Qwen3-8B",
        input_cost_per_1m=0.22,
        output_cost_per_1m=0.22,
        p_good_step_success=0.93,
        p_bad_step_recovery=0.50,
    ),
    3: Tier(
        id=3,
        name="Qwen3-30B-A3B",
        input_cost_per_1m=0.50,
        output_cost_per_1m=0.50,
        p_good_step_success=0.975,
        p_bad_step_recovery=0.72,
    ),
}

# Relative cost multipliers (Tier 1 = 1.0) — used for normalised reporting
RELATIVE_COST = {tid: t.input_cost_per_1m / TIERS[1].input_cost_per_1m for tid, t in TIERS.items()}


def step_cost_usd(tier_id: int, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost for one step given tier and token counts."""
    t = TIERS[tier_id]
    return (input_tokens * t.input_cost_per_1m + output_tokens * t.output_cost_per_1m) / 1_000_000


def step_cost_normalised(tier_id: int, input_tokens: int, output_tokens: int) -> float:
    """Cost normalised so Tier-1 cost = 1.0 per token."""
    return (input_tokens + output_tokens) * RELATIVE_COST[tier_id]
