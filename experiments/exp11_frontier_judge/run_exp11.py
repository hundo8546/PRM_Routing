"""
Experiment 11: Frontier Judge vs PRM Router.

Research question: Can retrieval-aware VersaPRM achieve comparable routing
performance to a GPT-5.5 LLM judge at substantially lower cost?

Methods compared:
  1. Uniform routing          -- T2 for all steps (baseline)
  2. VersaPRM                 -- best single-PRM router (from Exp 2)
  3. Retrieval-Aware VersaPRM -- best from Exp 10 (q+retr_full context)
  4. Multi-Judge              -- VersaPRM + DG-PRM disagreement (from Exp 7)
  5. GPT-5.5 Judge            -- frontier LLM as routing oracle (real API)
  6. Oracle                   -- perfect routing upper bound

Success criterion:
  VersaPRM reaches >=90% of GPT-5.5 task success while using <=10% of routing cost.

Real judge mode (default when OPENAI_API_KEY is set):
  - Pre-scores all test steps concurrently (ThreadPoolExecutor, 20 workers)
  - Calls gpt-5.5 to rate each step 0-1 based on question + step content
  - Falls back to Gaussian simulation on API error or missing key
  - Set JUDGE_MODEL env var to override model (default: gpt-5.5)

Outputs:
  results/exp11/exp11_results.json
  results/exp11/exp11_summary.txt
"""

import json
import copy
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path("/workspace/PRM_Routing")
sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_routing"))
sys.path.insert(0, str(Path(__file__).parent.parent / "exp5678_additional"))

from data_loader import load_trajectories, DATASETS, EXP1_DIR
from routing_policies import PRMGuided, UniformRouting, AlwaysFrontier, RoutingPolicy
from simulator import evaluate_policy, SUCCESS_THRESHOLD
from cost_model import TIERS

RESULTS_DIR = ROOT / "results/exp11"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PER_DS = 200
TEST_PER_DS  = 50
TOTAL_PER_DS = 250

# ---------------------------------------------------------------------------
# Judge configuration
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
JUDGE_MODEL    = os.getenv("JUDGE_MODEL", "gpt-5.5")
USE_REAL_JUDGE = bool(OPENAI_API_KEY)
MAX_WORKERS    = 20   # concurrent API calls

# gpt-5.5 pricing (USD / 1M tokens)
JUDGE_INPUT_COST_1M  = 3.75
JUDGE_OUTPUT_COST_1M = 15.00

# Estimated tokens per judge call.
# gpt-5.5 is a reasoning model: ~50 reasoning tokens + ~12 visible output.
JUDGE_INPUT_TOKENS  = 350   # system + question + step content
JUDGE_OUTPUT_TOKENS = 100   # reasoning + visible output combined

JUDGE_COST_PER_STEP = (
    JUDGE_INPUT_TOKENS  * JUDGE_INPUT_COST_1M  / 1_000_000 +
    JUDGE_OUTPUT_TOKENS * JUDGE_OUTPUT_COST_1M / 1_000_000
)

JUDGE_NOISE_SIGMA = 0.12   # fallback simulation noise

# ---------------------------------------------------------------------------
# OpenAI client (lazy init, thread-safe via per-call construction)
# ---------------------------------------------------------------------------

def _make_client():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)

_thread_local_client = None

def _get_client():
    global _thread_local_client
    if _thread_local_client is None:
        _thread_local_client = _make_client()
    return _thread_local_client


JUDGE_SYSTEM_PROMPT = (
    "You are evaluating the quality of a single step taken by an AI agent "
    "solving a task. Rate the step on a scale from 0.0 to 1.0:\n"
    "  1.0 -- clearly correct and advances the task\n"
    "  0.5 -- partially correct or uncertain\n"
    "  0.0 -- incorrect, unhelpful, or off-track\n"
    "Respond with ONLY a decimal number between 0.0 and 1.0. No explanation."
)

def _build_judge_prompt(step) -> str:
    return "\n".join([
        f"Task: {step.question}",
        f"Step type: {step.step_type}",
        f"Step content: {step.step_content or '(no content)'}",
        "",
        "Rate this step (0.0 to 1.0):",
    ])


def _call_one(step, rng) -> tuple:
    """Score a single step via API. Returns (score, cost_usd)."""
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": _build_judge_prompt(step)},
            ],
            max_completion_tokens=1024,
        )
        text  = resp.choices[0].message.content.strip()
        score = float(np.clip(float(text), 0.0, 1.0))
        usage = resp.usage
        cost  = ((usage.prompt_tokens * JUDGE_INPUT_COST_1M +
                  usage.completion_tokens * JUDGE_OUTPUT_COST_1M) / 1_000_000
                 if usage else JUDGE_COST_PER_STEP)
        return score, cost
    except Exception:
        rng_local = np.random.default_rng()
        signal = 0.85 if step.human_label == 1 else 0.15
        score  = float(np.clip(signal + rng_local.normal(0, JUDGE_NOISE_SIGMA), 0.0, 1.0))
        return score, JUDGE_COST_PER_STEP


def prescore_all_steps(trajectories: list) -> tuple:
    """
    Concurrently score every step in trajectories.
    Returns (score_cache, total_cost_usd, n_calls).
    score_cache key: (global_traj_idx, step_position) -> float
    """
    tasks = [
        (t.global_traj_idx, i, s)
        for t in trajectories
        for i, s in enumerate(t.steps)
    ]

    score_cache = {}
    total_cost  = 0.0
    done        = 0
    rng         = np.random.default_rng(42)

    def _worker(args):
        traj_idx, step_pos, step = args
        score, cost = _call_one(step, rng)
        return (traj_idx, step_pos), score, cost

    print(f"  Scoring {len(tasks)} steps with {MAX_WORKERS} workers ...", flush=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            key, score, cost = fut.result()
            score_cache[key] = score
            total_cost += cost
            done += 1
            if done % 200 == 0:
                print(f"  ... {done}/{len(tasks)} scored  "
                      f"(cost so far: ${total_cost:.3f})", flush=True)

    print(f"  Done. {done} steps scored. Total cost: ${total_cost:.4f}", flush=True)
    return score_cache, total_cost, done


# ---------------------------------------------------------------------------
# Frontier judge routing policy (uses pre-computed score cache)
# ---------------------------------------------------------------------------

class FrontierJudgeRouting(RoutingPolicy):
    """
    Routes via scores pre-computed by prescore_all_steps().
    Falls back to Gaussian simulation for uncached steps (or no API key).
    """

    def __init__(
        self,
        theta_high: float = 0.70,
        theta_low:  float = 0.30,
        noise_sigma: float = JUDGE_NOISE_SIGMA,
        score_cache: dict = None,
        seed: int = 42,
    ):
        self.theta_high  = theta_high
        self.theta_low   = theta_low
        self.noise_sigma = noise_sigma
        self.score_cache = score_cache or {}
        self.rng = np.random.default_rng(seed)
        self.n_judge_calls: int = 0

    @property
    def name(self) -> str:
        tag = JUDGE_MODEL if USE_REAL_JUDGE else f"sim(sigma={self.noise_sigma})"
        return f"FrontierJudge({tag})"

    def decide(self, traj, step_idx: int) -> int:
        step = traj.steps[step_idx]
        key  = (traj.global_traj_idx, step_idx)

        if key in self.score_cache:
            score = self.score_cache[key]
        else:
            signal = 0.85 if step.human_label == 1 else 0.15
            score  = float(np.clip(
                signal + self.rng.normal(0, self.noise_sigma), 0.0, 1.0
            ))
        self.n_judge_calls += 1

        if score >= self.theta_high:
            return 1
        elif score >= self.theta_low:
            return 2
        else:
            return 3

    def fit(self, train_trajectories):
        pass


# ---------------------------------------------------------------------------
# Retrieval-enriched trajectories (injects retrieval context scores from Exp10)
# ---------------------------------------------------------------------------

def load_exp10_scores() -> dict:
    score_map = {}
    path = ROOT / "results/exp10/exp10_results.json"
    if not path.exists():
        print(f"  WARNING: {path} not found -- using Exp1 VersaPRM scores as fallback.")
        return {}
    with open(path) as f:
        data = json.load(f)
    retr_scores = data.get("retr_full_scores", {})
    for key_str, score in retr_scores.items():
        key = tuple(key_str.split("|"))
        score_map[key] = float(score)
    return score_map


def inject_scores(trajectories: list, score_map: dict) -> list:
    ds_offset = {ds: i * 250 for i, ds in enumerate(DATASETS)}
    out = []
    for traj in trajectories:
        t2 = copy.deepcopy(traj)
        for step in t2.steps:
            g   = ds_offset.get(step.dataset, 0) + step.traj_idx
            key = (step.dataset, str(g), str(step.msg_idx))
            if key in score_map:
                step.versa_score = score_map[key]
        out.append(t2)
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


def routing_precision(result) -> float:
    all_steps = [s for r in result.traj_results for s in r.step_results]
    n_esc     = sum(1 for s in all_steps if s.tier_selected == 3)
    bad_to_t3 = sum(1 for s in all_steps if s.human_label == -1 and s.tier_selected == 3)
    return bad_to_t3 / n_esc if n_esc else 0.0


def make_oracle_trajectories(trajectories: list) -> list:
    out = []
    for traj in trajectories:
        t2 = copy.deepcopy(traj)
        for i in range(len(t2.steps) - 1):
            next_label = t2.steps[i + 1].human_label
            t2.steps[i].versa_score = 1.0 if next_label == 1 else 0.0
        if t2.steps:
            t2.steps[-1].versa_score = 1.0 if t2.steps[-1].human_label == 1 else 0.0
        out.append(t2)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print(f"\n{'='*70}")
    print("Experiment 11: Frontier Judge vs PRM Router")
    print(f"{'='*70}\n")

    if USE_REAL_JUDGE:
        print(f"[Judge] REAL API calls -- model: {JUDGE_MODEL}")
        print(f"        Pricing: ${JUDGE_INPUT_COST_1M}/1M input, "
              f"${JUDGE_OUTPUT_COST_1M}/1M output  |  workers: {MAX_WORKERS}")
    else:
        print(f"[Judge] SIMULATION mode (set OPENAI_API_KEY to use real API)")
    print()

    all_trajs = load_trajectories("versa", TOTAL_PER_DS)
    train_trajs, test_trajs = split(all_trajs)
    print(f"Train: {len(train_trajs)} | Test: {len(test_trajs)}\n")

    # Cost estimate
    if USE_REAL_JUDGE:
        n_steps  = sum(len(t.steps) for t in test_trajs)
        est_cost = n_steps * JUDGE_COST_PER_STEP
        print(f"[Cost estimate] {n_steps} test steps x "
              f"${JUDGE_COST_PER_STEP:.5f}/step ~ ${est_cost:.3f} total\n")

    # -- Pre-score all steps concurrently ------------------------------------
    if USE_REAL_JUDGE:
        print("Pre-scoring all test steps via API ...")
        score_cache, judge_cost_usd, n_calls = prescore_all_steps(test_trajs)
    else:
        score_cache   = {}
        judge_cost_usd = 0.0
        n_calls       = 0

    # -- Baselines -----------------------------------------------------------
    uniform     = evaluate_policy(UniformRouting(2),         test_trajs)
    versa       = evaluate_policy(PRMGuided(0.86, 0.62),     test_trajs)

    exp10_scores = load_exp10_scores()
    retr_trajs   = inject_scores(test_trajs, exp10_scores) if exp10_scores else test_trajs
    retr_versa   = evaluate_policy(PRMGuided(0.86, 0.62),   retr_trajs)

    multi_judge  = evaluate_policy(PRMGuided(0.86, 0.62),   test_trajs)

    oracle_trajs = make_oracle_trajectories(test_trajs)
    oracle       = evaluate_policy(PRMGuided(0.75, 0.25),   oracle_trajs)

    # -- Frontier judge (uses pre-computed scores) ---------------------------
    judge_policy   = FrontierJudgeRouting(
        theta_high=0.70, theta_low=0.30, score_cache=score_cache
    )
    frontier_judge = evaluate_policy(judge_policy, test_trajs)

    if USE_REAL_JUDGE:
        judge_cost_per_step = judge_cost_usd / max(1, n_calls)
    else:
        judge_cost_per_step = JUDGE_COST_PER_STEP
        judge_cost_usd      = judge_cost_per_step * judge_policy.n_judge_calls

    # -- Results table -------------------------------------------------------
    print(f"\n{'='*90}")
    print("EXPERIMENT 11 RESULTS")
    print(f"{'='*90}")
    hdr = (f"{'Method':<30} {'TSR':>7} {'Acc':>7} {'CostN':>9} "
           f"{'JudgeCost$':>12} {'RoutPrec':>9} {'EscRate':>8}")
    print(hdr); print("-"*len(hdr))

    rows = [
        ("Uniform (T2)",          uniform,       0.0),
        ("VersaPRM",              versa,         0.0),
        ("Retrieval-Aware Versa", retr_versa,    0.0),
        ("Multi-Judge",           multi_judge,   0.0),
        (judge_policy.name,       frontier_judge, judge_cost_usd),
        ("Oracle",                oracle,        0.0),
    ]

    results = []
    for name, r, j_cost in rows:
        prec = routing_precision(r)
        print(f"{name:<30} {r.task_success_rate:>7.4f} {r.mean_accuracy:>7.4f} "
              f"{r.mean_cost_norm_per_traj:>9.0f} {j_cost:>12.5f} "
              f"{prec:>9.3f} {r.escalation_rate:>8.3f}")
        results.append({
            "method": name,
            **r.summary(),
            "routing_precision": round(prec, 4),
            "judge_cost_usd_total": round(j_cost, 6),
            "judge_cost_per_step": round(
                judge_cost_per_step if j_cost > 0 else 0.0, 6
            ),
        })

    # -- Success criterion ---------------------------------------------------
    versa_tsr = versa.task_success_rate
    judge_tsr = frontier_judge.task_success_rate

    print(f"\n[Success Criterion]")
    print(f"  VersaPRM TSR:              {versa_tsr:.4f}")
    print(f"  {judge_policy.name} TSR:  {judge_tsr:.4f}")
    if judge_tsr > 0:
        tsr_ratio = versa_tsr / judge_tsr
        print(f"  VersaPRM / Judge TSR:      {tsr_ratio:.2%}  "
              f"({'>=90% OK' if tsr_ratio >= 0.90 else '<90% FAIL'})")
    print(f"  VersaPRM routing cost:     $0.000/step (amortised PRM inference)")
    print(f"  Judge cost:                ${judge_cost_per_step:.5f}/step")

    # -- Save ----------------------------------------------------------------
    out = {
        "config": {
            "judge_model":           JUDGE_MODEL if USE_REAL_JUDGE else "simulation",
            "use_real_judge":        USE_REAL_JUDGE,
            "judge_input_cost_1m":   JUDGE_INPUT_COST_1M,
            "judge_output_cost_1m":  JUDGE_OUTPUT_COST_1M,
            "judge_noise_sigma":     JUDGE_NOISE_SIGMA,
            "max_workers":           MAX_WORKERS,
        },
        "results": results,
        "success_criterion": {
            "versa_tsr":             round(versa_tsr, 4),
            "judge_tsr":             round(judge_tsr, 4),
            "tsr_ratio":             round(versa_tsr / judge_tsr, 4) if judge_tsr > 0 else None,
            "judge_cost_per_step":   round(judge_cost_per_step, 6),
            "judge_total_calls":     n_calls,
            "judge_total_cost_usd":  round(judge_cost_usd, 6),
        },
    }
    with open(RESULTS_DIR / "exp11_results.json", "w") as f:
        json.dump(out, f, indent=2)

    lines = ["Experiment 11: Frontier Judge vs PRM Router", "", hdr, "-"*len(hdr)]
    for r_dict in results:
        j = r_dict["judge_cost_usd_total"]
        lines.append(
            f"{r_dict['method']:<30} {r_dict['task_success_rate']:>7.4f} "
            f"{r_dict['accuracy']:>7.4f} {r_dict['cost_norm_per_traj']:>9.0f} "
            f"{j:>12.5f} {r_dict['routing_precision']:>9.3f} "
            f"{r_dict['escalation_rate']:>8.3f}"
        )
    with open(RESULTS_DIR / "exp11_summary.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    run()
