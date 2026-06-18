"""
Exp 1: Run all 4 PRMs on AgentProcessBench trajectories.

Saves per-step: {step_type, reward_score, human_step_label, final_task_success}
Pilot: 100 trajectories (25/dataset). Full run: 1000 trajectories (250/dataset, --n_per_dataset 250).
"""

import json
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/workspace/PRM_Routing")
DATA_DIR = ROOT / "benchmarks/AgentProcessBench/data/AgentProcessBench"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results/exp1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["hotpotqa", "gaia_dev", "bfcl", "tau2"]
SAMPLES_PER_DATASET = 25  # 4 × 25 = 100 total

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_step_type(msg: dict) -> str:
    """Infer step type from message content."""
    tool_calls = msg.get("tool_calls", "")
    content = msg.get("content", "")
    if tool_calls and tool_calls != "" and tool_calls != "[]":
        try:
            tc = json.loads(tool_calls) if isinstance(tool_calls, str) else tool_calls
            if tc:
                fn_name = tc[0].get("function", {}).get("name", "")
                if "search" in fn_name or "retriev" in fn_name:
                    return "retrieval"
                return "tool_call"
        except Exception:
            return "tool_call"
    if content:
        return "synthesis"
    return "unknown"


def load_trajectories(n_per_dataset: int = SAMPLES_PER_DATASET):
    """Load n_per_dataset trajectories from each dataset."""
    trajectories = []
    for ds in DATASETS:
        path = DATA_DIR / f"{ds}.jsonl"
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= n_per_dataset:
                    break
                traj = json.loads(line)
                traj["_dataset"] = ds
                trajectories.append(traj)
    print(f"Loaded {len(trajectories)} trajectories ({n_per_dataset} × {len(DATASETS)} datasets)")
    return trajectories


def extract_steps(traj: dict):
    """Return list of (msg_idx, message, step_label) for labeled assistant turns."""
    msgs = traj["messages"]
    labels = traj.get("step_labels", {})
    steps = []
    for idx_str, label in labels.items():
        idx = int(idx_str)
        msg = msgs[idx]
        steps.append((idx, msg, label))
    return steps


def build_step_context(traj: dict, msg_idx: int) -> str:
    """Build context string: question + prior turns up to (but not including) this step."""
    msgs = traj["messages"]
    parts = []
    q = traj.get("question", "")
    if q:
        parts.append(f"Task: {q}")
    for i, msg in enumerate(msgs):
        if i >= msg_idx:
            break
        role = msg["role"]
        content = msg.get("content", "")
        if role == "tool" and content:
            parts.append(f"[Tool result]: {str(content)[:300]}")
    return "\n".join(parts)


def build_step_text(msg: dict) -> str:
    """Render an agent step as a string."""
    tool_calls = msg.get("tool_calls", "")
    content = msg.get("content", "")
    if tool_calls and tool_calls not in ("", "[]"):
        try:
            tc = json.loads(tool_calls) if isinstance(tool_calls, str) else tool_calls
            fn = tc[0].get("function", {})
            return f"Action: {fn.get('name', 'tool')}({fn.get('arguments', '')})"
        except Exception:
            return str(tool_calls)[:300]
    return str(content)[:500]


# ---------------------------------------------------------------------------
# Qwen PRM
# ---------------------------------------------------------------------------

class QwenPRM:
    name = "qwen_prm"

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        model_dir = str(MODELS_DIR / "qwen_prm")
        print(f"Loading QwenPRM from {model_dir} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        self.step_sep_id = self.tokenizer.encode("<extra_0>")[0]

    def score(self, question: str, context: str, step_text: str) -> float:
        """Return probability [0,1] that the step is good."""
        # Format: system + user (question) + assistant (context + this step)
        # We treat this step as a single-step "solution"
        ctx_prefix = (context + "\n\n") if context.strip() else ""
        system_msg = "You are a helpful agent evaluating step quality in a task-solving pipeline."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question or "Solve the task."},
            {"role": "assistant", "content": ctx_prefix + step_text + "<extra_0>"},
        ]
        conv_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = self.tokenizer.encode(conv_str, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=False)
        logits = outputs[0]  # (1, seq_len, 2)
        token_masks = (input_ids == self.step_sep_id)
        probabilities = F.softmax(logits, dim=-1)
        masked = probabilities * token_masks.unsqueeze(-1)
        # Extract scores at step separator positions
        sample = masked[0]  # (seq_len, 2)
        valid = sample[sample.sum(-1) != 0]  # (n_steps, 2)
        if len(valid) == 0:
            return 0.5
        return float(valid[-1, 1].cpu())  # probability of label 1 (good) at last step


# ---------------------------------------------------------------------------
# VersaPRM
# ---------------------------------------------------------------------------

class VersaPRM:
    name = "versa_prm"

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        base_dir = str(MODELS_DIR / "versaprm/base")
        adapter_dir = str(MODELS_DIR / "versaprm/adapter")
        print(f"Loading VersaPRM base from {base_dir} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_dir, torch_dtype=torch.bfloat16, device_map="auto"
        )
        print("Applying VersaPRM LoRA adapter ...")
        self.model = PeftModel.from_pretrained(base_model, adapter_dir).eval()
        self.candidate_tokens = [12, 10]
        # Token 23535 is the step separator ' \n\n\n\n' in LLaMA tokenizer
        self.step_sep_token = 23535

    def score(self, question: str, context: str, step_text: str) -> float:
        input_text = (question or "Solve the task.") + " \n\n" + step_text + " \n\n\n\n"
        input_ids = torch.tensor([self.tokenizer.encode(input_text)]).to(self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[:, :, self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 1]
            step_mask = (input_ids == self.step_sep_token)
        if not step_mask.any():
            # fallback: score at last token
            return float(scores[0, -1].cpu())
        return float(scores[step_mask].mean().cpu())


# ---------------------------------------------------------------------------
# AgentRM
# ---------------------------------------------------------------------------

class AgentRM:
    name = "agent_rm"

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        model_dir = str(MODELS_DIR / "agentprm")
        print(f"Loading AgentRM backbone from {model_dir} ...")
        # The released weights are a LlamaModel backbone only (no score head).
        # We use the backbone as a prompted scorer via tied LM weights.
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        # Pre-compute Yes/No token ids for prompted scoring
        self.yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[-1]
        self.no_id = self.tokenizer.encode("No", add_special_tokens=False)[-1]
        print(f"  AgentRM loaded (yes_id={self.yes_id}, no_id={self.no_id})")

    def score(self, question: str, context: str, step_text: str) -> float:
        # Prompted scoring: P("Yes" | prompt) via tied LM weights
        prompt = (
            f"Task: {question or 'Complete the task.'}\n"
            f"Agent step: {step_text[:300]}\n"
            "Is this a good agent step that makes progress toward the goal? Answer Yes or No."
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=768
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs)
            hidden = out.last_hidden_state[0, -1]  # last token hidden state
            # Use embed_tokens as tied LM head: logits = hidden @ embed_tokens.T
            embed = self.model.embed_tokens.weight  # (vocab, hidden)
            logits = hidden @ embed.T  # (vocab,)
        pair = torch.stack([logits[self.yes_id], logits[self.no_id]])
        return float(F.softmax(pair, dim=0)[0].cpu())


# ---------------------------------------------------------------------------
# DG-PRM  (local LLM judge, simplified zero-shot mode)
# ---------------------------------------------------------------------------

class DGPRM:
    name = "dg_prm"

    # Generic reward criteria for agent steps
    CRITERIA = [
        "The step is logically correct and makes meaningful progress toward the task goal.",
        "The step uses the right tool or produces the right action given the available information.",
        "The step does not contain factual errors or hallucinated information.",
    ]

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # Use VersaPRM base (Llama-PRM800K) as local judge — distinct from the AgentRM backbone.
        judge_dir = str(MODELS_DIR / "versaprm/base")
        print(f"Loading DG-PRM local judge (Llama-PRM800K) from {judge_dir} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(judge_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            judge_dir, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()

    def score(self, question: str, context: str, step_text: str) -> float:
        """Score via average log-likelihood of 'Yes' vs 'No' over criteria."""
        scores = []
        for criterion in self.CRITERIA:
            prompt = (
                f"Task: {question or 'Complete the task.'}\n"
                f"Agent step: {step_text}\n\n"
                f"Criterion: {criterion}\n"
                "Does this step satisfy the criterion? Answer Yes or No."
            )
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=768
            ).to(self.model.device)
            with torch.no_grad():
                out = self.model(**inputs)
                logits = out.logits[0, -1]  # last token logits
            yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[-1]
            no_id = self.tokenizer.encode("No", add_special_tokens=False)[-1]
            pair_logits = torch.tensor([logits[yes_id], logits[no_id]])
            prob_yes = float(F.softmax(pair_logits, dim=0)[0].cpu())
            scores.append(prob_yes)
        return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_pilot(prm_name: str, limit: Optional[int] = None, n_per_dataset: int = SAMPLES_PER_DATASET):
    print(f"\n{'='*60}")
    print(f"Running Exp 1 with PRM: {prm_name}  ({n_per_dataset}/dataset)")
    print(f"{'='*60}")

    # Load trajectories
    trajectories = load_trajectories(n_per_dataset)
    if limit:
        trajectories = trajectories[:limit]

    # Load PRM
    prm_map = {
        "qwen": QwenPRM,
        "versa": VersaPRM,
        "agent": AgentRM,
        "dgprm": DGPRM,
    }
    if prm_name not in prm_map:
        print(f"Unknown PRM '{prm_name}'. Choose from: {list(prm_map)}")
        sys.exit(1)
    prm = prm_map[prm_name]()

    results = []
    total_steps = 0
    errors = 0

    for t_idx, traj in enumerate(trajectories):
        steps = extract_steps(traj)
        question = traj.get("question", "")
        final_label = traj.get("final_label", None)
        task_success = 1 if final_label == 1 else 0
        dataset = traj["_dataset"]

        for msg_idx, msg, step_label in steps:
            total_steps += 1
            step_type = infer_step_type(msg)
            context = build_step_context(traj, msg_idx)
            step_text = build_step_text(msg)

            try:
                reward_score = prm.score(question, context, step_text)
            except Exception as e:
                print(f"  [ERROR] traj={t_idx} step={msg_idx}: {e}")
                reward_score = None
                errors += 1

            results.append({
                "traj_idx": t_idx,
                "dataset": dataset,
                "msg_idx": msg_idx,
                "step_type": step_type,
                "reward_score": reward_score,
                "human_step_label": step_label,   # 1 or -1
                "final_task_success": task_success,
                "question": question[:100],
            })

        if (t_idx + 1) % 10 == 0:
            print(f"  Processed {t_idx+1}/{len(trajectories)} trajectories ({total_steps} steps, {errors} errors)")

    # Save results — prefix "full_" for 250/dataset runs, "pilot_" for 25/dataset
    prefix = "full" if n_per_dataset >= 250 else "pilot"
    out_path = RESULTS_DIR / f"{prefix}_{prm_name}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. {total_steps} steps scored, {errors} errors.")
    print(f"Results saved to {out_path}")

    # Quick summary
    scored = [r for r in results if r["reward_score"] is not None]
    if scored:
        pos = [r["reward_score"] for r in scored if r["human_step_label"] == 1]
        neg = [r["reward_score"] for r in scored if r["human_step_label"] == -1]
        print(f"\nQuick stats:")
        print(f"  Good steps (label=1):  n={len(pos)}, mean_score={np.mean(pos):.3f}" if pos else "  No good steps")
        print(f"  Bad steps  (label=-1): n={len(neg)}, mean_score={np.mean(neg):.3f}" if neg else "  No bad steps")
        if pos and neg:
            sep = np.mean(pos) - np.mean(neg)
            print(f"  Score separation (good - bad): {sep:+.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prm", required=True, choices=["qwen", "versa", "agent", "dgprm"],
                        help="Which PRM to run")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit total trajectories (for testing)")
    parser.add_argument("--n_per_dataset", type=int, default=SAMPLES_PER_DATASET,
                        help="Trajectories per dataset (max 250; use 250 for full Exp 1)")
    args = parser.parse_args()
    run_pilot(args.prm, args.limit, args.n_per_dataset)
