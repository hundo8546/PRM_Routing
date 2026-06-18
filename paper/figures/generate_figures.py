"""
Generate all publication figures for PRM-Guided Cost-Aware Routing paper.

Run from /workspace/PRM_Routing/:
    python3 paper/figures/generate_figures.py

Outputs (PDF + PNG):
    paper/figures/fig1_system_diagram.pdf
    paper/figures/fig2_prm_signal_by_step_type.pdf
    paper/figures/fig3_pareto_frontier.pdf
    paper/figures/fig4_routing_quality_summary.pdf
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT   = Path("/workspace/PRM_Routing")
FIGS   = ROOT / "paper/figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# ACM-friendly colour palette (colour-blind safe)
C = {
    "versa":    "#2166AC",   # blue  — VersaPRM
    "qwen":     "#D6604D",   # red   — Qwen
    "dgprm":    "#74ADD1",   # light blue — DG-PRM
    "agent":    "#FDAE61",   # orange — AgentRM
    "prm":      "#1B7837",   # green — PRM-Guided (proposed)
    "trim":     "#762A83",   # purple — TRIM
    "baar":     "#E08214",   # amber — BAAR
    "uniform":  "#555555",   # grey — Uniform
    "frontier": "#000000",   # black — Always-Frontier
    "daao":     "#AAAAAA",   # light grey — DAAO
    "oracle":   "#4DAC26",   # light green — Oracle
    "random":   "#BABABA",   # very light — Random
}

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    print(f"  Saved {name}.pdf / .png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: System Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_system():
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_facecolor("#FAFAFA")

    def box(x, y, w, h, label, sublabel="", color="#D0E8FF", textcolor="black"):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="#444", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
                ha="center", va="center", fontsize=8.5, fontweight="bold", color=textcolor)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                    ha="center", va="center", fontsize=7, color="#555", style="italic")

    def arrow(x1, x2, y):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))

    # Pipeline row 1: step → PRM scorer → router → tier
    box(0.1, 2.5, 1.8, 0.9, "Agent Step", "tool call / synthesis", "#E8E8E8")
    arrow(1.9, 2.3, 2.95)
    box(2.3, 2.5, 2.2, 0.9, "VersaPRM Scorer",  "+retrieval context", "#D0E8FF")
    arrow(4.5, 4.9, 2.95)
    box(4.9, 2.5, 2.0, 0.9, "Routing Controller", "PRMGuided θ_h/θ_l", "#D0F0D8")
    arrow(6.9, 7.3, 2.95)

    # Three tier branches
    arrow(7.3, 7.7, 3.3); arrow(7.3, 7.7, 2.95); arrow(7.3, 7.7, 2.6)
    ax.plot([7.3, 7.3], [2.6, 3.3], color="#333", lw=1.2)

    box(7.7, 3.1, 2.1, 0.5, "T1  Llama-3.1-8B",  "", "#FFF3CD")
    box(7.7, 2.6, 2.1, 0.5, "T2  Llama-3.1-70B", "", "#FFE0B2")
    box(7.7, 2.1, 2.1, 0.5, "T3  Qwen2.5-72B",   "", "#FFCCCC")

    # Score annotation
    ax.text(3.5, 2.35, "score ∈ [0,1]", ha="center", fontsize=7.5,
            color="#1B7837", style="italic")
    ax.text(5.95, 2.35, "score > θ_h → T1\nθ_l < s ≤ θ_h → T2\nscore ≤ θ_l → T3",
            ha="center", fontsize=6.8, color="#333")

    # Title & footnote
    ax.set_title("PRM-Guided Routing Pipeline", fontsize=10, pad=4, fontweight="bold")
    ax.text(5, 0.15, "VersaPRM scores each completed step; the controller selects the model tier for the next step.",
            ha="center", fontsize=7, color="#555")

    fig.tight_layout()
    save(fig, "fig1_system_diagram")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: PRM Signal Quality by Step Type (Exp 1)
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_prm_signal():
    # Data from Exp 1 full run analysis
    step_types  = ["Retrieval", "Tool Call", "Synthesis"]
    prms        = ["Qwen2.5-Math-PRM", "VersaPRM", "DG-PRM (local)", "AgentRM*"]
    colors      = [C["qwen"], C["versa"], C["dgprm"], C["agent"]]
    hatches     = ["///", "", "...", "xxx"]

    # Spearman correlations from exp1_analysis.py per-step-type output
    data = {
        "Qwen2.5-Math-PRM": [0.303, 0.038, 0.103],
        "VersaPRM":          [0.064, 0.149, 0.249],
        "DG-PRM (local)":    [0.126, 0.103, 0.128],
        "AgentRM*":          [-0.065, -0.106, -0.016],
    }

    n_types  = len(step_types)
    n_prms   = len(prms)
    x        = np.arange(n_types)
    width    = 0.18
    offsets  = np.linspace(-(n_prms-1)/2, (n_prms-1)/2, n_prms) * width

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bars = []
    for i, (prm, col, hatch) in enumerate(zip(prms, colors, hatches)):
        vals = data[prm]
        b = ax.bar(x + offsets[i], vals, width,
                   label=prm, color=col, alpha=0.85,
                   hatch=hatch, edgecolor="white", linewidth=0.5)
        bars.append(b)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(step_types)
    ax.set_ylabel("Spearman Correlation\n(score vs. human label)")
    ax.set_ylim(-0.18, 0.35)
    ax.set_title("Exp 1: PRM Step-Quality Signal by Step Type", fontweight="bold")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    ax.axhspan(-0.18, 0, alpha=0.04, color="red")
    ax.axhspan(0, 0.35, alpha=0.04, color="green")
    ax.text(2.85, -0.155, "anti-correlated", fontsize=7, color="#AA3333", ha="right")
    ax.text(2.85, 0.32, "positively correlated", fontsize=7, color="#1B7837", ha="right")
    ax.text(2.0, -0.001, "* degenerate output", fontsize=7, color="#888", ha="right", va="top")

    fig.tight_layout()
    save(fig, "fig2_prm_signal_by_step_type")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Cost–Accuracy Pareto Frontier (Exp 2)
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_pareto():
    with open(ROOT / "results/exp2/exp2_pareto.json") as f:
        pareto = json.load(f)
    with open(ROOT / "results/exp2/exp2_results.json") as f:
        fixed  = json.load(f)["results"]

    trim_pts = sorted(pareto["trim"],        key=lambda x: x["cost_norm_per_traj"])
    prm_pts  = sorted(pareto["prm_guided"],  key=lambda x: x["cost_norm_per_traj"])

    tc = np.array([p["cost_norm_per_traj"] for p in trim_pts]) / 1000
    ta = np.array([p["accuracy"]           for p in trim_pts])
    pc = np.array([p["cost_norm_per_traj"] for p in prm_pts])  / 1000
    pa = np.array([p["accuracy"]           for p in prm_pts])

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Pareto curves
    ax.plot(tc, ta, "--", color=C["trim"],  lw=1.8, label="TRIM sweep",       zorder=3)
    ax.plot(pc, pa, "-",  color=C["prm"],   lw=2.2, label="PRM-Guided sweep", zorder=4)

    # Fill region between curves (PRM-Guided advantage)
    # Interpolate on shared cost axis
    cost_shared = np.linspace(max(tc.min(), pc.min()), min(tc.max(), pc.max()), 200)
    ta_interp = np.interp(cost_shared, tc, ta)
    pa_interp = np.interp(cost_shared, pc, pa)
    ax.fill_between(cost_shared, ta_interp, pa_interp,
                    where=(pa_interp >= ta_interp),
                    alpha=0.15, color=C["prm"], label="PRM-Guided advantage")

    # Fixed policy points
    label_map = {
        "Always-Cheap (T1)":    ("AlwaysCheap",  C["daao"],    "v"),
        "Uniform (T2)":         ("Uniform (T2)", C["uniform"], "s"),
        "Always-Frontier (T3)": ("Frontier",     C["frontier"],"^"),
        "DAAO":                 ("DAAO",         C["daao"],    "D"),
        "BAAR":                 ("BAAR",         C["baar"],    "P"),
    }
    for r in fixed:
        if r["policy"] in label_map:
            lbl, col, mk = label_map[r["policy"]]
            ax.scatter(r["cost_norm_per_traj"]/1000, r["accuracy"],
                       color=col, marker=mk, s=60, zorder=5,
                       edgecolors="white", linewidth=0.5, label=lbl)

    # Annotate key gap
    ax.annotate("+10pp at\nsame cost",
                xy=(320, 0.385), xytext=(240, 0.44),
                arrowprops=dict(arrowstyle="->", color=C["prm"], lw=1.0),
                fontsize=7.5, color=C["prm"], ha="center")

    ax.set_xlabel("Normalised Cost per Trajectory (×10³ token-units)")
    ax.set_ylabel("Expected Task Accuracy")
    ax.set_title("Exp 2: Cost–Accuracy Pareto Frontier", fontweight="bold")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.9)
    ax.set_xlim(left=40)
    ax.set_ylim(bottom=0.12)

    fig.tight_layout()
    save(fig, "fig3_pareto_frontier")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Routing Quality Summary (Exp 3 scatter + Exp 10 bar)
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_routing_quality():
    fig = plt.figure(figsize=(6.5, 2.9))
    gs  = gridspec.GridSpec(1, 2, wspace=0.38)

    # ── Panel A: Exp 3 — Routing Precision vs Exp1 Spearman ──────────────────
    ax1 = fig.add_subplot(gs[0])

    prm_pts = [
        ("VersaPRM",         0.1664, 0.536, C["versa"],    "o"),
        ("Qwen-Math-PRM",    0.1108, 0.421, C["qwen"],     "s"),
        ("DG-PRM",           0.1210, 0.434, C["dgprm"],    "D"),
        ("AgentRM*",        -0.0569, 0.459, C["agent"],    "^"),
        ("Random",           0.000,  0.465, C["random"],   "v"),
        ("Oracle",           1.000,  0.865, C["oracle"],   "*"),
    ]

    for name, sp, prec, col, mk in prm_pts:
        ax1.scatter(sp, prec, color=col, marker=mk, s=80 if mk != "*" else 140,
                    edgecolors="#333", linewidth=0.5, zorder=4)
        offset = {"VersaPRM": (0.01, 0.010), "Qwen-Math-PRM": (0.01, -0.020),
                  "DG-PRM": (-0.01, 0.012), "AgentRM*": (0.01, -0.022),
                  "Random": (0.01, 0.010), "Oracle": (-0.02, -0.025)}.get(name, (0.01, 0.01))
        ax1.text(sp + offset[0], prec + offset[1], name,
                 fontsize=6.5, ha="left" if offset[0] > 0 else "right", color="#333")

    # Random baseline reference line
    ax1.axhline(0.465, color=C["random"], lw=1.0, ls="--", alpha=0.7, label="Random baseline")

    ax1.set_xlabel("Exp 1 Spearman Correlation")
    ax1.set_ylabel("Routing Precision\n(bad steps correctly escalated)")
    ax1.set_title("(a) Signal Quality → Routing Precision", fontsize=9, fontweight="bold")
    ax1.set_xlim(-0.15, 1.08)
    ax1.set_ylim(0.35, 0.92)
    ax1.text(0.95, 0.40, "* degenerate", fontsize=6.5, color="#888", ha="right")

    # ── Panel B: Exp 10 — Retrieval Context Impact ───────────────────────────
    ax2 = fig.add_subplot(gs[1])

    labels = ["A. Baseline\n(q+step)", "B. Exp9\n(q_comp)", "C. Exp10a\n(+retrieval)", "D. Exp10b\n(+retr_comp)"]
    accs   = [0.347, 0.390, 0.413, 0.402]
    cols   = [C["uniform"], C["qwen"], C["prm"], "#5FA858"]
    bars   = ax2.bar(range(4), accs, color=cols, alpha=0.85, edgecolor="white", linewidth=0.5, width=0.6)

    for i, (b, a) in enumerate(zip(bars, accs)):
        delta = a - accs[0]
        label = f"{a:.3f}" if i == 0 else f"{a:.3f}\n({delta:+.3f})"
        ax2.text(b.get_x() + b.get_width()/2, a + 0.004, label,
                 ha="center", va="bottom", fontsize=7, fontweight="bold" if i in (2,3) else "normal")

    ax2.axhline(accs[0], color=C["uniform"], lw=1.0, ls="--", alpha=0.6)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_ylabel("Routing Accuracy")
    ax2.set_ylim(0.30, 0.46)
    ax2.set_title("(b) Retrieval Context Impact (Exp 10)", fontsize=9, fontweight="bold")

    fig.suptitle("Exp 3 & 10: Routing Quality Analysis", fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig4_routing_quality_summary")
    plt.close(fig)


# ─── Run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures ...")
    fig1_system();          print("  [1/4] System diagram")
    fig2_prm_signal();      print("  [2/4] PRM signal by step type")
    fig3_pareto();          print("  [3/4] Pareto frontier")
    fig4_routing_quality(); print("  [4/4] Routing quality summary")
    print(f"\nAll figures saved to {FIGS}/")
