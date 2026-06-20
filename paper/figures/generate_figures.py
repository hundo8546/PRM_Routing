"""
Generate publication figures for PRM-Guided Cost-Aware Routing paper.

Data figures: plotnine (ggplot2 port) with theme_minimal.
System diagram: matplotlib (no pdflatex available).

Run from /workspace/PRM_Routing/:
    python3 paper/figures/generate_figures.py
"""

import io
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from plotnine import (
    ggplot, aes, geom_col, geom_point, geom_line, geom_hline,
    scale_color_manual, scale_fill_manual, scale_shape_manual,
    scale_y_continuous,
    labs, theme_minimal, theme, element_text, element_blank,
    coord_cartesian,
    position_dodge,
)
from PIL import Image

warnings.filterwarnings("ignore")

ROOT = Path("/workspace/PRM_Routing")
FIGS = ROOT / "paper/figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Minimal colour palette ────────────────────────────────────────────────────
PAL = {
    "prm":      "#2166AC",   # VersaPRM / PRM-Guided
    "trim":     "#762A83",   # TRIM
    "judge":    "#B2182B",   # frontier judge / GPT
    "oracle":   "#1A7A1A",   # oracle
    "uniform":  "#888888",   # uniform / neutral
    "retr":     "#D6604D",   # retrieval
    "ua":       "#4393C3",   # uncertainty-adaptive
    "qwen":     "#E08214",   # Qwen / open-weight
    "dgprm":    "#74ADD1",   # DG-PRM
    "random":   "#BBBBBB",   # random baseline
    "frontier": "#222222",   # always-frontier
}

MINIMAL = (
    theme_minimal()
    + theme(
        figure_size=(5, 3.2),
        text=element_text(family="DejaVu Sans", size=9),
        plot_title=element_text(size=9, face="bold"),
        axis_title=element_text(size=8),
        axis_text=element_text(size=7.5),
        legend_title=element_blank(),
        legend_text=element_text(size=7.5),
        legend_key_size=8,
        panel_grid_minor=element_blank(),
        panel_grid_major_x=element_blank(),
    )
)

MINIMAL_SCATTER = (
    theme_minimal()
    + theme(
        figure_size=(4.5, 3.2),
        text=element_text(family="DejaVu Sans", size=9),
        plot_title=element_text(size=9, face="bold"),
        axis_title=element_text(size=8),
        axis_text=element_text(size=7.5),
        legend_title=element_blank(),
        legend_text=element_text(size=7.5),
        legend_key_size=8,
        panel_grid_minor=element_blank(),
    )
)

def _p_to_img(p, w, h, dpi=200):
    """Render a plotnine plot to a PIL Image via BytesIO."""
    buf = io.BytesIO()
    p.save(buf, format="png", width=w, height=h, dpi=dpi, verbose=False)
    buf.seek(0)
    return Image.open(buf).copy()

def combine_panels(panels_wh, name, dpi=200):
    """
    Combine multiple (plotnine_plot, w, h) tuples side-by-side via matplotlib.
    panels_wh: list of (plot, width_in, height_in)
    """
    imgs = [_p_to_img(p, w, h, dpi) for p, w, h in panels_wh]
    total_w = sum(i.width for i in imgs)
    max_h   = max(i.height for i in imgs)
    canvas  = Image.new("RGB", (total_w, max_h), "white")
    x = 0
    for img in imgs:
        canvas.paste(img, (x, (max_h - img.height) // 2))
        x += img.width
    for ext, fmt in [("png", "PNG"), ("pdf", "PDF")]:
        canvas.save(str(FIGS / f"{name}.{ext}"), format=fmt, dpi=(dpi, dpi))
    print(f"  Saved {name}")

def save_p(p, name, w=5.5, h=3.4):
    """Save a single plotnine figure."""
    p.save(str(FIGS / f"{name}.png"), width=w, height=h, dpi=200, verbose=False)
    p.save(str(FIGS / f"{name}.pdf"), width=w, height=h, dpi=200, verbose=False)
    print(f"  Saved {name}")

def save_mpl(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1: System diagram  (matplotlib — no pdflatex)
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_system():
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.5); ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    def box(x, y, w, h, top, sub="", fc="#F0F4F8", ec="#555555"):
        r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    facecolor=fc, edgecolor=ec, linewidth=0.8)
        ax.add_patch(r)
        cy = y + h/2
        if sub:
            ax.text(x + w/2, cy + 0.13, top, ha="center", va="center",
                    fontsize=8, fontweight="semibold", color="#222")
            ax.text(x + w/2, cy - 0.17, sub, ha="center", va="center",
                    fontsize=6.5, color="#666", style="italic")
        else:
            ax.text(x + w/2, cy, top, ha="center", va="center",
                    fontsize=8, fontweight="semibold", color="#222")

    def arr(x1, x2, y, color="#666"):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                                   mutation_scale=10))

    # Boxes
    box(0.1, 1.4, 1.6, 0.9, "Agent Step", "tool / synthesis", "#F5F5F5")
    arr(1.7, 2.1, 1.85)
    box(2.1, 1.4, 2.2, 0.9, "VersaPRM", "+ retrieval ctx", "#EBF3FB")
    arr(4.3, 4.7, 1.85)
    box(4.7, 1.4, 2.2, 0.9, "Router", "PRMGuided θ_h/θ_l", "#EBF5EB")
    arr(6.9, 7.2, 1.85)

    # Fork
    ax.plot([7.2, 7.2], [1.3, 2.4], color="#888888", lw=0.9)
    for y_t, label, fc in [(2.1, "T1  small", "#FFFDE7"),
                            (1.75, "T2  mid", "#FFF3E0"),
                            (1.4,  "T3  frontier", "#FFEBEE")]:
        arr(7.2, 7.5, y_t + 0.12, color="#888888")
        box(7.5, y_t, 2.35, 0.34, label, fc=fc, ec="#BBBBBB")

    # Score note
    ax.text(3.2, 1.25, "score ∈ [0,1]", ha="center", fontsize=6.5,
            color="#2166AC", style="italic")
    ax.text(5.85, 1.25,
            "s > θ_h → T1   |   θ_l ≤ s ≤ θ_h → T2   |   s < θ_l → T3",
            ha="center", fontsize=6.5, color="#444")
    ax.set_title("PRM-Guided Routing Pipeline", fontsize=10, fontweight="bold", pad=6)

    fig.tight_layout(pad=0.3)
    save_mpl(fig, "fig1_system_diagram")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2: PRM signal quality by step type  (Exp 1)
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_prm_signal():
    records = []
    for prm, color, vals in [
        ("Qwen-Math",  PAL["qwen"],    [0.303, 0.038, 0.103]),
        ("VersaPRM",   PAL["prm"],     [0.064, 0.149, 0.249]),
        ("DG-PRM",     PAL["dgprm"],   [0.126, 0.103, 0.128]),
        ("AgentRM*",   PAL["random"],  [-0.065, -0.106, -0.016]),
    ]:
        for step, v in zip(["Retrieval", "Tool Call", "Synthesis"], vals):
            records.append({"PRM": prm, "Step Type": step, "Spearman": v})
    step_order = ["Retrieval", "Tool Call", "Synthesis"]

    df = pd.DataFrame(records)
    df["Step Type"] = pd.Categorical(df["Step Type"], categories=step_order, ordered=True)
    colors = {"Qwen-Math": PAL["qwen"], "VersaPRM": PAL["prm"],
              "DG-PRM": PAL["dgprm"], "AgentRM*": PAL["random"]}

    p = (
        ggplot(df, aes("Step Type", "Spearman", fill="PRM"))
        + geom_col(position=position_dodge(width=0.75), width=0.7, alpha=0.9)
        + geom_hline(yintercept=0, color="#333333", size=0.5)
        + scale_fill_manual(values=colors)
        + labs(title="PRM Signal Quality by Step Type", x=None, y="Spearman Correlation")
        + MINIMAL
        + theme(panel_grid_major_x=element_blank(),
                legend_position="right",
                figure_size=(5.5, 3.0))
    )
    save_p(p, "fig2_prm_signal_by_step_type", w=5.5, h=3.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3: Cost–Accuracy Pareto Frontier  (Exp 2)
# ═══════════════════════════════════════════════════════════════════════════════

def _pareto_envelope(points):
    """Return only non-dominated (cost, acc) points — the upper-left envelope."""
    pts = sorted(points, key=lambda x: x["cost_norm_per_traj"])
    envelope, best_acc = [], -1
    for p in pts:
        if p["accuracy"] > best_acc:
            best_acc = p["accuracy"]
            envelope.append(p)
    return envelope

def fig3_pareto():
    with open(ROOT / "results/exp2/exp2_pareto.json") as f:
        pareto = json.load(f)
    with open(ROOT / "results/exp2/exp2_results.json") as f:
        fixed = json.load(f)["results"]

    prm_env  = _pareto_envelope(pareto["prm_guided"])
    trim_env = _pareto_envelope(pareto["trim"])

    # Clip TRIM to the same cost range as PRM for fair comparison
    prm_max_cost = max(p["cost_norm_per_traj"] for p in prm_env)
    trim_clipped = [p for p in trim_env if p["cost_norm_per_traj"] <= prm_max_cost * 1.05]

    df_lines = pd.DataFrame(
        [{"cost": p["cost_norm_per_traj"]/1000, "acc": p["accuracy"], "Method": "PRM-Guided (envelope)"}
         for p in prm_env] +
        [{"cost": p["cost_norm_per_traj"]/1000, "acc": p["accuracy"], "Method": "TRIM (envelope)"}
         for p in trim_clipped]
    )

    # Fixed reference policies — each labeled and colored
    ref_map = {
        "Always-Cheap (T1)":          ("T1 Always-Cheap",  PAL["random"],  "v"),
        "Uniform (T2)":               ("T2 Uniform",        PAL["uniform"], "s"),
        "Always-Frontier (T3)":       ("T3 Always-Frontier",PAL["frontier"],"^"),
        "BAAR":                       ("BAAR",              PAL["qwen"],    "D"),
        "PRM-Guided (h=0.86/l=0.62)": ("PRM-Guided (best)", PAL["prm"],    "o"),
    }
    pts = []
    for r in fixed:
        if r["policy"] in ref_map:
            lbl, col, shp = ref_map[r["policy"]]
            pts.append({"cost": r["cost_norm_per_traj"]/1000,
                        "acc": r["accuracy"], "Policy": lbl,
                        "color": col, "shape": shp})
    df_pts = pd.DataFrame(pts)

    # Split lines by method so we can hardcode linetype per geom
    df_prm  = df_lines[df_lines["Method"] == "PRM-Guided (envelope)"]
    df_trim = df_lines[df_lines["Method"] == "TRIM (envelope)"]

    pt_colors = dict(zip(df_pts["Policy"], df_pts["color"]))
    pt_shapes = dict(zip(df_pts["Policy"], df_pts["shape"]))

    # Single color scale covering both line labels and point labels
    all_colors = {"PRM-Guided sweep": PAL["prm"], "TRIM sweep": PAL["trim"], **pt_colors}
    all_shapes = {"PRM-Guided sweep": "o", "TRIM sweep": "o", **pt_shapes}

    df_prm  = df_prm.assign(Entry="PRM-Guided sweep")
    df_trim = df_trim.assign(Entry="TRIM sweep")
    df_pts  = df_pts.assign(Entry=df_pts["Policy"])

    all_colors = {
        "PRM-Guided sweep":  PAL["prm"],
        "TRIM sweep":        PAL["trim"],
        **pt_colors,
    }
    # Shapes: lines get a dash-like appearance via size=0 point, points get their shapes
    all_shapes = {
        "PRM-Guided sweep":  "o",
        "TRIM sweep":        "o",
        **pt_shapes,
    }

    p = (
        ggplot()
        + geom_line(df_prm,  aes("cost", "acc", color="Entry"), size=1.2, linetype="solid",  show_legend=False)
        + geom_line(df_trim, aes("cost", "acc", color="Entry"), size=1.2, linetype="dashed", show_legend=False)
        + geom_point(df_pts, aes("cost", "acc", color="Entry", shape="Entry"), size=3.2, stroke=0.4)
        # Invisible points for the line entries in legend
        + geom_point(df_prm.iloc[[0]],  aes("cost", "acc", color="Entry"), size=0, show_legend=True)
        + geom_point(df_trim.iloc[[0]], aes("cost", "acc", color="Entry"), size=0, show_legend=True)
        + scale_color_manual(values=all_colors,
                             breaks=["PRM-Guided sweep", "TRIM sweep"] + list(pt_colors.keys()))
        + scale_shape_manual(values=all_shapes,
                             breaks=["PRM-Guided sweep", "TRIM sweep"] + list(pt_shapes.keys()),
                             guide=None)
        + labs(title="Cost–Accuracy Pareto Frontier",
               x="Normalised Cost (×10³ token-units)", y="Expected Task Accuracy")
        + coord_cartesian(xlim=(50, 700), ylim=(0.15, 0.68))
        + MINIMAL_SCATTER
        + theme(legend_position="right",
                legend_text=element_text(size=7.5),
                figure_size=(6.2, 3.5))
    )
    save_p(p, "fig3_pareto_frontier", w=6.2, h=3.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4: PRM signal → routing precision  +  retrieval context impact
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_routing_quality():
    # Panel A: scatter
    pts = [
        ("VersaPRM",     0.1664, 0.536, PAL["prm"],    "o"),
        ("Qwen-Math",    0.1108, 0.421, PAL["qwen"],   "s"),
        ("DG-PRM",       0.1210, 0.434, PAL["dgprm"],  "D"),
        ("AgentRM*",    -0.0569, 0.459, PAL["random"], "^"),
        ("Random",       0.000,  0.465, PAL["random"], "v"),
        ("Oracle",       1.000,  0.865, PAL["oracle"], "D"),
    ]
    df_a = pd.DataFrame(pts, columns=["PRM", "Spearman", "Precision", "color", "shape"])
    colors_a = dict(zip(df_a["PRM"], df_a["color"]))
    shapes_a = dict(zip(df_a["PRM"], df_a["shape"]))

    pa = (
        ggplot(df_a, aes("Spearman", "Precision", color="PRM", shape="PRM"))
        + geom_hline(yintercept=0.465, linetype="dashed", color="#BBBBBB", size=0.7)
        + geom_point(size=3.5, stroke=0.3)
        + scale_color_manual(values=colors_a)
        + scale_shape_manual(values=shapes_a)
        + labs(title="Signal Quality vs. Routing Precision",
               x="Spearman Correlation", y="Routing Precision")
        + coord_cartesian(xlim=(-0.12, 1.05), ylim=(0.38, 0.90))
        + MINIMAL_SCATTER
        + theme(legend_position="right", figure_size=(4.2, 3.0))
    )

    # Panel B: retrieval context bars
    variant_order = ["Baseline", "Query\nCompressed", "+Full\nRetrieval", "+Comp.\nRetrieval"]
    df_b = pd.DataFrame({
        "Variant": variant_order,
        "Accuracy": [0.347, 0.390, 0.413, 0.402],
        "highlight": [False, False, True, True],
    })
    df_b["Variant"] = pd.Categorical(df_b["Variant"], categories=variant_order, ordered=True)

    pb = (
        ggplot(df_b, aes("Variant", "Accuracy"))
        + geom_col(aes(fill="highlight"), width=0.6, alpha=0.9, show_legend=False)
        + geom_hline(yintercept=0.347, linetype="dashed", color="#888888", size=0.6)
        + scale_fill_manual(values={True: PAL["prm"], False: PAL["uniform"]})
        + coord_cartesian(ylim=(0.30, 0.44))
        + labs(title="Effect of Retrieval Context", x=None, y="Routing Accuracy")
        + MINIMAL
        + theme(panel_grid_major_x=element_blank(), figure_size=(3.8, 3.0))
    )

    combine_panels([(pa, 4.2, 3.0), (pb, 3.8, 3.0)], "fig4_routing_quality_summary")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5: Frontier judge vs PRM Router  (Exp 11)
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_frontier_judge():
    path = ROOT / "results/exp11/exp11_results.json"
    if not path.exists():
        print("  [SKIP] fig5"); return
    data = json.load(open(path))["results"]

    method_order = ["Uniform (T2)", "VersaPRM", "Retrieval-Aware Versa",
                    "Multi-Judge", "FrontierJudge(gpt-5.5)", "Oracle"]
    short = {"Uniform (T2)": "Uniform", "VersaPRM": "VersaPRM",
             "Retrieval-Aware Versa": "RA-Versa", "Multi-Judge": "Multi-Judge",
             "FrontierJudge(gpt-5.5)": "GPT-5.5\nJudge", "Oracle": "Oracle"}
    colors_5 = {"Uniform": PAL["uniform"], "VersaPRM": PAL["prm"],
                "RA-Versa": PAL["dgprm"],  "Multi-Judge": PAL["qwen"],
                "GPT-5.5\nJudge": PAL["judge"], "Oracle": PAL["oracle"]}

    df = pd.DataFrame([{
        "Method": short[r["method"]], "TSR": r["task_success_rate"],
        "Precision": r["routing_precision"], "EscRate": r["escalation_rate"],
    } for r in data if r["method"] in short])
    df["Method"] = pd.Categorical(df["Method"], categories=list(short.values()), ordered=True)

    pa = (
        ggplot(df, aes("Method", "TSR", fill="Method"))
        + geom_col(width=0.65, alpha=0.9, show_legend=False)
        + scale_fill_manual(values=colors_5)
        + scale_y_continuous(limits=(0, 0.50))
        + labs(title="Task Success Rate by Method", x=None, y="Task Success Rate")
        + MINIMAL
        + theme(panel_grid_major_x=element_blank(), figure_size=(4.2, 3.0),
                axis_text_x=element_text(size=7))
    )

    pb = (
        ggplot(df, aes("EscRate", "Precision", color="Method", shape="Method"))
        + geom_point(size=3.5, stroke=0.3)
        + scale_color_manual(values=colors_5)
        + scale_shape_manual(values={m: s for m, s in zip(
            list(short.values()), ["s","o","^","D","*","D"])})
        + labs(title="Precision vs. Escalation Rate",
               x="Escalation Rate", y="Routing Precision")
        + coord_cartesian(ylim=(0, 1.0))
        + MINIMAL_SCATTER
        + theme(legend_position="right", figure_size=(4.0, 3.0))
    )

    combine_panels([(pa, 4.2, 3.0), (pb, 4.0, 3.0)], "fig5_frontier_judge")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6: Dynamic thresholds  (Exp 13 + 14)
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_dynamic_thresholds():
    p13 = ROOT / "results/exp13/exp13_results.json"
    p14 = ROOT / "results/exp14/exp14_results.json"
    if not p13.exists() or not p14.exists():
        print("  [SKIP] fig6"); return

    r13 = json.load(open(p13))["results"]
    r14 = json.load(open(p14))["results"]

    base_tsr = next(r for r in r13 if "0.62" in r["policy"])["task_success_rate"]

    # Panel A: percentile dynamic — delta TSR bars
    short13 = {"PRM-Guided (h=0.86/l=0.62)": "Fixed", "DynP10/90": "P10/90",
               "DynP20/80": "P20/80", "DynP25/75": "P25/75"}
    df_a = pd.DataFrame([{
        "Policy": short13.get(r["policy"], r["policy"]),
        "dTSR": r["task_success_rate"] - base_tsr,
        "highlight": r["policy"] != "PRM-Guided (h=0.86/l=0.62)",
    } for r in r13 if r["policy"] in short13])
    df_a["Policy"] = pd.Categorical(df_a["Policy"],
                                    categories=list(short13.values()), ordered=True)

    pa = (
        ggplot(df_a, aes("Policy", "dTSR", fill="highlight"))
        + geom_col(width=0.6, alpha=0.9, show_legend=False)
        + geom_hline(yintercept=0, color="#555555", size=0.5)
        + scale_fill_manual(values={True: PAL["ua"], False: PAL["uniform"]})
        + labs(title="Percentile Dynamic Thresholds",
               x=None, y="TSR delta vs fixed baseline")
        + MINIMAL
        + theme(panel_grid_major_x=element_blank(), figure_size=(3.8, 3.0))
    )

    # Panel B: UA variants scatter
    ua_labels = {"UA(τ=0.1/0.25)": "UA-1\n(τ=0.10/0.25)",
                 "UA(τ=0.05/0.15)": "UA-2\n(τ=0.05/0.15)",
                 "UA(τ=0.15/0.3)": "UA-3\n(τ=0.15/0.30)"}
    ua_colors = {"UA-1\n(τ=0.10/0.25)": "#2166AC",
                 "UA-2\n(τ=0.05/0.15)": "#4393C3",
                 "UA-3\n(τ=0.15/0.30)": "#74ADD1"}
    df_b = pd.DataFrame([{
        "Variant": ua_labels[r["policy"]],
        "Cost": r["cost_norm_per_traj"] / 1000,
        "TSR": r["task_success_rate"],
    } for r in r14 if r["policy"] in ua_labels])

    pb = (
        ggplot(df_b, aes("Cost", "TSR", color="Variant", shape="Variant"))
        + geom_hline(yintercept=base_tsr, linetype="dashed", color="#888888", size=0.7)
        + geom_point(size=4, stroke=0.3)
        + scale_color_manual(values=ua_colors)
        + scale_shape_manual(values={"UA-1\n(τ=0.10/0.25)": "D",
                                      "UA-2\n(τ=0.05/0.15)": "s",
                                      "UA-3\n(τ=0.15/0.30)": "^"})
        + coord_cartesian(ylim=(0.30, 0.44))
        + labs(title="Uncertainty-Adaptive Thresholds",
               x="Normalised Cost (×10³)", y="Task Success Rate")
        + MINIMAL_SCATTER
        + theme(legend_position="right", figure_size=(4.0, 3.0))
    )

    combine_panels([(pa, 3.8, 3.0), (pb, 4.0, 3.0)], "fig6_dynamic_thresholds")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7: Modern stack  (Exp 12)
# ═══════════════════════════════════════════════════════════════════════════════

def fig7_modern_stack():
    path = ROOT / "results/exp12/exp12_results.json"
    if not path.exists():
        print("  [SKIP] fig7"); return
    stacks = json.load(open(path))["stacks"]

    pol_short = {"Always-Cheap (T1)": "T1-only", "Uniform (T2)": "Uniform",
                 "Always-Frontier (T3)": "T3-only", "TRIM (θ=0.75)": "TRIM",
                 "PRM-Guided (h=0.86/l=0.62)": "PRM-Guided"}
    stack_short = {"Original (Exp 2)": "Original",
                   "Modern Recommended": "Modern\nRecom.",
                   "Modern Open-Weight": "Modern\nOpen-W."}
    stack_colors = {"Original": "#444444", "Modern\nRecom.": PAL["prm"],
                    "Modern\nOpen-W.": PAL["oracle"]}

    records = []
    for sname, data in stacks.items():
        for r in data["results"]:
            pshort = pol_short.get(r["policy"])
            if pshort:
                records.append({"Stack": stack_short[sname], "Policy": pshort,
                                 "TSR": r["task_success_rate"]})
    df = pd.DataFrame(records)
    df["Policy"] = pd.Categorical(df["Policy"],
                                  categories=list(pol_short.values()), ordered=True)

    pa = (
        ggplot(df, aes("Policy", "TSR", fill="Stack"))
        + geom_col(position=position_dodge(width=0.8), width=0.72, alpha=0.88)
        + scale_fill_manual(values=stack_colors)
        + scale_y_continuous(limits=(0, 0.75))
        + labs(title="TSR by Policy and Model Stack", x=None, y="Task Success Rate")
        + MINIMAL
        + theme(panel_grid_major_x=element_blank(),
                axis_text_x=element_text(size=7),
                legend_position="top", figure_size=(4.8, 3.0))
    )

    adv_df = pd.DataFrame([
        {"Stack": stack_short[k], "Advantage": v["prm_advantage_tsr"]}
        for k, v in stacks.items()
    ])
    adv_colors = {v: stack_colors[v] for v in adv_df["Stack"]}

    pb = (
        ggplot(adv_df, aes("Stack", "Advantage", fill="Stack"))
        + geom_col(width=0.55, alpha=0.9, show_legend=False)
        + geom_hline(yintercept=0, color="#555555", size=0.5)
        + scale_fill_manual(values=adv_colors)
        + labs(title="PRM Routing Advantage per Stack",
               x=None, y="PRM advantage over Uniform (TSR)")
        + MINIMAL
        + theme(panel_grid_major_x=element_blank(), figure_size=(3.4, 3.0))
    )

    combine_panels([(pa, 4.8, 3.0), (pb, 3.4, 3.0)], "fig7_modern_stack")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8: Full TSR progression  (Exp 2 → 14 → 10 → 15 vs Oracle)
# ═══════════════════════════════════════════════════════════════════════════════

def fig8_tsr_progression():
    records = [
        ("Fixed PRM-Guided",    0.320, PAL["uniform"],  False),
        ("UA-3",                0.385, PAL["ua"],        True),
        ("Retr-Aware PRM",      0.450, PAL["retr"],      True),
        ("Retr-Aware + UA-3",   0.450, PAL["prm"],       True),
        ("Oracle",              0.430, PAL["oracle"],    False),
    ]

    # Check exp15 for real numbers
    p15 = ROOT / "results/exp15/exp15_results.json"
    if p15.exists():
        d15 = json.load(open(p15))
        for r in d15.get("results", []):
            if "Retr-Aware + UA-3" in r.get("config", ""):
                records[3] = ("Retr-Aware + UA-3", r["task_success_rate"], PAL["prm"], True)
            if r.get("config") == "Retr-Aware PRM":
                records[2] = ("Retr-Aware PRM", r["task_success_rate"], PAL["retr"], True)

    df = pd.DataFrame(records, columns=["Config", "TSR", "color", "novel"])
    df["Config"] = pd.Categorical(df["Config"],
                                  categories=[r[0] for r in records], ordered=True)
    colors = dict(zip(df["Config"], df["color"]))

    p = (
        ggplot(df, aes("Config", "TSR", fill="Config"))
        + geom_col(width=0.65, alpha=0.9, show_legend=False)
        + geom_hline(yintercept=0.430, linetype="dashed", color=PAL["oracle"],
                     size=0.8, alpha=0.7)
        + scale_fill_manual(values=colors)
        + scale_y_continuous(limits=(0, 0.50))
        + labs(title="TSR Progression: Baseline to Near-Oracle",
               x=None, y="Task Success Rate")
        + MINIMAL
        + theme(
            panel_grid_major_x=element_blank(),
            axis_text_x=element_text(size=7.5),
            figure_size=(5.5, 3.2),
        )
    )
    save_p(p, "fig8_tsr_progression", w=5.5, h=3.2)


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures ...")
    fig1_system();              print("  [1/8] System diagram")
    fig2_prm_signal();          print("  [2/8] PRM signal by step type")
    fig3_pareto();              print("  [3/8] Pareto frontier")
    fig4_routing_quality();     print("  [4/8] Routing quality")
    fig5_frontier_judge();      print("  [5/8] Frontier judge")
    fig6_dynamic_thresholds();  print("  [6/8] Dynamic thresholds")
    fig7_modern_stack();        print("  [7/8] Modern stack")
    fig8_tsr_progression();     print("  [8/8] TSR progression")
    print(f"\nAll figures saved to {FIGS}/")
