"""
Phase 2a: plots and a summary table for the d12 SFT recipe sweep (post-training open question 1).
Reads results/logs/nanochat-sft/<run>/metrics.jsonl (val bpb and the 400-problem ChatCORE curve during SFT) and
the full-test chat_eval logs (results/d12/sft_sweep/stage_logs/<run>_chat_eval.log, plus the phase-1 defaults in
results/d12/stage_logs/). Writes results/d12/plots/sft_sweep_curves.png, sft_sweep_final.png and
results/d12/sft_sweep/summary.md.
    python results/plot_sft_sweep.py
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from plot import PALETTE, INK, INK2, tidy  # noqa: E402

CHANCE = 0.25
# name -> (label, colour, linestyle)
RUNS = {
    "d12_ar_sft":           ("AR, default recipe (phase 1)",            PALETTE["ar"],     ":"),
    "d12_mdm_sft":          ("Diffusion, default recipe (phase 1)",     PALETTE["mdm"],    ":"),
    "sft2_ar_m9_lr0.3_x2":  ("AR, 3x MMLU share, 0.3x LR",              PALETTE["ar"],     "-"),
    "sft2_mdm_m9_lr0.3_x2": ("Diffusion, 3x MMLU share, 0.3x LR",       PALETTE["mdm"],    "-"),
    "sft2_mdm_m9_lr0.3_ep2": ("Diffusion, same recipe, two passes",     PALETTE["mdm"],    "--"),
    "sft2_mdm_m9_lr0.1_x2": ("Diffusion, 3x MMLU share, 0.1x LR",       PALETTE["mdm_2x"], "-"),
    "sft2_mdm_m3_lr0.3":    ("Diffusion, default share, 0.3x LR",       PALETTE["mdm_4x"], "-"),
}
EVAL_LOGS = {
    "d12_ar_sft": "results/d12/stage_logs/d12_ar_chat_eval_sft.log",
    "d12_mdm_sft": "results/d12/stage_logs/d12_mdm_chat_eval_sft.log",
}
TASKS = ["ARC-Easy", "ARC-Challenge", "MMLU"]


def load_metrics(name):
    path = f"results/logs/nanochat-sft/{name}/metrics.jsonl"
    if not os.path.exists(path):
        return None
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if not any("chatcore_cat" in r for r in recs):
        return None
    cfg_path = f"results/logs/nanochat-sft/{name}/config.json"
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    return {"name": name, "recs": recs, "cfg": cfg}


def curve(run, key):
    pts = [(r["step"], r[key]) for r in run["recs"] if key in r and r[key] is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def final_evals(name):
    path = EVAL_LOGS.get(name, f"results/d12/sft_sweep/stage_logs/{name}_chat_eval.log")
    out = {}
    if os.path.exists(path):
        for m in re.finditer(r"^(ARC-Easy|ARC-Challenge|MMLU|GSM8K|HumanEval) accuracy: ([\d.]+)%", open(path).read(), re.M):
            out[m.group(1)] = float(m.group(2)) / 100
    return out


def plot_curves(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for run in runs:
        label, color, ls = RUNS[run["name"]]
        for ax, key in zip(axes, ("val/bpb", "chatcore_cat")):
            xs, ys = curve(run, key)
            if xs:
                ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2, label=label, marker="o" if key == "chatcore_cat" else None, markersize=4)
    axes[1].axhline(0, color=INK2, linewidth=0.8, linestyle="-")
    tidy(axes[0], "SFT optimizer step", "validation bits per byte", "Likelihood on the SFT validation mixture")
    tidy(axes[1], "SFT optimizer step", "ChatCORE, categorical tasks", "Multiple choice during SFT (400 problems per task)")
    axes[0].text(0.01, 0.02, "AR: exact NLL. Diffusion: ELBO upper bound.", transform=axes[0].transAxes, fontsize=8, color=INK2)
    axes[1].legend(fontsize=8, frameon=False, loc="center right")
    for ax in axes:
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_final(names, evals, out):
    names = [n for n in names if evals.get(n)]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.8 / max(1, len(names))
    for i, n in enumerate(names):
        label, color, ls = RUNS[n]
        ys = [evals[n].get(t, 0) for t in TASKS]
        xs = [j + (i - (len(names) - 1) / 2) * width for j in range(len(TASKS))]
        bars = ax.bar(xs, ys, width=width * 0.92, color=color, alpha=0.45 if ls == ":" else (0.75 if ls == "--" else 1.0),
                      edgecolor="white", linewidth=1, label=label)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y + 0.004, f"{100*y:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.axhline(CHANCE, color=INK2, linewidth=1, linestyle="--")
    ax.text(len(TASKS) - 0.55, CHANCE + 0.004, "chance", fontsize=8, color=INK2, ha="right")
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(TASKS)
    ax.set_ylim(0.2, max(0.42, max(max(evals[n].get(t, 0) for t in TASKS) for n in names) + 0.04))
    tidy(ax, "", "accuracy, full test set", "After SFT: multiple-choice accuracy by recipe")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summary(runs, evals):
    lines = ["| run | model | recipe | steps | final val bpb | ChatCORE_cat (400/task) | ARC-E | ARC-C | MMLU | GSM8K |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        n = run["name"]
        label = RUNS[n][0]
        model, recipe = label.split(", ", 1)
        _, ys = curve(run, "val/bpb")
        cx, cy = curve(run, "chatcore_cat")
        e = evals.get(n, {})
        fmt = lambda k: f"{100*e[k]:.1f}%" if k in e else "-"
        lines.append(f"| {n} | {model} | {recipe} | {cx[-1] if cx else '-'} | {ys[-1]:.3f} | {cy[-1]:+.3f} | {fmt('ARC-Easy')} | {fmt('ARC-Challenge')} | {fmt('MMLU')} | {fmt('GSM8K')} |")
    return "\n".join(lines)


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    runs = [r for r in (load_metrics(n) for n in RUNS) if r]
    evals = {r["name"]: final_evals(r["name"]) for r in runs}
    os.makedirs("results/d12/plots", exist_ok=True)
    os.makedirs("results/d12/sft_sweep", exist_ok=True)
    plot_curves(runs, "results/d12/plots/sft_sweep_curves.png")
    plot_final([r["name"] for r in runs], evals, "results/d12/plots/sft_sweep_final.png")
    table = summary(runs, evals)
    with open("results/d12/sft_sweep/summary.md", "w") as f:
        f.write("# d12 SFT recipe sweep (phase 2a)\n\nGenerated by results/plot_sft_sweep.py. Steps are optimizer steps at 524,288 tokens; "
                "a run ends after one pass over its mixture unless the epochs are raised, so the 3x-MMLU mixture gives 1375 steps per pass. "
                "Validation bpb is the exact NLL for AR and the ELBO upper bound for the diffusion model, on the same SFT validation mixture.\n\n"
                f"{table}\n\n![curves](../plots/sft_sweep_curves.png)\n\n![final](../plots/sft_sweep_final.png)\n")
    print(table)
    print("\nwrote results/d12/plots/sft_sweep_{curves,final}.png and results/d12/sft_sweep/summary.md")


if __name__ == "__main__":
    main()
