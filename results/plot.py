"""
Build the result plots and summary table from synced run logs.

    bash results/sync_logs.sh prime-d12 d12      # pull logs from the GPU box
    python results/plot.py --phase d12           # writes results/d12/plots/*.png and results/d12/summary.md

Reads results/logs/<project>/<run>/metrics.jsonl (+ config.json) written by nanochat's MetricsLogger, and
results/<phase>/base_eval/<tag>_<step>.csv written by scripts/base_eval.py.
"""
import argparse
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical palette (validated, fixed slot order): AR blue, MDM orange, then aqua / yellow for extra MDM budgets
PALETTE = {"ar": "#2a78d6", "mdm": "#eb6834", "mdm_2x": "#1baf7a", "mdm_4x": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load_run(run_dir):
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "metrics.jsonl")) if l.strip()]
    cfg_path = os.path.join(run_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    batch = cfg.get("total_batch_size", -1)
    if not batch or batch < 0:
        batch = 524288
    return {"name": os.path.basename(run_dir), "cfg": cfg, "recs": recs, "batch": batch,
            "model_type": cfg.get("model_type", "ar")}


def series(run, key):
    xs, ys = [], []
    for r in run["recs"]:
        if key in r and r[key] is not None:
            xs.append(r["step"] * run["batch"] / 1e9)
            ys.append(r[key])
    return xs, ys


def style_for(run):
    """Colour follows the entity (run family); line style separates the two MDM likelihood estimates."""
    n = run["name"]
    if run["model_type"] == "ar":
        return PALETTE["ar"]
    if n.endswith("_4x"):
        return PALETTE["mdm_4x"]
    if n.endswith("_2x"):
        return PALETTE["mdm_2x"]
    return PALETTE["mdm"]


def tidy(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=12, fontweight="bold")
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def label_end(ax, xs, ys, text, color, dy=0):
    if xs:
        ax.annotate(text, (xs[-1], ys[-1]), xytext=(6, dy), textcoords="offset points", va="center",
                    fontsize=9, color=INK)


def plot_val_bpb(runs, out):
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    ar_final = None
    for run in runs:
        c = style_for(run)
        if run["model_type"] == "ar":
            xs, ys = series(run, "val/bpb")
            ax.plot(xs, ys, color=c, linewidth=2, label=f"{run['name']} (exact NLL)")
            label_end(ax, xs, ys, run["name"], c)
            if ys:
                ar_final = ys[-1] if ar_final is None else min(ar_final, ys[-1])
        else:
            xs, ys = series(run, "val/bpb")
            ax.plot(xs, ys, color=c, linewidth=2, label=f"{run['name']} ELBO (upper bound)")
            label_end(ax, xs, ys, f"{run['name']} ELBO", c, dy=6)
            xs2, ys2 = series(run, "val/bpb_l2r")
            ax.plot(xs2, ys2, color=c, linewidth=2, linestyle="--", label=f"{run['name']} left-to-right NLL")
            label_end(ax, xs2, ys2, f"{run['name']} L2R", c, dy=-6)
    if ar_final is not None:
        ax.axhline(ar_final, color=PALETTE["ar"], linewidth=1, linestyle=":")
        ax.annotate(f"AR final {ar_final:.3f}", (0, ar_final), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.set_ylim(top=min(2.0, ax.get_ylim()[1]))
    tidy(ax, "training tokens (billions)", "validation bits per byte", "Validation loss vs tokens: AR vs masked diffusion (d12)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_train_loss(runs, out):
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    for run in runs:
        xs, ys = series(run, "train/loss")
        c = style_for(run)
        lab = run["name"] + (" (next-token CE)" if run["model_type"] == "ar" else " (masked-diffusion loss, = -ELBO/token)")
        ax.plot(xs, ys, color=c, linewidth=2, label=lab)
        label_end(ax, xs, ys, run["name"], c)
    ax.set_ylim(top=min(6.0, ax.get_ylim()[1]))
    tidy(ax, "training tokens (billions)", "train loss (nats / token, EMA)", "Training loss vs tokens (d12)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_throughput(runs, out):
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    for run in runs:
        xs, ys = series(run, "train/tok_per_sec")
        ax.plot(xs, [y / 1e3 for y in ys], color=style_for(run), linewidth=2, label=run["name"])
        label_end(ax, xs, [y / 1e3 for y in ys], run["name"], style_for(run))
    tidy(ax, "training tokens (billions)", "thousand tokens / s", "Training throughput (single H100 PCIe)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def load_core(phase):
    """{tag: {task: centered}} from results/<phase>/base_eval/<tag>_<step>.csv"""
    out = {}
    for path in sorted(glob.glob(f"results/{phase}/base_eval/*.csv")):
        tag = re.sub(r"_\d{6}\.csv$", "", os.path.basename(path))
        rows = {}
        with open(path) as f:
            for row in csv.reader(f):
                if len(row) >= 3 and row[0].strip() not in ("Task", "CORE"):
                    rows[row[0].strip()] = float(row[2])
                elif row and row[0].strip() == "CORE":
                    rows["CORE"] = float(row[2])
        out[tag] = rows
    return out


def plot_core(core, out):
    tags = [t for t in core if "CORE" in core[t]]
    if not tags:
        return False
    tasks = [t for t in core[tags[0]] if t != "CORE"]
    tasks.sort(key=lambda t: core[tags[0]][t])
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    h = 0.8 / len(tags)
    for i, tag in enumerate(tags):
        vals = [core[tag].get(t, 0.0) for t in tasks]
        ys = [j + i * h for j in range(len(tasks))]
        color = PALETTE["ar"] if "_ar" in tag else PALETTE["mdm"]
        ax.barh(ys, vals, height=h * 0.9, color=color, label=f"{tag}  (CORE {core[tag]['CORE']:.3f})")
    ax.set_yticks([j + (len(tags) - 1) * h / 2 for j in range(len(tasks))])
    ax.set_yticklabels(tasks, fontsize=8)
    ax.axvline(0, color=INK2, linewidth=0.8)
    tidy(ax, "centered accuracy (0 = random, 1 = perfect)", "", "CORE tasks: centered accuracy per task")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def summary_table(runs, core):
    lines = ["| run | type | tokens (B) | steps | val bpb (final) | val bpb L2R | CORE | train time | tok/s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        xs, ys = series(run, "val/bpb")
        _, l2r = series(run, "val/bpb_l2r")
        steps = max((r["step"] for r in run["recs"]), default=0)
        tt = max((r.get("total_training_time", 0) or 0 for r in run["recs"]), default=0)
        _, tps = series(run, "train/tok_per_sec")
        tps_med = sorted(tps)[len(tps) // 2] if tps else 0
        c = core.get(run["name"], {}).get("CORE")
        lines.append(f"| {run['name']} | {run['model_type']} | {steps * run['batch'] / 1e9:.2f} | {steps} | "
                     f"{ys[-1]:.4f} | " + (f"{l2r[-1]:.4f}" if l2r else "-") + " | " +
                     (f"{c:.4f}" if c is not None else "-") + f" | {tt / 60:.0f} min | {tps_med / 1e3:.0f}K |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="d12")
    args = ap.parse_args()
    run_dirs = sorted(d for d in glob.glob("results/logs/nanochat/*") if os.path.exists(os.path.join(d, "metrics.jsonl")))
    runs = [load_run(d) for d in run_dirs]
    runs = [r for r in runs if r["name"].startswith(args.phase)]
    if not runs:
        raise SystemExit("no runs found; run results/sync_logs.sh first")
    plots = f"results/{args.phase}/plots"
    os.makedirs(plots, exist_ok=True)
    plot_val_bpb(runs, f"{plots}/val_bpb_vs_tokens.png")
    plot_train_loss(runs, f"{plots}/train_loss_vs_tokens.png")
    plot_throughput(runs, f"{plots}/throughput.png")
    core = load_core(args.phase)
    has_core = plot_core(core, f"{plots}/core_per_task.png")
    table = summary_table(runs, core)
    with open(f"results/{args.phase}/summary.md", "w") as f:
        f.write(f"# {args.phase} summary\n\nGenerated by results/plot.py from synced logs.\n\n{table}\n\n")
        f.write("![val bpb](plots/val_bpb_vs_tokens.png)\n\n![train loss](plots/train_loss_vs_tokens.png)\n\n![throughput](plots/throughput.png)\n")
        if has_core:
            f.write("\n![CORE per task](plots/core_per_task.png)\n")
    print(table)
    print(f"\nwrote {plots}/*.png and results/{args.phase}/summary.md")


if __name__ == "__main__":
    main()
