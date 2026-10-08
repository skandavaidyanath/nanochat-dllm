"""
Build the result plots and summary table from synced run logs.

    bash results/sync_logs.sh prime-d12 d12      # pull logs from the GPU box
    python results/plot.py --phase d12           # writes results/d12/plots/*.png and results/d12/summary.md

Reads results/logs/<project>/<run>/metrics.jsonl (+ config.json) written by nanochat's MetricsLogger,
results/<phase>/base_eval/<tag>_<step>.csv written by scripts/base_eval.py, and
results/<phase>/decode_bench.log written by scripts/infer_bench_mdm.py.

Figures:
  val_bpb_vs_tokens.png   main runs: AR vs MDM at 1x / 2x / 4x tokens
  train_loss_vs_tokens.png, throughput.png   same runs
  lr_sweep.png            Muon LR sweep, one panel per model type
  datarep.png             data-constrained ablation (10 epochs over 136M tokens) vs the full-data runs
  decode_bench.png        decoding throughput vs tokens-per-forward-pass, AR reference
  core_per_task.png       CORE centered accuracy per task
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
UNIQUE_TOKENS_REP10_B = 0.136  # 4 training shards of ClimbMix, ~34M usable tokens each


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


def val_key(run):
    """The likelihood we compare across model types: exact NLL for AR, left-to-right NLL for MDM."""
    return "val/bpb" if run["model_type"] == "ar" else "val/bpb_l2r"


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


def label_end(ax, xs, ys, text, _color=None, dy=0):
    if xs:
        ax.annotate(text, (xs[-1], ys[-1]), xytext=(6, dy), textcoords="offset points", va="center",
                    fontsize=9, color=INK)


def by_name(runs, name):
    for r in runs:
        if r["name"] == name:
            return r
    return None


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
    ax.set_ylim(0.78, 1.6)
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


def plot_lr_sweep(runs, phase, out):
    """One panel per model type; line style = Muon matrix LR (solid 0.02 default, dashed 0.01, dotted 0.04)."""
    styles = {"": ("-", "lr 0.02 (default)"), "_lr0.01": ("--", "lr 0.01"), "_lr0.04": (":", "lr 0.04")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150, sharey=False)
    found = False
    for ax, mt in zip(axes, ("ar", "mdm")):
        for suffix, (ls, lab) in styles.items():
            run = by_name(runs, f"{phase}_{mt}{suffix}")
            if run is None:
                continue
            found = True
            xs, ys = series(run, val_key(run))
            ax.plot(xs, ys, color=PALETTE[mt], linewidth=2, linestyle=ls, label=f"{lab}: final {ys[-1]:.3f}")
        ax.set_xlim(0.25, None)
        ax.set_ylim(0.82 if mt == "ar" else 1.02, 1.12 if mt == "ar" else 1.32)
        tidy(ax, "training tokens (billions)", "validation bits per byte" if mt == "ar" else "left-to-right NLL, bits per byte",
             f"{'AR' if mt == 'ar' else 'Masked diffusion'}: Muon LR sweep")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if found:
        fig.savefig(out)
    plt.close(fig)
    return found


def plot_datarep(runs, phase, out):
    """Full data (solid) vs the same token budget drawn from a 136M-token subset for ~10 epochs (dashed)."""
    pairs = [("ar", "AR"), ("mdm", "masked diffusion (L2R NLL)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    found = False
    for mt, lab in pairs:
        full, rep = by_name(runs, f"{phase}_{mt}"), by_name(runs, f"{phase}_{mt}_rep10")
        if full is None or rep is None:
            continue
        found = True
        for run, ls, what in ((full, "-", "full data"), (rep, "--", "10 epochs on 136M tokens")):
            xs, ys = series(run, val_key(run))
            axes[0].plot(xs, ys, color=PALETTE[mt], linewidth=2, linestyle=ls, label=f"{lab}, {what}: final {ys[-1]:.3f}")
            xs, ys = series(run, "train/loss")
            axes[1].plot(xs, ys, color=PALETTE[mt], linewidth=2, linestyle=ls, label=f"{lab}, {what}")
        if mt == "ar":
            xs, ys = series(rep, "val/bpb")
            i = min(range(len(ys)), key=lambda k: ys[k])
            axes[0].plot([xs[i]], [ys[i]], marker="o", markersize=8, color=PALETTE["ar"], markeredgecolor="white")
            axes[0].annotate(f"AR best {ys[i]:.3f} at {xs[i] / UNIQUE_TOKENS_REP10_B:.0f} epochs, then overfits",
                             (xs[i], ys[i]), xytext=(8, -14), textcoords="offset points", fontsize=8, color=INK)
    axes[0].set_xlim(0.2, None)
    axes[0].set_ylim(0.8, 1.35)
    axes[1].set_xlim(0.2, None)
    axes[1].set_ylim(2.0, 4.6)
    tidy(axes[0], "training tokens (billions)", "validation bits per byte", "Data-constrained: validation loss")
    tidy(axes[1], "training tokens (billions)", "train loss (nats / token, EMA)", "Data-constrained: train loss (memorisation)")
    for ax in axes:
        sec = ax.secondary_xaxis("top", functions=(lambda t: t / UNIQUE_TOKENS_REP10_B, lambda e: e * UNIQUE_TOKENS_REP10_B))
        sec.set_xlabel("epochs over the 136M-token subset", color=INK2, fontsize=8)
        sec.tick_params(colors=INK2, labelsize=8)
        ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    if found:
        fig.savefig(out)
    plt.close(fig)
    return found


def load_decode_bench(path):
    """rows of {model, steps_per_block, block, tokens_per_pass, batch, tok_s, s_gen} from infer_bench_mdm output."""
    rows = []
    if not os.path.exists(path):
        return rows
    pat = re.compile(r"^(AR \(KV cache\)|MDM block (\d+))\s+(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)")
    for line in open(path):
        m = pat.match(line.strip())
        if not m:
            continue
        model, block, steps, batch, tok_s, s_gen = m.groups()
        if model.startswith("AR"):
            rows.append({"model": "ar", "block": None, "steps": None, "tokens_per_pass": 1,
                         "batch": int(batch), "tok_s": float(tok_s), "s_gen": float(s_gen)})
        else:
            steps = int(steps.split("/")[0])
            rows.append({"model": "mdm", "block": int(block), "steps": steps, "tokens_per_pass": int(block) // steps,
                         "batch": int(batch), "tok_s": float(tok_s), "s_gen": float(s_gen)})
    return rows


def plot_decode_bench(rows, out):
    if not rows:
        return False
    batches = sorted({r["batch"] for r in rows})
    fig, axes = plt.subplots(1, len(batches), figsize=(3.6 * len(batches), 4.2), dpi=150, sharey=True)
    axes = list(axes) if len(batches) > 1 else [axes]
    for ax, b in zip(axes, batches):
        mdm = sorted([r for r in rows if r["model"] == "mdm" and r["batch"] == b], key=lambda r: r["tokens_per_pass"])
        ar = [r for r in rows if r["model"] == "ar" and r["batch"] == b]
        xs = [r["tokens_per_pass"] for r in mdm]
        ys = [r["tok_s"] for r in mdm]
        ax.plot(xs, ys, color=PALETTE["mdm"], linewidth=2, marker="o", markersize=6, markeredgecolor="white",
                label="masked diffusion (semi-AR, block 32)")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:,.0f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
        if ar:
            ax.axhline(ar[0]["tok_s"], color=PALETTE["ar"], linewidth=2, linestyle=":", label=f"AR with KV cache: {ar[0]['tok_s']:,.0f} tok/s")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        tidy(ax, "tokens decoded per forward pass", "tokens / s (log)" if ax is axes[0] else "", f"batch {b}")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Decoding throughput, d12 on one H100 PCIe (256 generated tokens)", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def decode_table(rows):
    if not rows:
        return ""
    batches = sorted({r["batch"] for r in rows})
    lines = ["| decoder | tokens / forward pass | " + " | ".join(f"batch {b} tok/s" for b in batches) + " |",
             "|---|---|" + "---|" * len(batches)]
    ar = {r["batch"]: r["tok_s"] for r in rows if r["model"] == "ar"}
    lines.append("| AR, KV cache | 1 | " + " | ".join(f"{ar.get(b, float('nan')):,.0f}" for b in batches) + " |")
    for tpp in sorted({r["tokens_per_pass"] for r in rows if r["model"] == "mdm"}):
        vals = {r["batch"]: r["tok_s"] for r in rows if r["model"] == "mdm" and r["tokens_per_pass"] == tpp}
        steps = next(r["steps"] for r in rows if r["model"] == "mdm" and r["tokens_per_pass"] == tpp)
        cells = []
        for b in batches:
            v = vals.get(b)
            cells.append(f"{v:,.0f} ({v / ar[b]:.1f}x)" if v is not None and b in ar else "-")
        lines.append(f"| MDM, block 32, {steps} steps/block | {tpp} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


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


def plot_core(core, tags, out):
    tags = [t for t in tags if t in core and "CORE" in core[t]]
    if not tags:
        return False
    tasks = [t for t in core[tags[0]] if t != "CORE"]
    tasks.sort(key=lambda t: core[tags[0]][t])
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    h = 0.8 / len(tags)
    for i, tag in enumerate(tags):
        vals = [core[tag].get(t, 0.0) for t in tasks]
        ys = [j + i * h for j in range(len(tasks))]
        if "_ar" in tag:
            color = PALETTE["ar"]
        elif tag.endswith("_4x"):
            color = PALETTE["mdm_4x"]
        elif tag.endswith("_2x"):
            color = PALETTE["mdm_2x"]
        else:
            color = PALETTE["mdm"]
        hatch = "///" if tag.endswith("_rep10") else None
        ax.barh(ys, vals, height=h * 0.9, color=color, hatch=hatch, edgecolor="white" if hatch else None,
                linewidth=0.5, label=f"{tag}  (CORE {core[tag]['CORE']:.3f})")
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
        _, ys = series(run, "val/bpb")
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
    phase = args.phase
    run_dirs = sorted(d for d in glob.glob("results/logs/nanochat/*") if os.path.exists(os.path.join(d, "metrics.jsonl")))
    runs = [load_run(d) for d in run_dirs]
    runs = [r for r in runs if r["name"].startswith(phase)]
    if not runs:
        raise SystemExit("no runs found; run results/sync_logs.sh first")
    main_names = [f"{phase}_ar", f"{phase}_mdm", f"{phase}_mdm_2x", f"{phase}_mdm_4x"]
    main_runs = [r for n in main_names for r in runs if r["name"] == n]
    plots = f"results/{phase}/plots"
    os.makedirs(plots, exist_ok=True)
    plot_val_bpb(main_runs, f"{plots}/val_bpb_vs_tokens.png")
    plot_train_loss(main_runs, f"{plots}/train_loss_vs_tokens.png")
    plot_throughput(main_runs, f"{plots}/throughput.png")
    has_lr = plot_lr_sweep(runs, phase, f"{plots}/lr_sweep.png")
    has_rep = plot_datarep(runs, phase, f"{plots}/datarep.png")
    bench = load_decode_bench(f"results/{phase}/decode_bench.log")
    has_bench = plot_decode_bench(bench, f"{plots}/decode_bench.png")
    core = load_core(phase)
    has_core = plot_core(core, main_names + [f"{phase}_ar_rep10", f"{phase}_mdm_rep10"], f"{plots}/core_per_task.png")
    table = summary_table(runs, core)
    with open(f"results/{phase}/summary.md", "w") as f:
        f.write(f"# {phase} summary\n\nGenerated by results/plot.py from synced logs.\n\n{table}\n\n")
        f.write("![val bpb](plots/val_bpb_vs_tokens.png)\n\n![train loss](plots/train_loss_vs_tokens.png)\n\n![throughput](plots/throughput.png)\n")
        if has_lr:
            f.write("\n![LR sweep](plots/lr_sweep.png)\n")
        if has_rep:
            f.write("\n![data-constrained](plots/datarep.png)\n")
        if has_bench:
            f.write("\n## Decoding throughput (scripts/infer_bench_mdm.py, 256 tokens after a 7-token prompt)\n\n")
            f.write(decode_table(bench) + "\n\n![decode bench](plots/decode_bench.png)\n")
        if has_core:
            f.write("\n![CORE per task](plots/core_per_task.png)\n")
    print(table)
    if has_bench:
        print()
        print(decode_table(bench))
    print(f"\nwrote {plots}/*.png and results/{phase}/summary.md")


if __name__ == "__main__":
    main()
