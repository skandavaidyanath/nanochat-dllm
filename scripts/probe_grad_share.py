"""
Post-training open question 2, gradient side: in a real SFT micro-batch, how much of the update comes from the
multiple-choice rows, for the AR model versus the diffusion model?

Builds micro-batches exactly as scripts/chat_sft.py does (same TaskMixture and epochs, same BOS-aligned best-fit
packing, same per-token loss and batch normaliser) but keeps a per-token source label, then splits the gradient by
source with the batch normaliser kept, so the per-source gradients sum to the training gradient:
    g = g_smoltalk + g_mmlu_letter + g_mmlu_end + g_gsm8k
(the MMLU answer letter and the <|assistant_end|> after it are separate sources, since only the letter carries the
exam skill). Reports, averaged over --batches micro-batches: supervised tokens, summed loss, ||g_src||, the share
||g_src|| / ||g|| and cos(g_src, g - g_src), per source and checkpoint. For the MDM the loss at a token is CE/p
when the token is masked and 0 otherwise, exactly as in training, so the letter's gradient is present only on the
micro-batches where it was masked; averaging over 16 micro-batches (one optimizer step) smooths that.

    python -m scripts.probe_grad_share --tag d12_ar --tag d12_mdm --batches 16 --mmlu-epochs 3

Writes $NANOCHAT_BASE_DIR/probes/grad_share_<tag>.json and prints a markdown table.
"""
import argparse
import json
import os
import time

import torch

from nanochat.common import compute_init, autodetect_device_type, get_base_dir, print0
from nanochat.checkpoint_manager import load_model
from nanochat.diffusion import MASK_TOKEN, mdm_loss
from tasks.common import TaskMixture
from tasks.smoltalk import SmolTalk
from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K

ap = argparse.ArgumentParser()
ap.add_argument("--tag", action="append", required=True, help="checkpoint tag (repeatable)")
ap.add_argument("--source", default="base", help="base|sft")
ap.add_argument("--batches", type=int, default=16, help="micro-batches to average; 16 = one optimizer step at the default grad accumulation")
ap.add_argument("--device-batch-size", type=int, default=16)
ap.add_argument("--max-seq-len", type=int, default=None, help="default: the checkpoint's")
ap.add_argument("--smoltalk-epochs", type=int, default=1)
ap.add_argument("--mmlu-epochs", type=int, default=3)
ap.add_argument("--gsm8k-epochs", type=int, default=4)
ap.add_argument("--mdm-min-mask-rate", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
_, _, _, _, device = compute_init(autodetect_device_type())

SRC = ["smoltalk", "mmlu_letter", "mmlu_end", "gsm8k"]  # per-token source labels 0..3; -1 = not supervised


def build_mixture():
    """Same tasks, same order and the same seed-42 shuffle as chat_sft, so the conversation stream is the training stream."""
    tasks, kinds = [], []
    for _ in range(args.smoltalk_epochs):
        tasks.append(SmolTalk(split="train")); kinds.append("smoltalk")
    for _ in range(args.mmlu_epochs):
        tasks.append(MMLU(subset="all", split="auxiliary_train")); kinds.append("mmlu")
    for _ in range(args.gsm8k_epochs):
        tasks.append(GSM8K(subset="main", split="train")); kinds.append("gsm8k")
    return TaskMixture(tasks), kinds


def micro_batches(tokenizer, mixture, kinds, is_mdm, max_seq_len, buffer_size=100):
    """chat_sft's bestfit packer (single rank) with a per-token source label."""
    row_capacity = max_seq_len + 1
    bos = tokenizer.get_bos_token_id()
    buf, cursor = [], 0

    def refill():
        nonlocal cursor
        while len(buf) < buffer_size:
            task_idx, _ = mixture.index_map[cursor]
            ids, mask = tokenizer.render_conversation(mixture[cursor])
            kind = kinds[task_idx]
            src, seen_letter = [], False
            for m in mask:
                if m == 0:
                    src.append(-1)
                elif kind == "smoltalk":
                    src.append(0)
                elif kind == "gsm8k":
                    src.append(3)
                else:  # MMLU: the first supervised token is the letter, the second is <|assistant_end|>
                    src.append(2 if seen_letter else 1)
                    seen_letter = True
            buf.append((ids, mask, src))
            cursor = (cursor + 1) % len(mixture)

    while True:
        rows, mrows, srows = [], [], []
        for _ in range(args.device_batch_size):
            row, mrow, srow = [], [], []
            while len(row) < row_capacity:
                refill()
                remaining = row_capacity - len(row)
                best, best_len = -1, 0
                for i, (ids, _, _) in enumerate(buf):
                    if best_len < len(ids) <= remaining:
                        best, best_len = i, len(ids)
                if best >= 0:
                    ids, mask, src = buf.pop(best)
                    row += ids; mrow += mask; srow += src
                else:
                    row += [bos] * remaining; mrow += [0] * remaining; srow += [-1] * remaining
                    break
            rows.append(row[:row_capacity]); mrows.append(mrow[:row_capacity]); srows.append(srow[:row_capacity])
        t = torch.tensor(rows, dtype=torch.long)
        m = torch.tensor(mrows, dtype=torch.int8)
        s = torch.tensor(srows, dtype=torch.long)
        if is_mdm:
            yield t[:, :-1].to(device), (m[:, :-1] == 1).to(device), s[:, :-1].to(device)
        else:
            inputs = t[:, :-1].to(device=device, dtype=torch.int32)
            targets = t[:, 1:].clone()
            targets[m[:, 1:] == 0] = -1  # padding has mask 0 too, so this also drops the pad targets
            yield inputs, targets.to(device), s[:, 1:].to(device)


def flat_grad(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).float() for g, p in zip(grads, params)])


out_dir = os.path.join(get_base_dir(), "probes")
os.makedirs(out_dir, exist_ok=True)
mixture, kinds = build_mixture()
print0(f"mixture: {len(mixture):,} conversations (SmolTalk x{args.smoltalk_epochs}, MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs})")
summary = {}
for tag in args.tag:
    t0 = time.time()
    torch.manual_seed(args.seed)
    model, tokenizer, meta = load_model(args.source, device, phase="train", model_tag=tag)
    model.eval()  # no dropout in nanochat; this only matters for determinism
    is_mdm = getattr(model.config, "model_type", "ar") == "mdm"
    mask_id = tokenizer.encode_special(MASK_TOKEN) if is_mdm else None
    max_seq_len = args.max_seq_len or meta.get("max_seq_len", model.config.sequence_len)
    params = [p for p in model.parameters() if p.requires_grad]
    acc = {k: {"tokens": 0.0, "loss": 0.0, "norm": 0.0, "share": 0.0, "cos": 0.0, "candidates": 0.0} for k in SRC}
    total_norm = 0.0
    loader = micro_batches(tokenizer, mixture, kinds, is_mdm, max_seq_len)
    for bi in range(args.batches):
        x, y, src = next(loader)
        if is_mdm:
            loss_tok, masked = mdm_loss(model, x, y, mask_id, reduction="none", eps=args.mdm_min_mask_rate)
            N = y.sum().clamp(min=1)  # maskable tokens: the training normaliser
            sup = masked
            cand = y
        else:
            loss_tok = model(x, y, loss_reduction="none").view(y.shape)
            sup = y != -1
            N = sup.sum().clamp(min=1)  # supervised tokens: the training normaliser (mean CE)
            cand = sup
        parts = {}
        for k, name in enumerate(SRC):
            sel = sup & (src == k)
            parts[name] = (loss_tok * sel).sum() / N
            acc[name]["tokens"] += sel.sum().item()
            acc[name]["candidates"] += (cand & (src == k)).sum().item()
            acc[name]["loss"] += parts[name].item()
        g = {name: flat_grad(parts[name], params) for name in SRC}
        model.zero_grad(set_to_none=True)
        g_total = g[SRC[0]].clone()
        for name in SRC[1:]:
            g_total += g[name]
        tn = g_total.norm().item()
        total_norm += tn
        for name in SRC:
            gn = g[name].norm().item()
            rest = g_total - g[name]
            acc[name]["norm"] += gn
            acc[name]["share"] += gn / max(tn, 1e-12)
            acc[name]["cos"] += (torch.dot(g[name], rest) / max(gn * rest.norm().item(), 1e-12)).item()
        del g, g_total, loss_tok
        print0(f"  {tag} micro-batch {bi+1}/{args.batches}: ||g||={tn:.4f}  " + "  ".join(f"{n}={parts[n].item():.4f}" for n in SRC))
    nb = args.batches
    res = {"tag": tag, "model_type": "mdm" if is_mdm else "ar", "batches": nb, "device_batch_size": args.device_batch_size,
           "max_seq_len": max_seq_len, "grad_norm": total_norm / nb,
           "sources": {name: {k: v / nb for k, v in acc[name].items()} for name in SRC}, "seconds": time.time() - t0}
    summary[tag] = res
    with open(os.path.join(out_dir, f"grad_share_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    del model, params
    if device.type == "cuda":
        torch.cuda.empty_cache()

print0("\nPer micro-batch averages (device batch %d x %s tokens). tokens = supervised tokens that carried loss (MDM: masked);"
       " candidates = maskable/supervised tokens of that source; loss = contribution to the batch loss; share = ||g_src|| / ||g||;"
       " cos = cosine between g_src and the rest of the gradient." % (args.device_batch_size, "seq"))
print0("\n| checkpoint | source | tokens | candidates | loss | grad norm | share of ||g|| | cos(src, rest) |")
print0("|---|---|---|---|---|---|---|---|")
for tag, res in summary.items():
    for name in SRC:
        s = res["sources"][name]
        print0(f"| {tag} | {name} | {s['tokens']:.1f} | {s['candidates']:.1f} | {s['loss']:.4f} | {s['norm']:.4f} | {100*s['share']:.1f}% | {s['cos']:+.3f} |")
    print0(f"| {tag} | total | | | | {res['grad_norm']:.4f} | 100% | |")
