"""
Post-training open question 2: is the diffusion model's slow multiple-choice learning a representation gap
(the frozen base model's features at the answer slot carry less about the answer) or an optimisation gap
(the features are there, but the gradient through a single <|mask|> position moves the network more slowly)?

Linear probe on frozen base-model features at the answer slot, MMLU auxiliary_train -> MMLU test.
The feature is the vector fed to lm_head (after the final norm), captured with a forward hook, at
  AR:  the <|assistant_start|> position of the completion prompt, where the AR model predicts the letter;
  MDM: the <|mask|> appended after <|assistant_start|>, where the MDM predicts the letter (as in chat_eval and in
       SFT at mask rate 1), and also the <|assistant_start|> position of the same masked input, to see where the
       information sits.
Reports the zero-shot letter accuracy of the frozen lm_head at the slot (the base model's own exam skill) and the
probe's test accuracy at several training-set sizes (sample efficiency of a linear readout).

    python -m scripts.probe_answer_slot --tag d12_ar --tag d12_mdm --n-train 20000 --n-test 2000

Writes $NANOCHAT_BASE_DIR/probes/answer_slot_<tag>.json and prints a markdown table.
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from nanochat.common import compute_init, autodetect_device_type, get_base_dir, print0
from nanochat.checkpoint_manager import load_model
from nanochat.diffusion import MASK_TOKEN
from tasks.mmlu import MMLU

ap = argparse.ArgumentParser()
ap.add_argument("--tag", action="append", required=True, help="checkpoint tag (repeatable)")
ap.add_argument("--source", default="base", help="base|sft")
ap.add_argument("--n-train", type=int, default=20000)
ap.add_argument("--n-test", type=int, default=2000)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--probe-sizes", default="500,2000,20000", help="training-set sizes for the probe (capped at --n-train)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
_, _, _, _, device = compute_init(autodetect_device_type())
probe_device = torch.device("cpu") if device.type == "mps" else device  # L-BFGS on a few MB: CPU is fine
LETTERS = ("A", "B", "C", "D")


@torch.no_grad()
def extract(model, tokenizer, task, n, is_mdm, mask_id, bos):
    """Frozen features at the answer slot. Returns {name: (n, C) float32}, labels (n,), zero-shot letter predictions (n,)."""
    captured = {}
    hook = model.lm_head.register_forward_hook(lambda m, inp, out: captured.__setitem__("x", inp[0].detach().float()))
    letter_ids = []
    for letter in LETTERS:
        ids = tokenizer.encode(letter)
        assert len(ids) == 1, f"letter {letter!r} is not a single token: {ids}"
        letter_ids.append(ids[0])
    letter_ids = torch.tensor(letter_ids, device=device)
    feats = {"slot": [], "prev": []} if is_mdm else {"slot": []}
    labels, zs_pred = [], []
    n = min(n, len(task))
    for i0 in range(0, n, args.batch_size):
        convs = [task[i] for i in range(i0, min(i0 + args.batch_size, n))]
        prompts = [tokenizer.render_for_completion(c) for c in convs]  # ends with <|assistant_start|>
        L = max(len(p) for p in prompts)
        if is_mdm:
            rows = [p + [mask_id] * (L + 1 - len(p)) for p in prompts]  # same padding as chat_eval
            slot = [len(p) for p in prompts]  # the <|mask|> right after <|assistant_start|>
        else:
            rows = [p + [bos] * (L - len(p)) for p in prompts]
            slot = [len(p) - 1 for p in prompts]  # the <|assistant_start|> token
        ids = torch.tensor(rows, dtype=torch.long, device=device)
        B = ids.size(0)
        r = torch.arange(B, device=device)
        c_slot = torch.tensor(slot, device=device)
        if is_mdm:
            # one forward pass, two selections: the mask and the token before it
            logits = model(ids, select=(torch.cat([r, r]), torch.cat([c_slot, c_slot - 1])))  # (2B, V)
            x = captured["x"]  # (2B, C)
            feats["slot"].append(x[:B].cpu())
            feats["prev"].append(x[B:].cpu())
            slot_logits = logits[:B]
        else:
            slot_logits = model(ids, select=(r, c_slot))  # (B, V)
            feats["slot"].append(captured["x"].cpu())
        zs_pred.append(slot_logits[:, letter_ids].argmax(dim=-1).cpu())
        labels.extend(LETTERS.index(c["messages"][-1]["content"]) for c in convs)
    hook.remove()
    return {k: torch.cat(v) for k, v in feats.items()}, torch.tensor(labels), torch.cat(zs_pred)


def fit_logreg(X, y, wd, steps=200):
    W = torch.zeros(X.size(1), len(LETTERS), device=probe_device, requires_grad=True)
    b = torch.zeros(len(LETTERS), device=probe_device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=steps, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, y) + wd * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def probe(Xtr, ytr, Xte, yte, seed):
    """Multinomial logistic regression on standardised features; weight decay picked on a 10% split of the training set."""
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr = ((Xtr - mu) / sd).to(probe_device)
    Xte = ((Xte - mu) / sd).to(probe_device)
    ytr, yte = ytr.to(probe_device), yte.to(probe_device)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(Xtr.size(0), generator=g).to(probe_device)
    n_val = max(1, Xtr.size(0) // 10)
    va, fit = perm[:n_val], perm[n_val:]
    acc = lambda W, b, X, y: ((X @ W + b).argmax(1) == y).float().mean().item()
    best = None
    for wd in (1e-4, 1e-3, 1e-2, 1e-1):
        W, b = fit_logreg(Xtr[fit], ytr[fit], wd)
        a = acc(W, b, Xtr[va], ytr[va])
        if best is None or a > best[0]:
            best = (a, wd)
    W, b = fit_logreg(Xtr, ytr, best[1])
    return {"val_acc": best[0], "wd": best[1], "train_acc": acc(W, b, Xtr, ytr), "test_acc": acc(W, b, Xte, yte)}


out_dir = os.path.join(get_base_dir(), "probes")
os.makedirs(out_dir, exist_ok=True)
train_task = MMLU(subset="all", split="auxiliary_train", stop=args.n_train)
test_task = MMLU(subset="all", split="test", stop=args.n_test)
sizes = [min(int(s), args.n_train) for s in args.probe_sizes.split(",")]
rows = []
for tag in args.tag:
    t0 = time.time()
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=tag)
    model.eval()
    is_mdm = getattr(model.config, "model_type", "ar") == "mdm"
    mask_id = tokenizer.encode_special(MASK_TOKEN) if is_mdm else None
    bos = tokenizer.get_bos_token_id()
    ftr, ytr, zs_tr = extract(model, tokenizer, train_task, args.n_train, is_mdm, mask_id, bos)
    fte, yte, zs_te = extract(model, tokenizer, test_task, args.n_test, is_mdm, mask_id, bos)
    del model
    probes: dict = {}
    for name in ftr:
        probes[name] = {str(n): probe(ftr[name][:n], ytr[:n], fte[name], yte, args.seed) for n in sizes}
    result = {"tag": tag, "model_type": "mdm" if is_mdm else "ar", "n_train": int(ytr.numel()), "n_test": int(yte.numel()),
              "feature_dim": int(ftr["slot"].size(1)), "zero_shot_test_acc": (zs_te == yte).float().mean().item(),
              "zero_shot_train_acc": (zs_tr == ytr).float().mean().item(),
              "label_majority_test": max((yte == k).float().mean().item() for k in range(len(LETTERS))),
              "probes": probes, "seconds": time.time() - t0}
    with open(os.path.join(out_dir, f"answer_slot_{tag}.json"), "w") as f:
        json.dump(result, f, indent=1)
    for name in ftr:
        rows.append((tag, name, result["zero_shot_test_acc"], [probes[name][str(n)]["test_acc"] for n in sizes]))
    print0(f"{tag}: done in {result['seconds']:.0f}s")

print0("\n| checkpoint | feature | zero-shot letter acc | " + " | ".join(f"probe acc @{n}" for n in sizes) + " |")
print0("|---|---|---|" + "---|" * len(sizes))
for tag, name, zs, accs in rows:
    print0(f"| {tag} | {name} | {100*zs:.1f}% | " + " | ".join(f"{100*a:.1f}%" for a in accs) + " |")
print0("\nfeature: slot = the position the model predicts the letter from (AR: <|assistant_start|>; MDM: the <|mask|> after it); "
       "prev = the <|assistant_start|> position of the masked MDM input. Chance is 25%.")
