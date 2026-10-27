"""
Masked diffusion language model (MDM) support for nanochat.

The "dLLM" variant keeps the GPT architecture, data, tokenizer and optimizer identical to the
autoregressive (AR) model and changes only three things:

1) attention is bidirectional (see GPTConfig.model_type == "mdm" in gpt.py)
2) a <|mask|> special token exists in the vocabulary
3) the training objective is the LLaDA / MDLM masked diffusion loss implemented here

Training objective (per sequence of maskable tokens x_1..x_N):
    sample a mask rate p ~ U(eps, 1), mask each maskable token independently with prob p,
    loss = (1/p) * sum_{masked i} CE(model(x_masked)_i, x_i),  normalized by N
In expectation this is the negative ELBO per token, i.e. an upper bound on the per-token NLL,
and it is in the same units (nats/token) as the AR cross-entropy loss. We use stratified
("low discrepancy") sampling of p across the batch to reduce variance.

Evaluation (both reported in bits per byte, bpb, so they are tokenizer-invariant):
    - "elbo": Monte Carlo estimate of the negative ELBO. Upper bound on the true NLL.
    - "l2r":  left-to-right NLL, i.e. -log p(x_i | x_<i) with all x_>=i masked. This is the exact
              NLL of a well-defined generative model (the L2R-decoded MDM), so it is directly
              comparable to the AR val loss. Estimated by sampling one position per row.

Inference: DiffusionEngine implements semi-autoregressive block decoding with low-confidence
remasking (LLaDA style), the same generate_batch() API as nanochat.engine.Engine, and the same
<|python_start|>...<|python_end|> calculator tool protocol so SFT/RL/eval code is shared.
"""
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist

from nanochat.engine import use_calculator, sample_next_token

MASK_TOKEN = "<|mask|>"
EPS = 1e-3  # minimum mask rate, as in LLaDA (avoids 1/p blowup and the all-clean no-op)


# -----------------------------------------------------------------------------
# Forward (noising) process and training loss

def _rand(shape, device, generator=None):
    """torch.rand that tolerates a generator living on a different device (e.g. a CPU generator for eval)."""
    if generator is None:
        return torch.rand(shape, device=device)
    if generator.device.type != device.type:
        return torch.rand(shape, generator=generator, device=generator.device).to(device)
    return torch.rand(shape, device=device, generator=generator)


def sample_mask_rates(B, device, generator=None, eps=EPS):
    """Stratified sample of B mask rates in [eps, 1]: one draw from each of B equal-width bins, shuffled."""
    u = _rand((B,), device, generator)
    t = (torch.arange(B, device=device, dtype=torch.float32) + u) / B
    if B > 1:
        perm = torch.argsort(_rand((B,), device, generator))
        t = t[perm]
    p = (1 - eps) * t + eps
    return p


def apply_mask(ids, maskable, p_mask, mask_id, generator=None):
    """
    ids: (B, T) long clean token ids. maskable: (B, T) bool. p_mask: (B,) float mask rate per row.
    Returns (noisy_ids, masked) where masked is a (B, T) bool of positions that were replaced by mask_id.
    """
    u = _rand(ids.shape, ids.device, generator)
    masked = (u < p_mask[:, None]) & maskable
    noisy = torch.where(masked, torch.full_like(ids, mask_id), ids)
    return noisy, masked


def mdm_loss(model, ids, maskable, mask_id, p_mask=None, generator=None, reduction="mean", eps=EPS, forced=None):
    """
    Masked diffusion loss (negative ELBO estimate), see module docstring.
    eps is the minimum mask rate: p ~ U(eps, 1). The LLaDA default 1e-3 gives importance weights 1/p up to 1000x;
    raising it (e.g. 0.1) bounds the weights at the cost of a biased ELBO. See docs/EXPERIMENT.md on SFT saturation.
    forced: optional (B, T) bool of maskable positions that are always masked and weighted 1 instead of 1/p
            (SFT option for short answers such as a multiple-choice letter: deterministic supervision, like the AR
            next-token loss at that slot, while the rest of the batch keeps the random-rate objective).
    reduction="mean": scalar = sum_{masked} CE/p / (#maskable tokens in the batch). Comparable to the AR
                      mean cross-entropy (nats per supervised token), as an upper bound.
    reduction="none": (loss_tok, masked) where loss_tok is (B, T) with CE/p at masked positions and 0 elsewhere.
    """
    B, T = ids.shape
    if p_mask is None:
        p_mask = sample_mask_rates(B, ids.device, generator, eps=eps)
    noisy, masked = apply_mask(ids, maskable, p_mask, mask_id, generator)
    if forced is not None:
        forced = forced & maskable
        masked = masked | forced
        noisy = torch.where(masked, torch.full_like(ids, mask_id), ids)
    targets = torch.where(masked, ids, torch.full_like(ids, -1))  # -1 = ignore_index
    loss_tok = model(noisy, targets, loss_reduction="none").view(B, T)  # 0 at ignored positions
    weight = (1.0 / p_mask)[:, None].expand(B, T)
    if forced is not None:
        weight = torch.where(forced, torch.ones_like(weight), weight)
    loss_tok = loss_tok * weight
    if reduction == "none":
        return loss_tok, masked
    n = maskable.sum().clamp(min=1)
    return loss_tok.sum() / n


# -----------------------------------------------------------------------------
# Likelihood evaluation in bits per byte

def _reduce_sum(*tensors):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        for t in tensors:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)


@torch.no_grad()
def evaluate_bpb_mdm(model, batches, steps, token_bytes, mask_id, mode="elbo", seed=0):
    """
    MDM counterpart of nanochat.loss_eval.evaluate_bpb.
    batches: iterator yielding (ids, maskable) with ids (B, T) long and maskable (B, T) bool.
    mode="elbo": stratified-p Monte Carlo estimate of the negative ELBO per byte (upper bound on NLL).
    mode="l2r":  left-to-right NLL per byte of the L2R-decoded model (one random position per row).
    Only tokens with token_bytes > 0 (i.e. non-special tokens) contribute nats and bytes, like the AR version.
    Deterministic given seed, so the same mask draws are used every time the eval runs.
    """
    assert mode in ("elbo", "l2r")
    device = model.get_device()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    batch_iter = iter(batches)
    for _ in range(steps):
        ids, maskable = next(batch_iter)
        if maskable.dtype != torch.bool:
            maskable = maskable > 0  # the SFT loader may send 0/1/2 (2 = forced-mask position); the metric ignores the distinction
        B, T = ids.shape
        counted = maskable & (token_bytes[ids] > 0)  # tokens that contribute to the metric
        total_bytes += token_bytes[ids][counted].sum()
        if mode == "elbo":
            loss_tok, masked = mdm_loss(model, ids, maskable, mask_id, generator=generator, reduction="none")
            total_nats += (loss_tok * counted).sum()
        else:
            # pick one maskable position per row uniformly, mask everything maskable at or after it
            u = _rand((B,), device, generator)
            n_maskable = maskable.sum(dim=1)  # (B,)
            k = (u * n_maskable).long().clamp(max=(n_maskable - 1).clamp(min=0))  # k-th maskable position
            cum = maskable.long().cumsum(dim=1) - 1  # index among maskable positions
            pos_hit = maskable & (cum == k[:, None])  # (B, T) one-hot of the chosen position
            pos = pos_hit.long().argmax(dim=1)  # (B,)
            suffix = maskable & (torch.arange(T, device=device)[None, :] >= pos[:, None])
            noisy = torch.where(suffix, torch.full_like(ids, mask_id), ids)
            targets = torch.where(pos_hit, ids, torch.full_like(ids, -1))
            loss_tok = model(noisy, targets, loss_reduction="none").view(B, T)
            # unbiased estimate of sum_i CE_i over counted positions: N_maskable * CE_pos * [pos counted]
            row_nats = (loss_tok * pos_hit * counted).sum(dim=1) * n_maskable.float()
            row_nats = torch.where(n_maskable > 0, row_nats, torch.zeros_like(row_nats))
            total_nats += row_nats.sum()
    _reduce_sum(total_nats, total_bytes)
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)


def mdm_batches_from_ar_loader(loader):
    """Adapt a nanochat pretraining loader yielding (inputs, targets) into (ids, maskable) batches."""
    for x, _y in loader:
        yield x, torch.ones_like(x, dtype=torch.bool)


# -----------------------------------------------------------------------------
# Conditional scoring of continuations (used by CORE and categorical chat evals)

@torch.no_grad()
def score_continuations(model, input_ids, start_idxs, end_idxs, mask_id, method="l2r", max_tokens_per_chunk=16384):
    """
    For each row b, score tokens input_ids[b, s:e] given the clean prefix input_ids[b, :s].
    Positions >= e (padding) are always masked. Returns (losses, predictions) in the layout of
    nanochat.core_eval.forward_model: losses[b, j-1] = -log p(x_j | context), predictions[b, j-1] = argmax,
    for j in [s, e); nan / -1 elsewhere.
      method="l2r": x_j is scored with x_<j clean and x_>=j masked (exact NLL of the L2R-decoded model).
                    Costs one forward per scored token (batched).
      method="mf":  all of x_[s:e) masked at once, one forward per row (mean-field approximation).
    Only the logits at scored positions are computed (GPT.forward(select=...)), and forwards are chunked to
    at most max_tokens_per_chunk tokens so memory stays bounded for long few-shot prompts.
    """
    assert method in ("l2r", "mf")
    device = input_ids.device
    B, T = input_ids.shape
    losses = torch.full((B, T), float("nan"), dtype=torch.float32, device=device)
    predictions = torch.full((B, T), -1, dtype=torch.long, device=device)
    pos_idx = torch.arange(T, device=device)
    # each variant is one noisy input row scored at one or more positions: (b, mask_from, [positions])
    variants = []
    for b, (s, e) in enumerate(zip(start_idxs, end_idxs)):
        if method == "l2r":
            variants.extend((b, j, [j]) for j in range(s, e))
        else:
            variants.append((b, s, list(range(s, e))))
    chunk_rows = max(1, max_tokens_per_chunk // T)
    for i0 in range(0, len(variants), chunk_rows):
        chunk = variants[i0:i0 + chunk_rows]
        rows = torch.stack([input_ids[b] for b, _, _ in chunk])  # (n, T)
        mask_from = torch.tensor([m for _, m, _ in chunk], device=device)
        ends = torch.tensor([end_idxs[b] for b, _, _ in chunk], device=device)
        masked = (pos_idx[None, :] >= mask_from[:, None]) | (pos_idx[None, :] >= ends[:, None])
        noisy = torch.where(masked, torch.full_like(rows, mask_id), rows)
        sel_rows = torch.tensor([n for n, (_, _, js) in enumerate(chunk) for _ in js], device=device)
        sel_cols = torch.tensor([j for _, _, js in chunk for j in js], device=device)
        logits_sel = model(noisy, select=(sel_rows, sel_cols))  # (num_scored, V)
        targets_sel = rows[sel_rows, sel_cols]
        ce = F.cross_entropy(logits_sel, targets_sel, reduction="none")
        pred = logits_sel.argmax(dim=-1)
        k = 0
        for b, _, js in chunk:
            for j in js:
                losses[b, j - 1] = ce[k]
                predictions[b, j - 1] = pred[k]
                k += 1
    return losses, predictions


# -----------------------------------------------------------------------------
# Sampling

def get_num_transfer_tokens(num_masked, steps):
    """Evenly spread num_masked unmaskings over steps (LLaDA schedule). Returns list of ints of length steps."""
    base, rem = divmod(num_masked, steps)
    return [base + (1 if i < rem else 0) for i in range(steps)]


class DiffusionEngine:
    """
    Drop-in replacement for nanochat.engine.Engine for MDM models (same generate_batch API).

    Semi-autoregressive block decoding: the response is generated block_size tokens at a time, left to
    right. Within a block, all positions start as <|mask|> and are unmasked over `steps_per_block`
    denoising steps; at each step the model predicts every masked position and the most confident
    predictions are committed ("low-confidence remasking"). steps_per_block == block_size means one
    token per step (max quality); fewer steps trades quality for speed.

    Rows complete when they emit <|assistant_end|> or <|bos|> (same as the AR engine). The calculator
    tool protocol is supported: when a finished block contains <|python_end|>, everything after it is
    discarded, the expression is evaluated, and <|output_start|>result<|output_end|> is force-inserted
    (mask=0) before decoding resumes.
    """

    def __init__(self, model, tokenizer, block_size=32, steps_per_block=None, remasking="low_confidence"):
        assert model.config.model_type == "mdm", "DiffusionEngine requires an MDM model"
        assert remasking in ("low_confidence", "random")
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.steps_per_block = steps_per_block if steps_per_block is not None else block_size
        self.remasking = remasking
        self.mask_id = tokenizer.encode_special(MASK_TOKEN)

    @torch.inference_mode()
    def generate_batch(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """
        Returns (results, masks): results[i] is the prompt followed by the generated tokens (terminal
        tokens excluded), masks[i] has 0 for prompt/forced tokens and 1 for sampled tokens.
        max_tokens bounds the number of appended tokens (sampled + forced), like the AR engine.
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start, python_end = get_special("<|python_start|>"), get_special("<|python_end|>")
        output_start, output_end = get_special("<|output_start|>"), get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        P = len(tokens)
        if max_tokens is None:
            max_tokens = self.model.config.sequence_len - P
        L = P + max_tokens
        B = num_samples
        seq = torch.full((B, L), self.mask_id, dtype=torch.long, device=device)
        seq[:, :P] = torch.tensor(tokens, dtype=torch.long, device=device)
        sampled = torch.zeros((B, L), dtype=torch.bool, device=device)  # 1 where the model sampled the token
        cur = [P] * B  # current length of each row
        completed = [False] * B
        in_python = [False] * B
        python_expr = [[] for _ in range(B)]
        pos_idx = torch.arange(L, device=device)

        while not all(completed) and any(c < L for c in cur):
            # active rows and their block windows [cur, cur + block)
            active = [b for b in range(B) if not completed[b] and cur[b] < L]
            win_lo = torch.tensor([cur[b] if b in active else L for b in range(B)], device=device)
            win_hi = torch.clamp(win_lo + self.block_size, max=L)
            window = (pos_idx[None, :] >= win_lo[:, None]) & (pos_idx[None, :] < win_hi[:, None])
            max_win = int((win_hi - win_lo).max().item())
            steps = max(1, min(self.steps_per_block, max_win))
            schedule = get_num_transfer_tokens(max_win, steps)
            for k in schedule:
                still_masked = window & (seq == self.mask_id)
                if not still_masked.any():
                    break
                sel = still_masked.nonzero(as_tuple=True)  # only the still-masked window positions matter
                logits_sel = self.model(seq, select=sel)  # (n, V): logits only at those positions
                x0_sel = sample_next_token(logits_sel, rng, temperature, top_k).view(-1)
                if self.remasking == "low_confidence":
                    scaled = logits_sel / temperature if temperature > 0 else logits_sel
                    conf_sel = F.softmax(scaled, dim=-1).gather(1, x0_sel.view(-1, 1)).view(-1)
                else:
                    conf_sel = torch.rand((x0_sel.numel(),), device=device, generator=rng)
                x0 = seq.clone()
                x0[sel] = x0_sel
                conf = torch.full((B, L), -float("inf"), device=device)
                conf[sel] = conf_sel
                # commit the k most confident still-masked positions in each row's window
                top_conf, top_pos = torch.topk(conf, k=min(k, L), dim=1)
                commit = torch.zeros_like(still_masked)
                valid = top_conf > -float("inf")
                commit.scatter_(1, top_pos, valid)
                seq = torch.where(commit, x0, seq)
                sampled = sampled | commit

            # block finished: scan new tokens for terminal tokens and tool calls, row by row
            for b in active:
                lo, hi = cur[b], int(win_hi[b].item())
                row = seq[b, lo:hi].tolist()
                new_cur = hi
                for off, tok in enumerate(row):
                    pos = lo + off
                    if tok == assistant_end or tok == bos:
                        completed[b] = True
                        new_cur = pos  # terminal token excluded from the result
                        break
                    if tok == python_start:
                        in_python[b] = True
                        python_expr[b] = []
                    elif tok == python_end and in_python[b]:
                        in_python[b] = False
                        forced = []
                        if python_expr[b]:
                            result = use_calculator(self.tokenizer.decode(python_expr[b]))
                            if result is not None:
                                forced = [output_start] + self.tokenizer.encode(str(result)) + [output_end]
                        python_expr[b] = []
                        new_cur = pos + 1
                        if forced:
                            n = min(len(forced), L - new_cur)
                            seq[b, new_cur:new_cur + n] = torch.tensor(forced[:n], dtype=torch.long, device=device)
                            sampled[b, new_cur:new_cur + n] = False
                            new_cur += n
                        break  # discard anything the model wrote after the tool call, regenerate
                    elif in_python[b]:
                        python_expr[b].append(tok)
                if new_cur < L and not completed[b]:
                    seq[b, new_cur:] = self.mask_id  # reset everything beyond the committed prefix
                    sampled[b, new_cur:] = False
                cur[b] = new_cur
                if cur[b] >= L:
                    completed[b] = True

        results, masks = [], []
        for b in range(B):
            n = cur[b]
            results.append(seq[b, :n].tolist())
            masks.append([0] * P + sampled[b, P:n].long().tolist())
        return results, masks


def make_engine(model, tokenizer, **mdm_kwargs):
    """Return the right inference engine for the model type."""
    if getattr(model.config, "model_type", "ar") == "mdm":
        return DiffusionEngine(model, tokenizer, **mdm_kwargs)
    from nanochat.engine import Engine
    return Engine(model, tokenizer)
