"""
Tests for the masked diffusion (MDM / dLLM) variant.

python -m pytest tests/test_diffusion.py -v
"""
import math
import torch
import torch.nn.functional as F
import pytest

from nanochat.gpt import GPT, GPTConfig
from nanochat.flash_attention import _sdpa_attention
from nanochat.diffusion import (
    sample_mask_rates, apply_mask, mdm_loss, evaluate_bpb_mdm, score_continuations,
    DiffusionEngine, get_num_transfer_tokens,
)

DEVICE = torch.device("cpu")
VOCAB = 300
MASK_ID = 299


def tiny_model(model_type, seed=0):
    torch.manual_seed(seed)
    config = GPTConfig(sequence_len=64, vocab_size=VOCAB, n_layer=2, n_head=2, n_kv_head=2, n_embd=64,
                       window_pattern="L", model_type=model_type)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=DEVICE)
    model.init_weights()
    # non-trivial projections so that attention actually mixes information across positions
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.1)
            block.mlp.c_proj.weight.normal_(std=0.1)
    model.eval()
    return model


class ByteTokenizer:
    """Byte-level tokenizer stub with nanochat's special tokens (ids 256+)."""
    SPECIAL = ["<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
               "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>", "<|mask|>"]

    def __init__(self):
        self._special = {name: 256 + i for i, name in enumerate(self.SPECIAL)}
        assert self._special["<|mask|>"] == 265

    def encode_special(self, s):
        return self._special[s]

    def get_bos_token_id(self):
        return self._special["<|bos|>"]

    def encode(self, s, prepend=None):
        tokens = list(s.encode("utf-8"))
        return ([prepend] + tokens) if prepend is not None else tokens

    def decode(self, tokens):
        return bytes([t for t in tokens if t < 256]).decode("utf-8", errors="replace")


# -----------------------------------------------------------------------------
# Attention

def test_mdm_attention_is_bidirectional_and_ar_is_causal():
    ids = torch.randint(0, VOCAB - 1, (1, 16))
    ids2 = ids.clone()
    ids2[0, 12] = (ids2[0, 12] + 1) % (VOCAB - 1)  # perturb a *later* token
    for model_type, should_change in [("mdm", True), ("ar", False)]:
        model = tiny_model(model_type)
        with torch.no_grad():
            a = model(ids)[0, :8]
            b = model(ids2)[0, :8]
        changed = not torch.allclose(a, b, atol=1e-5)
        assert changed == should_change, f"{model_type}: earlier logits changed={changed}"


def test_window_sizes_symmetric_for_mdm():
    ar = tiny_model("ar")
    mdm = tiny_model("mdm")
    assert all(right == 0 for _, right in ar.window_sizes)
    assert all(left == right for left, right in mdm.window_sizes)


def test_sdpa_bidirectional_window_matches_reference():
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 10, 8
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    left, right = 2, 3
    out = _sdpa_attention(q, k, v, (left, right), enable_gqa=False, causal=False)
    # reference: explicit masked softmax attention
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    mask = ((i - j) <= left) & ((j - i) <= right)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    scores = scores.masked_fill(~mask, float("-inf"))
    ref = F.softmax(scores, dim=-1) @ v
    assert torch.allclose(out, ref, atol=1e-5)
    # full context equals plain non-causal attention
    out_full = _sdpa_attention(q, k, v, (T, T), enable_gqa=False, causal=False)
    ref_full = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    assert torch.allclose(out_full, ref_full, atol=1e-5)


def test_select_logits_match_full_forward():
    for model_type in ("ar", "mdm"):
        model = tiny_model(model_type)
        ids = torch.randint(0, VOCAB - 1, (3, 20))
        rows = torch.tensor([0, 0, 1, 2, 2])
        cols = torch.tensor([0, 5, 19, 7, 3])
        with torch.no_grad():
            full = model(ids)
            sparse = model(ids, select=(rows, cols))
        assert sparse.shape == (5, VOCAB)
        assert torch.allclose(full[rows, cols], sparse, atol=1e-5)


# -----------------------------------------------------------------------------
# Masking and loss

def test_mask_rates_are_stratified():
    B = 8
    p = sample_mask_rates(B, DEVICE)
    assert p.shape == (B,) and (p > 0).all() and (p <= 1).all()
    bins = ((torch.sort(p).values - 1e-3) / (1 - 1e-3) * B).floor().long()
    assert bins.tolist() == list(range(B))  # exactly one draw per bin


def test_apply_mask_respects_maskable_and_rate():
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB - 1, (4, 2000))
    maskable = torch.ones_like(ids, dtype=torch.bool)
    maskable[:, :1000] = False
    p = torch.tensor([0.1, 0.5, 0.9, 1.0])
    noisy, masked = apply_mask(ids, maskable, p, MASK_ID)
    assert not masked[:, :1000].any()
    assert (noisy[masked] == MASK_ID).all() and (noisy[~masked] == ids[~masked]).all()
    rates = masked[:, 1000:].float().mean(dim=1)
    assert torch.allclose(rates, p, atol=0.06)


def test_mdm_loss_mean_and_none_are_consistent():
    model = tiny_model("mdm")
    torch.manual_seed(1)
    ids = torch.randint(0, VOCAB - 1, (4, 32))
    maskable = torch.ones_like(ids, dtype=torch.bool)
    g = torch.Generator().manual_seed(123)
    loss = mdm_loss(model, ids, maskable, MASK_ID, generator=g)
    g = torch.Generator().manual_seed(123)
    loss_tok, masked = mdm_loss(model, ids, maskable, MASK_ID, generator=g, reduction="none")
    assert loss_tok.shape == ids.shape
    assert (loss_tok[~masked] == 0).all() and (loss_tok[masked] > 0).all()
    assert torch.allclose(loss, loss_tok.sum() / maskable.sum())
    # sanity range for an untrained model (nats per token)
    assert 0 < loss.item() < 2 * math.log(VOCAB)


def test_mdm_loss_only_supervises_maskable_tokens():
    model = tiny_model("mdm")
    ids = torch.randint(0, VOCAB - 1, (2, 32))
    maskable = torch.zeros_like(ids, dtype=torch.bool)
    maskable[:, 16:] = True
    loss_tok, masked = mdm_loss(model, ids, maskable, MASK_ID, reduction="none")
    assert not masked[:, :16].any() and (loss_tok[:, :16] == 0).all()


# -----------------------------------------------------------------------------
# Likelihood evaluation

def test_evaluate_bpb_mdm_elbo_and_l2r():
    model = tiny_model("mdm")
    torch.manual_seed(2)
    token_bytes = torch.ones(VOCAB, dtype=torch.int32)
    token_bytes[256:] = 0  # special tokens do not count
    def batches():
        while True:
            ids = torch.randint(0, 256, (4, 32))
            yield ids, torch.ones_like(ids, dtype=torch.bool)
    elbo = evaluate_bpb_mdm(model, batches(), 3, token_bytes, MASK_ID, mode="elbo", seed=0)
    l2r = evaluate_bpb_mdm(model, batches(), 3, token_bytes, MASK_ID, mode="l2r", seed=0)
    for v in (elbo, l2r):
        assert math.isfinite(v) and 0 < v < 2 * math.log2(VOCAB)


def test_score_continuations_layout_and_l2r_exactness():
    model = tiny_model("mdm")
    torch.manual_seed(3)
    T = 12
    ids = torch.randint(0, 256, (2, T))
    starts, ends = [5, 7], [9, 12]
    losses, preds = score_continuations(model, ids, starts, ends, MASK_ID, method="l2r")
    for b, (s, e) in enumerate(zip(starts, ends)):
        assert torch.isnan(losses[b, :s - 1]).all() and torch.isnan(losses[b, e - 1:]).all()
        assert torch.isfinite(losses[b, s - 1:e - 1]).all()
    # direct check of one entry: row 0, score token j=6 with positions >= 6 masked
    j = 6
    noisy = ids[0:1].clone()
    noisy[0, j:] = MASK_ID
    with torch.no_grad():
        logits = model(noisy)[0, j]
    ref = F.cross_entropy(logits[None], ids[0, j][None]).item()
    assert abs(losses[0, j - 1].item() - ref) < 1e-4
    assert preds[0, j - 1].item() == logits.argmax().item()
    # mean-field variant has the same layout
    losses_mf, _ = score_continuations(model, ids, starts, ends, MASK_ID, method="mf")
    assert torch.isfinite(losses_mf[0, 4:8]).all() and torch.isnan(losses_mf[0, 8:]).all()


# -----------------------------------------------------------------------------
# Sampler

def test_transfer_schedule_sums_to_total():
    for n, steps in [(32, 32), (32, 8), (10, 4), (3, 8)]:
        sched = get_num_transfer_tokens(n, steps)
        assert len(sched) == steps and sum(sched) == n and min(sched) >= 0


def test_diffusion_engine_fills_every_position():
    model = tiny_model("mdm")
    tok = ByteTokenizer()
    # the tiny model has vocab 300 but the stub tokenizer's mask id is 265: make a model-side view consistent
    engine = DiffusionEngine(model, tok, block_size=8, steps_per_block=4)
    prompt = tok.encode("hi", prepend=tok.get_bos_token_id())
    for temperature in (0.0, 1.0):
        results, masks = engine.generate_batch(prompt, num_samples=3, max_tokens=20, temperature=temperature, top_k=50, seed=7)
        assert len(results) == 3 and len(masks) == 3
        for r, m in zip(results, masks):
            assert r[:len(prompt)] == prompt
            assert len(r) == len(m) and len(r) <= len(prompt) + 20
            assert engine.mask_id not in r
            assert m[:len(prompt)] == [0] * len(prompt) and all(x == 1 for x in m[len(prompt):])
    # determinism: same seed -> same samples
    r1, _ = engine.generate_batch(prompt, num_samples=2, max_tokens=12, temperature=1.0, seed=3)
    r2, _ = engine.generate_batch(prompt, num_samples=2, max_tokens=12, temperature=1.0, seed=3)
    assert r1 == r2


class ScriptedModel:
    """Model stub whose argmax at every position past the prompt follows a fixed script (relative to the prompt)."""
    def __init__(self, script, vocab=300, prompt_len=0):
        self.script = script
        self.vocab = vocab
        self.prompt_len = prompt_len
        from dataclasses import dataclass
        @dataclass
        class C:
            model_type: str = "mdm"
            sequence_len: int = 256
        self.config = C()

    def get_device(self):
        return DEVICE

    def __call__(self, ids, select=None):
        B, T = ids.shape
        logits = torch.zeros(B, T, self.vocab)
        for t in range(self.prompt_len, T):
            k = t - self.prompt_len
            tok = self.script[k] if k < len(self.script) else 0
            logits[:, t, tok] = 10.0
        if select is not None:
            return logits[select[0], select[1]]
        return logits


def test_diffusion_engine_stops_at_assistant_end():
    tok = ByteTokenizer()
    prompt = tok.encode("q", prepend=tok.get_bos_token_id())
    script = tok.encode("ab") + [tok.encode_special("<|assistant_end|>")] + tok.encode("zzz")
    engine = DiffusionEngine(ScriptedModel(script, prompt_len=len(prompt)), tok, block_size=4)
    results, masks = engine.generate_batch(prompt, num_samples=2, max_tokens=16, temperature=0.0)
    for r, m in zip(results, masks):
        assert r == prompt + tok.encode("ab")
        assert m == [0] * len(prompt) + [1, 1]


def test_diffusion_engine_tool_use_injects_output():
    tok = ByteTokenizer()
    prompt = tok.encode("q", prepend=tok.get_bos_token_id())
    ps, pe = tok.encode_special("<|python_start|>"), tok.encode_special("<|python_end|>")
    os_, oe = tok.encode_special("<|output_start|>"), tok.encode_special("<|output_end|>")
    # script: x <python_start> 1+1 <python_end> junk ... the junk after python_end must be discarded
    script = tok.encode("x") + [ps] + tok.encode("1+1") + [pe] + tok.encode("junkjunk")
    model = ScriptedModel(script, prompt_len=len(prompt))
    engine = DiffusionEngine(model, tok, block_size=8)
    results, masks = engine.generate_batch(prompt, num_samples=1, max_tokens=12, temperature=0.0)
    r, m = results[0], masks[0]
    expected_prefix = prompt + tok.encode("x") + [ps] + tok.encode("1+1") + [pe] + [os_] + tok.encode("2") + [oe]
    assert r[:len(expected_prefix)] == expected_prefix
    n_forced = 3  # <output_start> '2' <output_end>
    assert m[:len(prompt)] == [0] * len(prompt)
    assert m[len(prompt):len(prompt) + 6] == [1] * 6  # x <ps> 1 + 1 <pe>
    assert m[len(prompt) + 6:len(prompt) + 6 + n_forced] == [0] * n_forced
    assert len(r) <= len(prompt) + 12
