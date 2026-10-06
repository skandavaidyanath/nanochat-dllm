# AR LLM vs masked-diffusion LLM (dLLM) on nanochat

Goal: with everything else held fixed (architecture, tokenizer, data and data order, optimizer, schedule,
token budget, post-training data), measure

1. whether a masked-diffusion LM reaches the same pretraining validation loss as the autoregressive LM,
2. whether it post-trains (SFT, RL) as well, and
3. what the other trade-offs are (decoding cost vs quality, data repetition, infilling, eval cost).

## What differs between the two models

| | AR (`--model-type=ar`, nanochat default) | dLLM (`--model-type=mdm`) |
|---|---|---|
| attention | causal, sliding windows `(w, 0)` | bidirectional, symmetric windows `(w, w)` |
| vocabulary | shared tokenizer, `<|mask|>` unused | same tokenizer, `<|mask|>` used |
| objective | next-token cross-entropy | LLaDA/MDLM masked diffusion: mask each token w.p. `p~U(eps,1)`, loss `CE/p` on masked tokens, normalized by #tokens (= negative ELBO per token) |
| SFT | loss on assistant tokens (shifted) | same tokens are the maskable set, prompt stays clean, loss in place |
| RL (GSM8K) | on-policy REINFORCE with per-token log-probs | same, per-token log-prob replaced by the masked-ELBO estimate |
| decoding | KV-cached sampling | semi-autoregressive block decoding with low-confidence remasking (`DiffusionEngine`) |

Everything else is byte-identical: `nanochat/gpt.py` blocks, Muon/AdamW setup, LR/WD schedules, data loader, packing.

## Metrics

Pretraining (all in bits per byte on the held-out ClimbMix shard):

- `val/bpb` for AR = exact NLL.
- `val/bpb` for MDM = Monte Carlo estimate of the negative ELBO. An **upper bound** on the true NLL: if it is
  below the AR number the dLLM wins on likelihood outright; if above, the true gap is unknown.
- `val/bpb_l2r` for MDM = NLL of the left-to-right-decoded diffusion model (mask the suffix, score the next token).
  This is the exact likelihood of a well-defined generative model, so it is directly comparable to the AR number.
- CORE (22 DCLM tasks, likelihood-free for the comparison). MDM scores continuations with `--mdm-score=l2r`
  (exact, one forward per scored token) in `base_eval`; the cheap `mf` mode masks the whole continuation at once.

Post-training: ARC-Easy/Challenge, MMLU (categorical: argmax over answer letters at the answer position),
GSM8K and HumanEval (generative), ChatCORE = mean centered accuracy. Report absolute numbers and the delta over
each model's own SFT checkpoint for RL. Several SFT seeds because post-training is cheap.

## Phases

1. **d12 sweep** (`runs/exp_d12_sweep.sh`): matched-token pair at ratio 12 with a fixed 524K batch; Muon LR sweep
   {0.01, 0.02, 0.04} per architecture; MDM at 2x and 4x tokens to find the crossover; SFT on the matched pair.
2. **d24 pair** (`runs/exp_d24_pair.sh`): the speedrun config (ratio 8, fp8), both models, SFT, full evals, RL after
   inspecting the SFT gap.
3. **Trade-offs**: decoding steps vs accuracy vs tokens/s (`DiffusionEngine(steps_per_block=...)`), data-repetition
   regime (fewer shards, more epochs), infilling, eval cost.

## Logging

`--run=<name>` writes TensorBoard events and `metrics.jsonl` to `$NANOCHAT_BASE_DIR/logs/<project>/<name>/`.
`tensorboard --logdir ~/.cache/nanochat/logs` on the box, port-forward 6006 over SSH.

## Reading list

LLaDA (Nie et al. 2025), MDLM (Sahoo et al. 2024), SMDM scaling laws (Nie et al. 2024), Dream 7B (Ye et al. 2025),
d1 / diffu-GRPO (Zhao et al. 2025), LLaDA 1.5 / VRPO, "Diffusion beats AR in data-constrained settings"
(Prabhudesai et al. 2025), Zheng et al. 2024 on time-agnostic masked models, nanoLLaDA (Lukas-Xue).
