# Results tracker: AR LLM vs masked-diffusion LLM on nanochat

Design: [docs/EXPERIMENT.md](../docs/EXPERIMENT.md). Code: `--model-type=ar|mdm` on every nanochat script.

## How to update

```bash
bash results/sync_logs.sh prime-d12 d12   # pull metrics.jsonl / eval CSVs / stage logs from the GPU box
python results/plot.py --phase d12        # regenerate results/d12/plots/*.png and results/d12/summary.md
```

`results/logs/<project>/<run>/metrics.jsonl` is the raw record (also viewable with `tensorboard --logdir results/logs`).

## Phase 1: d12 (1.32B tokens at ratio 12, batch 524,288, single H100 PCIe on Prime Intellect)

Latest numbers and plots: [d12/summary.md](d12/summary.md)

| run | what | status | val bpb | L2R bpb | CORE | notes |
|---|---|---|---|---|---|---|
| d12_ar | AR baseline | done | 0.847 | n/a (exact) | 0.142 | 63 min, 349K tok/s |
| d12_mdm | MDM, matched tokens | done | 1.068 (ELBO) | 1.053 | 0.109 | 66 min, 332K tok/s (5% slower than AR with FA3); gap to AR shrank from 0.35 bpb at step 250 to 0.21 at the end |
| d12_ar SFT + chat eval | post-training | done | SFT val bpb 0.356 | | ChatCORE 0.084 | ARC-E 36.1 / ARC-C 32.5 / MMLU 31.2 / GSM8K 0.8 / HumanEval 7.9 | | | | |
| d12_mdm SFT + chat eval | post-training | done (fix pending) | SFT val bpb 0.624 | | ChatCORE -0.001 | ARC-E 25.6 / ARC-C 25.4 / MMLU 23.6 = chance, GSM8K 0.0 / HumanEval 0.0; see finding 1 | | | | |
| d12_mdm_2x | MDM, 2.64B tokens | done | 1.022 (ELBO) | 1.009 | 0.131 | 132 min; gap to AR-1x shrank to 0.16 bpb | | | | crossover search |
| d12_ar_lr0.01 / lr0.04 | AR LR sweep | done | 0.850 / 0.850 | | 0.147 / 0.140 | AR insensitive to Muon LR; default 0.02 is fine |
| d12_mdm_lr0.01 / lr0.04 | MDM LR sweep | done / eval running | 1.062 / 1.075 (ELBO) | 1.045 / 1.062 | 0.118 / - | MDM slightly prefers the lower LR (0.01) |
| d12_mdm_4x | MDM, 5.28B tokens | not started (credits exhausted) | | | | crossover search |

## Findings so far

**1. Diffusion SFT on nanochat saturates the logit softcap and loses multiple-choice discrimination (d12).**
After identical SFT, the AR model scores 36/33/31% on ARC-E/ARC-C/MMLU; the diffusion model scores chance.
Probing the answer position: the diffusion SFT model puts all mass on {A,B,C,D} but its four raw logits are
38-46, inside the saturated region of nanochat's `15*tanh(x/15)` cap, and identical across questions; the AR
model's raw letter logits are 15-29 and vary by 5-10 nats with the question; the pre-SFT diffusion model's
are 6-9 and do vary. The drift to saturation is ~0.12 raw logits/step from step 0 and is *independent of the
1/p loss weighting* (min mask rate 0.1 / 0.3 / 1.0 all saturate by step 200, `diag_mdm_sft_eps*`). Output-row
norms are unchanged, so the hidden state is what aligned with the shared "letter" direction. Working
explanation: in packed bidirectional training the model sees where the next conversation starts, so it is
certain the slot is one token; that gives a consistent gradient direction which Muon's orthogonalized updates
turn into steady drift until the cap saturates and the question-specific gradient (scaled by tanh') dies.
The AR model cannot see ahead, hedges, and never saturates. Fix candidates under test: lower unembedding LR
(`diag_mdm_sft_unemb0.0004`), softcap removed / raised to 50 (`diag_mdm_sft_cap0`, `cap50`).

## Phase 2: d24 (speedrun config, 8xH100 on Lightning)

Not started. Needs the 8xH100 VM (`lightning vm create ... --machine h100_x_8`) and a Lightning credit top-up.

## Phase 3: trade-offs

Decoding steps vs accuracy vs throughput; data-constrained regime (`--max-shards`); infilling. Not started.

## Cost log

| date | what | cost |
|---|---|---|
| 2026-10-06 | A100 smoke/benchmark pod (Massed Compute via Prime Intellect, 25 min) | ~$0.50 |
| 2026-10-06/07 | H100 PCIe pod for the d12 sweep (Massed Compute via Prime Intellect, $2.35/hr, 21.5 h) | ~$50 (credit exhausted) |

## Validation runs

- Mac (MPS, d6) and A100: all ten pipeline stages pass for both model types; logs in `a100_smoke/`.
- H100: 74/74 tests including FA3-vs-SDPA for bidirectional and windowed attention.
