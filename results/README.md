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
The AR model cannot see ahead, hedges, and never saturates. Diagnostics (200 SFT steps each, categorical ChatCORE on 400 problems/task; the AR reaches 0.067 at step 200):

| variant | ChatCORE_cat @200 | reading |
|---|---|---|
| packed, min mask rate 0.1 / 0.3 / 1.0 | 0.006 / 0.001 / -0.009 | 1/p weight variance is not the cause |
| packed, softcap off / 50 | -0.002 / -0.004 | saturation is a symptom, not the cause |
| packed, unembedding LR / 10 | 0.004 | output-layer drift is not the cause |
| MMLU-only (MDM, MDM eps=1, **AR control**) | 0.011 / 0.020 / **0.001** | degenerate: all collapse to uniform letters, val bpb 7-9; uninformative |
| padded (LLaDA-style, one conv/row, maskable end padding) | 0.008 | response-length shortcut is not the cause |
| MMLU-only at 0.1x LR: AR / MDM | **0.100 / 0.049** | the MDM *can* learn the letter task, at about half the AR's rate, once the LR is lowered |
| mixture at 0.1x and 0.3x LR (MDM), 0.1x (AR control) | pending | tuned-vs-tuned SFT learning rate |

Plain exam-format probe (no chat tokens, "Answer:" then next token / mask), ARC-Easy, 298 questions:
AR base 24%, MDM base 23%, MDM 2x base 26%, AR SFT 34%, MDM SFT 26%. Neither base model has the letter skill;
SFT gives it to the AR model only.

Local probe on 200 MMLU test questions in the training format: diffusion SFT model CE on the masked letter = 1.385
(uniform over A-D, chance = 1.386) in every context variant; AR SFT model CE 1.445 with 31.5% accuracy; the
pre-SFT diffusion model is uncalibrated (CE 1.5-1.9) and also non-discriminative.

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
