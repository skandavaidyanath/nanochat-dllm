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
| d12_mdm_lr0.01 / lr0.04 | MDM LR sweep | done | 1.062 / 1.075 (ELBO) | 1.045 / 1.062 | 0.118 / 0.102 | MDM slightly prefers the lower LR (0.01); see plots/lr_sweep.png |
| d12_mdm_4x | MDM, 5.28B tokens | done | 0.992 (ELBO) | 0.979 | 0.134 | 292 min; gap to AR-1x 0.13 bpb; CORE within noise of AR; see finding 2 |
| d12_ar_rep10 | AR, same budget, 10 epochs on 136M tokens | done | 1.101 (best 0.967 at 5 epochs) | n/a | 0.120 | overfits after 5 epochs; train loss 2.16 vs 2.80; see finding 3 |
| d12_mdm_rep10 | MDM, same budget, 10 epochs on 136M tokens | done | 1.072 (ELBO) | 1.055 | 0.085 | no overfitting: identical to the full-data MDM curve; see finding 3 |
| decode bench | AR (KV cache) vs MDM at 1-32 tokens/pass | done | | | | d12 base models, 256 tokens, batch 1/8/32; see finding 4 and plots/decode_bench.png |

## Findings so far

**1. Diffusion SFT is far slower and weaker than AR SFT on the same mixture (d12): the diffusion model
learns the multiple-choice skill only when the MC share is raised, the LR lowered and the run lengthened.**
After identical SFT, the AR model scores 36/33/31% on ARC-E/ARC-C/MMLU; the diffusion model scores chance,
and 0 on GSM8K / HumanEval. The first symptom found was logit-softcap saturation at the answer position
(raw letter logits 38-46 vs 15-29 for the AR model), but the diagnostics below show saturation is a symptom:
removing the cap, bounding the 1/p loss weights and lowering the unembedding LR all leave the MC score at
chance. What does move it: MC-only data at 0.1x LR (the MDM learns at about half the AR rate), and on the
mixture only a 3x MMLU share at 0.3x LR with twice the steps (0.028 at step 400 vs AR 0.067 at step 200).
Working explanation: with random masking the per-token gradient from a one-token answer slot is a tiny,
noisy share of a loss dominated by long SmolTalk responses, so the MC signal is drowned; the AR model gets a
clean next-token gradient at that slot every time. Diagnostics (200 SFT steps each, categorical ChatCORE on
400 problems/task; the AR reaches 0.067 at step 200):

| variant | ChatCORE_cat @200 | reading |
|---|---|---|
| packed, min mask rate 0.1 / 0.3 / 1.0 | 0.006 / 0.001 / -0.009 | 1/p weight variance is not the cause |
| packed, softcap off / 50 | -0.002 / -0.004 | saturation is a symptom, not the cause |
| packed, unembedding LR / 10 | 0.004 | output-layer drift is not the cause |
| MMLU-only (MDM, MDM eps=1, **AR control**) | 0.011 / 0.020 / **0.001** | degenerate: all collapse to uniform letters, val bpb 7-9; uninformative |
| padded (LLaDA-style, one conv/row, maskable end padding) | 0.008 | response-length shortcut is not the cause |
| MMLU-only at 0.1x LR: AR / MDM | **0.100 / 0.049** | the MDM *can* learn the letter task, at about half the AR's rate, once the LR is lowered |
| mixture at 0.1x / 0.3x LR (MDM); 0.1x (AR control) | 0.001 / 0.014; **AR 0.036** | lower LR does not rescue the MDM on the mixture; the AR still learns |
| mixture with 3x the MMLU share, 0.3x LR (MDM), 400 steps | 0.000 @200, 0.028 @400 | weak signal emerging with more MC share and more steps: consistent with the MC gradient being drowned by the diffusion loss on long responses |

Plain exam-format probe (no chat tokens, "Answer:" then next token / mask), ARC-Easy, 298 questions:
AR base 24%, MDM base 23%, MDM 2x base 26%, AR SFT 34%, MDM SFT 26%. Neither base model has the letter skill;
SFT gives it to the AR model only.

Local probe on 200 MMLU test questions in the training format: diffusion SFT model CE on the masked letter = 1.385
(uniform over A-D, chance = 1.386) in every context variant; AR SFT model CE 1.445 with 31.5% accuracy; the
pre-SFT diffusion model is uncalibrated (CE 1.5-1.9) and also non-discriminative.


**2. Pretraining: at d12 the diffusion model does not reach the AR validation loss even with 4x the tokens, but
its CORE score does catch up.** Left-to-right NLL (exact likelihood of the L2R-decoded model; the ELBO is
within 0.015 of it everywhere):

| tokens | AR | MDM (L2R) | gap | MDM CORE (AR 0.142, noise about +/-0.005) |
|---|---|---|---|---|
| 1.32B (1x) | 0.847 | 1.055 | 0.208 | 0.109 |
| 2.64B (2x) | | 1.009 | 0.162 | 0.131 |
| 5.28B (4x) | | 0.979 | 0.132 | 0.134 |

Each doubling buys the MDM about 0.64x the previous gain (0.047 then 0.030 bpb). A three-point power-law fit
puts its asymptote near 0.93 bpb for this model size, above the AR model's 1x result, so at d12 adding tokens
alone does not produce a likelihood crossover (consistent with the ~16x compute gap reported for MDMs in the
literature). The AR model was not trained on 2x/4x tokens, so this is "how much extra data the MDM needs to
match a fixed AR budget", not a compute-matched comparison. CORE behaves differently: by 4x the MDM is within
noise of the AR model (0.134 vs 0.140-0.147 across the AR LR sweep), winning bigbench_cs_algorithms, lambada,
squad, bigbench_operators and commonsense_qa and losing copa, winograd, hellaswag, piqa and arc_easy. Training
throughput is 332K vs 349K tok/s (5% slower, same FA3 kernels; the bidirectional model simply has no causal
skip). Plot: plots/val_bpb_vs_tokens.png.

**3. Data-constrained regime: the diffusion model does not overfit after 10 epochs; the AR model does.**
Same 1.32B-token budget and LR schedule, data restricted to the first 4 training shards (136M tokens, ~9.7
epochs; the validation shard is the 5th shard rather than the last one, but both models match their full-data
run to 0.0003 bpb at step 250, so the two val shards are equivalent).

| run | final val bpb | best val bpb | final train loss | CORE |
|---|---|---|---|---|
| AR, full data | 0.847 | 0.847 | 2.80 | 0.142 |
| AR, 10 epochs | **1.101** | 0.967 at 5 epochs | 2.16 (memorising) | 0.120 |
| MDM, full data (L2R) | 1.055 | 1.055 | 3.49 | 0.109 |
| MDM, 10 epochs (L2R) | **1.055** | 1.055 | 3.44 | 0.085 |

The MDM's validation curve on 136M tokens lies on top of its full-data curve for the entire run (ELBO 1.072
vs 1.066), while the AR model bottoms out at 5 epochs and then rises 0.134 bpb. At the final checkpoint the
diffusion model is better than the AR model by 0.046 bpb; an early-stopped AR model (0.967) still beats it by
0.088. Mechanism: every epoch presents a fresh random masking of each sequence, so the diffusion objective
sees a new prediction problem per pass, whereas next-token targets are identical every epoch. The benefit does
not carry to CORE at this scale: both models lose about 0.02 CORE on the repeated subset (the subset is also
a narrower slice of ClimbMix). Plot: plots/datarep.png.

**4. Decoding throughput: the diffusion model is faster only when it commits at least 2-4 tokens per forward
pass.** d12 base models, nanochat's Python engines, 256 generated tokens after a 7-token prompt, one H100 PCIe
(`scripts/infer_bench_mdm.py`, log in `d12/decode_bench.log`):

| decoder | tokens / pass | batch 1 | batch 8 | batch 32 |
|---|---|---|---|---|
| AR, KV cache | 1 | 97 tok/s | 768 | 3,013 |
| MDM, 32 steps / 32-token block | 1 | 0.6x | 0.5x | 0.5x |
| MDM, 16 steps | 2 | 1.6x | 1.0x | 0.9x |
| MDM, 8 steps | 4 | 3.1x | 2.9x | 2.6x |
| MDM, 4 steps | 8 | 6.3x | 6.1x | 5.7x |
| MDM, 1 step | 32 | 24x | 23x | 18x |

At one token per pass (the same number of forward passes as AR) the diffusion decoder is half the AR speed,
because it recomputes the whole block with no KV cache. Neither engine is optimised (no CUDA graphs, no prefix
cache for the MDM), so the absolute numbers are not production numbers; the relative picture is the result.
The accuracy side of this trade-off (how much quality is lost at 4-8 tokens per pass) could not be measured at
d12 because the d12 diffusion SFT model has no measurable GSM8K / HumanEval accuracy; it is deferred to d24.
Plot: plots/decode_bench.png.

## Phase 2: d24 (speedrun config, 8xH100 on Lightning)

Not started. `runs/exp_d24_pair.sh` (depth 24, ratio 8, fp8, device batch 16, 8 GPUs) trains the AR and MDM
models at matched tokens, then runs SFT + chat evals for both. Needs the 8xH100 VM
(`lightning vm create nanochat-d24 --machine h100_x_8 --volume-size 800`) and a Lightning credit top-up
(roughly $250-300 at $16/hr: two ~6 h pretrains, evals, SFT). Before launching, based on the d12 findings:

- Give the MDM its own SFT recipe (`runs/exp_posttrain.sh` currently uses the AR recipe for both): 3x MMLU
  share, Muon LR 0.3x, 2-3x the steps; keep the AR run on the default recipe and also run the AR on the MDM
  recipe as a control.
- Measure accuracy vs tokens-per-pass on GSM8K / HumanEval with the d24 MDM SFT model (finding 4).
- Make the job chain terminate the VM itself when it finishes (the d12 pod idled ~10 h after its last job).

## Phase 3: trade-offs

Done at d12: decoding throughput vs tokens-per-pass (finding 4), data-constrained regime (finding 3).
Open: accuracy vs decoding steps (needs a diffusion model with task accuracy, i.e. d24), infilling.

## Cost log

| date | what | cost |
|---|---|---|
| 2026-10-06 | A100 smoke/benchmark pod (Massed Compute via Prime Intellect, 25 min) | ~$0.50 |
| 2026-10-06 to 10-08 | H100 PCIe pod for the whole d12 phase (Massed Compute via Prime Intellect, $2.35/hr, 44.5 h; terminated 2026-10-08 18:20 UTC) | ~$105 by the hourly rate, of which ~$23 was ~10 h idle after the last job finished at 08:26 UTC on 10-08 |

## Validation runs

- Mac (MPS, d6) and A100: all ten pipeline stages pass for both model types; logs in `a100_smoke/`.
- H100: 74/74 tests including FA3-vs-SDPA for bidirectional and windowed attention.
