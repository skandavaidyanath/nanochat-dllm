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
| d12_mdm SFT + chat eval | post-training | done | SFT val bpb 0.624 | | ChatCORE -0.001 | ARC-E 25.6 / ARC-C 25.4 / MMLU 23.6 = chance, GSM8K 0.0 / HumanEval 0.0; see findings 1 and 5 | | | | |
| d12 SFT recipe sweep, 5 arms, + 2 probes | post-training, phase 2a | done 2026-10-24 | | | | best MDM recipe (3x MMLU share, 0.3x LR): ARC-E 30.1 / ARC-C 29.7 / MMLU 27.4; AR on the same recipe 35.0 / 31.3 / 29.9; see findings 5-6, `d12/sft_sweep/summary.md`, `d12/plots/sft_sweep_*.png` | | | | |
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

**5. Phase 2a: an MDM-specific SFT recipe gets the diffusion model off chance, and the lever is the share of
multiple-choice data, not the learning rate or the training time (d12, one H200, 2026-10-23).** Both d12 base
checkpoints, fresh optimizer (no pretraining momentum, unlike the phase-1 SFT runs). `scripts/chat_sft.py` stops
after one pass over its mixture, so the 3x-MMLU mixture (1.39M conversations) gives 1375 optimizer steps and the
`--num-iterations=29888` arms end at LR-schedule progress 0.74 (warmdown cut short, identically for the MDM and its
AR control); the two-pass arm doubles every dataset's epochs and gets the full warmdown; the default-share arm uses
`--num-iterations=20304` so it ends at the same schedule progress as the x2 arms. Full test sets after SFT:

| run | model | recipe | steps | final SFT val bpb | ARC-E | ARC-C | MMLU | GSM8K |
|---|---|---|---|---|---|---|---|---|
| d12_ar_sft (phase 1) | AR | default | 934 | 0.356 | 36.1 | 32.5 | 31.2 | 0.8 |
| d12_mdm_sft (phase 1) | MDM | default | 934 | 0.624 (ELBO) | 25.6 | 25.4 | 23.6 | 0.0 |
| sft2_ar_m9_lr0.3_x2 | AR | 3x MMLU share, 0.3x LR | 1375 | 0.363 | 35.0 | 31.3 | 29.9 | 2.0 |
| sft2_mdm_m9_lr0.3_x2 | MDM | 3x MMLU share, 0.3x LR | 1375 | 0.632 | **30.1** | **29.7** | **27.4** | 0.0 |
| sft2_mdm_m9_lr0.3_ep2 | MDM | same, two passes, full warmdown | 2749 | 0.590 | 29.8 | 28.1 | 26.7 | 0.0 |
| sft2_mdm_m9_lr0.1_x2 | MDM | 3x MMLU share, 0.1x LR | 1375 | 0.654 | 27.4 | 25.7 | 25.0 | 0.0 |
| sft2_mdm_m3_lr0.3 | MDM | default share, 0.3x LR | 934 | 0.632 | 25.6 | 26.2 | 25.7 | 0.0 |

- The recipe takes the MDM from chance to 30 / 30 / 27 on ARC-E / ARC-C / MMLU; the AR model on the same recipe
  gets 35 / 31 / 30 (slightly below its default 36 / 33 / 31). The gap shrinks from about 11 / 8 / 6 points to
  5 / 2 / 3, mostly by lifting the MDM.
- Two passes with a full warmdown lower the MDM's SFT ELBO from 0.632 to 0.590 but leave the accuracies unchanged:
  more training time is not the missing ingredient at this size.
- 0.1x LR is worse on every task, and the default MMLU share at 0.3x LR stays at chance (so the 0.3x LR alone does
  nothing, confirming the phase-1 diagnostic). The multiple-choice share of the mixture is the lever.
- The likelihood gap from pretraining survives SFT unchanged (AR 0.36 vs MDM 0.63 bpb on the SFT validation
  mixture), and the MDM still scores 0 on GSM8K and HumanEval under every recipe.
- Consequence for d24: the MDM gets the 3x-MMLU / 0.3x-LR / one-pass recipe; the AR model runs on its default
  recipe and on the MDM recipe as the control. Plots: `d12/plots/sft_sweep_curves.png`, `d12/plots/sft_sweep_final.png`.

**6. Phase 2a probes (post-training question 2): the base models carry no exam skill at the answer slot, the MDM's
features there are no worse than the AR model's, and the letter's share of an SFT gradient is half the AR model's.**
`scripts/probe_answer_slot.py` fits a multinomial logistic regression on the frozen vector fed to `lm_head` at the
answer position (MMLU auxiliary_train, 20k questions; MMLU test, 2k; weight decay picked on a held-out tenth):

| checkpoint | feature position | zero-shot letter acc | probe @500 | @2000 | @20000 |
|---|---|---|---|---|---|
| d12_ar | token before the letter (`<\|assistant_start\|>`) | 22.1% | 23.3% | 25.2% | 26.0% |
| d12_mdm | the `<\|mask\|>` at the letter | 25.2% | 24.2% | 24.4% | 26.1% |
| d12_mdm | token before the mask | 25.2% | 23.2% | 26.5% | 27.6% |

Chance is 25%: neither base model has a linearly readable answer at the slot, so SFT has to build the skill from
nothing in both, and the slow MDM learning is not a representation gap. `scripts/probe_grad_share.py` rebuilds the
real SFT micro-batches (same mixture, packing and loss normaliser) with per-token source labels and splits the
gradient by source (16 micro-batches, one optimizer step):

| checkpoint | letter tokens carrying loss / micro-batch | letter share of ‖g‖ | `<\|assistant_end\|>` share | SmolTalk share | GSM8K share |
|---|---|---|---|---|---|
| d12_ar | 19.5 | 9.3% | 3.8% | 97.4% | 4.1% |
| d12_mdm | 10.2 (of 19.5 maskable) | 4.7% | 2.9% | 98.9% | 4.6% |

The letter is masked on about half the draws (p ~ U(0, 1)), so the MDM gets the letter's gradient half as often,
and its SmolTalk gradient is larger (norm 1.00 vs 0.80), so the letter's share of each update is half the AR
model's. The same half-rate supervision is consistent with the phase-1 observation that on multiple-choice-only
data the MDM learned at about half the AR rate (0.049 vs 0.100), which was the part of question 2 left
unexplained: in expectation the 1/p weighting restores the gradient, but half the examples contribute nothing to a
given step and Muon normalises whatever is there.

**7. Phase 2b: forcing the answer to be masked does not help the diffusion model (d12, 2026-10-27).** `--mdm-force-mask-short=2`
always masks assistant responses of at most two tokens (the MMLU letter plus `<|assistant_end|>`) with loss weight 1,
so the letter gets the same deterministic supervision as the AR next-token loss, while long responses keep the
random-rate objective. On the 3x-MMLU / 0.3x-LR recipe (run `sft3_mdm_m9_lr0.3_force2`, same steps and schedule as
`sft2_mdm_m9_lr0.3_x2`): ARC-E 30.4 / ARC-C 26.1 / MMLU 25.8 against 30.1 / 29.7 / 27.4 with random masking, SFT ELBO
identical (0.632). So the half-rate supervision found by the gradient probe is not what holds the diffusion model
back; if anything the bursty 1/p-weighted updates of random masking are better than steady weight-1 updates. The
remaining SFT gap to the AR control (35.0 / 31.3 / 29.9) looks like a base-model gap, which only d24 can test.

**8. Phase 2b: AR-initialised diffusion, partial result (d12, 2026-10-27).** `scripts/base_train.py --init-from-tag=d12_ar
--model-type=mdm` loads the trained AR weights into the bidirectional model and continues with the diffusion loss
(fresh optimizer, 100-step warmup, 1260 steps planned = half the d12 budget, same data order; the mask token's
embedding starts untrained). Validation bpb (ELBO / left-to-right) against the from-scratch diffusion run at the same
step:

| step | AR-initialised MDM | from-scratch MDM | note |
|---|---|---|---|
| 0 | 2.277 / 2.335 | 3.170 / 3.171 | the AR weights already carry 0.9 bpb under the new objective, no training |
| 250 | 1.217 / 1.122 | 1.444 / 1.447 | L2R already below the scratch run's value at step 1250 (1.177) |
| 500 | 1.202 / 1.155 | 1.302 / 1.297 | L2R rose while ELBO fell: the model is giving up its next-token bias for the masked objective |
| 1250 / 1260 | lost | 1.185 / 1.177 | |
| 2520 | | 1.066 / 1.055 | the scratch run's full budget |

The step-500 numbers come from the live log; the metrics after step 300, the final checkpoint and the CORE eval were
lost when the pod terminated itself (see the cost log), so the question of where the adapted model lands at 1260 steps,
and whether it beats the scratch model's full-budget 1.055, is open. What is already clear: starting from AR weights
is worth most of the scratch run's first 1000 steps, and the adapted model is more "left-to-right" than a scratch
diffusion model (L2R below ELBO by 0.05 to 0.09, where the scratch run has them within 0.015). Re-running the 1260
steps plus CORE costs about $7 on a single H200; the plan is to fold it into the d24 pod.

## Phase 2: d24 pair

Not started. Lightning AI has been offline since 2026-10-16 (credential-compromise incident, no ETA), so the pair
will run through Prime Intellect: DataCrunch 8xH100 SXM spot ~$18/hr or on-demand ~$36/hr, Lambda 8xH100 SXM
$31.92/hr (stock changes by the minute; `results/wait_and_launch.sh` polls and launches). Estimate: the two
pretrains plus evals and default SFT in about 4 node-hours, the MDM-recipe SFT confirmation with its AR control in
about 1.5 h, so $100-200 depending on spot. `runs/exp_d24_pair.sh` (depth 24, ratio 8, fp8, device batch 16) still
needs the recipe split below. Pre-launch list:

- Done (phase 2a): the MDM SFT recipe is 3x MMLU share, 0.3x LR, one pass (finding 5); run the AR on its default
  recipe and on the MDM recipe as the control. Still to do: wire this into `runs/exp_d24_pair.sh`.
- Done: launcher with a pod-side deadline watchdog armed before any sync (`results/launch_pod.sh`), stock poller
  (`results/wait_and_launch.sh`), self-terminating chain with local ACK (`results/d12/pod_scripts/chain_sft_sweep.sh`,
  `results/poll_pod.sh`). Needs a d24 chain variant (pretrain both, evals, SFT both recipes).
- Measure accuracy vs tokens-per-pass on GSM8K / HumanEval with the d24 MDM SFT model (finding 4), if it has
  non-zero accuracy there.
- Fold the AR-initialised diffusion re-run (finding 8, 1260 steps + CORE at d12, ~45 min on one GPU) into the d24 pod.
- Results must not depend on the laptop: attach a Prime persistent disk (`prime disks create`, `pods create --disks`) and
  have the chain copy logs and evals there before it terminates the pod.

## Phase 3: trade-offs

Done at d12: decoding throughput vs tokens-per-pass (finding 4), data-constrained regime (finding 3).
Open: accuracy vs decoding steps (needs a diffusion model with task accuracy, i.e. d24), infilling.

## Cost log

| date | what | cost |
|---|---|---|
| 2026-10-06 | A100 smoke/benchmark pod (Massed Compute via Prime Intellect, 25 min) | ~$0.50 |
| 2026-10-06 to 10-08 | H100 PCIe pod for the whole d12 phase (Massed Compute via Prime Intellect, $2.35/hr, 44.5 h; terminated 2026-10-08 18:20 UTC) | ~$105 by the hourly rate, of which ~$23 was ~10 h idle after the last job finished at 08:26 UTC on 10-08 |
| 2026-10-22 | 4xH200 spot pod (DataCrunch via Prime Intellect, $11.66/hr) for the phase-2a SFT sweep; the ssh uploads from the Mac stalled and reset (macOS `IPQoS` issue, fixed with `IPQoS none`) and the chain never started | $15.20, nothing produced |
| 2026-10-23 | 1xH100 PCIe pod (Lambda via Prime Intellect, $3.29/hr) for the same sweep; the chain launch failed silently, a false 'already running' check hid it, and the laptop went offline, so the box idled 19.4 h before being terminated | $63.68, nothing produced. Fix: `results/launch_pod.sh` arms a pod-side deadline watchdog before anything else |
| 2026-10-23 to 10-24 | 1xH200 (Nebius via Prime Intellect, $5.40/hr) for phase 2a: both probes and the 5-arm SFT sweep, 20:45 to 00:17 UTC; the chain terminated the pod itself after the local ACK, the 8 h watchdog was never needed (one earlier Nebius pod failed at ssh setup and was terminated by the launcher within a second, $0.00) | $23.29 |
| 2026-10-27 | 1xH200 (Nebius via Prime Intellect, $5.40/hr) for phase 2b: forced-mask SFT run and the AR-initialised diffusion run, 21:44 to 23:54 UTC; the chain terminated the pod itself 20 min after finishing (the laptop had gone to sleep at ~22:40, so the local sync stopped and the adaptation run's results after step 500 were lost with the pod; the money side worked, the data side needs a persistent disk next time) | $14.28 |

## Validation runs

- Mac (MPS, d6) and A100: all ten pipeline stages pass for both model types; logs in `a100_smoke/`.
- H100: 74/74 tests including FA3-vs-SDPA for bidirectional and windowed attention.
