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
| d12_ar SFT + chat eval | post-training | running | | | | |
| d12_mdm SFT + chat eval | post-training | queued | | | | |
| d12_mdm_2x | MDM, 2.64B tokens | queued | | | | crossover search |
| d12_ar_lr0.01 / lr0.04 | AR LR sweep | queued | | | | |
| d12_mdm_lr0.01 / lr0.04 | MDM LR sweep | queued | | | | |
| d12_mdm_4x | MDM, 5.28B tokens | queued | | | | crossover search |

## Phase 2: d24 (speedrun config, 8xH100 on Lightning)

Not started. Needs the 8xH100 VM (`lightning vm create ... --machine h100_x_8`) and a Lightning credit top-up.

## Phase 3: trade-offs

Decoding steps vs accuracy vs throughput; data-constrained regime (`--max-shards`); infilling. Not started.

## Cost log

| date | what | cost |
|---|---|---|
| 2026-10-06 | A100 smoke/benchmark pod (Massed Compute via Prime Intellect, 25 min) | ~$0.50 |
| 2026-10-06 | H100 PCIe pod for the d12 sweep (Massed Compute via Prime Intellect, $2.35/hr) | running |

## Validation runs

- Mac (MPS, d6) and A100: all ten pipeline stages pass for both model types; logs in `a100_smoke/`.
- H100: 74/74 tests including FA3-vs-SDPA for bidirectional and windowed attention.
