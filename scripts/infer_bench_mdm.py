"""
Decoding throughput: AR engine (KV cache) vs DiffusionEngine at several denoising budgets.
Generates `--gen-tokens` tokens after a short prompt, batch sizes 1 and 8, reports tokens/s.
    python -m scripts.infer_bench_mdm --ar-tag d12_ar --mdm-tag d12_mdm --gen-tokens 256
"""
import argparse, time, torch
from nanochat.common import compute_init, autodetect_device_type, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.diffusion import DiffusionEngine

ap = argparse.ArgumentParser()
ap.add_argument("--ar-tag", default="d12_ar")
ap.add_argument("--mdm-tag", default="d12_mdm")
ap.add_argument("--source", default="base")
ap.add_argument("--gen-tokens", type=int, default=256)
ap.add_argument("--block-size", type=int, default=32)
ap.add_argument("--batch-sizes", default="1,8")
ap.add_argument("--repeats", type=int, default=3)
args = ap.parse_args()
_, _, _, _, device = compute_init(autodetect_device_type())
sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

def bench(engine, tokens, bs):
    engine.generate_batch(tokens, num_samples=bs, max_tokens=16, temperature=1.0)  # warmup
    sync(); t0 = time.time()
    for _ in range(args.repeats):
        out, _ = engine.generate_batch(tokens, num_samples=bs, max_tokens=args.gen_tokens, temperature=1.0, top_k=50)
    sync(); dt = (time.time() - t0) / args.repeats
    gen = sum(len(o) - len(tokens) for o in out)
    return gen / dt, dt

rows = []
ar, tok, _ = load_model(args.source, device, phase="eval", model_tag=args.ar_tag)
prompt = tok.encode("The history of the Roman Empire", prepend=tok.get_bos_token_id())
for bs in map(int, args.batch_sizes.split(",")):
    tps, dt = bench(Engine(ar, tok), prompt, bs)
    rows.append(("AR (KV cache)", "-", bs, tps, dt))
del ar
mdm, tok, _ = load_model(args.source, device, phase="eval", model_tag=args.mdm_tag)
for steps in (args.block_size, args.block_size // 2, args.block_size // 4, args.block_size // 8, 1):
    eng = DiffusionEngine(mdm, tok, block_size=args.block_size, steps_per_block=steps)
    for bs in map(int, args.batch_sizes.split(",")):
        tps, dt = bench(eng, prompt, bs)
        rows.append((f"MDM block {args.block_size}", f"{steps}/block", bs, tps, dt))
print0(f"\n{'model':16s} {'steps':10s} {'batch':>5s} {'tok/s':>9s} {'s/gen':>7s}   ({args.gen_tokens} tokens after a {len(prompt)}-token prompt)")
for m, st, bs, tps, dt in rows:
    print0(f"{m:16s} {st:10s} {bs:5d} {tps:9.0f} {dt:7.2f}")
