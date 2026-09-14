"""M-1: how big can the TTT batch actually be?

`bs_eff = clip(3e8/(L*48*256), 8, 128)` was written a priori and never checked against a
measurement, so two things are unknown: whether the 128 ceiling is anywhere near the card,
and whether the small batches the formula hands the long complexes (ACE2, L=931 -> 26) are
paying a real memory bill or an imaginary one. That matters because `L_div` is a WITHIN-BATCH
Pearson: a small batch is not just slower, it is a noisier estimate of the quantity being
optimised.

Runs real training steps (forward + backward + Adam) at increasing batch size and reports
torch.cuda.max_memory_allocated, stopping at the first OOM.
"""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bgmpnn import load_model, set_trainable
from e1_lambda_sweep import build, BG

GB = 1024 ** 3


def probe(ctx, S, bs, mode, lr, steps, dev):
    ps = set_trainable(ctx.model, mode)
    opt = torch.optim.Adam(ps, lr=lr)
    tgt = torch.randn(bs, device=dev)
    ctx.model.train()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        ix = np.random.randint(0, len(S), bs)
        m = int(np.random.randint(ctx.M))
        s = ctx.score(S[ix], m_idx=m)
        ss = (s - s.mean()) / (s.std() + 1e-8)
        loss = (ss * ss.flip(0)).mean() + 0.1 * ((s - tgt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    return (time.time() - t0) / steps, torch.cuda.max_memory_allocated() / GB, \
        torch.cuda.max_memory_reserved() / GB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assays", nargs="+", required=True)
    ap.add_argument("--bs", nargs="+", type=int, default=[26, 64, 128, 256, 512, 1024])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mode", default="decoder")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    dev = torch.device("cuda")
    assert "A100" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
    model = load_model(f"{BG}/training/cache/v_48_020.pt", dev)
    tot = torch.cuda.get_device_properties(0).total_memory / GB
    print(f"card {torch.cuda.get_device_name(0)}  total {tot:.1f} GiB  "
          f"free-at-start {(tot - torch.cuda.memory_reserved() / GB):.1f} GiB\n")
    for dms in a.assays:
        ctx, S, _ = build(dms, model, a.M, a.seed, dev)
        formula = max(8, min(128, int(3.0e8 / (ctx.L * 48 * 256))))
        print(f"=== {dms}  L={ctx.L}  n={len(S)}  formula bs={formula}")
        print(f"{'bs':>6s} {'s/step':>8s} {'peak alloc GiB':>15s} {'peak resv GiB':>14s}")
        for bs in a.bs:
            if bs > len(S):
                print(f"{bs:6d}  (> n, skipped)"); continue
            try:
                dt, al, rs = probe(ctx, S, bs, a.mode, a.lr, a.steps, dev)
                print(f"{bs:6d} {dt:8.3f} {al:15.2f} {rs:14.2f}", flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{bs:6d}  OOM", flush=True); break
        del ctx, S; torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
