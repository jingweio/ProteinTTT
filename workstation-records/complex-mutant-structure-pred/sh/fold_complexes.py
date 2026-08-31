#!/usr/bin/env python
"""ESMFold2-Fast 折 BindingGYM complex（WT × 5 seeds + mutant × 1 seed）。

设计（SOP §6.2 的四条必保留）：原子写、skip-if-exists 续跑、逐条 try/except、确定性顺序。
输出路径遵循 SOP §2.4。
"""
import argparse, csv, gzip, os, sys, time, traceback
csv.field_size_limit(sys.maxsize)

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', required=True)
ap.add_argument('--out-root', required=True)
ap.add_argument('--log-dir', required=True)
ap.add_argument('--num-loops', type=int, default=10)
ap.add_argument('--num-sampling-steps', type=int, default=68)
ap.add_argument('--num-diffusion-samples', type=int, default=1)
ap.add_argument('--kernel-backend', default='fused')
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()

import torch
gpu = torch.cuda.get_device_name(0)
print(f"GPU: {gpu}", flush=True)
assert 'A100' in gpu, gpu
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput

DIR = {'wt': 'BindingGYM-esmfold2-fast-predicted-wt-complex-structure',
       'mutant': 'BindingGYM-esmfold2-fast-predicted-mutant-complex-structure'}
for d in DIR.values():
    os.makedirs(os.path.join(a.out_root, d), exist_ok=True)
os.makedirs(a.log_dir, exist_ok=True)

_open = gzip.open if a.manifest.endswith('.gz') else open
rows = list(csv.DictReader(_open(a.manifest, 'rt')))
# WT 全部在前；mutant 按 assay 的单条成本升序（便宜的先跑完，随时终止都有完整 assay）
cost = {}
for r in rows:
    cost.setdefault(r['assay'], int(r['total_len']))
rows.sort(key=lambda r: (r['kind'] != 'wt', cost[r['assay']], r['id']))
if a.limit:
    rows = rows[:a.limit]
print(f"manifest {len(rows)} 条 | loops={a.num_loops} steps={a.num_sampling_steps} "
      f"samples={a.num_diffusion_samples} kernel={a.kernel_backend}", flush=True)

t0 = time.time()
model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
if a.kernel_backend and a.kernel_backend != 'none':
    model.set_kernel_backend(a.kernel_backend)
builder = ESMFold2InputBuilder()
print(f"model+ccd loaded in {time.time()-t0:.1f}s", flush=True)

mpath = os.path.join(a.log_dir, 'metrics.tsv')
fpath = os.path.join(a.log_dir, 'failures.tsv')
new = not os.path.exists(mpath)
mf = open(mpath, 'a', buffering=1)
if new:
    mf.write("id\tassay\tkind\tk\tseed\ttotal_len\tptm\tiptm\tplddt_mean\tseconds\n")
ff = open(fpath, 'a', buffering=1)

done = skipped = failed = 0
wall = time.time()
for n, r in enumerate(rows, 1):
    out = os.path.join(a.out_root, DIR[r['kind']], r['id'] + '.cif.gz')
    if os.path.exists(out) and os.path.getsize(out) > 0:
        skipped += 1
        continue
    try:
        chains = r['chains'].split(':')
        seqs = r['seqs'].split(':')
        spi = StructurePredictionInput(
            sequences=[ProteinInput(id=c, sequence=s) for c, s in zip(chains, seqs)])
        t = time.time()
        o = builder.fold(model, spi, num_loops=a.num_loops,
                         num_sampling_steps=a.num_sampling_steps,
                         num_diffusion_samples=a.num_diffusion_samples,
                         seed=int(r['seed']))
        dt = time.time() - t
        res = o[0] if isinstance(o, list) else o
        tmp = out + '.tmp'
        with gzip.open(tmp, 'wt') as fh:
            fh.write(res.complex.to_mmcif())
        os.replace(tmp, out)                                    # 原子写
        pl = float(res.plddt.mean()) if getattr(res, 'plddt', None) is not None else -1
        mf.write(f"{r['id']}\t{r['assay']}\t{r['kind']}\t{r['k']}\t{r['seed']}\t{r['total_len']}\t"
                 f"{float(res.ptm):.4f}\t{float(res.iptm):.4f}\t{pl:.4f}\t{dt:.2f}\n")
        done += 1
    except Exception as e:
        failed += 1
        ff.write(f"{r['id']}\t{type(e).__name__}\t{str(e)[:300].replace(chr(9),' ')}\n")
        traceback.print_exc(file=sys.stdout)
        torch.cuda.empty_cache()
    if n % 100 == 0 or n == len(rows):
        el = time.time() - wall
        rate = done / el if done else 0
        left = (len(rows) - n) / rate / 3600 if rate else 0
        print(f"[{n}/{len(rows)}] done={done} skip={skipped} fail={failed} "
              f"elapsed={el/3600:.2f}h rate={rate*3600:.0f}/h eta={left:.2f}h", flush=True)
mf.close(); ff.close()
print(f"DONE done={done} skipped={skipped} failed={failed} elapsed={(time.time()-wall)/3600:.2f}h", flush=True)
