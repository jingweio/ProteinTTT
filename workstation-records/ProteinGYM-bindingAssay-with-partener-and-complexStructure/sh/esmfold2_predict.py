#!/usr/bin/env python
"""ESMFold2 batch prediction for ProteinGym binding assays (monomer or complex).

Follows the shared SOP (Sources/datasets/protein-structure-prediction/
esmfold2-structure-prediction-sop.md): fp32 trunk + bf16 ESMC, fused kernels,
complex sampling 10 loops / 68 steps (SOP §9.4), atomic write + skip-if-exists.

Writes **PDB** (not mmCIF) so ProteinMPNN can consume it directly.
Only writes under --out-dir / --log-dir.
"""
import argparse, csv, os, time

csv.field_size_limit(1 << 24)
op = lambda d, i: os.path.join(d, f"{i}.pdb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--model", default="biohub/ESMFold2-Fast")
    ap.add_argument("--num-loops", type=int, default=10)          # SOP §9.4 complex default
    ap.add_argument("--num-sampling-steps", type=int, default=68)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--kernel-backend", choices=["fused", "none"], default="fused")
    ap.add_argument("--chunk-size", type=int, default=0, help=">0 caps trunk memory (SOP §4.2)")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/data/guoj0f/share/hf_cache"))
    ap.add_argument("--ccd-path", default=None)
    a = ap.parse_args()

    os.environ["HF_HOME"] = a.hf_home
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["ESMCFOLD_CCD_PATH"] = a.ccd_path or os.path.join(a.hf_home, "ccd.pkl")

    import torch
    from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    rows = sorted(csv.DictReader(open(a.manifest, newline="")), key=lambda r: int(r["seq_len"]))
    os.makedirs(a.out_dir, exist_ok=True); os.makedirs(a.log_dir, exist_ok=True)
    todo = [r for r in rows if not (os.path.exists(op(a.out_dir, r["id"]))
                                    and os.path.getsize(op(a.out_dir, r["id"])) > 0)]
    print(f"manifest={len(rows)} todo={len(todo)} (skip {len(rows)-len(todo)} done)", flush=True)
    if not todo:
        return 0

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    assert "A100" in torch.cuda.get_device_name(0)
    t0 = time.time()
    model = ESMFold2Model.from_pretrained(a.model, dtype=torch.float32).cuda().eval()
    model.set_kernel_backend(None if a.kernel_backend == "none" else a.kernel_backend)
    if a.chunk_size > 0:
        model.set_chunk_size(a.chunk_size)
    builder = ESMFold2InputBuilder()
    print(f"loaded {a.model} in {time.time()-t0:.1f}s  kernel={a.kernel_backend} "
          f"chunk={a.chunk_size or 'off'}  mem={torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

    tag = os.path.basename(a.manifest).replace(".csv", "")
    tsv = open(os.path.join(a.log_dir, f"{tag}_metrics.tsv"), "a")
    fails = open(os.path.join(a.log_dir, f"{tag}_failures.tsv"), "a")
    if tsv.tell() == 0:
        tsv.write("id\tn_chains\ttotal_len\tmean_plddt\tptm\tiptm\tsec\tpeak_gb\n")

    n_ok = n_fail = 0
    for k, r in enumerate(todo):
        _id, L = r["id"], int(r["seq_len"])
        try:
            t1 = time.time(); torch.cuda.reset_peak_memory_stats()
            cids, cseqs = r["chains"].split(":"), r["seqs"].split(":")
            spi = StructurePredictionInput(sequences=[
                ProteinInput(id=c, sequence=s) for c, s in zip(cids, cseqs)])
            res = builder.fold(model, spi, num_loops=a.num_loops,
                               num_sampling_steps=a.num_sampling_steps,
                               num_diffusion_samples=1, seed=a.seed, complex_id=_id)
            tmp = op(a.out_dir, _id) + ".tmp"
            # fold() returns MolecularComplex (no PDB writer); ProteinComplex has one.
            with open(tmp, "w") as f:
                f.write(res.complex.to_protein_complex().to_pdb_string())
            os.replace(tmp, op(a.out_dir, _id))
            pk = torch.cuda.max_memory_allocated() / 1e9
            tsv.write(f"{_id}\t{len(cids)}\t{L}\t{float(res.plddt.mean()):.4f}\t"
                      f"{float(res.ptm) if res.ptm is not None else float('nan'):.4f}\t"
                      f"{float(res.iptm) if getattr(res,'iptm',None) is not None else float('nan'):.4f}\t"
                      f"{time.time()-t1:.2f}\t{pk:.2f}\n"); tsv.flush()
            print(f"  [{k+1}/{len(todo)}] {_id} L={L} {time.time()-t1:.1f}s peak={pk:.1f}GB", flush=True)
            n_ok += 1
        except Exception as e:
            fails.write(f"{_id}\t{L}\t{type(e).__name__}: {str(e)[:300]}\n"); fails.flush()
            print(f"  [{k+1}/{len(todo)}] {_id} L={L} FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
            torch.cuda.empty_cache(); n_fail += 1
    tsv.close(); fails.close()
    print(f"DONE ok={n_ok} fail={n_fail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
