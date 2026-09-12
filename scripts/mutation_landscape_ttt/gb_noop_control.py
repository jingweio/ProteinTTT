"""G-B: the no-op control -- does this module reproduce the official frozen scores?

Nothing downstream is interpretable unless the score optimised by the TTT is the score the
benchmark reports. This runs the cached-encoder path at steps = 0 and compares it
elementwise against the stored zero-shot CSVs produced by compute_fitness_multi_pdb.py.
A mismatch means the harness is wrong, not that the TTT failed.
"""
import argparse, os, random, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bgmpnn import load_model, AssayContext

BG = "/data/guoj0f/share/BindingGYM"
REF = "/data/guoj0f/BindingGYM-zero-shot-proteinMPNN/scores"


def score_assay(model, dms_id, dms_file, pdb_dir, M, seed, device, batch=64, ctx_cache=None):
    """Replays the official control flow: seed once, group by (POI, chain_id), one randn
    per POI reused across that POI's variants, original row order restored at the end."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    df = pd.read_csv(dms_file)
    df["chain_id"] = df["chain_id"].fillna("")
    out = pd.Series(index=df.index, dtype=float)
    randn_by_poi = {}
    for (poi, chain_ids), g in df.groupby(["POI", "chain_id"]):
        pdb_file = os.path.join(pdb_dir, g["pdb_file"].values[0])
        if poi not in randn_by_poi:
            ctx = AssayContext(model, pdb_file, chain_ids, M, device)   # draws randn itself
            randn_by_poi[poi] = ctx.randn
        else:
            ctx = AssayContext(model, pdb_file, chain_ids, M, device, randn=randn_by_poi[poi])
        S = torch.stack([ctx.seq_to_S(eval(s)) for s in g["mutated_sequence"]])
        vals = []
        with torch.no_grad():
            for i in range(0, len(S), batch):
                vals.append(ctx.score(S[i:i + batch]).float().cpu())
        out.loc[g.index] = torch.cat(vals).numpy()
        if ctx_cache is not None:
            ctx_cache[(poi, chain_ids)] = ctx
    return df, out.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assays", nargs="+", required=True)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(0))
    assert "A100" in torch.cuda.get_device_name(0)
    model = load_model(f"{BG}/training/cache/v_48_020.pt", dev)
    nparam = sum(p.numel() for p in model.parameters())
    ndec = sum(p.numel() for p in model.decoder_layers.parameters()) + \
        sum(p.numel() for p in model.W_out.parameters())
    print(f"params total {nparam:,}   decoder+W_out {ndec:,} ({100*ndec/nparam:.1f}%)")

    idx = pd.read_csv(f"{BG}/input/BindingGYM.csv")
    rows = []
    for dms_id in a.assays:
        r = idx[idx.DMS_id == dms_id].iloc[0]
        df, got = score_assay(model, dms_id, f"{BG}/input/Binding_substitutions_DMS/{r.DMS_filename}",
                              f"{BG}/input/structures", a.M, a.seed, dev, a.batch)
        ref = pd.read_csv(f"{REF}/seed{a.seed}_M{a.M}/{dms_id}.csv")["global_score"].to_numpy()
        assert len(ref) == len(got), f"{dms_id}: {len(got)} vs {len(ref)} rows"
        d = np.abs(ref - got)
        rel = d / np.maximum(np.abs(ref), 1e-9)
        rows.append(dict(DMS_id=dms_id, n=len(got), max_abs=d.max(), max_rel=rel.max(),
                         corr=float(np.corrcoef(ref, got)[0, 1]), scale=float(np.abs(ref).mean())))
        print(f"{dms_id:42s} n={len(got):>6d}  max|d|={d.max():.3e}  "
              f"max rel={rel.max():.3e}  corr={rows[-1]['corr']:.8f}")
    t = pd.DataFrame(rows)
    if a.out: t.to_csv(a.out, index=False)
    # float32 accumulation over a few hundred residues is the only expected difference
    ok = bool((t.max_rel < 1e-5).all())
    print(f"\nG-B {'PASS' if ok else 'FAIL'}: max relative deviation {t.max_rel.max():.3e} "
          f"(tolerance 1e-5, float32 accumulation)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
