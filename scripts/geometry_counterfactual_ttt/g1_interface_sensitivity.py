"""G-1: is ProteinMPNN's score sensitive to interface geometry at all?

The gate for Geometry-Counterfactual TTT. The whole scheme rests on the WT complex being
more "compatible" with the WT sequence than an interface-disrupted counterfactual is:

    Compat(X_AB, S_WT)  >  Compat(X_hat_AB, S_WT)

If the frozen model cannot already tell them apart at all, there is no signal for a
contrastive objective to sharpen, and two independent lines of evidence in this project say
ProteinMPNN is largely partner-blind on BindingGYM. So measure the gap BEFORE building
anything: no training, WT sequence only.

Perturbation is a RIGID-BODY move of entity B: rotation about its own centroid and/or
translation. This leaves every intra-chain distance and dihedral untouched, so the only
thing that changes is the A-B relative pose -- the cleanest possible interface-only
counterfactual. Random coordinate noise would instead introduce clashes and broken geometry,
which the model could separate on trivial grounds.

Two readings:
  * gap = Compat(WT) - Compat(perturbed), in nats
  * gap / sd_mpnn -- sd_mpnn is the spread of this assay's own variant scores, so this says
    how large the interface disruption is IN UNITS OF the mutational variation the benchmark
    actually ranks. A gap far below 1 means disrupting the entire interface matters less than
    a typical single mutation.

Sanity arm: the same rigid move applied to the WHOLE complex. ProteinMPNN reads only
distances, so that is exactly SE(3)-invariant and its gap must be 0 to numerical precision.
A non-zero value there means the perturbation code is wrong, not that the model is sensitive.
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "structure_encoder_ttt"))
from bgmpnn import load_model, AssayContext                       # noqa: E402
from bgpdb import pdb_chain_slots                                 # noqa: E402

BG = os.environ.get("BG_ROOT", "/ibex/user/guoj0f/share/BindingGYM")
REFS = os.path.join(HERE, "..", "structure_encoder_ttt", "refs")
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))


def entity_of_slots(ctx, pdb_path, entity1):
    """0 = entity1, 1 = entity2, in tied_featurize packing order (designed chains first)."""
    order = ctx.designed + [c for c in ctx.all_chains if c not in ctx.designed]
    e1 = set(entity1)
    ent, real = [], []
    for c in order:
        s = pdb_chain_slots(pdb_path, c)
        assert len(s) == ctx.chain_lengths[c], f"chain {c}: {len(s)} vs {ctx.chain_lengths[c]}"
        ent += [0 if c in e1 else 1] * len(s)
        real += [x is not None for x in s]
    return np.asarray(ent), np.asarray(real)


def rotation_matrix(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def rigid_perturb(X, sel, valid_sel, t_ang, theta_deg, rng):
    """Rotate about the selected atoms' centroid, then translate. X: (M, L, 4, 3)."""
    Xp = X.clone()
    sub = Xp[:, sel]                                   # (M, n, 4, 3)
    # centroid from REAL residues only -- gap-padded slots sit at the origin and would drag it
    ref = sub[0][valid_sel].reshape(-1, 3)
    cen = ref.mean(0)
    R = torch.tensor(rotation_matrix(rng.randn(3), np.deg2rad(theta_deg)),
                     dtype=sub.dtype, device=sub.device)
    d = rng.randn(3); d = d / np.linalg.norm(d) * t_ang
    d = torch.tensor(d, dtype=sub.dtype, device=sub.device)
    Xp[:, sel] = (sub - cen) @ R.T + cen + d
    return Xp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assays", nargs="*", default=None)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gpu_assert", default="A100")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="g1")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0); print("GPU:", name, flush=True)
    assert a.gpu_assert in name, f"expected {a.gpu_assert}, got {name}"
    model = load_model(a.ckpt or f"{BG}/training/cache/v_48_020.pt", dev)
    ent_tab = pd.read_csv(f"{REFS}/entity_partition.csv").set_index("DMS_id")
    ref14 = pd.read_csv(f"{REFS}/g1e_canonical14.csv").set_index("DMS_id")
    idx = pd.read_csv(f"{BG}/input/BindingGYM.csv")
    assays = a.assays or list(ref14.index)
    print(f"{len(assays)} assays", flush=True)

    grid = ([("translate", t, 0.0) for t in (0.5, 1.0, 2.0, 4.0, 8.0)] +
            [("rotate", 0.0, th) for th in (2.0, 5.0, 10.0)] +
            [("whole_complex_sanity", 4.0, 5.0)])

    out_dir = a.out or f"{ROOT}/ibex-records/geometry-counterfactural-encoder-TTT/data"
    os.makedirs(out_dir, exist_ok=True)
    rows, t0 = [], time.time()
    for dms in assays:
        r = idx[idx.DMS_id == dms].iloc[0]
        df = pd.read_csv(f"{BG}/input/Binding_substitutions_DMS/{r.DMS_filename}")
        df["chain_id"] = df["chain_id"].fillna("")
        pdb = f"{BG}/input/structures/{df['pdb_file'].values[0]}"
        ctx = AssayContext(model, pdb, df["chain_id"].values[0], a.M, dev)
        X0 = ctx.X.clone()
        S_wt = ctx.S_wt[:1]
        with torch.no_grad():
            compat_wt = float(ctx.score(S_wt)[0])
        ent, real = entity_of_slots(ctx, pdb, ent_tab.loc[dms, "entity1"])
        valid = real & (ctx.mask[0].cpu().numpy() > 0)
        sd = float(ref14.loc[dms, "sd_mpnn"]) if dms in ref14.index else np.nan
        sel2 = np.flatnonzero(ent == 1)
        for kind, t_ang, th in grid:
            rng = np.random.RandomState(a.seed)
            for rep in range(a.reps):
                if kind == "whole_complex_sanity":
                    sel = np.arange(len(ent)); vs = valid
                else:
                    sel = sel2; vs = valid[sel2]
                ctx.X = rigid_perturb(X0, sel, vs, t_ang, th, rng)
                ctx._build_encoder_cache()
                with torch.no_grad():
                    c = float(ctx.score(S_wt)[0])
                rows.append(dict(DMS_id=dms, kind=kind, t_ang=t_ang, theta_deg=th, rep=rep,
                                 compat_wt=compat_wt, compat_pert=c, gap=compat_wt - c,
                                 gap_over_sd=(compat_wt - c) / sd if sd == sd else np.nan,
                                 L=int(len(ent)), n_entity2=int(len(sel2)), sd_mpnn=sd))
        ctx.X = X0; ctx._build_encoder_cache()
        g = pd.DataFrame(rows); g = g[g.DMS_id == dms]
        san = g[g.kind == "whole_complex_sanity"].gap.abs().max()
        tr8 = g[(g.kind == "translate") & (g.t_ang == 8.0)].gap.mean()
        print(f"{dms:44s} sanity|gap|max={san:.3e}  translate8A gap={tr8:+.3f} nats "
              f"({tr8/sd:+.3f} sd)  [{time.time()-t0:.0f}s]", flush=True)
        pd.DataFrame(rows).to_csv(f"{out_dir}/{a.tag}_interface_sensitivity.csv", index=False)

    t = pd.DataFrame(rows)
    san = t[t.kind == "whole_complex_sanity"].gap.abs().max()
    print(f"\n=== SANITY: whole-complex rigid move, max|gap| = {san:.3e} "
          f"(must be ~0; ProteinMPNN reads only distances) ===")
    print("\nmean over assays, gap in units of that assay's own variant-score sd:")
    for kind, col in (("translate", "t_ang"), ("rotate", "theta_deg")):
        sub = t[t.kind == kind]
        if len(sub):
            print(f"  {kind}:")
            for v, gg in sub.groupby(col):
                print(f"    {v:5.1f}  gap {gg.gap.mean():+8.3f} nats   "
                      f"{gg.gap_over_sd.mean():+6.3f} sd   (positive in "
                      f"{int((gg.groupby('DMS_id').gap.mean() > 0).sum())}/{gg.DMS_id.nunique()} assays)")


if __name__ == "__main__":
    main()
