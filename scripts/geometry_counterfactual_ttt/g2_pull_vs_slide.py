"""G-2: does the gap come from LOSING CONTACT, or from changing WHICH residues pair?

G-1 showed the frozen model's score drops sharply when entity B is moved. That alone does not
support the Geometry-Counterfactual premise, because pulling B away destroys the interface
outright -- a trivial thing to detect, and an 8 A translation is not a plausible complex. If
the model only reacts to "are they touching", then the counterfactual negative degenerates
into "monomer vs complex", a contrast BindingGYM already publishes (single-chain 0.3564 vs
complex 0.3970) and which needs no TTT.

Two matched arms at equal displacement |t|:

    pull  : t along  n = (centroid_B - centroid_A)/|.|   -> changes the separation
    slide : t within the plane orthogonal to n           -> preserves separation, repartners

A slide keeps the two centroids the same distance apart and keeps comparable contact area; it
changes which residue of A sits opposite which residue of B. So:

    slide gap ~ pull gap  => the model tracks pair-specific geometry  => premise holds
    slide gap << pull gap => the model tracks contact/separation only => premise fails

Also reported: n_contact, the number of cross-entity residue pairs within 5 A (the project's
interface definition), so "contact area was preserved" is measured rather than assumed.
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "structure_encoder_ttt"))
from bgmpnn import load_model, AssayContext                       # noqa: E402
sys.path.insert(0, HERE)
from g1_interface_sensitivity import entity_of_slots              # noqa: E402

BG = os.environ.get("BG_ROOT", "/ibex/user/guoj0f/share/BindingGYM")
REFS = os.path.join(HERE, "..", "structure_encoder_ttt", "refs")
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
CUT = 8.0   # CA-CA proxy for contact area; the project interface definition is
            # heavy-atom 5 A, which CA-CA 5 A badly under-counts (PSD95 gave n_contact0 = 2)


def n_contacts(X, selA, selB, cut=CUT):
    """Cross-entity CA pairs within `cut` -- a PROXY for contact area, not the project's
    interface definition (which is heavy-atom 5 A). CA-CA at 5 A under-counts badly: PSD95
    scored n_contact0 = 2. 8 A on CA is the usual coarse stand-in for heavy-atom 5 A, and
    side chains are invisible to ProteinMPNN anyway."""
    A = X[0, selA, 1, :]                       # CA
    B = X[0, selB, 1, :]
    d = torch.cdist(A[None], B[None])[0]
    return int((d < cut).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assays", nargs="*", default=None)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mags", type=float, nargs="*", default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--gpu_assert", default="A100")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="g2")
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

    out_dir = a.out or f"{ROOT}/ibex-records/geometry-counterfactural-encoder-TTT/data"
    os.makedirs(out_dir, exist_ok=True)
    rows, t0 = [], time.time()
    for dms in assays:
        r = idx[idx.DMS_id == dms].iloc[0]
        df = pd.read_csv(f"{BG}/input/Binding_substitutions_DMS/{r.DMS_filename}")
        df["chain_id"] = df["chain_id"].fillna("")
        pdb = f"{BG}/input/structures/{df['pdb_file'].values[0]}"
        ctx = AssayContext(model, pdb, df["chain_id"].values[0], a.M, dev)
        X0 = ctx.X.clone(); S_wt = ctx.S_wt[:1]
        with torch.no_grad():
            compat_wt = float(ctx.score(S_wt)[0])
        ent, real = entity_of_slots(ctx, pdb, ent_tab.loc[dms, "entity1"])
        valid = real & (ctx.mask[0].cpu().numpy() > 0)
        selA = np.flatnonzero((ent == 0) & valid)
        selB = np.flatnonzero((ent == 1) & valid)
        selB_all = np.flatnonzero(ent == 1)
        sd = float(ref14.loc[dms, "sd_mpnn"])
        cA = X0[0, selA, 1, :].mean(0)
        cB = X0[0, selB, 1, :].mean(0)
        n = (cB - cA); n = n / torch.linalg.norm(n)        # interface normal (centroid axis)
        n_np = n.cpu().numpy()
        c0 = n_contacts(X0, selA, selB)

        for mag in a.mags:
            for kind in ("pull", "slide"):
                rng = np.random.RandomState(a.seed)
                for rep in range(a.reps):
                    if kind == "pull":
                        d = n_np * mag                      # along the axis
                    else:
                        v = rng.randn(3)
                        v = v - np.dot(v, n_np) * n_np      # project out the axis component
                        d = v / np.linalg.norm(v) * mag     # orthogonal => separation preserved
                    Xp = X0.clone()
                    Xp[:, selB_all] = Xp[:, selB_all] + torch.tensor(
                        d, dtype=Xp.dtype, device=Xp.device)
                    ctx.X = Xp; ctx._build_encoder_cache()
                    with torch.no_grad():
                        c = float(ctx.score(S_wt)[0])
                    rows.append(dict(DMS_id=dms, kind=kind, mag=mag, rep=rep,
                                     gap=compat_wt - c, gap_over_sd=(compat_wt - c) / sd,
                                     n_contact0=c0, n_contact=n_contacts(Xp, selA, selB),
                                     sd_mpnn=sd))
        ctx.X = X0; ctx._build_encoder_cache()
        g = pd.DataFrame(rows); g = g[g.DMS_id == dms]
        p1 = g[(g.kind == "pull") & (g.mag == 1.0)]; s1 = g[(g.kind == "slide") & (g.mag == 1.0)]
        print(f"{dms:44s} 1A: pull {p1.gap_over_sd.mean():+6.2f} sd (contact "
              f"{c0}->{p1.n_contact.mean():.0f})   slide {s1.gap_over_sd.mean():+6.2f} sd "
              f"(contact {c0}->{s1.n_contact.mean():.0f})   [{time.time()-t0:.0f}s]", flush=True)
        pd.DataFrame(rows).to_csv(f"{out_dir}/{a.tag}_pull_vs_slide.csv", index=False)

    t = pd.DataFrame(rows)
    print("\n=== pull vs slide, mean over assays ===")
    print(f"{'mag':>5} {'pull sd':>9} {'slide sd':>9} {'slide/pull':>11} "
          f"{'pull contact%':>14} {'slide contact%':>15}")
    for mag, g in t.groupby("mag"):
        p = g[g.kind == "pull"]; s = g[g.kind == "slide"]
        cp = (p.n_contact / p.n_contact0).mean() * 100
        cs = (s.n_contact / s.n_contact0).mean() * 100
        ratio = s.gap_over_sd.mean() / p.gap_over_sd.mean() if p.gap_over_sd.mean() else np.nan
        print(f"{mag:5.1f} {p.gap_over_sd.mean():+9.3f} {s.gap_over_sd.mean():+9.3f} "
              f"{ratio:11.3f} {cp:13.1f}% {cs:14.1f}%")
    print("\nslide/pull ~ 1  => model tracks pair-specific geometry (premise holds)")
    print("slide/pull << 1 => model tracks contact/separation only (premise fails)")


if __name__ == "__main__":
    main()
