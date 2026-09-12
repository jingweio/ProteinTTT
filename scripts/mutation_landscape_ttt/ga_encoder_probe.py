"""G-A: is the interface information present in the FROZEN encoder embeddings?

decoder-only TTT is only possible if the decoder's input already distinguishes residues by
their distance to the partner entity. The decoder reads the encoder's node embeddings h_V
and the edge embeddings, all of which are frozen here, so if h_V does not carry that
information no amount of decoder tuning can express the prior. This probes h_V directly
with a linear model -- deliberately linear, because anything the decoder can do with a
representation, a linear read-out should at least partly see.

Two read-outs, since the probe record showed the effect is a gradient and not a threshold:
  classification  ->  is this residue within 5 A of the other entity (AUC)
  regression      ->  how far is it (Spearman against the true distance)

Reported both within-assay (5-fold) and leave-one-assay-out. Within-assay says "the
information is there for this complex", which is what a per-assay TTT needs; LOAO says
whether it is encoded in a shared, transferable way.
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bgmpnn import load_model, AssayContext

BG = "/data/guoj0f/share/BindingGYM"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = f"{ROOT}/workstation-records/mutation-landscape-TTT/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assays", nargs="*", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda:0"); assert "A100" in torch.cuda.get_device_name(0)
    model = load_model(f"{BG}/training/cache/v_48_020.pt", dev)

    ent = pd.read_csv(f"{ROOT}/local-records/binding-sites-analysis/data/entity_partition.csv").set_index("DMS_id")
    idx = pd.read_csv(f"{BG}/input/BindingGYM.csv")
    assays = a.assays or list(pd.read_csv(f"{OUT}/g1e_canonical14.csv").DMS_id)

    X_all, d_all, a_all = [], [], []
    for dms in assays:
        r = idx[idx.DMS_id == dms].iloc[0]
        df = pd.read_csv(f"{BG}/input/Binding_substitutions_DMS/{r.DMS_filename}")
        df["chain_id"] = df["chain_id"].fillna("")
        ctx = AssayContext(model, f"{BG}/input/structures/{r.pdb_file}", df["chain_id"].values[0], 1, dev)
        h = ctx.h_V[0].float().cpu().numpy()                       # (L, 128), frozen

        # Distances are computed in the packing space h_V lives in, so no WT<->PDB alignment
        # is involved -- that mapping is a known source of silent off-by-one errors, and the
        # question here ("does h_V encode proximity to the partner") does not need it. CA
        # coordinates, and the continuous distance rather than a cutoff, since the probe
        # record showed the effect is a gradient.
        ca = ctx.X[0, :, 1].float().cpu().numpy()                  # (L, 3)
        e1 = set(ent.loc[dms, "entity1"]); order = ctx.designed + [c for c in ctx.all_chains
                                                                  if c not in ctx.designed]
        lens = ctx.chain_lengths
        lab = np.concatenate([np.full(lens[c], 0 if c in e1 else 1) for c in order])
        assert len(lab) == len(h), f"{dms}: packed {len(lab)} vs h_V {len(h)}"
        A_, B_ = ca[lab == 0], ca[lab == 1]
        if len(A_) == 0 or len(B_) == 0:
            print(f"{dms}: one entity is empty, skipped"); continue
        dd = np.empty(len(ca))
        dd[lab == 0] = np.linalg.norm(A_[:, None, :] - B_[None, :, :], axis=2).min(1)
        dd[lab == 1] = np.linalg.norm(B_[:, None, :] - A_[None, :, :], axis=2).min(1)
        X_all.append(h); d_all.append(dd); a_all.append(np.full(len(h), dms))
        print(f"{dms:42s} L={len(h):>4d}  entity sizes {len(A_)}/{len(B_)}  "
              f"median CA-CA d {np.median(dd):.1f} A")

    X = np.vstack(X_all); d = np.concatenate(d_all); A = np.concatenate(a_all)
    y = (d <= 8.0).astype(int)      # CA-CA 8 A ~ the heavy-atom 5 A contact used elsewhere
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)

    rows = []
    for dms in sorted(set(A)):
        m = A == dms
        if len(set(y[m])) < 2 or m.sum() < 40:
            rows.append(dict(DMS_id=dms, n=int(m.sum()), auc_in=np.nan, rho_in=np.nan,
                             auc_loao=np.nan, rho_loao=np.nan)); continue
        # within-assay 5-fold
        pa = np.zeros(m.sum()); pr = np.zeros(m.sum())
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs[m], y[m]):
            pa[te] = LogisticRegression(max_iter=2000, C=1.0).fit(Xs[m][tr], y[m][tr]).predict_proba(Xs[m][te])[:, 1]
            pr[te] = Ridge(alpha=1.0).fit(Xs[m][tr], d[m][tr]).predict(Xs[m][te])
        # leave-one-assay-out
        tr = ~m
        qa = LogisticRegression(max_iter=2000, C=1.0).fit(Xs[tr], y[tr]).predict_proba(Xs[m])[:, 1]
        qr = Ridge(alpha=1.0).fit(Xs[tr], d[tr]).predict(Xs[m])
        rows.append(dict(DMS_id=dms, n=int(m.sum()),
                         auc_in=roc_auc_score(y[m], pa), rho_in=stats.spearmanr(pr, d[m]).statistic,
                         auc_loao=roc_auc_score(y[m], qa), rho_loao=stats.spearmanr(qr, d[m]).statistic))
    t = pd.DataFrame(rows); t.to_csv(f"{OUT}/ga_encoder_probe.csv", index=False)
    print("\n=== linear probe on the FROZEN encoder node embeddings ===")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nmedian within-assay  AUC {t.auc_in.median():.3f}   rho(d) {t.rho_in.median():+.3f}")
    print(f"median LOAO          AUC {t.auc_loao.median():.3f}   rho(d) {t.rho_loao.median():+.3f}")
    print(f"\nG-A: {'the information is present' if t.auc_in.median() > 0.7 else 'WEAK -- decoder-only may not be expressible'}")


if __name__ == "__main__":
    main()
