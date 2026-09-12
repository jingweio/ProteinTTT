"""D-1: did the TTT learn anything a single scalar could not have produced?

With an L2 anchor, the optimum over unconstrained score vectors of
    pearson(w, s) + lambda * ||s - s_frozen||^2
is a shift along w, i.e. s = s_frozen - c*w -- which is precisely the five-line rescoring
rule. So the only way decoder-only TTT can beat that rule is by being UNABLE to express an
arbitrary score vector, and therefore approximating the direction in a structured,
per-position way instead.

That makes the diagnosis concrete: regress the learned shift on the feature.
    R^2 near 1  -> the TTT reimplemented the scalar, with a GPU. No new information.
    clear residual, and residual correlated with the gain -> it learned something the
                                                             scalar cannot express.
"""
import os, sys, glob
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = f"{ROOT}/workstation-records/mutation-landscape-TTT/data"
TAU = 5.0


def main(tag="e1", suf="", agg="max"):
    sw = pd.read_csv(f"{OUT}/{tag}_lambda_sweep{suf}.csv")
    pr = np.load(f"{OUT}/{tag}_preds{suf}.npz")
    lab = pd.read_parquet(f"{ROOT}/local-records/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet")
    sdt = pd.read_parquet(f"{ROOT}/local-records/binding-sites-analysis/data/variant_site_dists.parquet")

    rows = []
    for _, r in sw.iterrows():
        key = f"{r.DMS_id}|{r.lam}"
        if key not in pr: continue
        s_new = pr[key]
        g = lab[lab.DMS_id == r.DMS_id]
        s_old = g.mpnn_score.to_numpy(float)      # the official frozen score, seed 1 / M=5
        y = g.DMS_score.to_numpy(float)
        sd = sdt[sdt.DMS_id == r.DMS_id]
        vi = sd.vi.to_numpy()
        f_ = np.exp(-sd.dist.to_numpy(float) / TAU)
        st = np.r_[0, np.flatnonzero(np.diff(vi)) + 1]
        w = {"max": np.fmax.reduceat, "sum": np.add.reduceat}[agg](f_, st)
        if agg == "mean": w = w / np.diff(np.r_[st, len(f_)])
        ds = s_new - s_old
        # R^2 of the learned shift explained by a single scalar multiple of the feature
        c = np.polyfit(w, ds, 1)
        fit = np.polyval(c, w)
        ss_res = ((ds - fit) ** 2).sum(); ss_tot = ((ds - ds.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot or 1.0)
        resid = ds - fit
        # what the same scalar shift would have scored on its own
        rho_scalar = stats.spearmanr(s_old + fit, y).statistic
        rows.append(dict(DMS_id=r.DMS_id, lam=r.lam, rho=r.rho, rho_base=r.rho_base,
                         gain=r.gain, r2_scalar=r2, rho_scalar_only=rho_scalar,
                         gain_beyond_scalar=r.rho - rho_scalar,
                         resid_rms_frac=float(np.sqrt((resid ** 2).mean()) /
                                              (np.sqrt((ds ** 2).mean()) or 1.0)),
                         rho_resid_y=stats.spearmanr(resid, y).statistic))
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/d1_diagnostic{suf}.csv", index=False)
    pd.set_option("display.width", 220)
    print("=== D-1: is the learned shift just c*w? ===")
    print(t.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    b = t.loc[t.groupby("DMS_id").rho.idxmax()]
    print("\nat each assay's best lambda:")
    print(b[["DMS_id", "lam", "rho", "rho_scalar_only", "gain_beyond_scalar", "r2_scalar",
             "resid_rms_frac"]].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nmedian R^2 explained by a single scalar: {b.r2_scalar.median():.4f}")
    print(f"median gain of the TTT OVER its own scalar shift: {b.gain_beyond_scalar.median():+.4f}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["e1"]))
