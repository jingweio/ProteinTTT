"""P2: the same two-group comparison, run on the DMS label and on the ProteinMPNN
prediction, so the two are directly comparable (identical estimators, identical
assay subset rule)."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "binding_sites"))
from bg_common import REC_DIR

PRED = os.path.normpath(os.path.join(REC_DIR, "..", "binding-sites-analysis-pred"))
COL, MIN_N = "iface_dist_5.0", 30
v = pd.read_parquet(f"{PRED}/data/variant_labels_with_mpnn.parquet")

def ovl(a, b, nbins=80):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if not np.isfinite(lo) or hi <= lo: return np.nan
    e = np.linspace(lo, hi, nbins + 1)
    pa = np.histogram(a, bins=e, density=True)[0]
    pb = np.histogram(b, bins=e, density=True)[0]
    return float(np.minimum(pa, pb).sum() * (e[1] - e[0]))

def two_group(a, b):
    """a = touches a binding site, b = does not."""
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = 2 * U / (len(a) * len(b)) - 1
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    tot = np.concatenate([a, b]).var()
    within = (len(a)*a.var() + len(b)*b.var()) / (len(a)+len(b))
    return dict(cliffs_delta=d, p_mwu=p, overlap_coef=ovl(a, b),
                mean_iface=a.mean(), mean_noniface=b.mean(),
                median_iface=np.median(a), median_noniface=np.median(b),
                sd_iface=a.std(ddof=1), sd_noniface=b.std(ddof=1),
                cohens_d=(a.mean() - b.mean()) / pooled, eta2=1 - within / tot,
                P_noniface_gt_iface=(1 - d) / 2)

rows = []
for dms, g in v.groupby("DMS_id"):
    m = g[COL].to_numpy()
    r = dict(DMS_id=dms, n_iface=int(m.sum()), n_noniface=int((~m).sum()))
    r["testable"] = r["n_iface"] >= MIN_N and r["n_noniface"] >= MIN_N
    for tag, col in [("dms", "DMS_score"), ("mpnn", "mpnn_score")]:
        a = g.loc[m, col].to_numpy(); b = g.loc[~m, col].to_numpy()
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if r["testable"]:
            r.update({f"{k}_{tag}": val for k, val in two_group(a, b).items()})
    r["rho_mpnn_vs_dms"] = stats.spearmanr(g.DMS_score, g.mpnn_score).statistic
    rows.append(r)

r = pd.DataFrame(rows)
for tag in ("dms", "mpnn"):                                     # BH-FDR per readout
    t = r["testable"].to_numpy(); p = r[f"p_mwu_{tag}"].to_numpy(dtype=float)
    q = np.full(len(r), np.nan); pv = p[t]; o = np.argsort(pv); mm = len(pv)
    adj = np.minimum.accumulate((pv[o] * mm / (np.arange(mm) + 1))[::-1])[::-1]
    qq = np.empty(mm); qq[o] = np.clip(adj, 0, 1); q[t] = qq
    r[f"q_BH_{tag}"] = q
r = r.sort_values("cliffs_delta_dms")
r.to_csv(f"{PRED}/data/stats_dms_vs_mpnn.csv", index=False)

pd.set_option("display.width", 300)
t = r[r.testable]
print(t[["DMS_id", "n_iface", "n_noniface", "cliffs_delta_dms", "cliffs_delta_mpnn",
         "overlap_coef_dms", "overlap_coef_mpnn", "eta2_dms", "eta2_mpnn",
         "q_BH_mpnn", "rho_mpnn_vs_dms"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\ntestable {len(t)}/25")
for tag in ("dms", "mpnn"):
    d = t[f"cliffs_delta_{tag}"]
    print(f"  {tag:5s} |delta| median {d.abs().median():.3f}  neg {int((d<0).sum())}/{len(t)}  "
          f"q<0.05 {int((t[f'q_BH_{tag}']<0.05).sum())}/{len(t)}  OVL median {t[f'overlap_coef_{tag}'].median():.3f}  "
          f"eta2 median {t[f'eta2_{tag}'].median():.3f}")
print(f"\n  |delta_mpnn| < |delta_dms| in {int((t.cliffs_delta_mpnn.abs() < t.cliffs_delta_dms.abs()).sum())}/{len(t)} assays")
print(f"  Spearman(delta_dms, delta_mpnn) over assays = {stats.spearmanr(t.cliffs_delta_dms, t.cliffs_delta_mpnn).statistic:.3f}"
      f"  (p={stats.spearmanr(t.cliffs_delta_dms, t.cliffs_delta_mpnn).pvalue:.3f})")
print(f"  median ratio |delta_mpnn|/|delta_dms| = {(t.cliffs_delta_mpnn.abs()/t.cliffs_delta_dms.abs()).median():.3f}")
