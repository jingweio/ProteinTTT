"""S7: extend the two-group comparison with means and two overlap measures."""
import numpy as np, pandas as pd
from scipy import stats
D = "/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-analysis/data"
v = pd.read_parquet(f"{D}/variant_labels.parquet")
COL = "iface_dist_5.0"

def ovl(a, b, nbins=80):
    """Overlap coefficient: integral of min(f_a, f_b) on shared bins. 1 = identical, 0 = disjoint."""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if not np.isfinite(lo) or hi <= lo: return np.nan
    e = np.linspace(lo, hi, nbins + 1)
    pa = np.histogram(a, bins=e, density=True)[0]
    pb = np.histogram(b, bins=e, density=True)[0]
    return float(np.minimum(pa, pb).sum() * (e[1] - e[0]))

rows = []
for dms, g in v.groupby("DMS_id"):
    a = g.loc[g[COL], "DMS_score"].to_numpy(); b = g.loc[~g[COL], "DMS_score"].to_numpy()
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    r = dict(DMS_id=dms, n_iface=len(a), n_noniface=len(b),
             mean_iface=a.mean() if len(a) else np.nan, mean_noniface=b.mean() if len(b) else np.nan,
             sd_iface=a.std(ddof=1) if len(a) > 1 else np.nan,
             sd_noniface=b.std(ddof=1) if len(b) > 1 else np.nan,
             median_iface=np.median(a) if len(a) else np.nan,
             median_noniface=np.median(b) if len(b) else np.nan)
    if len(a) >= 30 and len(b) >= 30:
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        d = 2 * U / (len(a) * len(b)) - 1
        _, pt = stats.ttest_ind(a, b, equal_var=False)
        r.update(cliffs_delta=d, p_mwu=p, p_welch=pt,
                 P_noniface_gt_iface=(1 - d) / 2, overlap_coef=ovl(a, b),
                 mean_diff=r["mean_iface"] - r["mean_noniface"],
                 cohens_d=(r["mean_iface"] - r["mean_noniface"]) /
                          np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2)),
                 testable=True)
    else:
        r.update(testable=False)
    rows.append(r)
r = pd.DataFrame(rows).sort_values("cliffs_delta")
t = r["testable"].to_numpy(); p = r["p_mwu"].to_numpy(dtype=float); q = np.full(len(r), np.nan)
pv = p[t]; o = np.argsort(pv); m = len(pv)
adj = np.minimum.accumulate((pv[o] * m / (np.arange(m) + 1))[::-1])[::-1]
qq = np.empty(m); qq[o] = np.clip(adj, 0, 1); q[t] = qq
r["q_BH"] = q
r.to_csv(f"{D}/stats_iface_vs_noniface_extended.csv", index=False)
pd.set_option("display.width", 320)
print(r[r.testable][["DMS_id","n_iface","n_noniface","mean_iface","mean_noniface","mean_diff","sd_iface","sd_noniface",
                     "cohens_d","cliffs_delta","P_noniface_gt_iface","overlap_coef","q_BH"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
tt = r[r.testable]
print(f"\noverlap coefficient: median {tt.overlap_coef.median():.3f}  range [{tt.overlap_coef.min():.3f}, {tt.overlap_coef.max():.3f}]")
print(f"P(non-iface > iface): median {tt.P_noniface_gt_iface.median():.3f}  range [{tt.P_noniface_gt_iface.min():.3f}, {tt.P_noniface_gt_iface.max():.3f}]")
print(f"|Cohen's d|: median {tt.cohens_d.abs().median():.3f}  max {tt.cohens_d.abs().max():.3f}")
