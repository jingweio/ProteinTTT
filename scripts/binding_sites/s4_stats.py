"""S4: does DMS_score differ between binding-site-touching and non-touching variants?"""
import numpy as np, pandas as pd
from scipy import stats

OUT = "/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/analysis_out"
v = pd.read_parquet(f"{OUT}/variant_labels.parquet")
COL = "iface_dist_5.0"
MIN_N = 30

def cliffs(a, b):
    """delta = P(a>b) - P(a<b), computed from the Mann-Whitney U."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0: return np.nan, np.nan, np.nan
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 2 * U / (n1 * n2) - 1, p, U

rows = []
for dms, g in v.groupby("DMS_id"):
    a = g.loc[g[COL], "DMS_score"].to_numpy()      # touches a binding site
    b = g.loc[~g[COL], "DMS_score"].to_numpy()     # does not
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    d, p, _ = cliffs(a, b)
    # depth-stratified: same comparison inside each n_mut stratum, weight by harmonic-ish n
    ds, ws = [], []
    for k, gg in g.groupby("n_mut"):
        aa = gg.loc[gg[COL], "DMS_score"].to_numpy(); bb = gg.loc[~gg[COL], "DMS_score"].to_numpy()
        aa = aa[np.isfinite(aa)]; bb = bb[np.isfinite(bb)]
        if len(aa) >= 10 and len(bb) >= 10:
            dd, _, _ = cliffs(aa, bb)
            ds.append(dd); ws.append(len(aa) * len(bb) / (len(aa) + len(bb)))
    d_str = float(np.average(ds, weights=ws)) if ds else np.nan
    rows.append(dict(DMS_id=dms, n_iface=len(a), n_noniface=len(b),
                     median_iface=np.median(a) if len(a) else np.nan,
                     median_noniface=np.median(b) if len(b) else np.nan,
                     cliffs_delta=d, p_mwu=p, n_strata=len(ds), cliffs_delta_depthstrat=d_str,
                     testable=(len(a) >= MIN_N and len(b) >= MIN_N)))
r = pd.DataFrame(rows).sort_values("cliffs_delta")
# Benjamini-Hochberg over testable assays
t = r["testable"].to_numpy()
p = r["p_mwu"].to_numpy(dtype=float)
q = np.full(len(r), np.nan)
if t.sum():
    pv = p[t]; o = np.argsort(pv); m = len(pv)
    adj = np.minimum.accumulate((pv[o] * m / (np.arange(m) + 1))[::-1])[::-1]
    qq = np.empty(m); qq[o] = np.clip(adj, 0, 1); q[t] = qq
r["q_BH"] = q
r.to_csv(f"{OUT}/stats_iface_vs_noniface.csv", index=False)
pd.set_option("display.width", 300)
print(r.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print()
tt = r[r.testable]
print(f"testable assays: {len(tt)}/25")
print(f"  q<0.05: {(tt.q_BH<0.05).sum()}   of which delta<0 (iface WORSE): {((tt.q_BH<0.05)&(tt.cliffs_delta<0)).sum()}"
      f"   delta>0 (iface BETTER): {((tt.q_BH<0.05)&(tt.cliffs_delta>0)).sum()}")
print(f"  |delta| median {tt.cliffs_delta.abs().median():.3f}  range [{tt.cliffs_delta.min():.3f}, {tt.cliffs_delta.max():.3f}]")
print(f"  depth-stratified delta vs raw: corr={np.corrcoef(tt.cliffs_delta, tt.cliffs_delta_depthstrat.fillna(0))[0,1]:.3f}")
