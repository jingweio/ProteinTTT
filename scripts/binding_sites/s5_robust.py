import numpy as np, pandas as pd
from scipy import stats
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # runnable from any cwd
from bg_common import OUT_DIR, REC_DIR
OUT = OUT_DIR
v = pd.read_parquet(f"{OUT}/variant_labels.parquet"); a = pd.read_csv(f"{OUT}/assay_summary.csv")

def delta(x, y):
    if len(x) < 5 or len(y) < 5: return np.nan, np.nan
    U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    return 2*U/(len(x)*len(y)) - 1, p

rows = []
for dms, g in v.groupby("DMS_id"):
    r = dict(DMS_id=dms, n=len(g), n_mut_median=int(g.n_mut.median()), n_mut_max=int(g.n_mut.max()),
             frac_singles=round(float((g.n_mut == 1).mean()), 3))
    for tag, col in [("d5A", "iface_dist_5.0"), ("dsasa", "iface_dsasa_1.0")]:
        d, p = delta(g.loc[g[col], "DMS_score"], g.loc[~g[col], "DMS_score"])
        r[f"delta_{tag}"] = d; r[f"p_{tag}"] = p
    s = g[g.n_mut == 1]
    d, p = delta(s.loc[s["iface_dist_5.0"], "DMS_score"], s.loc[~s["iface_dist_5.0"], "DMS_score"])
    r["n_singles"] = len(s); r["delta_singles"] = d; r["p_singles"] = p
    ok = np.isfinite(g.min_dist_to_partner) & np.isfinite(g.DMS_score)
    r["spearman_score_vs_dist"] = stats.spearmanr(g.DMS_score[ok], g.min_dist_to_partner[ok]).statistic if ok.sum() > 20 else np.nan
    rows.append(r)
r = pd.DataFrame(rows).merge(a[["DMS_id","L_mutated","n_lib_pos","n_site_res","frac_no_iface"]], on="DMS_id")
r["lib_coverage"] = (r.n_lib_pos / r.L_mutated).round(3)
pd.set_option("display.width", 320)
print(r[["DMS_id","n","frac_singles","n_lib_pos","L_mutated","lib_coverage","frac_no_iface",
         "delta_d5A","delta_dsasa","n_singles","delta_singles","spearman_score_vs_dist"]]
      .sort_values("lib_coverage").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print()
ok = r.frac_no_iface.notna()
print("Spearman(lib_coverage, frac_no_iface) over 25 assays =",
      round(stats.spearmanr(r.lib_coverage[ok], r.frac_no_iface[ok]).statistic, 3),
      " p =", f"{stats.spearmanr(r.lib_coverage[ok], r.frac_no_iface[ok]).pvalue:.2g}")
b = r.dropna(subset=["delta_d5A","delta_dsasa"])
print("delta agreement d5A vs dSASA: pearson", round(np.corrcoef(b.delta_d5A, b.delta_dsasa)[0,1], 3),
      " max|diff|", round((b.delta_d5A-b.delta_dsasa).abs().max(), 3))
s = r.dropna(subset=["delta_d5A","delta_singles"])
print("delta agreement all-variants vs singles-only:", len(s), "assays, pearson",
      round(np.corrcoef(s.delta_d5A, s.delta_singles)[0,1], 3))
print("singles-only: all negative?", (s.delta_singles < 0).all(), " values:",
      dict(zip(s.DMS_id.str[:18], s.delta_singles.round(3))))
sp = r.dropna(subset=["spearman_score_vs_dist"])
print("\nSpearman(DMS_score, min_dist_to_partner) > 0 in", (sp.spearman_score_vs_dist>0).sum(), "/", len(sp))
r.to_csv(f"{OUT}/robustness.csv", index=False)
