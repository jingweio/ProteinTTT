"""O3: replace the binary touch/no-touch split with K ordered bins of the soft feature.

The binary split says the two groups differ but overlap heavily. It cannot say whether
that difference is a GRADIENT in distance or a THRESHOLD at 5 A. Binning on the same
soft feature used in the complexTTT gates separates the two, and the answer explains
whether the binary label is a lossy summary or an adequate one.

Same framework as sections 4/5 of the overview, with K = 5 instead of K = 2:
  delta (2 groups)   ->  Spearman(bin index, score)
  eta^2 (2 groups)   ->  eta^2 over K groups
Bins are quantiles of d, so B1 = farthest from the other entity, B5 = closest.
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "binding_sites"))
from bg_common import REC_DIR

PRED = os.path.normpath(os.path.join(REC_DIR, "..", "binding-sites-analysis-pred"))
OUT = os.path.normpath(os.path.join(REC_DIR, "..", "binding-sites-overview"))
K, MIN_PER_BIN = 5, 30
EXCL_FOR_TTT = {"KRAS_DARPinK27_norfitness_5O2S", "KRAS_SOS1_norfitness_8BE4"}


v = pd.read_parquet(f"{PRED}/data/variant_labels_with_mpnn.parquet")
# Population for this section is the 14-assay working set: the 23 that are label-clean,
# restricted to those where the binary split of sections 4/5 is defined, so the graded
# view is directly comparable with them and with the complexTTT record.
_st = pd.read_csv(f"{PRED}/data/stats_dms_vs_mpnn.csv")
SET14 = sorted(set(_st[_st.testable].DMS_id) - EXCL_FOR_TTT)
v = v[v.DMS_id.isin(SET14)]

def eta2(y, lab):
    tot = np.var(y)
    within = sum(((lab == g).sum()) * np.var(y[lab == g]) for g in np.unique(lab)) / len(y)
    return 1 - within / tot if tot > 0 else np.nan

rows, per_bin = [], []
for dms, g in v.groupby("DMS_id"):
    d = g.min_dist_to_partner.to_numpy(float)
    y = g.DMS_score.to_numpy(float); m = g.mpnn_score.to_numpy(float)
    ok = np.isfinite(d) & np.isfinite(y) & np.isfinite(m); d, y, m = d[ok], y[ok], m[ok]
    # quantile bins of d; B1 = farthest. duplicates='drop' handles assays with few unique d
    try:
        b = pd.qcut(-d, K, labels=False, duplicates="drop")     # -d so B1 = largest d
    except ValueError:
        b = np.zeros(len(d), int)
    nb = len(np.unique(b))
    sizes = np.bincount(b)
    usable = nb >= 3 and sizes.min() >= MIN_PER_BIN
    r = dict(DMS_id=dms, n=len(d), n_bins=nb, min_bin_n=int(sizes.min()), usable=usable,
             excluded_for_ttt=dms in EXCL_FOR_TTT,
             d_min=d.min(), d_max=d.max(), n_unique_d=len(np.unique(np.round(d, 2))))
    if usable:
        r["rho_bin_dms"] = stats.spearmanr(b, y).statistic
        r["rho_bin_mpnn"] = stats.spearmanr(b, m).statistic
        r["eta2_bin_dms"] = eta2(y, b); r["eta2_bin_mpnn"] = eta2(m, b)
        med_y = np.array([np.median(y[b == i]) for i in range(nb)])
        med_m = np.array([np.median(m[b == i]) for i in range(nb)])
        r["mono_dms"] = int(np.all(np.diff(med_y) <= 0) or np.all(np.diff(med_y) >= 0))
        r["mono_mpnn"] = int(np.all(np.diff(med_m) <= 0) or np.all(np.diff(med_m) >= 0))
        # strict monotonicity is brittle with K=5; also report how many adjacent steps
        # move the expected way (closer to the interface -> lower score)
        r["steps_down_dms"] = int((np.diff(med_y) < 0).sum()); r["n_steps"] = nb - 1
        r["steps_down_mpnn"] = int((np.diff(med_m) < 0).sum())
        # threshold-vs-gradient: is there still a trend INSIDE each side of the 5 A cut?
        for tag, sel in (("far", d > 5.0), ("near", d <= 5.0)):
            if sel.sum() >= 100 and len(np.unique(d[sel])) > 2:
                r[f"rho_d_dms_{tag}"] = stats.spearmanr(d[sel], y[sel]).statistic
                r[f"p_d_dms_{tag}"] = stats.spearmanr(d[sel], y[sel]).pvalue
                r[f"n_{tag}"] = int(sel.sum())
        for i in range(nb):
            s = b == i
            per_bin.append(dict(DMS_id=dms, bin=i + 1, n=int(s.sum()),
                                d_lo=d[s].min(), d_hi=d[s].max(), d_med=float(np.median(d[s])),
                                dms_med=float(np.median(y[s])), dms_mean=float(y[s].mean()),
                                mpnn_med=float(np.median(m[s])),
                                dms_meanrank=float(stats.rankdata(y)[s].mean() / len(y)),
                                mpnn_meanrank=float(stats.rankdata(m)[s].mean() / len(m))))
    rows.append(r)

t = pd.DataFrame(rows); pb = pd.DataFrame(per_bin)
os.makedirs(f"{OUT}/data", exist_ok=True)
t.to_csv(f"{OUT}/data/graded_bins_per_assay.csv", index=False)
pb.to_csv(f"{OUT}/data/graded_bins_per_bin.csv", index=False)
pd.set_option("display.width", 300)

print(f"=== 口径：14-assay 工作集；K={K} 分位档（按 d），要求 >=3 档且每档 >=%d ===" % MIN_PER_BIN)
print(f"  14 个中可用: {int(t.usable.sum())}   不可用: {int((~t.usable).sum())}")
if (~t.usable).any():
    print(t[~t.usable][["DMS_id", "n", "n_bins", "min_bin_n", "n_unique_d"]].to_string(index=False))
u = t[t.usable]
print(f"\n=== 逐 assay（可用 {len(u)} 个） ===")
uu = u.copy()
print(uu[["DMS_id", "n", "n_bins", "rho_bin_dms", "rho_bin_mpnn",
          "eta2_bin_dms", "eta2_bin_mpnn", "mono_dms", "mono_mpnn"]]
      .sort_values("rho_bin_dms").to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
print(f"\n=== 汇总（{len(u)} 个可用 assay） ===")
print(f"  真值   Spearman(bin, DMS)  中位 {u.rho_bin_dms.median():+.3f}   >0 的有 {int((u.rho_bin_dms>0).sum())}/{len(u)}"
      f"   档中位数单调的 {int(u.mono_dms.sum())}/{len(u)}")
print(f"  MPNN   Spearman(bin, MPNN) 中位 {u.rho_bin_mpnn.median():+.3f}   >0 的有 {int((u.rho_bin_mpnn>0).sum())}/{len(u)}"
      f"   档中位数单调的 {int(u.mono_mpnn.sum())}/{len(u)}")
print(f"  eta^2  真值中位 {u.eta2_bin_dms.median():.3f}   MPNN 中位 {u.eta2_bin_mpnn.median():.3f}")
print(f"  相邻档「往下走」的步数占比：真值 {u.steps_down_dms.sum()}/{u.n_steps.sum()}"
      f" = {u.steps_down_dms.sum()/u.n_steps.sum():.0%}   MPNN {u.steps_down_mpnn.sum()}/{u.n_steps.sum()}"
      f" = {u.steps_down_mpnn.sum()/u.n_steps.sum():.0%}   (随机是 50%)")
print(f"  Spearman(rho_bin_dms, rho_bin_mpnn) 跨 assay = "
      f"{stats.spearmanr(u.rho_bin_dms, u.rho_bin_mpnn).statistic:+.3f}"
      f"  (p={stats.spearmanr(u.rho_bin_dms, u.rho_bin_mpnn).pvalue:.3f})")

print(f"\n=== 渐变 还是 阈值？—— 在 5 Å 两侧各自内部还有没有趋势 ===")
w = u.dropna(subset=["rho_d_dms_far"]) if "rho_d_dms_far" in u else pd.DataFrame()
for tag, lab in (("far", "只看 d > 5 Å（全是「不碰」）"), ("near", "只看 d <= 5 Å（全是「碰」）")):
    c = f"rho_d_dms_{tag}"
    if c not in u: continue
    s = u.dropna(subset=[c])
    print(f"  {lab:30s} n_assay={len(s):2d}   Spearman(d, DMS) 中位 {s[c].median():+.3f}"
          f"   >0 的 {int((s[c]>0).sum())}/{len(s)}   p<0.05 的 {int((s['p_d_dms_'+tag]<0.05).sum())}/{len(s)}")
