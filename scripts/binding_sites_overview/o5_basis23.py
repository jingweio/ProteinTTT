"""O5: every aggregate the overview quotes, recomputed on the 23-assay basis.

The document was written on all 25. Two assays carry duplicated labels (see
Sources/datasets/BindingGYM-issues), so the working basis becomes 23, and the nesting
23 -> 14 (binary testable) -> 13 (graded usable) is derived here rather than asserted.
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "binding_sites"))
from bg_common import OUT_DIR, REC_DIR

PRED = os.path.normpath(os.path.join(REC_DIR, "..", "binding-sites-analysis-pred"))
OVER = os.path.normpath(os.path.join(REC_DIR, "..", "binding-sites-overview"))
EXCL = ["KRAS_DARPinK27_norfitness_5O2S", "KRAS_SOS1_norfitness_8BE4"]

a = pd.read_csv(f"{OUT_DIR}/assay_summary.csv")
v = pd.read_parquet(f"{PRED}/data/variant_labels_with_mpnn.parquet")
st = pd.read_csv(f"{PRED}/data/stats_dms_vs_mpnn.csv")
a23, v23, st23 = a[~a.DMS_id.isin(EXCL)], v[~v.DMS_id.isin(EXCL)], st[~st.DMS_id.isin(EXCL)]

def line(lbl, old, new):
    flag = "  <-- 变了" if abs(old - new) > 5e-4 else ""
    print(f"  {lbl:44s} 25: {old:.4f}   23: {new:.4f}{flag}")

print("=== §0 / §1 的聚合量 ===")
line("frac 不碰界面 (D1, assay 未加权均值)", a.frac_no_iface.mean(), a23.frac_no_iface.mean())
line("frac 不碰界面 (D1, variant 加权)",
     (~v["iface_dist_5.0"]).mean(), (~v23["iface_dist_5.0"]).mean())
line("frac 不碰界面 (D0, assay 均值)", a.frac_no_iface_OTHERdef.mean(), a23.frac_no_iface_OTHERdef.mean())
line("frac 不碰界面 (D2 dSASA, variant 加权)",
     (~v["iface_dsasa_1.0"]).mean(), (~v23["iface_dsasa_1.0"]).mean())
print("  cutoff 敏感性 (variant 加权):")
for c in (4.0, 4.5, 5.0, 6.0):
    line(f"    {c} A", (~v[f"iface_dist_{c}"]).mean(), (~v23[f"iface_dist_{c}"]).mean())

print("\n=== §3.2 的分布形状与文库划分 ===")
for nm, df in (("25", a), ("23", a23)):
    z = df.frac_no_iface
    print(f"  {nm}: <=0.01 的 {int((z<=0.01).sum())}   0.37-0.97 的 {int(((z>=0.37)&(z<=0.97)).sum())}"
          f"   =1.000 的 {int((z>0.97).sum())}   其余 {int(((z>0.01)&(z<0.37)).sum())}")
    d, s_ = df[df.n_lib_pos <= 10], df[df.n_lib_pos >= 34]
    print(f"      设计型小文库 {len(d)} 个 ({d.n_var.sum():,} var, 均值 {d.frac_no_iface.mean():.3f})"
          f" | 饱和扫描 {len(s_)} 个 ({s_.n_var.sum():,} var, 均值 {s_.frac_no_iface.mean():.3f})")
    r = df.n_site_res / df.L_mutated
    print(f"      扫描型的 n_site/L: {r[df.n_lib_pos>=34].min():.3f}-{r[df.n_lib_pos>=34].max():.3f}"
          f" (中位 {r[df.n_lib_pos>=34].median():.3f})")

print("\n=== 集合的嵌套（从原始数据推导）===")
set23 = sorted(a23.DMS_id)
set14 = sorted(st23[st23.testable].DMS_id)
g = pd.read_csv(f"{OVER}/data/graded_bins_per_assay.csv")
graded_ok = set(g[g.usable].DMS_id)
set13 = sorted(set(set14) & graded_ok)
print(f"  23 (排除 2 个 label 重复的)          : {len(set23)}")
print(f"  14 = 23 中二值两侧都 >=30            : {len(set14)}")
print(f"  13 = 14 中还能切出 >=3 档且每档 >=30 : {len(set13)}")
print(f"  14 里切不出档的  : {sorted(set(set14)-graded_ok)}")
print(f"  分档可用但不在 14 里（本次不再纳入）: {sorted(graded_ok - set(set14) - set(EXCL))}")

print("\n=== §4 / §5 的汇总（14 口径）===")
k = st23[st23.testable]
for tag in ("dms", "mpnn"):
    d = k[f"cliffs_delta_{tag}"]
    print(f"  {tag:5s}: delta<0 {int((d<0).sum())}/{len(k)}   |delta| 中位 {d.abs().median():.3f}"
          f"   OVL 中位 {k[f'overlap_coef_{tag}'].median():.3f}   eta2 中位 {k[f'eta2_{tag}'].median():.3f}")
print(f"  |delta_mpnn|<|delta_dms| 的 {int((k.cliffs_delta_mpnn.abs()<k.cliffs_delta_dms.abs()).sum())}/{len(k)}"
      f"   中位比值 {(k.cliffs_delta_mpnn.abs()/k.cliffs_delta_dms.abs()).median():.3f}")
r1 = stats.spearmanr(k.cliffs_delta_dms, k.cliffs_delta_mpnn)
r2 = stats.spearmanr(k.rho_mpnn_vs_dms, k.cliffs_delta_mpnn)
print(f"  Spearman(delta_dms, delta_mpnn) = {r1.statistic:+.3f} (p={r1.pvalue:.2f})")
print(f"  Spearman(rho, delta_mpnn)       = {r2.statistic:+.3f} (p={r2.pvalue:.2f})")
flip = k.cliffs_delta_mpnn > 0
u = stats.mannwhitneyu(k.rho_mpnn_vs_dms[flip], k.rho_mpnn_vs_dms[~flip])
print(f"  符号翻转 {int(flip.sum())}/{len(k)}；翻转组 rho 中位 {k.rho_mpnn_vs_dms[flip].median():.3f}"
      f" vs 一致组 {k.rho_mpnn_vs_dms[~flip].median():.3f} (MWU p={u.pvalue:.2f})")
# 不碰界面组保留的 IQR 比例
ve = pd.read_csv(f"{OUT_DIR}/variance_explained.csv")
ve = ve[~ve.DMS_id.isin(EXCL)].dropna(subset=["eta2"])
print(f"  不碰界面组保留的 IQR 比例 中位 {(ve.iqr_noniface/ve.iqr_all).median():.2f}")
pd.DataFrame({"set23": pd.Series(set23)}).to_csv(f"{OVER}/data/basis_sets.csv", index=False)
