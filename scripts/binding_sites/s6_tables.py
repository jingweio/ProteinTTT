"""S6: render the markdown tables used in the record."""
import numpy as np, pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # runnable from any cwd
from bg_common import OUT_DIR, REC_DIR
D = OUT_DIR
a = pd.read_csv(f"{D}/assay_summary.csv"); s = pd.read_csv(f"{D}/binding_sites_per_chain.csv")
st = pd.read_csv(f"{D}/stats_iface_vs_noniface.csv"); rb = pd.read_csv(f"{D}/robustness.csv")
ve = pd.read_csv(f"{D}/variance_explained.csv")

def compact(ids):
    """['24','25','26','30'] -> '24-26,30'  (keeps non-numeric ids verbatim)"""
    nums, other, out = [], [], []
    for x in ids:
        (nums if x.lstrip("-").isdigit() else other).append(x)
    nums = sorted(int(x) for x in nums)
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j+1] == nums[j] + 1: j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}"); i = j + 1
    return ",".join(out + sorted(other))

# ---- Table A: binding-site ranges ----
print("### Table A  每个 assay 的 binding-site 残基（D1, heavy-atom <= 5 A to partner）\n")
print("| assay | 被突变链 | partner 链 | binding-site 残基（PDB 编号，逐链） | n_site / L |")
print("|---|---|---|---|---|")
for _, r in a.iterrows():
    g = s[s.DMS_id == r.DMS_id]
    parts, nsite, ltot = [], 0, 0
    for _, c in g.iterrows():
        ids = str(c.site_pdbres).split(";") if isinstance(c.site_pdbres, str) and c.site_pdbres else []
        parts.append(f"`{c.chain}`: {compact(ids) if ids else '(none)'}")
        nsite += len(ids); ltot += c.n_res
    print(f"| {r.DMS_id} | {r.mutated_chains} | {r.partner_chains} | " + "<br>".join(parts) + f" | {nsite}/{ltot} |")

# ---- Table B: library positions & variant split ----
print("\n\n### Table B  库位点 / variant 分组\n")
print("| assay | n_var | 库位点数 | 其中在 binding-site 上 | n(碰界面) | n(不碰) | frac 不碰 (D1) | frac 不碰 (D0，原口径) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for _, r in a.sort_values("frac_no_iface").iterrows():
    print(f"| {r.DMS_id} | {r.n_var:,} | {r.n_lib_pos} | {r.n_lib_pos_site} | {r.n_iface_var:,} | "
          f"{r.n_noniface_var:,} | {r.frac_no_iface:.3f} | {r.frac_no_iface_OTHERdef:.3f} |")

# ---- Table C: stats ----
print("\n\n### Table C  两组 DMS_score 的差异检验\n")
m = st.merge(ve[["DMS_id", "eta2", "iqr_noniface", "iqr_all"]], on="DMS_id").merge(
    rb[["DMS_id", "delta_dsasa", "delta_singles", "n_singles"]], on="DMS_id")
m = m[m.testable].sort_values("cliffs_delta")
print("| assay | n(碰) | n(不碰) | median 碰 | median 不碰 | Cliff's δ | q (BH) | δ depth-strat | δ dSASA 定义 | δ 仅单点 | η² |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for _, r in m.iterrows():
    f = lambda x: "—" if pd.isna(x) else f"{x:.3f}"
    q = "—" if pd.isna(r.q_BH) else ("<1e-99" if r.q_BH < 1e-99 else f"{r.q_BH:.1e}")
    print(f"| {r.DMS_id} | {r.n_iface:,} | {r.n_noniface:,} | {f(r.median_iface)} | {f(r.median_noniface)} | "
          f"**{f(r.cliffs_delta)}** | {q} | {f(r.cliffs_delta_depthstrat)} | {f(r.delta_dsasa)} | "
          f"{f(r.delta_singles)} | {f(r.eta2)} |")
print(f"\n不可检验的 9 个 assay（一侧子集 < 30）：" +
      ", ".join(f"`{d}`" for d in st[~st.testable].DMS_id))
