"""O2: the four tables for the consolidated overview, trimmed to the metrics we keep
(delta / OVL / eta^2 / rho). P(non>iface), Cohen's d and the BH q are dropped on purpose."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "binding_sites"))
from bg_common import OUT_DIR, REC_DIR

PRED = os.path.normpath(os.path.join(REC_DIR, "..", "binding-sites-analysis-pred"))
a = pd.read_csv(f"{OUT_DIR}/assay_summary.csv")
sites = pd.read_csv(f"{OUT_DIR}/binding_sites_per_chain.csv")
cmp_ = pd.read_csv(f"{PRED}/data/stats_dms_vs_mpnn.csv")
f = lambda x, n=3: "—" if pd.isna(x) else f"{x:.{n}f}"
sg = lambda x, n=3: "—" if pd.isna(x) else f"{x:+.{n}f}"

def compact(ids):
    nums, other, out = [], [], []
    for x in ids:
        (nums if x.lstrip("-").isdigit() else other).append(x)
    nums = sorted(int(x) for x in nums); i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1: j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}"); i = j + 1
    return ",".join(out + sorted(other))

print("=====TABLE1=====")
print("| assay | E1 中被突变的链 | 另一个 entity | binding-site 残基（PDB 编号，逐链） | n_site / L |")
print("|---|---|---|---|---|")
for _, r in a.iterrows():
    g = sites[sites.DMS_id == r.DMS_id]; parts, ns, lt = [], 0, 0
    for _, c in g.iterrows():
        ids = str(c.site_pdbres).split(";") if isinstance(c.site_pdbres, str) and c.site_pdbres else []
        parts.append(f"`{c.chain}`: {compact(ids) if ids else '(none)'}"); ns += len(ids); lt += c.n_res
    print(f"| {r.DMS_id} | {r.mutated_chains} | {r.partner_chains} | " + "<br>".join(parts) + f" | {ns}/{lt} |")

print("=====TABLE2=====")
print("| assay | n_var | 库位点数 | 其中在 binding-site 上 | n(碰) | n(不碰) | **frac 不碰** |")
print("|---|---:|---:|---:|---:|---:|---:|")
for _, r in a.sort_values("frac_no_iface").iterrows():
    print(f"| {r.DMS_id} | {r.n_var:,} | {r.n_lib_pos} | {r.n_lib_pos_site} | {r.n_iface_var:,} | "
          f"{r.n_noniface_var:,} | **{r.frac_no_iface:.3f}** |")

print("=====TABLE3=====")
t = cmp_.query("testable").sort_values("cliffs_delta_dms")
print("| assay | n 碰 / 不碰 | mean 碰 / 不碰 | median 碰 / 不碰 | **δ** | **OVL** | **η²** |")
print("|---|---:|---:|---:|---:|---:|---:|")
for _, r in t.iterrows():
    print(f"| {r.DMS_id} | {int(r.n_iface):,} / {int(r.n_noniface):,} | "
          f"{f(r.mean_iface_dms,2)} / {f(r.mean_noniface_dms,2)} | {f(r.median_iface_dms,2)} / {f(r.median_noniface_dms,2)} | "
          f"**{sg(r.cliffs_delta_dms)}** | **{f(r.overlap_coef_dms)}** | **{f(r.eta2_dms)}** |")

print("=====TABLE4=====")
print("| assay | **δ** 真值 | **δ** MPNN | 符号 | **OVL** 真值 | **OVL** MPNN | **η²** 真值 | **η²** MPNN | **ρ** |")
print("|---|---:|---:|:--:|---:|---:|---:|---:|---:|")
for _, r in t.iterrows():
    flip = np.sign(r.cliffs_delta_mpnn) != np.sign(r.cliffs_delta_dms)
    print(f"| {r.DMS_id} | {sg(r.cliffs_delta_dms)} | **{sg(r.cliffs_delta_mpnn)}** | "
          f"{'**翻转**' if flip else '一致'} | {f(r.overlap_coef_dms)} | {f(r.overlap_coef_mpnn)} | "
          f"{f(r.eta2_dms)} | {f(r.eta2_mpnn)} | {f(r.rho_mpnn_vs_dms)} |")
