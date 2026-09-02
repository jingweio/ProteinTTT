import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "binding_sites"))
from bg_common import REC_DIR
PRED = os.path.normpath(os.path.join(REC_DIR, "..", "binding-sites-analysis-pred"))
t = pd.read_csv(f"{PRED}/data/stats_dms_vs_mpnn.csv")
b = pd.read_csv(f"{PRED}/data/burial_contrast.csv")
o = pd.read_csv(f"{PRED}/data/burial_stratified_delta.csv")
f = lambda x, n=3: "—" if pd.isna(x) else f"{x:+.{n}f}"
g = lambda x, n=3: "—" if pd.isna(x) else f"{x:.{n}f}"

print("### Table P1  逐 assay：DMS 与 ProteinMPNN 的界面对比\n")
print("| assay | n 碰 / 不碰 | δ **DMS** | δ **MPNN** | 符号 | OVL DMS | OVL MPNN | η² DMS | η² MPNN | ρ(MPNN, DMS) |")
print("|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|")
for _, r in t.query("testable").sort_values("cliffs_delta_dms").iterrows():
    flip = "**翻转**" if np.sign(r.cliffs_delta_mpnn) != np.sign(r.cliffs_delta_dms) else "一致"
    print(f"| {r.DMS_id} | {int(r.n_iface):,} / {int(r.n_noniface):,} | {f(r.cliffs_delta_dms)} | "
          f"**{f(r.cliffs_delta_mpnn)}** | {flip} | {g(r.overlap_coef_dms)} | {g(r.overlap_coef_mpnn)} | "
          f"{g(r.eta2_dms)} | {g(r.eta2_mpnn)} | {g(r.rho_mpnn_vs_dms)} |")

print("\n\n### Table P2  burial 检验（partner 已删除的链内 Cβ 邻居数，10 Å）\n")
m = b.dropna(subset=["delta_dms"]).merge(o[["DMS_id","delta_mpnn_burialstrat","delta_dms_burialstrat"]], on="DMS_id", how="left")
print("| assay | 库位点 site / non-site | burial site | burial non-site | Δburial | δ MPNN | δ MPNN (burial 分层后) | δ DMS |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for _, r in m.sort_values("d_burial").iterrows():
    print(f"| {r.DMS_id} | {int(r.n_lib_site)} / {int(r.n_lib_nonsite)} | {g(r.burial_site,2)} | "
          f"{g(r.burial_nonsite,2)} | {f(r.d_burial,2)} | {f(r.delta_mpnn)} | {f(r.delta_mpnn_burialstrat)} | {f(r.delta_dms)} |")
