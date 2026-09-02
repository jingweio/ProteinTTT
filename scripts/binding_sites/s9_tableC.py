import numpy as np, pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # runnable from any cwd
from bg_common import OUT_DIR, REC_DIR
D = OUT_DIR
st = pd.read_csv(f"{D}/stats_iface_vs_noniface_extended.csv")
old = pd.read_csv(f"{D}/stats_iface_vs_noniface.csv")[["DMS_id","cliffs_delta_depthstrat"]]
rb = pd.read_csv(f"{D}/robustness.csv")[["DMS_id","delta_dsasa","delta_singles"]]
ve = pd.read_csv(f"{D}/variance_explained.csv")[["DMS_id","eta2"]]
m = st.merge(old,on="DMS_id").merge(rb,on="DMS_id").merge(ve,on="DMS_id")
m = m[m.testable].sort_values("cliffs_delta")
f = lambda x: "—" if pd.isna(x) else f"{x:.3f}"
print("| assay | n 碰 / 不碰 | **mean** 碰 / 不碰 | median 碰 / 不碰 | sd 碰 / 不碰 | Cliff's δ | P(不碰>碰) | **OVL** | Cohen's d | q (BH) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for _, r in m.iterrows():
    q = "<1e-99" if r.q_BH < 1e-99 else f"{r.q_BH:.1e}"
    print(f"| {r.DMS_id} | {int(r.n_iface):,} / {int(r.n_noniface):,} | **{f(r.mean_iface)} / {f(r.mean_noniface)}** | "
          f"{f(r.median_iface)} / {f(r.median_noniface)} | {f(r.sd_iface)} / {f(r.sd_noniface)} | "
          f"**{f(r.cliffs_delta)}** | {f(r.P_noniface_gt_iface)} | **{f(r.overlap_coef)}** | {f(r.cohens_d)} | {q} |")
print()
print("稳健性与分层（同 16 个 assay）\n")
print("| assay | δ 主口径 | δ depth-分层 | δ ΔSASA 定义 | δ 仅单点突变 | η² |")
print("|---|---:|---:|---:|---:|---:|")
for _, r in m.iterrows():
    print(f"| {r.DMS_id} | {f(r.cliffs_delta)} | {f(r.cliffs_delta_depthstrat)} | {f(r.delta_dsasa)} | {f(r.delta_singles)} | {f(r.eta2)} |")
