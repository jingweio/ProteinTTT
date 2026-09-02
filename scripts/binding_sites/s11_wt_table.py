import numpy as np, pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # runnable from any cwd
from bg_common import OUT_DIR, REC_DIR

w=pd.read_csv(f"{OUT_DIR}/wt_reference.csv"); v=pd.read_parquet(f"{OUT_DIR}/variant_labels.parquet")
rows=[]
for _,r in w.iterrows():
    g=v[v.DMS_id==r.DMS_id]
    if np.isfinite(r.wt_score):
        gt=g.DMS_score>r.wt_score; m=g["iface_dist_5.0"]
        gi=gt[m].mean() if m.sum()>=30 else np.nan
        gn=gt[~m].mean() if (~m).sum()>=30 else np.nan
    else: gi=gn=np.nan
    grp=("无 WT 行" if not np.isfinite(r.wt_score)
         else "锚点 ≈ 0" if abs(r.wt_score)<0.12 else "锚点 ≠ 0")
    rows.append(dict(DMS_id=r.DMS_id,grp=grp,wt=r.wt_score,pct=r.wt_pctile,
                     vmin=r.vmin,med=r["median"],vmax=r.vmax,fgt=r.frac_gt_wt,ngt=r.n_gt_wt,
                     gi=gi,gn=gn))
t=pd.DataFrame(rows)
order={"锚点 ≈ 0":0,"锚点 ≠ 0":1,"无 WT 行":2}
t=t.sort_values(["grp","fgt"],key=lambda c:c.map(order) if c.name=="grp" else c)
f=lambda x,n=3: "—" if not np.isfinite(x) else f"{x:.{n}f}"
print("| assay | 分组 | **WT score** | WT 分位 | variant 范围 (min / median / max) | 优于 WT 的比例 | n | 碰界面组 | 不碰组 |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for _,r in t.iterrows():
    print(f"| {r.DMS_id} | {r.grp} | **{f(r.wt)}** | {'—' if not np.isfinite(r.pct) else f'{r.pct:.1f}%'} | "
          f"{f(r["vmin"],2)} / {f(r["med"],2)} / {f(r["vmax"],2)} | {'—' if not np.isfinite(r.fgt) else f'{r.fgt:.1%}'} | "
          f"{'—' if not np.isfinite(r.ngt) else f'{int(r.ngt):,}'} | "
          f"{'—' if not np.isfinite(r.gi) else f'{r.gi:.1%}'} | {'—' if not np.isfinite(r.gn) else f'{r.gn:.1%}'} |")
