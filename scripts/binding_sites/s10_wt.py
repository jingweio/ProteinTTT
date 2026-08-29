"""S10: where does the wild type sit on each assay's DMS_score scale?"""
import os, numpy as np, pandas as pd
from bg_common import *

idx = load_index()
rows = []
for _, r in idx.iterrows():
    df = pd.read_csv(os.path.join(DMS_DIR, r["DMS_filename"]))
    is_wt = df["mutant"].map(lambda s: all(not v.strip() for v in parse_mut_dict(s).values()))
    y = df.loc[~is_wt, "DMS_score"].to_numpy(); y = y[np.isfinite(y)]
    n_wt = int(is_wt.sum())
    wt = float(df.loc[is_wt, "DMS_score"].iloc[0]) if n_wt else np.nan
    d = dict(DMS_id=r["DMS_id"], n_var=len(y), n_wt_rows=n_wt, wt_score=wt,
             vmin=y.min(), q25=np.percentile(y, 25), median=np.median(y),
             q75=np.percentile(y, 75), vmax=y.max(), mean=y.mean())
    if n_wt:
        d["wt_pctile"] = float((y < wt).mean() * 100)
        d["frac_gt_wt"] = float((y > wt).mean())
        d["n_gt_wt"] = int((y > wt).sum())
    rows.append(d)
t = pd.DataFrame(rows)
pd.set_option("display.width", 320)
print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\nassays with a WT row:", int(t.n_wt_rows.gt(0).sum()), "/ 25")
print("no WT row:", ", ".join(t.loc[t.n_wt_rows == 0, "DMS_id"]))
print("\nwt_score == 0 exactly:", int((t.wt_score == 0).sum()),
      " |wt_score| < 1e-6:", int((t.wt_score.abs() < 1e-6).sum()))
t.to_csv("../../local-records/binding-sites-analysis/data/wt_reference.csv", index=False)
