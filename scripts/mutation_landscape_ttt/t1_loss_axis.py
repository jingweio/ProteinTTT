"""T1: where does the decoder-TTT loss have to move the model?

The primary objective is a within-batch correlation between the interface prior
w = exp(-d/tau) and the model's predicted score. That makes "what the loss does" a
measurable one-dimensional statement: the model sits somewhere on corr(w, score), the
truth sits somewhere else, and the loss has to close the gap. Both ends are computable
from data we already have, before any training.

14-assay working set.
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "binding_sites"))
from bg_common import REC_DIR

WS = os.path.normpath(os.path.join(REC_DIR, "..", "..", "workstation-records", "mutation-landscape-TTT"))
PRED = os.path.normpath(os.path.join(REC_DIR, "..", "binding-sites-analysis-pred"))
TAU = 5.0

k = pd.read_csv(f"{WS}/data/g1e_canonical14.csv")
v = pd.read_parquet(f"{PRED}/data/variant_labels_with_mpnn.parquet")
v = v[v.DMS_id.isin(set(k.DMS_id))]

rows = []
for a, g in v.groupby("DMS_id"):
    d = g.min_dist_to_partner.to_numpy(float)
    s = g.mpnn_score.to_numpy(float); y = g.DMS_score.to_numpy(float)
    ok = np.isfinite(d) & np.isfinite(s) & np.isfinite(y); d, s, y = d[ok], s[ok], y[ok]
    w = np.exp(-d / TAU)
    rows.append(dict(DMS_id=a, n=len(d),
                     now=np.corrcoef(w, s)[0, 1],           # 模型当前在这根轴上的位置
                     target=np.corrcoef(w, y)[0, 1],        # 真值的位置
                     now_sp=stats.spearmanr(w, s).statistic,
                     target_sp=stats.spearmanr(w, y).statistic))
t = pd.DataFrame(rows); t["gap"] = t.target - t.now
t = t.sort_values("gap")
t.to_csv(f"{WS}/data/t1_loss_axis.csv", index=False)

f = lambda x: f"{x:+.3f}"
print("=====TABLE=====")
print("| assay | n | 模型现在 `corr(w,s)` | 真值 `corr(w,y)` | 要移动 |")
print("|---|---:|---:|---:|---:|")
for _, r in t.iterrows():
    flip = " ⚠️" if r.now > 0 else ""
    print(f"| {r.DMS_id}{flip} | {int(r.n):,} | {f(r.now)} | {f(r.target)} | **{f(r.gap)}** |")
print("=====SUMMARY=====")
print(f"model_now_median={t.now.median():+.3f} n_positive={int((t.now>0).sum())}/{len(t)}")
print(f"target_median={t.target.median():+.3f} n_negative={int((t.target<0).sum())}/{len(t)}")
print(f"gap_median={t.gap.median():+.3f} n_same_direction={int((t.gap<0).sum())}/{len(t)}")
print(f"flipped={list(t[t.now>0].DMS_id)}")
print(f"flipped_gap_range={t[t.now>0].gap.min():+.3f}..{t[t.now>0].gap.max():+.3f}")
