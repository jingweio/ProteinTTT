"""G1c: two things G1b left open.

(1) the shared c was picked to maximise the mean over the SAME 14 assays -> refit it
    leave-one-assay-out, so every number is out-of-sample.
(2) the interface label is binary. The parquet also carries the continuous distance to
    the other entity, and a zero-parameter distance feature is already known to score
    rho ~ 0.26 on its own -- so ask whether a continuous form does better than the
    binary one before designing anything around the binary label.
"""
import numpy as np, pandas as pd
from scipy import stats

REC = "/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records"
OUT = f"{REC}/mutation-landscape-TTT/data"
EXCLUDE = ["KRAS_DARPinK27_norfitness_5O2S", "KRAS_SOS1_norfitness_8BE4"]
GRID = np.linspace(-4, 4, 161)

v = pd.read_parquet(f"{REC}/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet")
v = v[~v.DMS_id.isin(EXCLUDE)]

FEATS = {
    "binary  1[d<=5A]":        lambda d: (d <= 5.0).astype(float),
    "soft    exp(-d/5A)":      lambda d: np.exp(-d / 5.0),
    "soft    1/(1+d/5A)":      lambda d: 1.0 / (1.0 + d / 5.0),
    "raw     -d (clipped 20A)": lambda d: -np.minimum(d, 20.0) / 20.0,
}
names, ids, base = list(FEATS), [], []
curves = {k: [] for k in names}
for dms, g in v.groupby("DMS_id"):
    s = g.mpnn_score.to_numpy(float); y = g.DMS_score.to_numpy(float)
    d = g.min_dist_to_partner.to_numpy(float)
    ok = np.isfinite(s) & np.isfinite(y) & np.isfinite(d); s, y, d = s[ok], y[ok], d[ok]
    z = (d <= 5.0)
    if z.sum() < 30 or (~z).sum() < 30: continue
    ids.append(dms); base.append(stats.spearmanr(s, y).statistic)
    sd = s.std() or 1.0
    for k, f in FEATS.items():
        w = f(d); w = (w - w.mean()) / (w.std() or 1.0)      # standardise so c is comparable
        curves[k].append([stats.spearmanr(s + c * sd * w, y).statistic for c in GRID])
base = np.array(base); n = len(ids)
print(f"{n} assays\n")
print(f"{'feature':26s}{'oracle c*':>11s}{'LOAO c':>9s}{'fixed c=-1':>12s}{'   LOAO/oracle':>15s}")
res = {}
for k in names:
    C = np.array(curves[k])                                   # n x len(GRID)
    star = C.max(axis=1)
    loao = np.array([C[i, C[np.arange(n) != i].mean(axis=0).argmax()] for i in range(n)])
    fix1 = C[:, int(np.abs(GRID + 1.0).argmin())]
    res[k] = dict(star=star, loao=loao, fix1=fix1)
    print(f"  {k:24s}{star.mean():>11.4f}{loao.mean():>9.4f}{fix1.mean():>12.4f}"
          f"{100*(loao.mean()-base.mean())/(star.mean()-base.mean()):>14.0f}%")
print(f"\n  baseline (no correction)  {base.mean():.4f}")
print(f"\n{'feature':26s}{'gain oracle':>13s}{'gain LOAO':>11s}{'gain fixed':>12s}")
for k in names:
    r = res[k]
    print(f"  {k:24s}{r['star'].mean()-base.mean():>+13.4f}"
          f"{r['loao'].mean()-base.mean():>+11.4f}{r['fix1'].mean()-base.mean():>+12.4f}")
bestk = max(names, key=lambda k: res[k]["loao"].mean())
print(f"\n  best out-of-sample feature: {bestk}   ->  {res[bestk]['loao'].mean():.4f} "
      f"({res[bestk]['loao'].mean()-base.mean():+.4f})")
pd.DataFrame({"DMS_id": ids, "rho_base": base,
              **{f"{k.split()[0]}_{t}": res[k][t] for k in names for t in ("star","loao","fix1")}}
             ).to_csv(f"{OUT}/g1c_loao.csv", index=False)
