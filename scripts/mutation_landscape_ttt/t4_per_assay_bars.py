"""T4: the per-assay reference lines a TTT result has to be read against.

A TTT number on one assay means nothing on its own. It needs two neighbours: the zero-shot
score it started from, and the best the five-line rescoring rule can do on that same assay
with an oracle c -- which is the ceiling of the "shift the score along the interface
feature" family, and therefore the bar TTT has to clear to have contributed anything.
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = f"{ROOT}/workstation-records/mutation-landscape-TTT/data"
GRID = np.linspace(-4, 4, 161)

lab = pd.read_parquet(f"{ROOT}/local-records/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet")
sdt = pd.read_parquet(f"{ROOT}/local-records/binding-sites-analysis/data/variant_site_dists.parquet")
set14 = list(pd.read_csv(f"{OUT}/g1e_canonical14.csv").DMS_id)

rows = []
for a in set14:
    g = lab[lab.DMS_id == a]
    s = g.mpnn_score.to_numpy(float); y = g.DMS_score.to_numpy(float)
    sd = sdt[sdt.DMS_id == a]; vi = sd.vi.to_numpy()
    f = np.exp(-sd.dist.to_numpy(float) / 5.0)
    st = np.r_[0, np.flatnonzero(np.diff(vi)) + 1]
    r = dict(DMS_id=a, n=len(y), n_mut_mean=float(g.n_mut.mean()),
             rho_zs=stats.spearmanr(s, y).statistic)
    for tag, w in [("max", np.fmax.reduceat(f, st)), ("sum", np.add.reduceat(f, st))]:
        wz = (w - w.mean()) / (w.std() or 1.0)
        c = np.array([stats.spearmanr(s + t * s.std() * wz, y).statistic for t in GRID])
        r[f"ceil_{tag}"] = c.max(); r[f"cstar_{tag}"] = GRID[c.argmax()]
    rows.append(r)
t = pd.DataFrame(rows)
t["headroom_max"] = t.ceil_max - t.rho_zs
t["headroom_sum"] = t.ceil_sum - t.rho_zs
t.to_csv(f"{OUT}/t4_per_assay_bars.csv", index=False)
pd.set_option("display.width", 240)
print(t[["DMS_id", "n", "n_mut_mean", "rho_zs", "ceil_max", "ceil_sum", "headroom_sum", "cstar_sum"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nmean over 14: zero-shot {t.rho_zs.mean():.4f}  ceiling(max) {t.ceil_max.mean():.4f}  "
      f"ceiling(sum) {t.ceil_sum.mean():.4f}")
