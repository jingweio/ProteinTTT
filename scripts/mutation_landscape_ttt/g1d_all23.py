"""G1d: the continuous feature works on ALL 23 assays, not just the 14.

G1/G1b/G1c reported on the 14 assays where the BINARY split has both groups non-empty.
That restriction is an artefact of the binary label: in the other 9 assays every variant
touches a binding site (or none does), so a 0/1 indicator is constant and buys nothing.
The continuous distance still varies in all 9, so the rescoring rule is defined on 23/23
-- which is also the set the published headline numbers are quoted on.
"""
import numpy as np, pandas as pd
from scipy import stats
REC = "/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records"
OUT = f"{REC}/mutation-landscape-TTT/data"
EXCL = ["KRAS_DARPinK27_norfitness_5O2S", "KRAS_SOS1_norfitness_8BE4"]
GRID = np.linspace(-4, 4, 161)

v = pd.read_parquet(f"{REC}/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet")
v = v[~v.DMS_id.isin(EXCL)]
ids, base, binary_ok, C = [], [], [], []
for dms, g in v.groupby("DMS_id"):
    s = g.mpnn_score.to_numpy(float); y = g.DMS_score.to_numpy(float)
    d = g.min_dist_to_partner.to_numpy(float)
    ok = np.isfinite(s) & np.isfinite(y) & np.isfinite(d); s, y, d = s[ok], y[ok], d[ok]
    z = d <= 5.0
    ids.append(dms); base.append(stats.spearmanr(s, y).statistic)
    binary_ok.append(bool(z.sum() >= 30 and (~z).sum() >= 30))
    w = np.exp(-d / 5.0); w = (w - w.mean()) / (w.std() or 1.0)
    sd = s.std() or 1.0
    C.append([stats.spearmanr(s + c * sd * w, y).statistic for c in GRID])
base = np.array(base); C = np.array(C); binary_ok = np.array(binary_ok); n = len(ids)

star = C.max(axis=1)
loao = np.array([C[i, C[np.arange(n) != i].mean(axis=0).argmax()] for i in range(n)])
fix1 = C[:, int(np.abs(GRID + 1.0).argmin())]
c_all = GRID[C.mean(axis=0).argmax()]

t = pd.DataFrame(dict(DMS_id=ids, binary_usable=binary_ok, rho_base=base,
                      rho_star=star, rho_loao=loao, rho_fixed1=fix1,
                      c_star=GRID[C.argmax(axis=1)]))
t["gain_loao"] = t.rho_loao - t.rho_base
pd.set_option("display.width", 250)
print(t.sort_values("gain_loao", ascending=False).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
t.to_csv(f"{OUT}/g1d_all23.csv", index=False)

def blk(m, lab):
    print(f"\n{lab}  (n={m.sum()})")
    print(f"  {'baseline':30s}{base[m].mean():.4f}")
    for arr, nm in ((star, "oracle per-assay c*"), (loao, "LOAO shared c"), (fix1, "fixed c = -1 (no fitting)")):
        print(f"  {nm:30s}{arr[m].mean():.4f}   ({arr[m].mean()-base[m].mean():+.4f})")
blk(np.ones(n, bool), "ALL 23 assays  << the set the published numbers use")
blk(binary_ok, "the 14 where the binary split is usable")
blk(~binary_ok, "the 9 the binary label cannot touch")
print(f"\nc* 符号: 负 {int((t.c_star<0).sum())}/{n}   共享 c (全 23) = {c_all:+.2f} sd")
