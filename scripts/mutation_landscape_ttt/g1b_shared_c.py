"""G1b: does the correction have to be per-target, or would one global constant do?

G1 showed a simple additive shift  s + c*z  captures essentially the whole ceiling.
That makes the design question concrete: c is one scalar per assay, so either
  - one shared c works for every target  -> no test-time adaptation is needed, a fixed
    prior would do, and complexTTT is not motivated; or
  - c* varies per target -> estimating it without labels IS the method.
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats

REC = "/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records"
OUT = f"{REC}/mutation-landscape-TTT/data"
EXCLUDE = ["KRAS_DARPinK27_norfitness_5O2S", "KRAS_SOS1_norfitness_8BE4"]
GRID = np.linspace(-4, 4, 161)                      # c in units of the assay's score sd

v = pd.read_parquet(f"{REC}/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet")
v = v[~v.DMS_id.isin(EXCLUDE)]
curves, rows = {}, []
for dms, g in v.groupby("DMS_id"):
    s = g.mpnn_score.to_numpy(float); y = g.DMS_score.to_numpy(float)
    z = g["iface_dist_5.0"].to_numpy(bool).astype(float)
    ok = np.isfinite(s) & np.isfinite(y); s, y, z = s[ok], y[ok], z[ok]
    if z.sum() < 30 or (len(z) - z.sum()) < 30: continue
    sd = s.std() or 1.0
    curve = np.array([stats.spearmanr(s + c * sd * z, y).statistic for c in GRID])
    curves[dms] = curve
    rows.append(dict(DMS_id=dms, rho_base=stats.spearmanr(s, y).statistic,
                     c_star=GRID[curve.argmax()], rho_star=curve.max()))
t = pd.DataFrame(rows); C = np.vstack([curves[d] for d in t.DMS_id])
t["gain_star"] = t.rho_star - t.rho_base

# one shared c, chosen to maximise the 14-assay mean
mean_curve = C.mean(axis=0)
c_shared = GRID[mean_curve.argmax()]
t["rho_shared"] = C[:, mean_curve.argmax()]
t["gain_shared"] = t.rho_shared - t.rho_base
# and the sign-only version: a fixed -1 sd shift
i_fix = int(np.abs(GRID - (-1.0)).argmin())
t["rho_fixed1"] = C[:, i_fix]; t["gain_fixed1"] = t.rho_fixed1 - t.rho_base

pd.set_option("display.width", 250)
print(t[["DMS_id", "rho_base", "c_star", "rho_star", "gain_star",
         "rho_shared", "gain_shared", "gain_fixed1"]]
      .sort_values("gain_star", ascending=False).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print(f"\n c* 的符号: 负 {int((t.c_star<0).sum())}/{len(t)}, 正 {int((t.c_star>0).sum())}, 零 {int((t.c_star==0).sum())}")
print(f" c* 范围 [{t.c_star.min():+.2f}, {t.c_star.max():+.2f}] sd,  中位 {t.c_star.median():+.2f} sd")
print(f"\n{'':38s}{'mean rho':>10s}{'gain':>9s}")
print(f"  {'baseline':36s}{t.rho_base.mean():>10.4f}{0:>+9.4f}")
print(f"  {'per-assay oracle c*':36s}{t.rho_star.mean():>10.4f}{t.gain_star.mean():>+9.4f}")
print(f"  {f'ONE shared c = {c_shared:+.2f} sd':36s}{t.rho_shared.mean():>10.4f}{t.gain_shared.mean():>+9.4f}")
print(f"  {'fixed c = -1.00 sd (sign only)':36s}{t.rho_fixed1.mean():>10.4f}{t.gain_fixed1.mean():>+9.4f}")
print(f"\n  => 共享常数拿到 per-assay oracle 的 {100*t.gain_shared.mean()/t.gain_star.mean():.0f}%；"
      f"剩下 {100*(1-t.gain_shared.mean()/t.gain_star.mean()):.0f}% 才是「必须逐 target 估计」的部分")
t.to_csv(f"{OUT}/g1b_shared_c.csv", index=False)
np.save(f"{OUT}/g1b_curves.npy", C); np.save(f"{OUT}/g1b_grid.npy", GRID)
