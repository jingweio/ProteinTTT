"""G1e: ONE authoritative table on the 14-assay basis.

Everything the record quotes is regenerated here from the raw per-variant data, so no
number in the document is transcribed. The 14-assay basis is fixed for the whole
complexTTT design phase (user decision 2026-09-10).
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats

REC = "/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records"
WSREC = REC.replace("local-records", "workstation-records")
OUT = f"{WSREC}/mutation-landscape-TTT/data"
EXCL = ["KRAS_DARPinK27_norfitness_5O2S", "KRAS_SOS1_norfitness_8BE4"]
GRID = np.linspace(-4, 4, 161)
FEATS = {"binary": lambda d: (d <= 5.0).astype(float),
         "soft":   lambda d: np.exp(-d / 5.0)}

v = pd.read_parquet(f"{REC}/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet")

# ---- the three assay sets, derived not asserted ----
all25 = sorted(v.DMS_id.unique())
set23 = [a for a in all25 if a not in EXCL]
def usable14(a):
    g = v[v.DMS_id == a]; z = g["iface_dist_5.0"].to_numpy(bool)
    return z.sum() >= 30 and (~z).sum() >= 30
set14 = [a for a in set23 if usable14(a)]
print(f"25 shipped: {len(all25)}   23 after excluding the 2 corrupted: {len(set23)}   "
      f"14 with a usable binary split: {len(set14)}")
print("dropped from 23 -> 14:")
for a in set23:
    if a in set14: continue
    g = v[v.DMS_id == a]; z = g["iface_dist_5.0"].to_numpy(bool)
    print(f"   {a:40s} n_touch={int(z.sum()):>6d} n_not={int((~z).sum()):>6d}")

# ---- per-assay curves on the 14 ----
rows, curves = [], {k: [] for k in FEATS}
for a in set14:
    g = v[v.DMS_id == a]
    s = g.mpnn_score.to_numpy(float); y = g.DMS_score.to_numpy(float)
    d = g.min_dist_to_partner.to_numpy(float)
    ok = np.isfinite(s) & np.isfinite(y) & np.isfinite(d); s, y, d = s[ok], y[ok], d[ok]
    sd = s.std() or 1.0
    rows.append(dict(DMS_id=a, n=len(y), sd_mpnn=sd,
                     frac_touch=float((d <= 5).mean()), rho_base=stats.spearmanr(s, y).statistic))
    for k, f in FEATS.items():
        w = f(d); w = (w - w.mean()) / (w.std() or 1.0)
        curves[k].append([stats.spearmanr(s + c * sd * w, y).statistic for c in GRID])
t = pd.DataFrame(rows); n = len(t)
i_fix = int(np.abs(GRID + 1.0).argmin())
for k in FEATS:
    C = np.array(curves[k])
    t[f"{k}_cstar"] = GRID[C.argmax(axis=1)]
    t[f"{k}_oracle"] = C.max(axis=1)
    t[f"{k}_loao"] = [C[i, C[np.arange(n) != i].mean(axis=0).argmax()] for i in range(n)]
    t[f"{k}_fixed1"] = C[:, i_fix]
# ceiling + null come from g1_ceiling.csv (the interleaving DP), restricted to the 14
ce = pd.read_csv(f"{OUT}/g1_ceiling.csv").set_index("DMS_id")
t["binary_ceiling"] = [ce.loc[a, "rho_ceiling"] for a in t.DMS_id]
t["binary_ceiling_null"] = [ce.loc[a, "rho_ceiling_null"] for a in t.DMS_id]
t["rho_base_sub"] = [ce.loc[a, "rho_base_sub"] for a in t.DMS_id]
t.to_csv(f"{OUT}/g1e_canonical14.csv", index=False)

pd.set_option("display.width", 300)
print("\n=== per-assay (14) ===")
print(t[["DMS_id", "n", "rho_base", "binary_cstar", "binary_loao", "soft_cstar", "soft_loao"]]
      .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print(f"\n=== 14-assay summary (baseline {t.rho_base.mean():.4f}) ===")
print(f"{'':44s}{'mean rho':>10s}{'gain':>9s}")
for lab, col in [("E1 binary, oracle per-assay c*", "binary_oracle"),
                 ("E1 binary, LOAO shared c", "binary_loao"),
                 ("E1 binary, fixed c = -1", "binary_fixed1"),
                 ("E2 soft exp(-d/5A), oracle per-assay c*", "soft_oracle"),
                 ("E2 soft exp(-d/5A), LOAO shared c", "soft_loao"),
                 ("E2 soft exp(-d/5A), fixed c = -1", "soft_fixed1")]:
    print(f"  {lab:42s}{t[col].mean():>10.4f}{t[col].mean()-t.rho_base.mean():>+9.4f}")
print(f"\n  E0 binary ceiling (interleaving DP)      {t.binary_ceiling.mean():.4f}"
      f"{t.binary_ceiling.mean()-t.rho_base_sub.mean():>+9.4f}   [on 2000-variant subsamples]")
print(f"  E0 same DP on a RANDOM label (null)      {t.binary_ceiling_null.mean():.4f}"
      f"{t.binary_ceiling_null.mean()-t.rho_base_sub.mean():>+9.4f}")
print(f"  E0 net of null                                     "
      f"{t.binary_ceiling.mean()-t.binary_ceiling_null.mean():>+9.4f}")
print(f"\n  c* sign: binary neg {int((t.binary_cstar<0).sum())}/{n}, soft neg {int((t.soft_cstar<0).sum())}/{n}")
print(f"  sd(mpnn_score) across the 14: {t.sd_mpnn.min():.2f} .. {t.sd_mpnn.max():.2f}"
      f"  ({t.sd_mpnn.max()/t.sd_mpnn.min():.1f}x)")
