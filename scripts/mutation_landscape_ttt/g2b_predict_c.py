"""G2b: can c be chosen at test time WITHOUT labels? (the pLDDT-analogue question)

The oracle c* is the per-target calibration complexTTT would have to supply. Ask two
things, both on the 14-assay set:
  Q1  do assays that share a target / a structure share a c*?
  Q2  does any LABEL-FREE feature of the WT complex or of the model's own scores predict
      c*, well enough to beat just using a constant c out of sample?

Q2 is the one that matters: a correlation is not enough -- a predictor only earns its
place if, leave-one-assay-out, predicted-c beats constant-c. Ten features are tested, so
p-values are BH-corrected and the LOAO comparison is the actual verdict.
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "binding_sites"))
from bg_common import load_index

REC = "/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records"
WSREC = REC.replace("local-records", "workstation-records")
OUT = f"{WSREC}/mutation-landscape-TTT/data"
GRID = np.linspace(-4, 4, 161)

lab = pd.read_parquet(f"{REC}/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet")
can = pd.read_csv(f"{OUT}/g1e_canonical14.csv")
idx = load_index().set_index("DMS_id")
sites = pd.read_csv(f"{REC}/binding-sites-analysis/data/binding_sites_per_chain.csv")

rows, curves = [], []
for a in can.DMS_id:
    g = lab[lab.DMS_id == a]
    s = g.mpnn_score.to_numpy(float); y = g.DMS_score.to_numpy(float)
    d = g.min_dist_to_partner.to_numpy(float); nm = g.n_mut.to_numpy(float)
    sd = s.std() or 1.0
    w = np.exp(-d / 5.0); w = (w - w.mean()) / (w.std() or 1.0)
    curves.append([stats.spearmanr(s + c * sd * w, y).statistic for c in GRID])
    st = sites[sites.DMS_id == a]
    n_site = sum(len(str(x).split(";")) for x in st.site_seqpos if isinstance(x, str) and x)
    L = int(st.n_res.sum())
    rows.append(dict(
        DMS_id=a, pdb=idx.loc[a, "pdb_file"], target=a.split("_")[0],
        # ---- label-free features: WT complex geometry ----
        f_mean_d=d.mean(), f_median_d=float(np.median(d)), f_std_d=d.std(),
        f_frac_iface=float((d <= 5).mean()), f_iface_frac_of_chain=n_site / L, f_L=float(L),
        # ---- label-free features: the model's own output ----
        f_sd_score=sd, f_corr_d_score=stats.spearmanr(d, s).statistic,
        # ---- label-free features: library shape ----
        f_n=float(len(g)), f_mean_nmut=nm.mean(),
    ))
t = pd.DataFrame(rows); C = np.array(curves); n = len(t)
t["c_star"] = GRID[C.argmax(axis=1)]
t["rho_base"] = can.set_index("DMS_id").loc[t.DMS_id, "rho_base"].to_numpy()
FEATS = [c for c in t.columns if c.startswith("f_")]
pd.set_option("display.width", 300)

print("=== Q1: 共享 target / 结构的 assay，c* 像不像？ ===")
for key in ("target", "pdb"):
    for k, grp in t.groupby(key):
        if len(grp) < 2: continue
        print(f"  {key}={k:16s} n={len(grp)}  c* = {[f'{v:+.2f}' for v in grp.c_star]}"
              f"   spread {grp.c_star.max()-grp.c_star.min():.2f} sd-units")
print(f"\n  全部 14 个 c* 的 spread: {t.c_star.max()-t.c_star.min():.2f} sd-units (sd {t.c_star.std():.2f})")

print(f"\n=== Q2a: {len(FEATS)} 个 label-free 特征与 c* 的相关（BH 校正） ===")
res = []
for f in FEATS:
    r = stats.spearmanr(t[f], t.c_star)
    res.append(dict(feature=f, rho=r.statistic, p=r.pvalue))
r = pd.DataFrame(res).sort_values("p")
m = len(r); r["q_BH"] = np.minimum.accumulate((r.p.values * m / (np.arange(m) + 1))[::-1])[::-1].clip(0, 1)
print(r.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

print(f"\n=== Q2b: 判据 —— LOAO 下，predicted-c 能不能打败 constant-c？ ===")
def loao_rho(cvals):
    return np.array([C[i, int(np.abs(GRID - cvals[i]).argmin())] for i in range(n)])
# baseline: constant c chosen on the other 13
const = np.array([GRID[C[np.arange(n) != i].mean(axis=0).argmax()] for i in range(n)])
out = [("constant c (LOAO)", loao_rho(const).mean())]
for f in FEATS:                                   # 1-feature linear fit on the other 13
    pred = []
    for i in range(n):
        m_ = np.arange(n) != i
        b, a0 = np.polyfit(t[f].values[m_], t.c_star.values[m_], 1)
        pred.append(np.clip(b * t[f].values[i] + a0, GRID[0], GRID[-1]))
    out.append((f"predicted c via {f}", loao_rho(np.array(pred)).mean()))
out.append(("oracle c* (upper bound)", C.max(axis=1).mean()))
base = t.rho_base.mean()
print(f"  {'method':34s}{'mean rho':>10s}{'gain':>9s}")
for name, v in out:
    print(f"  {name:34s}{v:>10.4f}{v-base:>+9.4f}")
print(f"\n  baseline (no correction) {base:.4f}")
best = max(out[1:-1], key=lambda x: x[1])
print(f"\n  最好的 predicted-c: {best[0]}  ->  {best[1]:.4f}"
      f"   vs constant-c {out[0][1]:.4f}   Δ = {best[1]-out[0][1]:+.4f}")
t.to_csv(f"{OUT}/g2b_predict_c.csv", index=False)

# ---------------------------------------------------------------------------
# Is c* even well-identified? A flat curve near the optimum makes argmax noise,
# and a "spread" between two assays then measures nothing.
# ---------------------------------------------------------------------------
print("\n=== Q1b: c* 被确定得有多好？（曲线在最优附近平不平） ===")
print(f"{'DMS_id':40s}{'c*':>7s}{'rho*':>8s}{'区间(rho>=rho*-0.002)':>24s}{'宽度':>7s}")
w95 = []
for i, a in enumerate(t.DMS_id):
    cur = C[i]; best = cur.max()
    ok = GRID[cur >= best - 0.002]
    lo, hi = ok.min(), ok.max()
    w95.append(hi - lo)
    print(f"{a:40s}{t.c_star.values[i]:>+7.2f}{best:>8.4f}{f'[{lo:+.2f}, {hi:+.2f}]':>24s}{hi-lo:>7.2f}")
t["c_plateau_width"] = w95
print(f"\n  平台宽度中位 {np.median(w95):.2f} sd-units；c* 的 assay 间 sd 是 {t.c_star.std():.2f}")
for key in ("target",):
    for k, grp in t.groupby(key):
        if len(grp) < 2: continue
        overlap_lo = max(GRID[C[i] >= C[i].max()-0.002].min() for i in grp.index)
        overlap_hi = min(GRID[C[i] >= C[i].max()-0.002].max() for i in grp.index)
        print(f"  {k:8s} n={len(grp)}  各自平台的交集 = "
              f"{'[%+.2f, %+.2f] 非空 ⇒ 一个共同 c 对全组都近最优' % (overlap_lo, overlap_hi) if overlap_hi>=overlap_lo else '空 ⇒ 真的不一致'}")
t.to_csv(f"{OUT}/g2b_predict_c.csv", index=False)
