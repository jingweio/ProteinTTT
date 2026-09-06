"""G1: how much Spearman could a perfect binding-site-aware correction possibly buy?

BindingGYM's metrics are rank-based, and the interface label is binary and available
label-free at test time (it comes from the WT complex structure). So "make ProteinMPNN
aware of the binding site" can only RE-INTERLEAVE the two groups -- the within-group
order is whatever the model already produces.

That makes the ceiling computable. Give an oracle the true DMS scores and let it pick
the best possible interleaving; nothing that uses (score, interface label) can beat it.

Three levels, each with a size-matched RANDOM-label null. The null is the whole point:
an oracle maximising over interleavings extracts some gain from a meaningless label
too, and only the gap between the two is information the interface label actually
carries.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats

REC = "/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records"
OUT = f"{REC}/mutation-landscape-TTT/data"
EXCLUDE = ["KRAS_DARPinK27_norfitness_5O2S", "KRAS_SOS1_norfitness_8BE4"]  # see BindingGYM-issues
SUB_N, N_REP, SEED = 2000, 5, 0

def spearman(a, b):
    return stats.spearmanr(a, b).statistic

def best_shift(s, y, z, n_grid=201):
    """max over c of Spearman(s + c*z, y); c swept in units of s's spread, plus limits."""
    sd = s.std() or 1.0
    best = -2.0
    for c in np.concatenate([np.linspace(-4, 4, n_grid) * sd, [-1e9 * sd, 1e9 * sd]]):
        r = spearman(s + c * z, y)
        if r > best: best = r
    return best

def best_interleaving(s, y, z):
    """Exact max Spearman over merges that keep each group's own order (by s).

    dp over (i, j): position i+j is filled from group A or group B. Maximising
    sum(position * rank(y)) maximises Spearman, because the position vector is a
    permutation of 1..n whatever the merge.
    """
    r = stats.rankdata(y)
    a = r[z][np.argsort(s[z], kind="stable")] if z.any() else np.array([])
    b = r[~z][np.argsort(s[~z], kind="stable")] if (~z).any() else np.array([])
    na, nb = len(a), len(b)
    NEG = -np.inf
    prev = np.full(nb + 1, NEG); prev[0] = 0.0
    for j in range(1, nb + 1):                       # i = 0 row
        prev[j] = prev[j - 1] + j * b[j - 1]
    for i in range(1, na + 1):
        cur = np.full(nb + 1, NEG)
        cur[0] = prev[0] + i * a[i - 1]
        j = np.arange(1, nb + 1)
        # cur[j] = max(prev[j] + (i+j)*a[i-1],  cur[j-1] + (i+j)*b[j-1])  -> needs a scan
        take_a = prev[1:] + (i + j) * a[i - 1]
        add_b = (i + j) * b
        run = cur[0]
        out = np.empty(nb)
        for k in range(nb):                          # sequential because cur[j-1] is needed
            run = max(take_a[k], run + add_b[k])
            out[k] = run
        cur[1:] = out
        prev = cur
    n = na + nb
    pos = np.arange(1, n + 1)
    # recover Spearman from the maximised sum(pos*r)
    return (prev[nb] / n - pos.mean() * r.mean()) / (pos.std() * r.std())

def _check_ceiling(s, y, z):
    """The s-sorted merge is itself a legal interleaving, so the ceiling can never sit
    below the plain Spearman. Anything else means the DP or its indexing is wrong."""
    c, b = best_interleaving(s, y, z), spearman(s, y)
    assert c >= b - 1e-9, f"ceiling {c:.4f} < baseline {b:.4f} -- DP is wrong"
    return c

v = pd.read_parquet(f"{REC}/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet")
v = v[~v.DMS_id.isin(EXCLUDE)]
rng = np.random.default_rng(SEED)
rows = []
for dms, g in v.groupby("DMS_id"):
    s = g.mpnn_score.to_numpy(float); y = g.DMS_score.to_numpy(float)
    z = g["iface_dist_5.0"].to_numpy(bool)
    ok = np.isfinite(s) & np.isfinite(y); s, y, z = s[ok], y[ok], z[ok]
    if z.sum() < 30 or (~z).sum() < 30:
        rows.append(dict(DMS_id=dms, n=len(y), testable=False,
                         rho_base=spearman(s, y))); continue
    zr = rng.permutation(z)                                     # size-matched null
    r = dict(DMS_id=dms, n=len(y), testable=True, frac_iface=float(z.mean()),
             rho_base=spearman(s, y), rho_label_only=spearman(z.astype(float), y),
             rho_shift=best_shift(s, y, z.astype(float)),
             rho_shift_null=best_shift(s, y, zr.astype(float)))
    ceil_t, ceil_n, base_sub = [], [], []
    for rep in range(N_REP):
        idx = rng.choice(len(y), size=min(SUB_N, len(y)), replace=False)
        ss, yy, zz = s[idx], y[idx], z[idx]
        if zz.sum() < 5 or (~zz).sum() < 5: continue
        base_sub.append(spearman(ss, yy))
        ceil_t.append(_check_ceiling(ss, yy, zz))
        ceil_n.append(_check_ceiling(ss, yy, rng.permutation(zz)))
    r.update(rho_base_sub=np.mean(base_sub), rho_ceiling=np.mean(ceil_t),
             rho_ceiling_null=np.mean(ceil_n), n_rep=len(ceil_t))
    rows.append(r)

t = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
t.to_csv(f"{OUT}/g1_ceiling.csv", index=False)
k = t[t.testable].copy()
for c in ("rho_shift", "rho_ceiling"):
    k[f"gain_{c}"] = k[c] - k["rho_base"]
    k[f"gain_{c}_null"] = k[f"{c}_null"] - k["rho_base"]
pd.set_option("display.width", 260)
print(k[["DMS_id", "n", "frac_iface", "rho_base", "rho_label_only", "rho_shift",
         "rho_shift_null", "rho_base_sub", "rho_ceiling", "rho_ceiling_null"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\n{len(t)} assays kept ({len(k)} testable). baseline mean rho = {t.rho_base.mean():.4f}")
print(f"\n{'':34s}{'mean':>9s}{'vs base':>10s}{'null':>9s}{'net of null':>13s}")
for c, lab in (("rho_shift", "additive shift  s + c*z"),
               ("rho_ceiling", "best interleaving (CEILING)")):
    print(f"  {lab:32s}{k[c].mean():>9.4f}{k[f'gain_{c}'].mean():>+10.4f}"
          f"{k[f'{c}_null'].mean():>9.4f}{(k[c] - k[f'{c}_null']).mean():>+13.4f}")
print(f"  {'interface label ALONE':32s}{k.rho_label_only.mean():>9.4f}")
print(f"\n  sanity: subsample baseline {k.rho_base_sub.mean():.4f} vs full {k.rho_base.mean():.4f} "
      f"(should match; the ceiling is computed on the same subsamples)")
