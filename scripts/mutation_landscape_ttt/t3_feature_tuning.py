"""Experiment 1: tune the interface feature w(.) before the TTT uses it.

tau = 5 A was inherited from the binary cutoff and has never been optimised, and the
aggregation over a variant's mutated sites was fixed to `min` only because that is what the
label table happened to store. Both are free parameters of the prior the TTT is about to be
trained on. Tuning them here means a bad TTT result cannot be blamed on an untuned prior.

Family:      w(v) = standardize_assay( AGG_i  f(d_i) )
Rule:        score' = score + c * sd(score) * w        (identical to the probe's rescoring rule)
Judged by:   per-assay Spearman, unweighted mean, on the 14-assay working set.

Selection over ~60 configs x 161 values of c overfits, so the same best-of is run on
distance-permuted data and reported as the null.
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy import stats
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
LREC = f"{ROOT}/local-records"
OUT = f"{ROOT}/workstation-records/mutation-landscape-TTT/data"
GRID = np.linspace(-4, 4, 161)
N_NULL = 3

# ---------------- feature family ----------------
TAUS = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
THETAS = [4.0, 4.5, 5.0, 6.0, 8.0]
FORMS = {}
for t in TAUS:
    FORMS[f"exp_tau{t:g}"] = (lambda d, t=t: np.exp(-d / t))
    FORMS[f"inv_tau{t:g}"] = (lambda d, t=t: 1.0 / (1.0 + d / t))
for th in THETAS:
    FORMS[f"bin_th{th:g}"] = (lambda d, th=th: (d <= th).astype(float))
FORMS["lin_negd"] = lambda d: -d
FORMS["neglogd"] = lambda d: -np.log(np.maximum(d, 1e-3))
AGGS = ["max", "mean", "sum"]          # f is decreasing in d, so max == f(min d) = the current feature
CONFIGS = [f"{a}|{f}" for a, f in itertools.product(AGGS, FORMS)]
ANCHOR = "max|exp_tau5"                # the probe's exact feature; must reproduce its numbers


def agg_feature(vals, idx, n, how):
    """vals: per-site f(d); idx: variant index of each site; n: #variants."""
    if how == "sum":
        return np.bincount(idx, weights=vals, minlength=n)
    if how == "mean":
        return np.bincount(idx, weights=vals, minlength=n) / np.bincount(idx, minlength=n)
    out = np.full(n, -np.inf)
    np.maximum.at(out, idx, vals)
    return out


def curves_for_assay(args):
    """Return (DMS_id, rho_base, {config: rho-vs-c curve}, {config: null curves})."""
    dms, s, y, dist, idx, n, seeds = args
    sd = s.std() or 1.0
    rho_base = stats.spearmanr(s, y).statistic
    ry = stats.rankdata(y); ry = (ry - ry.mean()) / np.linalg.norm(ry - ry.mean())

    def sweep(perm=None):
        """perm: variant permutation applied to w -- the null. Permuting the per-variant
        feature is exactly equivalent to permuting which variant owns which set of site
        distances, and standardisation is permutation-invariant."""
        out = {}
        for cfg in CONFIGS:
            how, fname = cfg.split("|")
            w = agg_feature(FORMS[fname](dist), idx, n, how)
            w = (w - w.mean()) / (w.std() or 1.0)
            if perm is not None: w = w[perm]
            cur = np.empty(len(GRID))
            for j, c in enumerate(GRID):
                rx = stats.rankdata(s + c * sd * w); rx -= rx.mean()
                cur[j] = rx @ ry / (np.linalg.norm(rx) or 1.0)
            out[cfg] = cur
        return out

    real = sweep()
    nulls = [sweep(np.random.default_rng(k).permutation(n)) for k in seeds]
    return dms, rho_base, real, nulls


def main():
    v = pd.read_parquet(f"{LREC}/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet")
    sd_tab = pd.read_parquet(f"{LREC}/binding-sites-analysis/data/variant_site_dists.parquet")
    set14 = list(pd.read_csv(f"{OUT}/g1e_canonical14.csv").DMS_id)

    jobs = []
    for a in set14:
        g = v[v.DMS_id == a].reset_index(drop=True)
        sdg = sd_tab[sd_tab.DMS_id == a]
        s, y = g.mpnn_score.to_numpy(float), g.DMS_score.to_numpy(float)
        ok = np.isfinite(s) & np.isfinite(y)
        assert ok.all(), f"{a}: non-finite score/label"          # the 14 are complete
        jobs.append((a, s, y, sdg.dist.to_numpy(float), sdg.vi.to_numpy(int), len(g),
                     list(range(1, N_NULL + 1))))

    with Pool(min(len(jobs), os.cpu_count() or 4)) as p:
        res = p.map(curves_for_assay, jobs)

    order = [r[0] for r in res]
    base = np.array([r[1] for r in res]); n = len(res)

    def summarise(pick):            # pick(i) -> {cfg: curve}
        rows = {}
        for cfg in CONFIGS:
            C = np.array([pick(i)[cfg] for i in range(n)])
            oracle = C.max(axis=1)
            loao = np.array([C[i, C[np.arange(n) != i].mean(axis=0).argmax()] for i in range(n)])
            rows[cfg] = dict(oracle=oracle.mean(), loao=loao.mean(),
                             cstar_med=float(np.median(GRID[C.argmax(axis=1)])))
        return rows

    real = summarise(lambda i: res[i][2])
    t = pd.DataFrame(real).T.rename_axis("config").reset_index().sort_values("loao", ascending=False)
    t["gain_loao"] = t.loao - base.mean()
    t["gain_oracle"] = t.oracle - base.mean()

    # null: identical best-of over configs and c, on permuted distances
    null_best = []
    for k in range(N_NULL):
        r = summarise(lambda i, k=k: res[i][3][k])
        null_best.append(dict(seed=k + 1,
                              loao=max(v_["loao"] for v_ in r.values()),
                              oracle=max(v_["oracle"] for v_ in r.values())))
    nb = pd.DataFrame(null_best)

    t.to_csv(f"{OUT}/t3_feature_tuning.csv", index=False)
    nb.to_csv(f"{OUT}/t3_feature_tuning_null.csv", index=False)

    anch = t[t.config == ANCHOR].iloc[0]
    print(f"baseline (14 assays)            {base.mean():.4f}")
    print(f"ANCHOR {ANCHOR:18s} LOAO {anch.loao:.4f}  oracle {anch.oracle:.4f}"
          f"   <- must match the probe's 0.4551 / 0.4689")
    print(f"\n=== top 12 of {len(CONFIGS)} configs, by LOAO ===")
    print(t.head(12).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n=== permuted-distance null (same best-of over {len(CONFIGS)} configs x {len(GRID)} c) ===")
    print(nb.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    best = t.iloc[0]
    print(f"\nbest LOAO {best.config}: {best.loao:.4f}  (null best {nb.loao.mean():.4f},"
          f" net {best.loao - nb.loao.mean():+.4f})")
    json.dump(dict(best_config=best.config, best_loao=float(best.loao),
                   best_oracle=float(best.oracle), anchor_loao=float(anch.loao),
                   anchor_oracle=float(anch.oracle), baseline=float(base.mean()),
                   null_loao_mean=float(nb.loao.mean()), assays=order),
              open(f"{OUT}/t3_best.json", "w"), indent=1)


if __name__ == "__main__":
    main()
