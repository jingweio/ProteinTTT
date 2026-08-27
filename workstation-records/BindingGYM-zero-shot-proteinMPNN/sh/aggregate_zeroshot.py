"""S4: 聚合 BindingGYM zero-shot ProteinMPNN 的逐 assay 分数，与官方发布值比对。

指标口径来自 refs/bindinggym_metrics.py —— 逐字移植官方
calc_metric.ipynb::calc_zero_shot_metric(top_test=False)。
聚合是先按 assay 算、再对 assay 取未加权均值（官方口径）。
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "refs"))
from bindinggym_metrics import bindinggym_metrics_one_assay  # noqa: E402

METRICS = ["Spearman", "AUC", "MCC", "NDCG", "AP"]


def one_run(score_dir, pred_col):
    rows = []
    for f in sorted(glob.glob(os.path.join(score_dir, "*.csv"))):
        df = pd.read_csv(f)
        if pred_col not in df.columns:
            continue
        m = bindinggym_metrics_one_assay(df, pred_col=pred_col, label_col="DMS_score")
        rows.append(dict(DMS_id=os.path.basename(f)[:-4], n=len(df), **m))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores_root", required=True)
    p.add_argument("--refs", required=True, help="dir holding ProteinMPNN_zero_shot_metric.csv")
    p.add_argument("--pred_col", default="global_score")
    p.add_argument("--out_csv", default=None)
    a = p.parse_args()

    ref = pd.read_csv(os.path.join(a.refs, "ProteinMPNN_zero_shot_metric.csv"))
    prior = pd.read_csv(os.path.join(a.refs, "prior_run_M1_per_assay.csv"))

    runs = {}
    for d in sorted(glob.glob(os.path.join(a.scores_root, "seed*_M*"))):
        t = one_run(d, a.pred_col)
        if not t.empty:
            runs[os.path.basename(d)] = t

    print(f"官方参考（25 assay 未加权均值）: " + "  ".join(
        f"{m}={ref[m].mean():.6f}" for m in METRICS if m in ref.columns))
    print(f"上次 M=1 复现: Spearman={prior.our_S.mean():.6f}  (官方 {prior.off_S.mean():.6f}, Δ={prior.our_S.mean()-prior.off_S.mean():+.6f})")
    print()
    print(f"{'run':<16}{'n_assay':>8}" + "".join(f"{m:>12}" for m in METRICS))
    for k, t in runs.items():
        print(f"{k:<16}{len(t):>8}" + "".join(f"{t[m].mean():>12.6f}" for m in METRICS))

    full = {k: t for k, t in runs.items() if len(t) == len(ref)}
    if full:
        print("\n=== 逐 assay 比对（仅覆盖全部 25 个 assay 的 run）===")
        for k, t in full.items():
            m = t[["DMS_id", "Spearman"]].merge(
                ref[["DMS_id", "Spearman"]], on="DMS_id", suffixes=("_ours", "_ref"))
            m["delta"] = m.Spearman_ours - m.Spearman_ref
            print(f"\n{k}:  mean {m.Spearman_ours.mean():.6f}  vs ref {m.Spearman_ref.mean():.6f}"
                  f"   Δ={m.delta.mean():+.6f}")
            print(f"   逐 assay |Δ|: mean {m.delta.abs().mean():.4f}  max {m.delta.abs().max():.4f}"
                  f"   >0.05: {(m.delta.abs()>0.05).sum()}/{len(m)}   Δ<0: {(m.delta<0).sum()}/{len(m)}")
            print(f"   偏差最大的 3 个:")
            for _, r in m.reindex(m.delta.abs().sort_values(ascending=False).index).head(3).iterrows():
                print(f"     {r.DMS_id:<40} ours={r.Spearman_ours:.4f} ref={r.Spearman_ref:.4f} Δ={r.delta:+.4f}")

    # seed 方差（在多 seed 都覆盖的 assay 上）
    seeds = {k: t for k, t in runs.items() if k.endswith("_M5")}
    if len(seeds) > 1:
        common = set.intersection(*[set(t.DMS_id) for t in seeds.values()])
        if common:
            print(f"\n=== seed 方差（{len(seeds)} 个 seed 共同覆盖的 {len(common)} 个 assay）===")
            piv = pd.DataFrame({k: t.set_index("DMS_id").loc[sorted(common), "Spearman"]
                                for k, t in seeds.items()})
            piv["sd"] = piv.std(axis=1, ddof=1)
            piv["range"] = piv.iloc[:, :len(seeds)].max(axis=1) - piv.iloc[:, :len(seeds)].min(axis=1)
            print(piv.round(4).to_string())
            print(f"\n逐 assay sd: median {piv.sd.median():.4f}  max {piv.sd.max():.4f}")
            sub = piv[[k for k in seeds]].mean(axis=0)
            print(f"子集均值逐 seed: " + "  ".join(f"{k}={v:.6f}" for k, v in sub.items()))
            print(f"子集均值的 sd = {sub.std(ddof=1):.6f}   range = {sub.max()-sub.min():.6f}")

    if a.out_csv:
        pd.concat([t.assign(run=k) for k, t in runs.items()]).to_csv(a.out_csv, index=False)
        print(f"\nwrote {a.out_csv}")


if __name__ == "__main__":
    main()
