#!/usr/bin/env python
"""汇总各 run 的逐 variant 分数 -> 逐 assay 指标 -> leaderboard。

指标口径【逐字复用】anchor 那轮已验证的 refs/bindinggym_metrics.py(移植自 BindingGYM 官方),
不重新实现。输出 Spearman / AUC / MCC / NDCG / AP 五项及 25-assay 均值。

打分列按 run 自动识别:design_score(MPNN 家族) / laser_score / adflip_score / stabddg_pred。
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd

SCORE_COLS = ["design_score", "laser_score", "adflip_score", "stabddg_pred"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_root", required=True)
    ap.add_argument("--metrics_py", required=True, help="refs/bindinggym_metrics.py")
    ap.add_argument("--index", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--official", default="", help="refs/ProteinMPNN_zero_shot_metric.csv")
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(a.metrics_py)))
    import bindinggym_metrics as BM
    # 逐字复用 anchor 那轮已验证的官方口径函数,不重新实现
    calc = BM.bindinggym_metrics_one_assay
    ids = pd.read_csv(a.index)["DMS_id"].tolist()
    os.makedirs(a.out_dir, exist_ok=True)

    rows = []
    for d in sorted(glob.glob(os.path.join(a.scores_root, "*/"))):
        run = os.path.basename(d.rstrip("/"))
        if run.startswith("smoke"): continue
        per = []
        for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
            df = pd.read_csv(f)
            col = next((c for c in SCORE_COLS if c in df.columns), None)
            if col is None: continue
            m = dict(calc(df, pred_col=col, label_col="DMS_score"))
            m["DMS_id"] = os.path.basename(f)[:-4]; m["n"] = len(df)
            per.append(m)
        if not per: continue
        pdf = pd.DataFrame(per).set_index("DMS_id")
        pdf.to_csv(os.path.join(a.out_dir, f"per_assay_{run}.csv"))
        num = pdf.select_dtypes(include=[np.number]).drop(columns=["n"], errors="ignore")
        r = num.mean().to_dict(); r["run_id"] = run; r["n_assay"] = len(pdf)
        r["missing"] = ",".join(i for i in ids if i not in pdf.index) or "-"
        rows.append(r)

    lb = pd.DataFrame(rows)
    front = ["run_id", "n_assay"] + [c for c in lb.columns if c not in ("run_id", "n_assay", "missing")] + ["missing"]
    lb = lb[front].sort_values(lb.columns[2], ascending=False)
    lb.to_csv(os.path.join(a.out_dir, "leaderboard.csv"), index=False)
    print(lb.to_string(index=False))

    if a.official and os.path.exists(a.official):
        o = pd.read_csv(a.official)
        sp = [c for c in o.columns if "spearman" in c.lower()]
        if sp:
            print(f"\n官方发布的 ProteinMPNN(25-assay 均值 {sp[0]}): {o[sp[0]].mean():.6f}")
            print("anchor 那轮复现: 0.391356")


if __name__ == "__main__":
    main()
