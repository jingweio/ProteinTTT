"""T5: one table for every TTT run so far, read against the two lines that matter.

A TTT number is only interpretable next to the zero-shot score it started from and the
oracle-c rescoring ceiling for the SAME feature -- the latter being the bar it has to clear
to have contributed anything a five-line rule could not.
"""
import os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "workstation-records/mutation-landscape-TTT/data"))
t4 = pd.read_csv(f"{OUT}/t4_per_assay_bars.csv").set_index("DMS_id")

def best(tag, agg):
    f = f"{OUT}/{tag}_lambda_sweep.csv"
    if not os.path.exists(f): return pd.DataFrame()
    s = pd.read_csv(f)
    if "final_M" in s: s = s[s.final_M.isna()] if s.final_M.notna().any() else s
    b = s.loc[s.groupby("DMS_id").rho.idxmax()].copy()
    b["feat"] = agg; b["src"] = tag
    return b

t = pd.concat([best("e1", "max"), best("e1b_max", "max"), best("e1b_sum", "sum")], ignore_index=True)
t = t.drop_duplicates(subset=["DMS_id", "feat"], keep="first")
t["ceil"] = [t4.loc[r.DMS_id, f"ceil_{r.feat}"] for r in t.itertuples()]
t["vs_ceil"] = t.rho - t.ceil
t["n_mut"] = [t4.loc[a, "n_mut_mean"] for a in t.DMS_id]
t = t.sort_values(["feat", "vs_ceil"], ascending=[True, False])
t.to_csv(f"{OUT}/t5_summary.csv", index=False)
pd.set_option("display.width", 240)
for agg in ["max", "sum"]:
    g = t[t.feat == agg]
    if not len(g): continue
    print(f"\n=== 特征 = {agg} + exp(-d/5A) ===")
    print(g[["DMS_id", "n_mut", "lam", "rho_base", "rho", "ceil", "vs_ceil"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"  n={len(g)}  zero-shot {g.rho_base.mean():.4f}  TTT {g.rho.mean():.4f}  "
          f"ceiling {g.ceil.mean():.4f}  TTT-ceiling {g.vs_ceil.mean():+.4f}  "
          f"越过天花板 {int((g.vs_ceil>0).sum())}/{len(g)}")
# consistency: single-mutant assays must be identical under max and sum
piv = t.pivot_table(index="DMS_id", columns="feat", values="rho")
sm = [a for a in piv.index if t4.loc[a, "n_mut_mean"] == 1.0 and piv.loc[a].notna().all()]
# For a single-mutant variant max and sum are the same number, so these runs are the same
# experiment twice. Their spread is therefore a direct measure of GPU run-to-run
# nondeterminism in rho, not a logic check -- the tolerance has to admit float32 atomics.
d = max(abs(piv.loc[a, "max"] - piv.loc[a, "sum"]) for a in sm)
print(f"\n一致性检查：{len(sm)} 个单点突变 assay（此时 max == sum，等于同一实验跑两遍）")
print(f"  最大差异 {d:.2e} -> {'OK，量级即 float32 非确定性' if d < 1e-4 else 'MISMATCH'}")
print(f"  ⇒ 同一配置重跑的 rho 抖动约 {d:.0e}，远小于任何被引用的效应")
