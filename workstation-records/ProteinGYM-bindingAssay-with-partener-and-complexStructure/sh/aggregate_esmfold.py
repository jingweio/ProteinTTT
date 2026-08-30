#!/usr/bin/env python
"""Four-way comparison on the 9 ProteinGym binding assays.

  (a) ProteinGym leaderboard top models
  (b) ProteinGym's official ProteinMPNN
  (c) ours, ProteinMPNN on the ESMFold2-predicted MONOMER
  (d) ours, ProteinMPNN on the ESMFold2-predicted COMPLEX (target + full partner)
"""
import argparse, glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
ap.add_argument("--official", required=True)
ap.add_argument("--crystal-scores", default="", help="optional: earlier BindingGYM-crystal run")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

def collect(d, suffix):
    r = {}
    for f in glob.glob(os.path.join(d, "*__*.csv")):
        assay, cond = os.path.basename(f)[:-4].rsplit("__", 1)
        t = pd.read_csv(f)
        r.setdefault(assay, {})[f"{cond}{suffix}"] = spearmanr(t.DMS_score, t.mpnn_design_ll).correlation
        r[assay]["n"] = len(t)
    return r

rec = collect(a.scores, "")
t = pd.DataFrame(rec).T.sort_index()
if a.crystal_scores and os.path.isdir(a.crystal_scores):
    c = pd.DataFrame(collect(a.crystal_scores, "_crystal")).T
    t = t.join(c.drop(columns=[x for x in c.columns if x == "n"]), how="left")

off = pd.read_csv(a.official)
off.columns = [c.replace("\n", " ").strip() for c in off.columns]
off = off.set_index("DMS ID").drop(columns=["Number of Mutants"], errors="ignore")
t = t.join(off, how="left")

cols = ["n", "ProteinMPNN", "esmfold_monomer", "esmfold_complex"]
cols = [c for c in cols if c in t.columns]
t["delta_partner"] = t.get("esmfold_complex") - t.get("esmfold_monomer")
t["delta_vs_official"] = t.get("esmfold_complex") - t["ProteinMPNN"]
show = cols + [c for c in ("monomer_crystal", "complex_crystal") if c in t.columns] + \
       ["delta_partner", "delta_vs_official"]
t[show].to_csv(os.path.join(a.out, "per_assay_esmfold.csv"))
pd.set_option("display.width", 250)
print("=== per-assay Spearman (target-chain score) ===")
print(t[show].to_string(float_format=lambda x: f"{x:.4f}"))

print("\n=== means over the 9 assays ===")
for c in show:
    if c == "n":
        continue
    print(f"  {c:22s} {t[c].mean():+.4f}")

for lab, x, y in [("partner effect (complex - monomer)", "esmfold_complex", "esmfold_monomer"),
                  ("vs official ProteinMPNN", "esmfold_complex", "ProteinMPNN"),
                  ("monomer vs official", "esmfold_monomer", "ProteinMPNN")]:
    if x in t and y in t:
        d = (t[x] - t[y]).dropna()
        p = wilcoxon(d).pvalue if len(d) > 2 else float("nan")
        print(f"  {lab:36s} mean {d.mean():+.4f}  positive {int((d>0).sum())}/{len(d)}  Wilcoxon p={p:.4f}")

num = off.select_dtypes("number"); num = num.loc[:, num.notna().all()]
board = num.mean().to_frame("mean_rho")
for c in ("esmfold_monomer", "esmfold_complex", "monomer_crystal", "complex_crystal"):
    if c in t:
        board.loc[f"[ours] {c}"] = t[c].mean()
board = board.sort_values("mean_rho", ascending=False)
board["rank"] = range(1, len(board) + 1)
board.to_csv(os.path.join(a.out, "leaderboard_esmfold.csv"))
print(f"\n=== leaderboard on these 9 assays ({len(board)} entries) ===")
print(board.head(12).to_string(float_format=lambda x: f"{x:.4f}"))
print("  ...")
print(board[board.index.str.startswith("[ours]") | (board.index == "ProteinMPNN")]
      .to_string(float_format=lambda x: f"{x:.4f}"))
