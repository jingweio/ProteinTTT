#!/usr/bin/env python
"""Aggregate ProteinMPNN complex/monomer scores and compare against ProteinGym's official baselines."""
import argparse, glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
ap.add_argument("--official", required=True, help="DMS_substitutions_Spearman_DMS_level.csv")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

rows = []
for f in sorted(glob.glob(os.path.join(a.scores, "*__*.csv"))):
    assay, cond = os.path.basename(f)[:-4].rsplit("__", 1)
    d = pd.read_csv(f)
    rows.append(dict(assay=assay, condition=cond, n=len(d),
                     rho_design=spearmanr(d.DMS_score, d.mpnn_design_ll).correlation,
                     rho_global=spearmanr(d.DMS_score, d.mpnn_global_ll).correlation))
s = pd.DataFrame(rows)
p = s.pivot(index="assay", columns="condition", values="rho_design")
p.columns = [f"MPNN_{c}" for c in p.columns]
p["n"] = s.groupby("assay").n.first()

off = pd.read_csv(a.official)
off.columns = [c.replace("\n", " ").strip() for c in off.columns]
off = off.set_index("DMS ID")
off = off.loc[[i for i in p.index if i in off.index]]

t = p.join(off, how="left")
t["delta_partner"] = t.MPNN_complex - t.MPNN_monomer
t["delta_vs_official"] = t.MPNN_complex - t["ProteinMPNN"]
cols = ["n", "ProteinMPNN", "MPNN_monomer", "MPNN_complex", "delta_partner", "delta_vs_official"]
t[cols].to_csv(os.path.join(a.out, "per_assay.csv"))
print("=== per-assay Spearman (target-chain score) ===")
print(t[cols].to_string(float_format=lambda x: f"{x:.4f}"))

print("\n=== means over the 9 assays ===")
print(f"  ProteinGym official ProteinMPNN : {t['ProteinMPNN'].mean():.4f}")
print(f"  ours, monomer (partner removed) : {t.MPNN_monomer.mean():.4f}")
print(f"  ours, complex (partner present) : {t.MPNN_complex.mean():.4f}")

d = (t.MPNN_complex - t.MPNN_monomer).dropna()
print(f"\n  partner effect: mean {d.mean():+.4f}  positive {int((d>0).sum())}/{len(d)}  "
      f"Wilcoxon p={wilcoxon(d).pvalue:.4f}")
d2 = (t.MPNN_complex - t["ProteinMPNN"]).dropna()
print(f"  vs official   : mean {d2.mean():+.4f}  positive {int((d2>0).sum())}/{len(d2)}  "
      f"Wilcoxon p={wilcoxon(d2).pvalue:.4f}")

# ---- decomposition: how much comes from the structure, how much from the partner
struct = (t.MPNN_monomer - t["ProteinMPNN"]).dropna()
print("\n=== decomposition (official -> monomer -> complex) ===")
print(f"  structure source (official -> monomer): mean {struct.mean():+.4f}  "
      f"positive {int((struct>0).sum())}/{len(struct)}  Wilcoxon p={wilcoxon(struct).pvalue:.4f}")
print(f"  partner          (monomer  -> complex): mean {d.mean():+.4f}  "
      f"positive {int((d>0).sum())}/{len(d)}  Wilcoxon p={wilcoxon(d).pvalue:.4f}")

# robustness of the partner effect
WEAK = ["DLG4_RAT_McLaughlin_2012", "YAP1_HUMAN_Araya_2012"]   # partner is a 5aa / 10aa peptide
SMALL = ["B2L11_HUMAN_Dutta_2010_binding-Mcl-1"]               # only 170 variants
print("\n=== partner effect, robustness ===")
for lab, keep in [("all 9", list(t.index)),
                  ("drop B2L11 (n=170)", [i for i in t.index if i not in SMALL]),
                  ("drop 5-10aa peptide partners", [i for i in t.index if i not in WEAK]),
                  ("drop both", [i for i in t.index if i not in WEAK + SMALL])]:
    dd = (t.loc[keep, "MPNN_complex"] - t.loc[keep, "MPNN_monomer"]).dropna()
    pv = wilcoxon(dd).pvalue if len(dd) > 2 else float("nan")
    print(f"  {lab:32s} n={len(dd)}  mean {dd.mean():+.4f}  "
          f"positive {int((dd>0).sum())}/{len(dd)}  Wilcoxon p={pv:.4f}")

# ranking of our variants among all official models on exactly these assays
# "Number of Mutants" is assay metadata, not a model -- it must not enter the leaderboard
num = off.select_dtypes("number").drop(columns=["Number of Mutants"], errors="ignore")
num = num.loc[:, num.notna().all()]
board = num.mean().to_frame("mean_rho")
for nm in ["MPNN_monomer", "MPNN_complex"]:
    board.loc[f"[ours] {nm}"] = t[nm].mean()
board = board.sort_values("mean_rho", ascending=False)
board["rank"] = range(1, len(board) + 1)
board.to_csv(os.path.join(a.out, "leaderboard.csv"))
print(f"\n=== leaderboard on these 9 assays ({len(board)} entries) ===")
print(board.head(15).to_string(float_format=lambda x: f"{x:.4f}"))
print("  ...")
print(board[board.index.str.startswith("[ours]") | (board.index == "ProteinMPNN")]
      .to_string(float_format=lambda x: f"{x:.4f}"))
