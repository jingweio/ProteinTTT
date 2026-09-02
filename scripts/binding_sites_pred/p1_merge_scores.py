"""P1: merge ProteinMPNN per-variant zero-shot scores onto the interface labels.

Three gates before anything is written:
  (a) row counts of the score CSVs match the shipped DMS CSVs
  (b) after filtering the WT row the same way, DMS_score agrees elementwise with
      variant_labels.parquet -> the positional join is proven, not assumed
  (c) per-assay Spearman recomputed from these scores reproduces the recorded
      seed1_M5 run -> we are using the right column and orientation
"""
import os, sys, ast, glob
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "binding_sites"))
from bg_common import OUT_DIR, DMS_DIR, parse_mut_dict

# Per-variant scores live on the workstation (329 MB, deliberately not in the repo).
# They are the 2026-08-27 seed1/M=5 run recorded on the proteinTTT-proteinGYM-reproduce
# branch; nothing here re-runs the GPU. Pulled on demand into a local cache.
WS = "guoj0f@10.67.24.41:/data/guoj0f/BindingGYM-zero-shot-proteinMPNN/scores/seed1_M5/"
SCORES = os.environ.get("MPNN_SCORES_DIR", "/tmp/mpnn_scores_seed1_M5")
if not (os.path.isdir(SCORES) and len(glob.glob(f"{SCORES}/*.csv")) == 25):
    os.makedirs(SCORES, exist_ok=True)
    print(f"fetching per-variant scores -> {SCORES}")
    if os.system(f"rsync -a {WS} {SCORES}/") != 0 or len(glob.glob(f"{SCORES}/*.csv")) != 25:
        raise SystemExit(f"could not obtain the 25 score CSVs; rsync manually:\n  rsync -a {WS} {SCORES}/")
REF = ("/home/guoj0f/repos/ProteinTTT/.claude/worktrees/proteinTTT-proteinGYM-reproduce/"
       "workstation-records/BindingGYM-zero-shot-proteinMPNN/results/per_assay_all_runs.csv")
PRED_OUT = os.path.join(os.path.dirname(OUT_DIR), "..", "binding-sites-analysis-pred", "data")
PRED_OUT = os.path.normpath(PRED_OUT)

lab = pd.read_parquet(f"{OUT_DIR}/variant_labels.parquet")
ref = pd.read_csv(REF).query("run == 'seed1_M5'").set_index("DMS_id")

rows, checks = [], []
for f in sorted(glob.glob(f"{SCORES}/*.csv")):
    dms = os.path.basename(f)[:-4]
    sc = pd.read_csv(f)
    src = pd.read_csv(os.path.join(DMS_DIR, f"{dms}.csv"))
    n_src = len(src)
    # (c) recompute the official metric on ALL rows (the official harness keeps the WT row)
    rho_now = sc["DMS_score"].rank().corr(sc["global_score"].rank())
    rho_ref = float(ref.loc[dms, "Spearman"])
    ident = bool(np.allclose(sc["design_score"], sc["global_score"], atol=1e-9))
    # (b) drop WT the same way the label table did, then align positionally
    is_wt = sc["mutant"].map(lambda s: all(not v.strip() for v in parse_mut_dict(s).values()))
    sv = sc[~is_wt].reset_index(drop=True)
    lv = lab[lab.DMS_id == dms].reset_index(drop=True)
    ok_n = len(sv) == len(lv)
    ok_lab = ok_n and bool(np.allclose(sv["DMS_score"], lv["DMS_score"], atol=1e-9, equal_nan=True))
    checks.append(dict(DMS_id=dms, n_score=len(sc), n_src=n_src, rows_match=len(sc) == n_src,
                       n_var_score=len(sv), n_var_label=len(lv), align_ok=ok_lab,
                       rho_now=rho_now, rho_ref=rho_ref, d_rho=abs(rho_now - rho_ref),
                       design_eq_global=ident))
    if ok_lab:
        out = lv.copy()
        out["mpnn_score"] = sv["global_score"].to_numpy()
        rows.append(out)

c = pd.DataFrame(checks)
pd.set_option("display.width", 250)
print(c[["DMS_id", "n_score", "rows_match", "n_var_score", "n_var_label", "align_ok",
         "rho_now", "rho_ref", "d_rho", "design_eq_global"]]
      .to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print(f"\n(a) rows_match        {c.rows_match.sum()}/25")
print(f"(b) align_ok          {c.align_ok.sum()}/25")
print(f"(c) max |d_rho|       {c.d_rho.max():.2e}   mean rho recomputed {c.rho_now.mean():.6f}  recorded {c.rho_ref.mean():.6f}")
print(f"    design==global    {c.design_eq_global.sum()}/25")
assert c.rows_match.all() and c.align_ok.all() and c.d_rho.max() < 1e-9, "GATE FAILED"

m = pd.concat(rows, ignore_index=True)
os.makedirs(PRED_OUT, exist_ok=True)
m.to_parquet(f"{PRED_OUT}/variant_labels_with_mpnn.parquet")
c.to_csv(f"{PRED_OUT}/merge_gates.csv", index=False)
print(f"\nWROTE {PRED_OUT}/variant_labels_with_mpnn.parquet  {m.shape}")
