"""G2a: does the rescoring GAIN replicate across decoding seeds?

Section 2.2 of the record says the gain is a PAIRED comparison on one fixed score vector,
so the per-assay seed sigma is not its error bar, and the gain's own reproducibility had
never been measured. Seeds 2-5 exist for 10 assays, 5 of which are in the 14-assay set,
so it can be measured with no GPU.

Caveat stated up front: those 5 are at the WEAK end of the effect -- this bounds the
noise, it does not confirm the strong assays.
"""
import os, glob, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "binding_sites"))
from bg_common import parse_mut_dict

REC = "/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records"
WSREC = REC.replace("local-records", "workstation-records")
OUT = f"{WSREC}/mutation-landscape-TTT/data"
GRID = np.linspace(-4, 4, 161)
SEEDS = [1, 2, 3, 4, 5]
DIRS = {1: "/tmp/mpnn_scores_seed1_M5", **{s: f"/tmp/mpnn_scores_seed{s}_M5" for s in (2, 3, 4, 5)}}

lab = pd.read_parquet(f"{REC}/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet")
set14 = set(pd.read_csv(f"{OUT}/g1e_canonical14.csv").DMS_id)
common = sorted(set14 & {os.path.basename(f)[:-4] for f in glob.glob(f"{DIRS[2]}/*.csv")})
print(f"assays with all 5 seeds AND in the 14-assay set: {len(common)}")
for a in common: print("   ", a)

rows = []
for a in common:
    lv = lab[lab.DMS_id == a].reset_index(drop=True)
    d = lv.min_dist_to_partner.to_numpy(float)
    y = lv.DMS_score.to_numpy(float)
    w = np.exp(-d / 5.0); w = (w - w.mean()) / (w.std() or 1.0)      # E2 soft feature
    for s in SEEDS:
        sc = pd.read_csv(f"{DIRS[s]}/{a}.csv")
        is_wt = sc["mutant"].map(lambda x: all(not v.strip() for v in parse_mut_dict(x).values()))
        sv = sc[~is_wt].reset_index(drop=True)
        assert len(sv) == len(lv), (a, s, len(sv), len(lv))
        assert np.allclose(sv.DMS_score, y, atol=1e-9, equal_nan=True), f"{a} seed{s}: label mismatch"
        x = sv.global_score.to_numpy(float); sd = x.std() or 1.0
        base = stats.spearmanr(x, y).statistic
        curve = np.array([stats.spearmanr(x + c * sd * w, y).statistic for c in GRID])
        rows.append(dict(DMS_id=a, seed=s, rho_base=base, c_star=GRID[curve.argmax()],
                         rho_star=curve.max(), gain_star=curve.max() - base,
                         rho_fix1=curve[int(np.abs(GRID + 1).argmin())],
                         gain_fix1=curve[int(np.abs(GRID + 1).argmin())] - base))
t = pd.DataFrame(rows)
t.to_csv(f"{OUT}/g2a_seed_reproducibility.csv", index=False)
pd.set_option("display.width", 250)

print("\n=== per-assay across 5 seeds ===")
g = t.groupby("DMS_id").agg(base_mean=("rho_base", "mean"), base_sd=("rho_base", "std"),
                            gain_fix1_mean=("gain_fix1", "mean"), gain_fix1_sd=("gain_fix1", "std"),
                            gain_star_mean=("gain_star", "mean"), gain_star_sd=("gain_star", "std"),
                            cstar_mean=("c_star", "mean"), cstar_sd=("c_star", "std"))
print(g.to_string(float_format=lambda x: f"{x:+.4f}"))

print(f"\n=== the question: is the GAIN more stable than the BASELINE? ===")
print(f"  sd of rho_base   across seeds, median over {len(common)} assays : {g.base_sd.median():.4f}")
print(f"  sd of gain(c=-1) across seeds, median                          : {g.gain_fix1_sd.median():.4f}")
print(f"  ratio (gain sd / base sd)                                      : {(g.gain_fix1_sd/g.base_sd).median():.2f}")
print(f"\n  per-assay MEAN of gain(c=-1) over seeds: {g.gain_fix1_mean.mean():+.4f}"
      f"   (single-seed value used in the record: "
      f"{t.query('seed==1').gain_fix1.mean():+.4f})")
print(f"\n  c* across seeds: sd median {g.cstar_sd.median():.2f} sd-units"
      f"   (mean c* {g.cstar_mean.mean():+.2f})")
print(f"\n  NOTE these {len(common)} assays are the weak-effect end of the 14; "
      f"their mean gain is {g.gain_fix1_mean.mean():+.4f} vs {0.0555:+.4f} over all 14.")
