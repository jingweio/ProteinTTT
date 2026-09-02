"""P5: head-to-head -- does the ProteinMPNN interface contrast look like the DMS one?"""
import os, sys
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "binding_sites"))
from bg_common import REC_DIR, SHORT

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
                     "axes.spines.right": False, "axes.spines.top": False,
                     "axes.linewidth": .8, "legend.frameon": False})
PRED = os.path.normpath(os.path.join(REC_DIR, "..", "binding-sites-analysis-pred"))
t = pd.read_csv(f"{PRED}/data/stats_dms_vs_mpnn.csv").query("testable")
t["is_kras"] = t.DMS_id.str.startswith("KRAS")
C_K, C_O, C_BAD = "#B64342", "#0F4D92", "#F6CFCB"      # A: KRAS / other / sign-flip band
C_MEAS, C_PRED  = "#272727", "#42949E"                  # B: measurement / prediction
# manual label offsets for the crowded cluster (dx, dy, ha)
LBL = {"KRAS_RAF1-RBD_norfitness_6VJJ": (-6, 5, "right"),
       "KRAS_DARPinK27_norfitness_5O2S": (6, 5, "left"),
       "KRAS_PICK3CG-RBD_norfitness_1HE8": (6, -7, "left"),
       "KRAS_RAF1_norfitness_6VJJ": (-6, -1.5, "right"),
       "CXCR4_CXCL12_enrich_8U4O": (6, -1.5, "left"),
       "KRAS_SOS1_norfitness_8BE4": (6, -6, "left"),
       "GB1_IgG-Fc_fitness_1FCC": (-6, 3, "right"),
       "PSD95_Tm2F_1BE9": (-6, 4, "right"),
       "HLA-A2_TAPBPR_meanscore_5WER": (6, -4, "left"),
       "ACE2_SARS2-RBD_enrich_6M17": (-6, 4, "right"),
       "PSD95_CRIPT_1BE9": (-6, -5, "right"),
       "SARS2-RBD_ACE2_deltaKd_6M0J": (6, -1.5, "left")}

fig = plt.figure(figsize=(12.4, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1], wspace=.26)

# --- A: delta_dms vs delta_mpnn -------------------------------------------------
ax = fig.add_subplot(gs[0])
ax.axhspan(0, .32, xmin=0, xmax=1, color=C_BAD, alpha=.45, zorder=0)
ax.text(-.755, .315, "sign flip: DMS says interface mutations are\nworse, ProteinMPNN says they are better",
        fontsize=7.2, color="#8A3330", va="top", ha="left")
lim = (-.78, .34)
ax.plot(lim, lim, ls=(0, (4, 3)), lw=.9, color="#767676", zorder=1)
ax.axhline(0, lw=.8, color="#272727", zorder=1)
for _, r in t.iterrows():
    c = C_K if r.is_kras else C_O
    ax.scatter(r.cliffs_delta_dms, r.cliffs_delta_mpnn, s=42, color=c, zorder=3,
               edgecolor="white", linewidth=.6)
    dx, dy, ha = LBL.get(r.DMS_id, (6, -1.5, "left"))
    ax.annotate(SHORT[r.DMS_id], (r.cliffs_delta_dms, r.cliffs_delta_mpnn),
                xytext=(dx, dy), textcoords="offset points", fontsize=6.1, color=c,
                zorder=3, ha=ha)
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel("Cliff's $\\delta$ on the measured DMS_score", fontsize=8.5)
ax.set_ylabel("Cliff's $\\delta$ on the ProteinMPNN score", fontsize=8.5)
rho = stats.spearmanr(t.cliffs_delta_dms, t.cliffs_delta_mpnn)
ax.set_title("A   Per-assay interface contrast: prediction vs measurement", fontsize=9, loc="left", pad=8)
ax.text(.02, .04, f"Spearman = {rho.statistic:+.3f}  (p = {rho.pvalue:.2f}, n = {len(t)})\n"
                  f"dashed = identity;  {int((t.cliffs_delta_mpnn.abs() < t.cliffs_delta_dms.abs()).sum())}/{len(t)} "
                  f"attenuated,  {int((t.cliffs_delta_mpnn > 0).sum())}/{len(t)} sign-flipped",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
ax.scatter([], [], s=42, color=C_K, label="KRAS assays (6)"); ax.scatter([], [], s=42, color=C_O, label="other (10)")
ax.legend(loc="lower right", fontsize=7.2, handletextpad=.3,
          bbox_to_anchor=(1.0, .12))

# --- B: paired effect-size / overlap / variance ---------------------------------
ax2 = fig.add_subplot(gs[1])
metrics = [("|Cliff's $\\delta$|", t.cliffs_delta_dms.abs(), t.cliffs_delta_mpnn.abs()),
           ("overlap coef. OVL", t.overlap_coef_dms, t.overlap_coef_mpnn),
           ("$\\eta^2$ explained", t.eta2_dms, t.eta2_mpnn)]
for k, (name, a, b) in enumerate(metrics):
    x0, x1 = k * 1.0 - .21, k * 1.0 + .21
    for u, v in zip(a, b):
        ax2.plot([x0, x1], [u, v], color="#767676", lw=.7, alpha=.55, zorder=2)
    ax2.scatter([x0] * len(a), a, s=26, color=C_MEAS, zorder=3, edgecolor="white", linewidth=.5)
    ax2.scatter([x1] * len(b), b, s=26, color=C_PRED, zorder=3, edgecolor="white", linewidth=.5)
    ax2.plot([x0 - .11, x0 + .11], [a.median()] * 2, color=C_MEAS, lw=2.4, zorder=4)
    ax2.plot([x1 - .11, x1 + .11], [b.median()] * 2, color=C_PRED, lw=2.4, zorder=4)
    ax2.text(k, .975, f"{a.median():.3f} → {b.median():.3f}", ha="center", fontsize=7.2, color="#272727")
    ax2.text(x0, -.075, "meas.", ha="center", fontsize=6.8, color=C_MEAS)
    ax2.text(x1, -.075, "pred.", ha="center", fontsize=6.8, color=C_PRED)
ax2.set_xticks(range(len(metrics))); ax2.set_xticklabels([m[0] for m in metrics], fontsize=8)
ax2.set_xlim(-.5, len(metrics) - .5); ax2.set_ylim(-.10, 1.03)
ax2.axhline(0, lw=.7, color="#CFCECE", zorder=1)
ax2.set_ylabel("value (16 testable assays)", fontsize=8.5)
ax2.set_title("B   Same statistic on the measurement (left) and the prediction (right)",
              fontsize=9, loc="left", pad=8)
ax2.text(.5, -.155, "black = measured DMS_score   ·   teal = ProteinMPNN score   ·   thick bar = median"
         "   ·   number above = median shift", transform=ax2.transAxes, ha="center",
         fontsize=7, color="#4D4D4D")

fig.suptitle("BindingGYM — the binding-site contrast the DMS measures is not the contrast ProteinMPNN predicts",
             fontsize=10.5, y=1.005)
fig.text(.5, -.035, "$\\delta$ < 0 = variants touching a binding site score lower.  ProteinMPNN: zero-shot "
                    "global_score, seed1 / M=5, scored on the WT complex.  Binding site = heavy atom within "
                    "5 Å of a never-mutated partner chain.", ha="center", fontsize=7.4, color="#4D4D4D")
fig.savefig(f"{PRED}/fig_dms_vs_mpnn_contrast.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{PRED}/fig_dms_vs_mpnn_contrast.pdf", bbox_inches="tight", metadata={"CreationDate": None})
print("saved")
