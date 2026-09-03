"""O1: three example assays, ground truth on top and ProteinMPNN below.

Read each column downwards: same assay, same variant split, measured vs predicted.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "binding_sites"))
from bg_common import REC_DIR, SHORT

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
                     "axes.spines.right": False, "axes.spines.top": False,
                     "axes.linewidth": .8, "legend.frameon": False})
C_IFACE, C_NON = "#B64342", "#0F4D92"
PRED = os.path.normpath(os.path.join(REC_DIR, "..", "binding-sites-analysis-pred"))
OUT = os.path.normpath(os.path.join(REC_DIR, "..", "binding-sites-overview"))
os.makedirs(OUT, exist_ok=True)

EXAMPLES = ["KRAS_RALGDS-RBD_norfitness_1LFD", "KRAS_RAF1-RBD_norfitness_6VJJ",
            "ACE2_SARS2-RBD_enrich_6M17"]
v = pd.read_parquet(f"{PRED}/data/variant_labels_with_mpnn.parquet")
st = pd.read_csv(f"{PRED}/data/stats_dms_vs_mpnn.csv").set_index("DMS_id")

fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.4))
for col, dms in enumerate(EXAMPLES):
    g = v[v.DMS_id == dms]; s = st.loc[dms]
    for row, (score_col, tag) in enumerate([("DMS_score", "dms"), ("mpnn_score", "mpnn")]):
        ax = axes[row, col]
        a = g.loc[g["iface_dist_5.0"], score_col].dropna().to_numpy()
        b = g.loc[~g["iface_dist_5.0"], score_col].dropna().to_numpy()
        pool = np.concatenate([a, b]); lo, hi = np.percentile(pool, [0.5, 99.5])
        edges = np.linspace(lo, hi, 55)
        for x, c, lab in ((b, C_NON, "does not touch"), (a, C_IFACE, "touches a binding site")):
            ax.hist(np.clip(x, lo, hi), bins=edges, density=True, color=c, alpha=.55,
                    edgecolor=c, linewidth=.35, label=lab, zorder=2)
            ax.axvline(np.median(x), color=c, lw=1.2, zorder=3)
        d, o = s[f"cliffs_delta_{tag}"], s[f"overlap_coef_{tag}"]
        flip = np.sign(s.cliffs_delta_mpnn) != np.sign(s.cliffs_delta_dms)
        col_d = "#B64342" if (tag == "mpnn" and flip) else "#272727"
        ax.text(.975, .96, f"$\\delta$ = {d:+.3f}\nOVL = {o:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=col_d, linespacing=1.4, zorder=4)
        if tag == "mpnn" and flip:
            ax.text(.975, .70, "SIGN FLIP", transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color="#B64342", zorder=4)
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_xlim(lo, hi); ax.margins(y=.34)
        ax.tick_params(axis="x", labelsize=7)
        if row == 0:
            ax.set_title(f"{SHORT[dms]}\n" + r"$\rho$ = " + f"{s.rho_mpnn_vs_dms:.3f}",
                         fontsize=9.5, pad=5)
        ax.set_xlabel("measured DMS_score" if row == 0 else "ProteinMPNN score", fontsize=8)
        if col == 0:
            ax.text(-.06, .5, "ground truth\n(DMS)" if row == 0 else "prediction\n(ProteinMPNN)",
                    transform=ax.transAxes, ha="right", va="center", fontsize=9.5,
                    rotation=90, linespacing=1.4)
axes[0, 0].legend(loc="upper left", fontsize=7.4, handlelength=1.1, handleheight=.8)
fig.suptitle("Same assay, same variant split — the contrast the DMS measures (top) "
             "vs the one ProteinMPNN predicts (bottom)", fontsize=10.5, y=.995)
fig.text(.5, -.02, "Density-normalised per panel; vertical lines = group medians.  "
                   "$\\delta$ = Cliff's $\\delta$ (0 = indistinguishable, <0 = the touching group scores lower);  "
                   "OVL = overlap coefficient (1 = identical);  $\\rho$ = that assay's benchmark Spearman.",
         ha="center", fontsize=7.6, color="#4D4D4D")
fig.tight_layout(rect=[.015, 0, 1, .97])
fig.savefig(f"{OUT}/fig_examples_dms_vs_mpnn.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/fig_examples_dms_vs_mpnn.pdf", bbox_inches="tight", metadata={"CreationDate": None})
print("saved", f"{OUT}/fig_examples_dms_vs_mpnn.png")
