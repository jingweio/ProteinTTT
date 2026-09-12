"""O4: two figures for the graded-bin section.

Global first, then detail (the reporting order the record follows):
  Figure 5  the trend itself -- one line per assay, truth beside prediction
  Figure 6  the per-assay distributions, K density curves per panel
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "binding_sites"))
from bg_common import REC_DIR, SHORT

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
                     "axes.spines.right": False, "axes.spines.top": False,
                     "axes.linewidth": .8, "legend.frameon": False})
C_FAR, C_NEAR = "#0F4D92", "#B64342"
RAMP = LinearSegmentedColormap.from_list("far2near", [C_FAR, "#B9A7E8", C_NEAR])
PRED = os.path.normpath(os.path.join(REC_DIR, "..", "binding-sites-analysis-pred"))
OUT = os.path.normpath(os.path.join(REC_DIR, "..", "binding-sites-overview"))

v = pd.read_parquet(f"{PRED}/data/variant_labels_with_mpnn.parquet")
t = pd.read_csv(f"{OUT}/data/graded_bins_per_assay.csv").query("usable")
pb = pd.read_csv(f"{OUT}/data/graded_bins_per_bin.csv")
order = list(t.sort_values("rho_bin_dms").DMS_id)

# ---------- Figure 5: the trend ----------
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
for ax, col, lab in ((axes[0], "dms_meanrank", "measured DMS_score"),
                     (axes[1], "mpnn_meanrank", "ProteinMPNN score")):
    curves = []
    for a in order:
        g = pb[pb.DMS_id == a].sort_values("bin")
        x = (g.bin.to_numpy() - 1) / (g.bin.max() - 1)
        ax.plot(x, g[col], color="#767676", lw=.9, alpha=.55, zorder=2)
        curves.append(np.interp(np.linspace(0, 1, 5), x, g[col].to_numpy()))
    med = np.median(np.array(curves), axis=0)
    ax.plot(np.linspace(0, 1, 5), med, color=C_NEAR if col.startswith("dms") else "#42949E",
            lw=3, zorder=4, label="median over assays")
    ax.axhline(0.5, color="#CFCECE", lw=.8, zorder=1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["B1\nfarthest", "B2", "B3", "B4", "B5\nclosest"], fontsize=7.5)
    ax.set_ylim(.2, .8); ax.set_ylabel("bin mean rank of the score  (0.5 = no effect)", fontsize=8.5)
    ax.set_title(f"{'A' if col.startswith('dms') else 'B'}   {lab}", fontsize=9.5, loc="left", pad=8)
    ax.legend(loc="lower left", fontsize=7.5)
    ax.text(.98, .96, f"median: {med[0]:.3f} at B1  $\\rightarrow$  {med[-1]:.3f} at B5"
                      f"   ($\\Delta$ = {med[-1]-med[0]:+.3f})",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#4D4D4D")
fig.suptitle("Graded binding-site bins — the measured effect is a gradient, the predicted one is flat",
             fontsize=10.5, y=1.0)
fig.text(.5, -.06, f"Bins are per-assay quantiles of d (distance from the variant's mutated positions to the "
                   f"other entity); B1 = farthest, B5 = closest.  One grey line per assay ({len(order)} usable), "
                   f"thick line = median.  Mean rank is scale-free, so assays are comparable.",
         ha="center", fontsize=7.6, color="#4D4D4D")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_graded_bins_trend.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/fig_graded_bins_trend.pdf", bbox_inches="tight", metadata={"CreationDate": None})

# ---------- Figure 6: per-assay distributions ----------
fig, axes = plt.subplots(4, 4, figsize=(13.0, 10.4))
for ax, a in zip(axes.ravel(), order):
    g = v[v.DMS_id == a]
    d = g.min_dist_to_partner.to_numpy(float); y = g.DMS_score.to_numpy(float)
    ok = np.isfinite(d) & np.isfinite(y); d, y = d[ok], y[ok]
    b = pd.qcut(-d, 5, labels=False, duplicates="drop"); nb = len(np.unique(b))
    lo, hi = np.percentile(y, [0.5, 99.5]); edges = np.linspace(lo, hi, 51)
    for i in range(nb):
        s = b == i
        ax.hist(np.clip(y[s], lo, hi), bins=edges, density=True, histtype="step",
                color=RAMP(i / max(nb - 1, 1)), lw=1.5, zorder=2 + i)
        ax.axvline(np.median(y[s]), color=RAMP(i / max(nb - 1, 1)), lw=1, ls=(0, (3, 2)), zorder=6)
    r = t[t.DMS_id == a].iloc[0]
    ax.set_title(SHORT[a] + ("  ★" if bool(r.excluded_for_ttt) else ""), fontsize=8.5, pad=3,
                 color="#767676" if bool(r.excluded_for_ttt) else "black")
    ax.text(.975, .96, f"$\\rho_{{bin}}$ = {r.rho_bin_dms:+.3f}\n$\\eta^2$ = {r.eta2_bin_dms:.3f}\nK = {nb}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color="#272727", linespacing=1.35)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=6.5); ax.set_xlim(lo, hi); ax.margins(y=.62)
for ax in axes.ravel()[len(order):]:
    ax.axis("off")
sm = plt.cm.ScalarMappable(cmap=RAMP, norm=mpl.colors.Normalize(0, 1))
cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=.018, pad=.045, aspect=60)
cb.set_ticks([0, 1]); cb.set_ticklabels(["B1  farthest from the other entity", "B5  closest"])
cb.ax.tick_params(labelsize=8)
fig.suptitle("Measured DMS_score by graded binding-site bin — one panel per assay, panels sorted by "
             "$\\rho_{bin}$  (★ = excluded from the complexTTT working set)", fontsize=10.5, y=.995)
fig.savefig(f"{OUT}/fig_graded_bins_per_assay.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/fig_graded_bins_per_assay.pdf", bbox_inches="tight", metadata={"CreationDate": None})
print("saved both figures")
