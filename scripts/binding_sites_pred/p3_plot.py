"""P3: the ProteinMPNN counterpart of the DMS figure.

Same layout, same panel order (sorted by the DMS Cliff's delta) so the two figures
can be read panel-by-panel against each other.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "binding_sites"))
from bg_common import REC_DIR, SHORT

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7.5,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7,
    "legend.frameon": False, "xtick.major.width": 0.7, "xtick.major.size": 2.5,
})
C_IFACE, C_NON = "#B64342", "#0F4D92"
PRED = os.path.normpath(os.path.join(REC_DIR, "..", "binding-sites-analysis-pred"))

v = pd.read_parquet(f"{PRED}/data/variant_labels_with_mpnn.parquet")
st = pd.read_csv(f"{PRED}/data/stats_dms_vs_mpnn.csv")
order = list(st.sort_values(["testable", "cliffs_delta_dms"], ascending=[False, True]).DMS_id)

fig, axes = plt.subplots(5, 5, figsize=(13.2, 12.4))
for ax, dms in zip(axes.ravel(), order):
    g = v[v.DMS_id == dms]
    a = g.loc[g["iface_dist_5.0"], "mpnn_score"].dropna().to_numpy()
    b = g.loc[~g["iface_dist_5.0"], "mpnn_score"].dropna().to_numpy()
    s = st[st.DMS_id == dms].iloc[0]
    pool = np.concatenate([a, b])
    lo, hi = np.percentile(pool, [0.5, 99.5])
    if hi <= lo: lo, hi = pool.min(), pool.max() + 1e-9
    edges = np.linspace(lo, hi, 61)
    for x, c in ((b, C_NON), (a, C_IFACE)):
        if len(x) < 10: continue
        ax.hist(np.clip(x, lo, hi), bins=edges, density=True, color=c, alpha=.55,
                edgecolor=c, linewidth=.35, zorder=2)
        ax.axvline(np.median(x), color=c, lw=1.1, zorder=3)
    lines = [(f"Spearman($\\rho$) = {s.rho_mpnn_vs_dms:.3f}", "#0F4D92")]   # the assay's benchmark metric
    if s.testable:
        # colour the delta by whether it agrees in sign with the DMS delta
        agree = np.sign(s.cliffs_delta_mpnn) == np.sign(s.cliffs_delta_dms)
        lines.append((f"$\\delta$ = {s.cliffs_delta_mpnn:+.2f}    OVL = {s.overlap_coef_mpnn:.2f}",
                      "#272727" if agree else "#B64342"))
        lines.append((f"($\\delta_{{DMS}}$ = {s.cliffs_delta_dms:+.2f})"
                      + ("" if agree else "   SIGN FLIP"), "#767676" if agree else "#B64342"))
    else:
        lines.append(("not comparable: " + ("all touch" if s.n_noniface == 0 else "none touch"), "#767676"))
        lines.append(("", "#767676"))
    lines.append((f"touches   n={int(s.n_iface):,}", C_IFACE))
    lines.append((f"does not  n={int(s.n_noniface):,}", C_NON))
    for k, (t, c) in enumerate(lines):
        ax.text(.975, .965 - k * .080, t, transform=ax.transAxes, ha="right", va="top",
                fontsize=6.2, color=c, zorder=4)
    ax.set_title(SHORT[dms], fontsize=8, pad=3, color="black" if s.testable else "#767676")
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=6.5, pad=1.5)
    ax.set_xlim(lo, hi); ax.margins(y=.72)
for ax in axes.ravel()[len(order):]:
    ax.axis("off")

fig.suptitle("BindingGYM — ProteinMPNN zero-shot score by whether a variant touches a binding site "
             "(same panel order as the DMS figure)", fontsize=10.5, y=.995)
fig.text(.5, .0125, "ProteinMPNN global_score, seed1 / M=5 (independent scale per panel; higher = more native-like)"
                    "  ·  panels sorted by the DMS Cliff's $\\delta$  ·  red = the MPNN $\\delta$ has the "
                    "OPPOSITE sign to the DMS $\\delta$", ha="center", fontsize=7.5, color="#4D4D4D")
fig.text(.5, .0005, "vertical lines = group medians  ·  OVL = overlap coefficient (1 = identical distributions)"
                    "  ·  a group with n < 10 is not drawn  ·  x-axis clipped to the 0.5-99.5 percentile",
         ha="center", fontsize=7.5, color="#4D4D4D")
fig.tight_layout(rect=[0, .028, 1, .985])
fig.savefig(f"{PRED}/fig_mpnn_distribution_by_interface.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{PRED}/fig_mpnn_distribution_by_interface.pdf", bbox_inches="tight",
            metadata={"CreationDate": None})
print("saved PNG + PDF")
