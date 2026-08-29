"""S8: per-assay DMS_score distributions, binding-site-touching vs not.

Core conclusion the figure must defend: the interface-touching group is shifted
low in essentially every assay, but the two distributions overlap heavily --
the interface label is a weak monotone stratifier, not a separable dichotomy.
Archetype: quantitative grid (25 small multiples, one per assay).
"""
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 7.5, "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.7, "legend.frameon": False,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})
C_IFACE, C_NON = "#B64342", "#0F4D92"          # signal family / neutral-baseline family
D = "/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-analysis"

SHORT = {
 "4D5_HER2_fitness_1N8Z":"4D5 – HER2", "5A12_Ang2_fitness_4ZFG":"5A12 – Ang2",
 "5A12_VEGF_fitness_4ZFF":"5A12 – VEGF", "Z-domain_ZpA963_HL1_fitness_2M5A":"Z-dom ZpA963 HL1",
 "Z-domain_ZpA963_HL2_fitness_2M5A":"Z-dom ZpA963 HL2", "Z-domain_ZSPA-1_LL1_fitness_1LP1":"Z-dom ZSPA-1 LL1",
 "Z-domain_ZSPA-1_LL2_fitness_1LP1":"Z-dom ZSPA-1 LL2", "CXCR4_CXCL12_enrich_8U4O":"CXCR4 – CXCL12",
 "hYAP65_peptide_FunctioncalScore_1JMQ":"hYAP65 – peptide", "GB1_IgG-Fc_fitness_1FCC":"GB1 – IgG-Fc",
 "GB1_IgG-Fc_fitness_1FCC_2016":"GB1 – IgG-Fc (2016)", "SARS2-RBD_ACE2_deltaKd_6M0J":"SARS2-RBD – ACE2",
 "KRAS_DARPinK27_norfitness_5O2S":"KRAS – DARPin K27", "KRAS_PICK3CG-RBD_norfitness_1HE8":"KRAS – PI3KCG-RBD",
 "KRAS_RAF1_norfitness_6VJJ":"KRAS – RAF1", "KRAS_RAF1-RBD_norfitness_6VJJ":"KRAS – RAF1-RBD",
 "KRAS_RALGDS-RBD_norfitness_1LFD":"KRAS – RALGDS-RBD", "KRAS_SOS1_norfitness_8BE4":"KRAS – SOS1",
 "BH3_Mcl-1_normed_3KZ0":"BH3 – Mcl-1", "BH3_Bcl-xL_normed_1PQ1":"BH3 – Bcl-xL",
 "HLA-A2_TAPBPR_meanscore_5WER":"HLA-A2 – TAPBPR", "PSD95_CRIPT_1BE9":"PSD95 – CRIPT",
 "PSD95_Tm2F_1BE9":"PSD95 – Tm2F", "ACE2_SARS2-RBD_enrich_6M17":"ACE2 – SARS2-RBD",
 "CD19_FMC63_Fitness_7URV":"CD19 – FMC63"}

v = pd.read_parquet(f"{D}/data/variant_labels.parquet")
st = pd.read_csv(f"{D}/data/stats_iface_vs_noniface_extended.csv")
wt = pd.read_csv(f"{D}/data/wt_reference.csv").set_index("DMS_id")["wt_score"].to_dict()
order = list(st.sort_values(["testable", "cliffs_delta"], ascending=[False, True]).DMS_id)

fig, axes = plt.subplots(5, 5, figsize=(13.2, 12.4))
for ax, dms in zip(axes.ravel(), order):
    g = v[v.DMS_id == dms]
    a = g.loc[g["iface_dist_5.0"], "DMS_score"].dropna().to_numpy()
    b = g.loc[~g["iface_dist_5.0"], "DMS_score"].dropna().to_numpy()
    s = st[st.DMS_id == dms].iloc[0]
    pool = np.concatenate([a, b])
    lo, hi = np.percentile(pool, [0.5, 99.5])
    if hi <= lo: lo, hi = pool.min(), pool.max() + 1e-9
    w = wt.get(dms, np.nan)
    if np.isfinite(w):                       # always keep the WT anchor inside the window
        pad = .03 * (hi - lo)
        lo, hi = min(lo, w - pad), max(hi, w + pad)
    edges = np.linspace(lo, hi, 61)
    MIN_DRAW = 10          # a density histogram of <10 points is noise blown up to full height
    for x, c, lab in ((b, C_NON, "non-interface"), (a, C_IFACE, "interface")):
        if len(x) < MIN_DRAW: continue
        ax.hist(np.clip(x, lo, hi), bins=edges, density=True, color=c, alpha=.55,
                edgecolor=c, linewidth=.35, label=lab, zorder=2)
        ax.axvline(np.median(x), color=c, lw=1.1, ls="-", zorder=3)
    if np.isfinite(w):                                   # wild-type anchor
        ax.axvline(w, color="#272727", lw=1.0, ls=(0, (3, 2)), zorder=5)
        ax.annotate("WT", xy=(w, .66), xycoords=("data", "axes fraction"),
                    ha="center", va="center", fontsize=6, color="#272727", zorder=6,
                    bbox=dict(fc="white", ec="none", pad=.8, alpha=.85))
    else:
        ax.annotate("no WT row", xy=(.5, .60), xycoords="axes fraction", ha="center",
                    va="center", fontsize=6.5, color="#767676", style="italic", zorder=6,
                    bbox=dict(fc="white", ec="none", pad=1.2, alpha=.85))
    ax.set_title(SHORT[dms], fontsize=8, pad=3,
                 color="black" if s.testable else "#767676")
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=6.5, pad=1.5)
    ax.set_xlim(lo, hi)
    # direct colour-coded labels instead of a legend
    lines = []
    if s.testable:
        lines.append((f"$\\delta$ = {s.cliffs_delta:.2f}    OVL = {s.overlap_coef:.2f}", "#272727"))
    else:
        lines.append(("not comparable: " + ("all touch" if s.n_noniface == 0 else "none touch"), "#767676"))
    lines.append((f"touches   n={int(s.n_iface):,}" +
                  (f"  mean {s.mean_iface:.2f}" if np.isfinite(s.mean_iface) else ""), C_IFACE))
    lines.append((f"does not  n={int(s.n_noniface):,}" +
                  (f"  mean {s.mean_noniface:.2f}" if np.isfinite(s.mean_noniface) else ""), C_NON))
    for k, (t, c) in enumerate(lines):
        ax.text(.975, .965 - k * .092, t, transform=ax.transAxes, ha="right", va="top",
                fontsize=6.2, color=c, zorder=4)
    ax.margins(y=.55)
for ax in axes.ravel()[len(order):]:
    ax.axis("off")

fig.suptitle("BindingGYM — DMS_score distribution by whether a variant touches a binding site "
             "(density-normalised per assay; vertical lines = group medians)", fontsize=10.5, y=.995)
fig.text(.5, .0125, "DMS_score (independent scale per panel; higher = better binding)  ·  panels sorted by Cliff's "
                    "$\\delta$; grey title = one group empty, not comparable  ·  black dashed line = wild-type score",
         ha="center", fontsize=7.5, color="#4D4D4D")
fig.text(.5, .0005, "OVL = overlap coefficient (1 = identical distributions)  ·  a group with n < 10 is not drawn  ·  "
                    "x-axis clipped to the 0.5-99.5 percentile, then widened so the WT anchor is always inside  ·  3 assays ship no WT row",
         ha="center", fontsize=7.5, color="#4D4D4D")
fig.tight_layout(rect=[0, .028, 1, .985])
fig.savefig(f"{D}/fig_dms_distribution_by_interface.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{D}/fig_dms_distribution_by_interface.pdf", bbox_inches="tight")
print("saved PNG + PDF")
