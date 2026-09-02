"""P4: why does the MPNN delta not track the DMS delta?

Hypothesis: ProteinMPNN's confidence tracks local packing density, so its
interface-vs-non-interface contrast is driven by BURIAL, not by partner contact.
Test it with burial computed from the mutated chain ALONE (partner deleted), so the
measure carries no partner information at all.
"""
import os, sys, difflib
import numpy as np, pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "binding_sites"))
from bg_common import *

PRED = os.path.normpath(os.path.join(REC_DIR, "..", "binding-sites-analysis-pred"))
CB_CUT = 10.0

def rep_atom(coords, names):
    """CB if present, else CA, else centroid -- the usual neighbour-count anchor."""
    for want in ("CB", "CA"):
        if want in names: return coords[names.index(want)]
    return coords.mean(axis=0)

def parse_named(path):
    ch, order = {}, {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"): continue
            el, nm = line[76:78].strip().upper(), line[12:16].strip()
            if el == "H" or (el == "" and nm.startswith("H")): continue
            c = line[21]; rid = line[22:26].strip() + line[26].strip()
            d = ch.setdefault(c, {})
            if rid not in d: d[rid] = [line[17:20].strip(), [], []]; order.setdefault(c, []).append(rid)
            d[rid][1].append((float(line[30:38]), float(line[38:46]), float(line[46:54]))); d[rid][2].append(nm)
    out = {}
    for c, d in ch.items():
        rs = order[c]
        out[c] = dict(resids=rs, seq="".join(THREE2ONE.get(d[k][0], "X") for k in rs),
                      rep=np.array([rep_atom(np.array(d[k][1]), d[k][2]) for k in rs]))
    return out

idx = load_index()
lab = pd.read_parquet(f"{PRED}/data/variant_labels_with_mpnn.parquet")
st = pd.read_csv(f"{PRED}/data/stats_dms_vs_mpnn.csv").set_index("DMS_id")
rows = []
for _, r in idx.iterrows():
    dms, ws = r["DMS_id"], parse_mut_dict(r["wildtype_sequence"])
    pdb = parse_named(os.path.join(PDB_DIR, r["pdb_file"]))
    df = pd.read_csv(os.path.join(DMS_DIR, r["DMS_filename"]))
    parsed = [parse_mut_dict(s) for s in df["mutant"]]
    mut_ch = sorted({c for md in parsed for c, v in md.items() if v.strip()})
    # intra-chain neighbour count: partner chains are DELETED, so this is partner-free
    burial, maps = {}, {}
    for c in mut_ch:
        p = pdb[c]; t = cKDTree(p["rep"])
        burial[c] = np.array([len(t.query_ball_point(x, CB_CUT)) - 1 for x in p["rep"]])
        sm = difflib.SequenceMatcher(a=p["seq"], b=ws[c], autojunk=False)
        maps[c] = {j + k + 1: i + k for i, j, n in sm.get_matching_blocks() for k in range(n)}
    # library positions, split by whether they are a binding site (from the record's table)
    sites = pd.read_csv(f"{OUT_DIR}/binding_sites_per_chain.csv").query("DMS_id == @dms")
    bi, bn = [], []
    for _, s in sites.iterrows():
        c = s["chain"]
        site = {int(x) for x in str(s["site_seqpos"]).split(";") if x and x != "nan"}
        libp = {int(x) for x in str(s["lib_pos"]).split(";") if x and x != "nan"}
        for q in libp:
            (bi if q in site else bn).append(burial[c][maps[c][q]])
    d = dict(DMS_id=dms, n_lib_site=len(bi), n_lib_nonsite=len(bn),
             burial_site=np.mean(bi) if bi else np.nan,
             burial_nonsite=np.mean(bn) if bn else np.nan)
    d["d_burial"] = d["burial_site"] - d["burial_nonsite"]
    if dms in st.index and st.loc[dms, "testable"]:
        d["delta_dms"] = st.loc[dms, "cliffs_delta_dms"]; d["delta_mpnn"] = st.loc[dms, "cliffs_delta_mpnn"]
    rows.append(d)

b = pd.DataFrame(rows)
b.to_csv(f"{PRED}/data/burial_contrast.csv", index=False)
pd.set_option("display.width", 250)
t = b.dropna(subset=["delta_dms", "d_burial"])
print(t[["DMS_id", "n_lib_site", "n_lib_nonsite", "burial_site", "burial_nonsite",
         "d_burial", "delta_dms", "delta_mpnn"]].sort_values("d_burial")
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nn = {len(t)} assays with both a burial contrast and a testable delta")
for tag in ("mpnn", "dms"):
    s = stats.spearmanr(t.d_burial, t[f"delta_{tag}"])
    print(f"  Spearman(d_burial, delta_{tag:4s}) = {s.statistic:+.3f}   p = {s.pvalue:.4f}")
s = stats.spearmanr(t.delta_dms, t.delta_mpnn)
print(f"  Spearman(delta_dms,  delta_mpnn)   = {s.statistic:+.3f}   p = {s.pvalue:.4f}")
print(f"\n  assays where the binding site is LESS buried than the rest of the library "
      f"(d_burial < 0): {int((t.d_burial < 0).sum())}/{len(t)}")
print(f"  of those, MPNN delta > 0 (interface looks MORE tolerated): "
      f"{int(((t.d_burial < 0) & (t.delta_mpnn > 0)).sum())}/{int((t.d_burial < 0).sum())}")

# ---------------------------------------------------------------------------
# Variant-level burial: stratify on it and see whether the interface label keeps
# any signal in the MPNN score once burial is matched. n=16 assays is underpowered
# for the assay-level correlation above; this runs inside each assay instead.
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("burial-stratified delta  (per variant: mean intra-chain neighbour count of its mutated positions)")
print("=" * 100)

pv = []
for _, r in idx.iterrows():
    dms, ws = r["DMS_id"], parse_mut_dict(r["wildtype_sequence"])
    pdb = parse_named(os.path.join(PDB_DIR, r["pdb_file"]))
    df = pd.read_csv(os.path.join(DMS_DIR, r["DMS_filename"]))
    parsed = [parse_mut_dict(s) for s in df["mutant"]]
    mut_ch = sorted({c for md in parsed for c, v in md.items() if v.strip()})
    burial, maps = {}, {}
    for c in mut_ch:
        p = pdb[c]; t = cKDTree(p["rep"])
        burial[c] = np.array([len(t.query_ball_point(x, CB_CUT)) - 1 for x in p["rep"]])
        sm = difflib.SequenceMatcher(a=p["seq"], b=ws[c], autojunk=False)
        maps[c] = {j + k + 1: i + k for i, j, n in sm.get_matching_blocks() for k in range(n)}
    vals = []
    for md in parsed:
        bs = [burial[c][maps[c][int(ps)]] for c, v in md.items() if v.strip() for _, ps, _ in muts_of(v)]
        vals.append(np.mean(bs) if bs else np.nan)
    keep = ~np.isnan(vals)
    pv.append(pd.DataFrame(dict(DMS_id=dms, burial=np.asarray(vals)[keep])))
pvb = pd.concat(pv, ignore_index=True)
assert len(pvb) == len(lab), (len(pvb), len(lab))
lab = lab.reset_index(drop=True); lab["burial"] = pvb["burial"].to_numpy()
lab.to_parquet(f"{PRED}/data/variant_labels_with_mpnn.parquet")

def delta(a, b):
    if len(a) < 10 or len(b) < 10: return np.nan, np.nan
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 2 * U / (len(a) * len(b)) - 1, p

out = []
for dms, g in lab.groupby("DMS_id"):
    s = st.loc[dms]
    if not s["testable"]: continue
    row = dict(DMS_id=dms, delta_dms=s["cliffs_delta_dms"], delta_mpnn=s["cliffs_delta_mpnn"])
    # 5 burial strata by quantile, delta recomputed inside each, weighted by stratum size
    q = pd.qcut(g["burial"], 5, duplicates="drop")
    for tag, col in [("mpnn", "mpnn_score"), ("dms", "DMS_score")]:
        ds, ws_ = [], []
        for _, gg in g.groupby(q, observed=True):
            m = gg["iface_dist_5.0"].to_numpy()
            d, _ = delta(gg.loc[m, col].to_numpy(), gg.loc[~m, col].to_numpy())
            if np.isfinite(d):
                ds.append(d); ws_.append(int(m.sum()) * int((~m).sum()) / len(gg))
        row[f"delta_{tag}_burialstrat"] = float(np.average(ds, weights=ws_)) if ds else np.nan
        row[f"n_strata_{tag}"] = len(ds)
    out.append(row)
o = pd.DataFrame(out).sort_values("delta_dms")
o.to_csv(f"{PRED}/data/burial_stratified_delta.csv", index=False)
print(o[["DMS_id", "delta_mpnn", "delta_mpnn_burialstrat", "delta_dms", "delta_dms_burialstrat", "n_strata_mpnn"]]
      .to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
oo = o.dropna(subset=["delta_mpnn_burialstrat", "delta_dms_burialstrat"])
for tag in ("mpnn", "dms"):
    raw, adj = oo[f"delta_{tag}"].abs(), oo[f"delta_{tag}_burialstrat"].abs()
    print(f"\n  {tag:4s}: |delta| median  raw {raw.median():.3f} -> burial-matched {adj.median():.3f}"
          f"   ({100*(1-adj.median()/raw.median()):+.0f}% change)")
    print(f"        retained fraction (median |adj|/|raw|) = {(adj/raw).median():.3f}")
