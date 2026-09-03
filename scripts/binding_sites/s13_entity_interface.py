"""S13: interface defined from the COMPLEX, not from the mutation data.

The earlier rule split chains into "ever mutated" vs "never mutated". That is a
data-driven proxy: it happens to recover the right two binding entities on this
benchmark, but it is a property of the library, not of the complex, and it would
silently mislabel an intra-entity interface (e.g. VH-VL packing) as a binding
interface if a library mutated only one chain of a two-chain entity.

Here the two entities are fixed per complex, mutation-free, by two independent routes:
  (1) metadata  -- the DMS_id names the two binding partners ("4D5_HER2" = 4D5 vs HER2).
                   Only the 3 three-chain assays need a chain assignment; for a
                   two-chain complex the partition is forced.
  (2) structure -- among all 2-partitions of the chains, the one with the SMALLEST
                   inter-partition contact. For a Fab this correctly cuts between the
                   Fab and the antigen, because the VH-VL interface is far larger than
                   the paratope-epitope one. Heuristic, and asserted against (1).

Output covers EVERY residue of EVERY chain -- both entities, and positions the library
never touches -- which the mutation-driven version could not provide.
"""
import os, sys, difflib, itertools
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bg_common import *

CUT = 5.0
# Curated chain -> entity for the only complexes with more than two chains.
# Read off the DMS_id: the antibody variable domains form one entity, the antigen the other.
CURATED = {
    "4D5_HER2_fitness_1N8Z":   ({"A", "B"}, {"C"}),   # A=VL, B=VH (Fab 4D5)      | C=HER2
    "5A12_Ang2_fitness_4ZFG":  ({"H", "L"}, {"A"}),   # H=VH, L=VL (Fab 5A12)     | A=Ang2
    "5A12_VEGF_fitness_4ZFF":  ({"H", "L"}, {"C"}),   # H=VH, L=VL (Fab 5A12)     | C=VEGF
}

def contact_count(pdb, E1, E2, cut=CUT):
    t1 = cKDTree(np.vstack([np.vstack(pdb[c]["coords"]) for c in E1]))
    t2 = cKDTree(np.vstack([np.vstack(pdb[c]["coords"]) for c in E2]))
    n1 = sum(1 for c in E1 for x in pdb[c]["coords"] if t2.query(x)[0].min() <= cut)
    n2 = sum(1 for c in E2 for x in pdb[c]["coords"] if t1.query(x)[0].min() <= cut)
    return n1 + n2

def min_contact_partition(pdb, chains):
    best = None
    for k in range(1, len(chains)):
        for g in itertools.combinations(chains, k):
            E1 = set(g); E2 = set(chains) - E1
            if min(E1) > min(E2): continue
            n = contact_count(pdb, E1, E2)
            if best is None or n < best[0]: best = (n, E1, E2)
    return best

idx = load_index()
part_rows, res_rows = [], []
for _, r in idx.iterrows():
    dms, ws = r["DMS_id"], parse_mut_dict(r["wildtype_sequence"])
    pdb = parse_pdb(os.path.join(PDB_DIR, r["pdb_file"]))
    chains = sorted(ws)
    n_struct, sE1, sE2 = min_contact_partition(pdb, chains)
    if dms in CURATED:
        cE1, cE2 = CURATED[dms]
        assert {frozenset(cE1), frozenset(cE2)} == {frozenset(sE1), frozenset(sE2)}, \
            f"{dms}: curated {cE1}|{cE2} != structural {sE1}|{sE2}"
        E1, E2, prov = cE1, cE2, "curated + structural agree"
    else:
        E1, E2, prov = sE1, sE2, "forced (2 chains)"
    # second-best partition, to record how safe the structural call was
    alts = [contact_count(pdb, set(g), set(chains) - set(g))
            for k in range(1, len(chains)) for g in itertools.combinations(chains, k)
            if min(set(g)) <= min(set(chains) - set(g))
            and {frozenset(g), frozenset(set(chains) - set(g))} != {frozenset(E1), frozenset(E2)}]
    part_rows.append(dict(DMS_id=dms, pdb=r["pdb_file"], n_chains=len(chains),
                          entity1="".join(sorted(E1)), entity2="".join(sorted(E2)),
                          provenance=prov, contact_chosen=n_struct,
                          contact_next_best=min(alts) if alts else np.nan))
    # every residue of every chain, distance to the OTHER entity
    for side, other in ((E1, E2), (E2, E1)):
        t = cKDTree(np.vstack([np.vstack(pdb[c]["coords"]) for c in other]))
        for c in sorted(side):
            p = pdb[c]
            sm = difflib.SequenceMatcher(a=p["seq"], b=ws[c], autojunk=False)
            r2s = {i + k: j + k + 1 for i, j, n in sm.get_matching_blocks() for k in range(n)}
            for i, rid in enumerate(p["resnums"]):
                d = t.query(p["coords"][i])[0].min()
                res_rows.append(dict(DMS_id=dms, chain=c,
                                     entity="E1" if c in E1 else "E2",
                                     seq_pos=r2s.get(i), pdb_resid=rid, wt_aa=p["seq"][i],
                                     min_dist_to_other_entity=round(float(d), 3),
                                     is_interface_5A=bool(d <= CUT)))

P = pd.DataFrame(part_rows); R = pd.DataFrame(res_rows)
P.to_csv(f"{OUT_DIR}/entity_partition.csv", index=False)
R.to_csv(f"{OUT_DIR}/interface_residues_all_chains.csv", index=False)
pd.set_option("display.width", 250)
print(P.to_string(index=False))
print(f"\nper-residue rows: {len(R):,}  ({R.is_interface_5A.sum():,} interface)  "
      f"assays {R.DMS_id.nunique()}  chains {len(R.groupby(['DMS_id','chain']))}")

# --- regression gate: the mutated-chain interface sets must be UNCHANGED ---
old = pd.read_csv(f"{OUT_DIR}/binding_sites_per_chain.csv")
bad = []
for _, o in old.iterrows():
    want = {int(x) for x in str(o["site_seqpos"]).split(";") if x and x != "nan"}
    got = set(R.query("DMS_id == @o.DMS_id and chain == @o.chain and is_interface_5A")
              ["seq_pos"].dropna().astype(int))
    if want != got: bad.append((o["DMS_id"], o["chain"], sorted(want ^ got)))
print(f"\nregression vs the mutation-driven definition: {len(old) - len(bad)}/{len(old)} "
      f"mutated chains identical" + ("" if not bad else f"\n  DIFFERENCES: {bad}"))
assert not bad, bad
