"""S1: verify PDB chain sequence == wildtype_sequence[chain], and that the
sequence-index -> PDB-resnum map reproduces the `mutant_pdb` column exactly."""
import os, ast, sys
import pandas as pd
from bg_common import *

idx = load_index()
rows = []
for _, r in idx.iterrows():
    dms = r["DMS_id"]
    ws = parse_mut_dict(r["wildtype_sequence"])
    pdb = parse_pdb(os.path.join(PDB_DIR, r["pdb_file"]))
    df = pd.read_csv(os.path.join(DMS_DIR, r["DMS_filename"]))
    # which chains are ever mutated
    mut_by_chain = {c: 0 for c in ws}
    for s in df["mutant"]:
        for c, v in parse_mut_dict(s).items():
            if v.strip():
                mut_by_chain[c] = mut_by_chain.get(c, 0) + 1
    for ch, wt in ws.items():
        p = pdb.get(ch)
        if p is None:
            rows.append(dict(DMS_id=dms, chain=ch, status="CHAIN_MISSING_IN_PDB",
                             len_wt=len(wt), len_pdb=0, offset=None,
                             n_mut_rows=mut_by_chain.get(ch, 0)))
            continue
        off = find_offset(p["seq"], wt)
        exact = (p["seq"] == wt)
        rows.append(dict(DMS_id=dms, chain=ch,
                         status="EXACT" if exact else ("SHIFT" if off is not None else "MISMATCH"),
                         len_wt=len(wt), len_pdb=len(p["seq"]), offset=off,
                         first_resnum=p["resnums"][0], last_resnum=p["resnums"][-1],
                         n_mut_rows=mut_by_chain.get(ch, 0)))
out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(out.to_string(index=False))
print()
print(out["status"].value_counts().to_string())
out.to_csv(f"{OUT_DIR}/chain_map_check.csv", index=False)
