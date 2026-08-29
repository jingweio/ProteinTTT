"""S2: build seq-index -> PDB-resnum map two independent ways and cross-check.
  (a) empirical: from the `mutant` vs `mutant_pdb` column pair (ground truth, but only
      defined at positions that are actually mutated somewhere in the assay)
  (b) structural: difflib alignment of the ATOM-record sequence to wildtype_sequence
"""
import os, json, difflib
import numpy as np, pandas as pd
from bg_common import *

def struct_map(pdb_seq, resnums, wt_seq):
    """-> dict seq_pos(1-based) -> resnum, plus stats"""
    sm = difflib.SequenceMatcher(a=pdb_seq, b=wt_seq, autojunk=False)
    m = {}
    matched = 0
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            m[j + k + 1] = resnums[i + k]
            matched += 1
    return m, matched

idx = load_index()
report = []
maps = {}
for _, r in idx.iterrows():
    dms, ws = r["DMS_id"], parse_mut_dict(r["wildtype_sequence"])
    pdb = parse_pdb(os.path.join(PDB_DIR, r["pdb_file"]))
    df = pd.read_csv(os.path.join(DMS_DIR, r["DMS_filename"]))
    emp = {}          # chain -> {pos: resnum}
    bad_wt = 0
    for ms, mps in zip(df["mutant"], df["mutant_pdb"]):
        md, pd_ = parse_mut_dict(ms), parse_mut_dict(mps)
        for ch in md:
            a, b = muts_of(md[ch]), muts_of(pd_.get(ch, ""))
            if len(a) != len(b):
                bad_wt += 1; continue
            for (w1, p1, m1), (w2, p2, m2) in zip(a, b):
                if w1 != w2 or m1 != m2:
                    bad_wt += 1; continue
                p1 = int(p1)
                if ws[ch][p1 - 1] != w1:
                    bad_wt += 1; continue
                prev = emp.setdefault(ch, {}).get(p1)
                if prev is not None and prev != p2:
                    bad_wt += 1
                emp[ch][p1] = p2
    for ch, wt in ws.items():
        p = pdb.get(ch)
        if p is None:
            report.append(dict(DMS_id=dms, chain=ch, note="no chain in PDB")); continue
        sm_, matched = struct_map(p["seq"], p["resnums"], wt)
        e = emp.get(ch, {})
        agree = sum(1 for k, v in e.items() if sm_.get(k) == v)
        missing = sum(1 for k in e if k not in sm_)
        conflict = len(e) - agree - missing
        report.append(dict(DMS_id=dms, chain=ch, len_wt=len(wt), n_struct_res=len(p["seq"]),
                           struct_cov=round(matched/len(wt), 4), n_emp_pos=len(e),
                           emp_agree=agree, emp_missing_in_struct=missing, emp_conflict=conflict,
                           bad_rows=bad_wt))
        maps[(dms, ch)] = {"struct": sm_, "emp": e}
rep = pd.DataFrame(report)
pd.set_option("display.width", 260)
print(rep.to_string(index=False))
print("\nTOTAL conflicts:", int(rep["emp_conflict"].fillna(0).sum()),
      " TOTAL emp positions missing coords:", int(rep["emp_missing_in_struct"].fillna(0).sum()))
rep.to_csv("../../local-records/binding-sites-analysis/data/residue_map_crosscheck.csv", index=False)
