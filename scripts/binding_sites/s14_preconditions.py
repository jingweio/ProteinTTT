"""S14: assert every structural precondition the binding-site definition relies on.

The definition in the record says things like "all non-hydrogen atoms" and "no
ligands". Those are claims about the 22 shipped PDB files, not universal truths, so
they are checked here rather than asserted in prose. Anything that fails means the
definition no longer describes what the code computes.
"""
import os, sys, glob, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bg_common import *
import difflib

STD = set("ACDEFGHIKLMNPQRSTVWY")
files = sorted(glob.glob(os.path.join(PDB_DIR, "*.pdb")))
rec, alt, icode, empty_elem, resnames, chains = collections.Counter(), set(), set(), 0, set(), 0
for f in files:
    seen = set()
    for line in open(f):
        rec[line[:6].strip()] += 1
        if not line.startswith("ATOM"): continue
        alt.add(line[16]); icode.add(line[26]); resnames.add(line[17:20].strip())
        if not line[76:78].strip(): empty_elem += 1
        seen.add(line[21])
    chains += len(seen)

print(f"files {len(files)}   chains {chains}   record types {dict(rec)}")
assert set(rec) <= {"ATOM", "TER", "END"}, f"unexpected records: {set(rec)}"
assert rec.get("HETATM", 0) == 0, "HETATM present: ligands/waters/metals would need a policy"
assert alt == {" "}, f"altLoc present: {alt - {' '}}"
assert empty_elem == 0, f"{empty_elem} ATOM lines have an empty element column"
assert resnames <= set(THREE2ONE), f"unmapped residue names: {resnames - set(THREE2ONE)}"
assert {THREE2ONE[r] for r in resnames} <= STD, "non-standard residue would alias to X"
print(f"  no HETATM / no altLoc / element column always set / residues = "
      f"{len(resnames)} standard types only")
print(f"  insertion codes in use: {sorted(icode - {' '}) or 'none'}   "
      f"(files: {[os.path.basename(f) for f in files if any(l.startswith('ATOM') and l[26] != ' ' for l in open(f))]})")

idx = load_index()
tot = cov = 0
lib_missing = 0
for _, r in idx.iterrows():
    ws = parse_mut_dict(r["wildtype_sequence"])
    pdb = parse_pdb(os.path.join(PDB_DIR, r["pdb_file"]))
    assert set(ws) <= set(pdb), f"{r['DMS_id']}: chains {set(ws) - set(pdb)} missing from structure"
    # only the mutant column: the sequence columns are huge and unused here
    df = pd.read_csv(os.path.join(DMS_DIR, r["DMS_filename"]), usecols=["mutant"])
    muts = [parse_mut_dict(x) for x in df["mutant"]]
    for c, wt in ws.items():
        sm = difflib.SequenceMatcher(a=pdb[c]["seq"], b=wt, autojunk=False)
        m = {j + k + 1 for i, j, n in sm.get_matching_blocks() for k in range(n)}
        tot += len(wt); cov += len(m)
        lib = {int(ps) for md in muts for cc, v in md.items()
               if cc == c and v.strip() for _, ps, _ in muts_of(v)}
        lib_missing += len(lib - m)
print(f"  sequence positions {tot:,}  with coordinates {cov:,}  without {tot-cov:,} "
      f"({100*(tot-cov)/tot:.2f}%)")
assert lib_missing == 0, f"{lib_missing} library positions have no coordinates"
print(f"  library positions without coordinates: 0  -> every scored mutation is evaluable")
print("\nALL PRECONDITIONS HOLD")
