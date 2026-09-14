"""PDB parsing shared by the probe and the TTT: residue ordering identical to
parse_PDB_biounits, with heavy atoms carried along and gap-padded slots marked None.

Split out of ga2_encoder_probe.py so callers that only need the geometry do not pull in
sklearn -- the TTT script does not use it, and the local smoke env has no sklearn.
The function body is unchanged.
"""
import numpy as np

ALPHA3 = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())

def pdb_chain_slots(path, chain):
    """Replicate parse_PDB_biounits' residue ordering EXACTLY, and carry heavy atoms along.

    Returns a list, one entry per packing slot, of either the residue's heavy-atom
    coordinates or None for a gap-padded slot.
    """
    res = {}                                     # resn -> resa -> list of heavy-atom xyz
    lo, hi = 10**6, -10**6
    for raw in open(path, "rb"):
        line = raw.decode("utf-8", "ignore")
        if line[:6] == "HETATM" and line[17:20] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
        if line[:4] != "ATOM" or line[21:22] != chain:
            continue
        if line[17:20] not in ALPHA3:
            continue
        elem = line[76:78].strip().upper() or line[12:16].strip()[:1]
        if elem == "H":
            continue
        tok = line[22:27].strip()
        if tok[-1].isalpha(): resa, resn = tok[-1], int(tok[:-1]) - 1
        else:                 resa, resn = "", int(tok) - 1
        lo, hi = min(lo, resn), max(hi, resn)
        res.setdefault(resn, {}).setdefault(resa, []).append(
            (float(line[30:38]), float(line[38:46]), float(line[46:54])))
    slots = []
    for n in range(lo, hi + 1):
        if n in res:
            for k in sorted(res[n]):
                slots.append(np.asarray(res[n][k]))
        else:
            slots.append(None)                   # the gap parse_PDB pads with X / NaN
    return slots

