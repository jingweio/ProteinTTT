"""Shared helpers: BindingGYM PDB parsing + sequence<->PDB-residue mapping."""
import ast, os
import numpy as np
import pandas as pd

ROOT = "/home/guoj0f/share/BindingGYM/input"
DMS_DIR = os.path.join(ROOT, "Binding_substitutions_DMS")
PDB_DIR = os.path.join(ROOT, "structures")

# Single source of truth for where artefacts go. Every script writes here;
# do not reintroduce a second output directory.
REC_DIR = "/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-analysis"
OUT_DIR = os.path.join(REC_DIR, "data")

THREE2ONE = {
 'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
 'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
 'THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M','HSD':'H','HSE':'H','HSP':'H',
 'HID':'H','HIE':'H','HIP':'H','CYX':'C','CYM':'C','ASH':'D','GLH':'E','LYN':'K',
}

def parse_pdb(path):
    """-> {chain: {'resnums': [int], 'seq': str, 'coords': [np.ndarray(n_heavy,3)]}} (heavy atoms only)."""
    chains = {}
    order = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            elem = line[76:78].strip()
            name = line[12:16].strip()
            if elem == "H" or (elem == "" and name.startswith("H")):
                continue
            resname = line[17:20].strip()
            ch = line[21]
            resnum = line[22:26].strip() + line[26].strip()   # resid string, keeps icode
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            d = chains.setdefault(ch, {})
            key = resnum
            if key not in d:
                d[key] = [resname, []]
                order.setdefault(ch, []).append(key)
            d[key][1].append(xyz)
    out = {}
    for ch, d in chains.items():
        rn = order[ch]
        out[ch] = {
            "resnums": rn,
            "seq": "".join(THREE2ONE.get(d[k][0], "X") for k in rn),
            "coords": [np.asarray(d[k][1], dtype=np.float64) for k in rn],
        }
    return out

def load_index():
    return pd.read_csv(os.path.join(ROOT, "BindingGYM.csv"))

def parse_mut_dict(s):
    return ast.literal_eval(s)

def muts_of(mutstr):
    """'A11C:D38C' -> [(wt, resid_str, mt)].  resid keeps any PDB insertion code ('52A')."""
    out = []
    for tok in mutstr.split(":"):
        tok = tok.strip()
        if not tok:
            continue
        out.append((tok[0], tok[1:-1], tok[-1]))
    return out

def find_offset(pdb_seq, wt_seq):
    """Return o such that pdb_seq[i] == wt_seq[i+o] for all i, else None (contiguous shift only)."""
    n, m = len(pdb_seq), len(wt_seq)
    for o in range(-m, m + 1):
        ok = True
        for i, c in enumerate(pdb_seq):
            j = i + o
            if j < 0 or j >= m or wt_seq[j] != c:
                ok = False
                break
        if ok:
            return o
    return None
