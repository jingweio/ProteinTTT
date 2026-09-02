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

# Short display names for the 25 assays (shared by every figure script).
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
