"""S3: define binding sites, label every variant, dump per-assay + per-variant tables."""
import os, json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from bg_common import *
import difflib

OUT = "/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/analysis_out"
os.makedirs(OUT, exist_ok=True)
CUTOFFS = [4.0, 4.5, 5.0, 6.0, 8.0]
PRIMARY = 5.0

# ---------- Shrake-Rupley SASA ----------
VDW = {"C":1.70,"N":1.55,"O":1.52,"S":1.80,"P":1.80,"SE":1.90,"X":1.70}
def _sphere(n=200):
    i = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2*i/n); th = np.pi*(1+5**0.5)*i
    return np.c_[np.cos(th)*np.sin(phi), np.sin(th)*np.sin(phi), np.cos(phi)]
SPH = _sphere(200)

def sasa_per_residue(coords_list, elems_list, probe=1.4):
    """coords_list: list per residue of (n,3); elems_list: matching element symbols."""
    xyz = np.vstack(coords_list)
    rad = np.array([VDW.get(e, 1.70) for es in elems_list for e in es]) + probe
    owner = np.concatenate([[i]*len(c) for i, c in enumerate(coords_list)])
    tree = cKDTree(xyz)
    maxr = rad.max()
    out = np.zeros(len(coords_list))
    nbrs = tree.query_ball_point(xyz, r=2*maxr)
    for a in range(len(xyz)):
        nb = np.array([b for b in nbrs[a] if b != a], dtype=int)
        pts = xyz[a] + rad[a]*SPH
        if len(nb):
            d = np.linalg.norm(pts[:, None, :] - xyz[nb][None, :, :], axis=2)
            acc = (d >= rad[nb][None, :]).all(axis=1)
        else:
            acc = np.ones(len(pts), bool)
        out[owner[a]] += 4*np.pi*rad[a]**2 * acc.mean()
    return out

# ---------- parse with elements ----------
def parse_pdb_elems(path):
    chains, order = {}, {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"): continue
            elem = line[76:78].strip().upper(); name = line[12:16].strip()
            if elem == "H" or (elem == "" and name.startswith("H")): continue
            if not elem: elem = name[0]
            ch = line[21]; rid = line[22:26].strip() + line[26].strip()
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            d = chains.setdefault(ch, {})
            if rid not in d:
                d[rid] = [line[17:20].strip(), [], []]; order.setdefault(ch, []).append(rid)
            d[rid][1].append(xyz); d[rid][2].append(elem)
    out = {}
    for ch, d in chains.items():
        rs = order[ch]
        out[ch] = dict(resids=rs,
                       seq="".join(THREE2ONE.get(d[k][0], "X") for k in rs),
                       coords=[np.asarray(d[k][1]) for k in rs],
                       elems=[d[k][2] for k in rs])
    return out

def struct_map(pdb_seq, resids, wt_seq):
    sm = difflib.SequenceMatcher(a=pdb_seq, b=wt_seq, autojunk=False)
    m = {}
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            m[j+k+1] = i+k          # seq pos (1-based) -> index into resids
    return m

# ---------- main ----------
idx = load_index()
assay_rows, site_rows, var_frames = [], [], []
for _, r in idx.iterrows():
    dms = r["DMS_id"]
    ws = parse_mut_dict(r["wildtype_sequence"])
    pdb = parse_pdb_elems(os.path.join(PDB_DIR, r["pdb_file"]))
    df = pd.read_csv(os.path.join(DMS_DIR, r["DMS_filename"]))
    parsed = [parse_mut_dict(s) for s in df["mutant"]]

    mutated_chains = sorted({c for md in parsed for c, v in md.items() if v.strip()})
    all_chains = sorted(ws.keys())
    partner_chains = [c for c in all_chains if c not in mutated_chains]
    zdomain_case = len(partner_chains) == 0
    # partner atom set per mutated chain
    def partner_atoms(ch):
        pcs = partner_chains if not zdomain_case else [c for c in all_chains if c != ch]
        return np.vstack([np.vstack(pdb[c]["coords"]) for c in pcs]) if pcs else np.zeros((0, 3))
    # "any other chain" (the old, buggy definition)
    def other_atoms(ch):
        ocs = [c for c in all_chains if c != ch]
        return np.vstack([np.vstack(pdb[c]["coords"]) for c in ocs]) if ocs else np.zeros((0, 3))

    maps, dmin_p, dmin_o = {}, {}, {}
    for ch in mutated_chains:
        p = pdb[ch]
        maps[ch] = struct_map(p["seq"], p["resids"], ws[ch])
        for tag, atoms, store in (("p", partner_atoms(ch), dmin_p), ("o", other_atoms(ch), dmin_o)):
            t = cKDTree(atoms) if len(atoms) else None
            store[ch] = np.array([t.query(c)[0].min() if t is not None else np.inf
                                  for c in p["coords"]])
    # --- SASA-based interface (dSASA > 1 A^2 for residues of mutated chains) ---
    dsasa = {}
    try:
        cx_c = [c for ch in all_chains for c in pdb[ch]["coords"]]
        cx_e = [e for ch in all_chains for e in pdb[ch]["elems"]]
        s_cx = sasa_per_residue(cx_c, cx_e)
        pos = 0; sl = {}
        for ch in all_chains:
            n = len(pdb[ch]["coords"]); sl[ch] = slice(pos, pos+n); pos += n
        for ch in mutated_chains:
            side = [ch] if zdomain_case else mutated_chains
            fr_c = [c for cc in side for c in pdb[cc]["coords"]]
            fr_e = [e for cc in side for e in pdb[cc]["elems"]]
            s_fr = sasa_per_residue(fr_c, fr_e)
            o = 0
            for cc in side:
                n = len(pdb[cc]["coords"])
                if cc == ch: dsasa[ch] = s_fr[o:o+n] - s_cx[sl[cc]]
                o += n
    except Exception as e:
        print("SASA failed", dms, e); dsasa = {}

    # binding-site sets
    sites = {}
    for cut in CUTOFFS:
        sites[("dist", cut)] = {ch: {p for p, i in maps[ch].items() if dmin_p[ch][i] <= cut}
                                for ch in mutated_chains}
    sites[("distOTHER", PRIMARY)] = {ch: {p for p, i in maps[ch].items() if dmin_o[ch][i] <= PRIMARY}
                                     for ch in mutated_chains}
    if dsasa:
        sites[("dsasa", 1.0)] = {ch: {p for p, i in maps[ch].items() if dsasa[ch][i] > 1.0}
                                 for ch in mutated_chains}

    # per-variant labels
    n_mut = np.zeros(len(df), int)
    lab = {k: np.zeros(len(df), bool) for k in sites}
    minid = np.full(len(df), np.inf)
    for i, md in enumerate(parsed):
        for ch, v in md.items():
            if not v.strip(): continue
            for _, ps, _ in muts_of(v):
                p = int(ps); n_mut[i] += 1
                minid[i] = min(minid[i], dmin_p[ch][maps[ch][p]])
                for k, S in sites.items():
                    if p in S[ch]: lab[k][i] = True
    keep = n_mut > 0
    vf = pd.DataFrame(dict(DMS_id=dms, DMS_score=df["DMS_score"].values, n_mut=n_mut,
                           min_dist_to_partner=minid,
                           **{f"iface_{k[0]}_{k[1]}": lab[k] for k in sites}))[keep]
    var_frames.append(vf)

    # per-assay summary
    prim = ("dist", PRIMARY)
    nres_sites = sum(len(sites[prim][ch]) for ch in mutated_chains)
    nres_tot = sum(len(maps[ch]) for ch in mutated_chains)
    mutated_pos = {ch: {int(ps) for md in parsed for c, v in md.items() if c == ch and v.strip()
                        for _, ps, _ in muts_of(v)} for ch in mutated_chains}
    n_libpos = sum(len(v) for v in mutated_pos.values())
    n_libpos_site = sum(len(mutated_pos[ch] & sites[prim][ch]) for ch in mutated_chains)
    assay_rows.append(dict(
        DMS_id=dms, pdb=r["pdb_file"], n_var=int(keep.sum()),
        mutated_chains="".join(mutated_chains), partner_chains="".join(partner_chains) or "(none: both sides mutated)",
        L_mutated=nres_tot, n_site_res=nres_sites, frac_site_res=round(nres_sites/nres_tot, 4),
        n_lib_pos=n_libpos, n_lib_pos_site=n_libpos_site,
        frac_lib_pos_site=round(n_libpos_site/n_libpos, 4) if n_libpos else np.nan,
        n_iface_var=int(vf[f"iface_dist_{PRIMARY}"].sum()),
        n_noniface_var=int((~vf[f"iface_dist_{PRIMARY}"]).sum()),
        frac_no_iface=round(float((~vf[f"iface_dist_{PRIMARY}"]).mean()), 4),
        frac_no_iface_OTHERdef=round(float((~vf[f"iface_distOTHER_{PRIMARY}"]).mean()), 4),
        frac_no_iface_dsasa=round(float((~vf["iface_dsasa_1.0"]).mean()), 4) if "iface_dsasa_1.0" in vf else np.nan,
        **{f"frac_no_iface_{c}A": round(float((~vf[f"iface_dist_{c}"]).mean()), 4) for c in CUTOFFS},
    ))
    for ch in mutated_chains:
        inv = {v: k for k, v in maps[ch].items()}
        S = sorted(sites[prim][ch])
        site_rows.append(dict(DMS_id=dms, chain=ch, n_res=len(maps[ch]), n_site=len(S),
                              site_seqpos=";".join(map(str, S)),
                              site_pdbres=";".join(pdb[ch]["resids"][maps[ch][p]] for p in S),
                              lib_pos=";".join(map(str, sorted(mutated_pos[ch]))),
                              lib_pos_site=";".join(map(str, sorted(mutated_pos[ch] & sites[prim][ch])))))
    print("done", dms, flush=True)

pd.DataFrame(assay_rows).to_csv(f"{OUT}/assay_summary.csv", index=False)
pd.DataFrame(site_rows).to_csv(f"{OUT}/binding_sites_per_chain.csv", index=False)
pd.concat(var_frames, ignore_index=True).to_parquet(f"{OUT}/variant_labels.parquet")
print("WROTE", OUT)
