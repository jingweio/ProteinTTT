#!/usr/bin/env python
"""
Build ProteinGym binding assays augmented with BindingGYM partner sequence + WT complex structure.

For every ProteinGym assay whose variant library can be located inside a BindingGYM assay,
emit: target sequence, partner sequence(s), the WT complex PDB, and a variant table whose
coordinates are validated three ways (ProteinGym target_seq / BindingGYM chain seq / PDB ATOM).

Nothing is hard-coded except the input roots: the PG<->BG pairing, the coordinate offset and
the PDB numbering shift are all DERIVED and then asserted.

Usage:  python build_pg_binding_complex.py --out <dir> [--min-overlap 0.10] [--iface-cutoff 8.0]
"""
import argparse, ast, json, os, shutil, sys, warnings
from collections import Counter, defaultdict
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

PG_REF = "/home/guoj0f/repos/ProteinGym/reference_files/DMS_substitutions.csv"
PG_DIR = "/home/guoj0f/share/ProteinGym/DMS_ProteinGym_substitutions"
BG_IDX = "/home/guoj0f/share/BindingGYM/input/BindingGYM.csv"
BG_DIR = "/home/guoj0f/share/BindingGYM/input/Binding_substitutions_DMS"
BG_PDB = "/home/guoj0f/share/BindingGYM/input/structures"

AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
       'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
       'TYR':'Y','VAL':'V'}


# ---------------------------------------------------------------- loaders
def load_pg(dms_id, filename):
    d = pd.read_csv(os.path.join(PG_DIR, filename))
    return d[["mutant", "DMS_score"]]


def load_bg(row):
    """-> dict(chain -> [(parts, score)]) restricted to records where that chain is the ONLY mutated one."""
    d = pd.read_csv(os.path.join(BG_DIR, row.DMS_filename))
    md = [ast.literal_eval(x) for x in d["mutant"]]
    per = defaultdict(list)
    for m, s in zip(md, d["DMS_score"].values):
        live = [k for k, v in m.items() if v]
        if len(live) != 1:
            continue
        ch = live[0]
        per[ch].append(([(t[0], int(t[1:-1]), t[-1]) for t in m[ch].split(":")], s))
    return per


def read_pdb(path):
    """-> (ca: dict chain -> {resid: aa1},  heavy: dict chain -> (N,3) array, resid array)"""
    ca, xyz, rid = defaultdict(dict), defaultdict(list), defaultdict(list)
    for ln in open(path):
        if not ln.startswith("ATOM"):
            continue
        if ln[76:78].strip() == "H" or ln[12:16].strip().startswith("H"):
            continue
        c, r = ln[21], int(ln[22:26])
        xyz[c].append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        rid[c].append(r)
        if ln[12:16].strip() == "CA" and r not in ca[c]:
            ca[c][r] = AA3.get(ln[17:20].strip(), "X")
    return ({c: v for c, v in ca.items()},
            {c: np.asarray(v) for c, v in xyz.items()},
            {c: np.asarray(v) for c, v in rid.items()})


# ---------------------------------------------------------------- derivations
def vote_offset(pg_singles, bg_recs):
    """PG_pos = BG_pos + offset, voted on (wt,mut) identity of single substitutions."""
    anchor = defaultdict(list)
    for m in pg_singles:
        anchor[(m[0], m[-1])].append(int(m[1:-1]))
    votes = Counter()
    for parts, _ in bg_recs:
        if len(parts) != 1:
            continue
        w, p, mt = parts[0]
        for pp in anchor.get((w, mt), []):
            votes[pp - p] += 1
    return votes.most_common(1)[0] if votes else (None, 0)


def vote_shift(ca_chain, bgseq):
    """BG_index = pdb_resid - shift; chosen by maximal residue-identity agreement."""
    if not ca_chain:
        return None, 0, 0
    lo = min(ca_chain)
    best = (None, -1, 0)
    for s in range(lo - 5, lo + 6):
        ok = tot = 0
        for r, aa in ca_chain.items():
            i = r - s
            if 1 <= i <= len(bgseq):
                tot += 1
                ok += (bgseq[i - 1] == aa)
        if ok > best[1]:
            best = (s, ok, tot)
    return best


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-overlap", type=float, default=0.10,
                    help="min fraction of BG records recovered in PG (or vice versa) to accept a pairing")
    ap.add_argument("--min-rho", type=float, default=0.90,
                    help="min Spearman on the shared variants; below this the two are DIFFERENT assays")
    ap.add_argument("--iface-cutoff", type=float, default=8.0)
    a = ap.parse_args()
    from scipy.stats import spearmanr

    os.makedirs(a.out, exist_ok=True)
    pgref = pd.read_csv(PG_REF)
    bgidx = pd.read_csv(BG_IDX)
    BG = {r.DMS_id: (r, load_bg(r)) for _, r in bgidx.iterrows()}

    # ---- stage 1: locate every PG assay inside BindingGYM (all 217 x 25, not just Binding)
    cand = []
    for _, pr in pgref.iterrows():
        d = load_pg(pr.DMS_id, pr.DMS_filename)
        pmut = dict(zip((tuple(sorted(m.split(":"))) for m in d.mutant), d.DMS_score))
        singles = [m for m in d.mutant if ":" not in m]
        for bid, (br, per) in BG.items():
            for ch, recs in per.items():
                off, nv = vote_offset(singles, recs)
                if off is None:
                    continue
                inter = [(pmut[k], s) for parts, s in recs
                         if (k := tuple(sorted(f"{w}{p+off}{mt}" for w, p, mt in parts))) in pmut]
                if len(inter) < 50:
                    continue
                cov_pg, cov_bg = len(inter) / len(pmut), len(inter) / len(recs)
                if max(cov_pg, cov_bg) < a.min_overlap:
                    continue
                rho = spearmanr([x[0] for x in inter], [x[1] for x in inter]).correlation
                cand.append(dict(PG=pr.DMS_id, PG_cat=pr.coarse_selection_type, BG=bid, ch=ch,
                                 off=off, n_common=len(inter), cov_pg=cov_pg, cov_bg=cov_bg, rho=rho))
    cand = pd.DataFrame(cand)
    cand.to_csv(os.path.join(a.out, "_all_candidate_pairings.csv"), index=False)

    # accept: same assay == high rank agreement. keep best BG per PG assay.
    ok = cand[cand.rho >= a.min_rho].sort_values("n_common", ascending=False)
    ok = ok.drop_duplicates("PG", keep="first")
    print(f"[stage1] {len(cand)} candidate pairings -> {len(ok)} accepted at rho>={a.min_rho}")

    # ---- stage 2: materialise each accepted pairing
    manifest = []
    for _, c in ok.sort_values("PG").iterrows():
        pr = pgref[pgref.DMS_id == c.PG].iloc[0]
        br, per = BG[c.BG]
        wt = ast.literal_eval(br.wildtype_sequence)
        tch = c.ch
        tseq_bg = wt[tch]
        partners = {k: v for k, v in wt.items() if k != tch}
        pdb_src = os.path.join(BG_PDB, br.pdb_file)
        ca, xyz, rid = read_pdb(pdb_src)

        shift, agree, tot = vote_shift(ca.get(tch, {}), tseq_bg)
        assert shift is not None and agree / max(tot, 1) > 0.95, \
            f"{c.PG}: PDB chain {tch} does not match BindingGYM sequence ({agree}/{tot})"

        # --- three-way WT validation.
        # HARD invariant: every position ProteinGym actually mutates must carry the same WT residue
        # in ProteinGym's target_seq and in the BindingGYM chain. Positions OUTSIDE the mutated
        # region may legitimately differ -- BindingGYM ships the *crystallisation construct*
        # sequence, which can carry tags / linkers / engineered extensions (e.g. 1BE9 chain A
        # positions 403-415 are construct, not native PSD-95). Those are reported, not fatal.
        d_pre = load_pg(pr.DMS_id, pr.DMS_filename)
        mut_pos = sorted({int(t[1:-1]) for m in d_pre.mutant for t in m.split(":")})
        bad = [(p, pr.target_seq[p - 1], tseq_bg[p - c.off - 1])
               for p in mut_pos
               if 1 <= p - c.off <= len(tseq_bg) and tseq_bg[p - c.off - 1] != "X"
               and pr.target_seq[p - 1] != tseq_bg[p - c.off - 1]]
        assert not bad, f"{c.PG}: WT residue disagrees at MUTATED positions, e.g. {bad[:3]}"
        # positions BindingGYM marks unknown ("X"): identity is absent, so they carry no structural
        # prior even where coordinates exist. Counted, and excluded from in_structure below.
        unk = {p for p in mut_pos
               if 1 <= p - c.off <= len(tseq_bg) and tseq_bg[p - c.off - 1] == "X"}

        chk = [(i, tseq_bg[i - 1], pr.target_seq[i + c.off - 1])
               for i in range(1, len(tseq_bg) + 1) if 1 <= i + c.off <= len(pr.target_seq)]
        mism = [x for x in chk if x[1] != x[2] and x[1] != "X"]
        div = ""
        if mism:
            lo, hi = min(x[0] for x in mism), max(x[0] for x in mism)
            div = f"target-chain idx {lo}-{hi} (PG pos {lo+c.off}-{hi+c.off}), {len(mism)} residues"

        # --- interface: target residues within cutoff of ANY partner heavy atom
        P = np.vstack([xyz[k] for k in xyz if k != tch])
        T, TR = xyz[tch], rid[tch]
        dmin = defaultdict(lambda: 1e9)
        for i in range(0, len(T), 4000):                       # chunk to bound memory
            blk = np.sqrt(((T[i:i+4000, None, :] - P[None, :, :]) ** 2).sum(-1)).min(1)
            for r, dd in zip(TR[i:i+4000], blk):
                dmin[int(r)] = min(dmin[int(r)], float(dd))
        iface_pg = {r - shift + c.off for r, dd in dmin.items() if dd <= a.iface_cutoff}
        resolved_pg = {r - shift + c.off for r in ca[tch] if 1 <= r - shift <= len(tseq_bg)} - unk

        # --- variant table
        d = load_pg(pr.DMS_id, pr.DMS_filename)
        rows = []
        for m, s in zip(d.mutant, d.DMS_score):
            ps = [int(t[1:-1]) for t in m.split(":")]
            rows.append(dict(
                mutant_PG=m,
                mutant_target_chain=":".join(f"{t[0]}{int(t[1:-1])-c.off}{t[-1]}" for t in m.split(":")),
                DMS_score=s, n_subs=len(ps),
                in_structure=all(p in resolved_pg for p in ps),
                touches_interface=any(p in iface_pg for p in ps)))
        vt = pd.DataFrame(rows)

        dst = os.path.join(a.out, c.PG)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(pdb_src, os.path.join(dst, "complex.pdb"))
        vt.to_csv(os.path.join(dst, "variants.csv"), index=False)
        with open(os.path.join(dst, "sequences.fasta"), "w") as fh:
            fh.write(f">target|chain_{tch}|{len(tseq_bg)}aa\n{tseq_bg}\n")
            for k, v in partners.items():
                fh.write(f">partner|chain_{k}|{len(v)}aa\n{v}\n")
        meta = dict(PG_assay=c.PG, PG_category=c.PG_cat, BG_assay=c.BG, pdb=br.pdb_file,
                    target_chain=tch, target_len=len(tseq_bg),
                    partner_chains=list(partners), partner_len=[len(v) for v in partners.values()],
                    offset_PG_minus_target=int(c.off), pdb_resid_minus_target_index=int(shift),
                    pairing_rho=float(c.rho), n_shared_with_BG=int(c.n_common),
                    n_variants=len(vt), n_in_structure=int(vt.in_structure.sum()),
                    n_touch_interface=int((vt.in_structure & vt.touches_interface).sum()),
                    target_X=tseq_bg.count("X"), partner_X=sum(v.count("X") for v in partners.values()),
                    wt_window_agree=f"{len(chk)-len(mism)}/{len(chk)}", construct_divergence=div,
                    n_mutpos=len(mut_pos), n_mutpos_unknown_X=len(unk),
                    iface_cutoff=a.iface_cutoff)
        json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=2)
        manifest.append(meta)
        print(f"  [ok] {c.PG:42s} {br.pdb_file:16s} tgt {tch}({len(tseq_bg)}) "
              f"partner {'+'.join(partners)}({'+'.join(str(len(v)) for v in partners.values())})  "
              f"struct {vt.in_structure.mean()*100:5.1f}%  iface {(vt.in_structure&vt.touches_interface).sum()/max(vt.in_structure.sum(),1)*100:5.1f}%")

    pd.DataFrame(manifest).to_csv(os.path.join(a.out, "manifest.csv"), index=False)
    print(f"\n[done] {len(manifest)} assays -> {a.out}/manifest.csv")


if __name__ == "__main__":
    main()
