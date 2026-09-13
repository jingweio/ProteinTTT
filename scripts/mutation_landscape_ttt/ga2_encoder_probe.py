"""G-A (corrected): what does the FROZEN encoder know about the interface?

Supersedes ga_encoder_probe.py, which had two defects found on 2026-09-13:

  1. parse_PDB walks `range(min_resn, max_resn+1)` and PADS crystallographic gaps with an
     'X' residue whose coordinates are NaN; tied_featurize then does `X[isnan] = 0.`, so
     those slots arrive at the encoder with coordinates AT THE ORIGIN and mask = 0. The old
     probe read ctx.X directly and silently treated them as real residues -- up to 17% of
     the rows for KRAS_PICK3CG. They are now excluded.
  2. It used CA-CA distance at 8 A instead of the heavy-atom 5 A definition every other
     analysis uses, on the belief that heavy-atom labels could not be indexed into h_V
     without a WT-sequence alignment. That was wrong: h_V is indexed by PDB residue, so the
     labels can be built directly in packing order from the PDB, with no WT alignment at
     all. This uses the same heavy-atom 5 A definition and the same metadata-derived entity
     partition as the rest of the project.

Reported per assay and as a mean, matching how BindingGYM itself is scored.
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bgmpnn import load_model, AssayContext

BG = "/data/guoj0f/share/BindingGYM"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = f"{ROOT}/workstation-records/mutation-landscape-TTT/data"
CUT = 5.0
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assays", nargs="*", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda:0"); assert "A100" in torch.cuda.get_device_name(0)
    model = load_model(f"{BG}/training/cache/v_48_020.pt", dev)
    ent = pd.read_csv(f"{ROOT}/local-records/binding-sites-analysis/data/entity_partition.csv").set_index("DMS_id")
    idx = pd.read_csv(f"{BG}/input/BindingGYM.csv")
    assays = a.assays or list(pd.read_csv(f"{OUT}/g1e_canonical14.csv").DMS_id)

    X_all, d_all, a_all, drop = [], [], [], []
    for dms in assays:
        r = idx[idx.DMS_id == dms].iloc[0]
        df = pd.read_csv(f"{BG}/input/Binding_substitutions_DMS/{r.DMS_filename}")
        df["chain_id"] = df["chain_id"].fillna("")
        pdb = f"{BG}/input/structures/{r.pdb_file}"
        ctx = AssayContext(model, pdb, df["chain_id"].values[0], 1, dev)
        h = ctx.h_V[0].float().cpu().numpy()
        valid_mask = ctx.mask[0].cpu().numpy() > 0            # tied_featurize's own validity

        order = ctx.designed + [c for c in ctx.all_chains if c not in ctx.designed]
        e1 = set(ent.loc[dms, "entity1"])
        slots, ent_of = [], []
        for c in order:
            s = pdb_chain_slots(pdb, c)
            assert len(s) == ctx.chain_lengths[c], f"{dms} chain {c}: {len(s)} vs {ctx.chain_lengths[c]}"
            slots += s; ent_of += [0 if c in e1 else 1] * len(s)
        assert len(slots) == len(h), f"{dms}: {len(slots)} slots vs h_V {len(h)}"
        ent_of = np.asarray(ent_of)
        ok = np.array([s is not None for s in slots]) & valid_mask   # drop padded + masked

        A = np.vstack([slots[i] for i in np.flatnonzero(ok & (ent_of == 0))])
        B = np.vstack([slots[i] for i in np.flatnonzero(ok & (ent_of == 1))])
        dd = np.full(len(slots), np.nan)
        for i in np.flatnonzero(ok):
            other = B if ent_of[i] == 0 else A
            dd[i] = np.sqrt(((slots[i][:, None, :] - other[None, :, :]) ** 2).sum(-1)).min()
        drop.append(dict(DMS_id=dms, L=len(h), kept=int(ok.sum()), dropped=int((~ok).sum()),
                         frac_dropped=float((~ok).mean()), frac_iface=float((dd[ok] <= CUT).mean())))
        X_all.append(h[ok]); d_all.append(dd[ok]); a_all.append(np.full(int(ok.sum()), dms))
        print(f"{dms:42s} L={len(h):>5d} 丢弃={int((~ok).sum()):>4d} "
              f"({(~ok).mean():.1%}) 界面比例={float((dd[ok]<=CUT).mean()):.3f}", flush=True)

    pd.DataFrame(drop).to_csv(f"{OUT}/ga2_coverage.csv", index=False)
    X = np.vstack(X_all); d = np.concatenate(d_all); A_ = np.concatenate(a_all)
    y = (d <= CUT).astype(int)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)

    rows = []
    for dms in assays:
        m = A_ == dms
        if len(set(y[m])) < 2 or m.sum() < 40:
            rows.append(dict(DMS_id=dms, n=int(m.sum()))); continue
        pa = np.zeros(m.sum()); pr = np.zeros(m.sum())
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs[m], y[m]):
            pa[te] = LogisticRegression(max_iter=2000).fit(Xs[m][tr], y[m][tr]).predict_proba(Xs[m][te])[:, 1]
            pr[te] = Ridge(alpha=1.0).fit(Xs[m][tr], d[m][tr]).predict(Xs[m][te])
        tr = ~m
        qa = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr]).predict_proba(Xs[m])[:, 1]
        qr = Ridge(alpha=1.0).fit(Xs[tr], d[tr]).predict(Xs[m])
        rows.append(dict(DMS_id=dms, n=int(m.sum()), frac_iface=float(y[m].mean()),
                         auc_in=roc_auc_score(y[m], pa), rho_in=stats.spearmanr(pr, d[m]).statistic,
                         auc_loao=roc_auc_score(y[m], qa), rho_loao=stats.spearmanr(qr, d[m]).statistic))
    t = pd.DataFrame(rows); t.to_csv(f"{OUT}/ga2_encoder_probe.csv", index=False)
    pd.set_option("display.width", 240)
    print("\n=== 冻结 encoder node embedding 的线性探针（重原子 5 A 定义，已排除缺口与 mask=0）===")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for c, lab in [("auc_in", "AUC within"), ("auc_loao", "AUC LOAO"),
                   ("rho_in", "rho(d) within"), ("rho_loao", "rho(d) LOAO")]:
        print(f"  {lab:16s} mean {t[c].mean():.3f}   median {t[c].median():.3f}")


if __name__ == "__main__":
    main()
