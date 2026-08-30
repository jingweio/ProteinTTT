#!/usr/bin/env python
"""Zero-shot ProteinMPNN scoring on ESMFold2-predicted structures.

Unlike the BindingGYM-crystal path, the structure here is folded FROM ProteinGym's
own target_seq, so chain A residue i == ProteinGym position i exactly: no offset to
solve, no truncation, no unknown residues -> 100% of variants are scorable.
Both facts are asserted, never assumed.

Protocol matches ProteinGym's official ProteinMPNN baseline: v_48_020,
backbone_noise=0, one random decoding order per variant, score = -mean NLL.
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--assay", required=True)
    ap.add_argument("--dms-dir", required=True)
    ap.add_argument("--dms-file", required=True)
    ap.add_argument("--condition", required=True)          # label only
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--target-bl", type=int, default=24000)
    ap.add_argument("--batch-size", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, os.environ["MPNN_UTILS_DIR"])
    from protein_mpnn_utils import ProteinMPNN, tied_featurize, parse_PDB, _scores

    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=128, edge_features=128,
                        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                        augment_eps=0.0, k_neighbors=ck['num_edges'])
    model.load_state_dict(ck['model_state_dict']); model.to(dev).eval()

    chains = sorted({l[21] for l in open(a.pdb) if l.startswith("ATOM")})
    tch, partners = "A", [c for c in chains if c != "A"]
    pdb_dict = parse_PDB(a.pdb, input_chain_list=chains)
    name = pdb_dict[0]["name"]
    X, S, mask, lengths, chain_M, chain_encoding_all, *_rest = tied_featurize(
        pdb_dict, dev, {name: ([tch], partners)}, None, None, None, None, None, ca_only=False)
    chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask = _rest[4], _rest[5], _rest[6], _rest[7]

    dec = "".join(ALPHABET[i] for i in S[0].cpu().numpy())
    tgt = pdb_dict[0][f"seq_chain_{tch}"].replace("-", "X")
    assert dec[:len(tgt)] == tgt, "target chain is not the leading block of the featurised sequence"

    d = pd.read_csv(os.path.join(a.dms_dir, a.dms_file))
    subs = [[(t[0], int(t[1:-1]) - 1, t[-1]) for t in m.split(":")] for m in d.mutant]
    bad = [(w, p) for ss in subs for w, p, _ in ss if p >= len(tgt) or dec[p] != w]
    assert not bad, f"{a.assay}: WT mismatch vs the folded structure, e.g. {bad[:5]}"
    print(f"[{a.assay}/{a.condition}] chains={chains} L={S.shape[1]} target={len(tgt)} "
          f"variants={len(d)}  offset=0 asserted, coverage=100%", flush=True)

    aa2i = {c: i for i, c in enumerate(ALPHABET)}
    f_row = np.fromiter((i for i, ss in enumerate(subs) for _ in ss), dtype=np.int64)
    f_pos = torch.tensor([p for ss in subs for _, p, _ in ss], device=dev)
    f_val = torch.tensor([aa2i[m] for ss in subs for _, _, m in ss], device=dev)
    f_row_t = torch.tensor(f_row, device=dev)

    L = S.shape[1]
    B = a.batch_size if a.batch_size > 0 else max(1, min(512, a.target_bl // L))
    Xb, mb, cMb, cMp, rib, ceb = (t.repeat(B, *([1] * (t.dim() - 1)))
                                  for t in (X, mask, chain_M, chain_M_pos, residue_idx, chain_encoding_all))
    mdes, mglo = mb * cMb * cMp, mb
    des, glo = [], []
    t0 = time.time()
    with torch.no_grad():
        for st in range(0, len(d), B):
            n = min(B, len(d) - st)
            lo, hi = np.searchsorted(f_row, [st, st + n])
            Sm = S.repeat(n, 1).clone()
            Sm[f_row_t[lo:hi] - st, f_pos[lo:hi]] = f_val[lo:hi]
            randn = torch.randn(n, L, device=dev)
            for att in range(6):
                try:
                    sb = max(1, n >> att); dd, gg = [], []
                    for j in range(0, n, sb):
                        k = min(sb, n - j)
                        lp = model(Xb[:k], Sm[j:j+k], mb[:k], (cMb*cMp)[:k], rib[:k], ceb[:k], randn[j:j+k])
                        dd.append((-_scores(Sm[j:j+k], lp, mdes[:k])).cpu().numpy())
                        gg.append((-_scores(Sm[j:j+k], lp, mglo[:k])).cpu().numpy())
                    d_, g_ = np.concatenate(dd), np.concatenate(gg); break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache(); d_ = None
                    print(f"  OOM at sub-batch {max(1, n >> att)}, halving", flush=True)
            assert d_ is not None, "OOM even at sub-batch 1"
            des.append(d_); glo.append(g_)
            if st % (B * 100) == 0:
                print(f"  {st+n}/{len(d)}  {(st+n)/max(time.time()-t0,1e-9):.0f} var/s", flush=True)

    d["mpnn_design_ll"] = np.concatenate(des); d["mpnn_global_ll"] = np.concatenate(glo)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"{a.assay}__{a.condition}.csv")
    d[["mutant", "DMS_score", "mpnn_design_ll", "mpnn_global_ll"]].to_csv(p, index=False)
    from scipy.stats import spearmanr
    print(f"[{a.assay}/{a.condition}] n={len(d)} wall={time.time()-t0:.0f}s "
          f"rho_design={spearmanr(d.DMS_score, d.mpnn_design_ll).correlation:.4f} "
          f"rho_global={spearmanr(d.DMS_score, d.mpnn_global_ll).correlation:.4f} -> {p}", flush=True)


if __name__ == "__main__":
    main()
