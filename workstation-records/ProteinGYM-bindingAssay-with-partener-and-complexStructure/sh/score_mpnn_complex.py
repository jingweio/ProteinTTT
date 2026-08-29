#!/usr/bin/env python
"""
Zero-shot ProteinMPNN scoring of ProteinGym binding assays on BindingGYM WT complexes.

Two conditions, identical in every other respect, so the difference isolates the partner:
  --condition complex   target chain designed, partner chain(s) present as fixed context
  --condition monomer   the SAME structure with partner chain(s) deleted

Protocol follows ProteinGym's official ProteinMPNN baseline
(proteingym/baselines/protein_mpnn/compute_fitness.py): v_48_020 checkpoint,
backbone_noise=0.0, one random decoding order per variant, score = -mean NLL.
Variants are batched along the batch dimension (each row gets its own decoding
order, exactly as in the official per-variant loop) purely for throughput.

Emits both scores:
  mpnn_design_ll  -mean NLL over the TARGET chain only   <- primary
  mpnn_global_ll  -mean NLL over all residues            <- official's `pmpnn_ll` analogue
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def load_mpnn(ckpt_path, device):
    sys.path.insert(0, os.environ["MPNN_UTILS_DIR"])
    from protein_mpnn_utils import ProteinMPNN
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = ProteinMPNN(ca_only=False, num_letters=21, node_features=128, edge_features=128,
                    hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                    augment_eps=0.0, k_neighbors=ck['num_edges'])
    m.load_state_dict(ck['model_state_dict']); m.to(device).eval()
    return m, ck


def write_monomer(src, dst, keep):
    with open(src) as f, open(dst, "w") as o:
        for ln in f:
            if ln.startswith(("ATOM", "HETATM", "TER")) and len(ln) > 21 and ln[21] not in keep:
                continue
            o.write(ln)


def min_resid(pdb, chain):
    v = [int(l[22:26]) for l in open(pdb)
         if l.startswith("ATOM") and l[21] == chain]
    return min(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--assay", required=True)
    ap.add_argument("--condition", choices=["complex", "monomer"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=0,
                    help="0 = auto from --target-bl and the complex length")
    ap.add_argument("--target-bl", type=int, default=24000,
                    help="batch*length budget; decoder activations scale as B*L*K*3H")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)

    sys.path.insert(0, os.environ["MPNN_UTILS_DIR"])
    from protein_mpnn_utils import tied_featurize, parse_PDB, _scores

    ad = os.path.join(a.dataset, a.assay)
    meta = json.load(open(os.path.join(ad, "meta.json")))
    tch, pchains = meta["target_chain"], meta["partner_chains"]
    shift = meta["pdb_resid_minus_target_index"]
    src = os.path.join(ad, "complex.pdb")
    os.makedirs(a.out, exist_ok=True)

    if a.condition == "monomer":
        pdb = os.path.join(a.out, f"{a.assay}__monomer.pdb")
        write_monomer(src, pdb, {tch}); chains = [tch]; vis = []
    else:
        pdb = src; chains = [tch] + list(pchains); vis = list(pchains)

    mr = min_resid(pdb, tch)
    pdb_dict = parse_PDB(pdb, input_chain_list=chains)
    name = pdb_dict[0]["name"]
    chain_id_dict = {name: ([tch], vis)}
    X, S, mask, lengths, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, omit_AA_mask, \
        residue_idx, dihedral_mask, _, _, _, _, _, _ = tied_featurize(
            pdb_dict, dev, chain_id_dict, None, None, None, None, None, ca_only=False)

    # ---- HARD coordinate check: target-chain index -> position in the featurised tensor.
    # parse_PDB gap-fills by residue number, and tied_featurize puts sorted(masked) before
    # sorted(visible), so the target chain starts at 0 -- both are asserted, never assumed.
    dec = "".join(ALPHABET[i] for i in S[0].cpu().numpy())
    tgt_fasta = open(os.path.join(ad, "sequences.fasta")).read().split("\n")[1]
    tgt_len = int(pdb_dict[0][f"seq_chain_{tch}"].__len__())
    assert dec[:tgt_len] == pdb_dict[0][f"seq_chain_{tch}"].replace("-", "X"), \
        "target chain is not the leading block of the featurised sequence"

    def s_index(target_idx):            # target-chain 1-based index -> tensor position
        return target_idx + shift - mr

    v = pd.read_csv(os.path.join(ad, "variants.csv"))
    v = v[v.in_structure].reset_index(drop=True)
    subs = [[(t[0], s_index(int(t[1:-1])), t[-1]) for t in m.split(":")]
            for m in v.mutant_target_chain]
    bad = [(i, w, p) for i, ss in enumerate(subs) for w, p, _ in ss
           if not (0 <= p < tgt_len) or dec[p] != w]
    assert not bad, f"WT residue mismatch in the featurised structure, e.g. {bad[:5]}"
    print(f"[{a.assay}/{a.condition}] chains={chains} L={S.shape[1]} target_len={tgt_len} "
          f"variants={len(v)}  coordinate check OK", flush=True)

    aa2i = {c: i for i, c in enumerate(ALPHABET)}
    # flatten every substitution into (variant_row, tensor_pos, new_aa) so a batch is applied
    # with one scatter instead of a Python double loop
    f_row = np.fromiter((i for i, ss in enumerate(subs) for _ in ss), dtype=np.int64)
    f_pos = torch.tensor([p for ss in subs for _, p, _ in ss], device=dev)
    f_val = torch.tensor([aa2i[mt] for ss in subs for _, _, mt in ss], device=dev)
    f_row_t = torch.tensor(f_row, device=dev)

    L = S.shape[1]
    B = a.batch_size if a.batch_size > 0 else max(1, min(512, a.target_bl // L))
    print(f"  batch_size={B} (L={L})", flush=True)
    Xb, mb, cMb, cMp, rib, ceb = (t.repeat(B, *([1] * (t.dim() - 1)))
                                  for t in (X, mask, chain_M, chain_M_pos, residue_idx, chain_encoding_all))
    mdes, mglo = (mb * cMb * cMp), mb
    des, glo = [], []
    t0 = time.time()
    with torch.no_grad():
        for st in range(0, len(v), B):
            n = min(B, len(v) - st)
            lo, hi = np.searchsorted(f_row, [st, st + n])
            Sm = S.repeat(n, 1).clone()
            Sm[f_row_t[lo:hi] - st, f_pos[lo:hi]] = f_val[lo:hi]
            randn = torch.randn(n, L, device=dev)
            for attempt in range(6):                     # halve on OOM rather than dying
                try:
                    d_ = g_ = None
                    sub_b = max(1, n >> attempt)
                    dd, gg = [], []
                    for j in range(0, n, sub_b):
                        k = min(sub_b, n - j)
                        lp = model(Xb[:k], Sm[j:j+k], mb[:k], (cMb * cMp)[:k],
                                   rib[:k], ceb[:k], randn[j:j+k])
                        dd.append((-_scores(Sm[j:j+k], lp, mdes[:k])).cpu().numpy())
                        gg.append((-_scores(Sm[j:j+k], lp, mglo[:k])).cpu().numpy())
                    d_, g_ = np.concatenate(dd), np.concatenate(gg)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  OOM at sub-batch {max(1, n >> attempt)}, halving", flush=True)
            assert d_ is not None, "OOM even at sub-batch 1"
            des.append(d_); glo.append(g_)
            if st % (B * 100) == 0:
                print(f"  {st+n}/{len(v)}  {(st+n)/max(time.time()-t0,1e-9):.0f} var/s", flush=True)
    v["mpnn_design_ll"] = np.concatenate(des)
    v["mpnn_global_ll"] = np.concatenate(glo)
    out = os.path.join(a.out, f"{a.assay}__{a.condition}.csv")
    v.to_csv(out, index=False)
    from scipy.stats import spearmanr
    print(f"[{a.assay}/{a.condition}] n={len(v)}  wall={time.time()-t0:.0f}s  "
          f"rho_design={spearmanr(v.DMS_score, v.mpnn_design_ll).correlation:.4f}  "
          f"rho_global={spearmanr(v.DMS_score, v.mpnn_global_ll).correlation:.4f}  -> {out}", flush=True)


if __name__ == "__main__":
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument("--checkpoint", required=True)
    model, _ck = load_mpnn(_ap.parse_known_args()[0].checkpoint, dev)
    print(f"checkpoint: num_edges={_ck['num_edges']} noise_level={_ck['noise_level']}", flush=True)
    main()
