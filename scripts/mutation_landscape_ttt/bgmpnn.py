"""Differentiable ProteinMPNN scoring with the encoder cached, matching BindingGYM exactly.

The TTT optimises a score and is then judged by a Spearman computed from a score; if those
two are not the same function, the experiment measures nothing. So the score here is the
official one from compute_fitness_multi_pdb.py:

    global_score(v) = - mean over M decoding orders of  sum_i  NLL_i * mask_i

summed over every residue of the complex, with no wild-type term and no length
normalisation (_scores has the division commented out). The only difference this module is
allowed to have from the official path is M, and score_variants() records the M it used.

Caching the encoder is not an optimisation but a direct consequence of the decoder-only
setting: the encoder reads X, mask, residue_idx and chain_encoding_all -- never S -- so for
a fixed WT complex its output is identical for every variant of that assay. The masks and
h_EXV_encoder built from it depend only on the decoding order, so they are cached per order
as well, and each training step runs three DecLayers and W_out.
"""
import os, sys, copy, hashlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
from protein_mpnn_utils import (ProteinMPNN, parse_PDB, tied_featurize,  # noqa: E402
                                cat_neighbors_nodes, gather_nodes)

CKPT_MD5 = "91d54c97a68bf551114f8c74c785e90f"      # official vanilla v_48_020
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA2I = {a: i for i, a in enumerate(ALPHABET)}
HIDDEN, NLAYERS = 128, 3


def load_model(ckpt_path, device):
    h = hashlib.md5(open(ckpt_path, "rb").read()).hexdigest()
    assert h == CKPT_MD5, f"checkpoint is not the official vanilla v_48_020: {h}"
    ck = torch.load(ckpt_path, map_location=device)
    if "noise_level" not in ck:
        ck = {"model_state_dict": ck, "noise_level": 0.2, "num_edges": 48}
    m = ProteinMPNN(ca_only=False, num_letters=21, node_features=HIDDEN, edge_features=HIDDEN,
                    hidden_dim=HIDDEN, num_encoder_layers=NLAYERS, num_decoder_layers=NLAYERS,
                    augment_eps=0.0, k_neighbors=ck["num_edges"])   # backbone_noise default is 0.0
    m.to(device); m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


class AssayContext:
    """One (structure, designed-chain-set) group: encoder run once, decoder run per batch."""

    def __init__(self, model, pdb_file, chain_ids, M, device, randn=None):
        self.model, self.device, self.M = model, device, M
        pdb = parse_PDB(pdb_file, ca_only=False)
        all_chains = [k[-1:] for k in pdb[0] if k[:9] == "seq_chain"]
        designed = [str(c) for c in chain_ids] if chain_ids else all_chains
        fixed = [c for c in all_chains if c not in designed]
        cid = {pdb[0]["name"]: (designed, fixed)}
        clones = [copy.deepcopy(pdb[0]) for _ in range(M)]
        f = tied_featurize(clones, device, cid, None, None, None, None, None, ca_only=False)
        self.X, self.S_wt, self.mask, self.chain_M = f[0], f[1], f[2], f[4]
        self.chain_enc, self.chain_M_pos, self.residue_idx = f[5], f[10], f[12]
        self.L = self.X.shape[1]
        self.designed, self.all_chains = designed, all_chains
        # tied_featurize packs masked (designed) chains first, then visible ones, using the
        # PDB chain sequences -- record the lengths so callers can index h_V by chain.
        self.chain_lengths = {c: len(pdb[0][f"seq_chain_{c}"]) for c in all_chains}
        # the official script draws one randn per POI and reuses it for every variant
        self.randn = torch.randn(self.chain_M.shape, device=device) if randn is None else randn
        self._build_encoder_cache()

    @torch.no_grad()
    def _build_encoder_cache(self):
        m = self.model
        E, E_idx_all = m.features(self.X, self.mask, self.residue_idx, self.chain_enc)
        E_idx = E_idx_all
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = m.W_e(E)
        ma = gather_nodes(self.mask.unsqueeze(-1), E_idx).squeeze(-1)
        ma = self.mask.unsqueeze(-1) * ma
        for layer in m.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, self.mask, ma)
        # the M clones are identical, so keep one copy and broadcast it over the batch
        self.h_V, self.h_E, self.E_idx = h_V[:1], h_E[:1], E_idx[:1]

        # S-independent: zeros_like(h_S) carries no sequence information
        h_EX = cat_neighbors_nodes(torch.zeros_like(m.W_s(self.S_wt[:1])), self.h_E, self.E_idx)
        self.h_EXV_encoder = cat_neighbors_nodes(self.h_V, h_EX, self.E_idx)

        # the official call passes chain_M*chain_M_pos, and forward() then multiplies by mask
        chain_M = self.chain_M * self.chain_M_pos * self.mask
        order = torch.argsort((chain_M + 0.0001) * torch.abs(self.randn))
        K = E_idx.shape[1]
        P = F.one_hot(order, num_classes=K).float()
        omb = torch.einsum("ij, biq, bjp->bqp",
                           (1 - torch.triu(torch.ones(K, K, device=self.device))), P, P)
        attend = torch.gather(omb, 2, E_idx).unsqueeze(-1)   # E_idx here keeps all M rows
        m1d = self.mask.view([self.mask.size(0), self.mask.size(1), 1, 1])
        self.mask_bw = m1d * attend                       # (M, L, K, 1)
        self.mask_fw = m1d * (1.0 - attend)
        self.h_EXV_fw = self.mask_fw * self.h_EXV_encoder  # (M, L, K, C)

    def seq_to_S(self, mutated_sequence):
        """Official substitution order: designed chains concatenated, in the order given."""
        S = self.S_wt[:1].clone()
        start = 0
        for c in self.designed:
            seq = mutated_sequence[c]
            S[:, start:start + len(seq)] = torch.tensor([AA2I[a] for a in seq], device=self.device)
            start += len(seq)
        return S[0]

    def score(self, S_batch, m_idx=None):
        """S_batch: (B, L) long. Returns (B,) global_score, differentiable in decoder params."""
        B = S_batch.shape[0]
        ms = range(self.M) if m_idx is None else [m_idx]
        E_idx = self.E_idx.expand(B, -1, -1)
        h_E = self.h_E.expand(B, -1, -1, -1)
        mask = self.mask[:1].expand(B, -1)
        total = 0.0
        for m in ms:
            h_S = self.model.W_s(S_batch)
            h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
            h_V = self.h_V.expand(B, -1, -1)
            fw = self.h_EXV_fw[m:m + 1]
            bw = self.mask_bw[m:m + 1]
            for layer in self.model.decoder_layers:
                h_ESV = bw * cat_neighbors_nodes(h_V, h_ES, E_idx) + fw
                h_V = layer(h_V, h_ESV, mask)
            lp = F.log_softmax(self.model.W_out(h_V), dim=-1)
            nll = F.nll_loss(lp.reshape(-1, lp.size(-1)), S_batch.reshape(-1),
                             reduction="none").view(B, -1)
            total = total + (nll * mask).sum(-1)
        return -total / len(list(ms))


def set_trainable(model, mode):
    """Freeze everything, then unfreeze what `mode` names. Returns the trainable parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    groups = {"decoder": [model.decoder_layers, model.W_out],
              "w_out": [model.W_out]}
    assert mode in groups, mode
    ps = []
    for mod in groups[mode]:
        for p in mod.parameters():
            p.requires_grad_(True); ps.append(p)
    return ps
