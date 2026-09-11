#!/usr/bin/env python
"""判别探针：LigandMPNN 的 use_sequence 到底哪个方向条件于其余残基？

读源码得到的结论与 CLI help 文字相反，故用实验定案，不靠推理：
  改变【其它】位点的氨基酸，看目标位点 idx 的 log_probs 变不变。
    变了  ⇒ 该设置条件于其余序列 (sequence-conditioned)
    没变  ⇒ 该设置只看 backbone
"""
import sys, os, argparse, numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)          # LigandMPNN repo 路径
ap.add_argument("--ckpt", required=True)          # proteinmpnn_v_48_020.pt
ap.add_argument("--pdb",  required=True)
ap.add_argument("--idx",  type=int, default=10)   # 被检查的位点
a = ap.parse_args()
sys.path.insert(0, a.repo)
from data_utils import parse_PDB, featurize, alphabet
from model_utils import ProteinMPNN

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, "| alphabet:", "".join(alphabet))
assert "".join(alphabet) == "ACDEFGHIKLMNPQRSTVWYX", "alphabet 与官方不一致，打分会错位"

ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
m = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128,
                num_encoder_layers=3, num_decoder_layers=3,
                k_neighbors=ck["num_edges"], device=dev, atom_context_num=1,
                model_type="protein_mpnn", ligand_mpnn_use_side_chain_context=0)
m.load_state_dict(ck["model_state_dict"]); m.to(dev).eval()

pd_, _, _, _, _ = parse_PDB(a.pdb, device=dev, parse_all_atoms=False)
pd_["chain_mask"] = torch.ones_like(pd_["mask"])
fd = featurize(pd_, model_type="protein_mpnn")
fd["batch_size"] = 1
fd["symmetry_residues"] = [[]]
L = fd["S"].shape[1]
print(f"L = {L}, 探测位点 idx = {a.idx}")

g = torch.Generator(device=dev).manual_seed(0)
randn_fixed = torch.randn([1, L], device=dev, generator=g)

def probe(use_sequence):
    outs = []
    for variant in (0, 1):
        fd2 = dict(fd)
        S = fd["S"].clone()
        if variant == 1:                       # 扰动【其它】位点，不动 idx
            far = [p for p in range(L) if abs(p - a.idx) > 5][:40]
            for p in far:
                S[0, p] = (int(S[0, p]) + 7) % 20
        fd2["S"] = S
        fd2["randn"] = randn_fixed
        with torch.no_grad():
            o = m.single_aa_score(fd2, use_sequence=use_sequence)
        outs.append(o["log_probs"][0, a.idx].cpu().numpy())
    d = float(np.abs(outs[0] - outs[1]).max())
    verdict = "条件于其余序列" if d > 1e-5 else "只看 backbone"
    print(f"  use_sequence={str(use_sequence):<5}  max|Δlog_probs| = {d:.3e}   ⇒ {verdict}")
    return d

print("\n=== single_aa_score ===")
d_true  = probe(True)
d_false = probe(False)
print("\n结论：条件于其余序列的是 use_sequence =",
      "True" if d_true > d_false else "False",
      f"  (ΔTrue={d_true:.2e} vs ΔFalse={d_false:.2e})")
