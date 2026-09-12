#!/usr/bin/env python
"""把 BindingGYM 的一个 assay 转成 StaB-ddG 的 csv 模式输入。

StaB-ddG 需要:
  pdb_dir/  {stem}.pdb + {stem}_{binder1}.pdb + {stem}_{binder2}.pdb
  csv       列 `#Pdb` = "{stem}_{binder1}_{binder2}"、列 `mutation` = SKEMPI 串

🔴 坐标系转换:BindingGYM 的 mutant 用【序列位】(1-based,链内),SKEMPI 串要【PDB 残基编号】。
   映射沿用已验证的规则:resnum = 该链最小编号 + (序列位-1)。
   每个突变都断言 WT 残基与结构一致 —— 对不上即 assert,绝不静默错位。

chain partition 从冻结的 refs/chain_partition.tsv 读,不重算(§1b-0)。
"""
import argparse, ast, os, sys, hashlib
import numpy as np, pandas as pd

AA3to1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G',
          'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
          'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}


def _align_map(obs, ref):
    """NW 比对,返回 {obs_idx: ref_idx}。"""
    n, m = len(obs), len(ref)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = -np.arange(n + 1); D[0, :] = -np.arange(m + 1)
    for i in range(1, n + 1):
        oi = obs[i - 1]
        for j in range(1, m + 1):
            D[i, j] = max(D[i-1, j-1] + (1 if oi == ref[j-1] else -1), D[i-1, j] - 1, D[i, j-1] - 1)
    i, j, out = n, m, {}
    while i > 0 and j > 0:
        if D[i, j] == D[i-1, j-1] + (1 if obs[i-1] == ref[j-1] else -1):
            out[i-1] = j-1; i -= 1; j -= 1
        elif D[i, j] == D[i-1, j] - 1: i -= 1
        else: j -= 1
    return out


def chain_residues(pdb):
    """{chain: {resnum: 1-letter aa}}，按 ATOM 行的第一个原子取 resname。"""
    out = {}
    for ln in open(pdb):
        if not ln.startswith("ATOM"): continue
        c, n, rn = ln[21], int(ln[22:26]), ln[17:20].strip()
        out.setdefault(c, {}).setdefault(n, AA3to1.get(rn, "X"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dms_mapping", required=True)
    ap.add_argument("--dms_input", required=True)
    ap.add_argument("--structure_folder", required=True)
    ap.add_argument("--partition_tsv", required=True)
    ap.add_argument("--dms_index", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    a = ap.parse_args()

    part = pd.read_csv(a.partition_tsv, sep="\t", comment="#")
    idx = pd.read_csv(a.dms_mapping); row = idx.iloc[a.dms_index]
    DMS_id = row["DMS_id"]
    pr = part[part.DMS_id == DMS_id]
    assert len(pr) == 1, f"{DMS_id}: chain_partition.tsv 里没有唯一记录"
    b1, b2 = pr.iloc[0]["binder1"], pr.iloc[0]["binder2"]

    pdb = os.path.join(a.structure_folder, row["pdb_file"])
    stem = os.path.splitext(os.path.basename(pdb))[0]
    wt_ref = ast.literal_eval(row["wildtype_sequence"])

    # 🔴 StaB-ddG 的 ppi_dataset.py:195 用
    #        mut_pos = int(mut[2:-1]) + chain_offset - 1
    #    即把 SKEMPI 串里的数字当作【seq_chain_X 内的 1-based 下标】,不是 PDB 残基编号。
    #    所以这里直接用它自己的 parse_PDB 取 seq_chain_X,再把 BindingGYM 序列比对上去,
    #    编号按比对到的下标生成 —— 不猜它的约定。
    sys.path.insert(0, os.environ.get("STABDDG_REPO", "."))
    from stabddg.mpnn_utils import parse_PDB as mpnn_parse
    pdict = mpnn_parse(pdb)[0]
    seqc, ref2idx = {}, {}
    for c in wt_ref:
        key = f"seq_chain_{c}"
        assert key in pdict, f"{DMS_id}: StaB-ddG 的 parse_PDB 里没有链 {c}"
        seqc[c] = pdict[key]
        am = _align_map(seqc[c], wt_ref[c])            # seq_chain 下标 -> BindingGYM 下标
        mism = [(k, seqc[c][k], wt_ref[c][v]) for k, v in am.items()
                if seqc[c][k] != wt_ref[c][v] and seqc[c][k] != "X"]
        assert not mism, f"{DMS_id} 链{c}: 比对后 WT 不符,前 3 处 {mism[:3]}"
        ref2idx[c] = {v: k for k, v in am.items()}     # BindingGYM 下标 -> seq_chain 下标
        if len(seqc[c]) != len(wt_ref[c]):
            print(f"  [map] {DMS_id} 链{c}: seq_chain 长 {len(seqc[c])} vs BindingGYM {len(wt_ref[c])},"
                  f" 比对映射 {len(am)} 个位点")

    df = pd.read_csv(os.path.join(a.dms_input, f"{DMS_id}.csv"))
    muts, skipped = [], 0
    for m in df["mutant"]:
        d = ast.literal_eval(m) if str(m).startswith("{") else {"?": m}
        toks, ok = [], True
        for c, s in d.items():
            for tk in str(s).split(":"):
                if not tk: continue
                wt, pos, mt = tk[0], int(tk[1:-1]), tk[-1]
                k = ref2idx.get(c, {}).get(pos - 1)     # BindingGYM 1-based -> seq_chain 0-based
                if k is None: ok = False; continue      # 该位点在结构里缺失
                assert seqc[c][k] == wt or seqc[c][k] == "X", \
                    f"{DMS_id}: 链{c} 序列位{pos} -> seq_chain[{k}]={seqc[c][k]} 但 mutant 说 {wt}"
                toks.append(f"{wt}{c}{k+1}{mt}")        # StaB-ddG 要 1-based seq_chain 下标
        if not toks: ok = False
        muts.append(",".join(toks) if ok else "")
        if not ok: skipped += 1

    out = os.path.join(a.out_root, DMS_id); os.makedirs(os.path.join(out, "pdbs"), exist_ok=True)
    os.system(f"cp {pdb} {out}/pdbs/{stem}.pdb")
    sys.path.insert(0, os.environ.get("STABDDG_REPO", "."))
    from stabddg.utils import extract_chains
    extract_chains(f"{out}/pdbs/{stem}.pdb", f"{out}/pdbs/{stem}_{b1}.pdb", b1,
                   f"{out}/pdbs/{stem}_{b2}.pdb", b2)

    keep = df.copy()
    keep["#Pdb"] = f"{stem}_{b1}_{b2}"
    keep["mutation"] = muts
    keep.to_csv(os.path.join(out, "all_rows.csv"), index=False)     # 保留全部行(含被跳过的)
    sub = keep[keep.mutation != ""].copy()
    sub[["#Pdb", "mutation"]].to_csv(os.path.join(out, "muts.csv"), index=False)
    print(f"[prep] {DMS_id}  partition={b1}_{b2}  rows={len(df)}  可打分={len(sub)}  跳过={skipped}")
    if skipped:
        print(f"        (跳过的是突变位点落在结构缺失残基上的 variant —— 会在汇总时如实标注)")


if __name__ == "__main__":
    main()
