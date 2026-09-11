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
    res = chain_residues(pdb)
    wt_ref = ast.literal_eval(row["wildtype_sequence"])

    # 每链的最小编号 + 跨度校验(与 score_bgym_mpnn.py 同一规则)
    lo = {}
    for c in wt_ref:
        assert c in res, f"{DMS_id}: 结构里没有链 {c}"
        ns = sorted(res[c]); lo[c] = ns[0]
        span = ns[-1] - ns[0] + 1
        assert span == len(wt_ref[c]), \
            f"{DMS_id} 链{c}: 编号跨度 {span} != BindingGYM 序列长 {len(wt_ref[c])}"

    df = pd.read_csv(os.path.join(a.dms_input, f"{DMS_id}.csv"))
    muts, skipped = [], 0
    for m in df["mutant"]:
        d = ast.literal_eval(m) if str(m).startswith("{") else {"?": m}
        toks, ok = [], True
        for c, s in d.items():
            for tk in str(s).split(":"):
                if not tk: continue
                wt, pos, mt = tk[0], int(tk[1:-1]), tk[-1]
                rn = lo[c] + pos - 1
                if rn not in res[c]:           # 结构缺该残基(晶体断链)
                    ok = False; continue
                assert res[c][rn] == wt or res[c][rn] == "X", \
                    f"{DMS_id}: 链{c} 序列位{pos}(resnum {rn}) 结构是 {res[c][rn]} 但 mutant 说 {wt}"
                toks.append(f"{wt}{c}{rn}{mt}")
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
