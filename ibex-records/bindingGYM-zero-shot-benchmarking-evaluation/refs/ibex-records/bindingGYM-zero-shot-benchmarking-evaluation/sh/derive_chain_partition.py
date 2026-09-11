#!/usr/bin/env python
"""为 BindingGYM 的 25 个 assay 推导 StaB-ddG 需要的 binding partition (binder1_binder2)。

BindingGYM.csv 只给了被突变的链(chain_id)和各链序列,没有给"界面两侧如何划分"。
StaB-ddG 的 --chains 需要 "ABC_DE" 形式。这里用结构接触图来定:
  1. 解析 PDB 重原子,算逐对链的接触数(任意重原子距离 < 5.0 A)
  2. 报告接触图 + 各链长度,供人工核对 assay 语义后定案

⚠️ 这是【定义实验的值】(§1b-0):结果冻结成 refs/chain_partition.tsv 并 commit,
   下游只读该文件,绝不在运行时重算。
"""
import sys, os, ast, itertools, numpy as np, pandas as pd

IDX = sys.argv[1]; STRUCT = sys.argv[2]
print(f"# numpy {np.__version__}  pandas {pd.__version__}  python {sys.version.split()[0]}")

def parse(pdb):
    """返回 {chain: (N,3) 重原子坐标}。只取 ATOM 行,跳过氢。"""
    ch = {}
    for ln in open(pdb):
        if not ln.startswith("ATOM"): continue
        el = ln[76:78].strip() or ln[12:16].strip()[0]
        if el == "H": continue
        c = ln[21]
        ch.setdefault(c, []).append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return {k: np.asarray(v, dtype=np.float32) for k, v in ch.items()}

def contacts(A, B, cut=5.0):
    """分块算 <cut A 的重原子对数,避免 N^2 内存爆掉。"""
    n = 0
    for i in range(0, len(A), 2000):
        d = np.linalg.norm(A[i:i+2000, None, :] - B[None, :, :], axis=-1)
        n += int((d < cut).sum())
    return n

idx = pd.read_csv(IDX)
rows = []
seen = {}
for _, r in idx.iterrows():
    pdb = os.path.join(STRUCT, r["pdb_file"])
    if not os.path.exists(pdb):
        print(f"!! 缺结构 {r['DMS_id']} -> {r['pdb_file']}"); continue
    if pdb not in seen: seen[pdb] = parse(pdb)
    ch = seen[pdb]
    wt = ast.literal_eval(r["wildtype_sequence"])
    lens = {c: len(s) for c, s in wt.items()}
    pairs = []
    for a, b in itertools.combinations(sorted(ch), 2):
        n = contacts(ch[a], ch[b])
        if n > 0: pairs.append((a, b, n))
    pairs.sort(key=lambda x: -x[2])
    rows.append(dict(DMS_id=r["DMS_id"], pdb=r["pdb_file"], mut_chains=r["chain_id"],
                     chains="".join(sorted(ch)),
                     lens=",".join(f"{c}:{lens.get(c,'?')}" for c in sorted(ch)),
                     contacts=" ".join(f"{a}-{b}:{n}" for a, b, n in pairs)))
    print(f"{r['DMS_id']:<38} chains={rows[-1]['chains']:<5} mut={str(r['chain_id']):<5} "
          f"| {rows[-1]['lens']:<28} | {rows[-1]['contacts']}")

pd.DataFrame(rows).to_csv(sys.argv[3], index=False)
print(f"\n写出 {sys.argv[3]}")
