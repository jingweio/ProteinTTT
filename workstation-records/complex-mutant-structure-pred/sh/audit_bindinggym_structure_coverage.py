#!/usr/bin/env python3
"""
BindingGYM 自带结构 vs DMS 突变位点的覆盖审计。

回答：BindingGYM 的 25 个 assay 里，被 DMS 突变的每一个位点，在 BindingGYM 随数据集分发的
WT complex 结构里是否有坐标？—— 这决定了"是否还需要外部的 predicted-complex-structure asset"。

三个输出：
  ① DMS mutant 串里的 WT 氨基酸 vs BindingGYM.csv 的 wildtype_sequence 是否一致
  ② 已解析残基 == wildtype_sequence 中所有非-X 位点（验证 X 就是结构缺口的占位符）
  ③ DMS 位点在结构中有坐标的比例

用法:  python3 audit_bindinggym_structure_coverage.py [BINDINGGYM_INPUT_DIR]
默认 BINDINGGYM_INPUT_DIR = /home/guoj0f/share/BindingGYM/input
"""
import sys, ast, re, collections
import pandas as pd

B = sys.argv[1] if len(sys.argv) > 1 else '/home/guoj0f/share/BindingGYM/input'

THREE = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
         'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
         'TYR':'Y','VAL':'V','MSE':'M'}


def resolved_chains(path):
    """{chain: 已解析残基的一维序列(按文件顺序)}"""
    ch = collections.defaultdict(dict)
    for line in open(path):
        if line.startswith(('ATOM', 'HETATM')):
            rn = line[17:20].strip()
            if rn in THREE:
                ch[line[21]][line[22:27]] = THREE[rn]   # (resSeq+iCode) 去重
    return {c: ''.join(v.values()) for c, v in ch.items()}


def dms_positions(csv_path):
    """{chain: {pos: wt_aa}}  —— mutant 列是逐链字典, 值形如 'A11C' 或 'I31V:L35V'"""
    pos = collections.defaultdict(dict)
    d = pd.read_csv(csv_path)
    for s in d['mutant'].astype(str):
        try:
            mm = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            continue
        for c, v in mm.items():
            for tok in (v.split(':') if v else []):
                m = re.match(r'^([A-Z])(\d+)([A-Z*])$', tok)
                if m:
                    pos[c][int(m.group(2))] = m.group(1)
    return pos, len(d)


def main():
    idx = pd.read_csv(f'{B}/BindingGYM.csv')
    rows, viol = [], []
    tot_id = tot = cov = 0
    for _, r in idx.iterrows():
        wt = ast.literal_eval(r['wildtype_sequence'])
        st = resolved_chains(f"{B}/structures/{r['pdb_file']}")

        # ② X-占位符假设
        for c, W in wt.items():
            S = st.get(c, '')
            nonX = [i for i, a in enumerate(W) if a != 'X']
            if not (len(nonX) == len(S) and all(W[i] == S[j] for j, i in enumerate(nonX))):
                viol.append((r['DMS_id'], c, len(nonX), len(S)))

        pos, n_var = dms_positions(f"{B}/Binding_substitutions_DMS/{r['DMS_filename']}")
        for c, pm in sorted(pos.items()):
            W = wt.get(c, '')
            ok   = sum(1 for p, a in pm.items() if p <= len(W) and W[p - 1] == a)   # ①
            has  = sum(1 for p in pm if p <= len(W) and W[p - 1] != 'X')            # ③
            tot_id += ok; tot += len(pm); cov += has
            rows.append(dict(DMS_id=r['DMS_id'], chain=c, n_var=n_var,
                             wt_len=len(W), pdb_res=len(st.get(c, '')),
                             seq_X=W.count('X'), dms_pos=len(pm),
                             pos_range=f"{min(pm)}-{max(pm)}",
                             wt_aa_match=f"{ok}/{len(pm)}",
                             has_coords=f"{has}/{len(pm)}"))

    df = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    print(df.to_string(index=False))
    print()
    print(f"① DMS 的 WT 氨基酸 vs wildtype_sequence : {tot_id}/{tot} = {tot_id/tot:.2%}")
    print(f"② X-占位符假设的反例                    : {viol if viol else '无'}")
    print(f"③ DMS 位点在自带 WT complex 中有坐标    : {cov}/{tot} = {cov/tot:.2%}")
    print()
    print(f"assay 数 {len(idx)} | 被突变的 (assay,chain) 数 {len(df)} | 总 variant 数 {int(df.groupby('DMS_id')['n_var'].first().sum())}")


if __name__ == '__main__':
    main()
