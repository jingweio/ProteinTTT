#!/usr/bin/env python
"""mutant vs WT(同 seed) 的 backbone deviation 分析。

四个口径（都用 CA + Kabsch 叠合）：
  rmsd_all        全复合物            —— 含链间 docking pose
  rmsd_mutchain   仅被突变链自身叠合  —— 链内 fold 变化
  rmsd_site_local 仅 binding-site 叠合 —— 界面局部构象
  rmsd_site_inchn 按突变链叠合后只测 site —— site 在链坐标系里的位移

噪声底：WT seed0 vs seed1..4，同样四个口径。
"""
import csv, gzip, os, sys, json, itertools, collections
import numpy as np

OUT_ROOT = sys.argv[1]
PROJ     = sys.argv[2]
OUT_JSON = sys.argv[3]

D = {'wt': 'BindingGYM-esmfold2-fast-predicted-wt-complex-structure',
     'mutant': 'BindingGYM-esmfold2-fast-predicted-mutant-complex-structure'}

# binding sites：1-based 链内位置（D1 口径，来自 bindingGYM-binding-sites-analysis 的 site_seqpos）
SITES = {
 'SARS2-RBD_ACE2_deltaKd_6M0J':  ('E', [85,114,115,117,121,123,124,141,143,144,152,154,155,157,161,164,166,168,169,170,173]),
 'CXCR4_CXCL12_enrich_8U4O':     ('R', [1,2,3,4,5,6,7,14,18,22,71,74,75,90,93,155,156,157,158,162,163,164,166,170,177,180,232,236,239,243,245,254,258,261,262,265,269]),
 'hYAP65_peptide_FunctioncalScore_1JMQ': ('A', [18,22,24,26,27,28,31,32,33,34,35]),
}


def load_ca(path):
    """返回 {chain: (N,3) ndarray}，按链内出现顺序（= 1-based 序列位置）。"""
    per = collections.defaultdict(list)
    with gzip.open(path, 'rt') as f:
        for l in f:
            if l.startswith('ATOM'):
                F = l.split()
                if F[2] == 'CA':
                    per[F[5]].append((float(F[15]), float(F[16]), float(F[17])))
    return {c: np.asarray(v, dtype=np.float64) for c, v in per.items()}


def kabsch_rmsd(P, Q):
    if P.shape != Q.shape or len(P) < 3:
        return float('nan')
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, W = np.linalg.svd(Pc.T @ Qc)
    if np.linalg.det(V) * np.linalg.det(W) < 0:
        S[-1] = -S[-1]; V[:, -1] = -V[:, -1]
    return float(np.sqrt(((Pc @ (V @ W) - Qc) ** 2).sum() / len(P)))


def kabsch_apply(P, Q):
    """把 P 叠合到 Q，返回叠合后的 P（以及两者的质心平移）。"""
    pm, qm = P.mean(0), Q.mean(0)
    Pc, Qc = P - pm, Q - qm
    V, S, W = np.linalg.svd(Pc.T @ Qc)
    if np.linalg.det(V) * np.linalg.det(W) < 0:
        S[-1] = -S[-1]; V[:, -1] = -V[:, -1]
    return (Pc @ (V @ W)), Qc


def four_metrics(A, B, assay):
    """A,B: {chain: CA}. 返回四个口径的 RMSD。"""
    chains = sorted(set(A) & set(B))
    ca_A = np.vstack([A[c] for c in chains])
    ca_B = np.vstack([B[c] for c in chains])
    out = {'rmsd_all': kabsch_rmsd(ca_A, ca_B)}

    mch, sites = SITES[assay]
    idx = [p - 1 for p in sites]
    if mch in A and mch in B and A[mch].shape == B[mch].shape:
        out['rmsd_mutchain'] = kabsch_rmsd(A[mch], B[mch])
        n = len(A[mch])
        ok = [i for i in idx if 0 <= i < n]
        if len(ok) >= 3:
            out['rmsd_site_local'] = kabsch_rmsd(A[mch][ok], B[mch][ok])
            al, bl = kabsch_apply(A[mch], B[mch])          # 按整条突变链叠合
            d = np.linalg.norm(al[ok] - bl[ok], axis=1)
            out['rmsd_site_inchn'] = float(np.sqrt((d ** 2).mean()))
        else:
            out['rmsd_site_local'] = out['rmsd_site_inchn'] = float('nan')
    else:
        out['rmsd_mutchain'] = out['rmsd_site_local'] = out['rmsd_site_inchn'] = float('nan')
    return out


METRICS = ['rmsd_all', 'rmsd_mutchain', 'rmsd_site_local', 'rmsd_site_inchn']

rows = list(csv.DictReader(gzip.open(PROJ + '/refs/fold_manifest.csv.gz', 'rt')))
assays = sorted({r['assay'] for r in rows if r['assay'] in SITES})
print(f"assay: {assays}", flush=True)

result = {}
for assay in assays:
    wt = {}
    for sd in range(5):
        p = os.path.join(OUT_ROOT, D['wt'], f"{assay}__WT__seed{sd}.cif.gz")
        if os.path.exists(p):
            wt[sd] = load_ca(p)
    ref = wt[0]
    # 噪声底：WT seed0 vs seed1..4（同口径），另外也给所有 seed 两两配对
    noise = {m: [] for m in METRICS}
    for a, b in itertools.combinations(sorted(wt), 2):
        f = four_metrics(wt[a], wt[b], assay)
        for m in METRICS:
            noise[m].append(f[m])
    noise_vs0 = {m: [] for m in METRICS}
    for sd in sorted(wt):
        if sd == 0: continue
        f = four_metrics(wt[0], wt[sd], assay)
        for m in METRICS:
            noise_vs0[m].append(f[m])

    per = []
    for r in rows:
        if r['assay'] != assay or r['kind'] != 'mutant':
            continue
        p = os.path.join(OUT_ROOT, D['mutant'], r['id'] + '.cif.gz')
        if not os.path.exists(p):
            continue
        f = four_metrics(load_ca(p), ref, assay)
        f.update(id=r['id'], k=int(r['k']),
                 dms=float(r['dms_score']) if r['dms_score'] not in ('', None) else float('nan'))
        per.append(f)
    print(f"  {assay}: {len(per)} mutants", flush=True)
    result[assay] = dict(n=len(per), noise_pairwise=noise, noise_vs_seed0=noise_vs0,
                         mutated_chain=SITES[assay][0], n_site=len(SITES[assay][1]),
                         per_mutant=per)

json.dump(result, open(OUT_JSON, 'w'))
print("WROTE", OUT_JSON, flush=True)
