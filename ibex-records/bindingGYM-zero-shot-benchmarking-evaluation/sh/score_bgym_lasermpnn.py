#!/usr/bin/env python
"""BindingGYM zero-shot 打分 —— LASErMPNN。

LASErMPNN 的 `utils/model.py:get_logits_for_score()` 是 teacher-forced 打分入口,
但 **repo 内无人调用**,故 harness 自写。

⚠️ 两个静默错位陷阱(都已加 assert):
 1. **AA 索引顺序与 ProteinMPNN 不同** —— LASErMPNN 按三字母码字母序
    (ALA=0,ARG=1,ASN=2,ASP=3,CYS=4,GLU=5,GLN=6,...),而 ProteinMPNN 是 ACDEFGHIKLMNPQRSTVWYX。
    一律用它自己的 `aa_short_to_idx`,并断言 WT 解码回来与 BindingGYM 一致。
 2. **chi teacher-forcing 的已知不一致** —— 打分位点 i 的 logits 只依赖解码序在它【之前】
    位点的 seq+chi;位点 i 自身的 chi 不参与 ⇒ 单点突变无问题。多点突变时,先解码的突变位点
    会带着【WT 的 chi】配【mutant 的 aa】。这是结构-only 输入的固有限制,如实记录、不掩盖。

BindingGYM 是 protein-protein,无小分子 ⇒ 按 run_inference 的 --ignore_ligand 路径清空 ligand 张量。
"""
import argparse, ast, os, sys, time
import numpy as np, pandas as pd, torch


def _align_map(obs, ref):
    """Needleman-Wunsch,返回 {观测下标: 参考下标}。与 score_bgym_mpnn.py 同一实现。"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="含 LASErMPNN 包的【父】目录")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dms_mapping", required=True)
    ap.add_argument("--dms_input", required=True)
    ap.add_argument("--structure_folder", required=True)
    ap.add_argument("--dms_index", type=int, required=True)
    ap.add_argument("--dms_output", required=True)
    ap.add_argument("--protocol", default="ar", choices=["ar", "jm"])
    ap.add_argument("--num_decoding_orders", type=int, default=5, help="平均多少个随机解码序(对齐 anchor 的 M=5)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run_id", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    from LASErMPNN.run_inference import (get_protein_hierview, ProteinComplexData,
                                         load_model_from_parameter_dict)
    from LASErMPNN.utils.constants import aa_short_to_idx, aa_idx_to_short
    assert aa_short_to_idx["A"] == 0 and aa_short_to_idx["R"] == 1 and aa_short_to_idx["C"] == 4, \
        f"LASErMPNN 的 AA 索引顺序变了,打分会错位: {aa_short_to_idx}"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    idx = pd.read_csv(a.dms_mapping); row = idx.iloc[a.dms_index]
    DMS_id = row["DMS_id"]
    out_csv = os.path.join(a.dms_output, f"{DMS_id}.csv")
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        print(f"[skip] {DMS_id} 已存在"); return

    model, params = load_model_from_parameter_dict(a.weights, str(dev), strict=True)
    model.eval()
    print(f"{DMS_id} | LASErMPNN | protocol={a.protocol} | M={a.num_decoding_orders}")

    df = pd.read_csv(os.path.join(a.dms_input, f"{DMS_id}.csv"))
    all_g, t0 = [], time.time()

    for (POI, chain_ids), g in df.groupby(["POI", "chain_id"], sort=False):
        pdb = os.path.join(a.structure_folder, g["pdb_file"].values[0])
        data = ProteinComplexData(get_protein_hierview(pdb), pdb)
        batch = data.output_batch_data(fix_beta=False)
        # BindingGYM 无小分子 —— 照 run_inference 的 --ignore_ligand 清空 ligand 输入
        u = batch.unprocessed_ligand_input_data
        u.lig_coords = torch.empty(0, 3); u.lig_batch_indices = torch.empty(0, dtype=torch.long)
        u.lig_subbatch_indices = torch.empty(0, dtype=torch.long)
        u.lig_burial_maskmask = torch.empty(0, dtype=torch.bool)
        u.lig_atomic_numbers = torch.empty(0, dtype=torch.long)

        batch.to_device(model.device)
        batch.construct_graphs(model.rotamer_builder, model.ligand_featurizer,
                               **params["model_params"]["graph_structure"],
                               protein_training_noise=0.0, ligand_training_noise=0.0,
                               subgraph_only_dropout_rate=0.0, num_adjacent_residues_to_drop=0,
                               build_hydrogens=params["model_params"]["build_hydrogens"])

        wt_idx = batch.sequence_indices.clone()
        chi = batch.chi_angles.clone()
        wt_seq = "".join(aa_idx_to_short[int(i)] for i in wt_idx)

        # 🔴 LASErMPNN 会丢掉 backbone 不完整的残基(实测 BH3_Bcl-xL 180 vs BindingGYM 229),
        #    所以不能要求数量相等 —— 用比对建立【观测下标 -> 参考下标】映射,未映射位不参与打分。
        wt_ref = ast.literal_eval(row["wildtype_sequence"])
        order = [c for c in str(chain_ids)] or sorted(wt_ref)
        ref = "".join(wt_ref[c] for c in order)
        amap = _align_map(wt_seq, ref)              # obs_idx -> ref_idx
        mism = [(k, wt_seq[k], ref[v]) for k, v in amap.items() if wt_seq[k] != ref[v] and wt_seq[k] != "X"]
        assert not mism, f"{DMS_id}: 比对后 WT 仍不符,前 3 处 {mism[:3]}"
        assert len(amap) >= 0.5 * len(wt_seq), \
            f"{DMS_id}: 比对只匹配上 {len(amap)}/{len(wt_seq)},映射不可信"
        ref2obs = {v: k for k, v in amap.items()}   # 参考下标 -> 观测下标
        if len(wt_seq) != len(ref):
            print(f"  [map] {DMS_id}: LASErMPNN 解析 {len(wt_seq)}/{len(ref)},"
                  f" 比对映射 {len(amap)} 个位点(未映射位不参与打分)")

        L = len(wt_seq)
        gen = torch.Generator(device="cpu").manual_seed(a.seed)
        randn = torch.randn([a.num_decoding_orders, L], generator=gen).to(model.device)

        scores, cache = [], {}
        for i in g.index:
            mseq = ast.literal_eval(g.loc[i, "mutated_sequence"])
            mut_full = "".join(mseq[c] for c in order)
            diff = [ref2obs[k] for k in range(len(ref))
                    if mut_full[k] != ref[k] and k in ref2obs]   # 观测坐标
            S = wt_idx.clone()
            for k, o in ref2obs.items():
                S[o] = aa_short_to_idx.get(mut_full[k], 20)

            if a.protocol == "ar":
                tot = 0.0
                for m in range(a.num_decoding_orders):
                    dorder = torch.argsort(torch.abs(randn[m]))
                    with torch.no_grad():
                        lg = model.get_logits_for_score(batch, dorder, S, chi)[0]
                    lp = torch.log_softmax(lg.float(), dim=-1)
                    tot += float(lp.gather(1, S.unsqueeze(-1)).sum())     # 未归一的 NLL 和(对齐官方 _scores)
                scores.append(tot / a.num_decoding_orders)
            else:
                key = tuple(diff)
                if key not in cache:
                    # 方向同 score_bgym_mpnn(probe 实测):要条件于其余序列,这组位点须【最后】解码
                    om = torch.zeros(L, device=model.device); om[diff] = 1.0
                    acc = None
                    for m in range(a.num_decoding_orders):
                        dorder = torch.argsort((om + 1e-4) * torch.abs(randn[m]))
                        with torch.no_grad():
                            lg = model.get_logits_for_score(batch, dorder, wt_idx, chi)[0]
                        lp = torch.log_softmax(lg.float(), dim=-1)
                        acc = lp if acc is None else acc + lp
                    cache[key] = (acc / a.num_decoding_orders).cpu().numpy()
                lp = cache[key]
                obs2ref = {v: k for k, v in ref2obs.items()}
                scores.append(sum(float(lp[o, aa_short_to_idx[mut_full[obs2ref[o]]]] -
                                        lp[o, aa_short_to_idx[ref[obs2ref[o]]]]) for o in diff))

        g = g.copy(); g["laser_score"] = scores
        all_g.append(g)
        print(f"  POI={POI} chains={chain_ids}: {len(g)} variants"
              + (f", {len(cache)} 个位点组合" if a.protocol == "jm" else ""))

    out = pd.concat(all_g).sort_index()
    out["seed"] = a.seed; out["run_id"] = a.run_id
    assert len(out) == len(df), f"行数变了 {len(out)} vs {len(df)}"
    os.makedirs(a.dms_output, exist_ok=True); out.to_csv(out_csv, index=False)
    print(f"[done] {DMS_id} rows={len(out)} wall={time.time()-t0:.0f}s -> {out_csv}")


if __name__ == "__main__":
    main()
