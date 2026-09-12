#!/usr/bin/env python
"""BindingGYM zero-shot 打分 —— ADFLIP (joint-masked 口径 B)。

【为什么不需要给 ADFLIP 加新 head】
ADFLIP 的 denoiser 本来就在 center 位输出 amino-acid logits：
    data, noisy = model.corrupt_data_by_sample(data, t, samples)   # samples 里 <MASK> 的残基
    logits, _   = model.model(noisy, t)                            #   连同其侧链原子一起被移除
把该 variant 的整组突变位点置 <MASK>、其余位点保持 WT(含全部侧链坐标)，一次 forward 得到的
logits 就是 p(aa_j | all-atom structure, rest-of-sequence) —— 正是所需的 structure–seq
compatibility likelihood。所谓 "header" 实为打分协议 wrapper，无新参数、无需训练。

score = Σ_j [ log p(mt_j) − log p(wt_j) ]，与其余三个模型的口径 B 严格同构。

【成本】masked 上下文只取决于位点集合、与替换成哪个 AA 无关 ⇒ 同一位点组合的所有 variant
共享一次 forward。全库 32,977 个 distinct 组合 vs 376,446 个 variant，约 11.4× 便宜。
"""
import argparse, ast, itertools, os, sys, time
import numpy as np, pandas as pd, torch


class Config:
    """🔴 ADFLIP_v1.pt 是用 test/benchmark.py 里定义在 __main__ 的 Config 类 pickle 的,
    反序列化时 unpickler 会去 __main__ 找同名类。本脚本作为 __main__ 运行,所以必须在这里
    原样定义它,否则 torch.load 报 "Can't get attribute 'Config' on <module '__main__'>"。
    定义逐字取自 ADFLIP/test/benchmark.py。"""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    def to_dict(self):
        return {k: (v.to_dict() if isinstance(v, Config) else v)
                for k, v in self.__dict__.items()}


def _align_map(obs, ref):
    """NW 比对,返回 {obs_idx: ref_idx}。与 score_bgym_mpnn.py 同一实现。"""
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


def build_model(repo, ckpt_path, device):
    sys.path.insert(0, repo)
    from data.residue_config import configure as configure_residues
    from model.discrete_flow_aa import DiscreteFlow_AA
    from model.zoidberg.zoidberg_GNN import Zoidberg_GNN
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    cfg.training.label_smoothing = True
    configure_residues(include_nonstd_amino_acids=getattr(cfg.data, "include_nonstd_amino_acids", True))
    z = cfg.zoidberg_denoiser
    den = Zoidberg_GNN(
        hidden_dim=z.hidden_dim, encoder_hidden_dim=z.hidden_dim, num_blocks=z.num_layers,
        num_heads=z.num_heads, k=z.k_neighbors,
        num_positional_embeddings=z.num_positional_embeddings, num_rbf=z.num_rbf,
        augment_eps=z.augment_eps, backbone_diheral=z.backbone_diheral, dropout=z.dropout,
        update_atom=z.update_atom, num_decoder_blocks=z.num_decoder_blocks,
        num_tfmr_heads=z.num_tfmr_heads, num_tfmr_layers=z.num_tfmr_layers,
        # 🔴 这三个是必须的:output_dim 决定输出词表大小(ckpt 是 33,漏传会退到默认 20
        #    并报 "size mismatch for model.layers.output.weight [33,128] vs [20,128]")。
        #    构造参数逐字对齐 ADFLIP/test/benchmark.py:248-266。
        number_ligand_atom=z.number_ligand_atom, mpnn_cutoff=z.mpnn_cutoff,
        output_dim=z.output_dim)
    # sidechain_packing=False:我们只要 logits,不做侧链重建(避免加载 PIPPack 权重)
    m = DiscreteFlow_AA(cfg, den, min_t=0.0, sidechain_packing=False)
    m.load_state_dict(ck["model"])        # 与 benchmark.py 一致:严格加载 ckpt["model"]
    print(f"  权重严格加载成功;output_dim={z.output_dim}")
    return m.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dms_mapping", required=True)
    ap.add_argument("--dms_input", required=True)
    ap.add_argument("--structure_folder", required=True)
    ap.add_argument("--dms_index", type=int, required=True)
    ap.add_argument("--dms_output", required=True)
    ap.add_argument("--t", type=float, default=1.0, help="flow 的 time step;只 mask 极少位点 ⇒ t≈1")
    ap.add_argument("--run_id", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = pd.read_csv(a.dms_mapping); row = idx.iloc[a.dms_index]
    DMS_id = row["DMS_id"]
    out_csv = os.path.join(a.dms_output, f"{DMS_id}.csv")
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        print(f"[skip] {DMS_id} 已存在"); return

    model, cfg = build_model(a.repo, a.ckpt, dev)
    from data.all_atom_parse import residue_tokens, pdb2data, index_to_token, restype_3to1

    df = pd.read_csv(os.path.join(a.dms_input, f"{DMS_id}.csv"))
    MASK = residue_tokens["<MASK>"]
    all_g, t0 = [], time.time()

    for (POI, chain_ids), g in df.groupby(["POI", "chain_id"], sort=False):
        pdb = os.path.join(a.structure_folder, g["pdb_file"].values[0])
        data = pdb2data(pdb, dev)
        des = data["is_center"].bool() & data["is_protein"].bool()
        if "backbone_mask" in data: des = des & data["backbone_mask"].bool()
        wt_tok = data["residue_token"][des].clone()
        wt_seq = "".join(restype_3to1.get(index_to_token[int(i)], "X") for i in wt_tok)

        # 🔴 ADFLIP 的 parser 残基集合与 BindingGYM 不同(实测 4D5_HER2 1015 vs 1041 ——
        #    它丢掉 backbone 不完整的残基),所以不能要求数量相等。
        #    另:pdb2data 只保留 ndarray 字段,链【字母】丢失,只剩数值型 data["chain_id"]。
        #    ⇒ 按链块切分 + 试所有排列,取错配最少的指派;再逐块 NW 比对建映射。
        wt_ref = ast.literal_eval(row["wildtype_sequence"])
        order = [c for c in str(chain_ids)] or sorted(wt_ref)
        ref = "".join(wt_ref[c] for c in order)
        roff = {}; _p = 0
        for c in order: roff[c] = _p; _p += len(wt_ref[c])

        cid = data["chain_id"][des].cpu().numpy()
        seen, blocks = [], []
        for v in cid.tolist():
            if v not in seen: seen.append(v)
        for v in seen: blocks.append(np.where(cid == v)[0])
        assert len(blocks) == len(order), \
            f"{DMS_id}: ADFLIP 解析出 {len(blocks)} 条链,BindingGYM 有 {len(order)} 条"

        best = None
        for perm in itertools.permutations(range(len(order))):
            amap, mis = {}, 0
            for bi, pi in enumerate(perm):
                ch = order[pi]; sel = blocks[bi]
                sub = "".join(wt_seq[p] for p in sel)
                am = _align_map(sub, wt_ref[ch])
                mis += sum(1 for k, v in am.items() if sub[k] != wt_ref[ch][v] and sub[k] != "X")
                mis += (len(wt_ref[ch]) - len(am))        # 未匹配上的也计入代价
                for k, v in am.items(): amap[int(sel[k])] = roff[ch] + v
            if best is None or mis < best[0]: best = (mis, amap, perm)
        nmis, amap, perm = best
        bad = [(k, wt_seq[k], ref[v]) for k, v in amap.items()
               if wt_seq[k] != ref[v] and wt_seq[k] != "X"]
        assert not bad, f"{DMS_id}: 最优指派下 WT 仍不符,前 3 处 {bad[:3]}"
        assert len(amap) >= 0.5 * len(ref), f"{DMS_id}: 只映射上 {len(amap)}/{len(ref)},不可信"
        ref2obs = {v: k for k, v in amap.items()}
        print(f"  [map] {DMS_id}: ADFLIP 解析 {len(wt_seq)}/{len(ref)}, 链块指派 {perm}, "
              f"映射 {len(amap)} 个位点(未映射位不参与打分)")

        tok_of = {}   # 1-letter AA -> ADFLIP token id
        for t_, i_ in residue_tokens.items():
            aa = restype_3to1.get(t_)
            if aa and aa not in tok_of: tok_of[aa] = i_

        scores, cache = [], {}
        for i in g.index:
            mseq = ast.literal_eval(g.loc[i, "mutated_sequence"])
            mut_full = "".join(mseq[c] for c in order)
            # diff 用【观测坐标】,未映射到结构的突变位点直接跳过
            diff = [ref2obs[k] for k in range(len(ref))
                    if mut_full[k] != ref[k] and k in ref2obs]
            key = tuple(sorted(diff))
            if key not in cache:
                samples = wt_tok.clone().unsqueeze(0)
                for o in diff: samples[0, o] = MASK
                with torch.no_grad():
                    _, noisy = model.corrupt_data_by_sample(dict(data), a.t, samples[0])
                    logits, _ = model.model(noisy, torch.tensor([[a.t]], device=dev))
                lp = torch.log_softmax(logits.view(-1, logits.shape[-1]).float(), dim=-1)
                cache[key] = {o: lp[o].cpu().numpy() for o in diff}
            c_ = cache[key]; obs2ref = {v: k for k, v in ref2obs.items()}
            s = 0.0
            for o in diff:
                k = obs2ref[o]
                s += float(c_[o][tok_of[mut_full[k]]] - c_[o][tok_of[ref[k]]])
            scores.append(s)

        g = g.copy(); g["adflip_score"] = scores
        all_g.append(g)
        print(f"  POI={POI} chains={chain_ids}: {len(g)} variants, {len(cache)} 次 forward")

    out = pd.concat(all_g).sort_index()
    out["run_id"] = a.run_id; out["adflip_t"] = a.t
    assert len(out) == len(df), f"行数变了 {len(out)} vs {len(df)}"
    os.makedirs(a.dms_output, exist_ok=True); out.to_csv(out_csv, index=False)
    print(f"[done] {DMS_id} rows={len(out)} wall={time.time()-t0:.0f}s -> {out_csv}")


if __name__ == "__main__":
    main()
