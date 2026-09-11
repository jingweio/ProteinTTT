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
import argparse, ast, os, sys, time
import numpy as np, pandas as pd, torch


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
        num_tfmr_heads=z.num_tfmr_heads, num_tfmr_layers=z.num_tfmr_layers)
    m = DiscreteFlow_AA(cfg, den)
    sd = ck.get("ema_model", ck.get("model", ck.get("state_dict", ck)))
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    missing, unexpected = m.load_state_dict(
        {k.replace("ema_model.", "").replace("online_model.", ""): v
         for k, v in sd.items()}, strict=False)
    print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    assert len(missing) < 20, f"权重加载明显不对,missing={missing[:10]}"
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

        # 🔴 硬校验：ADFLIP 解析出的 WT 必须与 BindingGYM 的 wildtype_sequence 拼接一致。
        #    这同时【建立了】位点索引映射 —— 对不上就 assert，绝不静默错位。
        wt_ref = ast.literal_eval(row["wildtype_sequence"])
        order = [c for c in str(chain_ids)] or sorted(wt_ref)
        ref = "".join(wt_ref[c] for c in order)
        assert len(wt_seq) == len(ref), f"{DMS_id}: ADFLIP 解析 {len(wt_seq)} 残基 vs BindingGYM {len(ref)}"
        bad = [(k, wt_seq[k], ref[k]) for k in range(len(ref)) if wt_seq[k] != ref[k] and wt_seq[k] != "X"]
        assert not bad, f"{DMS_id}: WT 不符，前 3 处 {bad[:3]}"
        off = {}; p = 0
        for c in order: off[c] = p; p += len(wt_ref[c])

        tok_of = {}   # 1-letter AA -> ADFLIP token id
        for t_, i_ in residue_tokens.items():
            aa = restype_3to1.get(t_)
            if aa and aa not in tok_of: tok_of[aa] = i_

        scores, cache = [], {}
        for i in g.index:
            mseq = ast.literal_eval(g.loc[i, "mutated_sequence"])
            mut_full = "".join(mseq[c] for c in order)
            diff = [k for k in range(len(ref)) if mut_full[k] != ref[k]]
            key = tuple(diff)
            if key not in cache:
                samples = wt_tok.clone().unsqueeze(0)
                for k in diff: samples[0, k] = MASK
                with torch.no_grad():
                    _, noisy = model.corrupt_data_by_sample(dict(data), a.t, samples[0])
                    logits, _ = model.model(noisy, torch.tensor([[a.t]], device=dev))
                lp = torch.log_softmax(logits.view(-1, logits.shape[-1]).float(), dim=-1)
                cache[key] = {k: lp[k].cpu().numpy() for k in diff}
            c_ = cache[key]
            s = 0.0
            for k in diff:
                s += float(c_[k][tok_of[mut_full[k]]] - c_[k][tok_of[ref[k]]])
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
