#!/usr/bin/env python
"""BindingGYM zero-shot 打分 —— ProteinMPNN / LigandMPNN 家族。

数据管线逐字对齐 BindingGYM 官方 baselines/protein_mpnn/compute_fitness_multi_pdb.py：
  * 用 DMS csv 自带的 `mutated_sequence`(逐链 dict)，不自己做位点映射 ⇒ 规避全部对齐陷阱
  * randn(解码序噪声) 每个 POI 抽一次、组内所有 variant 共享
  * _scores 为【未归一】的 NLL 求和(官方第 46 行的长度归一是注释掉的)
  * design_score 用 mask*chain_mask，global_score 用 mask，均对 M 个解码序取平均后取负

与官方的唯一差异：模型换成 LigandMPNN repo 的 ProteinMPNN(可开 ligand/side-chain context)。

两种口径：
  --protocol ar  autoregressive teacher-forced NLL          (官方口径，与 anchor 可直接续表)
  --protocol jm  joint-masked：把该 variant 的整组突变位点一起排到解码序最前，
                 一次 decoder pass 得到这 k 个位点的 log-probs(各自条件于其余位点保持原样)。
                 同一位点组合的所有 variant 共享同一次 pass ⇒ 组合库上省 10× 以上。

输出：原 DMS csv 全部列 + 打分列，不过滤、不改行序。
"""
import argparse, ast, copy, os, sys, time
import numpy as np, pandas as pd, torch


def _scores(S, log_probs, mask):
    """逐字复刻官方 protein_mpnn_utils._scores —— 注意长度归一是【注释掉】的。"""
    crit = torch.nn.NLLLoss(reduction="none")
    # NLLLoss 的 target 必须是 int64。LigandMPNN 的 featurize 产出 int32,
    # 而官方 tied_featurize 产出 int64 —— 这里显式 .long() 对齐,不改变数值语义。
    loss = crit(log_probs.contiguous().view(-1, log_probs.size(-1)),
                S.long().contiguous().view(-1)).view(S.size())
    return torch.sum(loss * mask, dim=-1)          # 不除以 mask.sum()


@torch.no_grad()
def joint_masked_logprobs(model, fd, pos_idx):
    """把 pos_idx 这组位点一起排到解码序最前，一次 pass 取它们的 log_probs。

    复刻 single_aa_score 的解码序构造，但 order_mask 在【整组】位点上置 0
    ⇒ 这组位点最先解码、彼此互不可见、条件于结构 + 其余位点由后续解码(即不可见)。
    返回 [B, len(pos_idx), 21]
    """
    from model_utils import cat_neighbors_nodes
    S_true0, mask0 = fd["S"], fd["mask"]
    randn, B = fd["randn"], fd["batch_size"]
    _, L = S_true0.shape
    dev = S_true0.device

    h_V_enc, h_E_enc, E_idx_enc = model.encode(fd)

    # 🔴 方向由 probe_use_sequence.py 实测定案(2026-09-12),不是推理:
    #    order_mask=1 的位点得到 ~|randn|(大) ⇒ 解码序【靠后】⇒ 能看到先解码的位点 = 条件于其余序列
    #    order_mask=0 的位点得到 ~1e-4|randn|(小) ⇒ 解码序【靠前】⇒ 看不到任何序列 = backbone-only
    #    我们要的是「mask 掉这组位点、条件于其余全部」⇒ 这组位点必须【最后】解码 ⇒ 置 1,其余置 0。
    #    (先前写反了:若按 ones/pos_idx=0,得到的是 backbone-only 分数,而非口径 B。)
    order_mask = torch.zeros(L, device=dev).float()
    order_mask[pos_idx] = 1.0
    decoding_order = torch.argsort((order_mask + 0.0001) * torch.abs(randn))

    E_idx = E_idx_enc.repeat(B, 1, 1)
    perm = torch.nn.functional.one_hot(decoding_order, num_classes=L).float()
    omb = torch.einsum("ij, biq, bjp->bqp",
                       (1 - torch.triu(torch.ones(L, L, device=dev))), perm, perm)
    mask_attend = torch.gather(omb, 2, E_idx).unsqueeze(-1)
    mask_1D = mask0.view([1, L, 1, 1])
    mask_bw, mask_fw = mask_1D * mask_attend, mask_1D * (1.0 - mask_attend)

    S_true = S_true0.repeat(B, 1)
    h_V = h_V_enc.repeat(B, 1, 1)
    h_E = h_E_enc.repeat(B, 1, 1, 1)
    mask = mask0.repeat(B, 1)

    h_S = model.W_s(S_true)
    h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
    h_EXV_enc = cat_neighbors_nodes(h_V, cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx), E_idx)
    h_EXV_enc_fw = mask_fw * h_EXV_enc
    for layer in model.decoder_layers:
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
        h_ESV = mask_bw * h_ESV + h_EXV_enc_fw
        h_V = layer(h_V, h_ESV, mask)
    lp = torch.nn.functional.log_softmax(model.W_out(h_V), dim=-1)
    return lp[:, pos_idx, :]


def _align_map(obs_seq, ref_seq):
    """把观测到的残基序列比对到 BindingGYM 参考序列,返回 {观测下标: 参考下标}。

    用于既有 insertion code 又有缺口的链(实例:4ZFF/4ZFG 的抗体重链 H —— Kabat 编号带
    A/B/C 插入码,同时还缺几个残基 ⇒ 「按序」和「按编号」两种映射都不成立)。
    简单 Needleman-Wunsch(match +1 / mismatch -1 / gap -1);两条序列本是同一蛋白,
    比对唯一且几乎全等,只用来吸收插入与缺口。
    """
    n, m = len(obs_seq), len(ref_seq)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = -np.arange(n + 1); D[0, :] = -np.arange(m + 1)
    for i in range(1, n + 1):
        oi = obs_seq[i - 1]
        for j in range(1, m + 1):
            D[i, j] = max(D[i-1, j-1] + (1 if oi == ref_seq[j-1] else -1),
                          D[i-1, j] - 1, D[i, j-1] - 1)
    i, j, out = n, m, {}
    while i > 0 and j > 0:
        if D[i, j] == D[i-1, j-1] + (1 if obs_seq[i-1] == ref_seq[j-1] else -1):
            out[i-1] = j-1; i -= 1; j -= 1
        elif D[i, j] == D[i-1, j] - 1: i -= 1
        else: j -= 1
    return out



def _code_stamp():
    """本打分脚本自身的 md5 —— 写进输出,并作为幂等跳过的判据。

    🔴 为什么需要:脚本改了语义之后,旧的(错的)输出会因为「文件已存在」被静默跳过而保留下来。
       实测事故(2026-09-12):LASErMPNN 的链映射修复后重跑,25 个 csv 里 23 个是修复【之前】写的,
       只有 2 个用了新代码,而 job 报 COMPLETED 25/25 —— 聚合出的 0.3614 是废数。
       改为按 md5 戳跳过:戳不一致就重算。
    """
    import hashlib, os
    return hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]


def _should_skip(out_csv, stamp):
    import os
    import pandas as _pd
    if not (os.path.exists(out_csv) and os.path.getsize(out_csv) > 0):
        return False
    try:
        old = _pd.read_csv(out_csv, nrows=1)
        if "code_stamp" in old.columns and str(old["code_stamp"].iloc[0]) == stamp:
            return True
        print(f"  [restale] {os.path.basename(out_csv)} 的 code_stamp 与当前脚本不符,重算")
    except Exception:
        pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="LigandMPNN repo 路径")
    ap.add_argument("--dms_mapping", required=True)
    ap.add_argument("--dms_input", required=True)
    ap.add_argument("--structure_folder", required=True)
    ap.add_argument("--dms_index", type=int, required=True)
    ap.add_argument("--dms_output", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_type", default="protein_mpnn",
                    choices=["protein_mpnn", "ligand_mpnn", "soluble_mpnn"])
    ap.add_argument("--use_side_chain_context", type=int, default=0)
    ap.add_argument("--use_atom_context", type=int, default=1)
    ap.add_argument("--protocol", default="ar", choices=["ar", "jm"])
    ap.add_argument("--k_neighbors", type=int, default=0,
                    help="裸 state_dict(无 num_edges 元数据)时必须显式给;有元数据时用于交叉校验")
    ap.add_argument("--designed_chains", default="index", choices=["index", "mutated"],
                    help="index=照 BindingGYM 的 chain_id(全部链,与 anchor 一致); "
                         "mutated=只把【实际携带突变】的链设为 designed,partner 设 fixed。"
                         "后者是让 side-chain context 能起作用的唯一途径 —— LigandMPNN 的 "
                         "model_utils.py:1252 `xyz_37_m * (1 - chain_mask)` 决定了侧链只对 "
                         "fixed(chain_mask=0) 残基生效;BindingGYM 的 chain_id 把所有链都列为 "
                         "designed ⇒ chain_mask 全 1 ⇒ 侧链 context 恒为零(实测两个 config "
                         "五项指标小数点后六位完全相同)。")
    ap.add_argument("--num_seq_per_target", type=int, default=5, help="M：平均多少个解码序")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run_id", required=True)
    a = ap.parse_args()
    assert a.seed != 0, "seed 必须非 0(官方 `if args.seed:` 把 0 当 falsy 会退回随机)"

    sys.path.insert(0, a.repo)
    from data_utils import parse_PDB, featurize, alphabet, restype_str_to_int
    from model_utils import ProteinMPNN
    assert "".join(alphabet) == "ACDEFGHIKLMNPQRSTVWYX", f"alphabet 不符: {''.join(alphabet)}"

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx = pd.read_csv(a.dms_mapping)
    row = idx.iloc[a.dms_index]
    DMS_id = row["DMS_id"]
    out_csv = os.path.join(a.dms_output, f"{DMS_id}.csv")
    STAMP = _code_stamp()
    if _should_skip(out_csv, STAMP):
        print(f"[skip] {DMS_id} 已存在且 code_stamp 一致"); return

    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    # StaB-ddG 的 stability_finetuned.pt 是【裸 state_dict】,没有 num_edges/noise_level 元数据
    # (它的 stage1 proteinmpnn.pt 有,为 48/0.2;架构与 ProteinMPNN 118/118 参数名一致、零形状不符)。
    if "model_state_dict" not in ck:
        assert a.k_neighbors, "裸 state_dict 必须显式给 --k_neighbors"
        ck = {"model_state_dict": ck, "num_edges": a.k_neighbors, "noise_level": "n/a"}
    elif a.k_neighbors:
        assert ck["num_edges"] == a.k_neighbors, \
            f"ckpt 的 num_edges={ck['num_edges']} 与 --k_neighbors={a.k_neighbors} 不符"
    atom_ctx = ck["atom_context_num"] if a.model_type == "ligand_mpnn" else 1
    model = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128,
                        num_encoder_layers=3, num_decoder_layers=3,
                        k_neighbors=ck["num_edges"], device=dev,
                        atom_context_num=atom_ctx, model_type=a.model_type,
                        ligand_mpnn_use_side_chain_context=a.use_side_chain_context)
    model.load_state_dict(ck["model_state_dict"]); model.to(dev).eval()
    print(f"{DMS_id} | {a.model_type} | k={ck['num_edges']} | noise={ck.get('noise_level','?')} "
          f"| sc_ctx={a.use_side_chain_context} | protocol={a.protocol} | M={a.num_seq_per_target}")

    df = pd.read_csv(os.path.join(a.dms_input, f"{DMS_id}.csv"))
    M = a.num_seq_per_target
    all_g, t0 = [], time.time()

    for (POI, chain_ids), g in df.groupby(["POI", "chain_id"], sort=False):
        pdb = os.path.join(a.structure_folder, g["pdb_file"].values[0])
        # 🔴 parse_atoms_with_zero_occupancy=True 是必须的,且是【口径正确】的选择:
        #    BindingGYM 全部 22 个结构的 occupancy 都是 0.00(多为同源模型 *_hm.pdb),
        #    LigandMPNN 默认会 select("occupancy > 0") ⇒ 选中 0 个原子、返回 None 再崩溃。
        #    而官方 BindingGYM 的 parse_PDB 根本不看 occupancy ⇒ anchor 那轮用的是全部原子。
        #    置 True 才与 anchor 逐原子一致。
        pdict, _, _, icodes, _ = parse_PDB(pdb, device=dev,
                                           parse_all_atoms=bool(a.use_side_chain_context),
                                           parse_atoms_with_zero_occupancy=True)
        assert pdict is not None and pdict["mask"].numel() > 0, f"{DMS_id}: parse_PDB 返回空"
        letters = np.array([str(c) for c in pdict["chain_letters"]])
        R_idx = pdict["R_idx"].cpu().numpy().astype(int)   # PDB 残基编号
        L = len(letters)
        if a.designed_chains == "mutated":
            mc = set()
            for m in g["mutant"]:
                d = ast.literal_eval(m) if str(m).startswith("{") else {}
                for ch, s in d.items():
                    if str(s).strip(): mc.add(ch)
            designed = sorted(mc) if mc else ([c for c in str(chain_ids)] or sorted(set(letters)))
            fixedc = [c for c in (str(chain_ids) or sorted(set(letters))) if c not in designed]
            print(f"  [chains] designed={''.join(designed)} fixed={''.join(fixedc) or '(无)'}"
                  + ("" if fixedc else "  ← 无 fixed 链 ⇒ 侧链 context 仍为零"))
        else:
            designed = [c for c in str(chain_ids)] if str(chain_ids) else sorted(set(letters))
        pdict["chain_mask"] = torch.tensor(
            [1 if c in designed else 0 for c in letters], device=dev).float()

        fd = featurize(pdict, use_atom_context=bool(a.use_atom_context),
                       number_of_ligand_atoms=atom_ctx, model_type=a.model_type)
        fd["batch_size"] = M; fd["symmetry_residues"] = [[]]

        # 🔴 两个 parser 的残基集合不同,必须显式补洞:
        #    BindingGYM 的 wildtype_sequence 按【残基编号跨度】索引;官方 ProteinMPNN 的
        #    parse_PDB 会按编号补洞(缺失位 mask=0,对 _scores 贡献为 0);而 LigandMPNN 的
        #    parse_PDB 只返回【观测到的】残基。
        #    实例 3KZ0 链A:观测 143 个、编号 172..321(跨度 150,缺 196-202),BindingGYM 序列长 150。
        #    ⇒ 按 (chain, 编号-该链最小编号) 建映射;未观测位不参与打分,与 anchor 的 mask=0 等价。
        wt_ref = ast.literal_eval(row["wildtype_sequence"])
        struct_seq = "".join(alphabet[int(i)] for i in fd["S"][0].cpu().numpy())
        pos_map = {}                  # chain -> {BindingGYM 序列位(0-based): featurize 下标}
        for c in designed:
            sel = np.where(letters == c)[0]
            assert len(sel) > 0, f"{DMS_id}: 结构里没有链 {c}"
            rn = R_idx[sel]; lo, hi = int(rn.min()), int(rn.max())
            span, nref, nobs = hi - lo + 1, len(wt_ref[c]), len(sel)
            obs_seq = "".join(struct_seq[p] for p in sel)
            # 三条映射路径,按可靠性排序;每条都在下面用 WT 逐位校验,选错会被 assert 拦住。
            if nobs == nref and sum(obs_seq[k] != wt_ref[c][k] and obs_seq[k] != "X"
                                    for k in range(nref)) == 0:
                mode = "sequential"                       # 无缺口(可含 insertion code)
                pos_map[c] = {k: int(sel[k]) for k in range(nref)}
            elif span == nref:
                mode = "resnum-span"                      # 有缺口、编号干净(如 3KZ0 链A)
                pos_map[c] = {int(r) - lo: int(p) for r, p in zip(rn, sel)}
            else:
                # 既有 insertion code 又有缺口(如 4ZFF/4ZFG 的抗体重链 H):比对定位
                mode = "align"
                am = _align_map(obs_seq, wt_ref[c])
                pos_map[c] = {v: int(sel[k]) for k, v in am.items()}
            if nobs != nref or mode != "sequential":
                print(f"  [map] {DMS_id} 链{c}: mode={mode}, 观测 {nobs}/{nref}, "
                      f"编号 {lo}..{hi}(跨度 {span}), 映射到 {len(pos_map[c])} 个位点"
                      f"{'' if nobs==nref else '(未映射位不参与打分,与 anchor 的 mask=0 一致)'}")
            bad = [(k, struct_seq[p], wt_ref[c][k]) for k, p in pos_map[c].items()
                   if struct_seq[p] != wt_ref[c][k] and struct_seq[p] != "X"]
            assert not bad, f"{DMS_id} 链{c}: WT 不符,前 3 处 {bad[:3]}"

        gen = torch.Generator(device=dev).manual_seed(a.seed)
        randn = torch.randn([M, L], device=dev, generator=gen)   # 每个 POI 一次，组内共享
        fd["randn"] = randn
        S_wt = fd["S"].clone()

        des, glo, cache = [], [], {}
        for i in g.index:
            mseq = ast.literal_eval(g.loc[i, "mutated_sequence"])
            S = S_wt.clone()
            for c in designed:
                seqc = mseq[c]; ks = sorted(pos_map[c])
                tok = torch.tensor([restype_str_to_int[seqc[k]] for k in ks],
                                   device=dev, dtype=S.dtype)   # dtype 必须与 S 一致(int32)
                S[0, [pos_map[c][k] for k in ks]] = tok
            fd["S"] = S

            if a.protocol == "ar":
                with torch.no_grad():
                    o = model.score(fd, use_sequence=True)
                lp = o["log_probs"]
                Sb = S.repeat(M, 1)
                d = _scores(Sb, lp, (fd["mask"] * fd["chain_mask"]).repeat(M, 1))
                gl = _scores(Sb, lp, fd["mask"].repeat(M, 1))
                des.append(-float(d.mean())); glo.append(-float(gl.mean()))
            else:
                diff = np.where((S[0] != S_wt[0]).cpu().numpy())[0]
                key = tuple(diff.tolist())
                if key not in cache:
                    fd["S"] = S_wt                     # masked 上下文 = WT，与替换成哪个 AA 无关
                    cache[key] = joint_masked_logprobs(
                        model, fd, torch.tensor(diff, device=dev, dtype=torch.long)
                    ).mean(0).cpu().numpy() if len(diff) else np.zeros((0, 21))
                lp = cache[key]
                s = sum(float(lp[j, int(S[0, p])] - lp[j, int(S_wt[0, p])])
                        for j, p in enumerate(diff))
                des.append(s); glo.append(s)

        g = g.copy(); g["design_score"] = des; g["global_score"] = glo
        all_g.append(g)
        if a.protocol == "jm":
            print(f"  POI={POI} chains={chain_ids}: {len(g)} variants, {len(cache)} 次 pass")

    out = pd.concat(all_g).sort_index()
    out["seed"] = a.seed; out["run_id"] = a.run_id; out["code_stamp"] = STAMP
    assert len(out) == len(df), f"行数变了 {len(out)} vs {len(df)} —— 不允许过滤"
    os.makedirs(a.dms_output, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[done] {DMS_id}  rows={len(out)}  wall={time.time()-t0:.0f}s -> {out_csv}")


if __name__ == "__main__":
    main()
