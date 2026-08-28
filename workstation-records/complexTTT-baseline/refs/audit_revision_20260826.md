# 修订指令 — complexTTT_baseline_plan_20260826-095745.md

适用范围：`/home/guoj0f/repos/ProteinTTT/.claude/worktrees/proteinTTT-proteinGYM-reproduce/workstation-records/complexTTT-baseline/complexTTT_baseline_plan_20260826-095745.md`（325 行）。行号均指该文件当前版本。

两处需要先说明的**审计间冲突裁决**（下文按裁决结果写）：
- **噪声地板 0.008**：success-criterion 维度要求整块删除；feasibility-budget 维度在官方代码路径上**直接实测**得 σ(M=20)=0.0093（6 seeds）。以直接实测为准 ⇒ M=20 那一格的数值**保留**，但 M=5 那一格（0.0136）被两次独立 6-seed 实测否证（0.0187 / 0.0205），必须改；绝对值（0.6919）与「可外推到 25 assay」两点仍然不成立。
- **S0 预算**：success-criterion 维度用 223/s 平铺推出「M=5 ⇒ 2.34 h」；feasibility-budget 维度实测官方脚本三个完整 assay + 四个长复合物子集。以实测为准 ⇒ 全量 S0 在 M=20 下 ≈28 A4500-h、在官方 M=5 下 ≈14–16 h，「2.34 h」和「9.4 h」都不能用。

---

## 1. BLOCKING corrections

### B1. checkpoint 钉错了模型（4/6 个审计维度独立命中）

- **plan 说**（line 248）：`checkpoint：/home/guoj0f/repos/StaB-ddG/model_ckpts/proteinmpnn.pt`；§1 的噪声地板、§2 全节的否证、§7 S0 的「复现 +0.3970」都建立在它上面。
- **实际**：BindingGYM 官方 launcher 载入的是 `{checkpoint_folder}/v_48_020.pt`（**vanilla**）；StaB-ddG 那份按其 README 是 **soluble** 权重。两文件字节数相同（6,681,301 B）、meta 相同（`{'num_edges':48,'noise_level':0.2}`）、118 个 key 相同 —— 但 **118/118 个 tensor 全部不同**，max|Δ| = 9.0867（`decoder_layers.2.dense.W_out.weight`），全局 cosine 0.0090，relative L2 1.346，‖θ‖₂ = 373.94 vs 410.90。可测后果：官方路径 + vanilla 在 BH3_Mcl-1 上 M=20 mean = 0.6662（published 0.6625），而 soluble 记的是 0.6919 —— 差 0.026 ≈ 2.8σ。
- **证据**：`modelzoo/proteinmpnn/run.py:25`；`StaB-ddG/README.md:50`「proteinmpnn.pt # soluble ProteinMPNN weights」；md5 698982b1bda2b0d42e26538e64c93fda vs 91d54c97a68bf551114f8c74c785e90f；sha256(v_48_020) `c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd`；`Sources/complex-ttt-evidence/logs/noise.log:1` `||theta||_2=373.9`。
- **精确编辑**：
  1. line 248 改为 `checkpoint：/home/guoj0f/repos/BindingGYM/training/cache/v_48_020.pt`（vanilla，本机已有），并写入 sha256，harness 启动时 assert 该 hash。
  2. §2 开头加一行：「§2.1–2.3 与 §1 噪声地板全部跑在 StaB-ddG 的 **soluble** 权重上，与官方 v_48_020 **零个 tensor 相同**（cosine 0.0090）；这些数字的**绝对值与 BindingGYM 榜单不可比**，且『vanilla 权重上是否同样崩坏』未测。」措辞用「不是同一个模型」，**不要**写成「soluble 更差」——单 assay 上 0.6822 vs 0.6701 的差落在 0.019 的 run-to-run σ 内。
  3. §7 S1a 必须包含一格「vanilla v_48_020 上重跑 BH3_Mcl-1 的 §2.1 lr 扫描」作为桥接，否则 §2 无法被引用为对本 plan 的否证。

### B2. 「复用官方口径」与「M=20」自相矛盾；官方口径是 M=5（4/6 命中）

- **plan 说**（line 245-246）：「M=20 个随机解码顺序取平均」+「复用 BindingGYM 官方 compute_fitness_multi_pdb.py 的口径」。
- **实际**：官方 launcher 两个分支都硬编码 `--num_seq_per_target 5`；`NUM_BATCHES = args.num_seq_per_target` 就是解码顺序数（`batch_clones` 复制 5 份结构、`randn_1` 形状 [5,L]、`native_score.mean()` 对这 5 行取均值）。`BATCH_COPIES = args.batch_size` 在全文件中只被赋值、**从不使用**（grep 全文唯一命中 line 58），所以 `--batch_size 8` 是死参数，有效 M 既不是 40 也不是 8，就是 5。M 不只改方差也改期望（soluble 路径上 0.6667→0.6817→0.6919→0.7017 单调升向确定性极限；官方 vanilla 路径上 M=5 mean 0.6701 vs M=20 mean 0.6662，无显著系统位移 —— 故只有「M 改变期望、方向朝确定性极限」这一点可移植，+0.0102 这个具体数值不可引用）。M=20 的额外成本实测为 M=5 的 1.4–2.4×（不是 4×）。
- **证据**：`run.py:57,76`；`compute_fitness_multi_pdb.py:57,58,212,217,250`；`noise.log:4-7`；实测 M20/M5 wall 比 41.5/23.7(L=116)、50.6/35.6(L=173)、65.6/43.2(L=229)、147.6/61.2(L=1107)。
- **精确编辑**：line 245 改为「**S0 gate 用官方 M=5**（`--num_seq_per_target 5`，唯一与 refs/ 可比的设定）；frozen-vs-TTT 的配对比较可用 M=20，但两臂必须严格相同 M，且 M=20 需标注为 *deviation from official protocol*，其数字**禁止**与 `refs/ProteinMPNN_zero_shot_metric.csv` 直接比较」。line 29 表头改为「M 同时改变期望值与方差」。line 246 删除「复用官方口径」这个笼统说法，替换为 B3 的显式清单。

### B3. §8 从未写 backbone_noise，而 §1 的地板是在 0.1 Å 注入噪声下测的（MISSING，3/6 命中）

- **plan 说**：§8 line 243-248 号称「frozen 与 TTT 两臂必须完全一致」，但通篇不出现 backbone noise。
- **实际**：官方推理 `--backbone_noise` default **0.00**，launcher 从不传该 flag，模型以 `augment_eps=args.backbone_noise`=0 构造 ⇒ 官方唯一随机源是解码顺序；checkpoint 里的 `noise_level=0.2` 只被打印、从不使用。而产生 §1 地板的脚本以 `augment_eps=0.1` 构造模型、并在**每个 draw** 注入 `nf=0.1*torch.randn_like(X)`；`ProteinFeatures.forward` 的噪声分支**没有 `self.training` 保护**，`model.eval()`/`no_grad` 下依然生效。所以 §1 的「MC 解码噪声地板」实为「解码顺序 + 0.1 Å backbone 抖动」的混合地板。额外陷阱：StaB-ddG 的 `ProteinMPNN` 类默认 `k_neighbors=32, augment_eps=0.1`，直接复用该 class 而不覆盖这两个参数是**静默错误**。
- **证据**：`compute_fitness_multi_pdb.py:294`（default 0.00）、`:165`、`:34-35`、`:164/173`；`baselines/protein_mpnn/protein_mpnn_utils.py:966-967`；`scripts/mpnn_noise.py:11-12,44`；`StaB-ddG/stabddg/mpnn_utils.py:592-595`。
- **精确编辑**：§8 加「结构输入」小节，写死 `backbone_noise = 0.00`，并 assert `model.features.augment_eps == 0.0`；构造参数全部钉住 `hidden_dim=128, num_encoder_layers=num_decoder_layers=3, k_neighbors=checkpoint['num_edges']=48, ca_only=False, num_letters=21`；明确禁止复用 StaB-ddG 的 `ProteinMPNN` class（默认值不同）。§1 表格加脚注说明其协议差异。

### B4. S0 的两个容差在数学上不可能通过（4/6 命中，且被 10 次实跑证实）

- **plan 说**（line 191-192）：per-assay |Δ Spearman| ≤ 0.008（1σ），且 25-assay mean 复现 +0.3970 ± 0.002。
- **实际**：`refs/` 里的参考值是**一次未设种的 M=5 单抽**（`--seed` default=0，判断写成 falsy 的 `if args.seed:` ⇒ 每次运行重抽；launcher 从不传 `--seed`），所以任何 Δ 都含参考侧自己的采样噪声。官方路径 BH3_Mcl-1 6 seeds 两次独立实测：M=5 得 0.6701±0.0205（range 0.6514–0.7071）与 0.6701±0.0187；M=20 得 0.6662±0.0093。published 0.6625 正落在该 range 内，正是单抽应有的行为。⇒ σ(Δ) = √(0.0205²+0.0093²) = 0.0225，`≤0.008` 只有 0.36σ ⇒ **完美 harness 也有约 72%（≈18/25）的 assay 随机判失败**；实跑 4 个 assay 对参考值的 Δ 为 −0.0021 / −0.0111 / +0.0399 / +0.0059，**2/4 已越界**，Z-domain_ZpA963_HL2 超出 5×。mean 侧：combined SE ≈ 0.0225/√25 = 0.0045，是所要求 ±0.002 的 2.25 倍（另一维度按 M=20 独立算得 ±0.002 仅 0.88σ、假失败率 38%）。此外**逐 assay 唯一有实测的校准点就会失败**：BH3_Mcl-1 published 0.6625 vs soluble frozen 0.6919 ⇒ Δ=+0.0294 = 自称 1σ 的 3.67 倍（该偏差同时被 checkpoint 与 backbone noise 污染，因此只能断言「校准点不可比 + gate 必然判失败」，**不能**据此断言 harness 有 bug）。
- **证据**：`compute_fitness_multi_pdb.py:26-33,293`；`run.py` 无 `--seed`（grep 无命中）；`refs/ProteinMPNN_zero_shot_metric.csv` 第 19 行 0.6625111983616473、25 行 mean 0.3969502467239151；两组 6-seed 实测；`BindingGYM_overview.md:709` 独立记录该不可复现性。
- **精确编辑**：line 191-192 整段重写为 §6 给出的 S0a/S0b 双段判据（per-assay 用「参考值落入我们自己 5-seed 的 [min,max]」或 |Δ| ≤ 2σ_combined ≈ 0.045；mean 用 ±0.009），并加一句：「`refs/` 是**单次未设种 M=5 抽样**，逐 assay 逐位复现在原理上不可能；且 `randn_1` 按 POI 缓存、被该 assay 全部 variant 共用，所以解码噪声**不随 n 衰减**，大 assay 不更安全。」

### B5. §1 噪声地板的适用范围被过度外推（多维度命中）

- **plan 说**（line 27-36）：地板「实测过的（BH3_Mcl-1，8 replicates）」，M=5 = ±0.0136，M=20 = ±0.0080，并以 0.008/√25 推出全部阈值。
- **实际**：(a) M=5 那一格被两次独立 6-seed 官方实测否证 ⇒ 应为 **0.019–0.021**（plan **低估**了 M=5 噪声）；M=20 那一格站得住（官方实测 0.0093，与 0.0080 在 n=6/8 下不可区分）。(b) 地板只在 **1/25** 个 assay 上测过（n=518，**不是 517**；L=173 中只有 **166** 个位点进 mask），却被搬给全部 25 个 assay：该 benchmark n 跨 518→92,891、L 跨 56→1107。(c) 机制上这个外推特别不安全 —— `randn_1` 按 POI 抽一次并缓存、被该 assay **所有 variant 共用**，是共模扰动，对 Spearman 的方差由「排序函数对解码顺序的敏感度」决定，**没有 1/√n 抑制**，n→∞ 也不趋零；因此不能用任何缩放律代替逐 assay 实测。(d) 顺带纠正：审计稿里「n 从 943 到 92,891」有误，真实范围是 518→92,891，`5A12_Ang2` 是 944 且不是最小；plan §4 表格里也没有出现过 943。
- **证据**：`noise.log:1`（`L=173 n_var=518`，全日志仅此一个 assay）；`compute_fitness_multi_pdb.py:216-220,225`；`assay_stats.json`（518,518,600,944,1577,1577,2080…45476,92891）；两组官方 6-seed σ；`tied_featurize` 实跑得 9/25 个 assay 有 gap（BH3_Mcl-1 166/173、BH3_Bcl-xL 180/229、KRAS_PICK3CG-RBD 915/1107、4D5_HER2 1015/1041 等）。
- **精确编辑**：line 29-33 表格改为三列（M / soluble+0.1Å 旧测 / 官方 vanilla+0.00 新测），M=5 行写 0.019–0.021，M=20 行并列 0.0080(旧)/0.0093(官方)；删掉 line 35 的 `0.008/√25` 这一步，替换为「逐 assay σ 必须各自实测（≥5 seeds），阈值由**实测 σ 向量**给出」；把「n≈517」与 §9 的规模数字对齐到 `assay_stats.json`；补一句「L=173 中实际计分位点为 166」。

### B6. 阈值把单臂 σ 当成配对差 σ，且从未规定 seed 配对

- **plan 说**（line 35-36）：0.008/√25 ≈ 0.0016 ⇒ mean 阈值取 > +0.005（「约 3σ」）。
- **实际**：判据检验的是**两臂之差**。两臂各自独立抽解码顺序 ⇒ sd(Δ̄) = √2·σ/√25 = 0.0023(M=20) / 0.0038(M=5)，3σ 应为 **+0.0068 / +0.0115**；+0.005 实际只有 2.21σ / 1.30σ，单侧假阳性率 1.4% / 9.7%，不是自称的 3σ。更划算的修法：两臂共用同一组 `randn_1`（同 `--seed`、同 POI 缓存），共模噪声在配对差里几乎完全抵消。而官方默认行为恰好是**不配对**（`--seed` default=0 + `if args.seed:` ⇒ 每次重抽），plan 通篇没有规定 seed 配对。
- **证据**：`compute_fitness_multi_pdb.py:26-29,216-220,293`；复算 √2·0.008/5 = 0.00226，0.005/0.00226 = 2.21σ。
- **精确编辑**：line 35-36 重写：(i) 判据量是**配对差 Δ̄**；(ii) 两臂必须显式传同一 `--seed` 并共用同一 `randn_1_dic`（写进 §8）；(iii) 阈值由**实测 sd(Δ̄)** 定，不得由单臂 σ 除 √25 推。开跑前必须先做 **frozen-vs-frozen 同 seed 零效应对照**，Δ 必须恒等于 0（不为 0 说明 seed 未真正配对）。

### B7. +0.005 与真实可检测效应量差一个数量级；两种主张被混在一节（2/6 命中）

- **plan 说**（line 25、36）：「其差必须超过 MC 解码噪声地板」+「> +0.005」是唯一成功判据。
- **实际**：用已发布指标矩阵原样复算 —— ProteinMPNN − MPNN-1chain 的逐 assay 配对差 mean=0.0406、sd_d=0.0526、SE=0.0105 ⇒ **MDE80 = +0.0295 @25 assay / +0.0394 @14 cluster**；更贴近 frozen-vs-TTT 的「同权重、换输入」对照（ESM2-allchain vs ESM2）mean=+0.0001、sd_d=0.0697 ⇒ MDE80 = +0.0390 / +0.0522。即同一模型的一个输入扰动，逐 assay Δ 的 sd 是 0.07 而总均值 0.0001 —— 这正是 frozen-vs-TTT 差值零分布的形状；+0.005 只有 0.36 个 SE。同时 paper 自己的 mean 不确定度是 ±0.03（见 M4），而 LOAO 实算：删掉单个 assay 就能让 25-assay mean 移动 **0.01246** = 阈值的 2.49 倍。两条限定必须写清：(a) 0.0526/0.0697 是**代理** sd_d（换模型/换输入），不是实测的 TTT Δ 异质性；(b) 这一项只对「TTT 能改善 complex binding zero-shot」这种**可推广性**主张构成方差，对「在这固定 25 个 assay 上均值更高」这种**榜单**主张不构成 —— 而 plan §1 把两者混在一节、§11 用的是前者。注：证据包自己的 full-blueprint-raw.md:442 已把同一组数字列为 blocking 风险 R4 并据此换掉 primary endpoint，当前 plan 相对该文档是统计上的**退步**。
- **证据**：复算 `_spearman_matrix.csv`（25×14），与 `full-blueprint-raw.md:442` 逐位吻合；同文件 :24；LOAO max|Δmean| = 0.01246014092747394。
- **精确编辑**：§1 拆成两条互不替代的判据（见 §6 的 A/B/C），并在 §11 明确写：按 §3.1 外推的期望增益 +0.0018 比 (C) 的 MDE 小 16 倍 ⇒ **均值层面的正、负结论都不可判定**；阈值不得低于 LOAO max shift 0.0125。

### B8. 全篇没有任何统计检验（MISSING）

- **plan 说**：只有点阈值（line 25、36、191、209）。
- **实际**：无零假设、无 α、无检验力、无 CI，也未处理 25 个 assay 的相关结构 —— 点阈值不是检验。统计单位可直接从仓库核实：`training/cache/BindingGYM_cluster.tsv` 28 行 / 15 个 cluster representative，剔除 3 个非 benchmark 流感 assay 独占的那一簇后恰为 **14 簇 / 25 assay**。
- **证据**：`full-blueprint-raw.md:553-554`（该文档自己开的处方）；`BindingGYM_cluster.tsv` 实核。
- **精确编辑**：§1 新增「统计检验」小节，开跑前锁定、不得跑完再选：primary = 逐 assay 配对 **Wilcoxon signed-rank**（α=0.05 双侧，n=24，见 M16）；secondary = **14-cluster bootstrap**（重采样整簇、20,000 次，KRAS 6 个先内部平均）的 Δ̄ 95% CI；辅助 = **win/loss/tie**（|Δρ|>0.05）与 **leave-one-assay-out** 均值敏感性。全部报 CI，而不是只报是否过线。

### B9. S1a 的判定门槛必然被噪声触发，「不跑 S1b」这条分支永远走不到

- **plan 说**（line 207-209）：3 lr × 7 step × 4 assay，「若三个 lr 的曲线都单调下降，S1b 不跑；若任一 lr 出现 > +0.008（1σ）的窗口，才展开 S1b」。
- **实际**：这是 3 lr × 6 个非零 step × 4 assay = **72 次逐 assay 比较**，每次用 1σ 单侧门槛。两臂独立抽样下 per-assay per-step 的 sd(Δ) = √2×0.008 = 0.0113(M=20) ⇒ P(噪声 Δ>0.008)=0.24 ⇒ **期望 17.3 个假窗口，P(至少一个)≈1.00**；官方 M=5 下 sd(Δ)=0.019–0.027 ⇒ 期望 24+ 个。⇒ 50 GPU-h 的 S1b 会被纯噪声触发，而「都单调下降」这条分支实际不可达（噪声本身足以打破单调）。加重项：这 4 个 assay 只有 **3 个 MMseqs2 cluster**（PSD95 两个同簇；两个 BH3 各自单簇），且两对各自共用同一突变库（1,577 / 518），所以「跨 4 assay 取均值」也不能当 4 个独立单元用。
- **证据**：plan line 207-209；`noise.log:5-6` + √2 换算；`BindingGYM_cluster.tsv`；`assay_stats.json`。
- **精确编辑**：判据改为「在同一 lr 曲线上**跨 assay 取配对均值**再判（把 72 次比较压成 lr 条数），门槛 = 3σ of **实测** sd(Δ̄) 并做 Holm 校正」，外加对独立噪声鲁棒的**形态判据**（同一 lr 在 k/k 个 assay 上 Δ>0 且 step-wise 单调上升）；正文注明 cluster 数。lr 网格与 assay 集合见 B15 与 §6。

### B10. §2.4 的机制论证对 Spearman 是空的 ⇒ 「机制普适」与 D2 的理由链断裂

- **plan 说**（line 72-78）：score(s') = log p(s'|X_AB) − log p(s_WT|X_AB)，「任何以 δ_WT 为最优解的目标都在对打分公式自己的分母做梯度上升……这不是超参问题，是目标函数的方向问题」；§2 结尾据此断言「机制论证是普适的」，§11 与 D2 据此跳过 S1b。
- **实际**：log p(s_WT|X_AB) 在固定 θ 下对该 assay 的**所有 variant 是同一个常数**，减常数保序 ⇒ Spearman(score) ≡ Spearman(log p(s'|X_AB))。「TTT 在优化分母」在数学上**完全不能解释**实测的排序退化。两点细节：论文 §3.3 确实把打分定义为 log p_mut/p_wt（所以公式不是 plan 编的），但**官方代码既不减 WT 也不做长度归一化** —— `_scores` 里的 `/ torch.sum(mask,-1)` 是被注释掉的，返回 mask 上 NLL 的**求和**，`design_score = -1*ns_mean`；sum 与 mean、以及 WT 偏移，在同一 assay 内都只差常数，同样保序。真实机制是**熵坍缩/校准破坏**：最大化 log p(WT|X) 把 p_θ 推向 δ_WT，所有非 WT 序列的 likelihood 一起塌向下界、丢失细粒度区分度。这个机制预测「损伤幅度 ∝ 模型当前携带的排序信息量」，因而**不预测 benchmark 级普适失败**。
- **证据**：`protein_mpnn_utils.py:46`；`compute_fitness_multi_pdb.py:244-263`（全文件无 WT forward）；`mpnn_noise.py:46-51`（`w` 在一个 draw 内恒定）；`mpnn_rescue.py:66`；bgym.txt:237-244。
- **精确编辑**：重写 §2.4 —— 明写「WT 项是 per-assay 常数 ⇒ 对 Spearman 无影响 ⇒ 『在优化分母』不能解释实测退化」，机制改述为熵坍缩/相对校准破坏，并注明官方打分器用的是 mask 上 NLL 求和（未长度归一化）。**删除**「机制论证是普适的」。§11 的「机制已经清楚」与 **D2（跳过 S1b）的理由链必须撤销**，改为「机制只预测损伤随 baseline 强度递增，不预测 benchmark 级普适失败 ⇒ 需分层实测」。同时 §4 后果 1 的「log-odds ratio 的位点独立假设」不适用于这个打分器（单次 teacher-forced 自回归 pass），须改写。

### B11. 「单调」从未被测量 —— 证据包里没有任何 step-wise 曲线

- **plan 说**（line 45、55）：「单调下降，没有操作窗口」；§11「实测在 1 个 assay 上单调崩坏」；D2 据此跳过 S1b。
- **实际**：noise.log 全 13 行，lr 扫描每行**只有 step0 与 step30 两个端点**（`r0` 与 `r30`，30 步循环内不打分）；rescue.log 同样只有端点。唯一会打中间步（1,2,3,5,10,20,30）的 `scripts/mpnn_ttt.py:87-89` 其日志**不在 logs/ 里**（`grep -l gradnorm logs/*` 无命中）。两点连线无法区分「单调下降」与「先涨后跌」。而且 lr 方向上本身就**不单调**：lr=0.3 的 Spearman +0.4783 高于 lr=0.1 的 +0.3884（NLL 亦然，0.8323 vs 0.3867）。另：mpnn_ttt.py 的 step-wise 打分每步只有 1 个 M=5 draw（σ≈0.019），分辨不出 +0.008 的窗口。
- **证据**：`noise.log`（13 行）；`mpnn_noise.py:63,82`；`mpnn_rescue.py:76,87,92,103`；`mpnn_ttt.py:51,87-89`；证据包 `README.md:9,35`（同样的错误表述，且把 lr 截断在 1e-1）。
- **精确编辑**：line 55 改为「只测了 step0 与 step30 两个端点，**中间轨迹未测**；lr 方向上也不单调（lr=0.3 的 +0.4783 高于 lr=0.1 的 +0.3884）」；证据包 README:9/35 同改。撤销 D2 的这条理由：单调性本身未测，S1a 的 step-wise 曲线是 plan 里**唯一能测它**的实验，不得在结果出来前预判。把 mpnn_ttt.py 的实际输出补进 logs/，或注明从未成功跑完。

### B12. §9 的「partner-blind 恒等于 0.000」前提为假（4/6 命中）

- **plan 说**（line 284）：「partner-blind 打分器在这个指标上恒等于 0.000（同一条链、同一结构 ⇒ 同一分数）」；line 203、280「同一 PDB 1BE9」。
- **实际**：三对**全部使用不同的结构文件** —— PSD95 `1BE9_hm.pdb` vs `1BE9Tm2F_hm.pdb`（md5 不同，145,809 vs 146,295 B）；BH3 `3KZ0_hm.pdb` vs `1PQ1_hm.pdb`；5A12 `4ZFF_CHL.pdb` vs `4ZFG.pdb`。逐原子比对（两个维度独立复算，一致）：PSD95 chain A 的 **N/CA/C 逐位完全相同**，只有 O（RMSD 0.032 Å，max 0.187 Å）与侧链不同 ⇒ 因为 ProteinMPNN 只看 N/CA/C/O 且训练含 0.2 Å 噪声，这一对的 partner-blind specificity 实际上 **≈0.000（远低于噪声地板），但不是恒等式**；BH3 与 5A12 则**不为 0**（BH3 被打分的链本身就不同：23 aa chain C vs 33 aa chain B，位点偏移 +2；5A12 连链集合都不同，CHL vs AHL，H 链有 4 个 token 差异）。**更根本的一点**：本 plan 要打的 baseline 是**全复合物 ProteinMPNN，它本来就不是 partner-blind**（paper §4.1「we input full protein complex structures」；harness 把 `chain_id` 里每条链都标为 designed 并纳入计分 mask），其自身 specificity 已经非零 —— published per-assay Spearman 已经不相等（PSD95 0.3863 vs 0.2073；5A12 0.4775 vs 0.1171）。所以这个指标**不能把 complexTTT 与它自己的 baseline 分开**。「≡0」只对 `--focus 1` 的纯序列 baseline 成立。
- **证据**：`input/BindingGYM.csv` 的 pdb_file/chain_id 列；两次独立逐原子比对；bgym.txt:289；`compute_fitness_multi_pdb.py:205-208,245`；`results/ProteinMPNN_zero_shot_metric.csv`；`baselines/esm/compute_fitness_multi_pdb.py:215-219`（`--focus` default 1）。
- **精确编辑**：删除 line 284 的「恒等于 0.000」与括注「同一结构」，替换为逐对表：PSD95 → **≈0.000，远低于噪声地板**，且仅对 focus-only 序列模型成立；BH3 → 任何模型都不为 0（被打分链 23 vs 33 aa）；5A12 → 任何模型都不为 0（4 个 H 链 token 差异，且它们是**未解析残基占位 X**，不是生物学差异）。line 272 与 286 的「任何 partner-blind 方法结构上做不到」「baseline 结构上做不到」必须删除，改为「对比对象是 `--focus 1` 的**序列** baseline，不是 ProteinMPNN」，并把**frozen 全复合物 ProteinMPNN 自己实测的 specificity** 列为强制 comparator（S0 的交付物之一）。line 203/280 的「同一 PDB 1BE9」改为「两个不同的同源模型文件（N/CA/C 逐位相同、O RMSD 0.032 Å）」。

### B13. §9 的指标 `1 − ρ` 方向错误，会被退化的模型最大化

- **plan 说**（line 274，D3 提为并列 headline，越大越好）。
- **实际**：无信息的打分器给 ρ≈0 ⇒ specificity≈1.0，**超过** label 隐含的 oracle 目标（PSD95 1−0.4348=0.565；BH3 1−0.5919=0.408）；且无界（5A12 的 oracle 是 1−(−0.1452)=1.145>1）。plan 自己 §2.1 的 lr=0.1 崩塌（0.6921→0.3884）正是会把这个指标**推高**的情形。
- **证据**：plan line 274、280-282、296；`crosspartner.json` 的三个 label ρ 逐位复现；plan line 51。
- **精确编辑**：把 `1 − ρ` 换成 **oracle 锚定的有号量** |ρ_model_cross − ρ_label_cross|，或换成 **matched/mismatched-partner 配对检验**（把 assay-1 的 variant 集分别在 partner-1 与 partner-2 复合物上打分，要求 matched 那一侧对 assay-1 自己的 label 给出更高 Spearman）；写明取值范围与零分布。按现定义它必须从 §9、D3、§11.3 移除。

### B14. §9 完全没有跨 assay 的 variant 对齐程序（MISSING）

- **plan 说**：line 276 表头「同一突变库」，line 203/281「同一 1,576-variant 库 / 同一 518-variant 库」，但全篇无对齐规程。
- **实际**：实测原始 `mutant` 字符串重叠 —— PSD95 1577/1577（字符串与行序都相同，可直接 join）；**BH3 = 0**；**5A12 = 0**；KRAS-RAF1 对 12,086。BH3 需要 chain 重映射（C→B）**加** +2 的位点偏移（`I6A:L10A:R11D:I13A` ≡ `I8A:L12A:R13D:I15A`），裸 key join 只得 1 个共享 key 且那是 WT 空 key ⇒ **真实重叠 0**，任何按 mutation key 配对这两个 assay 的代码会静默产生空 join。5A12 **不需要偏移搜索**，它只是因为未突变的抗原链在 mutant dict 里换了字母（C vs A）；只按被突变链（H,L）做 key 即得 534。
- **证据**：四个 per-assay CSV 的实算；offset sweep（0/1/3/10 → 1；+2 → 518）；`crosspartner.json`；参考实现 `Sources/complex-ttt-evidence/scripts/crosspartner.py`（`keyset()` / `align_offset()` / `canon()`）。
- **精确编辑**：§9 新增「Variant Alignment」小节，五步写死：(1) key 只取**被突变链**（这一步单独就修好 5A12）；(2) 逐对的 chain 对应表；(3) 由被突变链的序列比对求位点偏移并设 identity 闸门（BH3 需 C→B、+2）；(4) 只在对齐后的交集上计算；(5) **报交集大小及其占每个 assay 的比例**。引 `crosspartner.py` 为参考实现。并在 S1a harness 里加断言：BH3 对对齐后 join 必须返回 518 行。§5.1 同时补一句同类坑的说明。

### B15. S1a 的 4 个 assay 既不能推广，也系统性避开了 §4 自己认定的最大技术风险（2/6 命中）

- **plan 说**（line 202-206、309-310）：4 个 assay + 两对 partner-swap「把否证做实」，成本仅 ~2.5 h。
- **实际**：这 4 个 assay 只有 **3 个 cluster、2 个蛋白家族**，两对内部各自共用同一突变库（1,577 / 518，同一 PDB），有效独立单元 ≈2（BH3 对 label ρ=0.5919、PSD95 对 0.4348）；其中 BH3 那一簇正是 prior 已测的簇 ⇒ 实际只新增 1 簇。而且它们全部落在 benchmark 的**极端一侧**：多链同改数 **0/0/0/0**（全 benchmark 21.32%、7/25 个 assay 有、6/25 过半），depth_max 只有 1（PSD95，纯单点）与 5（BH3）（全 benchmark depth_max=21、35.07% 的 variant 深度 ≥3），n 是 25 个里最小的 4 个（518/518/1577/1577，中位 5,585），共 4,190 = 全 benchmark variant 的 **1.11%**，4/4 都是 `_hm` 同源模型；published MPNN mean ρ = **0.4779** vs benchmark 0.39695，且 BH3 两个落在 ρ≥0.62 的最高分层（该层仅 4/25 assay、7.12% variant）。⇒ §4 自称「整个提案最大的技术风险」（多链同改、深突变）在 S1a 里**覆盖率为零**。
- **证据**：`BindingGYM_cluster.tsv`；`assay_stats.json`；`refs/ProteinMPNN_zero_shot_metric.csv` 排序；Σn×L 实算（S1a = 586,716）。
- **精确编辑**：把 S1a 换成 §6 的 **S1a′（7 assay）**：保留 BH3×2 + PSD95×2，新增 `Z-domain_ZpA963_HL2_fitness_2M5A`（n×L=69,600，全 benchmark 最便宜；55.2% 多链；非 `_hm` 实验坐标；兼作 specificity 零对照）、`Z-domain_ZpA963_HL1_fitness_2M5A`（92.9% 多链、最低 baseline 层、paper §4.4 点名家族）、`hYAP65_peptide_FunctioncalScore_1JMQ` 随机子采样 2,000（depth_max=21）；预算够再加 `KRAS_RAF1_norfitness_6VJJ` 子采样 2,000。分层报告：低/中/高 baseline 三层、多链 vs 单链、depth<3 vs ≥3。§11 line 309 的「4 个 assay 把它做实」改为「3 个 cluster / 2 个家族、全部为纯单链浅突变小 assay」，结论范围写死为「partner-swap 小 assay 上的负对照」，**不得**表述为「把否证做实」。

### B16. S0 预算低估 2.1–3.0×（4/6 命中，且经直接计时）

- **plan 说**（line 252-256、263）：46.5 s ⇒ 223 sequence-scoring/s ⇒ S0 全量 376,446×20 = 7.53 M ⇒ **~9.4 h**。
- **实际**：算术自洽，但成本模型把打分当成与长度无关 —— ProteinMPNN 每个 variant 对**整个复合物**做一次 forward，成本随 n×L_total 增长且比线性更差（`_get_rbf` 建 [B,L,L] 张量）。A4500 实测（官方脚本、v_48_020、M=20，完整 assay）：Z-ZpA963_HL2 (n=600,L=116) 41.5 s；BH3_Mcl-1 (518,173) **50.6 s**；BH3_Bcl-xL (518,229) 65.6 s —— 同 n、1.30× 墙钟，正好是 229/173。100-variant 子集：L=652 → 0.97 s/variant，L=1041 → 1.12，L=1107 → 1.21（对比 L=173 的 0.089）。按 t_var = 4.074e-4·L + 6.194e-7·L² 对 Σ n·L_total = **135,744,347** 求和 ⇒ 全量 S0 **≈28.3 A4500-h @M=20**（纯线性下界 19.5 h），官方 M=5 下 **≈14–16 h**。⇒ plan 低估 2.1–3.0×。注意 223/s 这个锚点本身是**对的**（官方实测 205/s，差 9%），只是**只在 L≈173 成立**：同一脚本在 L=1107 只有 16.5/s，跨 benchmark 12× 差异。
- **证据**：上列全部计时；`protein_mpnn_utils.py:525-529`；Σn 与 Σn·L 由 `input/BindingGYM.csv` + 25 个 DMS CSV 复算。
- **精确编辑**：§8 表格单位从「×M=20 scorings」改为「n × L_total residue-variants（× M）」，新增 per-assay n·L_total 与实测小时列；S0 行改为「≈28 A4500-h @M=20 / ≈14–16 h @官方 M=5（实测外推；19.5 h 为纯线性下界）」；223/s 改写为「= 205/s（官方脚本实测）**@L=173**；同脚本在 L=1107 仅 16.5/s」，删除任何「该速率是 benchmark 常量」的暗示。line 261 的「S0 + S1a ≈ 12 GPU-h」改为按所选 S0 子集的正确总数（缩减 gate ≈2.8 GPU-h；全量 ≈30 GPU-h）。

### B17. TTT 侧的 run-to-run 方差从未测量，而阈值假装它是 0（MISSING）

- **plan 说**：§1/§7/§8 的地板只考虑打分侧；line 212 写「25 assay × 5 seeds」但未定义 seed 指什么。
- **实际**：TTT 自身的随机性（masking、每步重抽的解码顺序与 backbone 噪声、优化器路径）很可能是配对 Δ 里最大的一项，却完全未测。证据包里 §2.1 的「±0.0147 / ±0.0106 / ±0.0031 / ±0.0152」**不是** TTT-seed 方差：那是对**同一个已训练模型 m2** 做 3 次打分复算；§2.2 是 2 次打分取均值；§2.3 是**单次打分、无误差棒**。⇒ 目前没有任何数字能界定 TTT 侧噪声。
- **证据**：`mpnn_noise.py:63,78,82,84`、:66-72（训练侧随机源）；`mpnn_rescue.py:76,87,92,103`；plan line 212。
- **精确编辑**：在 S1a 之前加廉价前置测量：固定 1–2 个 assay、固定 (lr, steps)，跑 ≥5 个 **TTT seed** × 同一打分 seed，报 sd(Δ) 的 TTT 分量；阈值改为 3 × √(sd²_TTT + sd²_score,paired)/√n。line 212 的「5 seeds」明确写成「**TTT seed**；打分 seed 与 frozen 臂共用同一 `randn_1`」。

---

## 2. MAJOR corrections

### M1. §2 对**自己的**证据的口径写错了，「−4σ」与「单调」的显著性判断随之失效（3/6 命中）
- plan line 42-43 写「M=20，8 replicates」并把它用作 §2.1–2.3 的统一出处。实际：**§2.1 = M=5、3 次打分复算**；**§2.2 = M=20、2 次**；**§2.3 = M=20、单次、无误差棒**；只有 §1 的地板表用了 8 replicates（且只有 NLL 那一行是 8 draws）。后果：(a) n=3 的 std 自身相对不确定度 ~50%，不能用来算「σ 倍数」；(b) lr=0.01 那行的「← −4σ」实为 ~3.0σ（对 M=5 的 0.0136）或 ~2.2σ（对官方实测 0.019）；(c) **lr=0.001 那行的 Δ = 0.6935−0.7054 = −0.0119 在 M=5 的 1σ 之内，根本不是下降** —— 更强的读法：4 行 lr 扫描的 step0 都是同一 frozen ckpt 的独立估计，pooled mean 0.6931、std 0.0079（正好 = 0.0136/√3，说明该列离散度全是噪声），以 pooled frozen 为基准 lr=1e-3 的 Δ = **+0.0004 ≈ 0**。
- 证据：`mpnn_noise.py:55,63,78,82,84`；`mpnn_rescue.py:76,87,92,103`；`noise.log:5,10-13`。
- 编辑：line 43 改为逐小节标注真实口径，并全部加注「soluble 权重 + augment_eps=0.1」。§2.1 表格加一列「Δ / σ_paired」，lr=1e-3 标 **n.s.**，删除或改写「← −4σ」；加一行说明 4 个 step0 的 std = 0.0136/√3。「单调」按 B11 处理。§11「把握是高的」下调为「1 个 assay、薄 replicate 的机制性证据」。

### M2. §2.3「放大损伤 1.6×」过度精确且无误差棒
- 1.6× 只在 lr=0.1 成立（0.3235 vs 0.5058 = **1.56×**）；lr=0.01 是 **1.82×**（0.0484 vs 0.0879）。每个端点都只是**单个 M=20 draw**（无 replicate），按 σ≈0.008 估配对 Δ 的 SE≈0.011 ⇒ lr=0.01 那行 complex Δ 仅 ~4σ；而 **StaB 打分器自身的 σ 从未测过**（它多算 2 条单链，噪声应更大）。
- 编辑：标题改为「StaB 参数化放大损伤 ~1.6–1.8×（lr=0.1 为 1.56×、lr=0.01 为 1.82×；每个端点均为单个 M=20 draw，误差棒未测，StaB 打分器噪声地板未测）」，或在 S1a′ 补 ≥3 replicate 并先测 StaB 打分器的 σ。

### M3. §2 的「证据强度边界」写窄了 —— 真正的问题不是「只有 1 个 assay」，而是「这个 assay 在极端一侧」
- BH3_Mcl-1 的 published ProteinMPNN ρ=0.66251 在 25 个里**排第 2**；n=518 **并列最小**（0.14% 的 variant）；其所在 ρ≥0.62 分层只占 4/25 assay、7.12% variant；且它自己天花板极低（M=inf proxy 0.7017 只比 M=20 高 0.0098，有界指标在接近上界处任何扰动都更可能向下）。⇒ 这轮实测给出的是**损伤上界**，对 benchmark mean 的方向性没有约束。
- 需要**降一档**的一句：「高 baseline 更容易被 TTT 打坏」目前只有两个极值点（ProteinGym 复现里最差 −0.1926 @baseline 0.675、最好 +0.5137 @0.140），逐 assay 的 baseline–Δ 相关性**没有实测**（`workstation-records/proteinTTT-repro/results/` 是空目录），且记录里对最差三个的归因是「37–44 残基的短 stability assay」而不是「高 baseline」。
- 编辑：按上述改写 §2 结尾的边界声明，并加一句「若要用它支撑外推，需先从 ProteinGym 复现导出 per-assay Δ 并实测该相关性」。§11「我给的把握是高的」改为「对高 baseline assay 把握高，对 benchmark mean 未定」。

### M4. §1 完全没提 benchmark 自己的误差棒（3/6 命中）
- 论文 Table 6 / Table 11 报 ProteinMPNN **0.40 ± 0.03**（1000 次 bootstrap）。复算已发布逐 assay Spearman：mean 0.39695、sd 0.17064、SEM 0.03413（有放回略小 ≈0.0334 ⇒ 四舍五入 0.03）；SI notebook 的 `tmp = df.sample(frac=1,replace=True)` 确认重采样对象是 **per-assay 指标表**（即「换一批 assay 会怎样」）。⇒ 它**不是**配对 frozen-vs-TTT 的尺子（配对比较时这一项抵消），但它决定了「超过 +0.3970 上榜」这个**绝对主张**的不确定度 —— ±0.03 是 +0.005 的 6–7 倍，在 25 个 assay 上不可能被 TTT 量级的效应撼动。附带记录论文自身口径矛盾：Table 6 caption 说「from the set of assays」，A.7 正文说「from each assay」，**代码站 caption 一边**。
- 编辑：§1 加一段区分两种主张与两把尺子（见 §6 的 A/B/C），并记下该 caption/A.7 矛盾。

### M5. 「MPNN-1chain +0.3564」这一行不可复现，且不在论文里
- 论文 Table 2 的 13 个模型里没有这一行。0.3564 是 `results/ProteinMPNN_single_zero_shot_metric.csv` 的均值（复算 0.356353），但仓库里**没有任何脚本生成该文件**，也不可能是「另一个 score 列」——我在 25 个 assay 上验过 `mask*chain_M*chain_M_pos == mask`（`chain_id` 覆盖全部链、`fixed_chain_list` 空），所以 `design_score ≡ global_score`，而两个 CSV 在 25/25 个 assay 上都不同（逐 assay 差 mean +0.0406、max +0.1474、min −0.0642）。它其实是作者自己的 `target_seq` ablation（见 `BindingGYM_SI.ipynb`），只是从未进论文、输入也未发布；「MPNN-1chain」是 plan 自造的名字。
- 编辑：line 16 删除该行，或注明「repo CSV only（`ProteinMPNN_single_zero_shot_metric.csv`）；作者未发表的 target_seq ablation；生成脚本与输入均不在仓库 —— **不可复现**」。并审查 §4 后果 3 与 §9 里任何依赖「+0.041 partner benefit」的论证，改锚到 S0 能实测的量上。

### M6. §4 后果 3 的「6/25 个 assay 里 partner 链也在变」是错的（2/6 命中）
- 真正**同时突变界面两侧**的只有 **4/25**（四个 Z-domain assay：1LP1 chain A 55 aa ZSPA-1 / B 54 aa Z-domain；2M5A A/B 各 58 aa）。另外两个「多链同改过半」的 assay 改的是**同一侧抗体的两条链**：4D5_HER2（chain_id ABC，突变 A=214 aa VL 与 B=220 aa VH，抗原 chain C 607 aa **从不突变**）、5A12_VEGF / 5A12_Ang2（突变 H/L，抗原 C 96 aa / A 220 aa 从不突变）。含**任何**多链 variant 的是 **7/25**，多链过半的是 6/25。第二半也错：正是在这些 assay 上，focus-only 序列 baseline **并非 partner-blind** —— `DMS_file_for_LLM` 把每条被突变过的链都放进 `focus_chains` 并拼进被打分序列。
- 编辑：line 146-147 按上述重写；**删除**「任何 partner-blind 方法结构上做不到的部分」，并注明 `utils/data_utils.py:80-86,109-116` 的拼接行为。§4 表头把「同时改多条链」拆成「multi-chain」与「both-sides-of-interface」两列。

### M7. n_eff = 2.02 被误引（2/6 命中）
- 2.02 是**6 个 KRAS assay** 的 label 有效自由度（15 对配对 ρ = 0.677–1.000，PC1 解释 65.6%），不是 25-assay benchmark 也不是 14 簇的 n_eff。把它与「≈14 簇」并置，等于断言整个 benchmark 的 n_eff≈2 —— 那会让 plan 自己推荐的 14-cluster mean 变成无用而不是充分。
- 编辑：line 158 改为「KRAS 家族 6 个 assay 之间 label 配对 ρ = 0.677–1.000（15 对），PC1 解释 65.6%，n_eff ≈ 2.02/6 ⇒ 这 6 个必须先内部平均为 1 个统计单元；官方 MMseqs2 聚类给出 14 簇，见 `training/cache/BindingGYM_cluster.tsv`」。

### M8. §5.3 的 interface 标签定义错误，且分层在 9–10/25 个 assay 上不可执行
- 产生 0.4295 的标签是「任何被突变残基与**结构中任意其它链**的最小重原子距离 < 5 Å」，**不区分 binding interface 与其它链间接触**。对抗体与 Z-domain assay 这使「interface-touching」变成同义反复：4D5_HER2 的 `frac_variants_no_iface_mut` = 0.000、5A12_VEGF = 0.001、四个 Z-domain 全 = 0.000（VH–VL packing 也被算作界面）。可执行性：**7/25** 个 assay 的 non-interface 子集为 0（Z-domain×4、GB1_1FCC_2016、BH3×2），4D5_HER2=1、5A12_VEGF=24、5A12_Ang2=120 ⇒ 以 200 variant 为下限，**10/25 不满足、15/25 满足**。直接打到 S1a：**两个 BH3 完全无法分层**，两个 PSD95 的 interface 子集只有 285/1,576 与 304/1,576（≈18%）。
- 编辑：把标签重定义为「与该 assay 中**从不被突变的 partner 链**的重原子距离 < 5 Å」，重算 `frac_variants_no_iface_mut` 后再引用任何数字；对 4D5 与 5A12 明确把 H–L 接触**排除**在界面集之外。§5.3 加执行条件：「分层只在 non-interface 子集 ≥200 variants 的 assay 上报（15/25 满足）」，并报每个子集的 n。

### M9. §6 的表头一行用的是**子集受限**相关，与所有已发表数字不可比
- `rho_burial_on_iface` = 0.2816 是 `sp(b_nbc, iface_mask)`，只在 interface-touching variant 上算 Spearman；而 ESM2 (+0.2851)、ProteinMPNN (+0.3970) 是**全 variant 集**。同类可比的零参数量是 `rho_burial_complex` = **+0.2444**（低于 ESM2），全集最强的是 `rho_dmin` = **+0.2598**（仍低于 ESM2 约 0.025、低于 MPNN 约 0.14）。`rho_burial_on_noniface`（0.1927）只在 17/25 个 assay 上有定义。
- 编辑：把 `rho_burial_on_iface` 从对比表移出（或移入清楚标注「interface-subset only, not comparable」的分块），结论改为「全变体集上最强的零参数特征是 `rho_dmin` +0.2598 / `rho_burial_complex` +0.2444，**都低于** ESM2 的 +0.2851」。§6 的定性结论（MPNN 有真实信号、高出约 0.14）仍成立。

### M10. §8「降本备选」把成本大头认错了
- 按实测成本模型：`KRAS_PICK3CG-RBD_1HE8` **6.45 h（22.8%，n=19,203，L=1107）**、`SARS2-RBD_ACE2_6M0J` 4.31 h（15.2%，L=791）、`GB1_IgG-Fc_1FCC` 3.85 h（13.6%）、`5A12_VEGF_4ZFF` 3.23 h（11.4%）、`KRAS_SOS1_8BE4` 2.80 h（9.9%）—— 五个占 72.9%。GB1 是第 3 不是第 1；**Z-domain_ZSPA-1_LL1 只占 0.65 h = 2.3%（排第 10）**，因为 L=109。删掉 plan 点名的那两个只省 15.9%（剩 23.8 h）。
- 编辑：line 263 换成上面的实测小时排序，并注明三个长复合物（1HE8 L=1107、1N8Z L=1041、6M17 L=931）是成本与**显存**的双重驱动。选子集时必须按 **n·L_total** 排序，不能按 n（否则漏掉 L=1107 那个）。

### M11. S1b 的 ~50 h 低估约 3×
- 一趟 25-assay M=20 实测 ≈28.3 h ⇒ 5 seeds ≈ **142 A4500-h**（官方 M=5 下 ≈75 h）；若 frozen 臂也按 seed 重打分做配对，则 ×2 ≈ 283 h。TTT 训练本身可忽略（1.66 M params，<1 s/assay）。
- 编辑：line 259 改为「~142 A4500-h @M=20（~75 h @官方 M=5），若 frozen 臂逐 seed 重打分则 ×2」，并写明哪一臂重打分。

### M12. M=20 在最长的复合物上没有显存余量（MISSING，已实测复现 OOM）
- batch 维就是 M 份结构副本 ⇒ 显存 ∝ M×L。在 A4500（19.57 GiB）上、另有 ~8.85 GiB 被占用时，官方脚本在 `KRAS_PICK3CG-RBD`（L=1107）M=20 下 **OOM**（`protein_mpnn_utils.py:1097`，「Tried to allocate 1.52 GiB … 83.69 MiB is free … this process has 10.83 GiB in use」）；同输入 M=5（61.2 s）与 M=10（76.5 s）成功，M=20 在同租户退出后成功（147.6 s）。⇒ M=20 × L≈1041–1107 需约 11–15 GiB，只能在**空闲** A4500 上跑。涉及 1HE8(1107)、1N8Z(1041)、6M17(931)，6M0J(791) 接近。另：`run.py` 按 assay round-robin 分片（`i = idx % gpu_count`），但**不要**给 run.sh 传 `0,1` —— cuda:1 是 Pascal TITAN X（~3–4× 慢、12 GB，机器级规范禁止使用）；分片粒度是「一个 assay 一个进程」，所以全量 M=20 的 Amdahl 下界是**最大单个 assay = 6.45 h（1HE8）**；有 assay 级 resume（`if os.path.exists(f'./output/{DMS_id}.csv')`，仅当在 modelzoo/proteinmpnn 下调用时有效）但**没有 assay 内 checkpoint**。
- 编辑：§8 加「显存与设备」小节写入上述全部内容（含 pre-flight 的空闲显存断言、L≥931 时降到 M=5–10 并逐 assay 记录 M、所有小时数均为**独占** A4500 cuda:0）。

### M13. 数据根目录与运行环境全篇未指定（MISSING）
- `/home/guoj0f/repos/BindingGYM/input` **不存在**；`modelzoo/config.sh` 指向相对 `../../input/...` 加三个他人硬编码路径（`/home/zhangjx/anaconda3/...`、`/home/zhangjx/.cache/torch/hub/checkpoints`、`/home/zhangjx/af2_database/.../uniref100.fasta`）⇒ run.sh 按现状不可执行。数据确实在本机，但只存在于**另一个项目的 git worktree** 里：`/home/guoj0f/repos/H3-DDG/.claude/worktrees/reproduce/data/input`（25/25 assay CSV、22/22 structures、22 个 .a2m 共 633 MB、input.zip 166,467,341 B）—— 一个可被删除的位置，而 plan 默默依赖它。
- 编辑：§8 加「inputs」块写明绝对数据根，并把 input/ 拷贝或 symlink 进 `/home/guoj0f/repos/BindingGYM/input`；在 plan 的 sh/ 下放一份本地化 config.sh（python = `/home/guoj0f/anaconda3/envs/stabddg/bin/python`，已验证可跑；checkpoint_folder = `/home/guoj0f/repos/BindingGYM/training/cache`）；记录 zenodo record 12514160 为重下来源。

### M14. S2 的 MSA 成本判断错误，D4 的问题问错了层级
- BindingGYM 的 input.zip **自带 MSA** 且已在本机：22 个 .a2m、633 MB、**22/22 个 POI 全覆盖** ⇒ 「同源检索是这一段的主要开销」不成立，D4 的 POI 级覆盖率问题答案是 100%。真正的缺口在**逐链**：每个 alignment 的 query 只覆盖被突变链。实测 query 长度 vs L_total —— 3KZ0_hm 23/173 = 13%（只有 BH3 肽）、1HE8_hm 166/1107 = 15%、6M0J 194/791 = 25%、1N8Z_hm 434/1041 = 42%（重+轻链，无 HER2）、4ZFF_CHL 432/528、4ZFG 432/652、1BE9_hm 115/120；**全复合物覆盖只有 2 个 POI / 4 个 assay**（2M5A 116=116、1LP1 109=109）。
- 编辑：改写 line 232-233 与 D4：MSA 已就地、22/22 POI 覆盖；把开放问题重述为「S2 的 q 是否能只定义在被突变链上（全复合物 q 仅 2M5A 与 1LP1 可用）」，再决定是否需要新跑 MMseqs2（新检索需 uniref100，~100 GB+）。

### M15. S0 的判据与 D1 的建议互斥
- 在 10-assay 子集上「25-assay mean 复现 +0.3970」不可计算，而子集自己的均值没有参考值 ⇒ 选 D1 会静默作废一半 gate。子集 gate 是可构造的，参考均值已算好：**10 smallest by n = 0.348913**；**10 smallest by Σn·L = 0.356467**；S1a 那 4 个 = 0.477881；全 25 = 0.396950。
- 编辑：S0 判据拆成 S0a（子集逐 assay + 子集自己的参考均值，把数值写进 plan 免得日后重推不一致）与 S0b（25-assay mean，明确延后到全量）。

### M16. §5.1 要求「处理重复 assay」与 S0 要求「复现 25-assay +0.3970」冲突，plan 从未裁决
- 复现 +0.3970 必须**保留全部 25 个 assay**（`results/ProteinMPNN_zero_shot_metric.csv` 25 行 mean=0.39695）；而「处理重复」意味着剔除一个并报 n=24。plan 既未说哪一臂用 25 哪一臂用 24，也未说剔除哪一个。证据包已裁决过：剔除 `KRAS_SOS1_norfitness_8BE4`（其 99.0% 的行与 5O2S 逐值重复）并报 n=24。
- 编辑：§7 加一句「S0 gate 在**全部 25 个 assay** 上比对；S1b 及所有 Δ 报告改为 **n=24**（剔除 `KRAS_SOS1_norfitness_8BE4`），并同时报 14-cluster 版本。两个 n 在文中不得混用」。这条单独作为新决策 **D6**，不要塞在 D5 里。

### M17–M21. §9 表格与 S1a 的对本身各有硬缺陷
- **M17（BH3 对）**：partner 身份与**被打分链自身的构造**混淆 —— 被突变链在 Mcl-1 assay 是 23 aa chain C，在 Bcl-xL assay 是 33 aa chain B（N 端 +2、C 端 +8），partner chain A 是 150 aa vs 196 aa 的两个不同同源模型。任何非零 specificity 都混了链长、链重标与结构来源。编辑：要么把比较限制在对齐后的 23 残基核心并把额外残基列为未控协变量，要么改用单一构造上的 mismatched-partner 对照；不得写「同一库」而不加对齐 caveat。
- **M18（PSD95 对）**：**不是 partner swap** —— partner chain B 是同一 5 残基 CRIPT 肽的单点变体（KQTSV vs KQFSV，T3F），chain A 序列完全相同、chain B 从不被突变（0/1577）。整对的真实 partner 信号就是**1 个残基**，之上还叠了一次独立松弛（O RMSD 0.032 Å）与不同的解码顺序 seed（POI 不同）。编辑：line 203/280 改述；删除对这一对的「决定性」；加 **mismatched-partner 对照**（把 CRIPT 的 variant 集在 Tm2F 复合物上打分，反之亦然）以把 partner 分量与坐标/MC 噪声分开。
- **M19（5A12 行）**：既非同库也非同 PDB —— n=29,981（4ZFF_CHL.pdb, CHL）vs n=944（4ZFG.pdb, AHL），原始 key 重叠 0，对齐后 **533–534**，占 Ang2 的 56.5%、占 VEGF 的 **1.8%**；抗原是不同蛋白（96 aa vs 220 aa）。ρ=−0.145（复算 −0.142）是在这个 533-variant 交集上算的，无 n 无 CI。编辑：表头改为「同一被突变蛋白、不同 partner；三对的 PDB 文件各不相同」，加 `n_shared` 列（PSD95 1,576 / BH3 517 需 offset +2 / 5A12 **533，Ang2 的 56.5%、VEGF 的 1.8%**）与 bootstrap CI。
- **M20（指标上界 / 负对照 / 排除规则）**：有 3 对 assay 的 (POI, chain_id, wildtype_sequence) **逐位相同** —— `KRAS_RAF1_6VJJ`/`KRAS_RAF1-RBD_6VJJ`（245 aa，12,086 个共享 key 上 `mutated_sequence` 亦逐位相同）、`Z-domain_ZpA963_HL1`/`HL2`（116 aa）、`Z-domain_ZSPA-1_LL1`/`LL2`（109 aa）。对这些对，**任何**只看输入的模型（含 complexTTT）必然给同一分数 ⇒ specificity **结构性恒为 0**，与 partner-blind 不可区分，而 label 却分歧（Spearman 0.876 / 0.835 / 0.306）。**注意**：KRAS_RAF1 这一对**不是**第二个重复数据缺陷 —— label 确实是不同测量（max|ΔDMS_score| = 1.1216、ρ=0.876、n=12,677 vs 23,162），与 DARPinK27/SOS1（max|Δ|=0.0、ρ=1.0）性质不同。编辑：§9 加「排除规则 + 负对照」：输入指纹相同的对必须排除在 specificity 指标外（点名 KRAS_RAF1 vs RAF1-RBD）；把 `Z-domain_ZpA963_HL1/HL2` 同时用作**零对照**。引用 label 一致性时用 **Spearman 0.876 / 0.835 / 0.306**，不要用 overview §7.1.4 表里的 Pearson 0.927 / 0.872 / −0.076（并注明该表列名有误）。
- **M21（statistic 的自由度与误差棒）**：按现在的写法这个 co-headline 只有 **n=2 对**（其中一对只有 1 个残基的 partner 信号、另一对有长度混淆），既无 cluster-level 聚合也无 CI —— 而 §5.2 对 mean-Spearman headline 要求这两样。可用的最大合法 partner-swap 测试床被完全忽略：KRAS 簇有 6 assay / 15 对、对齐重叠 9,957–19,520，剔除 DARPinK27|SOS1（label ρ=1.0）与 RAF1|RAF1-RBD（输入相同）后**13 对可用**（6 个 assay 覆盖 5 个不同 partner；被突变 KRAS 链 165–168 aa，分布在 A/B/R 三个链字母、5 个 PDB 文件，需按 M14 的对齐程序处理）。此外 specificity 的**噪声地板从未测量**：ρ(score1,score2) 继承 MC 解码方差，而 `randn_1` 按 POI 缓存、每对的两个 POI 都不同 ⇒ 两个被相关的分数向量是在**不同解码顺序**下产生的，即使模型完全相同也会把 ρ 压低。编辑：§9 扩到 13 对 KRAS（点名两个被排除的对及理由）；写明 cluster 级聚合规则（并注意 `BindingGYM_cluster.tsv` 把 PSD95×2、5A12×2、GB1×2 各并成一簇 ⇒ **specificity 必须逐对报，绝不能报 cluster-level mean**；反过来 BH3 Mcl-1 / Bcl-xL 明明共用同一 517-variant 库、label ρ=0.592，却被官方分成两个单簇 ⇒ cluster 聚合会把这对相关 assay 双重计数，plan 使用的聚类必须显式合并它们）；对齐后固定两个 assay 的解码顺序（同 seed、同排列），并用 ≥8 replicates 测出 specificity 的噪声地板后才允许宣称任何非零值。

### M22. 漏掉了论文里最直接的一条反面证据（adaptation 收益上界）
- Table 5 / §4.3：用 benchmark 中 4/5 assay 的**真实标签**微调 ProteinMPNN，Spearman = **0.42**，相对 zero-shot 0.40 只涨 **+0.02**（原文「showing a slight improvement… the improvements are marginal」）。⇒ 有标签、跨 20 个 assay 也只买到 +0.02，一个无标签、只看单个 WT complex 的 test-time adaptation 期望收益必须远低于此。独立交叉验证：overview §7.1.5 实测 inter_cluster ProteinMPNN = 0.422（+0.025），且 **8/25 个 assay 反而变差**（最惨 Z-domain_ZSPA-1_LL1 −0.294）—— 比论文自称的「两个 outlier」多得多。
- 编辑：§3 新增 3.4 写入上述内容，并据此把 §1 的现实目标区间锚定为 **[+0.005, +0.02]**，让 S1/S2 的功率计算有真实效应量锚点。

### M23. 漏掉了论文对「adaptation 在哪些 assay 上反向」的点名归因
- §4.4：「the two outlier points for ProteinMPNN, which fall below the diagonal line, are derived from the same study on protein co-evolution [60]」，机制解释是「A 上有害的突变在 B 发生互补突变后可能变中性」；ref [60] = Yang et al., Science 381(6656):eadh1720 (2023)；SI 数据表把 Z-domain 的 ZpA963 与 ZSPA-1 两行（共 4 个 assay）都归到该 DOI。这 4 个 assay 的多链同改比例实测 92.9% / 55.2% / 99.6% / 95.9%。
- 编辑：§4 新增「已发表印证」，措辞用「**唯一被点名归因**的 adaptation 反向家族」而非「唯一确定会变差」，并注明 overview §7.1.5 实测是 8/25 变差。同时把至少 1 个 Z-domain assay 纳入 S1a′（见 B15）。

### M24. 15/25 个结构是同源模型 —— 与目标方向无关的第二个损伤源（MISSING）
- POI 带 `_hm` 的有 15 个（含 refutation 用的 `3KZ0_hm` 与 S1a 的**全部 4 个**）；非 `_hm` 的 10 个是 6M0J、6M17_BE、6VJJ×2、1LP1×2、2M5A×2、4ZFF_CHL、4ZFG。WT-likelihood TTT 的目标是 max log p(WT|X)，X 是建模坐标时，梯度里有一部分是在拟合建模误差，且幅度随结构质量逐 assay 变化 ⇒ 负结果在 10 个实验坐标 assay 上未必同样成立，而 S1a 里这类 assay 占 **0** 个。措辞注意：BindingGYM 发布的 PDB **全部剥掉了 header**（22 个文件 `grep ^EXPDTA` 全空、首行即 ATOM），所以只能写「非同源模型的实验坐标」，**不能**写「晶体结构」。
- 编辑：§4 或 §5 加这条 benchmark 事实，并在 S1a′ 里至少纳入 1 个非 `_hm` assay（推荐 `Z-domain_ZpA963_HL2_fitness_2M5A`）。

### M25. 打分/训练的 mask 与 X token 政策未写（MISSING）
- `X` 在这些 CSV 里标记结构中未解析的残基：parse_PDB 用 seq index 20（'X'）与 NaN 坐标填充，`tied_featurize` 随后把这些位置 `mask=0`，所以 `mask_for_loss = mask*chain_M*chain_M_pos` **已经**把它们排除在 frozen 打分之外。⇒ harness 事实上有政策，但未写明，且两处 load-bearing：(a) **TTT loss 必须套用同一个坐标有效性 mask**，否则 complexTTT 会被训练去在这些位置输出 `X`，直接破坏 §8 的「两臂完全一致」；(b) 一对 assay 的计分残基数不同，会改变 `_scores` 的归一化。实测：BH3_Mcl-1 chain A 有一段 7 残基 X run，BH3_Bcl-xL chain A 有一段 **49** 残基 X run（0-index offset 28）；**5A12 对的 H 链差异只有 4 个 X 位点**（1-indexed 134,135,140,141：4ZFF 为 X，4ZFG 为 S,S,G,G）—— 即 5A12「partner-blind 输入不同」这件事整体上是未解析残基的产物，不是生物学。9/25 个 assay 有 gap（BH3_Mcl-1 166/173、BH3_Bcl-xL 180/229、KRAS_PICK3CG-RBD 915/1107、HLA-A2_TAPBPR 582/644、CD19_FMC63 445/497、4D5_HER2 1015/1041、KRAS_SOS1 605/643、5A12_VEGF 520/528、5A12_Ang2 648/652），其余 16 个为零 gap。
- 编辑：§8 加「X / mask 政策」小节写死上述；§9 加注 5A12 的跨 assay 输入差异是 4 个未解析占位符。§2/§8 的「L=173」改为「173 中 166 个计分位点」。

### M26. 被打分的 variant 集合从未定义（MISSING）
- 官方指标脚本 assert 每一行输入都被打分（`assert df.shape[0] == orig_df.shape[0]`），官方打分器**只读 `mutated_sequence`**（从不读 `mutant`），所以不过滤任何行、WT 行也被打分，且分数只在 `mask` 上求和。而先前的 harness 两条都违反：`if ok:` 丢弃 `mutant` 与 `wildtype_sequence` 不一致的行，且分数是**无 mask** 的 `(one_hot(b,21)*lp).sum((1,2))` —— 在没有坐标的位置也计 NLL（gap 是 `wildtype_sequence`/`mutated_sequence` 里真实存在的 `X` 字符）。
- 编辑：§8 写死：(a) 逐行按 `mutated_sequence` 原样打分、不做一致性过滤、不排除 WT 行，并按 assay assert `n_scored == n_rows`；(b) loss/score mask 用 `tied_featurize` 的 `mask`。

### M27. 指标实现与分数列未钉（MISSING）
- 已发布数字用的是 **`global_score`**（不是 `design_score`；顺带：本 benchmark 里两者恒等，因为 `chain_id` 覆盖全部链、`fixed_chain_list` 空，我在 25/25 上验过 `mask*chain_M*chain_M_pos == mask`）。`refs/*.csv` 是指标**输出**，生成脚本**不在工作树里**，只存在于 git 历史的**唯一一个 commit**：`git show ee4e25e:calc_metric.py`（`994243c^` 亦解析到 ee4e25e）。仓库里还有第二套互不一致的 Hit 家族实现：`training/main.py:186` `precision = (top_pred_idxs[:min(k,m)]<m).mean()` vs `calc_metric.py:36` `hit = (top_pred_idxs[:min(k,n)][:min(k,m)]<m).mean()`。
- 编辑：§8 写明分数列 = `global_score`；指标代码 = 从 `ee4e25e` 恢复的 `calc_metric.py`（`calc_zero_shot_metric` + `calc_two_extreme_metric`），vendored 进本项目以做版本钉定；明确 `training/main.py:186` 的 BottomHit 实现不同、**不得**使用。line 247 的「refs/ 里官方发布文件的同一套 6 指标」改为「论文 Table 2 的 6 指标，取自 refs CSV 的 18 列中的对应列」（六个指标本身是论文自己的 headline 集，属性归错了对象而非数量错）。

### M28. 链布局的假设未做断言（MISSING）
- 官方打分器把突变序列按 **`chain_id` 字符串顺序**从 index 0 起连续写入 `S`，并依赖 `tied_featurize` 用同一顺序；BindingGYM 的 fork 里 `masked_chains.sort()` 被**注释掉**（注释写「sort 不适合数据格式」），所以布局是 `designed_chain_list + fixed_chain_list` = chain_id 字符串序。这里之所以安全纯属巧合：25 个 `chain_id` 字符串本来就按字母升序（我逐个验过 `list(chain_id)==sorted(chain_id)` 25/25、`set(chain_id)` == PDB `seq_chain_*` key 集 25/25、`Σ len(wildtype_sequence[c]) == len(S)` 25/25）—— 所以复用上游 ProteinMPNN（`.sort()` 是启用的）恰好一致。
- 编辑：§8 写明 vendored **BindingGYM 版** `baselines/protein_mpnn/protein_mpnn_utils.py`（不得替换成上游或 StaB-ddG 的副本），并在每个 assay 打分前断言上述三条。若 TTT 臂换用自研批量打分器，还必须加一条与官方脚本在某个 assay 上的 per-variant 分数一致性检查（`design_score` 的 Pearson）。

### M29. §3.2 的「85% 抄写」不迁移到 ProteinMPNN ⇒ 「三条独立证据」实为两条
- §2.2 的标签也错：`mpnn_rescue.py:84` 的 loss 只在 15% 被 mask 的位点算 CE，对应 ProteinTTT 的 `loss_kind="cross_entropy"`，**不是** fitness 任务用的 `unnormalized_cross_entropy`（`base.py:1130-1157` 完全忽略 `mask` 参数，在全部 bs*seq_len token 上算 CE）⇒「15% mask 输入 + 全位点 loss」这个组合从未被测过。但**理由要改**：ProteinMPNN 在 ProteinTTT 的分类里是 `model_kind="autoregressive"`，而「85% 梯度来自抄写」是**双向 MLM 特有**的病理 —— 自回归解码器预测 s_i 时永远看不到 s_i（teacher forcing 只给 s_{<i}），不存在那 85%。
- 编辑：§2.2 标题改为「masked-only denoise 变体（= `loss_kind='cross_entropy'`，**不是** fitness 任务的 all-token `unnormalized_cross_entropy`）」；§3.2 加限定「这条只对双向 MLM 成立，不迁移到自回归 MPNN」；§3 与 §11 的「三条独立证据」改为**两条**。S1a′ 可加第三个 arm（15% mask + 全位点 loss）作完整性对照，但注明其与 §2.1 的差别仅是输入扰动、优先级低于分层扩样。

### M30. 论文摘要的点数与 assay 数是拼接的（可报给作者的第二项）
- 摘要「half a million high-quality points from twenty-five assays」与 §3.1「508,962 data points」：508,962 恰是 **28** 个 curated assay 的和，而 shipped 的 **25** 个只有 **376,446** 行。被剔除的 3 个流感 assay（CR9114_FluAH1 65,094、CR9114_FluAH3 65,535、CR6261 1,887 = 132,516 点 = 26.0%）在 `BindingGYM_cluster.tsv`（28 行）与 FoldX notebook 里都存在，但 `results/` 下无任何记录；它们还携带最深的多点数据（depth_max 16/16/11）。
- 编辑：§5 加此条，并在 D5 里列为第二项要报给作者的内容。

---

## 3. MISSING items the plan must add

按严重性排列。已在 §1/§2 展开细节的，这里只列「必须新增的条目」并指回编号，不重复论证。

1. **官方协议的完整显式清单**（B2/B3/M25/M26/M27/M28）。§8 现在只有三行，必须变成一份可断言的规格：ckpt v_48_020.pt + sha256、M（S0 用 5 / 配对比较用 20，两臂相同）、`--backbone_noise 0.00` + assert `augment_eps==0.0`、构造参数（hidden 128 / 3+3 层 / k=48 / ca_only=False / 21 letters）、`--seed` 显式且两臂共享、分数列 `global_score`、mask 来自 `tied_featurize`、逐行打分不过滤、指标代码 `ee4e25e:calc_metric.py`、vendored BindingGYM 版 utils、三条链布局断言、X 政策。
2. **统计检验章节**（B8）+ **两把尺子/两种主张的区分**（B7/M4）+ **LOAO 下界 0.0125**。
3. **TTT 侧方差的前置测量**（B17）与 **seed 配对的零效应对照**（B6）。
4. **逐 assay 噪声地板的实测计划**（B5）：官方条件下 ≥5 seeds，覆盖极端 —— `BH3_Mcl-1`(518，兼作与历史值的对照)、`BH3_Bcl-xL`、`Z-domain_ZpA963_HL2`(最便宜)、`PSD95_CRIPT`、`GB1_IgG-Fc_fitness_1FCC`(92,891，最大 n)、`KRAS_PICK3CG-RBD`(L=1107，最大 L)、`4D5_HER2_fitness_1N8Z`(2,076/2,080 多链)、`hYAP65_peptide_FunctioncalScore_1JMQ`(depth 21)。报 σ 向量而不是单一 0.008。
5. **§9 的 Variant Alignment 小节 + 排除规则 + 负对照 + 噪声地板 + 逐对 n 与 CI + KRAS 13 对**（B14/M19/M20/M21），以及 **mismatched-partner 对照**（M18）。
6. **§8.1 preconditions 清单**（M12/M13）：切到 vanilla v_48_020；本地化 config.sh 与数据根；L≥931 需独占 A4500 或降 M；S2 的 q 只定义在有 alignment 的链上。并明确写「**数据可用性不在风险清单里**」——25/25 assay CSV、22/22 结构、22/22 MSA、参考指标、v_48_020 全部在本机，`/home` 余 967 G，官方脚本已被端到端跑通 3 个 assay。
7. **BindingGYM 已发表的 adaptation 上界**（M22）与 **§4.4 的 outlier 归因**（M23）。
8. **同源模型这一独立损伤源**（M24）。
9. **重复 assay 的 n=25 / n=24 裁决，作为新决策 D6**（M16）。
10. **摘要点数拼接问题，作为 D5 的第二项**（M30）。
11. **specificity 的 baseline 是 frozen 全复合物 ProteinMPNN 自己的实测值**，列为 S0 的交付物（B12）。

---

## 4. Minor / wording

- line 203/280/281/§8：变体计数口径不一致 —— PSD95 写 1,576（mutant 数）、BH3 写 518（行数），而 §8 的 4,190 = 1577+1577+518+518（行数）。统一为「1,577 行（1,576 mutant + 1 WT）」与「518 行（517 mutant + 1 WT）」。
- §4 line 124：「376,446」是 **25 个 shipped CSV 的行数**，含 **22 条 WT 行** ⇒ variant 数是 **376,424**（`iface_baselines.csv` 的 N 列之和正是这个，两个参考文件用了两个分母）；3 个 assay 无 WT 行（HLA-A2_TAPBPR、Z-ZSPA-1_LL1、LL2）。所有百分比按 376,424 计（singles 9.329%、≥3 35.070%、多链 21.317%，标称精度不变）。建议补一行「depth==2 占 55.6%（209,296）⇒ **90.7% 的数据是多点突变**」以加强 §4 结论 1。
- §4 line 125：「148/217 纯单点 vs 9.3%」是两种单位。拆成两行：「纯单点 assay 数 148/217 (68.2%) vs 5/25 (20%)」与「variant 级单点占比 28.24% (696,311/2,465,767) vs 9.33% (35,116/376,424)」（3.0× 而非 all-vs-9.3%）。BindingGYM 的 5 个纯单点 assay 是 ACE2_SARS2-RBD_enrich_6M17、CXCR4_CXCL12_enrich_8U4O、HLA-A2_TAPBPR_meanscore_5WER、PSD95_CRIPT_1BE9、PSD95_Tm2F_1BE9。
- §4/§7/§9 的 assay id 全部改成 canonical DMS_id：`Z-domain_ZSPA-1_LL1_fitness_1LP1`、`hYAP65_peptide_FunctioncalScore_1JMQ`（`Functioncal` 是上游拼写，**代码里不得「修正」**）、`GB1_IgG-Fc_fitness_1FCC`（注意存在两个 GB1 1FCC assay：92,891 行的 `..._1FCC` 与 22,176 行的 `..._1FCC_2016`）。S1a 现有的四个 id 已是 canonical。
- line 157-158：「约 14 个簇」改为「14 个簇（官方 `training/cache/BindingGYM_cluster.tsv`，25 assay；28 assay 时为 15）」，并列出真实组成：KRAS 6、Z-domain 4、5A12 2、GB1 2、PSD95 2、其余 9 个单簇（4D5、ACE2_SARS2-RBD、BH3_Bcl-xL、BH3_Mcl-1、CD19、CXCR4、HLA-A2、hYAP65、SARS2-RBD_ACE2）。plan 现在的枚举漏了 5A12 那一簇，重构出来是 15 不是 14。
- line 160：「43% 的 variant 完全不碰界面」是**25 个 assay 的未加权 assay 均值**，不是 variant 比例。variant 加权是 **39.68%**（149,368/376,424），且分布双峰：9/25 ≤0.1%，11/25 ≥61%（CD19 96.5%、HLA-A2 87.7%、CXCR4 87.4%）⇒「43%」不描述任何一个 assay。
- §5.1 可加一句强化：19,226 个真 variant（19,227 含 WT 空 key）= `KRAS_SOS1_8BE4` 的 **99.0%**，即该 assay 几乎整体是 5O2S 的换链副本。次高相关对是 GB1_IgG-Fc_fitness_1FCC vs _1FCC_2016（label ρ=0.960，160 个共享 variant）。
- line 15：把 +0.3970 标注为「由 `results/ProteinMPNN_zero_shot_metric.csv` 复算（mean 0.396950）」而不是「已发表」——论文只发布 0.40（Table 2）与 0.40±0.03（Table 6/11），且**完全不写** checkpoint / M / backbone_noise（`grep -niE "v_48|soluble|num_seq|backbone_noise"` 命中 0）⇒ 协议的唯一权威是代码。
- §8 line 252：吞吐单位改为「residue-scoring 单位/s = n×L/wall」；并注明 223/s 是自研 bs=48 批量打分器的数字，官方逐 variant 脚本在同一 assay 实测 205/s。
- 全文把「晶体结构」改为「非同源模型的实验坐标」（发布的 PDB 无 EXPDTA，无法判定方法）。
- 引用 overview §7.1.4 的「两套排序相关性」表时注明其列名有误（实为 Pearson）。
- §2.4 的公式按代码口径补注：官方分数是 mask 上 NLL 的**求和**取负、对 M 个解码顺序取均值，无 WT 项、无长度归一化；两者在 assay 内是保序常数，但使**跨 assay 的绝对分数不可比**。

---

## 5. What survived audit unchanged

这些经多维度独立核查后**逐位成立**，可以直接保留并在修订版里标为「已验证」：

**数据层面**
- §5.1 的 KRAS 重复 assay 结论**完全复现**：19,533 / 19,425 行、裸 `mutant` 字符串 join = **0** 行重叠、抹掉 chain id 后 **19,227** 个共享 key、max|Δlabel| = **0.0**、逐值相同比例 1.000、ρ = **1.0**、chain_id 是 AB vs RS。这是全 plan 里最干净的一条发现。
- §4 逐 assay 表格的**每一个数字**都从原始 CSV 复算无误（Z-LL1 45,476/3/45,436/9/45,285；5A12_VEGF 29,981/54/29,751/9/24,452；4D5 2,080/0/2,079/9/2,076；hYAP65 18,407/288/11,091/**21**/0；GB1 92,891/1,045/0/2/0），depth_max=21 正确，Σ 行数 376,446 正确，三个汇总百分比在标称精度下正确。
- 「**14 个簇**」是精确值而非估计（官方 `BindingGYM_cluster.tsv` 28 行 / 15 representative，剔除 3 个流感 assay 独占的那一簇）。§5.2 要求「必须同时报 cluster-level mean」这条判断是对的，且被证据包独立支持。
- §9 的三个 label ρ（**0.435 / 0.592 / −0.145**）全部复现（复算 0.4348 / 0.5919 / −0.1452）。
- 聚合口径正确：+0.3970 是 25 个 assay 的**未加权均值**（0.396950），且 25/25 全为正 ⇒ signed 与 absolute mean 一致。
- §1 榜单表里其余各行均复现（MPNN_single 0.3564、TranceptEVE 0.3432、PiFold 0.3380、ESM-if1 0.3378、ESM2 0.2851）。

**协议与噪声**
- **M=20 的噪声地板 ≈0.008 站得住**：官方代码路径 + vanilla ckpt + backbone_noise=0.00 上 6 seeds 实测 σ = **0.0093**，与 0.0080 在 n=6/8 下不可区分。由它导出的「M=20 单 assay 1σ 量级」这个判断是对的（只有 M=5 那一行与外推方式要改）。
- **223 sequence-scoring/s 这个锚点是公允的**：官方脚本在同一 assay 同一 M 实测 50.6 s ⇒ 205/s，差 9%（尽管两者 batching 完全不同）。只是它只在 L≈173 成立。
- **§8 的 S1a 两行预算是保守的**：一趟打分实测/拟合 ≈5.4 min（plan 写 ~6 min）、21 趟 ≈1.9 h（plan 写 ~2.5 h）；4,190 的 variant 计数精确。原因也对：那 4 个 assay 全在 L=120–229，贴近计时锚点。
- **D1「先跑 10 个中小 assay（~1.5 h）」的建议成立且偏保守**（最便宜的 10 个实测 0.83 h @M=20）。
- `design_score ≡ global_score` 在本 benchmark 上恒等（`chain_id` 覆盖全链、`fixed_chain_list` 空，25/25 验证）—— 这意味着 plan 未指定分数列并没有造成数值错误，只是缺钉定。
- **「frozen 与 TTT 两臂必须完全一致」**这条纪律本身是对的，也正是它让上面一串协议缺口变成必须补的清单而不是随意项。
- **「S0 不过就不跑 S1」的 gate 纪律**是对的，且理由（gate 抓出过三个真 bug）成立 —— 需要改的只是 gate 的**容差**，不是 gate 本身。

**判断层面**
- §6 的定性结论成立：零参数几何特征打平的是 **PLM（ESM2 0.2851）而不是 ProteinMPNN**，MPNN 高出约 **0.14** ⇒ baseline 有真实信号、因此更难被超过。（只有表头那一行 `rho_burial_on_iface` 不可比。）
- §2.1 中 **lr ≥ 0.01 的崩坏是真的**（Δ=0.0403 ≈ 3σ vs M=5 地板，≈2.2σ vs 官方实测地板）；把 §1 的 §2.4 机制改述为熵坍缩后，**「以 δ_WT 为最优解的目标会破坏排序」这个方向性判断依然站得住**，只是它只预测「损伤 ∝ baseline 携带的排序信息量」，不预测 benchmark 级普适失败。
- §5.3 要做 **interface vs non-interface 分层**的直觉是对的（只是标签定义与可执行性要修）。
- §11 的定位判断 —— **把 S1 当作必要的负对照而不是希望所在，S2 才是有机会的那一段** —— 与所有六个维度的证据一致；BindingGYM 自己 Table 5 的 0.40→0.42 上界（M22）反而进一步支持它。
- 资源可用性：**没有任何东西因为缺数据/缺权重/缺磁盘而不可跑**（25/25 CSV、22/22 结构、22/22 MSA、v_48_020、967 G 空闲，官方脚本已端到端跑通）。所有 blocker 都是 plan 的编辑或调度约束。

---

## 6. Recommended revised success criterion and revised S0/S1a design

以下可直接替换 plan 的 §1、§7 S0、§7 S1a 与 §8 预算表。

### 6.1 §1 替换文本 —— 打分协议（先钉死，一切阈值依赖它）

```
打分协议（S0 与 S1 两臂逐字相同，harness 启动时全部 assert）
- checkpoint: /home/guoj0f/repos/BindingGYM/training/cache/v_48_020.pt  (vanilla)
  sha256 c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd
  md5    91d54c97a68bf551114f8c74c785e90f
  禁止使用 StaB-ddG/model_ckpts/proteinmpnn.pt（soluble，118/118 tensor 不同，cosine 0.0090）
- M: S0 与 refs/ 比对时 = 5（官方 --num_seq_per_target 5）；frozen-vs-TTT 配对比较可用 20，
     但两臂必须同 M，且 M=20 的数字禁止与 refs/ 直接比较。--batch_size 是死参数（BATCH_COPIES 从不使用）。
- --backbone_noise 0.00；assert model.features.augment_eps == 0.0
  （checkpoint 里的 noise_level=0.2 只被打印；StaB-ddG 的 ProteinMPNN class 默认 k=32/eps=0.1，禁止复用）
- 构造: hidden_dim=128, num_encoder_layers=num_decoder_layers=3,
        k_neighbors=checkpoint['num_edges']=48, ca_only=False, num_letters=21
- --seed: 显式传入；frozen 臂与 TTT 臂共用同一 seed 与同一 randn_1_dic（randn_1 按 POI 缓存、
  该 assay 全部 variant 共用 ⇒ 解码噪声是共模扰动，不随 n 衰减）
- 分数列 global_score = -1 * mean_over_M( Σ_i mask_i · NLL_i(mutant) )
  （无 WT 项、无长度归一化；assay 内保序，跨 assay 绝对值不可比）
- 逐行按 mutated_sequence 打分，不做一致性过滤、不排除 WT 行，assert n_scored == n_rows
- loss/score mask = tied_featurize 的 mask（X/未解析位点 mask=0；TTT loss 必须套同一 mask）
- 指标代码 = vendored `git show ee4e25e:calc_metric.py`（training/main.py:186 的 BottomHit 不同，禁用）
- vendored BindingGYM 版 protein_mpnn_utils.py；每 assay assert
  list(chain_id)==sorted(chain_id)、set(chain_id)==PDB seq_chain keys、Σlen(wt_seq[c])==S.shape[1]
- 数据根: <绝对路径>/input（BindingGYM/input 需 symlink；zenodo record 12514160）
- 设备: 独占 A4500 cuda:0（assert 'A4500' in torch.cuda.get_device_name(0)）；
  L>=931 的 assay（1HE8/1N8Z/6M17）在 M=20 下需 11–15 GiB，非独占时降到 M=5–10 并记录 per-assay M
```

### 6.2 §1 替换文本 —— 三种主张、三把尺子

```
(A) 绝对榜单主张（"超过 +0.3970 上榜"）
    不确定度 = ±0.03（论文 Table 6/11，对 assay 集合 bootstrap；复算 sd 0.1706 / √25 = 0.0341）
    ⇒ 在这固定的 25 个 assay 上，TTT 量级的效应不可能撼动它。本 plan 不宣称 (A)。

(B) 配对榜单主张（"同一 harness 上，这 25 个 assay 的 Δ̄ > 0"）  ← 本 plan 的主判据
    assay 集合项抵消，只剩 replicate 噪声（含 TTT 侧）。
    阈值 = 3 × 实测 sd(Δ̄)。当前最佳先验：sd(Δ̄) 在两臂 seed 配对下应远低于
    √2·σ/√24；未配对时 = 0.0023 (M=20) / 0.0039–0.0043 (M=5) ⇒ 3σ = +0.007 ~ +0.013。
    最终阈值由 F1–F3 前置测量确定，且不得低于 LOAO max shift = 0.0125。
    plan 原来的 +0.005 = 2.21σ (M=20) / 1.30σ (M=5)，不是 3σ，作废。

(C) 方法学主张（"TTT 改善 complex binding zero-shot"）
    配对跨 assay 检验。代理 sd_d = 0.0526（MPNN vs MPNN-1chain）/ 0.0697（同权重换输入）
    ⇒ MDE80 = +0.030 @24 assay / +0.039 @14 cluster。（两者是代理量，非实测 TTT Δ 异质性。）
    按 §3.1 外推的期望增益 +0.0018 比 MDE 小 16 倍 ⇒ 均值层面的正、负结论都不可判定。
    现实目标区间锚定于 BindingGYM 自己的有标签 adaptation 上界：0.40 → 0.42（Table 5/§4.3;
    overview §7.1.5 实测 0.422 且 8/25 assay 变差）⇒ [+0.005, +0.02]。

统计检验（开跑前锁定，不得跑完再选）
    primary   : 逐 assay 配对 Wilcoxon signed-rank，n=24（剔除 KRAS_SOS1_norfitness_8BE4），α=0.05 双侧
    secondary : 14-cluster bootstrap（重采样整簇，20,000 次）的 Δ̄ 95% CI；KRAS 6 个先内部平均
                （cluster 划分引 training/cache/BindingGYM_cluster.tsv，但必须手动把
                 BH3_Mcl-1_normed_3KZ0 与 BH3_Bcl-xL_normed_1PQ1 合为一簇 —— 二者共用同一 517-variant
                 库、label ρ=0.592，官方文件却把它们分成两个单簇）
    辅助      : win/loss/tie（|Δρ|>0.05）+ leave-one-assay-out 均值敏感性
    specificity: 逐对报，绝不做 cluster-level mean（PSD95×2 / 5A12×2 / GB1×2 各被官方并成一簇）
```

### 6.3 前置测量 F1–F4（在 S0 之后、S1a 之前，全部 < 3 GPU-h）

| id | 内容 | 规模 | 交付 |
|---|---|---|---|
| F1 | frozen-vs-frozen **同 seed** 零效应对照 | 2 assay × 1 seed | Δ 必须恒等于 0；不为 0 ⇒ seed 未真正配对，停 |
| F2 | 逐 assay 打分噪声 σ | 8 assay × ≥5 seeds，官方条件（M=5 与 M=20 各一组） | σ 向量（含 BH3_Mcl-1 518、GB1 92,891、KRAS_PICK3CG-RBD L=1107、4D5_HER2 多链、hYAP65 depth21） |
| F3 | **TTT 侧** run-to-run 方差 | 2 assay、固定 (lr,steps)、≥5 TTT seed × 同一打分 seed | sd_TTT ⇒ 阈值 = 3×√(sd²_TTT + sd²_score,paired)/√n |
| F4 | specificity 噪声地板 | 3 对、≥8 replicates、两 assay 固定同一解码顺序 | 地板值；未测出前不得宣称任何非零 specificity |

### 6.4 §7 S0 替换文本

```
S0a — 子集 harness gate（先跑这个；~2.0 h @M=20，~1.0 h @官方 M=5）
  assay 集 = 按 n·L_total 最便宜的 10 个（CXCR4_8U4O, 5A12_Ang2_4ZFG, hYAP65_1JMQ,
    Z-ZSPA-1_LL2, Z-ZpA963_HL1, PSD95_Tm2F, PSD95_CRIPT, BH3_Bcl-xL, BH3_Mcl-1, Z-ZpA963_HL2)
    + 强制加 2 个覆盖用 assay：4D5_HER2_fitness_1N8Z（L=1041、3 链、验 M=20 显存路径）
      与 KRAS_RAF1_norfitness_6VJJ（覆盖 KRAS 簇）
  合计实测 1.95 h @M=20。（不要把 1HE8/4ZFF/5O2S 放进 gate —— 那是 10.7 h。）
  判据（每个 assay 各自）:
    我们跑 ≥5 个显式 seed，得到 [min,max] 与 mean±σ_ours；
    PASS 当 refs/ 的参考值落入我们的 [min,max]，或 |mean_ours − ref| ≤ 2σ_combined，
    σ_combined = √(σ_ours² + σ_ref²)，σ_ref 用 F2 在该 assay 上实测的 M=5 σ
    （BH3_Mcl-1 上已实测 σ_M5 = 0.019–0.021、σ_M20 = 0.0093 ⇒ σ_combined ≈ 0.0225 ⇒ 容差 ≈ 0.045）
  子集均值判据: 与写死的参考子集均值比 —— 10-smallest-by-n = 0.348913，
    10-smallest-by-Σn·L = 0.356467（本 gate 用后者 + 2 个覆盖 assay 时须重算并写死）
  额外交付物（不是可选项）:
    (i) 逐 assay σ 向量；(ii) frozen 全复合物 ProteinMPNN 自己的 partner-conditional specificity
    （三对，逐对报 n 与 CI）；(iii) 全部 assert 的通过记录；(iv) 一个长 assay 的端到端 wall time。

S0b — 全量 25 assay（可与 S1a 并行排队）
  成本：≈14–16 h @官方 M=5；≈28 A4500-h @M=20（实测外推，Σn·L=135,744,347；纯线性下界 19.5 h）
  判据：25-assay mean 落在 0.396950 ± 0.009（= 2 × combined SE 0.0045）。
  比对必须用**全部 25 个 assay**（+0.3970 是 25-assay 均值）；此后所有 Δ 报告改为 n=24。
  声明：refs/ 是一次未设种的 M=5 单抽，逐 assay 逐位复现在原理上不可能；
        randn_1 按 POI 缓存 ⇒ 解码噪声不随 n 衰减，大 assay 不更安全。
```

### 6.5 §7 S1a′ 替换文本

```
S1a′ — 分层小样本（7 assay，Σn·L ≈ 1.11 M ≈ 1.9× 原 S1a）
  保留（refutation 锚点 + partner-swap + 高/低 baseline 两端）:
    BH3_Mcl-1_normed_3KZ0 (518, L=173, baseline 0.6625, 单簇)
    BH3_Bcl-xL_normed_1PQ1 (518, L=229, 0.6554, 单簇)
    PSD95_CRIPT_1BE9 (1577 行, L=120, 0.3863)
    PSD95_Tm2F_1BE9 (1577 行, L=120, 0.2073)  ← 与上一个同簇；且这一对不是 partner swap，
                                                 partner 只差 1 个残基（KQTSV vs KQFSV）
  新增（覆盖 §4 自认的最大风险面）:
    Z-domain_ZpA963_HL2_fitness_2M5A (600, L=116; 55.2% 多链; 非 _hm 实验坐标; 全 benchmark 最便宜)
    Z-domain_ZpA963_HL1_fitness_2M5A (2904, L=116; 92.9% 多链; 最低 baseline 层; paper §4.4 点名家族)
    hYAP65_peptide_FunctioncalScore_1JMQ 随机子采样 2,000 (depth_max=21)
  预算够再加: KRAS_RAF1_norfitness_6VJJ 子采样 2,000（覆盖 6/25 assay、30.4% variant 的 KRAS 簇）
  覆盖声明（必须写进正文）: 这 7 个覆盖 5 个 cluster / 4 个家族；多链同改 3 个、depth≥3 有 2 个、
    非同源模型结构 2 个；baseline 跨 0.136–0.663。仍不可外推到 25 assay。

  配置:
    lr ∈ {1e-4, 3e-4, 1e-3, 3e-3}      ← ≥1e-2 已被 prior 证明必崩，不再花预算
    steps ∈ {0,1,2,3,5,10,20,30}       ← step-wise 打分（prior 只有 step0/step30 两个端点，
                                          "单调" 从未被测量）
    每个 step ≥3 个打分 replicate，M=20，打分 seed 与 frozen 臂共用同一 randn_1
    TTT seed ≥3（与打分 seed 分开记账，见 F3）
    第三个 loss arm（15% mask + 全位点 loss = fitness 任务真正用的 unnormalized_cross_entropy）
      作完整性对照，优先级低于分层扩样
    附加 arm: vanilla v_48_020 上重跑 BH3_Mcl-1 的 lr 扫描，作为 §2（soluble 权重）的桥接

  判定（不再用 72 次 1σ 比较 —— 那期望产生 17–24 个假窗口，P(至少一个)≈1.00）:
    主判据: 每个 lr 的每个 step，取**跨 7 assay 的配对均值 Δ̄₇**，
            与 3 × 实测 sd(Δ̄₇)（来自 F2/F3）比较，并对 4 个 lr 做 Holm 校正 ⇒ 32 次比较压成 4 条曲线
    形态判据（对独立噪声鲁棒）: 同一 lr 在 7/7 个 assay 上 Δ>0 且 step-wise 单调上升
    双侧读数: 既判"有没有超过 3σ 的窗口"，也判"小 lr 是不是只是零结果"
    partner-specificity: 用 oracle 锚定量 |ρ_model_cross − ρ_label_cross| 或 matched/mismatched
      配对检验（把 CRIPT 的 variant 集在 Tm2F 复合物上打分，反之亦然；BH3 同理），
      对齐程序见 §9（BH3 需 chain C→B + offset +2，assert join = 518 行），
      并与 F4 的地板、以及 S0a 交付的 frozen ProteinMPNN 自身 specificity 比较

  成本: 一趟打分 ≈9.6 min @M=20；全网格 ≈4.7 GPU-h（含 replicate 后按实测 σ 重算）
```

### 6.6 §8 预算表替换

| 项 | 工作量（n × L_total × M） | 估时（独占 A4500） |
|---|---|---|
| S0a 子集 gate（12 assay，M=20，5 seeds 已计入 F2） | ~4.0 M residue-variant·M | **~2.0 h**（M=5 时 ~1.0 h） |
| S0b 全量 25 assay，一趟 | Σn·L = 135,744,347 | **≈14–16 h @M=5 / ≈28.3 h @M=20**（线性下界 19.5 h） |
| F1–F4 前置测量 | — | **< 3 h** |
| S1a′ 一趟打分（7 assay） | ~1.11 M | **~9.6 min** |
| S1a′ 全网格（4 lr × 7 step × 7 assay × 3 replicate + TTT） | — | **~4.7 h** |
| S1b 全量 25 assay × 5 TTT seed（仅在 S1a′ 出窗口） | — | **~142 h @M=20 / ~75 h @M=5**；frozen 臂逐 seed 重打分则 ×2 |

承诺范围改为：**S0a + F1–F4 + S1a′ ≈ 10 GPU-h**（原文「S0 + S1a ≈ 12 GPU-h」按错误的 9.4 h 算，且不含任何 replicate）；S0b 单独排一次独占长跑。成本大头（按实测）是 `KRAS_PICK3CG-RBD_1HE8` 6.45 h (22.8%)、`SARS2-RBD_ACE2_6M0J` 4.31 h、`GB1_IgG-Fc_1FCC` 3.85 h、`5A12_VEGF_4ZFF` 3.23 h、`KRAS_SOS1_8BE4` 2.80 h —— 不是 plan 写的 GB1 + Z-domain LL1（后者只占 0.65 h = 2.3%）。1HE8 的 6.45 h 同时是全量 S0 的 Amdahl 下界（分片粒度是一个 assay 一个进程，且无 assay 内 checkpoint）。

### 6.7 决策表需要改的项

- **D1**：保留「先 10 个」，但按 Σn·L 选并强制加入 `4D5_HER2_fitness_1N8Z` 与 `KRAS_RAF1_norfitness_6VJJ`（合计 1.95 h）；并把「全量 S0 = 9.4 h」改为 14–28 h。注：§5.1 的重复 assay 检查是纯 label join，不需要 GPU 时间。
- **D2**：**撤销「直接进 S2」的现有理由链**（机制论证对 Spearman 为空、单调性从未测量、§2 跑在另一个模型上）。改为：S1a′ 的 step-wise 曲线是唯一能测单调性的实验，必须先跑完；S1b 的取舍改在 S1a′ 结果之后、并以 142/75 GPU-h 的真实成本重新权衡。
- **D3**：在指标按 B13 重定义、且 S0a 交出 frozen ProteinMPNN 自己的 specificity 之前，**不能**把它定为并列 headline。
- **D4**：MSA 的 POI 级覆盖率问题已答（22/22，本机 633 MB）；改问「S2 的 q 能否只定义在被突变链上」（全复合物覆盖仅 2M5A 与 1LP1）。
- **D5**：加第二项 —— 摘要把 28-assay 的 508,962 点数与 25-assay 的 assay 数拼接（shipped 只有 376,446 行）。
- **新增 D6**：S0 用 n=25 比对、S1b 及所有 Δ 报告用 n=24（剔除 `KRAS_SOS1_norfitness_8BE4`），两个 n 在文中不得混用。