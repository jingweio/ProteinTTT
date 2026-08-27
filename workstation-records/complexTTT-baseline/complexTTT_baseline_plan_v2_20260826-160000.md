# complexTTT-baseline — experiment plan **v2**  (2026-08-26; status: PLANNED, 未启动任何 GPU 任务)

> **v2 说明**：v1（`complexTTT_baseline_plan_20260826-095745.md`）经一轮对 BindingGYM 论文 +
> 代码仓库 + `BindingGYM/reproduce/BindingGYM_overview.md` 的系统审计，发现 **17 条 blocking**、
> **30 条 major**。v1 保留在目录里作为记录，**不要照它执行**。本文件是唯一可执行版本。
> 审计的完整修订指令存档于 `refs/audit_revision_20260826.md`。

把 ProteinTTT 的 test-time training 协议搬到 complex binding：对 pretrained **ProteinMPNN**
在 BindingGYM 的 WT 数据（complex structure + 全部链序列）上做自监督微调，零样本设定。

---

## 0a. v3 修订（2026-08-27）—— 用户指定的删改 + 一轮新分析的更正

按用户要求：**删除**「ProteinGym 复现的三条新证据」一节（原 v1 §3 / v2 §4）与「ProteinMPNN 超越
几何 baseline」一节（原 v1 §6）。两节的内容分别被下面 §A/§C 的更强证据取代。

一轮针对 (4)(6)(10) 的分析（3 路独立分析 + 对抗复核）**推翻了 v2 里若干条我自己的表述**，
逐条更正见 `refs/analysis_q4_q6_q10_20260827.md`。对本 plan 最要紧的六条：

| # | v2 的说法 | 更正 |
|---|---|---|
| **N1** | 未使用 | ⭐ **ProteinTTT 自己 Table 2 的 Binding 列增益均值只有 +0.0012、Stability 为负**（`ttt.txt:435-450`）。**这一条足以支撑负面预期，且不需要任何机制论证。** 计划若要往下走，必须先回答「凭什么突破 +0.0012」 |
| **N2** | §4.1「0.3970 落在 TTT 给 +0.019 那一档」 | ❌ **算错了**。复算 ESM2-150M = 0.3868、650M = 0.4138 ⇒ 0.3970 落在 **150M 与 650M 之间**，最近的上一档是 650M，其实测增益 **+0.0010**。**按同一映射逻辑结论应反过来** |
| **N3** | §3.1「单调下降、没有操作窗口」 | ❌ 正确表述是「**大 lr 下急剧退化，而真正的工作区间从未被检验**」。lr=1e-3 的 Δ=−0.0119 落在 1σ 内；masked-denoise 臂只测了 2 个 lr、**都 ≥1e-2**；而 ProteinTTT fitness 用的是 **AdamW 4e-5–4e-3** |
| **N4** | 「TTT 主要改善 low MSA depth」（引 paper） | ❌ **与 paper 自己的 Table A4 矛盾**：ESM2-35M 分层增益 Low **+0.0051** < Medium **+0.0437**。且 Table A4 的 Low/High baseline 与 leaderboard 不一致 —— **该表不可用于机制推理** |
| **N5** | §12「partner-blind 恒等于 0.000」（已改为"≈0"） | ⚠️ 仍需再改：partner-blind 打分器对两个 partner 给出**同一个常数向量**，相关系数是**未定义**而非 0 ⇒「唯一 baseline 做不到的读数」目前是**同义反复、不可 falsify**。必须先规定约定 |
| **N6** | 未使用 | ⭐ **KRAS 重复 assay 给出一把免费的尺子**：同一批 label、不同结构+partner ⇒ ProteinMPNN 的 per-assay ρ 差 **0.0948**（0.4040 vs 0.3092），14 个模型 mean \|Δ\| = **0.0613**；序列模型只有 0.004–0.010，**结构模型 0.07–0.28**。ProteinMPNN 那 0.0948 是 §2(B) 的 MDE（0.021）的 **4.5 倍** |

| **N7** | §11「BindingGYM 零 MSA ⇒ S2 的主增益路径开箱不可用」 | ❌ **错了。BindingGYM 官方提供 MSA**，在 `input/msas/`（22 个 `.a2m`，633 MB，来自 Zenodo record 12514160 的 `input.zip`）。我早前那份 rsync 来源是 session scratchpad，**它本身缺 `msas/`**，所以我的 find 与后续论证都建立在一个不完整的副本上。已补齐：`~/share/BindingGYM/input/msas` 与 `/data/guoj0f/share/BindingGYM/input/msas` 两侧都有。⇒ **S2（backbone-conditioned homolog denoising）的 q 有现成来源，不需要自建 paired MSA** —— 这是本 plan 唯一被**放松**的约束。但见下方 MSA 的三个口径限制 |

**MSA 的实际口径（已核，25/25 assay）**：
- **query = 被突变链的拼接**，不是完整复合物（raw query 长度 25/25 精确等于被突变链总长）
- **文件按结构名索引**（22 个文件覆盖 25 个 assay）⇒ 共享结构的 assay 共享 MSA
- **对齐核心（大写列）常远短于 query**：BH3_Mcl-1 仅 **13/23**、hYAP65 34/46、PSD95 87/115；
  SARS2-RBD 是唯一全覆盖（194/194）。多链突变时更糟：4D5 的 query 434（A 214 + B 220）
  而核心只有 **220** ⇒ **实际只对齐了其中一条链**
- 深度跨度极大：**521**（GB1）到 **756,017**（PSD95），median 14,724

**新增的一条正面证据**（C.2 实测，BH3 peptide 在 3KZ0/1PQ1 上 518/518 对齐）：
**匹配骨架 0.6813 vs 错配 partner 0.3713（Δ = +0.310）**；而**错骨架（0.3713）比完全没有 partner
（0.5975）更差 0.226** ⇒ **固定 WT 骨架是双向赌注、不是免费先验**，而 BindingGYM **15/22 个结构带 `_hm`**。

⚠️ 该组实测跑在 **StaB-ddG 的 soluble 权重**上（每臂 n=1 seed、只有一对 peptide），作单点证据保留、
不可当定量结论。

## 0. v1 → v2 的实质性改动（先看这个）

| # | v1 的说法 | 实际情况 | 后果 |
|---|---|---|---|
| **B1** | checkpoint 用 `StaB-ddG/model_ckpts/proteinmpnn.pt` | 那是 **soluble** 变体（其 README:50 自陈）；官方用 **vanilla** `v_48_020.pt`。**118/118 个 tensor 全部不同**，cosine 0.0090，‖θ‖₂ 373.94 vs 410.90 | v1 的 §1 噪声地板与 §2 全部否证**跑在另一个模型上**，绝对值与榜单不可比 |
| **B2** | M=20 且"复用官方口径" | 官方 launcher 硬编码 `--num_seq_per_target 5`；`BATCH_COPIES = args.batch_size` **全文件只出现一次 = 死参数** | 自相矛盾。S0 必须用 M=5 |
| **B3** | §8 未提 backbone noise | 官方推理 `--backbone_noise` **default 0.00**；而 v1 地板的测量脚本用 `augment_eps=0.1`，且噪声分支**无 `self.training` 保护** | v1 地板是"解码顺序 + 0.1 Å 抖动"的混合量，非官方条件 |
| **B4** | S0 判据 per-assay \|Δ\| ≤ 0.008、mean ±0.002 | `refs/` 是**一次未设种的 M=5 单抽**（`if args.seed:` + `default=0` ⇒ 0 是 falsy，每次重随机；launcher 从不传 `--seed`） | 完美 harness 也约 72% 的 assay 随机判失败。实跑 4 个 assay 已 2/4 越界 |
| **B5** | 地板可按 0.008/√25 外推 | `randn_1` **按 POI 缓存、被该 assay 全部 variant 共用** ⇒ 共模扰动，**不随 n 衰减**；且 M=5 的 σ 实测为 **0.019–0.021**（v1 写 0.0136，低估） | 大 assay 并不更稳；不能用任何缩放律代替逐 assay 实测 |
| **B6** | 阈值 = 单臂 σ/√25 | 判据量是**配对差**，sd(Δ̄) = √2·σ/√n；且官方默认**不配对** | +0.005 实际只有 2.21σ (M=20) / 1.30σ (M=5)，不是自称的 3σ |
| **B10** | 机制 = "TTT 在对打分公式的分母 `log p(WT)` 做梯度上升"，且"机制论证是普适的" | **官方打分器根本没有 WT forward**（`_scores` 返回 mask 上 NLL 的**求和**，长度归一化被注释掉）；即便有，减 per-assay 常数**对 Spearman 恒等保序** | ⇒ 这个机制**数学上无法解释**实测退化。必须改述为**熵坍缩/相对校准破坏**，且它**不预测 benchmark 级普适失败** |
| **B11** | "单调下降，没有操作窗口" | `noise.log` 每个 lr **只有 step0 与 step30 两个端点**；lr 方向上本身不单调（lr=0.3 的 0.4783 > lr=0.1 的 0.3884） | "单调"**从未被测量**。S1a 的 step-wise 曲线是唯一能测它的实验 |
| **B12** | PSD95 两个 assay "同一 PDB 1BE9" ⇒ partner-blind 恒等于 0.000 | 三对**全部用不同结构文件**（实核 md5）：`1BE9_hm.pdb`/`1BE9Tm2F_hm.pdb`、`3KZ0_hm`/`1PQ1_hm`、`4ZFF_CHL`/`4ZFG`；且 **ProteinMPNN 吃完整复合物，本来就不是 partner-blind** | §9 的 co-headline 论证前提为假 |
| **B14** | "同一突变库"可直接配对 | 实测原始 `mutant` 字符串重叠：PSD95 1577/1577，**BH3 = 0**，**5A12 = 0**（BH3 需 chain C→B **加** 位点 +2；5A12 只需按被突变链取 key） | 裸 join 会**静默产生空集** |
| **B15** | S1a 4 个 assay "把否证做实" | 这 4 个只有 **3 个 cluster / 2 个家族**，多链同改 **0/0/0/0**，depth_max 仅 1 与 5，n 是 25 个里最小的 4 个（占全 benchmark variant 的 **1.11%**），MPNN mean ρ=0.4779 vs benchmark 0.397 | **系统性避开了 §5 自己认定的最大风险面**，且不可外推 |
| **B16** | S0 全量 ≈9.4 h | 成本随 **n × L_total** 增长且超线性（`_get_rbf` 建 [B,L,L]）。Σn·L = **135,744,347** ⇒ **≈28 h @M=20 / ≈14–16 h @M=5** | 低估 2.1–3.0× |
| **B17** | "25 assay × 5 seeds" | v1 引用的 ±0.0147 等**不是 TTT-seed 方差**，是对同一个已训练模型的多次打分复算 | **TTT 侧 run-to-run 方差从未测量**，阈值等于假装它是 0 |

**审计确认逐位成立的部分**（保留并标为已验证）：KRAS 重复 assay（19,227 个共享 key、max\|Δlabel\|=0.0、ρ=1.0）；
§5 逐 assay 表格每一个数字；"14 个簇"是**精确值**（官方 `BindingGYM_cluster.tsv`）；三个 label ρ（0.435/0.592/−0.145）；
+0.3970 是 25 assay 未加权均值；M=20 地板 ≈0.008（官方条件实测 0.0093）；223/s 锚点（官方实测 205/s，**但只在 L≈173 成立**）；
零参数几何特征打平 ESM2 而非 ProteinMPNN；**"S1 是必要负对照、S2 才是希望所在"这个定位与全部六个维度的证据一致**；
**没有任何东西因缺数据/权重/磁盘而不可跑**。

---

## 1. 打分协议（先钉死；S0 与 S1 两臂逐字相同，harness 启动时全部 assert）

```
checkpoint  /home/guoj0f/repos/BindingGYM/training/cache/v_48_020.pt   (vanilla)
            sha256 c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd
            md5    91d54c97a68bf551114f8c74c785e90f
            ⛔ 禁用 StaB-ddG/model_ckpts/proteinmpnn.pt（soluble；118/118 tensor 不同，cosine 0.0090）
M           与 refs/ 比对时 = 5（官方 --num_seq_per_target 5）
            frozen-vs-TTT 配对比较可用 20，但两臂必须同 M，且 M=20 的数字禁止与 refs/ 直接比较
            --batch_size 是死参数（BATCH_COPIES 从不被使用）
噪声        --backbone_noise 0.00；assert model.features.augment_eps == 0.0
            （ckpt 里的 noise_level=0.2 只被打印；⛔ 禁止复用 StaB-ddG 的 ProteinMPNN class，
              其默认 k_neighbors=32 / augment_eps=0.1）
构造        hidden_dim=128, num_encoder_layers=num_decoder_layers=3,
            k_neighbors=checkpoint['num_edges']=48, ca_only=False, num_letters=21
seed        显式传入；frozen 臂与 TTT 臂共用同一 seed 与同一 randn_1_dic
            （randn_1 按 POI 缓存、该 assay 全部 variant 共用 ⇒ 解码噪声是共模扰动，不随 n 衰减）
分数        global_score = -1 * mean_over_M( Σ_i mask_i · NLL_i(mutant) )
            无 WT 项、无长度归一化（_scores 里 / torch.sum(mask,-1) 被注释掉）
            assay 内保序；跨 assay 绝对值不可比
            本 benchmark 上 design_score ≡ global_score（chain_id 覆盖全链、fixed_chain_list 空，25/25 验证）
打分范围    逐行按 mutated_sequence 打分，不做一致性过滤、不排除 WT 行，assert n_scored == n_rows
mask        = tied_featurize 的 mask（X/未解析位点 mask=0；**TTT loss 必须套同一 mask**）
指标        vendored `git show ee4e25e:calc_metric.py`（training/main.py:186 的 BottomHit 实现不同，禁用）
每 assay assert  list(chain_id)==sorted(chain_id)；set(chain_id)==PDB seq_chain keys；
                 Σlen(wt_seq[c]) == S.shape[1]
数据根      /home/guoj0f/share/BindingGYM/input  （已从 session scratchpad 持久化，327 MB；
            BindingGYM/input 已 symlink 过去；25/25 assay CSV、22/22 结构、md5 已核）
设备        独占 A4500 cuda:0（assert 'A4500' in torch.cuda.get_device_name(0)）
            L>=931 的 assay（1HE8 / 1N8Z / 6M17）在 M=20 下需 11–15 GiB；非独占时降到 M=5–10 并记录 per-assay M
```

---

## 2. 三种主张、三把尺子（**不可互相替代**）

| | 主张 | 不确定度 | 本 plan 是否宣称 |
|---|---|---|---|
| **(A)** | 绝对榜单：">+0.3970 上榜" | **±0.03**（论文 Table 6/11，对 assay 集合 bootstrap；复算 sd 0.1706/√25 = 0.0341） | **不宣称** —— TTT 量级的效应不可能撼动它 |
| **(B)** | 配对榜单：同一 harness 上这 25 个 assay 的 **Δ̄ > 0** | assay 集合项抵消，只剩 replicate 噪声（**含 TTT 侧**） | ✅ **主判据** |
| **(C)** | 方法学："TTT 改善 complex binding zero-shot" | 代理 sd_d = 0.0526（MPNN vs MPNN-1chain）/ **0.0697**（同权重换输入）⇒ **MDE80 = +0.030 @24 assay / +0.039 @14 cluster** | 明确标为**不可判定** |

**(B) 的阈值**：3 × **实测** sd(Δ̄)，由 F1–F3 确定，且**不得低于 LOAO max shift = 0.0125**（实算：删掉单个 assay
就能让 25-assay mean 移动 0.01246）。未配对时的先验估计 3σ = +0.007 (M=20) / +0.013 (M=5)。
**v1 的 +0.005 作废**（= 2.21σ / 1.30σ）。

**(C) 的现实目标区间**锚定于 BindingGYM 自己**有标签** adaptation 的上界：**0.40 → 0.42**
（paper Table 5 / §4.3；overview §7.1.5 实测 0.422，且 **8/25 个 assay 变差**）⇒ **[+0.005, +0.02]**。
按 §4.1 外推的期望增益 +0.0018 比 (C) 的 MDE **小 16 倍** ⇒ **均值层面的正、负结论都不可判定**。

### 统计检验（开跑前锁定，不得跑完再选）

| | 内容 |
|---|---|
| primary | 逐 assay **配对 Wilcoxon signed-rank**，**n=24**（剔除 `KRAS_SOS1_norfitness_8BE4`），α=0.05 双侧 |
| secondary | **14-cluster bootstrap**（重采样整簇，20,000 次）的 Δ̄ 95% CI；KRAS 6 个先内部平均。cluster 划分引 `training/cache/BindingGYM_cluster.tsv`，**但必须手动把 BH3_Mcl-1_3KZ0 与 BH3_Bcl-xL_1PQ1 合为一簇**（二者共用同一 518-variant 库、label ρ=0.592，官方文件却分成两个单簇） |
| 辅助 | win/loss/tie（\|Δρ\|>0.05）+ leave-one-assay-out 均值敏感性 |
| specificity | **逐对报，绝不做 cluster-level mean**（PSD95×2 / 5A12×2 / GB1×2 各被官方并成一簇） |

> **n 的使用规则（D6）**：S0 用 **n=25** 与 refs/ 比对；S1b 及所有 Δ 报告用 **n=24**。两个 n 在文中不得混用。

---

## 3. Blocking prior：字面版已被实测否证 —— **但适用范围比 v1 声称的窄**

> ⚠️ **§3.1–3.3 的全部数字跑在 StaB-ddG 的 soluble 权重上**，与官方 `v_48_020` **零个 tensor 相同**
> （cosine 0.0090）。这些数字的**绝对值与 BindingGYM 榜单不可比**，且"vanilla 权重上是否同样崩坏"**未测**。
> 措辞上不要写成"soluble 更差" —— 单 assay 上 0.6822 vs 0.6701 的差落在 0.019 的 run-to-run σ 内。
> ⇒ **S1a′ 必须包含一格"在 vanilla v_48_020 上重跑 BH3_Mcl-1 的 lr 扫描"作为桥接**，否则 §3 无法被引用。

### 3.1 lr ≥ 1e-2 的崩坏是真的（assay = `BH3_Mcl-1_normed_3KZ0`，n=518，L=173，其中 **166** 个位点进 mask）

| lr | NLL/res step0 → step30 | Spearman step0 → step30 |
|---|---|---|
| 0.001 | 1.8218 → 1.7686 | +0.7054 → +0.6935 |
| 0.01 | 1.8854 → **1.4652** | +0.6879 → **+0.6476** （Δ=0.0403 ≈ 3σ vs M=5 地板、≈2.2σ vs 官方地板） |
| 0.1 | 1.8193 → **0.3867** | +0.6921 → **+0.3884** |
| 0.3 | 1.8444 → 0.8323 | +0.6868 → **+0.4783** ← **注意：高于 lr=0.1** |

**⚠️ "单调"从未被测量**：每个 lr 只有 **step0 与 step30 两个端点**，中间轨迹（1,2,3,5,10,20,30）没有日志。
两点连线无法区分"单调下降"与"先涨后跌"。而 **lr 方向上本身就不单调**（见 lr=0.3 那行）。

### 3.2 masked-denoise 目标同样退化 / StaB 分解放大损伤

masked-denoise：lr=0.01 → +0.6889→+0.6536（2 次打分取均值）；lr=0.1 → +0.6827→+0.4192。
StaB：frozen 0.7001/0.6600 → lr1e-2 0.6517/0.5721 → lr1e-1 0.3766/0.1542。
⚠️ StaB 那组是**单次打分、无误差棒**，"放大 1.6×"这个比值不应被当作精确量。

### 3.3 🔴 机制改述（v1 的 §2.4 论证是错的）

**v1 说**：`score = log p(s') − log p(s_WT)`，TTT 在最大化分母 ⇒ 自我抵消 ⇒ "机制普适"。

**实际**：
1. 官方打分器**没有 WT forward pass** —— `_scores` 返回 `torch.sum(loss * mask, dim=-1)`
   （`/ torch.sum(mask,-1)` **被注释掉**），`global_score = -1 * mean_over_M(...)`。
2. 即便有 WT 项，它在固定 θ 下对该 assay **所有 variant 是同一个常数**，减常数**保序**
   ⇒ Spearman(score) ≡ Spearman(log p(s'|X_AB))。

⇒ **"TTT 在优化分母"在数学上完全不能解释实测的排序退化。**

**真实机制：熵坍缩 / 相对校准破坏。** 最大化 log p(WT|X) 把 p_θ 推向 δ_WT，所有非 WT 序列的
likelihood 一起塌向下界、丢失细粒度区分度。**这个机制预测"损伤幅度 ∝ 模型当前携带的排序信息量"，
因而不预测 benchmark 级普适失败。**

> 连带修正：v1 §4 后果 1 说的"log-odds ratio 的位点独立假设失效"**不适用于这个打分器** ——
> 它是单次 teacher-forced 自回归 pass，本身不假设位点独立。

---

## 5. BindingGYM 的 variant 结构与 ProteinGym 根本不同（全部数字已复核）

| | ProteinGym | **BindingGYM** |
|---|---|---|
| assay | 217 | 25 |
| variant | 2,465,767 | **376,446**（shipped 行数；论文摘要的"half a million"是 28-assay 的 508,962，**拼接了两个口径**） |
| 单点突变 | 148/217 assay 纯单点 | **9.3%** |
| ≥3 点 | — | **35.1%**（最深 **21**） |
| 多链同改 | 不存在 | **21.3%**；**7/25 个 assay 有、6/25 过半** |

| assay | n | singles | depth≥3 | depth_max | 多链同改 |
|---|---|---|---|---|---|
| Z-domain_ZSPA-1_LL1_1LP1 | 45,476 | 3 | 45,436 | 9 | **45,285** |
| 5A12_VEGF_fitness_4ZFF | 29,981 | 54 | 29,751 | 9 | **24,452** |
| 4D5_HER2_fitness_1N8Z | 2,080 | **0** | 2,079 | 9 | **2,076** |
| hYAP65_peptide_1JMQ | 18,407 | 288 | 11,091 | **21** | 0 |
| GB1_IgG-Fc_1FCC | 92,891 | 1,045 | 0 | 2 | 0 |

---

## 6. Benchmark 缺陷（协议里必须处理）

1. **重复 assay**（已复现，最干净的一条）：`KRAS_DARPinK27_5O2S`（19,533 行）与 `KRAS_SOS1_8BE4`（19,425 行）
   裸 `mutant` join = **0 行**（chain id 是 AB vs RS），抹掉 chain id 后 **19,227 个共享 key、
   max\|Δlabel\| = 0.0、逐值相同比例 1.000、ρ = 1.0**。⇒ 所有 Δ 报告剔除后者，n=24。
   *（此项为纯 label join，不需要 GPU 时间。）*
2. **真实统计单元 = 14 簇**（精确值，来自官方 `BindingGYM_cluster.tsv`：28 行 / 15 representative，
   剔除 3 个流感 assay 独占的那簇）。KRAS 占 6 个 assay / 30.4% variant。**必须同时报 cluster-level**。
3. **43% 的 variant 不碰界面**（`frac_variants_no_iface_mut` 均值 0.4295）。⚠️ 但 interface 标签定义与
   可执行性在 9–10/25 个 assay 上有问题（审计 M8），分层前需先修定义。
4. **`refs/` 是单次未设种 M=5 抽样** ⇒ 逐 assay 逐位复现**在原理上不可能**。

---

## 7. 关于 baseline 本身：ProteinMPNN **不是** partner-blind

论文 §4.1 原话：*"Unlike previous studies where predicted monomer structures from AlphaFold were used,
we input **full protein complex structures** into our structure-based methods."*

- harness 把 `chain_id` 里**每条链**都标为 designed 并纳入计分 mask（25/25 assay：`fixed_chain_list` 为空）
- 官方消融：**ProteinMPNN 0.397 vs ProteinMPNN_single 0.356（+0.041）**
  ⚠️ 但**仓库里没有任何脚本能产生 `_single`** ⇒ 这个 +0.041 **不可复现**，引用必须注明
- 对照组：**ESM2 0.2851 vs ESM2_all_seq 0.2852** —— 序列模型拼上 partner 序列**增益为零**（未进 paper）
- 全 benchmark **只有 5/13 个 baseline 真正吃到 partner**（ProteinMPNN / PiFold / ByProt / ESM-IF1 / PPIformer）；
  **SaProt 走的是 ESM 那条 `--focus 1` 路径，只看被突变的链**

---

## 8. 前置测量 F1–F4（在 S0 之后、S1a′ 之前；全部 < 3 GPU-h）

| id | 内容 | 规模 | 交付 / 停止条件 |
|---|---|---|---|
| **F1** | frozen-vs-frozen **同 seed** 零效应对照 | 2 assay × 1 seed | **Δ 必须恒等于 0**；不为 0 ⇒ seed 未真正配对，**停** |
| **F2** | 逐 assay 打分噪声 σ | 8 assay × ≥5 seeds，M=5 与 M=20 各一组 | σ 向量（须含 BH3_Mcl-1 n=518、GB1 n=92,891、KRAS_PICK3CG-RBD L=1107、4D5_HER2 多链、hYAP65 depth21） |
| **F3** | **TTT 侧** run-to-run 方差 | 2 assay、固定 (lr,steps)、≥5 **TTT seed** × 同一打分 seed | sd_TTT ⇒ 阈值 = 3×√(sd²_TTT + sd²_score,paired)/√n |
| **F4** | specificity 噪声地板 | 3 对、≥8 replicates、两 assay 固定同一解码顺序 | 地板值；**未测出前不得宣称任何非零 specificity** |

---

## 9. S0 — harness validation gate

### S0a：子集 gate（先跑；实测 **1.95 h @M=20**，~1.0 h @官方 M=5）

按 Σn·L 最便宜的 10 个：`CXCR4_8U4O`、`5A12_Ang2_4ZFG`、`hYAP65_1JMQ`、`Z-ZSPA-1_LL2`、
`Z-ZpA963_HL1`、`PSD95_Tm2F`、`PSD95_CRIPT`、`BH3_Bcl-xL`、`BH3_Mcl-1`、`Z-ZpA963_HL2`
**+ 强制加 2 个覆盖用**：`4D5_HER2_1N8Z`（L=1041、3 链、验 M=20 显存路径）与
`KRAS_RAF1_6VJJ`（覆盖 KRAS 簇）。
⛔ 不要把 `1HE8` / `4ZFF` / `5O2S` 放进 gate —— 那是 10.7 h。

**判据（每个 assay 各自）**：我们跑 **≥5 个显式 seed** 得 [min,max] 与 mean±σ_ours；
**PASS** 当 refs/ 参考值落入 [min,max]，或 `|mean_ours − ref| ≤ 2σ_combined`，
`σ_combined = √(σ_ours² + σ_ref²)`（BH3_Mcl-1 上已实测 σ_M5 = 0.019–0.021、σ_M20 = 0.0093
⇒ σ_combined ≈ 0.0225 ⇒ 容差 ≈ **0.045**）。

**子集均值判据**：与写死的参考子集均值比（10-smallest-by-Σn·L = 0.356467；加 2 个覆盖 assay 后须重算并写死）。

**强制交付物**：(i) 逐 assay σ 向量；(ii) **frozen 全复合物 ProteinMPNN 自己的 partner-conditional
specificity**（三对，逐对报 n 与 CI）；(iii) 全部 assert 的通过记录；(iv) 一个长 assay 的端到端 wall time。

### S0b：全量 25 assay（可与 S1a′ 并行排队）

成本 **≈14–16 h @官方 M=5 / ≈28.3 h @M=20**（实测外推，Σn·L = 135,744,347；纯线性下界 19.5 h）。
判据：25-assay mean 落在 **0.396950 ± 0.009**（= 2 × combined SE 0.0045）。比对必须用**全部 25 个**。

---

## 10. S1a′ — 分层小样本（**7 assay**，Σn·L ≈ 1.11 M ≈ v1 的 1.9×）

**保留**（refutation 锚点 + 高/低 baseline 两端）：
`BH3_Mcl-1_3KZ0`(518, L=173, base 0.6625) · `BH3_Bcl-xL_1PQ1`(518, L=229, 0.6554) ·
`PSD95_CRIPT_1BE9`(1577, L=120, 0.3863) · `PSD95_Tm2F_1BE9`(1577, L=120, 0.2073)
> ⚠️ PSD95 那一对**不是 partner swap** —— partner 只差 1 个残基（KQTSV vs KQFSV），且两者同簇。

**新增**（覆盖 §5 自认的最大风险面）：
`Z-ZpA963_HL2_2M5A`(600, L=116；**55.2% 多链**；**非 `_hm` 实验坐标**；全 benchmark 最便宜) ·
`Z-ZpA963_HL1_2M5A`(2904, L=116；**92.9% 多链**；最低 baseline 层；paper §4.4 点名家族) ·
`hYAP65_1JMQ` 随机子采样 2,000（**depth_max=21**）
预算够再加 `KRAS_RAF1_6VJJ` 子采样 2,000（覆盖 6/25 assay、30.4% variant 的 KRAS 簇）。

**覆盖声明（必须写进正文）**：这 7 个覆盖 **5 个 cluster / 4 个家族**；多链同改 3 个、depth≥3 有 2 个、
非同源模型结构 2 个；baseline 跨 **0.136–0.663**。**仍不可外推到 25 assay。**

**配置**：
- `lr ∈ {1e-4, 3e-4, 1e-3, 3e-3}` ← ≥1e-2 已被 prior 证明必崩，不再花预算
- `steps ∈ {0,1,2,3,5,10,20,30}` ← **step-wise 打分**（prior 只有两个端点，单调性从未被测量）
- 每个 step ≥3 个打分 replicate，M=20，打分 seed 与 frozen 臂**共用同一 randn_1**
- TTT seed ≥3（与打分 seed **分开记账**，见 F3）
- 附加 arm：**vanilla v_48_020 上重跑 BH3_Mcl-1 的 lr 扫描**（§3 的桥接，不可省）
- 低优先级 arm：15% mask + 全位点 loss（对应 fitness 真正用的 `unnormalized_cross_entropy`）

**判定**（不再用 72 次 1σ 比较 —— 那期望产生 17–24 个假窗口，P(至少一个)≈1.00）：
- **主判据**：每个 lr 的每个 step，取**跨 7 assay 的配对均值 Δ̄₇**，与 3 × 实测 sd(Δ̄₇)（来自 F2/F3）比较，
  并对 4 个 lr 做 **Holm 校正** ⇒ 32 次比较压成 4 条曲线
- **形态判据**（对独立噪声鲁棒）：同一 lr 在 **7/7** 个 assay 上 Δ>0 **且** step-wise 单调上升
- **双侧读数**：既判"有没有超过 3σ 的窗口"，也判"小 lr 是不是只是零结果"

**成本**：一趟打分 ≈9.6 min @M=20；全网格 **≈4.7 GPU-h**。

---

## 11. S2 — 唯一有机会的一段：换掉目标函数的最优解

S1 的所有目标（WT-NLL、masked-denoise）最优解都是 δ_WT。替换为
**backbone-conditioned homolog denoising**：

$$\mathcal{L}(\theta) = \mathbb{E}_{s' \sim q}\ \mathbb{E}_{M \subset P}\Big[-\tfrac{1}{|M|}\sum_{i \in M} \log p_\theta(s'_i \mid s'_{\setminus M},\, X_{AB})\Big]$$

**最优解是 `p_θ = q ≠ δ_WT`** —— 与 S1 的唯一本质区别。
§4.2 的 +0.0160 vs +0.0014（11×）是这个方向在**单体 fitness 上的实证**；
§4.2 末的分类学结论（该机制的最优解是**家族条件分布**、耦合信息存活）是它的机制解释。

- 停步信号：**held-out interface confidence** —— `c(θ) = (1/|H|) Σ_{i∈H} log p_θ(x_i | x_\H, X_AB)`，
  `H` 永不进 masking 集合。（ProteinTTT 在 fitness 任务上**不启用** confidence function，这是本 plan 新增。）
- MSA 可得性**已确认**：22/22 POI 覆盖，本机 633 MB。
  ⚠️ 但**全复合物覆盖仅 2M5A 与 1LP1** ⇒ 待定问题改为"q 能否只定义在被突变链上"（D4）。

---

## 12. Partner-conditional specificity — **指标与前提都要重做**

### 12.1 v1 的两处错误

1. **"partner-blind 恒等于 0.000"为假**（B12）。三对全部用不同结构文件（md5 已核）：

| 对 | 结构 | chain_id | partner-blind 是否为 0 |
|---|---|---|---|
| PSD95 CRIPT / Tm2F | `1BE9_hm.pdb` / `1BE9Tm2F_hm.pdb` | AB / AB | **≈0.000 但非恒等式**（chain A 的 N/CA/C 逐位相同，O RMSD 0.032 Å）；且**只对 `--focus 1` 序列模型成立** |
| BH3 Mcl-1 / Bcl-xL | `3KZ0_hm` / `1PQ1_hm` | **AC / AB** | **任何模型都不为 0**（被打分链 23 aa chain C vs 33 aa chain B） |
| 5A12 VEGF / Ang2 | `4ZFF_CHL` / `4ZFG` | **CHL / AHL** | **任何模型都不为 0**（H 链 4 个 token 差异，且是**未解析残基占位 X**，非生物学差异） |

**更根本**：我们要打的 baseline 是**全复合物 ProteinMPNN，它本来就不是 partner-blind**，
其 published per-assay Spearman 已经不相等（PSD95 0.3863 vs 0.2073；5A12 0.4775 vs 0.1171）
⇒ **这个指标不能把 complexTTT 与它自己的 baseline 分开**。

2. **`1 − ρ` 方向错误**（B13）：无信息打分器给 ρ≈0 ⇒ specificity ≈ 1.0，**超过** label 隐含的 oracle
   （PSD95 0.565、BH3 0.408），且无界（5A12 oracle = 1.145）。v1 §2.1 的 lr=0.1 崩塌恰恰会**推高**它。

### 12.2 修正后的定义

用 **oracle 锚定的有号量** `|ρ_model_cross − ρ_label_cross|`，或 **matched/mismatched 配对检验**：
把 assay-1 的 variant 集分别在 partner-1 与 partner-2 复合物上打分，要求 matched 那一侧对 assay-1
自己的 label 给出更高 Spearman。写明取值范围与零分布，并与 **F4 的地板** 及 **S0a 交付的
frozen ProteinMPNN 自身 specificity** 比较。

### 12.3 Variant Alignment（v1 完全缺失 —— 会静默产生空 join）

实测原始 `mutant` 字符串重叠：**PSD95 1577/1577**（可直接 join）· **BH3 = 0** · **5A12 = 0** · KRAS-RAF1 对 12,086。

五步写死：
1. key **只取被突变链**（这一步单独就修好 5A12 → 534）
2. 逐对的 chain 对应表
3. 由被突变链序列比对求位点偏移并设 identity 闸门（**BH3 需 C→B 且 +2**：`I6A:L10A:R11D:I13A` ≡ `I8A:L12A:R13D:I15A`）
4. 只在对齐后的交集上计算
5. **报交集大小及其占每个 assay 的比例**

参考实现：`Sources/complex-ttt-evidence/scripts/crosspartner.py`（`keyset()` / `align_offset()` / `canon()`）。
**harness 断言**：BH3 对对齐后 join 必须返回 **518 行**。

---

## 13. 预算（实测外推，非平铺估计）

| 项 | 工作量 | 估时（独占 A4500） |
|---|---|---|
| S0a 子集 gate（12 assay，M=20，5 seeds） | ~4.0 M residue-variant·M | **~2.0 h**（M=5 时 ~1.0 h） |
| **F1–F4 前置测量** | — | **< 3 h** |
| S1a′ 一趟打分（7 assay） | ~1.11 M | ~9.6 min |
| **S1a′ 全网格** | 4 lr × 8 step × 7 assay × 3 replicate + TTT | **~4.7 h** |
| S0b 全量 25 assay，一趟 | Σn·L = 135,744,347 | **≈14–16 h @M=5 / ≈28.3 h @M=20** |
| S1b 全量 × 5 TTT seed（仅在 S1a′ 出窗口） | — | **~142 h @M=20 / ~75 h @M=5**（frozen 臂逐 seed 重打分则 ×2） |

**承诺范围：S0a + F1–F4 + S1a′ ≈ 10 GPU-h。** S0b 单独排一次独占长跑。

**成本大头（实测）**：`KRAS_PICK3CG-RBD_1HE8` **6.45 h (22.8%)** · `SARS2-RBD_ACE2_6M0J` 4.31 h ·
`GB1_IgG-Fc_1FCC` 3.85 h · `5A12_VEGF_4ZFF` 3.23 h · `KRAS_SOS1_8BE4` 2.80 h。
**不是 v1 写的 GB1 + Z-domain LL1**（后者只占 0.65 h = 2.3%）。1HE8 的 6.45 h 同时是全量 S0 的
**Amdahl 下界**（分片粒度是一个 assay 一个进程，无 assay 内 checkpoint）。

吞吐锚点更正：**205 sequence-scoring/s @L=173**（官方脚本实测）；同一脚本在 **L=1107 只有 16.5/s**
⇒ 跨 benchmark 12× 差异，**该速率不是常量**。

---

## 14. 需要拍板的决策

| | 决策 | 建议 |
|---|---|---|
| **D1** | S0 先跑子集还是全量？ | **先 12 个（1.95 h）**，按 Σn·L 选 + 强制加 `4D5_HER2` 与 `KRAS_RAF1`。全量改记为 14–28 h |
| **D2** | S1a′ 后是否直接进 S2？ | **撤销 v1「直接进 S2」的理由链**（机制论证对 Spearman 为空、单调性从未测量、§3 跑在另一个模型上）。改为：**S1a′ 的 step-wise 曲线是唯一能测单调性的实验，必须先跑完**；S1b 的取舍在其后、并以 142/75 GPU-h 的真实成本重新权衡 |
| **D3** | specificity 作并列 headline？ | **在按 §12.2 重定义、且 S0a 交出 frozen ProteinMPNN 自身 specificity 之前，不能** |
| **D4** | S2 的 MSA | 覆盖率已答（22/22 POI）。改问：**q 能否只定义在被突变链上**（全复合物 MSA 覆盖仅 2/22） |
| **D5** | 报 benchmark 缺陷？ | 与论文一起报。**加第二项**：摘要把 28-assay 的 508,962 点数与 25-assay 的 assay 数拼接（shipped 只有 376,446 行） |
| **D6** | n 的口径 | **S0 用 n=25；S1b 及所有 Δ 报告用 n=24**（剔 `KRAS_SOS1_8BE4`）。两者不得混用 |

---

## 15. 总体判断（v2 修订）

**我对"字面版打不动 ProteinMPNN"的把握，比 v1 声称的低。** 原因：

- v1 的三条"独立同向证据"里，**第一条（机制）被否证**（§3.3：对 Spearman 数学上为空），
  **第三条（85% 抄写梯度）不能直接移植到 ProteinMPNN**（§4.3）。真正站得住的是：
  修正后的熵坍缩机制（但它**只预测"损伤 ∝ baseline 携带的排序信息量"，不预测普适失败**）、
  lr ≥ 1e-2 的实测崩坏（但在**另一个模型权重**上）、以及 §4.1 的规模趋势。
- 而 §2 (C) 的 MDE 分析说明：**期望增益比可检测下限小 16 倍 ⇒ 均值层面的正、负结论都不可判定。**

**⇒ 本 plan 的正确定位不是"证明它失败"，而是"把不可判定性本身做实，并把基础设施建好"。**
三个理由让这仍然值得做：

1. **S0 的 harness 是所有后续工作的公共基础设施**，与目标函数无关。ProteinGym 那轮的 gate
   抓出了三个真 bug（MSA 子区段截断、weight-tying、ProSST 改名）。
2. **S1a′ 的 step-wise 曲线是 plan 里唯一能测"单调性"的实验** —— 而单调性是 v1 用来跳过 S1b 的
   核心依据，它从未被测量过。
3. **S2 的方向有独立实证支持**（§4.2 的 11×、以及"最优解是家族条件分布"的机制），
   且它与 S1 只差一个目标函数。

**S1 是必要的负对照，不是希望所在** —— 这个定位与全部六个审计维度的证据一致，
且被 BindingGYM 自己 Table 5 的**有标签** adaptation 上界（0.40 → 0.42，且 8/25 assay 变差）进一步支持。

---

## 16. Change log

- 2026-08-26 09:57 — v1 建 project、写 plan（未启动任何 GPU 任务）。
- 2026-08-26 16:00 — **v2**：经一轮对论文 + 代码仓库 + `BindingGYM_overview.md` 的系统审计
  （6 维度 × 审计 + 对抗复核，13 个 agent），发现 **17 blocking / 30 major**，全文重写。
  其中 B1/B2/B4/B5/B10/B12 六条我自己逐条复核过（md5 对比、`if args.seed:` falsy、`BATCH_COPIES`
  死参数、`randn_1_dic` 按 POI 缓存、`_scores` 无 WT 项且长度归一化被注释、三对结构文件 md5 不同）。
  完整修订指令存档 `refs/audit_revision_20260826.md`。
- 2026-08-26 16:00 — **数据持久化**：BindingGYM input 原本只存在于 session scratchpad（327 MB），
  已 rsync 到 `/home/guoj0f/share/BindingGYM/input`（25/25 CSV、22/22 结构、md5 已核），
  并把 `BindingGYM/input` symlink 过去。**仍未启动任何 GPU 任务。**
