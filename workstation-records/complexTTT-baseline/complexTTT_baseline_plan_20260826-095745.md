# complexTTT-baseline — experiment plan  (created 2026-08-26 09:57; status: PLANNED, 未启动任何 GPU 任务)

把 ProteinTTT 的 test-time training 协议搬到 **complex binding** 上：对 pretrained **ProteinMPNN**
在 BindingGYM 的 **WT 数据（complex structure + 全部链的序列）** 上做自监督微调，
零样本设定，目标是打 BindingGYM zero-shot 榜。

---

## 1. Goal 与成功判据

**要打的对象**（BindingGYM 已发表 zero-shot，25 assay 的 mean signed Spearman）：

| 模型 | mean ρ |
|---|---|
| **ProteinMPNN** | **+0.3970** ← 现任第一，也是我们的 baseline |
| MPNN-1chain | +0.3564 |
| TranceptEVE | +0.3432 |
| PiFold | +0.3380 |
| ESM-IF1 | +0.3378 |
| ESM2 | +0.2851 |

**成功判据（必须在开跑前锁定，否则事后一定会被"看起来涨了"骗）：**

> **ProteinMPNN + complexTTT 的 25-assay mean signed Spearman，
> 减去同一 harness 上 frozen ProteinMPNN 的值，其差必须超过 MC 解码噪声地板。**

噪声地板是实测过的（BH3_Mcl-1，8 replicates）：

| 解码采样数 M | frozen Spearman | 标准差 |
|---|---|---|
| M=1 | +0.6667 | ±0.0366 |
| M=5 | +0.6817 | ±0.0136 |
| **M=20** | **+0.6919** | **±0.0080** |

⇒ **M=20 下单 assay 的 1σ 是 0.008。** 25 个 assay 平均后约 0.008/√25 ≈ **0.0016**，
所以 mean 上的判定阈值取 **> +0.005**（约 3σ）才算真效应。

---

## 2. 🔴 必须先说：这个提案的**字面版已经被实测否证了**

2026-08-19 我在本机 A4500 上用真实 BindingGYM 数据 + `StaB-ddG/model_ckpts/proteinmpnn.pt`
跑过这件事（assay = `BH3_Mcl-1_normed_3KZ0`，L=173，518 variants，M=20，8 replicates）。

### 2.1 WT-likelihood 目标：单调下降，没有操作窗口

| lr | NLL/res step0 → step30 | Spearman step0 → step30 |
|---|---|---|
| 0.001 | 1.8218 → 1.7686 | +0.7054 → **+0.6935 ± 0.0147** |
| 0.01 | 1.8854 → **1.4652** | +0.6879 → **+0.6476 ± 0.0106** ← −4σ |
| 0.1 | 1.8193 → **0.3867** | +0.6921 → **+0.3884 ± 0.0031** |
| 0.3 | 1.8444 → 0.8323 | +0.6868 → +0.4783 ± 0.0152 |

**目标函数确实在下降**（NLL 1.82 → 0.39），**但 Spearman 同步崩掉**。
学得越好，打分越差 —— 而且**单调**，没有"先涨后跌"的窗口可挑。

### 2.2 换成 ProteinTTT 自己的 masked-denoise 目标，一样

| 配置 | Spearman |
|---|---|
| masked-denoise lr=0.01 | +0.6889 → **+0.6536** |
| masked-denoise lr=0.1 | +0.6827 → **+0.4192** |

### 2.3 StaB 热力学分解**放大**损伤 1.6×

| 配置 | complex-only ρ | StaB ρ |
|---|---|---|
| frozen | +0.7001 | +0.6600 |
| WT-NLL TTT lr=0.01 | +0.6517 | **+0.5721** |
| WT-NLL TTT lr=0.1 | +0.3766 | **+0.1542** |

### 2.4 机制：目标函数的最优解 = 打分公式的分母

$$\text{score}(s') = \log p_\theta(s' \mid X_{AB}) - \underbrace{\log p_\theta(s_{\text{WT}} \mid X_{AB})}_{\text{TTT 正在最大化的量}}$$

任何以 **δ_WT 为最优解**的自监督目标，都是在对打分公式自己的分母做梯度上升。
ProteinMPNN 是 inverse-folding 模型，`p(sequence | backbone)`；在 WT complex 上做 TTT
就是教它 `p(WT | WT-backbone) → 1`。**这不是超参问题，是目标函数的方向问题。**

> ⚠️ **这轮否证的证据强度边界**：只测了 **1 个 assay**。机制论证是普适的，
> 但"25 assay 上也如此"这个断言**没有实测支撑**。这正是本 plan 的 S1 要补的。

---

## 3. 今天（2026-08-25）的 ProteinGym 复现给出了三条新证据，都指向同一方向

### 3.1 TTT 增益随基座变强而衰减 —— 而 ProteinMPNN 是榜首

| 模型 | ProteinGym rank | TTT 增益（我们实测） |
|---|---|---|
| ESM2 (35M) | 84 / 97 | **+0.0189** |
| ESM2 (650M) | 45 / 97 | +0.0010 |
| ProSST (K=2048) | **3 / 97** | **+0.0018** |

ProteinMPNN 在 BindingGYM 上是 **rank 1**。按这条趋势，期望增益落在**区间最底部**。

### 3.2 实际目标函数比想象的更"抄 WT"

fitness 任务用的是 `unnormalized_cross_entropy`，我读代码 + 实测确认它
**完全忽略 mask 参数**，在**全部 token**（含未 mask 的 85% 和 BOS/EOS）上算 loss：

| | 参与 loss 的位置 | 占比 | loss |
|---|---|---|---|
| `cross_entropy`（只算 mask 位） | 168 | 14.6% | 1.8013 |
| **`unnormalized_cross_entropy`** | **1152** | **100%** | **0.5068** |

拆开看：被 mask 位置的平均 loss 1.8013，**未被 mask 的残基位置只有 0.2864（输入已给答案，等于抄写）**。
⇒ **约 85% 的梯度来自"把已知位置原样抄出来"** —— §2.4 的分母问题比字面理解的更严重。

### 3.3 即使在最弱的基座上，也只有 77% 的 assay 受益

ESM2-35M 上逐 assay 增益 **167/217 为正**，最差 −0.1926。
最差的三个都是 **37–44 残基的短 stability assay** —— 正是"抄 WT"信号最强的情形。

---

## 4. 被低估的一点：BindingGYM 的 variant 结构和 ProteinGym 根本不同

这条我认为是**整个提案最大的技术风险**，且和 §2 的否证独立。

| | ProteinGym | **BindingGYM** |
|---|---|---|
| assay 数 | 217 | 25 |
| variant 总数 | 2,465,767 | 376,446 |
| **单点突变占比** | 148/217 个 assay **纯单点** | **9.3%** |
| **≥3 点突变占比** | — | **35.1%**（最深 21 点） |
| **同时改多条链的 variant** | 不存在 | **21.3%**；**6/25 个 assay 过半** |

逐 assay 看（部分）：

| assay | n | singles | depth≥3 | depth_max | 多链同改 |
|---|---|---|---|---|---|
| Z-domain_ZSPA-1_LL1_1LP1 | 45,476 | 3 | 45,436 | 9 | **45,285** |
| 5A12_VEGF_fitness_4ZFF | 29,981 | 54 | 29,751 | 9 | **24,452** |
| 4D5_HER2_fitness_1N8Z | 2,080 | **0** | 2,079 | 9 | **2,076** |
| hYAP65_peptide_1JMQ | 18,407 | 288 | 11,091 | **21** | 0 |
| GB1_IgG-Fc_1FCC | 92,891 | 1,045 | 0 | 2 | 0 |

**后果：**
1. log-odds ratio 的**位点独立假设**在这里大面积失效（深度 3–21）。
   ProteinMPNN 是自回归解码，本身不假设独立 —— 这是它领先的原因之一，但也意味着
   **分数对解码顺序敏感**，必须 M 次采样平均（噪声地板的来源）。
2. "one protein is all you need" 的**单一 WT 参考**在多链同改时更弱：
   variant 和 WT 的编辑距离中位数远大于 1，TTT 把模型往 WT 拉得越紧，
   对远处 variant 的排序能力越差。
3. **6/25 个 assay 里 partner 链也在变** —— 这类 assay 才是真正"complex-specific"的，
   也是任何 partner-blind 方法结构上做不到的部分。

---

## 5. 另外两条已知的 benchmark 缺陷（会影响读数，必须在协议里处理）

1. **重复 assay**：`KRAS_DARPinK27_norfitness_5O2S`（19,533）与
   `KRAS_SOS1_norfitness_8BE4`（19,425）抹掉 chain id 后 join，
   **19,227 个共享 variant 的 label 逐值完全相同**（max\|Δ\|=0.0，ρ=1.0）。
   裸 key join 会得到 0 行重叠（chain id 是 AB vs RS），必须先做 chain 对齐才看得见。
2. **assay 高度相关**：KRAS 家族占 6 个 assay、Z-domain 占 4 个、GB1 占 2 个、PSD95 占 2 个……
   真实统计单元约 **14 个簇而非 25 个 assay**（n_eff ≈ 2.02）。
   ⇒ 报 mean 时**必须同时报 cluster-level mean**，否则 KRAS 一族会主导结论。
3. **43% 的 variant 完全不碰界面**（`frac_variants_no_iface_mut` 均值 0.4295）。
   ⇒ 分层报告 **interface-touching vs non-interface** 两个子集，否则"binding-specific"是空话。

---

## 6. 一个正面信号：ProteinMPNN 确实有超越几何 baseline 的信号

零参数结构特征（25 assay mean signed ρ，实测）：

| 特征 | mean ρ |
|---|---|
| `rho_burial_on_iface`（界面残基的埋藏度） | +0.2816 |
| `rho_dmin`（跨链最小重原子距离） | +0.2598 |
| `rho_burial_complex`（CB 10Å 邻居计数） | +0.2444 |
| `rho_isiface`（是否界面残基） | +0.2295 |
| **ESM2（已发表）** | **+0.2851** ← 与零参数特征持平 |
| **ProteinMPNN（已发表）** | **+0.3970** ← **明显高出，有真实信号** |

**这修正了我之前一个过于笼统的说法**：零参数几何量打平的是 **PLM（ESM2 0.2851）**，
**不是 ProteinMPNN**。MPNN 高出 0.11，说明它学到的东西超越"这个位置埋得深不深"。
⇒ **好消息是 baseline 有真本事；坏消息是它因此更难被超过。**

---

## 7. 设计：三段式，每段都是可独立止损的 gate

### S0 — harness validation gate（**不做 TTT，必须先过**）

用我们自己的打分器复算 25 个 assay 的 frozen ProteinMPNN 分数，与
`refs/ProteinMPNN_zero_shot_metric.csv`（BindingGYM 官方发布的 per-assay 6 指标）**逐 assay 比对**。

- **通过条件**：per-assay \|Δ Spearman\| 落在 M=20 噪声地板内（≤ 0.008，即 1σ），
  且 25-assay mean 复现 **+0.3970 ± 0.002**。
- **S0 不过就不跑 S1。** 今天的 ProteinGym 复现里这条纪律救了三次
  （MSA 子区段截断、weight-tying bug、ProSST 改名 assay）—— 没有 gate，
  TTT 的 Δ 与 harness bug 无法区分。
- 成本：一趟 frozen 打分（见 §8）。

### S1 — 字面版 complexTTT baseline（**用户要的那个**）

**分两层，先小后大：**

**S1a — 决定性小样本（4 assay，step-wise 曲线）**
- assay：`PSD95_CRIPT_1BE9` / `PSD95_Tm2F_1BE9`（同一 PDB 1BE9、同一 1,576-variant 库、不同 partner）
  + `BH3_Mcl-1_normed_3KZ0` / `BH3_Bcl-xL_normed_1PQ1`（同一库、不同 partner）
- 为什么选这四个：**两对 partner-swap** —— 既能测 TTT 效果，又能测
  §9 的 partner-conditional specificity，而且都是小 assay（518–1,577 variants），便宜。
- 配置：`lr ∈ {1e-4, 1e-3, 1e-2}` × `steps ∈ {0,1,2,5,10,20,30}`（同一条 run 内逐步打分）
- **判定**：若三个 lr 的曲线**都单调下降**（复现 §2.1），S1b 不跑，直接进 S2。
  若任一 lr 出现 **> +0.008（1σ）的窗口**，才展开 S1b。

**S1b — 全量 25 assay**（仅在 S1a 出现操作窗口时跑）
- 用 S1a 选出的最优 (lr, steps)，跑满 25 assay × 5 seeds
- 报 assay-level mean、**cluster-level mean（14 簇）**、interface / non-interface 分层

### S2 — 一个有机制依据的目标函数改动

**只改一件事：让目标函数的最优解不是 δ_WT。**

S1 的所有目标（WT-NLL、masked-denoise）最优解都是"完美复现 WT"，
所以都在优化打分分母。最省事且自洽的替代是
**backbone-conditioned homolog denoising**：

$$\mathcal{L}(\theta) = \mathbb{E}_{s' \sim q}\ \mathbb{E}_{M \subset P}\Big[-\tfrac{1}{|M|}\sum_{i \in M} \log p_\theta(s'_i \mid s'_{\setminus M},\, X_{AB})\Big]$$

其中 `q` = 同源序列分布（threaded 到固定的 WT backbone 上）。
**最优解是 `p_θ = q ≠ δ_WT`** —— 这是与 S1 的唯一本质区别。

- 停步信号：**held-out interface confidence** —— 留出一部分界面残基
  `H`，`c(θ) = (1/|H|) Σ_{i∈H} log p_θ(x_i | x_\H, X_AB)`，且 `H` 永不进 masking 集合。
  这样有一个**无标签**的选步依据（对应 ProteinTTT 结构任务里 pLDDT 的角色 ——
  fitness 任务本来是没有这个模块的，见 §7 备注）。
- **成本风险**：需要 MSA。25 个 assay 的同源检索是这一段的主要开销，
  **S2 只在 S1 明确失败后才启动，且先只在 S1a 那 4 个 assay 上验证机制**。

> 备注：ProteinTTT 在 fitness 任务上**不启用** confidence function
> （Table A3 的 fitness 行只写 30 步，代码里 `_ttt_eval_step` 返回 `confidence=None`）。
> S2 引入 held-out-interface confidence 是**本 plan 的新增**，不是照搬。

---

## 8. 评测口径与预算

### 打分口径（frozen 与 TTT 两臂必须完全一致）

- ProteinMPNN 自回归 log-likelihood，**M=20 个随机解码顺序取平均**（噪声地板见 §1）
- 复用 BindingGYM 官方 `baselines/protein_mpnn/compute_fitness_multi_pdb.py` 的口径，
  指标用 `refs/` 里官方发布文件的同一套 6 指标（Spearman / AUC / MCC / NDCG / AP / UnbiasHit@10）
- checkpoint：`/home/guoj0f/repos/StaB-ddG/model_ckpts/proteinmpnn.pt`

### 实测吞吐外推

早前实测：BH3（518 variants，L=173）M=20 一趟 **46.5 s** ⇒ 约 **223 sequence-scoring/s**。

| 项 | variant 数 | ×M=20 | 估时 |
|---|---|---|---|
| **S0** frozen 全 25 assay | 376,446 | 7.53 M | **~9.4 h** |
| **S1a** 4 assay 一趟打分 | 4,190 | 83.8 k | ~6 min |
| **S1a** 全网格（3 lr × 7 step × 4 assay + TTT） | — | — | **~2.5 h** |
| **S1b** 全量 25 assay × 5 seeds | — | — | **~50 h**（仅在 S1a 出窗口才跑） |

⇒ **先只承诺 S0 + S1a ≈ 12 GPU-h**，其余按 gate 结果再定。

**降本备选**：S0 的 9.4 h 大头在 GB1（92,891）和 Z-domain LL1（45,476）两个巨型 assay。
若只求"harness 正确性"，可先在 **10 个中小 assay** 上过 gate（~1.5 h），
全量 S0 与 S1b 一起跑。**建议采用这个** —— 见 §10 决策 D1。

---

## 9. Headline 指标的建议：不要只报 mean

§3.1 的趋势说明"mean 提升"很可能拿不到。但有一个读数是
**任何 partner-blind 方法结构上做不到的**，而 complexTTT 天然可以：

**partner-conditional specificity** = `1 − ρ(score_partner1, score_partner2)`

实测的 label 层面 divergence（同一突变库、同一 PDB、不同 partner）：

| assay 对 | label ρ | 说明 |
|---|---|---|
| PSD95 CRIPT vs Tm2F | **0.435** | 同一 PDB 1BE9、同一 1,576-variant 库 |
| BH3 Mcl-1 vs Bcl-xL | 0.592 | 同一 518-variant 库 |
| 5A12 VEGF vs Ang2 | **−0.145** | 方向都反了 |

**partner-blind 打分器在这个指标上恒等于 0.000**（同一条链、同一结构 ⇒ 同一分数）。
⇒ 即使 mean 打不动榜，"能否区分同一蛋白面对不同 partner"是一个**干净、可发表、
且 baseline 结构上做不到**的读数。建议把它作为**并列 headline**，而不是备选。

---

## 10. 需要你拍板的决策

| | 决策 | 我的建议 |
|---|---|---|
| **D1** | S0 gate 先跑全量 25 assay（9.4 h）还是先跑 10 个中小 assay（1.5 h）？ | **先 10 个**。gate 的目的是验 harness，不需要全量；省下 8 h 留给 S1a。 |
| **D2** | S1a 出现单调下降（大概率）后，是否直接进 S2？还是先补 S1b 把"25 assay 上也失败"钉死？ | **直接进 S2**。S1b 在已知失败的配置上花 50 h 只是把否证做厚，机制已经清楚。 |
| **D3** | headline 用 mean Spearman，还是 mean + partner-conditional specificity 并列？ | **并列**。理由见 §9。 |
| **D4** | S2 的 MSA 检索：用 ProteinGym 已有的 MSA（覆盖不全），还是为 25 个 complex 重新跑 MMseqs2？ | 先查覆盖率再定；`proteinttt/utils/msa.py` 已有 MSAServer，可复用。 |
| **D5** | 是否把 §5 的 BindingGYM 重复 assay bug 报给作者？ | 与论文一起报，先在记录里留证。 |

---

## 11. 我的总体判断（说在前面，免得跑完才说）

**字面版 complexTTT 大概率打不动 ProteinMPNN，我给的把握是高的** —— 三条独立证据同向：
机制上目标函数在优化打分分母（§2.4）；实测在 1 个 assay 上单调崩坏（§2.1–2.3）；
ProteinGym 上 TTT 增益随基座变强而衰减，而 ProteinMPNN 是 BindingGYM 榜首（§3.1）。

**但这个 baseline 仍然值得建**，理由有三：
1. 我的否证只有 **1 个 assay**，S1a 用 **4 个 assay + 两对 partner-swap** 把它做实，
   成本仅 ~2.5 h。**一个被严格测量过的负结果，是后续任何方案的立足点。**
2. S0 的 harness 是**所有后续工作的公共基础设施** —— 无论目标函数怎么改，
   都要用它打分。今天 ProteinGym 的经验是：gate 本身就抓出了三个真 bug。
3. §9 的 partner-conditional specificity 是**这条路线真正的独特读数**，
   而它**不依赖 mean 提升成立**。即使 S1 全线失败，这个指标仍然能产出结论。

**S2 才是真有机会的那一段**，它和 S1 的唯一区别是"目标函数的最优解不是 δ_WT"。
建议把 S1 定位成**必要的负对照**，而不是希望所在。

---

## 12. Change log

- 2026-08-26 09:57 — 建 project、写 plan。**尚未启动任何 GPU 任务。**
  已就地化参考文件：`refs/ProteinMPNN_zero_shot_metric.csv`（官方发布的 per-assay 6 指标，
  S0 gate 的 ground truth）、`refs/iface_baselines.csv`、`refs/bindinggym_published_spearman_matrix.csv`。
