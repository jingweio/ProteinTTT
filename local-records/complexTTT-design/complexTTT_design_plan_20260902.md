# complexTTT 设计：理解与计划（草案）

**分支** `BindingGYM-complexTTT-design` · **日期** 2026-09-02 · **状态** 🟠 **草案，未获批准，未写任何代码**

**这份文档是什么**：把「我理解你要做什么」「基底事实告诉我们什么」「有哪几条路可走」「我推荐哪条、为什么」
一次摊开，供你拍板。**姊妹文档** `BindingGYM_zeroshot_protocol_20260902.md` 是评测契约与架构基底（口径已定案），
本文只讲**设计**。两份要一起读。

**证据等级**（沿用姊妹文档）：🟢 我逐行读代码/亲自跑命令验证 · 🔵 agent 产出且我复核过关键数字 ·
🟡 agent 产出，我未独立复核 —— **用之前请自己再确认**。

---

## 0. 一页纸

| 项 | 内容 |
|---|---|
| **目标** | per-assay 在 WT 复合物结构 + WT target 序列 + WT partner 序列上做 test-time training，提高 ProteinMPNN 在 BindingGYM 上的 **zero-shot** 打榜数字 |
| **已定约束** | 只改 θ；打分公式冻结；界面先验只能进 TTT 的 loss 权重，**不能**进打分 mask |
| **参考点** | 官方 0.3970 · 噪声底 25-assay 均值 σ≈0.0050 · **要涨 >≈0.010 才算数** |
| **天花板** | 14 张 zero-shot 表的逐 assay oracle = **0.4819**（+0.085）；**结构族天花板只有 +0.050** |
| **最大的坏消息** | headroom 最大的 8 个 assay（占总量 71.6%）**8/8 的 oracle 都是序列/MSA 模型**；MPNN 在 7/25 上已经是 oracle |
| **推荐方案** | **A：同源去噪（homolog denoising）on complex backbone**，界面加权作为 **ablation 轴**而非前提 |
| **推荐理由** | 它同时绕开了已被否证的 δ_WT 目标，**并且把 71.6% headroom 所在的 evolutionary 信号，通过 θ 注入一个结构模型 —— 而打分函数一行不改** |
| **必须先拍板的** | 见 §7，三个决策点 |

---

## 1. 我对目标的理解

### 1.1 你的原话（我不改写，作为需求锚点）

> 目前在 BindingGYM 的 zero-shot evaluation 上，效果不错的是 pretrained ProteinMPNN，所以我们目前就是想在
> ProteinMPNN 上先去 design complexTTT，目标是通过我的一套 complexTTT，提高以 ProteinMPNN 为模型基地的模型
> 在 BindingGYM 上的 zero-shot 能力，test-time training over WT-complex-structure、wt-target-protein-sequence、
> wt-partner-protein-sequence；
> 我目前的理解是 BindingGYM 这种 binding-assay 的 DMS-score prediction，对于 structural information 的要求很高，
> 或者说 structure-side information 提供了很多 binding determinant 信息，所以我觉得如何通过 complexTTT，
> 增加 ProteinMPNN 对于当前 WT-complex-structure 的刻画能力 / encoding 能力非常关键，
> 特别是围绕 binding-sites 展开的一系列 prior-knowledge 应该是我们 TTT 的重点。

### 1.2 我的形式化

对每个 assay $a$，给定 **label-free** 的三元组
$\big(X_a^{WT},\ s_a^{\text{target}},\ s_a^{\text{partner}}\big)$，求一次权重更新

$$\theta_a \;=\; \operatorname*{arg\,min}_{\theta}\ \mathcal{L}_{\text{TTT}}\big(\theta;\ X_a^{WT},\ s_a^{WT}\big),\qquad \theta_a^{(0)}=\theta_{\text{pretrained}}$$

然后**用完全不变的官方打分函数**给该 assay 全部 variant 打分：

$$\text{score}_a(v)=-\frac{1}{5}\sum_{m=1}^{5}\sum_{j:\text{mask}_j=1}\mathrm{NLL}\big(S_j^{(v)}\mid X_a^{WT},S^{(v)},\pi_m\big)\Big|_{\theta_a}$$

目标：$\dfrac{1}{25}\sum_a \rho_a$ 显著高于 $\theta_{\text{pretrained}}$ 的 0.3970。

### 1.3 三处我做了解释、需要你确认没理解偏

1. **「zero-shot」= 不碰 `DMS_score`**，不是「不训练」。TTT 更新 θ 但只用结构与 WT 序列 ⇒ 仍是 zero-shot。
   这个定义与 BindingGYM 官方一致（官方 zero-shot 侧 🟢 `grep -rn "DMS_score" modelzoo/ --include=*.py` 为空）。
2. **「per-assay」** = 每个 assay 独立训一次、独立 reset。25 个 assay 之间不共享 θ。
3. **「提高 zero-shot 能力」的判据 = 25 个 assay 未加权平均的 Spearman**（官方聚合口径）。
   若你其实想要的是别的 headline（例如 partner-conditional specificity），设计会完全不同 —— 见 §7-③。

---

## 2. 已经定下来的三条约束

**① θ-only，打分公式 byte-identical。**（你 2026-09-02 拍板）
可动的只有权重。M=5、共享解码顺序、全残基求和、两列输出、`sort_index()`、`to_csv(index=False)` 全部冻结。

> ⚠️ **一处技术例外必须说明**：为了让 TTT 臂与 baseline 臂可配对，`randn_1` 仍必须从全局 CUDA RNG 上摘到独立
> `torch.Generator`（否则 ProteinTTT 的 `self.train()` + ProteinMPNN 的 `dropout=0.1` 一跑就把 philox 流推走，
> 两臂解码顺序不再相同，🟡 已实测）。这**不改打分公式**，但会改变数值 ⇒
> **验收基准是「我们自己重跑的 baseline」，不是官方那 25 个数。** 官方 0.3970 只能作口径对标，不能作 md5 锚点。

**② 界面先验只能进 TTT 的 loss 权重，不能进打分 mask。**（§3-① 的直接推论）

**③ TTT 目标函数的最优解不能是 $\delta_{WT}$。**（§3-⑤ 的直接推论）

---

## 3. 五条改变计划的基底事实

### ① 🔵 界面限制打分是陷阱，不是杠杆

agent 新跑的 25-assay 实验（fixed seed 42, M=5, paired —— 同一份 NLL 张量套不同 mask）：

| mask | mean ρ | median | **paired Δ vs FULL** | wins | worst |
|---|---:|---:|---:|---:|---|
| FULL（官方） | +0.3976 | +0.3859 | — | — | — |
| target 链 only | +0.3874 | +0.3737 | **−0.0102** | 5/25 | −0.059 |
| **target 界面 ≤5 Å** | **+0.2470** | +0.2245 | **−0.1506** | **5/25** | **−0.689 (CD19)** |

**保真度**：它的 FULL 均值 0.3976 vs 官方 0.3970（差 0.0006）。

- **机制是 tie 塌缩**（实测）：≤5 Å 下分数完全相同的 variant 比例 —— CXCR4 **30.9%**、CD19 **20.8%**、
  6M17 15.6%、HLA-A2 13.2%；**9/25 个 assay >1%**，而 FULL mask 下最大只有 2.9%。CD19 的 −0.689 就是这个。
- **判别式已确认**：`ρ(界面残基数/target 链长, Δ_t5) = +0.613, p=0.0011`
  ⇒ **只在 target 链本身已经界面饱和（ratio ≳0.24）时才有用**。
- 语料里那个 exemplar（PSD95_CRIPT，+0.028）是 5 个赢家之一，而**同一个 PDB 上的姊妹 assay
  PSD95_Tm2F 是 −0.057**。

> **口径交代（agent 自报的四条限制，我未复跑）**：variant 上限 3,000（影响 10 个 assay）；单一 seed 42；
> **4 个 Z-domain 行与官方偏差大到无法用 seed 噪声解释**（ZSPA-1_LL1 它 0.1927 vs 官方 0.3066）——
> **那 4 行不能当设计证据**；「双侧界面」列有 bug（`b5 == t5` 全 25 行）。
> ≤8 Å / ≤10 Å 两列是 **label-informed** 的（阈值是看着 label 挑的），只有 ≤5 Å 是合法的 zero-shot 选择。

**⇒ 推论**：界面先验必须以 **TTT 的 loss 权重**形式进入，打分 mask 保持 FULL。
这同时杀掉了一个看似聪明的 θ-only 变体：「训 θ 使非界面残基的 NLL 对突变不敏感」——
那等价于同一个 mask，会产生同样的 tie。

### ② 🟡 `wt-partner-protein-sequence` 作为 TTT 输入几乎无效

partner ablation（target-only 打分 mask，**n=2 assay**）：

| 处理 | PSD95_CRIPT | BH3_Mcl-1 |
|---|---|---|
| 完整 | 0.3615 | 0.6682 |
| partner **序列打乱** | 0.3468（−0.015） | 0.6469（−0.021）← **在噪声底内** |
| partner → 全 `X` | 0.2371（−0.124） | 0.3646（−0.304） |
| partner **整条删除** | 0.3029（−0.059） | 0.5791（−0.089） |

**模型需要 partner 槽位上有「某个合理残基」，不需要是「对的那个」。**

叠加两条 🟢 架构事实：

- **编码器逐 bit 与序列无关** —— 打乱 `S` 后 `h_V`、`h_E`、`E_idx` 全部 `torch.equal → True`（`max|d|=0.0`），
  而全模型 log_probs 移动 3.03。`features.forward(X, mask, residue_idx, chain_labels)` 签名里根本没有 `S`。
  ⇒ **partner 序列对「结构编码能力」的影响恰好是 0。**
- **「界面」在架构里只有 1 个 bit**：`PositionalEncodings` 把全部跨链边压成 one-hot 的 bin 65（共 66 个 bin），
  对应 `linear.weight[:,65]` 一个 16 维向量。🟡 消融它只让 NLL 动 0.0015–0.0063 nats。
  另外 🔵 `num_chain_embeddings=16` 在整个文件里**只出现在两处 `__init__` 签名、从未被引用** ⇒
  模型**分不清抗体 VH↔VL packing 与真正的抗原接触**。🟡 4ZFF 实测：库位点平均有 11.11 个跨链邻居，
  但只有 **2.11** 个在真正的 VEGF partner 上 —— **81% 的「界面信号」是 H↔L**。

**⇒ 推论**：你需求里的第三个输入 `wt-partner-protein-sequence`，在当前架构下**基本是个空槽**。
partner 起作用的方式是**它的 backbone 占据了 k-NN 图的邻居槽位**，不是它的序列身份。
任何以「拟合 partner 序列似然」为目标的 TTT，瞄准的是一个总价值约 0.02 的通道。

### ③ 🟢 零参数的距离尺就有 ProteinMPNN 的 65%（我亲验）

用 `variant_labels.parquet` 的 `min_dist_to_partner`（突变位点到 partner 的最小重原子距离），
**单一全局符号、零参数**：

```
signed Spearman 均值 = 0.2597    vs   ProteinMPNN 0.3970       (65.4%)
在 5/25 个 assay 上打败 ProteinMPNN：
  BH3_Bcl-xL_1PQ1      0.699  vs  0.655
  Z-domain_ZpA963_HL1  0.526  vs  0.136      ← 差 3.9 倍
  PSD95_CRIPT_1BE9     0.486  vs  0.386
  ACE2_SARS2-RBD_6M17  0.401  vs  0.265
  Z-domain_ZSPA-1_LL1  0.374  vs  0.307
```

> **口径**：signed rho 在 21/25 个 assay 上为正，所以用单一全局符号是合法的（无需 per-assay 选符号，
> 那才会是 label 依赖）。分母是 parquet 的 376,424 行（= 官方 376,446 − 22 个 WT 行）。

**⇒ 推论**：**任何 complexTTT 的论文里必须有这一行 baseline**，否则审稿人会替你补上。
而且它同时是一条**正面**证据：在那 5 个 assay 上，MPNN 拿着 1.66 M 参数**没能把界面几何转成排序**，
说明「让模型更好地用界面几何」确实有空间 —— 但也说明那空间可能一把零参数的尺子就够了。

### ④ 🟢 headroom 是「演化的」不是「结构的」（我亲验）

对 `results/` 下 14 张 zero-shot 逐 assay 表取 per-assay max 作 oracle：

```
ProteinMPNN mean 0.3970  |  ORACLE mean 0.4819  |  headroom +0.0850
MPNN 自己就是 oracle 的 assay：7/25（零 headroom）
```

**headroom 最大的 8 个（占总 headroom 71.6%），oracle 全部是序列/MSA 模型：**

| assay | MPNN | ORACLE | headroom | argmax |
|---|---:|---:|---:|---|
| Z-domain_ZpA963_HL1_2M5A | 0.1356 | 0.4742 | **0.3386** | EVE |
| KRAS_SOS1_8BE4 | 0.3092 | 0.5530 | 0.2438 | ESM2 |
| PSD95_CRIPT_1BE9 | 0.3863 | 0.5851 | 0.1988 | TranceptEVE |
| KRAS_DARPinK27_5O2S | 0.4040 | 0.5799 | 0.1759 | TranceptEVE |
| KRAS_PICK3CG-RBD_1HE8 | 0.4686 | 0.6171 | 0.1485 | EVE |
| KRAS_RAF1_6VJJ | 0.4743 | 0.6177 | 0.1435 | TranceptEVE |
| hYAP65_peptide_1JMQ | 0.1153 | 0.2572 | 0.1419 | ESM2_all_seq |
| KRAS_RAF1-RBD_6VJJ | 0.4920 | 0.6229 | 0.1310 | TranceptEVE |

🟡 agent 另报：结构族（ByProt / ESM-if1 / PiFold / PPIformer / ProteinMPNN_single）的天花板只有 **+0.050**。

**⇒ 这是对「binding DMS 对 structure 要求很高」这个假设最直接的反证。**
更准确的表述应该是：**在 ProteinMPNN 已经赢的那 7 个 assay 上，结构确实是主导信息；
但剩下的 71.6% 的钱不在结构侧。** 一个纯结构的 complexTTT，瞄准的恰好是 headroom 最少的那部分。

> ⚠️ 一条同方向的坏消息（🟡）：`KRAS_DARPinK27_5O2S` 的 label 是错的（装了 SOS1 的数据，见姊妹 memory），
> 它在上表里排第 4。修掉之后这一行的 headroom 会变，8 个的 71.6% 要重算。

### ⑤ δ_WT 类目标已被实测否证 —— 但否证范围比记的窄

🟢 我读了原始脚本与日志（`Sources/complex-ttt-evidence/`），**用的就是 ProteinMPNN**，BH3_Mcl-1，
1.66 M 参数全量 SGD 30 步：

| 目标 | lr | WT-NLL/res 降到 | Spearman |
|---|---|---:|---|
| —（frozen, M=5） | — | 1.82 | +0.6817 ± 0.0136 |
| WT-NLL | 1e-3 | 1.7686 | +0.7054 → +0.6935 ± 0.0147（−0.012，≈1σ） |
| WT-NLL | 1e-2 | 1.4652 | +0.6879 → +0.6476（−0.040） |
| WT-NLL | **1e-1** | **0.3867** | +0.6921 → **+0.3884**（−0.304） |
| WT-NLL | 3e-1 | 0.8323 | +0.6868 → +0.4783 |
| masked-denoise (15%→X) | 1e-2 | — | +0.6889 → +0.6536 |
| masked-denoise (15%→X) | 1e-1 | — | +0.6827 → +0.4192 |

**注意 lr=0.3 那行**：它降得比 lr=0.1 **少**（0.83 vs 0.39），伤害也**小**（0.478 vs 0.388）。
⇒ **伤害不跟 lr 走，跟「目标函数被优化了多少」走。** 这比单调 lr 曲线强得多 ——
它说的是**这个目标函数本身错了，优化得越成功排序越差**。

我的机理解读（未验证，是假说）：预训练模型的排序能力来自在百万蛋白上校准过的 $p(S\mid X)$；
在单个样本上朝它的 WT 收敛 = 把这个分布塌成 $\delta_{WT}$，于是 $\log p(\text{mut})$ 退化成
「到 WT 的距离」而不是「这个突变与这个结构有多兼容」。

**🟢 四条限定（我读脚本读出来的，说明这次否证比 memory 里记的窄）**：

1. **只做了一个 assay：BH3_Mcl-1** —— 而它恰好是全 benchmark **partner 占比最高的**（150/173 = 86.7%）。
   损失 `-(one_hot(S_wt)*lp).sum()/L` 因此约 **87% 花在「复现 Mcl-1」上**，而 Mcl-1 在所有 variant 里从不变化。
   **这个目标函数几乎整个用在了不含信号的地方。**
2. **不是 BindingGYM 的 ckpt/设置**：用 `StaB-ddG/model_ckpts/proteinmpnn.pt`，`augment_eps=0.1` +
   显式 `fix_backbone_noise=0.1·randn`；BindingGYM 打分时 `augment_eps=0`。
3. **全参数 SGD**，没试过任何参数子集（encoder-only / LoRA / norm-only / 那 16 个界面参数）、无早停。
4. **只测了两个目标族，而它们都以 $\delta_{WT}$ 为最优解。**

（一条**不**构成 confound：那个脚本的打分减了 `wt_ll`，但它在 assay 内是常数，不改排序。）

**⇒ 被否证的精确命题是**：「以 $\delta_{WT}$ 为最优解的目标 + 全参数更新 + 在最 partner-dominated 的那个 assay 上」。
**不是「ProteinMPNN 上的 TTT」。**

### ⑥ 🟡 无引导的权重移动本身就是抽奖（这条决定对照组怎么设）

编码器加相对高斯噪声 `W += s·std(W)·N(0,1)`，mean Δρ（PSD95 / BH3）：

| s | 1e-3 | 1e-2 | 1e-1 | 0.2 | 0.3 | 0.5 | enc 重初始化 |
|---|---|---|---|---|---|---|---|
| PSD95 | +0.0001 | +0.0010 | +0.0002 | −0.0429 | −0.2147 | −0.3169 | **−0.3455** |
| BH3 | +0.0001 | −0.0000 | −0.0042 | −0.0554 | −0.3046 | −0.5052 | **−0.8957** |

**每个位移档的 Δ 均值都 ≤ 0**，但 **s=0.1 上 13 抽取最好的一次给 +0.0565 / +0.0371** ——
**比整个带标签微调的 headroom（+0.025）还大。**

**验收包络**（120 次扰动，按分数向量移动量分箱）：

| 1 − ρ(扰动后, baseline) | n | mean Δρ | **sd Δρ** |
|---|---:|---:|---:|
| ≤0.005 | 31 | +0.0012 | 0.0053 |
| 0.005–0.02 | 34 | −0.0073 | 0.0163 |
| 0.02–0.05 | 14 | −0.0002 | **0.0291** |
| 0.05–0.15 | 18 | −0.0291 | 0.0484 |
| 0.15–0.40 | 9 | −0.1022 | 0.1639 |
| 0.40–1.10 | 14 | −0.1929 | 0.2179 |

**⇒ 对照组不能是「不做 TTT」**，必须是 **N 个「参数范数匹配」的随机扰动**；
增益必须过上表在实测位移处的 sd 包络（例如 ρ=0.95 时 sd=0.029 ⇒ 需要 ≈+0.06）。
**报告里必须写实测位移量，不能只写 lr × steps。**

---

## 4. 与原始假设的张力（诚实陈述）

你的假设：**「structure-side information 提供了很多 binding determinant 信息 ⇒ 增强 ProteinMPNN 对 WT 复合物的
刻画能力是关键，binding-sites 先验是重点。」**

支持它的证据：

- 🟢 complex 相对 single-chain 确实有 **+0.0406**（0.3970 vs 0.3564）—— partner 进图是有用的。
- 🟢 §3-③：在 5 个 assay 上，**零参数距离尺打败了 1.66 M 参数的 MPNN** ⇒ MPNN 确实没把界面几何用好，有空间。
- 🟡 编码器重初始化毁掉一切（−0.35 / −0.90）⇒ 结构编码确实是承载信号的地方。

反对它的证据（更强）：

- 🟢 §3-④：71.6% 的 headroom 在 8 个 assay 上，**8/8 的 oracle 是序列/MSA 模型**；结构族天花板只有 +0.050；
  MPNN 在 7/25 上已经是 oracle。
- 🔵 §3-①：界面限制打分 **mean Δ −0.1506，只赢 5/25**。
- 🟡 §3-②：partner **序列**打乱只掉 0.015–0.021（噪声底内）；「界面」在架构里只有 1 个 bit 且近乎惰性。
- 🟡 跨链边只占全部边的 **7.6%**（pooled）；**1,191/2,220 = 53.6% 的库位点根本没有任何跨链邻居**；
  平均 19.3% 的 variant 一条跨链边都不碰。⇒ 一个不加权的全复合物 TTT loss，**>90% 的梯度花在单体结构上**。
- 🟡 信号在**位点分辨率**而非界面分辨率：单突变上 η²(位点身份) = **0.553**，η²(替换成什么氨基酸) = 0.059，
  **η²(二值界面标签) = 0.0176**。
- 🟡 **15/22 个结构是同源模型（`_hm`）** ⇒「更好地刻画这个界面」在 15/25 个 assay 上可能是「拟合建模误差」。
- 🔵 覆盖不是瓶颈：346 个 Cb < 8 Å 到 partner 的库位点，k=48 图**一个都没漏**。**加大 k 买不到界面覆盖。**

**我的结论**：假设**不是错的，但被放错了位置**。
正确的表述是 —— **ProteinMPNN 不是「一个会看界面但看得不够准的模型」，而是「一个几乎不看界面的模型」**
（1 个 bit、7.6% 的边、53.6% 库位点无跨链边）。而 θ-only 又意味着我们**不能给它装新的界面通道**
（那要改 featurization，已被 §2-① 排除）。

⇒ **在 θ-only 下，「增强界面刻画」能做的上限，是让梯度更多地落在那 7.6% 的边和 46% 的库位点上。**
这是一个真实但**有界**的杠杆，值得做成 ablation 轴，**不适合当作整个方法的立论**。

---

## 5. 三个候选方案

### 方案 A（推荐）：同源去噪 on complex backbone，界面加权作 ablation 轴

**目标函数**

$$\mathcal{L}_A(\theta)=\sum_{h\in\mathcal{H}_a}\pi_h\sum_{j\in\text{target}} w_j\cdot\mathrm{CE}\Big(p_\theta\big(\cdot\mid X_a^{WT},\,S^{(h)},\,\pi_m\big)_j,\ S_j^{(h)}\Big)$$

- $\mathcal{H}_a$：该 assay 的 **target 链同源序列**，来自 22 个已分发的 `.a2m`（depth 3,207–756,017）。
  把同源序列 thread 到 **WT 复合物 backbone** 上，**partner 链固定为 WT 且不计 loss**。
- $w_j$：位点权重。**arm 0：$w_j\equiv 1$；arm 1..k：$w_j = 1+\lambda\cdot[\,j\text{ 有跨链边}\,]$**，
  λ 取到界面位点占梯度约 50%（🟡 agent 估 5–10×）。

**为什么推荐**

1. **绕开已被否证的失败模式**：最优解是**同源分布**而不是 $\delta_{WT}$（§3-⑤）。
2. **它瞄准了钱真正在的地方**：71.6% 的 headroom 由序列/MSA 模型持有（§3-④）。
   这个方案的本质是 —— **在不改一行打分函数的前提下，把 evolutionary 信号通过 θ 注入一个结构模型**。
   这是 θ-only 约束下唯一能触及那 71.6% 的路径。
3. **它把你的假设变成一个被测量的结果而不是前提**：界面加权是 ablation 轴，
   跑完就能回答「界面先验到底值多少」，而不是假设它值很多。
4. 🟢 MSA 是**已分发资产**，合法可用。

**已知风险 / 必须先处理**

- 🟡 `.a2m` 的 query = **仅被突变的 focus 链**按字母序拼接，22/22 长度吻合，**partner 从不进 query**。
  ⇒ 18/22 严格 target-side，**拿不到 complex-level 进化信号**。这个方案注入的是**单体演化约束**，
  不是界面共进化。**必须在论文里写清楚，别声称是 co-evolution。**
- 🟡 MSA depth 跨 3 个数量级 ⇒ per-assay 的有效样本量差异巨大，需要一个 depth-aware 的步数/采样策略
  （**不能看着 Spearman 调，那是泄漏**）。
- 4 个 Z-domain assay 两条链都突变、无固定 partner（🟢 54,563 variant = 14.5%）⇒ 需要显式策略（§7-①）。
- 🟡 transductive 提醒：`msa_BindingGYM.py:29-36` 的 `focus_chains` 是遍历整张 variant 表的 `mutant` 列推出来的
  ⇒ **只要用了分发的 `.a2m`，就已间接消费了 variant 列表**。论文必须声明。

### 方案 B：只做界面加权的自一致目标（不引入 MSA）

$$\mathcal{L}_B(\theta)=\sum_j w_j\cdot \mathrm{KL}\Big[p_\theta\big(S_j\mid X\big)\ \Big\|\ p_\theta\big(S_j\mid X+\epsilon\big)\Big],\qquad \epsilon\sim 0.2\,\text{Å}$$

或等价的 `CE(p_θ(·|X+noise), S_WT)` 形式（`augment_eps=0.2` 正是 ckpt 训练时的同分布噪声）。

- **优点**：零外部依赖、可动部件最少、最贴近你原本的想法（纯粹「更好地刻画这个复合物」）。
- **缺点（先验 upside 很窄）**：
  - 🟡 §C.4 实测：一个**冻结的、被扰动 0.3 Å 的结构**只让分数变 −0.027 / **+0.020**
    ⇒ **分数根本不受 backbone 精度限制**，这条目标瞄准的量本来就不大。
  - 🟡 53.6% 的库位点没有跨链边 ⇒ 界面加权在它们身上是恒等操作。
  - 🟡 15/22 是同源模型 ⇒ 「对建模误差鲁棒」和「拟合建模误差」是同一个方向的两种说法。
  - $\mathcal{L}_B$ 的一个平凡最优解是**让 $p_\theta$ 与 $X$ 无关**（完全平滑）—— 必须加锚定项防止塌缩，
    而最自然的锚定项就是 $\delta_{WT}$，又踩回 §3-⑤。**这条需要额外的设计工作才能站住。**

### 方案 C：先不定目标，先补那个缺失的关键数字

workflow 自己指出：**最决定性的数字还没测** —— ProteinMPNN 逐 variant 分数与 `min_dist_to_partner` 的
**偏相关**，即「MPNN 的 0.3970 里有多少已经就是那把零参数距离尺」。

```
ρ( MPNN_score , DMS )              = 0.3970   ← 已知
ρ( min_dist   , DMS )              = 0.2597   ← 我已验
ρ( MPNN_score , min_dist )         = ?        ← 缺
partial ρ( MPNN , DMS | min_dist ) = ?        ← 缺，这个才是关键
rank-ensemble(MPNN, min_dist) vs 0.4819 ceiling = ?
```

分数在 workstation `10.67.24.41:/data/guoj0f/BindingGYM-zero-shot-proteinMPNN/scores/seed1_M5/`（25/25 齐全），
**纯 CPU 就能算**。

- 若偏相关**高**（MPNN 基本就是距离尺的平滑版）⇒「增强结构刻画」这个方向本身要重想。
- 若偏相关**低**（MPNN 的信号与几何正交）⇒ 方案 A 的立论更硬，且「界面加权」的预期收益要下调。
- **代价**：多花半天，但不写任何会丢的代码。

### 我的推荐

**A**，且把 **C 作为 A 的第 0 步**（它便宜、纯 CPU、并且会直接改变 A 里 λ 该怎么设）。
B 作为 A 的对照臂之一，不作为主线。

---

## 6. 无论选哪条都必须做的事

### 6.1 三个对照臂（缺一不可）

| 臂 | 内容 | 作用 |
|---|---|---|
| **baseline'** | 用独立 Generator 重跑的 no-TTT baseline | 新锚点（§2-① 的例外） |
| **random-θ** | **N ≥ 13 个参数范数匹配的随机扰动**，取分布而非最好那次 | §3-⑥：否则你测的是抽奖 |
| **zero-param** | `min_dist_to_partner` 单一全局符号 | §3-③：审稿人一定会问 |

### 6.2 验收门槛

1. 报告 **ρ(TTT 分数, baseline' 分数)** 的实测位移，查 §3-⑥ 的 sd 包络，**增益必须超过该位移档的 sd**。
2. **逐 assay 报告 + cluster-collapsed 均值**（🟢 25 assay ≈ 14 簇；🟡 KRAS 一家 6 个 assay、占总 headroom 43.8%
   ⇒ **只动一个 KRAS assay 就能造出 +0.01 的均值**）。
3. 报告**最差 assay 的 Δ**。🟡 官方带标签微调在 8/25 上是负的，ZSPA-1_LL1 从 0.3066 掉到 **0.0121** ——
   这正是要防的失败模式。
4. `--ttt_steps 0` 时 25 个 CSV 必须与 baseline' **逐个 md5 相同**。

### 6.3 必须先修的四个 bug（否则结果无意义）

| # | 位置 | 症状 |
|---|---|---|
| 1 | 🟢 `base.py:391 best_confidence = 0` + `:483 if confidence > best_confidence` + `:618 if best_state is not None` | ProteinMPNN 的自然 confidence 是**负**对数似然 ⇒ `best_state` 永远 `None` ⇒ **rollback 整段被跳过**；`confidence_collapse_ratio` 早停也因 `:489 best_confidence > 0` 一起失效。**两道保险被一个负号全关掉。** 修法：用天然为正的 confidence，如 $\exp(-\mathrm{NLL}/L)\in(0,1]$ |
| 2 | 🟡 `base.py:1189-1193 _ttt_unnormalized_cross_entropy_loss` | **忽略它自己的 `mask` 参数**（实测 mask 取 2/7、7/7、None 都返回 2.7580）⇒ loss 会算到 padding 与 438 个 ghost 残基上 |
| 3 | 🟢 `augment_eps` | **不受 `self.training` 门控**，且是 forward 第一行 ⇒ 会**重采 k-NN 图**。打分前必须显式复位 0.0，否则 🟡 最多 −0.19 ρ 且不可复现 |
| 4 | 🟡 `lora_target_replace_module` 默认 `"MultiheadAttention"` | ProteinMPNN 里 **43 个 `nn.Linear` / 16 个 `LayerNorm` / 1 个 `Embedding`，零个 MultiheadAttention** ⇒ 默认配置**匹配不到任何模块**，LoRA 静默不生效。另 🟡 `lora_diffusion` 未安装，`base.py:174-178` 在 `lora_rank>0` 时会抛异常 |

### 6.4 其它已知会静默出错的地方

- 🔵 **绝不要从 `training/protein_mpnn_utils.py` import** —— 它的 `ProteinMPNN.forward(data, ...)` 是**另一个契约**：
  收 PyG batch、**缓存一个解码顺序**、**eval 模式返回 `(wt_scores - scores)` 标量而不是 `log_probs`**、
  且用 `chain_M` 而非 `mask` 做 loss mask。
  ⚠️ **由此还引出一条**：官方带标签微调的 0.4217 是**用另一个打分泛函**产生的 ⇒
  「headroom 只有 +0.025」这个说法要打折扣，不能直接与 zero-shot 的 0.3970 相减。
- 🔵 `ProteinMPNN.forward` 返回的已经是 `F.log_softmax`（`:1101`，实测 `exp().sum()=1.0`），
  而 `base.py` 会把 `_ttt_predict_logits` 的输出喂给 `cross_entropy` ⇒ **双重 log_softmax**。
- 🔵 `_scores` 是**求和**（1HE8 上 ~1582）不是均值（~1.73）⇒ 直接喂优化器会让**有效 lr 随 L 变化**，
  1HE8 比 1JMQ 热 20 倍。
- 🔵 `crop_size` 默认 1024 会**切断链并重建 k-NN 图**；多个复合物超过它（1HE8 L=1107、1N8Z L=1041）。
- 🔵 assay 之间必须 `ttt_reset()`：`@preserve_model_state` 恢复 mode/requires_grad/device，**但不恢复权重**。
- 🟢 `ProteinMPNN.__init__` 对所有 dim>1 参数跑 `xavier_uniform_`（`:1055-1057`）⇒ **权重加载必须在构造之后**。

### 6.5 成本（不是约束）

🟡 全参数一步（B=1, L=1107）：**109.7 ms**，峰值 **2,416.6 MiB**；params+grads+Adam = 25.34 MiB。
Σ L over 25 assays = 9,931。**25 assay × 100 步 ≈ 100–200 s A4500 时间。**
B=5 全参数需 11,963 MiB —— A4500(20 GB) 够，**TITAN X(12 GB) 不够**。
⇒ **选 PEFT 是为了控制过拟合，不是为了省显存**（🟡 反直觉：norms-only 在 B=1 时反而更贵，1,179.7 MiB，
因为激活占主导；冻结**靠前**的模块才省激活）。

---

## 7. 需要你拍板的三个决策点

**① 4 个 Z-domain assay（🟢 54,563 variant = 14.5%）没有固定 partner，怎么办？**
两条链都被突变（ZSPA-1_LL1 99.6%、LL2 95.9%、ZpA963_HL1 92.9%、HL2 55.2%）。
你需求里的 `wt-partner-protein-sequence` 在它们上**不存在**。选项：(a) 两条链都当 target；
(b) 排除并显式报告（但它们占 14.5%，且 ZpA963_HL1 是 headroom 最大的那个）；(c) 其它。

**② 那 6 个 test-time 输入逐字节相同的 assay（🟢 占 headline 的 24%）怎么报告？**
`1LP1`→ZSPA-1 LL1/LL2、`2M5A`→ZpA963 HL1/HL2、`6VJJ`→KRAS_RAF1/RAF1-RBD，
md5(`chain_id`+`wildtype_sequence`+`pdb_file`) 三对全部 IDENTICAL ⇒
**WT-only TTT 对它们必然给出完全相同的 θ 和完全相同的分数**。
这既省算力，**也是最诚实的自检：若增益全落在这 6 个上，那不是 assay-specific 适配。**

**③ headline 到底是什么？**
若是 25-assay 均值 Spearman：§3-④ 说结构族天花板只有 +0.050，且钱在序列侧 ⇒ 方案 A 是唯一够得着的路。
若你愿意换成 **partner-conditional specificity**（partner-blind 打分器在该指标上恒等于 0.000，
是唯一 baseline 结构上做不到的读数）—— 那设计会完全不同，而且 §3-② 说当前架构在这个指标上先天很弱。
**这个决定在写第一行代码前必须定。**

---

## 8. 若批准 A，执行顺序（每阶段带 kill criterion）

| 阶段 | 内容 | 产出 | kill criterion |
|---|---|---|---|
| **S0** | 方案 C：拉 workstation 的 `seed1_M5` 分数，算 MPNN × min_dist 的偏相关 + rank-ensemble vs 0.4819 | 一张 25 行表 | 偏相关 >0.8 ⇒ 停下重议方向 |
| **S1** | 修 §6.3 的四个 bug；写 `compute_fitness_multi_pdb_ttt.py`（三处改动）；重跑 baseline' | md5 清单 + baseline' 25 行 | `--ttt_steps 0` 的 md5 对不上 ⇒ 不许往下走 |
| **S2** | random-θ 对照：N≥13 个范数匹配扰动，**全 25 assay** | §3-⑥ 那张包络表的 25-assay 版 | — （这是门槛本身，不 kill） |
| **S3** | 方案 A arm 0（$w_j\equiv1$），先在 3–5 个 assay 上定 lr/步数，**用 held-out assay 调，不看目标 assay 的 Spearman** | 逐 assay Δ | Δ 不过 S2 的包络 ⇒ 目标函数不成立，回 §5 重选 |
| **S4** | 方案 A arm 1..k（界面加权 λ 扫描），**排除 CXCR4 / HLA-A2 / CD19**（🟡 它们大多数 variant 无法被界面重排序） | λ–Δ 曲线 | λ 曲线平坦 ⇒ 界面先验被证伪，如实报告 |
| **S5** | 全 25 assay + cluster-collapsed 均值 + 最差 assay + 那 6 个同输入 assay 的单独一栏 | 最终表 | — |

**先做 S0–S2 再碰 S3。** S1/S2 的产出即使方法失败也是可复用的资产。

---

## 9. 本文尚未关闭的问题

1. 🟡 §3-① 的 25-assay 实验有四条自报限制（3,000 上限 / 单 seed / **4 个 Z-domain 行无法解释** /
   `b5` 列有 bug），**那 4 行不能当设计证据**。
2. 🟡 §3-② 与 §3-⑥ 都只在 **2 个 assay**（PSD95_CRIPT、BH3_Mcl-1）上测过。
3. 🟢 §3-⑤ 只在 **1 个 assay**（BH3_Mcl-1，最 partner-dominated 的那个）上测过。
4. **未测**：TTT 的权重更新如何**重分配逐残基方差**。一个可能的失败模式（未验证）：
   TTT 均匀降低 NLL，于是把「73% 方差、ρ=−0.019」的噪声块动得比界面块还多。
5. **未测**：只有 Spearman。BindingGYM 还评 AUC / MCC / NDCG / AP / TopHit@10 等，
   🟡 TopHit@10 在任何 mask/θ 变化下可能表现完全不同。
6. **未定**：`X`（index 20）该不该进 `_ttt_get_non_special_tokens()` —— 它是模型训练时见过的合法未知残基 token，
   不是「预测我」的标记。两种答案会改变 loss 触及哪些残基，**没有任何文件能决定这个**。
7. 🟡 `ProteinMPNN_single` 的口径不明：agent 用「删掉 partner 链」复现得 0.3029/0.5791 vs 官方 0.2809/0.5844
   （BH3 接近，**PSD95 差 0.022**）。「single」到底指截断 PDB 还是完整复合物 + target-only 打分 mask，
   **未解决** —— 而这两者作为「partner 值多少」的对照是很不一样的。
8. 🟡 反向传播穿过 `features` **未测**：`topk`（`:946`）对邻居**选择**不可导。
   只训权重时无妨（坐标是常数），但任何优化坐标的变体会被它卡住。
9. 🟡 `TTTModule` + `ProteinMPNN` 的组合**从未端到端跑通过**。

---

## 附：证据来源

- 🟢：本 session 内我逐行读 `/home/guoj0f/repos/BindingGYM`、`proteinttt/`、
  `/home/guoj0f/repos/Sources/complex-ttt-evidence/` 的代码与日志，以及在
  `~/anaconda3/envs/proteingym-ttt` 与 `~/anaconda3/bin/python` 里跑的验证脚本。
- 🔵/🟡：两轮 workflow ——
  `wf_98617ef3-936`（评测契约，14 agent / 1,639,470 token / 470 tool calls / 57 min / 0 error）与
  `wf_d264429a-66c`（架构基底，9 agent / 1,113,948 token / 290 tool calls / 3.1 h / 0 error）。
  journal 在 `~/.claude/projects/.../subagents/workflows/<run_id>/journal.jsonl`。
- 姊妹文档：`BindingGYM_zeroshot_protocol_20260902.md`（评测契约与架构基底，口径已定案）。
- 相关 memory：`bindinggym-zeroshot-protocol-and-ttt-hooks`、`complex-ttt-wt-likelihood-refuted`、
  `proteinmpnn-partner-blind-on-bindinggym`、`bindinggym-wt-anchor-per-assay`、
  `bindinggym-binding-sites-artifacts`、`bindinggym-43pct-no-interface-resolved`。
