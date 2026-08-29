# ProteinMPNN 的 binding 信号来自 partner 还是 intrinsic constraint？

> created 2026-08-29 · status: PLANNED · 分支 `proteinTTT-proteinGYM-reproduce` · project `proteinTTT-repro` · task `binding_assay_with_partner`
> 总预算 ≤ 8 A100-h（计划 5.4 h + 1.6 h contingency）。**Stage 0 = 0 GPU。**
> 方案来源：3 路独立设计 × 2 hostile reviewer（44 条 flaw，2 个设计被判 fatally-confounded）后的合并版。
> **本文档区分「我已复核」与「agent 声称、未复核」，见 §6。**

---

## 1. 问题与当前证据

**问题：** ProteinMPNN 在 binding DMS 上的 zero-shot 信号，是来自 **binding partner**，
还是来自单链的 intrinsic constraint（folding stability + 通用结构相容性）？

**用户假说：** 「它算的是 structure–sequence compatibility；只给单体就只能捕捉 folding stability。
给它 partner 序列 + 复合物结构，binding 预测会大幅提升。」

### 1.1 已完成的零 GPU 测量（Stage 0 的一部分，本人实测）

用 workstation 上已有的 `seed1_M5` per-variant `design_score` + MAVEdb 的 KRAS 真值。

**(a) partner-swap 混淆矩阵，label = MoCHI 分解出的 ΔΔG（|Spearman|，越大越强）**

| 打分所用结构 | K27 | SOS1 | RALGDS | RAF1RBD | PIK3CG | K55 | **ΔΔG_fold** | 对角线 rank |
|---|---|---|---|---|---|---|---|---|
| 5O2S KRAS–K27 | **.180** | .138 | .125 | .154 | .087 | .097 | **.388** | 1/6 |
| 8BE4 KRAS–SOS1 | .127 | .153 | .205 | .234 | .203 | .183 | **.399** | 5/6 |
| 1LFD KRAS–RALGDS | .118 | .171 | **.278** | .222 | .217 | .192 | **.403** | 1/6 |
| 6VJJ KRAS–RAF1 | .065 | .115 | .318 | **.274** | .295 | .284 | .260 | 4/6 |
| 6VJJ KRAS–RAF1-RBD | .117 | .156 | .231 | **.229** | .220 | .194 | **.387** | 2/6 |
| 1HE8 KRAS–PIK3CG | .140 | .158 | .244 | .238 | **.234** | .178 | **.396** | 3/6 |

```
配对 Δ(对角线 − 其余五个均值) = +0.0440   正号 5/6   Wilcoxon p = 0.062
对角线平均排名 2.67 (随机 3.5)            排列检验 p = 0.144
```

**(b) 最重要的一列是 ΔΔG_fold。** ΔΔG_fold 与 ΔΔG_bind **近正交**（实测 Spearman 0.083–0.234），
所以下面这个对比不是共享成分造成的假象：

| | \|ρ\| vs ΔΔG_fold | 最好的 binding | 倍数 |
|---|---|---|---|
| 均值（6 行） | **0.372** | 0.248 | **1.57×** |
| 配对 Wilcoxon | | | p = 0.0625，5/6 |

⇒ **在两个几乎独立的量上，ProteinMPNN 明显更贴 folding。**
partner-specificity **存在、非零，但只有 folding 的约 2/3**。

**(c) label 选择本身会改变结论。** 我最初用 raw fitness 做同一张矩阵，结论明显更弱：

| label | partner 之间相关 | 与 folding 相关 | PC1 | 配对 Δ | p |
|---|---|---|---|---|---|
| raw fitness | 0.65–0.87 | **0.54–0.64** | **79.7%** | +0.0317 | 0.219 |
| **ΔΔG_bind** | 0.37–0.72 | **0.08–0.23** | 63.3% | **+0.0440** | **0.062** |

raw fitness 被一个共享成分（大部分是 folding）主导 ⇒ **后续一律用 ΔΔG，不用 raw fitness。**

**(d) 零参数几何对照。** 同一张矩阵换成「突变位点到 partner 链的最小重原子距离」：
配对 Δ = +0.0015、对角线平均排名 3.40（随机 3.5）⇒ **纯几何完全没有 partner-specificity**。
ProteinMPNN 比它强（配对差 +0.031），但 n=5 时 p = 0.438。

### 1.2 这些测量还不能回答的

上面全是**观察性**的（换 label、换结构），不是**干预性**的。要回答「给 partner 会不会提升」，
必须做受控 arm 对照。而且 KRAS 家族的 label 之间本就相关 0.37–0.72，
**对角线检验这条轴在 KRAS 上功效有限**（见 §5 限制 4）。

---

## 2. 全局约定（一次钉死，避免事后 forking）

| 项 | 定义 |
|---|---|
| **score** | `s(v) = −(1/L_mut)·Σ_{i∈被突变链} NLL_i`，**higher-is-better**，与 BindingGYM 的 `design_score` 同向。分母固定为被突变链长度，**所有 arm 相同**；partner 自身的 NLL 永不进分数。 |
| **预测方向** | 所有 primary 统计量的 support 方向为**正**。「与预测相反且显著」是第三种结局，单独解释。 |
| **per-variant 主量** | `Δ(v) = s_N(v) − s_X(v)`；**WT-centered** `Δ̃(v) = Δ(v) − Δ(WT)`。 |
| **primary 统计量** | `ρ_id = Spearman(Δ̃, DMS)`（单个相关，null 恰为 0）。**不用两个 ρ 相减** —— Δρ 有 attenuation-asymmetry 造成的结构性正 null。 |
| **分层** | 按 **k=48 图的 1-hop 可达性**（突变位点的 48 近邻里是否含 partner 残基），**不按连续距离阈值**。同时把 hop 数 / partner 占据的邻居 slot 数作为连续协变量。 |
| **误差** | 一律 **position-clustered bootstrap**（重采样单位 = 残基位点，整位点连同其 ~16–19 个变体）。 |
| **seed** | 每个 (assay, seed) 共享同一个 `randn`。共享 randn 只在 score 层配对，**ρ 层不完全抵消**，故每个 contrast 旁必须报 across-seed SD。 |

### Arms（`augment_eps = 0`；partner 恒在 `fixed_chain_list`，因此恒被最先解码、完全可见）

| Arm | 内容 | 回答什么 |
|---|---|---|
| **N** | partner 坐标 + 真实序列 | 参照 |
| **X** | 与 N 逐比特相同的坐标/mask/randn/chain_M，**只把 partner 的序列 token 全设为 `'X'`**（index 20） | **primary：partner 的序列身份通道** |
| **Xd** | 同上，partner 换成**等长的无关真实蛋白序列**（K=8 draw） | X-token 的 OOD 对照 + within-assay null |
| **A** | 删掉 partner 全部原子 | presence / geometry 通道 |
| **O** | partner backbone 刚体搬到非 epitope 表面 patch，接触原子数匹配 | **occlusion null**（只在 KRAS 做） |
| **BG** | BindingGYM 原生口径 | 复现锚点，不参与配对 |

**已删除的 arm 及理由：** `+500 Å 平移`（与直接删除逐比特相同，k-NN 图早已断开）；
`composition-matched scramble`（效应与 partner 长度共线：77→0.029、941→0.179、475→0.314，
是 OOD 冲击不是身份消融）。

> ⚠️ 为什么 primary 不是「删掉 partner」：删除同时改变**序列长度、k=48 邻居图、自回归解码顺序**，
> 三者混在一起。**N vs X 保持几何完全不变，只切断「partner 是谁」这一条信息通道。**

---

## 3. 分阶段执行

### Stage 0 — 0 GPU（部分已完成）

**S0.1 正确性 harness（必须先跑，否则后面全废）**

- 按 CSV `chain_id` 顺序**显式构造**链拼接，**不要**依赖 `tied_featurize` 的
  `all_chains = masked_chains + visible_chains`（`protein_mpnn_utils.py:227`，
  `.sort()` 在 225–226 行被注释掉 —— **已复核**）。
  一旦把 partner 设为 fixed，链顺序就与 CSV 不一致；1HE8（KRAS 是 B 链，941+166=1107）
  与 1LFD（87+167=254）长度刚好填满、**不会报错**。
- 硬 assert（每 assay × 每 arm）：`_S_to_seq(S[0], chain_M_被突变链)` 必须逐字符等于
  该 assay 的被突变链 `wildtype_sequence`。
- MAVEdb join unit test：`hgvs_pro` 可直接构 key（已验 174,794 行 100% 一致）；
  1LFD 编号 −200；**6VJJ 需 shift −1**（参考序列多一个前导 `G`）。assert join 行数 > 20000。

**S0.2 【headline，0 GPU】BindingGYM 的 0.391 有多少是 partner 够不到的？**

对 21 个有 partner 的 assay，用 `seed1_M5`：算每个突变位点的 1-hop 可达性与 hop 数，
得 `ρ_all` / `ρ_reach` / `ρ_unreach`，以及 unreachable 变体在 rank-covariance 中的份额 `w`。
`f = median(ρ_unreach / ρ_all)`。若 f 高，则 partner 能买到的**上限**
`gain_max ≈ (1−w)·(1 − ρ_reach)` —— 不花 GPU 就把「加 partner 会大幅提升」量化成一个上界。

MDE：每 assay unreachable 位点约 60 个 → SE(ρ_unreach) ≈ 0.13，比值 SE ≈ 0.2；
21 个 assay 取 median → SE ≈ 0.05。足以区分 f=0.85 与 f=0.50。

**S0.3 label 侧天花板（KRAS ΔΔG）**
用 `*_std` 列做 disattenuation，报 `rel(ΔΔG_bind,i | ΔΔG_fold 残差)`。
ceiling control：用现有 `design_score` 算 `ρ(score, ΔΔG_bind,i | ΔΔG_fold)`，
Steiger/Williams dependent-correlation test。**若一个已知能给 0.39 的分数在残差上也只有 ~0.05，
说明这个 label 没有可检测 headroom，Stage 3 就不该跑。**

**S0.4 null 标定**（全部 0 GPU，全部写进预注册）：mismatched-label Γ、contact-count oracle、
几何 oracle 对角线、decoding-order σ、position ICC。

#### 🚦 GATE G0 —— 是否花 Stage 1 的 0.6 GPU-h
GO 需**同时**满足：
- **G0a**：≥3/5 KRAS partner 的 `ΔΔG_bind | ΔΔG_fold` 残差 split-half reliability **≥ 0.50**
- **G0b**：**≥6 个 assay** 各有 **≥30 个独立 1-hop 可达位点**且 **≥300 个可达变体**

任一失败 → **STOP，0 GPU 花费。** 结论写成：问题在 label / panel 层面就没有 power，
需要一个 **epitope 互不重叠**的家族（例如同一抗原的多抗体 panel）才能问这个问题。

> S0.2 的 f 值**不是 gate**。f 高只会把 Stage 1/2 的 target 重定义为可达子集，不 kill 实验。

---

### Stage 1 — 0.6 GPU-h：机制门（partner 序列到底进不进分数）

**4 个 pilot assay**（4 个独立 cluster，partner 长度 65–597）：
`KRAS_RAF1-RBD_6VJJ`(p77, 实验结构)、`SARS2-RBD_ACE2_6M0J`(p597, 实验结构)、
`GB1_IgG-Fc_1FCC`(p206)、`CXCR4_CXCL12_8U4O`(p65)。

每 assay：arms {N, X, Xd×1, A, BG} × 800 可达变体 + WT 行 × 3 seeds ≈ 48k forwards ≈ **0.6 GPU-h**。

**两个 readout，都不用 Spearman，所以 power 高：**

1. **`R_id = SD_v[Δ̃] / SD_v[δ_seed]`**，`δ_seed` = 同臂跨 seed 的配对差。
   *为什么不是 `mean|Δ|`*：Δ 里含一个对所有变体近似恒定的 offset（partner 可见后界面残基
   NLL 的整体位移），而 Spearman 对该常数免疫。真正驱动 primary 的是 SD。
2. **序列通道 recovery ladder**：界面位点上 `mean log p(WT residue)`（**连续量，不是 argmax
   recovery** —— 32 位点的 argmax 二项 SE = 8.8 pp，分辨不出东西）。
   预注册顺序 **X < Xd < N**（错误身份必须比无身份**更差**）。

**MDE：** `R_id` 每 assay 可检出 ≥ 1.15；`Δ mean log p(WT)` 每 assay MDE ≈ 0.12 nats，
4 assay 合并 ≈ 0.06 nats。

#### 🚦 GATE G1 —— 是否花 Stage 2 的 3.2 GPU-h
GO 需**同时**满足：
- **R_id ≥ 1.0 在 ≥3/4 pilot assay**（partner 身份对分数的扰动**不小于**重抽一次解码顺序）
- **`mean log p(WT)|_N − mean log p(WT)|_X ≥ +0.05 nats`**，position-paired 95% CI 排除 0，≥3/4 assay

**NO-GO → STOP，总花费 0.6 GPU-h。** 结论（强、可发表、**不需要任何 label**）：
*ProteinMPNN v_48_020 在 partner 完全可见、几何完全相同的条件下，其被突变链分数对 partner
序列身份的响应不超过 decoding-order 噪声；「把 partner 序列交给它」这一具体 remedy
在该 checkpoint 上无机制基础。*

---

### Stage 2 — 3.2 GPU-h：panel 上的正式检验

**8 个 assay / 6 个 cluster**：`6VJJ(RAF1-RBD)`、`8BE4(SOS1)`、`5O2S(K27)`、`1LFD(RALGDS)`、
`6M0J`、`6M17`、`1FCC`、`8U4O`、`5WER`。
（6M0J 与 6M17 合并为一个 cluster —— 同一 SARS2-RBD/ACE2 界面从两侧测；6 个 KRAS 只算一个 cluster。）

**排除并记录理由：** 4 个 Z-domain（两条链都突变，无 partner）；PSD95 ×2（partner 5 aa）、
hYAP65（10 aa）；4D5_1N8Z（2076/2080 变体全在界面，无 unreachable 对照）；
1HE8（partner 941 aa 且 DMS 序列与同源模型仅 79.6% identity，降级为 sensitivity 行）。

规模：8 × 5 arms × 1200 变体（600 reachable + 600 unreachable）× 5 seeds ≈ 240k forwards ≈ 2.6 h；
加 within-assay null（Xd 的 K=8 独立 decoy × 400 变体 × 8 assay × 1 seed ≈ 26k）≈ 0.3 h。合计 ≈ **3.0 h**。

**统计量（只有一个 primary）：**
`ρ_id(assay) = Spearman(Δ̃, DMS)` 在 **1-hop 可达位点**上，K=5 seed 后取均值，position-clustered bootstrap。
Pooling：**per-assay 表格为 primary** + 跨 6 个 cluster 的**单边 exact sign test**
（n=6 时最小可达 p = 1/64 = 0.016）。**不用 median + cluster bootstrap**（n=6 时 percentile CI 严重塌缩）。

**必须同时报（不是可选）：** 每 arm 每 stratum 的 **level** `ρ(s_arm, DMS)`（不只是差值）；
每个 contrast 的 across-seed SD；`|Δ̃|` 对 hop 数 / partner 占据 slot 数的连续回归；
singles-only 子分析；Kendall tau-b 与 disattenuated Spearman 并列。

**MDE（诚实的）：** 每 assay `SE(ρ_id) ≈ 0.16` → 单 assay MDE ≈ **0.40**。
6 cluster 合并 → SE ≈ 0.066，pooled MDE ≈ **0.17**（有异质性时 ≈ 0.19）。
要把 pooled MDE 压到 0.10 需要 **≈ 620 个独立界面位点**，本 panel 拿不出来。**这是硬约束。**

#### 🚦 GATE G2 —— 是否花 Stage 3 的 1.6 GPU-h
- **GO**：pooled `ρ_id ≥ +0.15`，单边 sign-test p < 0.05，**且**超过 within-assay Xd–Xd null 的
  95 百分位 ≥ 0.05，**且**超过 mismatched-label null。
- **NO-GO / STOP**：pooled `ρ_id` 的 90% CI 完全落在 **[−0.08, +0.08]**（预注册的 equivalence bound）。
- **INCONCLUSIVE（预注册的第三种结局）**：落在 0.08–0.15 之间 → 报 effect size 与 MDE，
  **不宣称任何方向**，并说明需要多少个新的独立界面位点才能定论。

---

### Stage 3 — 1.6 GPU-h：KRAS ΔΔG 分解（只在 G2 = GO 时跑）

5 个结构 × arms {N, X, A, O} × 2522 个 complete-case 单突变 × 3 seeds ≈ 151k forwards ≈ **1.6 h**。

回答**唯一**一个问题：**partner 带来的那部分变化，追踪的是 binding 还是 folding？**
- 主统计量：`ρ(Δ̃, ΔΔG_bind,i)` vs `ρ(Δ̃, ΔΔG_fold)`，**common complete cases**，
  用 `*_std` 做 disattenuation，用 **Steiger/Williams dependent-correlation test** 直接比较。
  **不要**拿残差化 label 去和原始 label 比大小（残差化会移走两个 label 共享的可靠方差）。
- 对称残差化：同时报 `a_resid`（abundance 对 binding 的残差）。两个方向不对称即为 attenuation artifact。
- **O 臂**：错位 occluder 给出 occlusion null。若 Δ 在错误 occluder 上仍 ≈ +0.07，
  则该效应测的是遮挡不是信息。

**不跑：** 5×5 confusion matrix 的对角线检验 —— Stage 0 已证明该轴在 KRAS 家族**不可识别**
（几何 oracle 对角线 D ≈ 0，p ≥ 0.19）。作为 **bounded null** 引用，不消耗 GPU。

---

## 4. Confound register —— 每一条要么 designed out，要么声明为限制

| # | Confound | 处理 |
|---|---|---|
| 1 | `tied_featurize` 在 partner 变 fixed 后重排 chain（1HE8/1LFD 会静默填满且不报错） | 显式 chain 拼接 + 每 assay/arm 的 `_S_to_seq` assert |
| 2 | visibility 与 identity 混淆 | primary 改为 **N vs X**（同 chain_M、同 X、同 randn、同 L，只差 partner 的 S token） |
| 3 | 长度 / 解码顺序 / k=48 图随 partner 改变 | primary 完全不动几何；A 臂用 randn 子向量 + pin `residue_idx` |
| 4 | `mean|Δ|` 的 gate 几乎必然 fail | WT-centering + 用 SD 而非 mean |
| 5 | 符号约定未声明 | `s = −mean NLL`，support 方向 = 正，钉死 |
| 6 | scramble 的 OOD 冲击随 partner 长度放大 | 改用 X-token + 等长真实 decoy（Xd），K=8 draw 做 within-assay null |
| 7 | Δρ 的 attenuation-asymmetry 正 null | primary 用单个相关 `ρ(Δ̃, y)`（null 恰为 0） |
| 8 | burial/occlusion 可产生零信息量的 Δ | 几何完全不动 + O 臂 + contact-count oracle + mismatched-label null |
| 9 | variant bootstrap 低估 SE | position-clustered bootstrap |
| 10 | n=6 cluster 上 median+bootstrap 无 power | per-assay 表为 primary + exact sign test + 预注册 inconclusive 带 |
| 11 | +500 Å 平移与 delete 逐比特相同 | 删除该 arm |
| 12 | DIST 更高可能是高信号饱和而非零信号对照 | 报 levels；改用 hop-count 连续回归 |
| 13 | 「>12 Å」不是因果零（3-hop 传播覆盖 100% 位点） | 明确 UNREACH = **1-hop** 控制 |
| 14 | 残差化会 kill 信号 | disattenuation + Steiger/Williams + 对称残差化 + ceiling control |
| 15 | MAVEdb join 的编号偏移 | unit test，assert join > 20000 行 |
| 16 | fixed-chain 同时改 visibility / loss mask / normalizer | 只存一次 per-residue `log_probs`，离线导出；normalizer 固定为 L_mut |
| 17 | multiplicity（12+ quasi-primary） | 每 stage **一个** primary，其余 exploratory + FDR |

### 声明为限制（无法 designed out，必须写进结论）

1. **单 checkpoint**：`v_48_020` 的 null 不能推广到别的架构，也不说明「partner-aware 建模在原理上不可行」。
2. **同源模型**：8/9 panel 结构带 `_hm`，且 DMS 的 `wildtype_sequence` 会覆盖 PDB 序列
   （1HE8 差 192/941 = 79.6% identity；5WER 53/370；8BE4 35/475）—— **这几行的 N 臂已部分与 X 臂合流**，单独作为一行报告。
3. **无配体（已复核）**：5 个 KRAS PDB 的 **HETATM 全为 0**，没有 GTP/GDP/Mg²⁺；
   而 switch I/II 同时是部分 epitope 和核苷酸口袋。ProteinMPNN 只吃 N/CA/C/O，物理上无法修。
   ⇒ KRAS 的任何阴性结果是 **KRAS-specific 上界**，不可外推到「所有蛋白–蛋白界面」。
4. **对角线 / partner-identity 判别轴在 KRAS 上不可识别**：4/5 partner 的 exclusive epitope 为 0 个位点；
   几何 oracle 对角线 D ≈ 0（p ≥ 0.19）。回答它需要 epitope 互不重叠的家族。
5. **接触面占比不可比**：k=48 覆盖 PSD95 系统 40%、1LFD 19%、8BE4 8%（partner 占 48-NN slot 的比例，
   8BE4 最高 33/48）。必须作为显式协变量报告，行间比较需谨慎。
6. **Pseudo-replication**：6 个 KRAS assay = 1 个 cluster；6M0J/6M17 = 1 个 cluster。有效 n = 6。

---

## 5. 证据强度标注

**✅ 本人已复核：**
- §1.1 全部数字（混淆矩阵、ΔΔG vs fitness 的口径差异、几何 baseline）
- `protein_mpnn_utils.py:227` 的 `all_chains = masked_chains + visible_chains`，sort 在 225–226 被注释
- 5 个 KRAS 结构的 HETATM = 0
- `hgvs_pro` 构 key 与 `aa_seq` 100% 一致；6VJJ 需 shift −1
- BindingGYM 25 assay 的 partner 链清单（21 个可删，4 个 Z-domain 无 partner）
- decoding-order σ：per-assay median 0.0184 / max 0.0581

**⚠️ agent 声称、尚未复核**（跑之前应逐条验证）：
- `+500 Å 平移与 delete 逐比特相同（max|ΔS| = 0）`
- `scramble 效应与 partner 长度共线：77→0.029 / 941→0.179 / 475→0.314`
- `CORE/DIST 的 SD 比值 = 14.2`（mean|Δ| 比值仅 0.73 / 1.08）
- `6VJJ DIST +0.58 vs CORE +0.28；6M0J DIST +0.665 vs CORE +0.176`
- `position ICC = 0.595`，design effect 11.7
- `variant bootstrap 低估 SE 1.7–3.1×`
- `mismatched-label Γ = +0.073 ± 0.031`
- `配对 contrast 跨 seed 摆动 0.111`
- 同源模型的 identity 缺口（1HE8 79.6% 等）
- 吞吐 `94k variants/GPU-h`（GPU 预算全依赖这个数，**必须先实测**）

---

## 6. 数据与前置

```
本地  /home/guoj0f/share/BindingGYM/input/{BindingGYM.csv,Binding_substitutions_DMS,structures,msas}
      /tmp/claude-224072/kras_src/mavedb/          ← KRAS 真值(ΔΔG + fitness)，⚠️ 会被清理，先转出去
远端  10.67.24.41:/data/guoj0f/BindingGYM-zero-shot-proteinMPNN/scores/
        seed1_M5 = 25 assay 齐全；seed2–5_M5 = 各 10 个。列 design_score / global_score
代码  /home/guoj0f/repos/BindingGYM/baselines/protein_mpnn/
```

**开跑前两件事：** 把 MAVEdb 那份从 `/tmp` 转到持久位置；把 workstation 的 scores rsync 到本地。

## 7. 关联

- 同 repo `workstation-records/BindingGYM-zero-shot-proteinMPNN/zeroshot_proteinmpnn_20260827-154500.md`（0.3914 vs 官方 0.3970 的复现）
- 同上 `BindingGYM_KRAS_provenance_audit_20260828.md` §9（partner-blindness 的初版观察，**已被本文档 §1.1 修正**）
- ⚠️ `KRAS_DARPinK27_norfitness_5O2S` 的 label 是错的（装的是 SOS1 数据）。本文档 §1.1 用的是
  **MAVEdb 真值**，不受影响；但任何用 BindingGYM 原始 label 的分析都要先修。
