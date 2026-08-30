# complex-mutant-structure-pred — ESMFold2-Fast 折 BindingGYM mutant complex

（created 2026-08-30 00:33；updated 2026-08-30；status: **READY** —— 环境已就绪并校验，待启动）

## 1. 目标

用 `biohub/ESMFold2-Fast` 给 BindingGYM 的 variant 折 mutant complex 结构，回答两个问题：

- **(a)** mutant 的 backbone 相对 WT 变化大不大？**single-site 与 multi-site 是否不同？**
- **(b)** mut-vs-WT 的 backbone deviation 与 **within-assay 的 DMS score** 相关吗？
  （方向假设：deviation 越大 → 结合越差 → DMS score 越低；符号随各 assay 的 score 语义而定）

## 2. 三条决定设计的前提

**① WT 也必须自己折，不能用 BindingGYM 的晶体结构当对照。**
10 个入选 assay 里有 4 个的 WT 结构含未解析残基（`X`）：`CD19_FMC63_7URV` 52 个（10.5%）、
`4D5_HER2_1N8Z` 26 个、`5A12_VEGF_4ZFF` 8 个、`BH3_Mcl-1_3KZ0` 7 个。
拿"预测的 mutant"去比"不完整的晶体 WT"，测到的差异会混进**预测器与晶体的系统差**，而不是突变效应。
→ **WT 与 mutant 走完全相同的 pipeline、相同配置，只比彼此。**

**② seed 噪声可能盖过突变信号 —— 必须先量噪声底。**
实测（同一条 WT，只换 seed，全复合物 CA-RMSD）：

| assay | L | ipTM | seed0 vs seed1 | seed0 vs seed2 |
|---|--:|--:|--:|--:|
| `5A12_VEGF_4ZFF` | 528 | 0.70 | **11.59 Å** | **13.59 Å** |
| `SARS2-RBD_ACE2_6M0J` | 791 | 0.91 | 0.73 Å | 3.73 Å |

**ipTM 低的体系近乎随机。** 而文献反复测出 co-folding 对点突变高度不敏感 —— 若不先拿到逐 assay 的
噪声底，(a)(b) 两个结论都无法解读。→ **Stage-0 先折每个 assay 的 WT × 5 seeds。**

> **多 seed 在这里的用途是「量噪声」，不是「取平均」，也不是官方那条提精度的协议。**
> 论文的推荐是**多 sample/多 seed 按 ipTM 挑 top-1**（Fig 2D：横轴是 seed 预算，扫到 1024 seeds；
> Fast 版 AbAg 45%→65%）—— 是**选优**，不是平均；3D 坐标处在各自的刚体坐标系里，本来也没法有意义地平均。
> 我们跑 WT × 5 seeds 是为了拿到**该 assay 的 seed-to-seed 差异分布**（5 seeds = 10 个两两配对），
> 作为解读 mutant-vs-WT 差异的基线。**成本 2 分钟**（3 seeds 也够用，省下的是 1 分钟，没必要省）。
>
> ⚠️ **由此带出一个未决**：mutant 每条只跑 1 seed，所以「mutant(seed0) vs WT(seed0)」的差异里**含一份完整
> seed 噪声**。三条出路：① 统计上减去噪声底（最省，本计划默认）；② mutant 也用
> `num_diffusion_samples=5` + ipTM top-1 压噪声（+21%，**未实测**，Stage-0 顺带量）；
> ③ mutant 也跑多 seed（**精确 ×N 成本**，最后才考虑）。

**③ 已有的 2,080 个 FoldX 结构帮不上忙。**
FoldX 是 rigid-backbone（`BuildModel` 只重排侧链），产出的 mutant 与 WT **backbone 逐原子相同**，
对 ProteinMPNN / ESM-IF1 这类 backbone-only scorer 在构造上不可见。这正是要用 ESMFold2 的原因。
（唯一例外用途：`4D5_HER2_1N8Z` 上可做 **ESMFold2 vs FoldX vs WT** 三方对照。）

## 3. 要跑的 5 个 assay

按「目标 (a) 的 single/multi 受控对照 + k 跨度」与「目标 (b) 的样本量 + DMS 动态范围」选，
兼顾 L 谱与界面类型。**每 assay 分层抽样 ≤1,500**（within-assay Spearman 的 SE ≈ 0.026，够用）。

| # | assay | L | n | single | multi | k_max | DMS sd | s/个 | 折数 | A100-h | 入选理由 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 1 | `SARS2-RBD_ACE2_deltaKd_6M0J` | 791 | 21,872 | 3,134 | 18,737 | **10** | 1.86 | 12.3 | 1,500 | 5.1 | 目标(a)最强：k=1..10 连续，single/multi 都上千 |
| 2 | `CD19_FMC63_Fitness_7URV` | 497 | 3,886 | 1,618 | 2,267 | 3 | **4.97** | 5.3 | 1,500 | 2.2 | single/multi 最均衡；DMS 动态范围全库最大 |
| 3 | `hYAP65_peptide_1JMQ` | **56** | 18,407 | 288 | 18,118 | **21** | 1.15 | 1.5 | 1,500 | 0.6 | k 跨度最大、最便宜，最快见到答案 |
| 4 | `CXCR4_CXCL12_enrich_8U4O` | 360 | 5,585 | 5,584 | 0 | 1 | 1.43 | 3.4 | 1,500 | 1.4 | **99.98% 纯单点** —— 单点到底动不动 backbone 的直接检验 |
| 5 | `GB1_IgG-Fc_fitness_1FCC` | 262 | 92,891 | 1,045 | 91,845 | 2 | 1.01 | 2.5 | 1,500 | 1.0 | 全库最大 assay，抽样池最深 |

**这 5 个的覆盖**：k = 1 / 1–2 / 1–3 / 1–10 / 1–21（跨度完整）；L = 56 / 262 / 360 / 497 / 791；
界面类型 = 短肽-结构域 / 天然 PPI / 趋化因子-受体 / 抗体-抗原 / 天然 PPI。

**刻意排除的（有信号再加回来，按此优先级）**

| assay | 取舍 |
|---|---|
| `4D5_HER2_1N8Z` | **第一个该加回来的** —— 唯一有 2,080 个 FoldX 结构可做三方对照。但 L=1041 要 9.1 h（几乎翻倍预算），且 **0 个 single**，对目标 (a) 的核心对照无贡献 |
| `KRAS_RAF1_6VJJ` / `KRAS_RALGDS_1LFD` | 便宜（各 1.0 h）、可交叉验证，但 k_max 仅 2、sd ~0.5，信息量与已选的重叠 |
| `KRAS_DARPinK27_5O2S` | **数据污染** —— 装的是 SOS1 的数据（根因在 Nature 补充表），已定案，永久排除 |
| `KRAS_PICK3CG_1HE8` | 最贵（10.4 h）但 k_max=2、sd 0.499，且 **chain A 有 192 个 `X`（17%）** |
| `5A12_Ang2_4ZFG` | DMS sd 仅 **0.079**，没有动态范围，做不了目标 (b) |

## 4. 成本

**t(L) 全区间实测**（`esmfold2` env，torch 2.11.0+cu128，`kernel=fused`，`10 loops / 68 steps / 1 sample`，A100-80GB）：

| L | 109 | 245 | 262 | 528 | 791 | 931 | 1041 | 1107 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| s/结构 | 1.89 | 2.68 | 2.08 | 5.30 | 12.74 | 17.16 | 21.91 | 24.86 |
| 峰值显存 GB | 12.98 | 13.37 | 13.42 | 15.51 | 19.00 | 21.42 | 23.60 | 25.02 |

拟合 `t(L) = 1.499 + 1.237e-5·L² + 6.147e-9·L³`（8 点最小二乘，残差 ±0.55 s）—— **无外推**。

| | 结构数 | A100-h |
|---|--:|--:|
| mutant（5 assay，每个 ≤1,500） | 7,500 | **10.4** |
| WT × 5 seeds（噪声底） | 25 | 0.04（2 分钟） |
| **合计** | **7,525** | **10.4** ≈ **单卡 10 小时** |
| 若 mutant 也用 `samples=5` + ipTM top-1 | 7,525 | 12.6（+21% **未实测**） |
| 参考：这 5 个不抽样跑全量 | 142,641 | 157（单卡 6.5 天） |

显存最高 19.0 GB（L=791），A100-80GB 宽裕。⚠️ 这是**共享卡** —— 他人占 48.9 GiB 时长序列曾 OOM，
起跑前重做 GPU 预检，必要时 `model.set_chunk_size(32)`。

## 5. 配置与路径

| 项 | 值 |
|---|---|
| env | `esmfold2` @ workstation（`conda activate esmfold2`），SOP §2.3 |
| 模型 | `biohub/ESMFold2-Fast` |
| 采样 | `num_loops=10, num_sampling_steps=68, num_diffusion_samples=1`（SOP §9.4）。⚠️ `fold()` 默认是 20/200，必须显式传 |
| kernel | `model.set_kernel_backend("fused")` —— L>400 收益 4.5–7.7×，数值偏差远小于 seed 噪声（SOP §4.2） |
| 抽样 | 每 assay 按突变点数 k **分层**抽 ≤1,500，保证各 k 档都有样本 |
| 输出 | `/data/guoj0f/share/our-predicted-structure/ProteinTTT-bindingGYM-mutation-structure-analysis/complex-mutant-structure-pred/`<br>├ `BindingGYM-esmfold2-fast-predicted-wt-complex-structure/`<br>└ `BindingGYM-esmfold2-fast-predicted-mutant-complex-structure/` |

**未定**：`num_diffusion_samples=5` + ipTM top-1 能压 seed 噪声，成本模型估 +21% 但**从未实测**。
Stage-0 顺带测一次，再决定 mutant 用不用。

## 6. 执行顺序（可随时终止，越早的越快出结果）

| Stage | 内容 | A100-h | 产出 |
|---|---|--:|---|
| **0** | 5 个 WT × 5 seeds；顺带实测 `samples=5` 的真实开销 | 0.04 | **逐 assay 的 ipTM 与 seed 噪声底** —— 决定哪些 assay 的结论可信 |
| **1** | #3 hYAP65 → #5 GB1 → #4 CXCR4 | 3.0 | 短 L 快速见效；先看有没有信号 |
| **2** | #2 CD19 | 2.2 | single/multi 最均衡 + DMS 动态范围最大 |
| **3** | #1 SARS2-RBD | 5.1 | 目标(a)最强证据（k=1..10） |

## 7. 分析口径（跑完之后）

- **backbone deviation 必须分两层**：链**内** fold 变化 vs 链**间** docking pose 变化。
  全复合物 CA-RMSD 会被后者主导（§2 那 11.6 Å 大概率就是 pose 漂移）。
  `res.complex` 自带官方 `.rmsd()` / `.lddt_ca()` / `.dockq()`，逐链算 + 算界面指标。
- **任何 deviation 都要先减去该 assay 的 seed 噪声底**，否则测的是噪声。
- 目标 (b) 用 **within-assay Spearman**（deviation vs DMS score），不跨 assay 合并。

## 关联

- [`predicted_complex_structure_assets_20260829-000600.md`](predicted_complex_structure_assets_20260829-000600.md) —— 现成 mutant complex 资产只覆盖 0.55%
- [`structure_pseudolabel_methods_survey_20260829-231404.md`](structure_pseudolabel_methods_survey_20260829-231404.md) —— FoldX 不动 backbone；co-folding 对点突变不敏感的证据
- SOP：`repos/Sources/datasets/protein-structure-prediction/esmfold2-structure-prediction-sop.md`
