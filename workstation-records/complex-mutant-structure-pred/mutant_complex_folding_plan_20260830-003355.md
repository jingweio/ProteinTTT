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

**③ 已有的 2,080 个 FoldX 结构帮不上忙。**
FoldX 是 rigid-backbone（`BuildModel` 只重排侧链），产出的 mutant 与 WT **backbone 逐原子相同**，
对 ProteinMPNN / ESM-IF1 这类 backbone-only scorer 在构造上不可见。这正是要用 ESMFold2 的原因。
（唯一例外用途：`4D5_HER2_1N8Z` 上可做 **ESMFold2 vs FoldX vs WT** 三方对照。）

## 3. 要跑的 10 个 assay

按「目标 (a) 的 single/multi 受控对照 + k 跨度」与「目标 (b) 的样本量 + DMS 动态范围」选，
兼顾 L 谱与界面类型覆盖。**每 assay 分层抽样 ≤1,500**（within-assay Spearman 的 SE ≈ 0.026，够用）。

| # | assay | L | n | single | k_max | DMS sd | s/个 | 折数 | A100-h | 入选理由 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 1 | `SARS2-RBD_ACE2_deltaKd_6M0J` | 791 | 21,872 | 3,134 | **10** | 1.86 | 12.3 | 1,500 | 5.1 | 目标(a)最佳：k=1..10 连续，single/multi 都上千 |
| 2 | `CD19_FMC63_Fitness_7URV` | 497 | 3,886 | 1,618 | 3 | **4.97** | 5.3 | 1,500 | 2.2 | single/multi 最均衡；DMS 动态范围全库最大 |
| 3 | `hYAP65_peptide_1JMQ` | **56** | 18,407 | 288 | **21** | 1.15 | 1.5 | 1,500 | 0.6 | k 跨度最大、最便宜，最快见到答案 |
| 4 | `GB1_IgG-Fc_fitness_1FCC` | 262 | 92,891 | 1,045 | 2 | 1.01 | 2.5 | 1,500 | 1.0 | 全库最大 assay，中位 L |
| 5 | `KRAS_RAF1_norfitness_6VJJ` | 245 | 12,677 | 1,188 | 2 | 0.51 | 2.3 | 1,500 | 1.0 | 天然 PPI，single-vs-double 干净对照 |
| 6 | `CXCR4_CXCL12_enrich_8U4O` | 360 | 5,585 | 5,584 | 1 | 1.43 | 3.4 | 1,500 | 1.4 | **99.98% 纯单点** —— 单点到底动不动 backbone 的直接检验 |
| 7 | `5A12_VEGF_fitness_4ZFF` | 528 | 29,981 | 54 | 9 | 0.96 | 5.9 | 1,500 | 2.4 | 唯一入选的 antibody-antigen，3 链 |
| 8 | `4D5_HER2_fitness_1N8Z` | 1041 | 2,080 | 0 | 9 | 1.19 | 21.8 | 1,500 | 9.1 | 唯一有 FoldX 结构可做三方对照 |
| 9 | `KRAS_RALGDS-RBD_1LFD` | 254 | 20,341 | 2,544 | 2 | 0.45 | 2.4 | 1,500 | 1.0 | 与 #5 交叉验证：同蛋白不同 partner |
| 10 | `BH3_Mcl-1_normed_3KZ0` | 173 | 518 | 170 | 5 | 0.49 | 1.9 | 518 | 0.3 | 小而均衡，n=518 可跑全量 |

**刻意排除的三个**

| assay | 原因 |
|---|---|
| `KRAS_DARPinK27_5O2S` | **数据污染** —— 装的是 SOS1 的数据（根因在 Nature 补充表），已定案 |
| `KRAS_PICK3CG-RBD_1HE8` | L=1107 最贵（10.4 h，28% 预算），但 k_max 仅 2、sd 0.499；且 **chain A 有 192 个 `X`（17%）** |
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
| mutant（10 assay，每个 ≤1,500） | 14,018 | **24.2** |
| WT × 5 seeds（噪声底） | 50 | 0.08 |
| **合计** | **14,068** | **24.3** ≈ **单卡 1 天** |
| 参考：这 10 个不抽样跑全量 | 208,238 | 240（单卡 10 天） |

显存最高 25.0 GB（L=1107），A100-80GB 宽裕。⚠️ 但这是**共享卡** —— 他人占 48.9 GiB 时 L=1107 曾 OOM，
长 L 的 assay 起跑前重做 GPU 预检，必要时 `model.set_chunk_size(32)`。

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
| **0** | 10 个 WT × 5 seeds；顺带测 `samples=5` 的真实开销 | ~0.2 | **逐 assay 的 ipTM 与 seed 噪声底** —— 决定哪些 assay 的结论可信 |
| **1** | #3 hYAP65 → #5 KRAS_RAF1 → #4 GB1 → #9 KRAS_RALGDS → #10 BH3 | 3.9 | 短 L 快速见效；先看有没有信号 |
| **2** | #6 CXCR4 → #2 CD19 → #7 5A12_VEGF | 6.0 | 单点专项 + 最均衡对照 + antibody-antigen |
| **3** | #1 SARS2-RBD → #8 4D5_HER2 | 14.2 | 目标(a)最强证据 + FoldX 三方对照 |

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
