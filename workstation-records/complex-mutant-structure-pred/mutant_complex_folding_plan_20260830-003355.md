# complex-mutant-structure-pred — ESMFold2-Fast 折 BindingGYM mutant complex

（created 2026-08-30 00:33；updated 2026-08-30；status: **RUNNING** —— Stage-0 已完成，主任务运行中）

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

## 3. 要跑的 3 个 assay（每个抽样 2,000）

原选 5 个，**Stage-0 的噪声底数据剔除了其中 2 个**（见 §4）。最终：

| # | assay | L | n | 抽样 | 其中单点 | k_max | WT ipTM | **噪声底** | s/个 | A100-h |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | `SARS2-RBD_ACE2_6M0J` | 791 | 21,872 | 2,000 | 300 | **10** | **0.91** | **2.58 Å** | 12.5 | **6.94** |
| 2 | `CXCR4_CXCL12_8U4O` | 360 | 5,585 | 2,001 | **2,000** | 1 | 0.70–0.81 | **1.10 Å** | 4.6 | 2.56 |
| 3 | `hYAP65_1JMQ` | 56 | 18,407 | 2,000 | 288（全部） | **21** | 0.40 | 2.60 Å | 1.85 | 1.03 |
| | **合计** | | | **6,001** | **2,588** | | | | | **10.5** |

抽样按 k 分层，**k=1 档保底 300**（目标 (a) 的核心对照）。`CXCR4` 是纯单点 assay，2,000 全是单点；
`hYAP65` 的单点源数据只有 288 条，已全取。

**剔除的两个（Stage-0 实测，WT 结构已留存作为证据）**

| assay | WT ipTM | 噪声底 | 剔除理由 |
|---|--:|--:|---|
| `GB1_IgG-Fc_1FCC` | **0.15–0.18** | 5.69 Å | 界面预测失败。**链内 RMSD 仅 0.38 Å 但全复合物 5.69 Å** —— 两条链各自折得很稳，相对摆放却是随机的 |
| `CD19_FMC63_7URV` | 0.51–0.71 | 5.32 Å | ipTM 波动大，且**逐链就有 6.08 Å**，连链内 fold 都不稳 |

## 4. Stage-0 结果（2026-08-30，已完成）

5 个 assay × 5 个 WT seed = 25 条，**30/30 成功，0 失败**，pipeline 全链路验证通过。

**seed 噪声底**（同一条 WT 换 seed 的两两 CA-RMSD，Kabsch 对齐）：

| assay | L | WT ipTM | 全复合物（中位/min/max） | 逐链最大（链内 fold） |
|---|--:|--:|--:|--:|
| `CXCR4_CXCL12_8U4O` | 360 | 0.70–0.81 | **1.10** / 0.77 / 2.98 | 1.69 |
| `SARS2-RBD_ACE2_6M0J` | 791 | 0.91 | **2.58** / 0.66 / 3.73 | 2.37 |
| `hYAP65_1JMQ` | 56 | 0.40 | **2.60** / 1.71 / 3.09 | 2.60 |
| ~~`CD19_FMC63_7URV`~~ | 497 | 0.51–0.71 | 5.32 / 3.70 / 11.97 | 6.08 |
| ~~`GB1_IgG-Fc_1FCC`~~ | 262 | 0.15–0.18 | 5.69 / 1.10 / 8.47 | **0.38** |

> **ipTM 与噪声底强相关** —— ipTM 越低，seed 之间越随机。`GB1` 是最清楚的例子：
> 链内 0.38 Å（每条链都折得很好）但全复合物 5.69 Å，说明**噪声全在链间 docking pose**。
> 这也证实了 §7 那条口径：**全复合物 RMSD 会被 pose 主导，必须逐链分开算。**

**读结果时的硬约束**：三个入选 assay 里只有 `SARS2-RBD`（ipTM 0.91）算结构可信。
`hYAP65` 的噪声底 2.60 Å 对 L=56 的体系是相当大的比例 —— **若它给出「无差异」，可能只是噪声底本来就高**，不能直接当阴性证据。

## 4b. 成本（实测）

**t(L)**（`esmfold2` env / torch 2.11.0+cu128 / `kernel=fused` / `10 loops / 68 steps / 1 sample` / A100-80GB）：

| L | 56 | 109 | 245 | 262 | 360 | 497 | 528 | 791 | 931 | 1041 | 1107 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| s/结构 | 1.85 | 1.89 | 2.68 | 2.08 | 4.6 | 8.7 | 5.30 | **12.5** | 17.16 | 21.91 | 24.86 |

⚠ 首条要多付 ~2.5 s 的 Triton autotune（L=791 实测 15.05 s → 之后稳定 12.41–12.52 s）。
⚠ Stage-0 期间曾观测到 L=791 达 22 s，隔离复测为 12.5 s —— 判定为当时的**瞬时干扰**（共享卡），稳态取 12.5 s。

| | 结构数 | A100-h |
|---|--:|--:|
| mutant（3 assay × 2,000） | 6,001 | **10.5** |
| WT × 5 seeds | 15 | 已完成（Stage-0） |
| **合计** | 6,016 | **≈ 10.5 小时** |

## 4c. 模型置信度记录（全部 6,026 个结构，2026-08-31）

来源 `results/fold_logs/metrics.tsv`（每条一行：`ptm / iptm / plddt_mean / seconds`）。
⚠ **ESMFold2 的三个指标都是 0–1 尺度**，与 AF2 PDB 的 B-factor（0–100）不可直接比大小。

### 4c-1 WT × 5 seeds：三个指标的 seed 间波动

| assay | pLDDT (min/中位/max) | pTM | **ipTM** | ipTM 极差 |
|---|---|---|---|--:|
| `SARS2-RBD_ACE2_6M0J` | 0.8999 / **0.9007** / 0.9018 | 0.8625 / 0.8918 / 0.8954 | 0.9054 / **0.9108** / 0.9149 | **0.0095** |
| `CXCR4_CXCL12_8U4O` | 0.7775 / **0.7848** / 0.8190 | 0.8051 / 0.8096 / 0.8530 | 0.7006 / **0.7170** / 0.8061 | 0.1055 |
| `hYAP65_1JMQ` | 0.6611 / **0.6618** / 0.6697 | 0.4786 / 0.4905 / 0.4960 | 0.3910 / **0.4066** / 0.4151 | 0.0241 |
| ~~`CD19_FMC63_7URV`~~ | 0.7261 / 0.7390 / 0.7475 | 0.6369 / 0.6559 / 0.7069 | 0.5073 / 0.5781 / 0.7069 | **0.1996** |
| ~~`GB1_IgG-Fc_1FCC`~~ | 0.8387 / 0.8456 / 0.8497 | 0.7004 / 0.7035 / 0.7063 | 0.1480 / **0.1831** / 0.1848 | 0.0368 |

> **pLDDT / pTM 稳，ipTM 不稳。** 三个指标里只有 ipTM 的 seed 波动大（`CD19` 极差 0.1996、
> `CXCR4` 0.1055），而 pLDDT 极差最大只有 0.0415。**换 seed 主要改变的是"界面"而不是"折叠"** ——
> 与 §4 的 RMSD 结论（`GB1` 链内 0.38 Å、全复合物 5.69 Å）完全一致。
> → **筛质量必须看 ipTM，且必须知道它自身的 seed 波动有多大**；只看 pLDDT 会漏掉界面失败
> （`GB1` pLDDT 0.85 看着很好，ipTM 只有 0.18）。

### 4c-2 mutant（6,001 条）的置信度分布，与 WT 中位对比

| assay | 指标 | n | p5 | 中位 | p95 | WT 中位 | Δ(mut−WT) |
|---|---|--:|--:|--:|--:|--:|--:|
| `SARS2-RBD` | pLDDT | 2,000 | **0.8685** | 0.8973 | 0.9007 | 0.9007 | −0.0034 |
| | pTM | 2,000 | **0.7597** | 0.8906 | 0.8935 | 0.8918 | −0.0012 |
| | **ipTM** | 2,000 | **0.2422** | 0.9006 | 0.9113 | 0.9108 | **−0.0102** |
| `CXCR4` | pLDDT | 2,001 | 0.7711 | 0.7871 | 0.8172 | 0.7848 | +0.0023 |
| | pTM | 2,001 | 0.8013 | 0.8155 | 0.8513 | 0.8096 | +0.0059 |
| | **ipTM** | 2,001 | 0.7068 | 0.7294 | 0.8060 | 0.7170 | **+0.0124** |
| `hYAP65` | pLDDT | 2,000 | 0.6103 | 0.6611 | 0.6734 | 0.6618 | −0.0007 |
| | pTM | 2,000 | 0.4260 | 0.4857 | 0.5027 | 0.4905 | −0.0048 |
| | **ipTM** | 2,000 | 0.3244 | 0.3992 | 0.4154 | 0.4066 | **−0.0075** |

> 🔴 **形态不是"整体平移"，而是"中位几乎不动 + 一条向下的长尾"。**
> 最清楚的是 `SARS2-RBD` 的 ipTM：中位 0.9006（与 WT 的 0.9108 差 0.01，落在 seed 波动量级内），
> 但 **p5 只有 0.2422** —— 即约 5% 的 mutant 让界面预测彻底崩掉。
> → **不能用"中位没变"就断言"突变无影响"**；要看的是尾部，以及尾部里是哪些 variant。

### 4c-3 ⭐ ipTM 随突变点数 k 单调下降（对目标 (a) 是正面信号）

| assay | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `SARS2-RBD` | **0.909**<br>(n=300) | 0.904<br>(709) | 0.898<br>(513) | 0.891<br>(260) | 0.884<br>(113) | 0.881<br>(38) | **0.819**<br>(30) | **0.799**<br>(30) |
| `hYAP65` | **0.401**<br>(288) | 0.400<br>(717) | 0.400<br>(717) | 0.396<br>(151) | 0.390<br>(30) | 0.377<br>(30) | — | — |
| `CXCR4` | 0.729<br>(2000) | 纯单点 assay，无 k 梯度 | | | | | | |

> **两个有 k 梯度的 assay 都单调下降，且 `SARS2-RBD` 的降幅（0.909 → 0.799，跨 0.11）
> 远超它自己的 WT seed 波动（0.0095，约 12×）。** 这是本次跑出来的第一个**超出噪声底的信号**：
> **突变点数越多，模型对界面越不确定。**
> ⚠ 但这是 **ipTM（模型自评置信度）**，不是 backbone 几何。它**不能直接回答目标 (a)** ——
> 目标 (a) 问的是"backbone 变了多少"，要等 RMSD 算完。两者可能一致，也可能 ipTM 只是反映
> "序列越偏离训练分布、模型越没底"。

### 4c-4 mutant 落在 WT seed 区间内的比例

| assay | pLDDT | pTM | ipTM |
|---|--:|--:|--:|
| `SARS2-RBD` | 18.9% | 85.5% | 35.4% |
| `CXCR4` | 83.0% | 86.6% | 91.9% |
| `hYAP65` | 37.4% | 50.6% | 69.8% |

> ⚠ **这张表容易误读，别拿它当显著性检验。** 比例低不等于"差异显著" —— 分母是**5 个 seed 撑出来的
> 区间**，样本量 5 的极差本来就严重低估真实分布宽度。`SARS2-RBD` 的 pLDDT 只有 18.9% 落在区间内，
> 纯粹是因为它的 WT 区间只有 0.0019 宽。要做显著性判断，得用 §7 的口径（逐链 RMSD 减噪声底），
> 而不是这个比例。

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

## 6. 执行

一条 `nohup` 长任务，按 assay 单条成本升序（便宜的先跑完，随时终止都有完整 assay）：
`hYAP65`（1.0 h）→ `CXCR4`（2.6 h）→ `SARS2-RBD`（6.9 h）。

脚本 `sh/fold_complexes.py` 保留 SOP §6.2 的四条设计：**原子写**（`.tmp` → `os.replace`）、
**skip-if-exists 续跑**、**逐条 try/except**（写 `failures.tsv`）、**确定性顺序**。
崩了或被挤掉，原样重启即可继续。

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
