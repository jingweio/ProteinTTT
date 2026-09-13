# structure-TTT：从 **encoder** 侧提升 ProteinMPNN 的 BindingGYM zero-shot 能力

**写于 2026-09-13。** 本文档记录的是**另一条分支上已经做完的相关工作与已知事实**，
供 structure-TTT 方向参考。它不包含实验规划。

---

## 0. 问题

给定某个 assay 的 WT complex structure，能不能在 test time **只用结构本身**（零 DMS label）
把 ProteinMPNN 的 **encoder** 对这个复合物、特别是对 binding-sites 的**细粒度几何**编码得更好，
从而提升它在 BindingGYM 上的 zero-shot DMS-score 预测？

**效果预期**：只要**优于预训练 ProteinMPNN 的 zero-shot**即可。
这一侧的定位是**学一个更好的基础表征**，为后续的 mutation-landscape-TTT 提供更好的起点，
不需要直接去比那些更强的后处理方法。

---

## 1. 评测口径

### 1.1 基线与指标

- 指标：**per-assay Spearman（预测分数 vs DMS_score），再对 assay 取未加权平均**。
- **预训练 ProteinMPNN zero-shot 基线**：**23-assay `0.3939`**、**14-assay `0.3903`**。
- 打分口径（官方 `global_score`）：把 variant 的突变序列做 autoregressive teacher-forced，
  在 mask 上**求和** NLL 再取负，**无 WT 项、无长度归一化**，对 **M=5** 个解码顺序取均值。
  ⚠️ 因为是 NLL 求和且不做长度归一化，**跨 assay 的绝对分数不可比**（尺度差 3.3×）；
  只有 assay 内部的排序有意义。

### 1.2 binding-sites 的定义

**不同定义会切出不同的结果，所以先把定义写死。** 全项目统一使用下面这一个：

1. **先把复合物的链分成两个 binding entity**，**依据是元数据、不是结构启发式** ——
   `DMS_id` 里已经写明了两个结合方（如 `4D5_HER2` = 抗体 4D5 对 HER2），
   据此把链分到 `entity1` / `entity2`。
   完整对照表：`/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-analysis/data/entity_partition.csv`
   （含 `provenance` 列，标明每一条的来源）。
   ⚠️ **刻意不用「按接触面自动划分」这类启发式**，因为它在多链复合物上会误判。
2. **残基级判据**：设残基 `r` 属于 entity `E`，则
   ```
   d(r) = min { ‖x_a − x_b‖₂ :  a ∈ 重原子(r),  b ∈ 重原子(另一个 entity 的全部残基) }
   r ∈ binding-sites   ⟺   d(r) ≤ 5.0 Å
   ```
   - **重原子**（含侧链），不是只看 CA；**排除氢**。
   - 距离是到**另一个 entity**，**不是到「任何其它链」** ——
     这个区别很重要：抗体的 VH–VL 之间、以及同一 entity 内部的链间接触，**不算界面**。
     （早期版本用过「任何其它链」的定义，会把 VH–VL packing 误计为界面，已废弃。）
3. **变体级判据**（供逐 variant 的分析用）：变体 `v` 的所有突变位点里，
   只要有一个落在 binding-sites 上，就算「碰界面」。
   连续版本则取 `d(v) = min_i d(p_i)`（或对各位点的 `f(d_i)` 做聚合）。

**逐残基距离的产物**（23+ assay 全链，含 `d(r)` 与 `is_interface_5A`）：
`/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-analysis/data/interface_residues_all_chains.csv`

> **cutoff 的敏感性**已单独验证过（4.0 / 4.5 / 5.0 / 6.0 / 8.0 Å 都算了），
> 5.0 Å 是常用值，结论对它不敏感；各 cutoff 的结果见
> `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-analysis/data/variant_labels.parquet` 的 `iface_dist_*` 列。

### 1.3 25 / 23 / 14 分别是哪些 assay

**三套嵌套的 assay 集合**，报任何数字都必须说明用的是哪一套：

- **25** = BindingGYM 官方全集。
- **23** = 25 减去 **2 个 label 被污染的 assay**：`KRAS_DARPinK27_norfitness_5O2S`、
  `KRAS_SOS1_norfitness_8BE4`（根因在 Nature 补充表，不在 BindingGYM 本身；**一律排除**）。
- **14** = 23 中**「碰界面 / 不碰界面」两组都有足够样本（各 ≥ 30）**的那些。
  其余 9 个是**全碰或全不碰**，二值指示量恒定 ⇒ 任何基于二值界面标签的分析在它们上恒为 0，无法评。
  ⚠️ **连续距离在 23 个上都有变化**，所以只有**需要二值切分**的分析才受限于 14。

| # | DMS_id | n variants | 25 | 23 | 14 | 碰界面比例 |
|---:|---|---:|:--:|:--:|:--:|---:|
| 1 | `4D5_HER2_fitness_1N8Z` | 2,079 | ✅ | ✅ | — | 0.999 |
| 2 | `5A12_Ang2_fitness_4ZFG` | 943 | ✅ | ✅ | ✅ | 0.869 |
| 3 | `5A12_VEGF_fitness_4ZFF` | 29,980 | ✅ | ✅ | — | 0.000 |
| 4 | `ACE2_SARS2-RBD_enrich_6M17` | 2,185 | ✅ | ✅ | ✅ | 0.174 |
| 5 | `BH3_Bcl-xL_normed_1PQ1` | 517 | ✅ | ✅ | — | 1.000 |
| 6 | `BH3_Mcl-1_normed_3KZ0` | 517 | ✅ | ✅ | — | 1.000 |
| 7 | `CD19_FMC63_Fitness_7URV` | 3,885 | ✅ | ✅ | ✅ | 0.035 |
| 8 | `CXCR4_CXCL12_enrich_8U4O` | 5,584 | ✅ | ✅ | ✅ | 0.126 |
| 9 | `GB1_IgG-Fc_fitness_1FCC` | 92,890 | ✅ | ✅ | ✅ | 0.630 |
| 10 | `GB1_IgG-Fc_fitness_1FCC_2016` | 22,175 | ✅ | ✅ | — | 1.000 |
| 11 | `HLA-A2_TAPBPR_meanscore_5WER` | 3,344 | ✅ | ✅ | ✅ | 0.123 |
| 12 | `KRAS_DARPinK27_norfitness_5O2S` | 19,532 | ✅ | ❌ | — | 0.459 |
| 13 | `KRAS_PICK3CG-RBD_norfitness_1HE8` | 19,202 | ✅ | ✅ | ✅ | 0.203 |
| 14 | `KRAS_RAF1-RBD_norfitness_6VJJ` | 23,161 | ✅ | ✅ | ✅ | 0.257 |
| 15 | `KRAS_RAF1_norfitness_6VJJ` | 12,676 | ✅ | ✅ | ✅ | 0.390 |
| 16 | `KRAS_RALGDS-RBD_norfitness_1LFD` | 20,340 | ✅ | ✅ | ✅ | 0.272 |
| 17 | `KRAS_SOS1_norfitness_8BE4` | 19,424 | ✅ | ❌ | — | 0.512 |
| 18 | `PSD95_CRIPT_1BE9` | 1,576 | ✅ | ✅ | ✅ | 0.181 |
| 19 | `PSD95_Tm2F_1BE9` | 1,576 | ✅ | ✅ | ✅ | 0.193 |
| 20 | `SARS2-RBD_ACE2_deltaKd_6M0J` | 21,871 | ✅ | ✅ | ✅ | 0.309 |
| 21 | `Z-domain_ZSPA-1_LL1_fitness_1LP1` | 45,476 | ✅ | ✅ | — | 1.000 |
| 22 | `Z-domain_ZSPA-1_LL2_fitness_1LP1` | 5,583 | ✅ | ✅ | — | 1.000 |
| 23 | `Z-domain_ZpA963_HL1_fitness_2M5A` | 2,903 | ✅ | ✅ | — | 1.000 |
| 24 | `Z-domain_ZpA963_HL2_fitness_2M5A` | 599 | ✅ | ✅ | — | 1.000 |
| 25 | `hYAP65_peptide_FunctioncalScore_1JMQ` | 18,406 | ✅ | ✅ | ✅ | 0.527 |

**被排除出 14 的那 9 个，及其原因（碰 / 不碰 的样本数）：**

| DMS_id | 碰界面 | 不碰界面 |
|---|---:|---:|
| `4D5_HER2_fitness_1N8Z` | 2,076 | **3** |
| `5A12_VEGF_fitness_4ZFF` | **0** | 29,980 |
| `BH3_Bcl-xL_normed_1PQ1` | 517 | **0** |
| `BH3_Mcl-1_normed_3KZ0` | 517 | **0** |
| `GB1_IgG-Fc_fitness_1FCC_2016` | 22,175 | **0** |
| `Z-domain_ZSPA-1_LL1_fitness_1LP1` | 45,476 | **0** |
| `Z-domain_ZSPA-1_LL2_fitness_1LP1` | 5,583 | **0** |
| `Z-domain_ZpA963_HL1_fitness_2M5A` | 2,903 | **0** |
| `Z-domain_ZpA963_HL2_fitness_2M5A` | 599 | **0** |

> **这三套集合的完整推导与逐 assay 明细**（含每个 assay 的界面残基数、变体数、基线 Spearman）：
> `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-overview/BindingGYM_binding_sites_overview_20260903.md`
> 生成脚本：`/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/scripts/binding_sites_overview/o5_basis23.py`、
> `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/scripts/mutation_landscape_ttt/g1e_canonical14.py`

---

## 2. 已知事实：encoder 对界面的编码 —— 二值层面好，细粒度弱

### 2.1 标签是怎么建的（与 §1.2 **完全同一个定义**）

探针要的是**逐残基**标签，索引必须与 encoder 输出 `h_V` 的行一一对应。

`h_V` 的第 `i` 行对应 **ProteinMPNN 内部打包顺序**下的第 `i` 个残基：
`tied_featurize` 按「designed chains 在前、其余在后」拼接，每条链按 **PDB 残基顺序**排列。
所以标签**直接按 PDB 残基顺序在打包空间里算**即可 —— **完全不涉及 WT 序列**，
也就没有 WT↔PDB 对齐的问题。用的就是 §1.2 的重原子 5 Å 定义与元数据 entity 划分。

**但有两处必须排除，否则会静默算错：**

1. **缺口填充位**：官方 `parse_PDB` 是按 `range(min_resn, max_resn+1)` 遍历的，
   **晶体学缺失的残基号会被补成一个 `X` 残基、坐标为 NaN**；
   随后 `tied_featurize` 执行 `X[isnan] = 0.`，于是这些位置**带着坐标 (0,0,0) 进入 encoder**，
   同时 `mask` 被置 0。
   ⚠️ **它们在 `h_V` 里占着行，但不是真实残基** —— 若不排除，会被当作「离所有东西都很远」的残基。
   实测占比：`KRAS_PICK3CG-RBD` **17.3%**、`CD19_FMC63` 10.5%、`HLA-A2_TAPBPR` 9.6%，其余 assay 为 0。
2. **`mask == 0` 的位置**：`tied_featurize` 自己标出的无效位。

### 2.2 两种读出分别是怎么算的

对每个 assay，用**冻结**的 encoder 跑一遍它的 WT complex，拿到每残基 128 维的 `h_V`，
在其上训**线性**模型（特征先按维度做标准化）：

| 读出 | 模型 | 目标 | 报什么 |
|---|---|---|---|
| **AUC（二值）** | logistic regression | `1[d(r) ≤ 5 Å]` | ROC-AUC |
| **ρ（连续）** | **ridge regression（α=1）** | **`d(r)` 这个实数**（单位 Å，不做变换） | **预测值与真实 `d(r)` 之间的 Spearman 秩相关** |

**⇒ AUC 衡量「碰不碰」分得开不开；ρ 衡量「多远」估得准不准。**
ρ 用 Spearman 而不是 R²，是因为我们关心的是**能否把残基按远近排对**，
而不是能否还原 Å 的绝对数值。

两种切分：

- **within-assay（5-fold）**：在同一个 assay 的残基内部做 5 折交叉验证，
  报出的是**留出折**上的预测。回答「**对这个复合物**，信息在不在 `h_V` 里」。
- **leave-one-assay-out（LOAO）**：用其余 13 个 assay 的**全部**残基训练，在留出的那个上测。
  回答「这个编码是**跨复合物共享**的，还是每个结构各背一套」。

### 2.3 结果

**预测的粒度是「每一个氨基酸残基」** —— `h_V` 的每一行是一个残基的 128 维表示，
两种读出都是**对该残基**预测一个量（是不是界面残基 / 离另一个 entity 多远）。
所以下表的 `n` 是**参与训练与评测的残基数**，不是变体数。

**「排除」列的含义**：从 `L` 个打包槽位里被剔除、**不参与探针**的那些，两个来源（取并集）：

| 来源 | 判据 | 为什么必须剔除 |
|---|---|---|
| **缺口填充位** | 该槽位在 PDB 里**没有对应残基**（`parse_PDB` 按残基号连续遍历时补出来的） | 它不是真实残基，坐标被置成 (0,0,0)，算出的距离是伪造的 |
| **`mask == 0`** | `tied_featurize` 自己标记的无效位（骨架原子缺失等） | 模型自己都不把它当有效残基 |

> 实测 14 个里只有 4 个有非零排除（`KRAS_PICK3CG-RBD` 17.3%、`CD19_FMC63` 10.5%、
> `HLA-A2_TAPBPR` 9.6%、`5A12_Ang2` 0.6%），其余 10 个为 0。
> 逐 assay 的排除计数见 `ga2_coverage.csv`（路径见附录 A.2）。

**逐 assay 明细**（`L` = 复合物总残基数）：

| assay | L | 排除 | 参与探针的残基 | 界面比例 | AUC within | AUC LOAO | ρ(d) within | ρ(d) LOAO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `5A12_Ang2_fitness_4ZFG` | 652 | 4 | 648 | 0.056 | 0.875 | 0.879 | 0.586 | 0.065 |
| `ACE2_SARS2-RBD_enrich_6M17` | 931 | 0 | 931 | 0.046 | 0.951 | 0.970 | 0.601 | 0.373 |
| `CD19_FMC63_Fitness_7URV` | 497 | 52 | 445 | 0.092 | 0.911 | 0.886 | 0.636 | 0.204 |
| `CXCR4_CXCL12_enrich_8U4O` | 360 | 0 | 360 | 0.169 | 0.820 | 0.851 | 0.557 | 0.423 |
| `GB1_IgG-Fc_fitness_1FCC` | 262 | 0 | 262 | 0.137 | 0.946 | 0.940 | 0.432 | 0.456 |
| `HLA-A2_TAPBPR_meanscore_5WER` | 644 | 62 | 582 | 0.124 | 0.891 | 0.897 | 0.677 | 0.449 |
| `KRAS_PICK3CG-RBD_norfitness_1HE8` | 1107 | 192 | 915 | 0.038 | 0.918 | 0.946 | 0.390 | 0.144 |
| `KRAS_RAF1-RBD_norfitness_6VJJ` | 245 | 0 | 245 | 0.127 | 0.829 | 0.968 | 0.293 | 0.445 |
| `KRAS_RAF1_norfitness_6VJJ` | 245 | 0 | 245 | 0.127 | 0.829 | 0.968 | 0.293 | 0.445 |
| `KRAS_RALGDS-RBD_norfitness_1LFD` | 254 | 0 | 254 | 0.110 | 0.844 | 0.958 | 0.489 | 0.357 |
| `PSD95_CRIPT_1BE9` | 120 | 0 | 120 | 0.167 | 0.713 | 0.820 | 0.331 | 0.441 |
| `PSD95_Tm2F_1BE9` | 120 | 0 | 120 | 0.175 | 0.692 | 0.824 | 0.326 | 0.453 |
| `SARS2-RBD_ACE2_deltaKd_6M0J` | 791 | 0 | 791 | 0.054 | 0.885 | 0.948 | 0.247 | 0.174 |
| `hYAP65_peptide_FunctioncalScore_1JMQ` | 56 | 0 | 56 | 0.339 | 0.688 | 0.824 | 0.375 | 0.625 |
| **mean（14 assay 未加权）** | | | | 0.126 | **0.842** | **0.905** | **0.445** | **0.361** |
| *median* | | | | *0.125* | *0.859* | *0.918* | *0.411* | *0.432* |

**三条结论：**

1. **二值界面信息在 `h_V` 里，且线性可读** —— AUC 均值 **0.842**（随机为 0.5）。
2. **这个编码跨复合物共享** —— LOAO 均值 **0.905**，**比 within-assay 的 0.842 还高**，
   因为 LOAO 用了 13 个 assay 的残基训练、样本多一个量级。
   若编码是每个结构各背一套，LOAO 应当明显更差。
3. 🔴 **细粒度明显更弱** —— 「多远」的 ρ 均值只有 **0.445**（within）/ **0.361**（LOAO），
   而「碰不碰」的 AUC 是 0.842 / 0.905。
   **这正是 structure-TTT 可能有空间的地方。**

> **为什么同时给 mean 和 median**：BindingGYM 的官方口径是**逐 assay 指标取未加权平均**，
> 所以 **mean 是主指标**；median 一并列出只是为了看分布有没有被个别 assay 拉偏。
> 本例两者差别不大（AUC within 0.842 vs 0.859）。

## 3. 一条方向性证据：encoder 编码得越好的 assay，zero-shot 倾向于越好（**未达显著**）

### 3.1 具体测了什么

**这是一个跨 assay 的相关性分析，n = 14（上面 §1.2 表里标了 14 的那些）。**

对每个 assay，我们手上有两个独立得到的数：

| 量 | 怎么来的 | 含义 |
|---|---|---|
| **`AUC_within`** | §2 的 G-A 探针，within-assay 5-fold | **这个复合物的 encoder 表征，把界面残基分得有多开** |
| **`ρ_zeroshot`** | 预训练 ProteinMPNN 在该 assay 上的 per-assay Spearman | **这个 assay 的 zero-shot DMS 预测有多准** |

然后问：**这 14 对数之间有没有关系？** 即「encoder 把界面编码得好的那些 assay，
是不是恰好也是 ProteinMPNN 预测得准的那些」。用 Spearman 秩相关来测。

### 3.2 结果

| 探针指标 | 对照量 | Spearman | **p 值** |
|---|---|---:|---:|
| **`AUC_within`（二值界面）** | **`ρ_zeroshot`** | **+0.519** | **0.057** |
| `ρ_within`（连续距离） | `ρ_zeroshot` | −0.128 | 0.664 |

**`p` 是什么**：在「两者其实毫无关系」这个零假设下，仅凭随机涨落也能得到
**不小于观测到的 |Spearman|** 的概率。`p = 0.057` 意思是：如果 encoder 质量与 zero-shot
表现真的无关，那么 14 个 assay 上出现 ≥ 0.519 这么强的相关，有约 **5.7%** 的机会纯属偶然。
习惯上以 0.05 为界，**所以 0.057 是「接近显著但没过线」**。

**逐 assay 明细**（按 zero-shot 从高到低）：

| assay | `ρ_zeroshot` | `AUC_within` | `ρ_within` |
|---|---:|---:|---:|
| `SARS2-RBD_ACE2_deltaKd_6M0J` | 0.6967 | 0.885 | 0.247 |
| `CD19_FMC63_Fitness_7URV` | 0.6032 | 0.911 | 0.636 |
| `KRAS_RALGDS-RBD_norfitness_1LFD` | 0.5872 | 0.844 | 0.489 |
| `KRAS_PICK3CG-RBD_norfitness_1HE8` | 0.5020 | 0.918 | 0.390 |
| `GB1_IgG-Fc_fitness_1FCC` | 0.5005 | 0.946 | 0.432 |
| `KRAS_RAF1_norfitness_6VJJ` | 0.4871 | 0.829 | 0.293 |
| `KRAS_RAF1-RBD_norfitness_6VJJ` | 0.4413 | 0.829 | 0.293 |
| `HLA-A2_TAPBPR_meanscore_5WER` | 0.4116 | 0.891 | 0.677 |
| `PSD95_CRIPT_1BE9` | 0.3672 | 0.713 | 0.331 |
| `ACE2_SARS2-RBD_enrich_6M17` | 0.2760 | 0.951 | 0.601 |
| `CXCR4_CXCL12_enrich_8U4O` | 0.2014 | 0.820 | 0.557 |
| `PSD95_Tm2F_1BE9` | 0.1944 | 0.692 | 0.326 |
| `5A12_Ang2_fitness_4ZFG` | 0.1074 | 0.875 | 0.586 |
| `hYAP65_peptide_FunctioncalScore_1JMQ` | 0.0886 | 0.688 | 0.375 |
| **mean** | **0.3903** | **0.842** | **0.445** |

### 3.3 怎么解读

**因此现在能说的是**：
`AUC_within` 与 `ρ_zeroshot` **呈中等强度的正相关，方向一致，但在 n=14 上未达统计显著**。
它**支持但不证实**「改善 encoder 的界面表征 → 改善 zero-shot」这条链。

**即便它显著，也只是相关，不是因果。** 至少三个混淆项可能**同时**驱动这两个量：

1. **结构质量** —— 分辨率高、缺失残基少的结构，encoder 编码得好是自然的，
   同时 ProteinMPNN 打分也更可靠。
2. **复合物大小 `L`** —— 残基数多则探针训练样本多、AUC 可能偏高；`L` 也影响打分的尺度与方差。
3. **assay 本身的噪声水平** —— DMS 噪声大的 assay，`ρ_zeroshot` 上限本来就低，与 encoder 无关。

**要把「相关」推进到「支持因果」，最直接的是做干预**：不训练，直接人为改动冻结的 `h_V`，
重新打分，看 per-assay Spearman 是否随之变化。这只需要前向打分，不需要训练。

**方向 `u` 怎么来**：§2 的二值探针是一个 logistic regression，对标准化后的 `h̃_V(r)` 拟合
`logit(r) = w · h̃_V(r) + b`。`w ∈ R¹²⁸` 就是「沿哪个方向走，探针越认为这是界面残基」，
取单位化 **`u = w / ‖w‖`**。
探针的标签来自结构（重原子 5 Å），**不含任何 DMS label**，所以用 within-assay 的探针拟合 `u`
不构成泄漏。（也可以取 ridge 回归预测连续 `d` 的权重方向 —— 那是「多远」的方向，
和「碰不碰」的方向不是同一个，两个都值得试。）

**怎么改 —— 要缩放投影，不是整体平移**：

```
h̃_V'(r) = h̃_V(r) + α · ( u · h̃_V(r) ) · u
```

即把每个残基**自己**在 `u` 上的分量乘 `(1 + α)`。

| `α` | 含义 |
|---|---|
| `0` | 不变（对照原点） |
| `> 0` | **放大**界面轴：界面与非界面残基在这根轴上被推得更开 |
| `= −1` | **完全消除**这根轴上的界面信息（消融） |
| `< −1` | **反转**界面轴 |

> 🔴 **不要写成 `h_V + α·u` 这种整体平移。** 那样每个残基的探针 logit 都增加同一个常数
> `α‖w‖`，**残基之间按界面程度的相对次序完全没变**，`h_V` 里的界面信息量一点没动 ——
> 它测的是「把表征沿某方向整体搬一下会怎样」，不是「界面表征的强弱变化会怎样」。

**必须先验证干预确实生效**：改完之后把探针在 `h̃_V'` 上重跑一遍。
`α > 0` 时 AUC 应当**上升**，`α = −1` 时应当**掉到 0.5 附近**。
**这一步不过，说明干预压根没起作用，后面的分数变化无法归因。**

**🔴 单独跑主臂读不出任何东西，必须配对照** —— 因为只要幅度够大，
沿任何方向改 `h_V` 都会让模型变差。没有对照就分不清「沿界面方向改」和「改 `h_V` 这件事本身」。

| 臂 | 是什么 | 它排除什么 |
|---|---|---|
| **界面方向 `u`**，`α` 扫过 `−1.5 … +1.5` | 主臂 | — |
| **随机方向**（同幅度，多个随机种子） | 取随机单位向量替代 `u` | **「任何扰动都会有这个效果」** |
| **打乱分量的 `u`** | 把 `u` 的 128 个分量随机置换后再用 | 保留 `u` 的数值分布，只破坏「哪个值属于哪一维」 |

⚠️ **幅度匹配要小心**：换成投影形式后 `‖Δh̃‖ = |α| · |u · h̃|`。
真实的 `u` 指向**有实际方差**的方向、投影大；随机方向的投影小得多。
**所以同一个 `α` 并不意味着同一个扰动幅度** —— 必须按 `‖Δh̃‖`（或它引起的分数标准差变化）
显式对齐各臂，否则比的又是幅度而不是方向。

**预期的 signature**：主臂应当**关于 `α` 不对称**（放大与消融给出方向相反的效果），
随机臂应当**近似对称且只会恶化**。
若主臂与随机臂在同幅度下**形状相同**，说明 `h_V` 的界面方向并不特别，
「改善 encoder 的界面表征 → 改善 zero-shot」这条链就没有得到支持。

**反面参照**：`ρ_within`（连续距离）与 `ρ_zeroshot` 的相关是 **−0.128, p=0.664**，
不但不显著、方向还是反的。
⇒ **目前这条证据只覆盖「二值界面编码质量」，细粒度距离编码与 zero-shot 表现之间没有观察到关系。**

---

## 4. 会静默出错的地方（纪律与陷阱）

**这一节全部来自实际踩过的坑 —— 它们的共同特征是：不报错、结果看起来合理、但已经错了。**

### 4.1 索引与对齐

1. 🔴 **`parse_PDB` 会把晶体学缺口补成假残基。** 它按 `range(min_resn, max_resn+1)` 遍历，
   缺失的残基号被补成 `X` + NaN 坐标，随后 `tied_featurize` 执行 `X[isnan] = 0.`
   ⇒ **这些位置带着坐标 (0,0,0) 进入模型**，在 `h_V` 里占着行但不是真实残基（`mask` 被置 0）。
   实测占比最高 **17.3%**（`KRAS_PICK3CG-RBD`）。
   **任何逐残基的分析都必须先按 `mask` 和「是否为填充位」过滤。**
   （这条已经造成过一次真实的静默错误，见 §2.1。）
2. **官方 `parse_PDB` 与自己写的 PDB 解析器，残基列表会不一致**（实测 **9/25** 个结构不同）。
   除了上一条的缺口填充，它还会 `if resi not in alpha_3: continue` 丢掉非标准残基，
   并把 `HETATM MSE` 当作 `ATOM`。**跨解析器映射索引之前必须逐链比对序列并 assert。**
3. **WT 序列位置 ↔ PDB 残基索引不是一回事。** 有偏移、有 gap。
   实测案例：`6VJJ` 有 **+1 偏移**；`4ZFF` / `4ZFG` 带 **Kabat insertion code**（残基号形如 `100A`）。
   解析 PDB 时残基 id 必须取 `line[22:26].strip() + line[26].strip()`，否则 insertion code 被吞掉。
   ⚠️ **但逐残基的标签根本不需要这个映射** —— `h_V` 按 PDB 残基索引，直接在 PDB 空间算即可。
   早先为「绕开对齐」而改用 CA–CA 距离是不必要的（见 §2.1）。
4. **`tied_featurize` 的打包顺序是「designed chains 在前，其余在后」**，
   且用的是 **PDB 链序列**长度，不是 WT 序列长度。`h_V` 的行索引跟着这个顺序走。
5. **数据对齐后必须按文件数 / 行数核对**，不能只 `test -f`。
   曾经出现过 symlink 指向 worktree 外部、rsync 过去变成断链，而 `ls | wc -l` 只显示 1 的情况。

### 4.2 实验设计

4. **先检查目标函数的最优解是什么。** 这条线上已经有过两次教训：
   - 「让模型对 WT 序列的 likelihood 更高」这个 TTT 目标：最优解不是想要的东西，**已被否证**；
   - 无约束的「最大化两组分数分布差异」：最优解让分数**塌缩成界面特征的一个函数**，
     实测那个塌缩点是 **0.2564**，**比 0.3903 的基线还差 0.13**。
   **动手前先问：如果这个 loss 被完美优化，分数会变成什么？**
5. **任何「取最大」的统计量都必须配 null。** 扫超参、选配置、选最优 `c`，都会对样本过拟合。
   做法：把关键变量（如距离）随机置换后，跑**完全相同**的选择程序，把它的结果作为 null 报出来。
6. **判读标准要写成定量阈值，不要写成是非题。** 实测教训：预先写的是
   「置换 null 也涨 ⇒ 增益不来自先验」，结果 null 涨了、但只占总增益的 19% 且从不越线 ——
   二值判据在这里给不出结论。
7. **打分函数必须与评测用的逐行相同。** 改了模型路径之后，
   先验证「不训练时能否复现官方分数」。我们做过这个对照，最大相对偏差 **3.0e-7**（float32 累加量级）。
8. **口径不许混。** 每个数字都要说明是 14 / 23 / 25 assay；不同实验用不同子集时**分开报**。
   曾经把两次不同配置的运行拼在一起算均值，得出过错误的结论。
9. **单调不变性**：`Spearman(f(d), ·)` 对 `d` 的任意**单调**变换**恒定不变**。
   所以「换一个距离形状」（`exp(−d/τ)` → `1/(1+d/τ)` → `−d`）**改不动任何秩层面的量**，
   只对**加性组合**与 **Pearson** 有杠杆。别在这上面白费力气。
10. **label-free 的模型选择是这条线上反复出现的未解问题。**
    已测 10 个静态 WT 特征来预测最优超参，**9 个比用常数更差**，最好的只拿回 **28%** 的空间，
    且相关性**过不了 10 重检验的 BH 校正**。
    ProteinTTT 在结构预测上可以用 **pLDDT** 选最佳模型，**ProteinMPNN 这边至今没找到类比**。
    ⚠️ 但**只测过静态特征** —— 适配后模型**自身的行为**（自监督目标的 held-out 值、
    TTT 前后一致性、与独立预测器的 ensemble agreement）**从未测过**。

### 4.3 工程

11. **新增命令行参数后，必须先用最小配置 smoke test 那条确切路径再投长任务。**
    实测教训：没做这件事，三个子任务秒退，白等了 8.5 小时的 GPU 空档。
12. **M（解码顺序数）在 `AssayContext` 构造时就烙进了所有张量**（mask、randn 等）——
    只改 `ctx.M` 不重建会静默出错。
13. **`rsync -a` 从 workstation 回流会覆盖本地写的 record md**（它按差异传、不按新旧传）。
    回流必须 `--exclude '*.md' --update`。
14. **matplotlib 只有 DejaVu Sans 可用**，中文会**静默**变成豆腐块 ⇒ 图上的文字一律用英文。
15. **`pandas` 里 `median` / `agg` 这类名字会被属性遮蔽** —— `df.median` 返回的是方法不是列。
    列名别取这些，取了就用 `df["median"]`。
16. **stdout 会被缓冲** —— 长任务用 `python -u`，并让结果**逐步落盘**，
    否则跑几小时看不到任何进度，中途失败会丢掉已完成的部分。

---

## 附录 A：现成可复用的资产（绝对路径）

### A.1 代码（在 worktree `bindingGYM-binding-sites-analysis` 内）

| 东西 | 绝对路径 |
|---|---|
| **可微打分 + encoder 缓存**（`AssayContext` 已缓存 encoder 输出，改 `h_V` 后只需跑 decoder） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/scripts/mutation_landscape_ttt/bgmpnn.py` |
| **encoder 线性探针（现行，重原子 5 Å，已排除缺口）** | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/scripts/mutation_landscape_ttt/ga2_encoder_probe.py` |
| *旧版探针（CA–CA 8 Å，含缺口填充位；仅供追溯，勿用）* | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/scripts/mutation_landscape_ttt/ga_encoder_probe.py` |
| **no-op 对照**（验证打分函数与官方逐行一致） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/scripts/mutation_landscape_ttt/gb_noop_control.py` |
| 官方 `protein_mpnn_utils.py` 的 vendored 副本（md5 `56fc8e171b6d97dc9a048259f4eb3a77`） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/scripts/mutation_landscape_ttt/vendor/protein_mpnn_utils.py` |
| 界面计算流程（entity 划分、逐残基距离） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/scripts/binding_sites/` |

### A.2 数据产物（本地）

| 东西 | 绝对路径 |
|---|---|
| **逐残基**界面距离（重原子，全链，9,493 行） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-analysis/data/interface_residues_all_chains.csv` |
| **entity 划分**（由元数据确定，非结构启发式） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-analysis/data/entity_partition.csv` |
| 逐 variant 的界面标签 + MPNN zero-shot 分数（376,424 行） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet` |
| **逐突变位点**距离（1,173,273 对，可换任意聚合方式） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-analysis/data/variant_site_dists.parquet` |
| G-A 探针逐 assay 结果（**现行**，§2 的原始数据） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/workstation-records/mutation-landscape-TTT/data/ga2_encoder_probe.csv` |
| 各 assay 被排除的残基数（缺口填充位统计） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/workstation-records/mutation-landscape-TTT/data/ga2_coverage.csv` |
| 逐 assay 的 zero-shot ρ 与其他参照量 | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/workstation-records/mutation-landscape-TTT/data/t4_per_assay_bars.csv` |
| 14-assay 集合的推导与逐 assay 明细 | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/workstation-records/mutation-landscape-TTT/data/g1e_canonical14.csv` |

### A.3 BindingGYM 本体与模型权重

| 东西 | 绝对路径 |
|---|---|
| 官方代码仓库（评测口径与 split 的唯一权威） | `/home/guoj0f/repos/BindingGYM` |
| 官方 input（DMS csv + 结构 + 索引 `BindingGYM.csv`） | `/home/guoj0f/share/BindingGYM/input`（本地）<br>`/data/guoj0f/share/BindingGYM/input`（workstation） |
| 官方打分脚本 | `/data/guoj0f/share/BindingGYM/baselines/protein_mpnn/compute_fitness_multi_pdb.py`（workstation） |
| **ckpt `v_48_020.pt`**（md5 `91d54c97a68bf551114f8c74c785e90f`） | `/data/guoj0f/share/BindingGYM/training/cache/v_48_020.pt`（workstation） |
| 官方 zero-shot 参考分数（seed 1, M=5，329 MB，逐 variant） | `/data/guoj0f/BindingGYM-zero-shot-proteinMPNN/scores/seed1_M5/`（workstation） |
| conda env（numpy 1.24.4 / scipy 1.10.1，**别升级**） | workstation `bindinggym-zs-mpnn`（base 在 `/data/guoj0f/miniconda3`） |

### A.4 相关记录文档

| 内容 | 绝对路径 |
|---|---|
| binding-sites 定义、数据统计、真值与预测分布的逐 assay 对比 | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-overview/BindingGYM_binding_sites_overview_20260903.md` |
| 界面先验价值的摸底（含 rescoring 规则、`c` 的 label-free 选法否定结论） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/workstation-records/mutation-landscape-TTT/probe_binding_site_insight_value_20260909-160514.md` |
| decoder 侧 TTT 的现状与结论 | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/workstation-records/mutation-landscape-TTT/transduc_decoder_ttt_interface_divergence-organized.md` |
| decoder 侧 TTT 的操作日志（含全部踩坑细节） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/workstation-records/mutation-landscape-TTT/transduc_decoder_ttt_interface_divergence-tracking.md` |
| BindingGYM 两个污染 assay 的调查 | `/home/guoj0f/repos/Sources/datasets/BindingGYM-issues` |
