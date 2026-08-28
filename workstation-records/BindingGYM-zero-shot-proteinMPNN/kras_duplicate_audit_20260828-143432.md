# BindingGYM KRAS 重复数据 — 溯源审计

> ## ⚠️ 本文档的归因结论已被推翻（2026-08-28 晚）
> 拿到源头数据后确认：**错误不在 BindingGYM，而在 Nature 论文的 Supplementary Table 4 本身** ——
> 其 `BindingPCA DARPin K27` 与 `BindingPCA DARPin K55` 两块数据被 `BindingPCA SOS1` 整块覆写。
> BindingGYM 是**忠实传播**（读取无误 98.7–99.3%）。
> **定案：`KRAS_SOS1_8BE4` 是对的，`KRAS_DARPinK27_5O2S` 是错的**（本文档 §3.5 的结构启发式指向相反，是错的）。
> 完整证据链（bioRxiv v1 / MAVEdb / 原始 DiMSum 三个独立来源）见：
> **`/home/guoj0f/repos/Sources/datasets/BindingGYM_KRAS_provenance_audit_20260828.md`**
>
> 本文档保留作为**发现过程与 benchmark 内影响评估**的记录；§1/§2/§3.6/§4 的实测数字仍然成立，
> **§3.3（归因到 curation 第 3 步）与 §3.5（未定案）已过时**。

> created 2026-08-28 14:34 · status: DONE（调查完成，一处归属未定案）
> **只读调查**：全程未修改任何代码或数据集，仅新增本文档。
> 关联：[`zeroshot_proteinmpnn_20260827-154500.md`](zeroshot_proteinmpnn_20260827-154500.md)（本 project 的复现记录）

---

## 0. TL;DR

`KRAS_DARPinK27_norfitness_5O2S` 与 `KRAS_SOS1_norfitness_8BE4` 这两个 assay 的
`DMS_score` 列在 **19,227 个共同 mutant 上逐点位相同**（`max|Δ| = 0.0`，非"近似"）。

- 二者**都在 25 个 benchmark assay 之内**，官方 14 个 zero-shot 模型的结果全部含它们。
- 二者的**测量环境与 metric 物理含义一致**（同一篇源文章、同一个 raw 文件、同一套 yeast-growth 选择、同一种归一化）——
  正因为一致，才更说明问题：**同一套协议下针对两个不同 partner 的两次选择实验，不可能给出逐位相同的浮点数**。
- **25 个 assay 的 300 对全量扫描：这是唯一一处重复**（75 对有实质重叠，其余 eq=0）。
  同体系近邻对（GB1×2 r=0.979、RAF1 vs RAF1-RBD r=0.927、PSD95×2）label 高度相关但**都不相同** ⇒
  扫描有分辨力，这不是 BindingGYM 的编码惯例，而是**一处孤立的 curation 错误**。
- 已排除：不是整文件复制、不是 DARPin **K55**、不是 **abundance**（三条都用数据否掉了）。
- **未定案**：无法确定是哪一个文件拿错了列。结构界面检验给出弱的方向性提示（偏向"共享向量属于 DARPin K27"），但 p = 0.208，不足以下结论。

**对我们工作的结论：** 复现口径不用改（必须跟官方同 25 个）；但 complexTTT 的 Δ 分析应
主报 n=25 + 附 n=23 敏感性检验，不要单剔其中一个（见 §4.3）。

---

## 1. 问题(1)：突变操作、测量环境、metric 物理含义是否一致？

### 1.1 突变对象：都是 KRAS，且是同一个突变库

两个 assay **突变的都是 KRAS 本身**（partner 不带突变）：

| assay | PDB | 被突变链 | 链长 | partner 链 | partner 长 | 复合物全长 |
|---|---|---|---|---|---|---|
| `KRAS_DARPinK27_..._5O2S` | `5O2S_hm.pdb` | **A** | 165 | B = DARPin K27 | 156 | 321 |
| `KRAS_SOS1_..._8BE4` | `8BE4_hm.pdb` | **R** | 168 | S = SOS1 | 475 | 643 |

突变集合：**19,533 vs 19,425，交集 19,227**（98.4% / 99.0%）。
深度分布（交集）：单点 2,461 + 双点 16,765。

### 1.2 差集 100% 由**结构覆盖**解释，不是实验差异

| | 只在 DARPinK27 (306) | 只在 SOS1 (198) |
|---|---|---|
| 主导位点 | **118 (123), 119 (121), 120 (62)** | **166 (73), 167 (111), 168 (14)** |
| 其余 | 63/66/71/72/78/82/84/101/112/113（双突变里的搭档位） | 133/137/146/149/159（同上） |

对照两条 KRAS 链序列：

```
5O2S chain A (165 aa): ...MVLVGNK CDL PSRTVDTKQA ... DAFYTLVREIRK          ← 止于 165
8BE4 chain R (168 aa): ...MVLVGNK XXX PSRTVDTKQA ... YTLVREIRK HKE         ← 到 168，但 118-120 未解析
```

- 118–120 在 5O2S 解析出来（`CDL`）、在 8BE4 是 `XXX` ⇒ 这些 mutant 只能留在 DARPinK27 文件
- 166–168 只存在于 8BE4 的更长构建 ⇒ 只能留在 SOS1 文件

⇒ **两个文件的 mutant list 是从同一份母表按各自结构过滤出来的两个子集。**
这一步 per-assay 的处理**是对的**（也正因如此，排除了"整个文件被复制"的假设，见 §3.4）。

### 1.3 测量环境与 metric：一致

来自 ProteinGym `reference_files/DMS_substitutions.csv`（对同一篇源文章的**独立第三方 curation**）：

```
title               : The energetic and allosteric landscape for KRAS inhibition
first_author / year : Weng / 2022 (bioRxiv) → Nature 2024, DOI 10.1038/s41586-023-06954-0
selection_assay     : Yeast growth
raw_DMS_filename    : kras_fitness.xlsx          ← 六个 partner 共用同一个原始工作簿
raw_DMS_phenotype   : fitness
region_mutated      : 2-188
```

BindingGYM 侧 assay 名里的 `norfitness` = **normalized fitness**，实测口径一致：

| assay | n | WT_score | min | max | mean | std | 位点范围 |
|---|---|---|---|---|---|---|---|
| **DARPinK27** | 19,533 | **−0.019953** | **−1.971** | **0.867** | −0.6253 | **0.4575** | 2–165 |
| PICK3CG-RBD | 19,203 | −0.004941 | −1.747 | 0.627 | −0.5880 | 0.4995 | 2–166 |
| RAF1 | 12,677 | −0.008706 | −1.859 | 0.581 | −0.8353 | 0.5077 | 3–65 |
| RAF1-RBD | 23,162 | −0.008841 | −1.450 | 0.258 | −0.4567 | 0.3924 | 3–168 |
| RALGDS-RBD | 20,341 | −0.000423 | −1.603 | 0.282 | −0.5351 | 0.4541 | 2–167 |
| **SOS1** | 19,425 | **−0.019953** | **−1.971** | **0.867** | −0.6257 | **0.4575** | 2–168 |

**六个 assay 的 WT 都归一到 ≈ 0，量纲一致** ⇒ metric 物理含义相同（"相对 WT 的 log 富集比 / 生长适应度"，
数值本身不等于 ΔΔG，但单调相关；BindingGYM 论文明确说 "the DMS score does not directly equal ΔG but correlates with it"）。

> ⚠️ 注意最后一行与第一行：**WT_score / min / max 三个值逐位相同**，mean 与 std 的
> 微小差异（0.0004 / 0.0000）完全由那 306 + 198 个非共同 variant 造成。

**所以对问题(1)的回答：** 突变对象、突变库、测量协议、metric 归一方式**全部一致**。
这恰恰是问题所在 —— 一致的协议下，针对 DARPin K27 与 SOS1 的两次**独立选择实验**
不可能产出 19,227 个位相同的浮点数。

---

## 2. 问题(2)：labels 是否一模一样？

**是，bit-exact，不是"高度相关"。**

```
共同 mutant           : 19,227
max|Δ|                : 0.000e+00
#exact-equal          : 19,227 / 19,227
Pearson               : 1.0000000000
行顺序                : 前 309 行逐行对齐（第 1 行 WT = −0.0199530896045886，
                        第 2 行 A11C = 0.100154266347276），之后因增删错位
```

六个 KRAS assay 的完整两两比较（`ov` = 共同 mutant 数，`eq` = 逐点严格相等数）：

| pair | ov | **eq** | pearson |
|---|---|---|---|
| **DARPinK27 vs SOS1** | 19,227 | **19,227** | **+1.0000** |
| RAF1 vs RAF1-RBD | 12,086 | 0 | +0.9267 |
| PICK3CG-RBD vs RALGDS-RBD | 16,880 | 0 | +0.8834 |
| RALGDS-RBD vs DARPinK27 | 17,472 | 0 | +0.7506 |
| RALGDS-RBD vs SOS1 | 17,388 | 0 | +0.7507 |
| PICK3CG-RBD vs DARPinK27 | 16,502 | 0 | +0.6686 |
| PICK3CG-RBD vs SOS1 | 16,293 | 0 | +0.6671 |
| RALGDS-RBD vs RAF1 | 117 | 0 | +0.5533 |
| RAF1-RBD vs RALGDS-RBD | 139 | 0 | +0.5695 |
| PICK3CG-RBD vs RAF1-RBD | 140 | 0 | +0.4850 |
| PICK3CG-RBD vs RAF1 | 116 | 0 | +0.4480 |
| DARPinK27 vs RAF1-RBD | 139 | 0 | +0.4088 |
| SOS1 vs RAF1-RBD | 139 | 0 | +0.4088 |
| DARPinK27 vs RAF1 | 116 | 0 | +0.3885 |
| SOS1 vs RAF1 | 116 | 0 | +0.3885 |

**15 对里只有 1 对 exact-equal。** 其余 14 对都有真实差异，这才是"不同 partner ⇒ 不同 binding readout"应有的样子。

> 附带：`RAF1` vs `RAF1-RBD` 是**镜像对照** —— 同一个 `6VJJ.pdb`、label 却不同（eq=0, r=0.9267）。
> 所以 BindingGYM 里同时存在"同结构 / 不同 label"与"同 label / 不同结构"两种配对。

---

## 3. 问题(3)：物理/生物定义 + 溯源

### 3.1 两个 assay 的生物学定义

源文章一次性测了 KRAS 对**多个结合伙伴**的突变效应（BindingGYM 论文原文：
*"[14] conducted screenings for the KRAS protein against **seven** different proteins"*），
BindingGYM 取了其中 6 个 partner，ProteinGym 只取了 1 个（`binding-DARPin_K55`）+ 1 个 abundance。

| assay | partner 是什么 | 生物学含义 |
|---|---|---|
| `KRAS_DARPinK27` | **DARPin K27** —— 人工设计的 ankyrin repeat binder（非天然） | KRAS 突变对"与合成抑制剂结合"的影响；**药物/binder 工程视角** |
| `KRAS_SOS1` | **SOS1** —— 天然 guanine nucleotide exchange factor (GEF) | KRAS 突变对"与上游激活因子结合"的影响；**天然信号通路视角** |

两者的 KRAS 结合表位并不相同（本文档 §3.5 由结构算出：
K27 界面 26 个残基、switch-I 主导；SOS1 界面 35 个残基、switch-II 主导；Jaccard 仅 0.33）。
⇒ **生物学上必然是两组不同的测量。**

### 3.2 数据从哪里来

| 层级 | 内容 | 证据来源 |
|---|---|---|
| 源文章 | Weng et al., *The energetic and allosteric landscape for KRAS inhibition*, **10.1038/s41586-023-06954-0** | BindingGYM 论文 **Table 10 (A.14 List of data sources)**，六行 KRAS 全指向该 DOI |
| 原始文件 | **`kras_fitness.xlsx`** —— 一个工作簿，各 partner 一列/一表 | ProteinGym `DMS_substitutions.csv` 的 `raw_DMS_filename` 字段（两条 RASK 记录都指向它） |
| 测量方式 | Yeast growth 选择 → log 富集比 → 相对 WT 归一 | ProteinGym `selection_assay`；BindingGYM `new_dataset_construction_guide.md` §3 第 1/2 条 |
| BindingGYM 分发 | `zenodo.org/records/12514160/files/input.zip` → `input/Binding_substitutions_DMS/*.csv` | README "Download Data" |

**关键点：六个 KRAS assay 出自同一个 Excel 工作簿的不同列/表。**
这正是"选错列"这一类错误最容易发生的数据形态。

### 3.3 错在 curation 的哪一步

BindingGYM 的 `new_dataset_construction_guide.md` 把流程分成 7 步。逐步对照：

| 步骤 | 本例是否正确 | 证据 |
|---|---|---|
| 1. 确定参考序列 | ✅ 正确 | 两个文件的 KRAS 链序列与各自 PDB 完全一致（`PDBseq == CSVseq`） |
| 2. 搜复合物晶体结构 | ✅ 正确 | 5O2S = KRAS–DARPin K27，8BE4 = KRAS–SOS1，配对无误 |
| 3. 从 NGS 结果算 binding score | ❓ **疑点所在** | 源是同一个 xlsx 的多列，两个 assay 取到了同一列 |
| 4. 确认突变位点 | ✅ 正确 | 位点编号与参考序列对齐，per-structure 过滤（118–120 / 166–168）逻辑正确 |
| 5. **一致性核对** | ❌ **失效** | 指南写 *"We compare the results with the main findings of the article to ensure consistency"* —— 若做了跨 assay 的一致性检查，两列 bit-exact 必然暴露 |
| 6. 复合物同源建模 | ✅ 正确 | `_hm` 结构与各自参考序列匹配 |
| 7. 数据精修（置信度过滤） | — | 不影响本结论 |

**定位：错误发生在第 3 步（分数列的抽取/合并），并且第 5 步的一致性核对没有覆盖到"跨 assay 重复"这种错误。**

从错误的**形状**还能再收窄一层：

> **mutant list 不同（per-structure 正确过滤），score 列相同。**

这不是"文件被整份复制"，而更像**下游 merge 时 key 里丢了 partner 维度** ——
按 `mutant` 取分、取到了同一列。这个形状具体且自洽，比"随机出错"更符合一个 pipeline bug。

### 3.4 已排除的三个替代解释

| 假设 | 检验 | 结论 |
|---|---|---|
| **H1：BindingGYM 有意让多个 partner 共用一张"通用 KRAS binding fitness"表** | 其余 14 对 label 全不同（eq=0）；且论文把"同一蛋白对不同 partner"当作**卖点**（Table 10 前的说明） | ❌ 排除 |
| **H2：整个文件被复制** | mutant list 不同（19,533 vs 19,425），且差集由结构覆盖精确解释 | ❌ 排除 |
| **H3：共享列其实是 DARPin K55 或 abundance（被误当成第 6/7 列）** | 与 ProteinGym 的独立 curation 对比（全深度，含双突变）：<br>vs `binding-DARPin_K55`：ov=17,945/17,848，**exact_eq = 0**，pearson 0.7152/0.7159<br>vs `abundance`：ov=18,695/18,590，**exact_eq = 0**，pearson 0.4804/0.4760 | ❌ 排除，共享列是真正的第 6/7 列之一 |

### 3.5 哪一个文件拿错了？—— **未定案**

做了一个结构界面富集检验：**真实 partner 的数据，其损伤应集中在该 partner 的界面上。**

方法：单点突变的逐位点平均 `DMS_score` → 用 ProteinGym `abundance`（folding 对照）线性回归扣掉
折叠成分 → 检验残差在界面残基上是否显著更负。界面 = 到 partner 链的最小重原子距离 < 5 Å。

先在已知正确的 assay 上验证方法（AUC 用扣除 folding 后的残差）：

| 对照 assay | #界面 | #非界面 | AUC_raw | **AUC_resid** | MWU p |
|---|---|---|---|---|---|
| RALGDS-RBD (1LFD) | 15 | 147 | 0.783 | **0.874** | 9.6e-07 |
| PICK3CG-RBD (1HE8) | 16 | 146 | 0.581 | **0.642** | 3.2e-02 |
| RAF1-RBD (6VJJ) | 17 | 147 | 0.547 | **0.555** | 2.3e-01 ← **方法在此失效** |

**方法的功效不稳定**（RAF1-RBD 基本等于随机），因此下面的结果只能当**方向性提示**：

| 待判 | #界面 | AUC_resid | MWU p |
|---|---|---|---|
| 共享向量 → **K27 界面**（5O2S） | 26 | **0.786** | 2.1e-06 |
| 共享向量 → **SOS1 界面**（8BE4） | 35 | 0.668 | 1.2e-03 |

由于两个界面重叠 15 个残基（Jaccard 0.33），又做了**只用判别位点**的对照：

```
K27-only  (11): 3, 24, 25, 29, 36, 38, 39, 41, 42, 43, 52
SOS1-only (20): 5, 17, 18, 55, 56, 57, 58, 59, 60, 61, 64, 65, 68, 69, 73, 95, 99, 102, 103, 105
（K27 界面 switch-I 主导：switchI∩=11, switchII∩=5；SOS1 界面 switch-II 主导：switchI∩=8, switchII∩=12）
```

| label 向量 | K27-only 均残差 | SOS1-only 均残差 | 差 | MWU p |
|---|---|---|---|---|
| **共享向量** | **−0.1694** | −0.0602 | **−0.109** | **0.208** |
| 对照 RALGDS-RBD | −0.1265 | −0.1576 | +0.031 | 0.984 |
| 对照 PICK3CG-RBD | −0.0774 | −0.1980 | +0.121 | 0.421 |
| 对照 ProteinGym K55 | −0.0277 | −0.2636 | +0.236 | 0.066 |

**读法：** 共享向量是唯一一个偏向 K27-only 残基的（差为负），三个对照都偏向另一侧。
方向上与"共享向量是真实的 DARPin K27 数据、SOS1 文件拿错了列"一致 ——
**但 p = 0.208，不显著，且方法在 RAF1-RBD 上失效，不能据此定案。**

**定案需要的东西：** Weng et al. 的 supplementary（`kras_fitness.xlsx`）原始六列。
本地没有该文章 PDF / 补充数据，需联网获取。

### 3.6 全量重复扫描：这是 benchmark 内的**唯一**一处

对 25 个 assay 做了完整的 300 对两两扫描（抹掉 chain id 后按 mutant 对齐）。
其中 **75 对**有实质重叠（≥50 个共同 mutant），逐点严格相等的比例如下：

| pair | ov | eq | 比例 | r | 判定 |
|---|---|---|---|---|---|
| **KRAS_DARPinK27_5O2S == KRAS_SOS1_8BE4** | 19,227 | **19,227** | **100.0%** | +1.0000 | **重复** |
| CXCR4_CXCL12_8U4O vs ACE2_SARS2-RBD_6M17 | 77 | 1 | 1.3% | −0.1527 | 单值巧合，非重复 |
| CXCR4_CXCL12_8U4O vs CD19_FMC63_7URV | 76 | 1 | 1.3% | +0.2608 | 单值巧合，非重复 |
| 其余 72 对 | — | **0** | 0.0% | — | 无重复 |

同一体系的"近邻对"都通过了检验（label 高度相关但**不相同**），说明扫描有分辨力：

| 近邻 pair | ov | eq | r |
|---|---|---|---|
| GB1_IgG-Fc_1FCC vs GB1_IgG-Fc_1FCC_2016 | 160 | **0** | +0.9789 |
| KRAS_RAF1 vs KRAS_RAF1-RBD（同一个 6VJJ） | 12,086 | **0** | +0.9267 |
| KRAS_PICK3CG-RBD vs KRAS_RALGDS-RBD | 16,880 | **0** | +0.8834 |
| Z-domain_ZpA963_HL1 vs HL2（同一个 2M5A） | 143 | **0** | +0.8782 |
| PSD95_CRIPT vs PSD95_Tm2F（同一个 1BE9） | 1,577 | **0** | +0.4795 |

⇒ **`KRAS_DARPinK27` / `KRAS_SOS1` 是 BindingGYM 25-assay benchmark 内唯一一处 label 重复。**

---

## 4. 影响评估

### 4.1 两个 assay 都在 benchmark 内

```
Binding_substitutions_DMS/ 磁盘文件 : 28
BindingGYM.csv（benchmark 索引）    : 25
被排除的 3 个 : CR6261_FluAH1_logKd_3GBN / CR9114_FluAH1_logKd_4FQI / CR9114_FluAH3_logKd_4FQY

KRAS_DARPinK27_norfitness_5O2S  in benchmark: True
KRAS_SOS1_norfitness_8BE4       in benchmark: True
```

`results/*_zero_shot_metric.csv` 共 14 个模型，**assay 集合逐一 == 索引的 25 个**（`same_as_index = True` ×14）。
⇒ 官方发表的 zero-shot 数字全部建立在含这两个 assay 的集合上。

### 4.2 对榜单：榜首稳，中段洗牌

ProteinMPNN 逐 assay：`DARPinK27 ρ = 0.4040`，`SOS1 ρ = 0.3092`（同一组 label，差 **0.0948**）。

```
全 25 个            0.3970   ← 官方 headline
剔除 SOS1  (n=24)   0.4006   Δ = +0.0037
剔除 DARPin(n=24)   0.3967   Δ = −0.0003
两个都剔除 (n=23)   0.4005   Δ = +0.0035
```

| model | n=25 | 剔 SOS1 | 剔两个 | rank |
|---|---|---|---|---|
| **ProteinMPNN** | 0.3970 | 0.4006 | 0.4005 | **1 → 1 → 1** |
| ProteinMPNN_single | 0.3564 | 0.3577 | 0.3578 | 2 → 2 → 2 |
| TranceptEVE | 0.3432 | 0.3376 | 0.3270 | **3 → 5 → 5** |
| PiFold | 0.3380 | 0.3383 | 0.3415 | 4 → 4 → **3** |
| ESM-IF1 | 0.3378 | 0.3402 | 0.3308 | 5 → **3** → 4 |
| Tranception | 0.3213 | 0.3178 | 0.3100 | 6 → 6 → 6 |
| EVE | 0.3190 | 0.3099 | 0.2985 | 7 → 7 → 7 |
| ESM2_all_seq / ESM2 | 0.2852 / 0.2851 | 0.2744 / 0.2739 | 0.2622 / 0.2620 | 8/9 不变 |
| ByProt / SaProt | 0.2762 / 0.2720 | 0.2693 / 0.2693 | 0.2618 / 0.2606 | **10↔11 互换** |
| ESM1v / ProGen2 / PPIformer | 0.2594 / 0.2547 / 0.1902 | — | — | 12/13/14 不变 |

**ProteinMPNN 第一名在三种口径下都不动** ⇒ 我们"打榜对手 = ProteinMPNN"的前提不受影响。
但 **rank 3–5 不稳定**，引用 BindingGYM 中段排名要小心。

### 4.3 对我们自己的工作

1. **复现口径不改。** `zeroshot_proteinmpnn_20260827-154500.md` 里的 0.391356 vs 官方 0.396950
   是在同一个 25 assay 集合上比的 —— 这是**必须**的，换集合就没法比。
2. **complexTTT 的 Δ 分析：主报 n=25 + 附 n=23 敏感性检验。**
   理由：剔 SOS1 让 ProteinMPNN 涨 0.0037，剔 DARPinK27 几乎不动（−0.0003）——
   两者不对称，**在没定案之前"剔哪一个"本身就会注入 ~0.004 的偏移（≈ MDE 0.021 的 18%）**。
   单剔其一等于替 BindingGYM 做了一个我们证据不足的判断。
3. **"免费的尺子"仍然成立，但要标成 upper bound。**
   label 固定、输入变，这一点严格成立，所以 ρ 的差只来自输入。但共享向量至多对一个
   partner 是正确的，所以 0.0948 里混了"label 与 partner 不匹配"的额外惩罚。

   同时修正一处此前的表述：这把尺子量的是**整个输入表示**的敏感度，不只是结构 ——

   | 类别 | 模型 | \|Δρ\| |
   |---|---|---|
   | 纯序列 | ESM2 / ESM1v / ProGen2 | **0.004–0.010** |
   | MSA-based | EVE / Tranception / TranceptEVE | **0.034 / 0.090 / 0.100** |
   | 结构 | ProteinMPNN / SaProt / ESM-IF1 | **0.095 / 0.132 / 0.277** |

   MSA 模型也大幅漂移，因为两边的 `.a2m`（`5O2S_hm` vs `8BE4_hm`）与 query 序列都不同。
   14 模型 mean \|Δ\| = 0.0613。

---

## 5. 复核方式（全部只读，本地可跑）

用到的数据（均为只读）：

```
/home/guoj0f/share/BindingGYM/input/Binding_substitutions_DMS/KRAS_*.csv   # 6 个 assay
/home/guoj0f/share/BindingGYM/input/structures/{5O2S_hm,8BE4_hm,1HE8_hm,1LFD_hm,6VJJ}.pdb
/home/guoj0f/share/BindingGYM/input/BindingGYM.csv                          # 25 assay 索引
/home/guoj0f/repos/BindingGYM/results/*_zero_shot_metric.csv                # 官方 14 模型结果
/home/guoj0f/share/ProteinGym/DMS_ProteinGym_substitutions/RASK_HUMAN_Weng_2022_{abundance,binding-DARPin_K55}.csv
/home/guoj0f/repos/ProteinGym/reference_files/DMS_substitutions.csv         # 源文章元数据
/home/guoj0f/repos/Sources/datasets/BindingGYM (NIPS 2024).pdf              # Table 10
/home/guoj0f/repos/BindingGYM/new_dataset_construction_guide.md             # curation 7 步
```

**mutant key 的构造（关键，否则裸 join 得 0 行重叠）：**
两个文件的 `chain_id` 分别是 `AB` 与 `RS`，`mutant` 列是 `{'A': 'A11C', 'B': ''}` 这样的
逐链字典。必须**抹掉 chain id**、把所有非空突变串拼接后排序，才能对齐：

```python
def key(s):
    d = ast.literal_eval(s)
    p = [v for v in d.values() if v]
    return ":".join(sorted(":".join(p).split(":"))) if p else "__WT__"
```

---

## 6. 未决事项与建议

| # | 事项 | 状态 |
|---|---|---|
| U1 | 确定**哪一个**文件拿错了列 | **未定案**。需 Weng et al. supplementary（`kras_fitness.xlsx`）的原始六列。定案后 D6 可从"两个都剔（n=23）"退回"剔错的那一个（n=24）"，保住一个真实 assay |
| U2 | 源文章说 screening 了 7 个 partner，BindingGYM 收了 6 个、ProteinGym 收的 K55 不在 BindingGYM 的 6 个里 ⇒ 至少 7 列存在 | 已核实 BindingGYM 侧 6 个的名字；第 7 个（K55）确认未被 BindingGYM 收录 |
| U3 | 是否向 BindingGYM 作者报告 | 建议报告（含本文档的可复现检验）。**需用户决定**，本 session 未做任何对外动作 |
| U4 | 其余 assay 是否有同类问题 | ✅ **已完成**（见 §3.6）：300 对全量扫描，75 对有实质重叠，**只有这 1 对重复**。已确认是孤例 |

