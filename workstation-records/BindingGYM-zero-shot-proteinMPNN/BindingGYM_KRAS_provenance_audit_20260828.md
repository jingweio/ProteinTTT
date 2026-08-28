# BindingGYM KRAS 数据溯源审计 —— 追到源头

> 2026-08-28 · 只读调查，未修改任何代码或数据集
> 前一份文档（问题的发现与 benchmark 内影响）：
> `ProteinTTT/.claude/worktrees/proteinTTT-proteinGYM-reproduce/workstation-records/BindingGYM-zero-shot-proteinMPNN/kras_duplicate_audit_20260828-143432.md`
> **本文档推翻了前一份的归因结论**，见 §0。

---

## 0. TL;DR —— 结论翻转

前一份文档把 `KRAS_DARPinK27` 与 `KRAS_SOS1` 的 label 重复归因为 **BindingGYM 的 curation 错误**。
拿到源头数据后，这个归因是**错的**：

| | 前一份文档的判断 | **实测后的结论** |
|---|---|---|
| 错在哪一层 | BindingGYM 的 curation 第 3 步 | **Nature 论文的 Supplementary Table 4 本身** |
| BindingGYM 的角色 | 引入错误 | **忠实传播**（读取无误，98.7–99.3% 逐点相同） |
| 哪个文件是错的 | 未定案（结构检验弱指向 K27） | **定案：`KRAS_DARPinK27` 是错的**，`KRAS_SOS1` 是对的 |
| 错误范围 | 一对 assay | **Nature ST4 的 K27 与 K55 两整块**都被 SOS1 覆写 |
| 能否修复 | 需要源数据 | **可以** —— MAVEdb / bioRxiv / CRG 原始数据都有真实 K27，覆盖 97.1% |

**一句话：Nature `s41586-023-06954-0` 的 Supplementary Table 4 里，`BindingPCA DARPin K27`
和 `BindingPCA DARPin K55` 两块数据被 `BindingPCA SOS1` 整块覆写了。真实的 K27/K55 在该文件中不存在。**

---

## 1. 谁对谁错 —— 一览

| 数据发布 | K27 | K55 | SOS1 | 其它 partner | 判定 |
|---|---|---|---|---|---|
| **原始 DiMSum 输出**（CRG，作者上传） | ✅ 真实 | ✅ 真实 | ✅ 真实 | ✅ | **干净**（最底层） |
| **bioRxiv 预印本 Supp. Table 4**（2022-12-08, v1） | ✅ 真实 | ✅ 真实 | ✅ 真实 | ✅ | **干净** |
| **MAVEdb `urn:mavedb:00000115`** | ✅ 真实 | ✅ 真实 | ✅ 真实 | ✅ | **干净** |
| **Nature Supp. Table 4**（正式发表版） | ❌ = SOS1 | ❌ = SOS1 | ✅ 真实 | ✅ | **污染** |
| **BindingGYM**（读 Nature ST4） | ❌ = SOS1 | 未收录 | ✅ 真实 | ✅ | 忠实传播了污染 |
| **ProteinGym**（读另一版本） | 未收录 | ✅ 真实 | 未收录 | — | **未受影响** |

⇒ 污染**只存在于 Nature 的正式发表版补充材料**。预印本是干净的，说明是**从预印本到正刊的
生产环节**引入的。

---

## 2. 数据源与获取路径（全部可复核）

| 来源 | 获取方式 | 落盘 |
|---|---|---|
| Nature Supp. Tables 3/4/5 | `media.springernature.com/.../41586_2023_6954_MOESM{4,5,6}_ESM.xlsx`（CC-BY open access） | `ST3/ST4/ST5.xlsx`，ST4 = 191,752 × 13 |
| bioRxiv v1 Supp. Tables 1–5 | `biorxiv.org/content/biorxiv/early/2022/12/08/2022.12.06.519122/DC{1..5}/embed/media-{n}.xlsx`；**只有 v1，无 v2/v3** | `biorxiv/` |
| MAVEdb | `api.mavedb.org/api/v1/experiment-sets/urn%3Amavedb%3A00000115` → 28 个 score set（21 fitness = 7 assay × 3 block，+7 个 ΔΔG）；每个 `/scores` 取 CSV | `mavedb/kras_mavedb_fitness_long.csv`（174,794 行） |
| **原始 DiMSum 输出** | GitHub `lehner-lab/krasddpcams` → README "Required Data" → CRG OneDrive 匿名共享 → SharePoint REST API → **23 个 `CW_RAS_*.RData`**（每 assay × 每 block 一个） | `krasddpcams_DATA/`（26 MB），合并后 202,173 × 35 |

> ⚠️ 全部写在 `/tmp/claude-224072/kras_src/`，**未进入任何 repo，未改动任何既有数据**。

**关键元数据**（Nature 页面 Data availability）：
```
DNA sequencing  : SRA BioProject PRJNA907205
Fitness & ΔΔG   : Supplementary Tables 4 and 5；MAVEdb urn:mavedb:00000115
Code            : github.com/lehner-lab/MoCHI, github.com/lehner-lab/krasddpcams
处理管线         : DiMSum v1.2.9
```

---

## 3. 证据链 (1)：Nature ST4 本身被污染

### 3.1 行数就已经异常

ST4 的 9 个 assay：

```
AbundancePCA                          27615
BindingPCA RAF1RBD                    26820
BindingPCA RALGDSRBD                  22771
BindingPCA SOS1                       22096   ┐
BindingPCA DARPin K27                 22096   ├─ 三者行数完全相同
BindingPCA DARPin K55                 22096   ┘   连分 block 计数都一样 (12705/6001/3390)
BindingPCA PIK3CGRBD                  21982
BindingPCA full length RAF1           13402
BindingPCA RAF1RBD coexpression GAP   12874
```

### 3.2 全部 8 个数值列逐点相同

按 `(block, aa_seq)` 对齐，21,950 个共同变体：

| 列 | K27~K55 | K27~SOS1 | K55~SOS1 | 对照 RAF1RBD~RALGDSRBD |
|---|---|---|---|---|
| `fitness` | 21950/21950 (100%) | 21950/21950 | 21950/21950 | **0**/21690 |
| `sigma` | 100% | 100% | 100% | 0 |
| `growthrate` | 100% | 100% | 100% | 0 |
| `growthrate_sigma` | 100% | 100% | 100% | 0 |
| `nor_gr` | 100% | 100% | 100% | 0 |
| `nor_gr_sigma` | 100% | 100% | 100% | 0 |
| `nor_fitness` | 100% | 100% | 100% | 0 |
| `nor_fitness_sigma` | 100% | 100% | 100% | 0 |

连 WT 行都一样：三者的 `fitness` 都是 `[0.00685532580079933, 0.00313482769024313, -0.0077702339915976]`
（三个 block），而 RAF1RBD 是 `[0.000785…]`、RALGDSRBD 是 `[0.001447…]`。

⇒ **不是"某一列算错"，是整块数据被替换。**

### 3.3 污染是在「预印本 → 正刊」的修订中引入的 —— 行数算术精确闭合

两版 ST4 并排（本地已存两份原件，见 §11）：

| assay | **bioRxiv v1** (2022-12-08) | **Nature** (published) | |
|---|---|---|---|
| `BindingPCA DARPin K27` | **28,209** | **22,096** | ← 被 SOS1 覆写 |
| `BindingPCA DARPin K55` | **26,404** | **22,096** | ← 被 SOS1 覆写 |
| `BindingPCA SOS1` | 22,096 | 22,096 | 未变 |
| `AbundancePCA` | 27,615 | 27,615 | 未变 |
| `BindingPCA RAF1` → `RAF1RBD` | 26,820 | 26,820 | 仅改名 |
| `BindingPCA RALGDS` → `RALGDSRBD` | 22,771 | 22,771 | 仅改名 |
| `BindingPCA PIK3CG` → `PIK3CGRBD` | 21,982 | 21,982 | 仅改名 |
| `BindingPCA full length RAF1` | — | **13,402** | ← 新增 |
| `BindingPCA RAF1RBD coexpression GAP` | — | **12,874** | ← 新增 |
| **总行数** | **175,897**（7 assay） | **191,752**（9 assay） | |

```
  bioRxiv 总行数                175,897
  − K27 被覆写损失 (28209-22096)  −6,113
  − K55 被覆写损失 (26404-22096)  −4,308
  + 新增 full length RAF1        +13,402
  + 新增 RAF1RBD coexpr GAP      +12,874
  ────────────────────────────────────────
  =                             191,752      Nature 实际 191,752   ✅ 精确吻合
```

⇒ **这一版修订同时做了三件事：新增 2 个 assay、给 3 个 assay 改名、把 K27 与 K55 两块换成了 SOS1。**
行数差在个位上闭合，说明不是"部分数值被改写"，而是**整块替换** —— 与重新组装该表时
两个 assay 的数据源指错了这一解释完全一致。

**这也解释了为什么预印本是干净的**：污染是重投/修订环节的产物，与原始实验和 DiMSum 处理无关。

---

## 4. 证据链 (2)：三个独立来源都是干净的

### 4.1 原始 DiMSum（最底层）

23 个 `.RData`，文件名即 assay × block：

```
CW_RAS_binding_K27_{1,2,3}_fitness_replicates_fullseq.RData   → 28,209 行
CW_RAS_binding_K55_{1,2,3}_fitness_replicates_fullseq.RData   → 26,404 行
CW_RAS_binding_SOS_{1,2,3}_fitness_replicates_fullseq.RData   → 22,096 行
```

**行数就互不相同**，而 Nature ST4 里三者都是 22,096（= SOS1 的行数）。

### 4.2 bioRxiv 预印本 Supp. Table 4

行数：K27 **28,209** / K55 **26,404** / SOS1 **22,096** —— 与原始 DiMSum 完全一致。
逐列检查：8 个数值列**没有一个** byte-identical。
示例（block1 WT）：`fitness` K27 = 0.04916533，K55 = 0.00420306，SOS1 = 0.00685533 —— 三个不同的数。
对照 RAF1 vs RALGDS：r = 0.9083，identical = False（管线既能识别"相似但不同"，也能识别"不同"）。

### 4.3 MAVEdb

21 个 fitness score set（7 assay × 3 block），block1 计数：K27 = 14,859 / K55 = 12,607 / SOS1 = 12,637。
两两比较**没有任何一对**存在 bit-identical 值：

| pair | ov | eq | pearson |
|---|---|---|---|
| K27 vs K55 | 24,391 | **0** | +0.7188 |
| K27 vs SOS1 | 21,191 | **0** | +0.6704 |
| K55 vs SOS1 | 20,150 | **0** | +0.7412 |
| （其余 18 对） | — | **0** | 0.47–0.91 |

ΔΔG score set 同样三者不同（K55~K27 r=0.4225、K55~SOS1 r=0.4943、K27~SOS1 r=0.4391）。

---

## 5. 证据链 (3)：定案 —— 被复制进去的是 **SOS1**

把 ST4 那组三元向量分别与原始 DiMSum / MAVEdb 的各 partner 比：

| ST4 的块 | vs 真实 K27 | vs 真实 K55 | vs 真实 SOS1 |
|---|---|---|---|
| `BindingPCA DARPin K27` | r = +0.6492, **eq = 0** | r = +0.6999, **eq = 0** | **r = +1.000000, eq = 21,889/21,889** |
| `BindingPCA DARPin K55` | r = +0.6492, eq = 0 | r = +0.6999, eq = 0 | **r = +1.000000** |
| `BindingPCA SOS1` | r = +0.6492, eq = 0 | r = +0.6999, eq = 0 | **r = +1.000000** |

用 MAVEdb 的 `nor_fitness` 口径同样成立（21,712/21,947 逐点相同，r = 1.0000）。

⇒ **ST4 的三块都是 SOS1 的测量。真实的 K27 与 K55 在 Nature 补充材料中完全缺失。**

（这也推翻了前一份文档里那个"结构界面检验方向性偏向 K27"的弱提示 —— 当时 p = 0.208、
且方法在 RAF1-RBD 对照上失效，已明确标注为不可据此定案。事实是相反的：SOS1 才是对的那个。
**记为一次结构启发式判据失败的实例。**）

---

## 6. BindingGYM 做对了什么、做错了什么

### 6.1 做对的（复核后确认）

**(a) 六个 KRAS assay 都是对 ST4 `nor_fitness` 的忠实复制**

| BindingGYM assay | 对应 ST4 assay | 位点坐标系 | ov | 逐点相同 | r |
|---|---|---|---|---|---|
| `KRAS_DARPinK27_..._5O2S` | BindingPCA DARPin K27 | 0 | 19,533 | 19,280 (98.7%) | 1.0000 |
| `KRAS_SOS1_..._8BE4` | BindingPCA SOS1 | 0 | 19,425 | 19,176 (98.7%) | 1.0000 |
| `KRAS_RALGDS-RBD_..._1LFD` | BindingPCA RALGDSRBD | 0 | 20,341 | 19,843 (97.6%) | 1.0000 |
| `KRAS_PICK3CG-RBD_..._1HE8` | BindingPCA PIK3CGRBD | 0 | 19,203 | 18,858 (98.2%) | 1.0000 |
| `KRAS_RAF1-RBD_..._6VJJ` | BindingPCA RAF1RBD | **−1** | 23,162 | 22,498 (97.1%) | 1.0000 |
| `KRAS_RAF1_..._6VJJ` | BindingPCA full length RAF1 | **−1** | 12,677 | 12,582 (99.3%) | 1.0000 |

剩余 1–3% 非逐点相同：ST4 里同一个 `(block, aa_seq)` 存在重复行，取值规则不同所致，非错误。

**(b) 25 个 assay 全部内部自洽 —— 0 个问题**

对每条 variant 校验 `mutant` 串 ↔ `wildtype_sequence` ↔ `mutated_sequence`：

```
25 个 assay，1,133,000+ 个 substitution
  WT 残基不符 : 0
  突变残基不符 : 0
  位点越界     : 0
  链定义不符   : 0
  DMS_score NaN: 0
```

**(c) mutant list 的 per-structure 过滤是正确的**
`KRAS_DARPinK27`（5O2S，165 aa）保留位点 118–120、砍掉 166–168；
`KRAS_SOS1`（8BE4，168 aa 但 118–120 未解析为 `XXX`）反之。逻辑正确。

**(d) 全量重复扫描：只有这一处**
25 个 assay 的 300 对全扫（修正 6VJJ 坐标系后，77 对有 ≥50 共同 variant）：
只有 `DARPinK27 == SOS1` 一对重复。同体系近邻对全部通过：

| 近邻 pair | ov | eq | r |
|---|---|---|---|
| GB1_1FCC vs GB1_1FCC_2016 | 160 | 0 | +0.9789 |
| KRAS_RAF1 vs RAF1-RBD（同 6VJJ） | 12,086 | 0 | +0.9267 |
| KRAS_RAF1-RBD vs RALGDS-RBD | 19,520 | 0 | +0.9018 |
| Z-domain ZpA963 HL1 vs HL2（同 2M5A） | 143 | 0 | +0.8782 |
| PSD95_CRIPT vs PSD95_Tm2F（同 1BE9） | 1,577 | 0 | +0.4795 |

### 6.2 真正可归到 BindingGYM 的问题（两条，都不严重）

**(i) 一致性核对未覆盖"跨 assay 重复"。**
`new_dataset_construction_guide.md` 第 5 步写明 *"We compare the results with the main findings of
the article to ensure consistency and reduce unnecessary errors"*。若这一步包含一次跨 assay 的
pairwise 相等性检查（成本 <1 分钟），必然拦下。**这是唯一可归责的环节** —— 不是引入错误，
而是**没有拦下上游错误**。

**(ii) 位点坐标系在 assay 之间不统一。**
两个 6VJJ assay 的参考序列多一个前导 `G`（`GMTEYKLVVVGA…`，168 aa，来自 6VJJ 构建体），
位点编号整体 +1；其余四个用规范 KRAS 编号（`MTEYKLVVVGAG…`）。

- **内部完全自洽**（7,364 个 substitution，0 个不符），**不影响 assay 内排序**，因此**不影响 zero-shot 榜单**。
- 但会影响任何**跨 assay 汇总位点**的分析（inter-assay split、位点级统计），也会让
  naive 的跨 assay 变体对齐严重低估重叠（本审计中重叠从 ~140 跳到 11k–19k）。

### 6.3 不是问题的（我自己踩的坑，记录以免复现）

- **Z-domain 4 个 assay 的"重复 mutant"是假象。** 它们**两条链都突变**，
  `{'A':'L17I','B':'I31V:L35V'}` 与 `{'A':'L17I:I31V','B':'L35V'}` 是不同分子，
  被我"抹掉 chain id"的 key 合并了。⇒ **该 key 只在单链突变的 assay 上安全。**
- **`full length RAF1` 一度"对不上"是我的 parsing bug** —— 它的 `aa_seq` 长度是 **63**（block1 覆盖残基 2–64），
  不是 187，被我的长度检查静默丢弃。

---

## 7. ProteinGym 未受影响

ProteinGym 从同一篇文章收了 2 个 assay（`RASK_HUMAN_Weng_2022_binding-DARPin_K55` 与 `_abundance`），
其 `reference_files/DMS_substitutions.csv` 记 `raw_DMS_filename: kras_fitness.xlsx`。实测：

| ProteinGym assay | vs 真实 K55 | vs 真实 K27 | vs 真实 SOS1 |
|---|---|---|---|
| `binding-DARPin_K55` | **r = +1.000000** (n=24,873) | +0.6779 | +0.6903 |
| `abundance` | — | — | — |
| `abundance` vs 真实 abundance | **r = +1.000000** (n=26,012) | | |

⇒ ProteinGym 用的是**未污染的版本**（预印本或原始 DiMSum 谱系），其 K55 是真实的 K55。
**ProteinGym 侧无需任何修正。**

> 附带口径差异：ProteinGym 用的是 `fitness` 列，BindingGYM 用的是 `nor_fitness`（分 block 归一）列。
> 两者对同一 assay 的 ρ 可差 ~0.037（本审计实测，见 §8 脚注），做跨 benchmark 对比时不能混。

---

## 8. 影响量化 —— 实测重算

用我们自己复现跑出的 per-variant ProteinMPNN 分数（`seed1_M5`，25/25 assay 齐全），
把 `KRAS_DARPinK27` 的 label 换成 MAVEdb 的真实 K27（同 `nor_fitness` 口径、同 block1 优先规则）。

**先验证程序正确性**（对照组：BG label 本来就对，换成同口径的真实值应几乎不变）：

| assay | 现用 ρ | 同子集 ρ | 修正后 ρ | Δ | n |
|---|---|---|---|---|---|
| SOS1（对照） | +0.3095 | +0.3094 | +0.3094 | **−0.0001** ✅ | 19,424 |
| RALGDS-RBD（对照） | +0.5872 | +0.5872 | +0.5872 | **+0.0000** ✅ | 20,340 |
| **DARPin K27（修正对象）** | +0.4029 | +0.4046 | **+0.4079** | **+0.0033** | 18,974 |

**结论：ProteinMPNN 在这个 assay 上 ρ 从 0.4046 → 0.4079，聚合影响 ≈ +0.0033/25 = +0.00013。**
即 **BindingGYM 的 headline 0.3970 基本不受影响**（远小于其 bootstrap 标准误 0.03）。

> 校准细节：MAVEdb `score` ≡ ST4 `nor_fitness`（max|diff| ~1e-16）。
> "block1 优先"规则在 SOS1 上复现了 BG 的 19,367/19,424 = 99.7% 的取值，故用同一规则构造 K27。
> 脚注：若误用原始 `fitness` 列而非 `nor_fitness`，对照组 SOS1 会假性变动 +0.0369 —— 比要测的效应还大。

**但"数值影响小"不等于"问题不重要"：**
这个 assay 在过去一年里以 "KRAS–DARPin K27 binding" 的名义被 14 个模型评测、
被 inter-assay split 当作一个独立的 target 使用，而它测的其实是 KRAS–SOS1。
**它的语义是错的，即使它的数值恰好不改变排名。**

---

## 9. 一个意外发现：ProteinMPNN 在这批数据上基本是 **partner-blind** 的

既然拿到了 6 个 partner 的真实 landscape，就能做一个之前做不了的检验：
**用某个复合物结构打出来的分，去预测各个 partner 的真实 landscape。**
若模型真在建模 partner-specific 的界面能量学，**对角线应显著更高**。

（每一行内部是同一组模型分数、只换 label，所以行内比较是干净的；跨行不可比。）

| 打分所用结构 | K27 | SOS1 | RALGDS | RAF1RBD | PIK3CG | K55 | **abundance** |
|---|---|---|---|---|---|---|---|
| **5O2S** (KRAS–DARPin K27) | *0.4079* | 0.4028 | 0.3953 | **0.4345** | 0.3187 | 0.4330 | 0.3711 |
| **8BE4** (KRAS–SOS1) | 0.3213 | *0.3094* | 0.3653 | 0.3708 | 0.3252 | 0.3808 | **0.3946** |
| **1LFD** (KRAS–RALGDS) | 0.3749 | 0.4720 | ***0.5872*** | 0.5300 | 0.5374 | 0.4874 | 0.3626 |

*斜体 = 对角线（结构与 partner 匹配）*

- **3 行里只有 1 行（1LFD）对角线最高。**
- 5O2S 行：预测 **RAF1RBD（0.4345）比预测它自己的 K27（0.4079）还准**。
- 8BE4 行：预测 **abundance（0.3946，纯折叠稳定性，与结合无关）比预测 SOS1（0.3094）准得多**。

⇒ **ProteinMPNN 在 BindingGYM-KRAS 上的 zero-shot 信号，主要来自 KRAS 自身的内在突变约束
（折叠稳定性 + 效应叶通用约束），而非 partner-specific 的界面能量学。**

**这直接冲击 complexTTT 的核心前提。** 计划里假定"把模型往 WT-complex 结构上推能提升
partner-specific 预测"—— 但预训练模型在这批数据上根本没在做 partner-specific 预测。
在设计 S1/S2 之前，这条必须先正面处理。

> 限定：只用了 3 个结构、1 个模型、1 个 seed，且三个结构的建模质量不同（`_hm` 同源模型 vs 晶体）。
> 这是一条**需要扩大验证**的观察，不是定论。扩到 6 个结构 × 多 seed 的成本约 1.5 GPU-h。

---

## 10. 修复方案

**可以完整修复，且不需要重跑任何 GPU。**

1. `KRAS_SOS1_norfitness_8BE4` —— **不动**，它是对的。
2. `KRAS_DARPinK27_norfitness_5O2S` —— label 列替换为 MAVEdb `urn:mavedb:00000115` 的真实
   DARPin K27 `score`（同 `nor_fitness` 口径），覆盖 **18,974 / 19,533 = 97.1%**，
   未覆盖的 559 个变体删除或标 NaN。
3. 可选补一个 `KRAS_DARPinK55_*` assay（MAVEdb 有完整数据，BindingGYM 原本就漏收了这个 partner）。
4. 评估口径：主结果仍报 n=25 以对齐官方；另报"修正 K27 后"的 n=25 作为并列结果
   （**前一份文档建议的 n=23 敏感性检验现在不需要了** —— 既然定了案，直接修复比剔除更好）。

**对我们自己工作的更新：**
- `zeroshot_proteinmpnn_20260827-154500.md` 的复现口径**不用改**（同官方 25 个才可比）。
- complexTTT plan 的 **D6 可以关闭**：不再需要"两个都剔"，改为"修复 K27"。
- complexTTT plan 需要新增一条 blocking 项：**§9 的 partner-blindness**。

---

## 11. 复核方式

**已归档到 `Sources/datasets/` 的两份原件**（可长期引用）：

| 文件 | 大小 | md5 | 内容 |
|---|---|---|---|
| `Weng2024_KRAS_ddPCA_SupplTable4__NATURE-published.xlsx` | 23,540,334 B | `f03fe18f33777766669039cf9a19badc` | 191,752 × 13，9 assay，**含污染** |
| `Weng2024_KRAS_ddPCA_SupplTable4__BIORXIV-preprint-v1.xlsx` | 22,035,507 B | `228e2988f102070ef7d3cb049b7e4a3d` | 175,897 × 13，7 assay，**干净** |

两者结构相同：sheet = `['README','TableS4']`，13 列
`block, aa_seq, Nham_aa, WT, fitness, sigma, growthrate, growthrate_sigma, nor_gr, nor_gr_sigma, nor_fitness, nor_fitness_sigma, assay`。

下载 URL：
```
Nature  : https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-023-06954-0/MediaObjects/41586_2023_6954_MOESM5_ESM.xlsx
bioRxiv : https://www.biorxiv.org/content/biorxiv/early/2022/12/08/2022.12.06.519122/DC4/embed/media-4.xlsx?download=true
```
（Nature 版为 CC-BY open access；bioRxiv 为预印本公开材料）

其余源数据（未归档，留在 scratchpad）：
```bash
/tmp/claude-224072/kras_src/ST4.xlsx                                   # 同上 Nature 版
/tmp/claude-224072/kras_src/biorxiv/                                   # bioRxiv v1 Supp. Tables（干净）
/tmp/claude-224072/kras_src/mavedb/kras_mavedb_fitness_long.csv        # MAVEdb 21 score sets（干净）
/tmp/claude-224072/kras_src/krasddpcams_DATA/*.RData                   # 原始 DiMSum 23 个文件（干净）
/tmp/claude-224072/kras_src/out/kras_singles_fitness_matrix.csv        # 3,320 单点 × 9 assay 宽表
/tmp/claude-224072/kras_src/out/PROVENANCE.txt                         # URL / 日期 / 映射
```

**两个必须注意的对齐陷阱：**

1. **`aa_seq` 单独不是唯一键** —— 同一序列在 block1/2/3 都会出现。必须用 `(block, aa_seq)`。
2. **BindingGYM 的 `mutant` 是逐链字典**（`{'A': 'A11C', 'B': ''}`），且 chain id 各 assay 不同
   （`AB` vs `RS`）。必须抹掉 chain id 才能跨 assay 对齐：
   ```python
   def key(s):
       d = ast.literal_eval(s); p = [v for v in d.values() if v]
       return ":".join(sorted(":".join(p).split(":"))) if p else "__WT__"
   ```
   ⚠️ **此 key 只在单链突变的 assay 上安全**（Z-domain 4 个 assay 两条链都突变，会被错误合并，见 §6.3）。
3. **6VJJ 的两个 assay 需要 shift −1** 才能与其余 assay / 源表对齐（见 §6.2-ii）。
4. 源表 `aa_seq` 长度有两种：**187**（缺首位 M，覆盖残基 2–188）与 **63**（`full length RAF1`，
   block1，覆盖残基 2–64）。按长度硬过滤会静默丢数据。

---

## 12. 未决与建议

| # | 事项 | 状态 |
|---|---|---|
| V1 | 向 **Nature / 通讯作者**报告 ST4 的 K27/K55 被 SOS1 覆写 | **建议做**，证据链完整且可复核。需用户决定，本 session 未做任何对外动作 |
| V2 | 向 **BindingGYM 作者**报告（附修复方法：改用 MAVEdb） | 建议做，同上 |
| V3 | ProteinGym 是否需要通知 | **不需要**，其 K55/abundance 均为真实值 |
| V4 | §9 partner-blindness 扩大验证（6 结构 × 多 seed × 多模型） | 待定，约 1.5 GPU-h |
| V5 | `BindingPCA full length RAF1` 与 `RAF1RBD coexpression GAP` 无 MAVEdb 对应，仅存在于 ST4/原始 DiMSum | 已确认原始 DiMSum 有（13,402 / 12,874 行），可交叉验证，本次未做 |
| V6 | 其余 24 个 assay 是否也存在"上游源数据错误" | **未做** —— 本次只穷举了 BindingGYM 内部的重复，没有逐一回溯 20 篇源文章 |
