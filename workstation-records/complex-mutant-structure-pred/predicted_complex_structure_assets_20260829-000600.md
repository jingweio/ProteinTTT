# complex-mutant-structure-pred — investigation record

（created 2026-08-29 00:06; status: **DONE**；无 GPU 消耗，纯资产调研 + 本地/在线核验）

## 调研问题（用户三问）

1. **AlphaFold 与 ESMFold 官方是否 release 了 predicted-complex-structure 的 data asset？**
   （注意区分：release 了 *模型/权重/代码* ≠ release 了 *批量预测产物*。用户问的是后者。）
2. 若有，**是否覆盖 BindingGYM 里的 mutation？**
3. 具体到一个 BindingGYM mutant，**有没有能方便定位的 asset？**

---

## 0. 一页结论

| 问题 | 答案 |
|---|---|
| **AlphaFold 官方有 complex asset 吗？** | **有，但是 2026-03 才有**，且不是 AF3。AFDB 在 2026-03-16 首次加入预测复合物（NVIDIA + Steinegger 组 + DeepMind + EMBL-EBI），用的是 **AlphaFold-Multimer v2.3.0 权重、OpenFold/ColabFold 推理**。规模：~7.59 M heterodimer + ~21–23 M homodimer，其中仅 **80,248** 条 heterodimer 过质量阈值、可单条 URL 直取，其余只能从 FTP 的 shard tar 里挖。 |
| **ESMFold 官方有 complex asset 吗？** | **没有，而且结构上不可能有。** 两代 Atlas（Meta 617 M / Biohub 2026-05 的 1.1 B）都是严格 one-sequence→one-structure 的**单体**库；Biohub 的 Lance schema 只有一个 `sequence` 列、一个 `ptm`、**没有 `iptm`、没有 chain/partner 列**。ESMFold2（MIT，开权重）**能**预测复合物，但那是模型不是 asset。 |
| **覆盖 BindingGYM 的 WT 复合物吗？** | 部分。AFDB complex 层在 **UniProt-pair 层面**命中 BindingGYM 的 **6/22 个复合物结构（7/25 assay）**，但都是 full-length canonical 构建体，与 BindingGYM 的截断链对不上。 |
| **覆盖 BindingGYM 的 mutation 吗？** | **376,446 个 variant 里，官方 AlphaFold/ESMFold 资产覆盖 0 个。** 二者都以 UniProt accession（或 canonical 序列 MD5）为键，schema 里**没有 variant 字段**，是构造性的零覆盖，不是"暂时没做"。 |
| **那有没有别的资产覆盖到 BindingGYM mutant？** | **有，且只有一个：BindingGYM 自己 Zenodo 的 v1**（record `12200340`，被 v2 顶掉、README/论文/仓库都不指向它）——里面是 `4D5_HER2_fitness_1N8Z` 的 **2,080 个 FoldX mutant complex**，1:1 覆盖该 assay 全部 variant。占全 benchmark **2,080 / 376,446 = 0.55%**。 |
| **能方便定位吗？** | AFDB complex：**能**（`GET /api/complex/{UniProt}` 一次调用，但 accession 是不可推导的 16 位数字）。BindingGYM v1 的 mutant 结构：**能，但没有任何官方索引**——我实测出映射键（见 §7）。其余：无。 |

> **一句话**：官方那边，AlphaFold 今年确实第一次有了 complex asset，ESMFold 没有也不会有；但两者对 BindingGYM 的**突变体**是**零覆盖且构造性零覆盖**。唯一真的能直接拿到 BindingGYM mutant complex 坐标的，是 BindingGYM 自己一个被淹没的 Zenodo 旧版本，只覆盖 25 个 assay 中的 1 个。

---

## 1. 方法与证据等级

- **本地实测**（我自己跑的，最高可信）：BindingGYM 结构覆盖审计、Zenodo zip 的中央目录解析与逐条比对、AbBiBench 8.2 GB 压缩包的完整流式清点、序列/突变串交叉比对。脚本在 `sh/`，产物在 `refs/`。
- **在线核验**（我自己 curl 的）：Zenodo versions API、HuggingFace repo/CSV、GitHub API 与源码、AbBiBench tar 的 gzip 流解压。
- **workflow 调研**（16 个 agent，8 维度 × survey→adversarial verify，2.25 M token / 1,107 次工具调用，0 error）：AFDB、AF-Multimer/AF3、ESMFold、第三方 complex DB、mutant-structure asset、BindingGYM 溯源、DIY 预测器、逐链覆盖表。**所有关键 URL 由 verify agent 独立复取**。
- ⚠️ 我的知识截止是 2026-05，而 AFDB 的 complex release 是 2026-03/05 — 这次是**靠强制 live 核验才抓到的**，凡是 2026 年的状态我都要求 agent 给出实取证据。

**本记录中凡标 ✅ 的结论是我本人复核过的；标 ⚠️ 的是 agent 报告但我未独立复核的。**

---

## 2. BindingGYM 到底需要覆盖什么（全部本地实测 ✅）

脚本：`sh/audit_bindinggym_structure_coverage.py`；逐 assay 表：`refs/bindinggym_structure_coverage.csv`

- **25 个 benchmark assay，22 个结构文件，20 个 distinct PDB 模板，376,446 个 variant。**
- 每个 assay 都是 **2–3 链复合物**；DMS 只突变其中 1 条链（4 个 Z-domain + 4D5/5A12 抗体那几个突变 2 条）。
- **✅ DMS 突变位点在 BindingGYM 自带的 WT 复合物里 100% 有坐标（2,220 / 2,220 个 (assay,chain,pos)）**，且 DMS 的 WT 氨基酸与 `wildtype_sequence` **100% 吻合**。
- **✅ `wildtype_sequence` 里的 `X` 就是结构中无坐标残基的占位符** —— 22 个文件、53 个 (assay,chain) 全部满足「已解析残基 == 所有非-X 位点，按序」，零反例。

> **推论（很重要）**：BindingGYM 的 **WT 侧结构需求已经被它自己完整满足**。任何外部 predicted-complex asset 在 WT 侧的边际价值都很小。**真正的缺口只在 mutant 侧。**

### 2.1 `_hm` 到底是什么（本地 + 论文双向坐实 ✅）

- 22 个文件全部 **header 被剥、B-factor 与 occupancy 归零**，provenance 读不出来。
- 论文原文（`new_dataset_construction_guide.md` 第 6 条 + NeurIPS 2024 workshop PDF）：
  > "For sequences that do not precisely match with the sequences in the crystal structures, we employ **homology modeling using BioPython and OpenMM**."
- ⚠️ agent 逐原子比对 RCSB 的结论：**所有 22 个文件的 N–CA–C 主链与沉积条目逐原子一致（0.000 Å）**，`_hm` 的真实含义是"把论文的 reference 序列 thread 到晶体骨架上、只重建侧链"。改动的残基数：`1N8Z_hm` 179、`5WER_hm` 76、`1LFD_hm` 19…；7 个不带 `_hm` 的改了 0 个。
- ⚠️ 但 verify agent 把 headline 判为 OVERSTATED：**只有 N/CA/C 逐原子一致**；羰基 O 有 97.93% 动过（最大 2.20 Å）、CB 99.52%、其余侧链 100%（最大 5.94 Å）。**这条有操作后果**——BindingGYM 的 `protein_mpnn_utils.py` 用 `sidechain_atoms = ['N','CA','C','O']`，被重建的 O 会进模型特征。
- **✅ 仓库里没有任何 AlphaFold / ESMFold / 预测结构**（grep 只命中 vendored 的 baseline 代码）。
- **✅ 15 个带 `_hm`、7 个不带**（我最初目测的 16/6 是错的）。
- ⚠️ **所有 HETATM 被删** ⇒ 六个 KRAS 复合物的 GDP/GNP 核苷酸全部缺失（114,341 variants ≈ 30% 的 benchmark）。任何重新折叠/重新对接若把核苷酸加回来，就不再与已发表 baseline 可比。

### 2.2 所有 baseline 都在单一 WT backbone 上打分（⚠️ agent 逐行核过源码）

ProteinMPNN 每个 assay 只 `parse_PDB` 一次并缓存，只覆写 `S`；ESM-IF1 直接 `coords_list = [coords] * len(seq_list)`；SaProt 用 WT 的 3Di token 配 mutant 氨基酸；`training/dataset.py` 同理。**已发表的 BindingGYM 全部数字都是在 rigid-WT-backbone 假设下产生的。**

---

## 3. Q1 — AlphaFold 官方：有，但今年才有，且不是 AF3

| | |
|---|---|
| **资产** | AFDB "NVIDIA dataset" predicted protein complexes |
| **入口** | `https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvda/` |
| **时间** | 2026-03-16 首发（1.7 M 高置信 homodimer），2026-05-19 更新（加 ~80 k 高置信 heterodimer）；`heterodimer_metadata.csv` last-modified 2026-07-17 |
| **规模** | ⚠️ heterodimer metadata CSV = 2,454,028,021 bytes / **7,594,357 行**，其中 `passes_quality_threshold=true` **80,248** 条；homodimer metadata 5.6 GB |
| **模型** | ⚠️ CIF 内的 `_ma_software` 写明 **AlphaFold-Multimer v2.3.0 权重**，推理走 **OpenFold-TRT+cuEq / ColabFold 1.6.0**，`providerId=NVIDIA`。**不是 AF3，也不是 DeepMind 跑的** |
| **候选来源** | STRING v12.0 physical links，限定 16 个 model organism + 30 个 WHO global-health proteome；**只做 dimer** |
| **许可** | CC-BY 4.0 |

**几条必须写清楚的限定：**

- ⚠️ **只有 dimer**，没有 trimer 及以上 ⇒ BindingGYM 的抗体体系（Fab 本身就是 H+L 两条链，加抗原 = 3 链：1N8Z / 4ZFF / 4ZFG）**结构上无法表示**。
- ⚠️ **99.47% 是同物种对**（40,427 / 7,594,357 = 0.53% 跨物种）⇒ SARS2 spike × 人 ACE2、菌源 protein A/G × 人 IgG 这类跨物种界面基本被排除。
- ⚠️ **还有第四类缺口，不是上面任何一条能解释的**：CXCR4 (P61073) 与 CXCL12 (P48061) 在全表 **0 行**；YAP1 (P46937) 0 行（而它的 partner WBP1 有 61 行）；DLG4 有 271/284 行但从不与 CRIPT 配对。**"人源 = 被覆盖"是错的。**
- ⚠️ **AlphaFold 3 本身没有任何批量预测资产**。变化的是获取方式：代码 2026-06-09 由 CC BY-NC-SA 改为 **Apache 2.0**；权重 2026-07-23 起**免申请直链下载**（`storage.googleapis.com/alphafold3/af3.bin.zst`），但**权重条款未放开**——仅限非商业组织、不得再分发、**且输出不得用于训练结构预测模型**（见 §10 的许可闸门）。
- ⚠️ **AlphaFold Server 不是替代路线**：几十 jobs/天（第三方来源写 30/day，官方页是 JS SPA 取不到正文，此数字标 unconfirmed），且 ToS 禁止用输出训练结构预测模型。
- ⚠️ **AlphaMissense 是最容易被误读的东西**：它只出**分数**，不出结构。AFDB FAQ-22 明说它 "does not predict the change in protein structure ... upon mutation"。网上那个"AlphaFold structures with AlphaMissense scores"（Zenodo 10255502）是**第三方**把分数写进 B-factor 列的 WT 单体，不是 mutant 结构。

---

## 4. Q1 — ESMFold 官方：没有，而且结构上不可能有

| 资产 | 状态 |
|---|---|
| **Meta ESM Metagenomic Atlas**（617 M，v2023_02） | ⚠️ 网站/`fetchPredictedStructure`/`foldSequence`/多 GB bulk 下载 **2026-08-28 仍活**；但 **MMseqs2 序列检索与 Foldseek 结构检索已挂**（`searchSequence`/`searchStructure` 返回 503/502）⇒ **知道 MGYP 号能取，但没法按序列反查**。严格单体、纯 MGnify 天然序列。 |
| **Biohub ESM Atlas v1**（2026-05-27，~1.1 B 结构 / 6.8 B 序列） | ⚠️ Lance schema 只有一个 `sequence`、一个 `ptm`，**无 `iptm`、无 chain 列** ⇒ 一序列一结构的单体库，schema 层面的铁证。 |
| **ESMFold 的 multimer 能力** | ⚠️ 是**官方代码默认**（`esmfold.py`: `chain_linker="G"*25`, `residue_index_offset=512`），不是社区 hack。但 **Meta 从未用它发布任何 complex 资产**，也从未把 ESMFold 当 multimer predictor 做过评测；**托管 API 直接拒绝** `:` 多链语法（HTTP 422），且硬上限 400 残基。 |
| **ESMFold2**（Biohub，2026-05，MIT 开权重） | ⚠️ **真的能做复合物**（返回 `result.iptm` / `result.complex`），自称在 Foldbench 的 protein-protein 与 antibody-antigen 上 DockQ pass-rate 超越同类。但这是**模型**，Biohub 在 billion 规模上**只跑了单体模式**。⚠️ 另需注意：其 model card 说训练数据含 **PDB + AFDB** ⇒ BindingGYM 的 22 个结构几乎肯定在训练集里，重预测有 leakage 顾虑。 |

**结论：ESMFold 家族对 BindingGYM 的 WT 复合物和 mutant 都是 0 覆盖**，且前者是结构性的（单体模型 → 单体库）。

> ⚠️ 一条 verify agent 挖出的细微修正：Biohub Atlas 的**序列层**（6.8 B）里其实**有** BindingGYM 相关序列，包括 mutant —— BindingGYM 的 KRAS4B 1-169 WT 构建体是 UniParc `UPI00085E21D7`，KRAS4B G12D 是 `UPI00018CBEAD`，GB1 B1 是 `UPI0000111872`，trastuzumab VH 是 `UPI00001117B4`。但它们 **`ptm=None`、没有存储结构**；官方 API 的 `fold_on_miss=true` 可以**按需现折**，但只出单链。所以"零 mutant"要精确表述为：**没有任何 BindingGYM 序列在 Atlas 里有预计算结构**。

---

## 5. Q2 — 覆盖 BindingGYM 的 mutation 吗？

### 5.1 官方资产：**0 / 376,446，且是构造性的零**

- AFDB 单体层按 UniProt canonical 序列建索引，序列查询是 **精确 MD5 匹配** ⇒ ⚠️ 查 KRAS G12D 直接 HTTP 404（生物学上最著名的点突变都查不到）。
- AFDB complex 层按 **UniProt accession 对**建索引，metadata schema 的列是 `uniprot_ac_1` / `uniprot_ac_2`，**没有 mutation/variant 字段**。CIF 的 `_struct_ref` 把链映射到 full-span UNP accession。
- **isoform ≠ point mutant**：AFDB v5/v6 加了 40,054 条 splice isoform，这与突变体无关。
- ESM 两代 Atlas 同理（见 §4）。

> ⚠️ **最需要警惕的混淆**：AFDB complex 的 preprint 自己把 "**variant effects at interfaces**" 当卖点写在摘要里。那指的是把突变**映射到** WT 结构上，**不是**库里有 mutant 结构。

### 5.2 BindingGYM 的 WT 复合物：AFDB 命中 6/22（⚠️ agent 全表 grep + 我未复核）

| BindingGYM 结构 | UniProt 对 | AFDB modelEntityId | ipTM | 可直取？ |
|---|---|---|---|---|
| 8BE4 KRAS×SOS1 | P01116×Q07889 | `AF-0000000210539519` | 0.865 | ✅ 网站可见 |
| 1HE8 KRAS×PIK3CG | P01116×P48736 | `AF-0000000210539413` | 0.850 | ✅ 网站可见 |
| 1PQ1 Bcl-xL×Bim (mouse) | Q64373×O54918 | `AF-0000000210453793` | 0.674 | ✅ 网站可见 |
| 6VJJ KRAS×RAF1 (×2 assay) | P01116×P04049 | `AF-0000000210539270` | 0.824 | ❌ 只在 `shard_1710_batch_2.tar` |
| 1LFD KRAS×RALGDS | P01116×Q12967 | `AF-0000000210539530` | 0.709 | ❌ `shard_1828_batch_2.tar` |
| 5WER HLA-A×TAPBPR | P04439×Q9BX59 | `AF-0000000210561025` | 0.693 | ❌ `shard_1709_batch_0.tar` |

**但这 6 个也不是 drop-in**：AFDB 给的是 **full-length canonical** 链（KRAS 189 aa、SOS1 1333 aa），而 BindingGYM 是截断域（8BE4_hm 的 R=168 / S=475）。要用必须重新裁剪与重编号 —— 正好会撞上本 repo memory 里记的那几个坑（chain id、6VJJ 的 +1 偏移）。

⚠️ 另外 verify agent 做了个更狠的一致性检验：把 BindingGYM 全部 **47 条 WT 链序列**丢进 AFDB 的精确序列 API，**47/47 全部 404** —— 因为 BindingGYM 的链是晶体学构建体（截断、融合），不是 canonical 全长。所以"AFDB 覆盖 WT"只在 **accession 层面**成立，在**序列层面是 0/47**。

### 5.3 结构上永远进不来的部分

BindingGYM 有 **9 条工程化/设计链没有任何 UniProt 条目**：DARPin K27（5O2S）、Z-domain affibody ZpA963 / ZSPA-1（2M5A / 1LP1）、trastuzumab-4D5（1N8Z）、5A12 Fab（4ZFF/4ZFG）、FMC63 scFv（7URV）。**任何以 accession 为键的数据库在原理上都够不到它们**，涉及 7–9 个 assay。
（⚠️ 一个限定：ZpA963/ZSPA-1 的**母体骨架** protein A Z-domain 在 UniProt 里有（P38507 / P02976），有 AFDB 条目 —— 所以准确说法是"工程化变体没有条目，母体骨架有"。）

---

## 6. Q3 — 能不能方便定位？

| 资产 | 定位方式 | 好用吗 |
|---|---|---|
| AFDB 单体 | `GET /api/prediction/{acc}`；文件 `AF-{acc}-F1-model_v6.{cif,pdb,bcif}` | 好。⚠️ **但 `_v4` 已死**：per-accession 端点只发当前版本，26 个测试 accession 的 `_v4`/`_v5` 全部 404。硬编码 `_v4` 的老 pipeline 会静默取不到。（v4 仍以 FTP 批量归档形式存在） |
| AFDB complex（高置信 80 k） | ⚠️ `GET https://alphafold.ebi.ac.uk/api/complex/{UniProt}` **一次调用返回该蛋白参与的全部复合物**（含 modelEntityId、两侧 accession、ipTM/ipSAE/pDockQ）；文件 `/files/AF-{16位数字}-model_v1.{pdb,cif}` | 可用。但 **accession 是不可推导的 16 位数字**，无法从 UniProt 对拼出来，必须走 API。 |
| AFDB complex（低置信 7.5 M） | 只能先下 2.45 GB metadata CSV，读 `local_tar_name` 列，再下对应 shard tar（0.8–1.6 GB） | 难用。 |
| ESM Atlas（Biohub） | ⚠️ `GET /esm/protein/api/v1alpha1/uniprot/{acc}` 与 `/proteins/{md5(sequence)}`，无需鉴权；`fold_on_miss=true` 可现折 | 检索机制干净（key = 序列 MD5），但**只出单链**，且只对 Atlas 已收录序列生效。 |
| ESM Atlas（Meta 旧版） | 知道 MGYP 号能取；⚠️ **按序列/结构检索已下线** | 不可用于定位。 |
| **BindingGYM mutant 结构** | **无任何官方索引**——我实测出了映射键，见 §7 | 见下 |

---

## 7. ✅ 真正覆盖到 BindingGYM mutant 的唯一资产（我逐条验证）

**Zenodo record `12200340`** —— BindingGYM 同一个 concept DOI (`10.5281/zenodo.12200339`) 的 **v1**，被 v2 (`12514160`, `input.zip`) 顶掉。**论文、README、GitHub 仓库都不指向它**，concept DOI 直接跳到 v2，所以极易漏掉。

```
dataset_4D5_HER2_fitness_1N8Z.zip   538,237,327 bytes  (解压 2.17 GB, 4,179 条)
└── batch_0/  batch_1/  batch_2/
      ├── individual_list.txt                  FoldX 突变清单
      ├── 1N8Z_hm_renumbered_<i>.pdb           mutant complex   ← 共 2,080 个
      ├── WT_1N8Z_hm_renumbered_<i>.pdb        配套 WT          ← 共 2,080 个
      └── Average_/Dif_/Raw_/PdbList_*.fxout   FoldX 能量表
```

**✅ 我的验证（全部自己跑的）：**
1. Zenodo versions API：该 concept 确有 2 个版本，v1 = 12200340（2024-06-21，538 MB）。
2. HTTP range 读取 zip 的 EOCD + 完整中央目录（565,535 bytes）：**4,179 条，4,160 个 .pdb + 12 个 .fxout + 3 个 .txt**，解压后 2.17 GB。
3. 正则计数：**mutant PDB 恰好 2,080 个，配套 WT 恰好 2,080 个** = 该 assay 的 variant 数（2,080）。
4. Range-fetch + inflate 三个 batch 的 `individual_list.txt`，与本地 `4D5_HER2_fitness_1N8Z.csv` 的 `mutant_pdb` 列逐条比对：**每个 batch 取首、次、末三条，9/9 全部精确吻合**（含 batch 边界 row 999→1000、1999→2000）。

**✅ 定位键（本记录的原创产物，官方无任何文档）：**

```
BindingGYM 第 n 行（0-based，按 CSV 原始行序）
    →  batch_{n // 1000} / 1N8Z_hm_renumbered_{n % 1000 + 1}.pdb
    配套 WT： batch_{n // 1000} / WT_1N8Z_hm_renumbered_{n % 1000 + 1}.pdb
```

**覆盖率：2,080 / 376,446 = 0.55%**，25 个 assay 里的 **1 个**。其余 24 个 assay 的 FoldX 结构论文里说生成过（"we use FoldX to generate the complex structure for each mutant"）但**从未发布**。⚠️ 大概率的原因：**FoldX 的 EULA 禁止再分发** —— GearBind 的 README 写得很直白："we can not provide FoldX-generated HER2 and CR3022 mutant structures **due to license restrictions**."

---

## 8. ⚠️→❌ 一个被我亲手否掉的"好消息"：AbBiBench

workflow 的 verify agent 报了一个很亮眼的发现，说 **AbBiBench 的 Zenodo `16557372`（8.23 GB FoldX mutant 抗体-抗原复合物）1:1 覆盖 BindingGYM 的三个抗体 assay，合计 33,005 variants = 8.8% 的 benchmark**。它的依据是 CSV 行数完全吻合：

| AbBiBench CSV | 行数 | BindingGYM assay | 行数 |
|---|---|---|---|
| `4d5_her2_benchmarking_data_trimmed` | 2,080 | `4D5_HER2_fitness_1N8Z` | 2,080 |
| `5a12_ang2_benchmarking_data` | 944 | `5A12_Ang2_fitness_4ZFG` | 944 |
| `5a12_vegf_benchmarking_data` | 29,981 | `5A12_VEGF_fitness_4ZFF` | 29,981 |
| `3gbn_h1_benchmarking_data` | 1,887 | `CR6261_FluAH1_logKd_3GBN`（被砍） | 1,887 |
| `4fqi_h1_benchmarking_data` | 65,094 | `CR9114_FluAH1_logKd_4FQI`（被砍） | 65,094 |
| `4fqi_h3_benchmarking_data` | 65,535 | `CR9114_FluAH3_logKd_4FQY`（被砍） | 65,535 |

**✅ 行数我复核过，六个全部精确吻合**（确实是同一批 DMS library）。**但 agent 从行数吻合直接推了结构覆盖，没查压缩包内容。我查了，结论相反：**

我用 `curl | gzip -dc | tar -tv` 流式清点了整个 8.2 GB 包（清单存在 `refs/abbibench_mutant_structure_listing.txt.gz`，逐体系计数在 `refs/abbibench_mutant_structure_inventory.csv`）：

| 体系目录 | mutant PDB 数 | 与 BindingGYM 的关系 |
|---|---|---|
| `4fqi` | 65,536 | 对应**被砍掉**的 CR9114 流感 assay |
| `aayl49_ML` / `aayl51` / `aayl49` | 8,953 / 4,320 / 4,312 | 无关 |
| `2fjg` | 2,223 | 无关 |
| `3gbn` | 1,917 | 对应**被砍掉**的 CR6261 流感 assay |
| `1mlc` | 1,229 | 无关 |
| `1n8z` | **419** | AbBiBench 自己那个 419 行的数据集，**不是** BindingGYM 的 2,080 行 4D5_HER2 |
| `1mhp` | 37 | 无关 |
| **`4d5_her2` / `4zfg` / `4zff`** | **0 / 0 / 0** | **BindingGYM 的三个 benchmark 抗体 assay 一个结构都没有** |
| 合计 | **88,946** | |

**✅ 交叉验证**（把包内文件名的突变串、以及 AbBiBench CSV 的 `heavy_chain_seq` 与 BindingGYM 的 `mutant`/`mutated_sequence` 逐一比对）：

- 突变串交集：**全部为 0**。位点编号口径其实是**一致的**（BindingGYM 的位点集是 AbBiBench 的子集，例如 3gbn：BindingGYM `{28,30,58,59,62,74,75,76,77,79,104}` ⊂ AbBiBench `{6,28,30,45,58,59,62,74,75,76,77,79,93,104}`），零重叠是因为**两边的 WT 参考序列不同**。
- 完整序列交集：**六个 assay 全部为 0**。原因是**构建体边界不同** —— AbBiBench 用 **VH 域（121 aa）**，BindingGYM 用 **完整 Fab 重链（220 aa）**；4D5 那组我验证了 AbBiBench 的 121-aa VH **正是** BindingGYM 220-aa 链 B 的**子串**。

> **修正后的结论**：AbBiBench 是一个真实存在、命名规范（`mutant_structure/<pdbid>/<pdbid>_<chains>_<muts>.pdb`）的 FoldX mutant-complex 资产，但 **它不覆盖 BindingGYM 的任何一个 benchmark assay**。它覆盖的是 BindingGYM **砍掉的**那三个流感 assay 所对应的 library，而且要用还得跨构建体重新对齐。**"8.8% 覆盖"这个数字是错的，正确答案是 0%。**

---

## 9. 第三方 WT complex 资产（⚠️ 全部来自 agent，我未复核）

| 资产 | 规模 | 对 BindingGYM |
|---|---|---|
| **Predictomes** (predictomes.org) | 1,614,047 个人源蛋白对，AF-Multimer + SPOC；top-16k tar = 53 GB，全量 ~4.32 TB | **第三方里覆盖最好的**，含 10 个 BindingGYM 体系，且是**唯一有 CXCR4-CXCL12 的**。但 KRAS-PIK3CG（SPOC 排名 #72,134）与 PSD95-CRIPT（#767,162，SPOC 0.000）不在 top-16k 里，要拿得下 4.32 TB |
| **humanPPI** (RoseTTAFold2, conglab) | 29,246 个 PPI 的最佳模型，63 GB | 人源限定；按 segment 建模 |
| **huintaf2** (Burke 2023 NSMB) | 65,484 对，3,137 高置信 | 只命中 MCL1-BCL2L11、BCL2L1-BAD 两对，pDockQ ~0.39 |
| **ModelArchive** `ma-bak-cepc` (Humphreys 2021) | 1,106 个酵母复合物 | 与 BindingGYM 零重叠 |
| **GWYRE 2.0** (gwyre.org) | 46,969 个人源二元复合物（9,048 实验 + 37,921 AF2-Multimer） | 典型的"把 variant **映射到** WT 结构"资源，**不含 mutant 坐标** |
| **Complex Portal / hu.MAP 3.0** | — | **完全没有 3D 坐标**，只有组分表和分数。Complex Portal 把 AF-Multimer 整合列为**未来计划** |
| **SKEMPI 2.0** | 7,085 个突变条目、**345 个 PDB** | mutant-indexed 但**只发 WT 结构**。⚠️ 修正："连实验界也没有 mutant complex 结构"是过头的——RCSB 里有 15,038 个"多蛋白实体 + 至少一条链带注释突变"的条目；真正没有的是**WT 配对、ΔΔG 标注、覆盖 DMS 的索引** |
| **ATLAS / ThermoMPNN-D / MuToN / MdrDB / AbDesign** | TCR-pMHC / Tsuboyama 单体 / SKEMPI 单链 / 蛋白-配体 / 抗体 CDR-H3 | 都是真的预计算 mutant 结构资产，但**与 BindingGYM 零重叠** |

---

## 10. 如果要自己生成

### 10.1 成本（⚠️ agent 估算，verify 已上修下界）

- **22 个 WT 复合物：0.15–0.38 A100-GPU-h**，一个下午的事。
- **376,446 个 mutant：可辩护区间 ~3,000–6,500 A100-GPU-h**（agent 原报 2,000–6,500，verify 指出下界靠 t^2.149 外推到 256-token bucket 得到 ≈3 s/次，物理不可达；加 20–30 s/次的地板后下界移到 3,046–3,817）。约 $13k 的 2026 serverless H200 价，或本地 A4500 上 0.6–2.2 GPU-年。
- **MSA 可跨 mutant 复用**（22 个复合物 / ~47 个 (structure, chain) 槽位），这是最大的一笔节省。⚠️ 但有个 gate：AF3 的 `input.md` 明写 "The first sequence is exactly equal to the query sequence" —— **WT 的 a3m 不能原样喂 mutant，row 0 必须改写**，写错会静默出错。
- ⚠️ AFDB complex release **公开了它用的全部 MSA**（`collaborations/nvda/msas/`，三个子发布），对命中的那 6 个 pair 可以直接复用。

### 10.2 但真正的瓶颈是**精度不是算力**（verify 的最重要一条）

⚠️ Hitawala & Gray, *mAbs* 2025;17(1):2545601：**AF3 在抗体上的 high-accuracy docking 成功率只有 10.2%**（纳米抗体 13.3%），Boltz-1 4.08%、Chai-1 0%；"AF3's **65% failure rate** for antibody and nanobody docking"。而 BindingGYM 至少 9 个 assay 的 partner 是抗体/scFv/DARPin/affibody（~112k variants）—— **恰好是 MSA 最浅、模型公认最弱的体系**。

**更直接的证据在 AbBiBench 自己的 leaderboard 上（我读到的 ✅）**：它把 AF3 和 Boltz-2 当 "Structure Prediction" baseline 逐 mutant 折叠过，结果是——

| 排名 | 类型 | 模型 | 平均 Spearman |
|---|---|---|---|
| 🥇 1 | Inverse Folding | **ProteinMPNN**（单一 WT backbone） | **0.30** |
| 4 | Structure Prediction | Boltz-2（逐 mutant 折叠） | 0.13 |
| 5 | Biophysics | FoldX | 0.12 |
| **17** | Structure Prediction | **AF3**（逐 mutant 折叠） | **−0.02** |

**在抗体体系上，逐 mutant 重折叠（AF3）比在单一 WT backbone 上做 inverse folding（ProteinMPNN）差得多。** 这对本 repo 的 complexTTT 路线是很硬的一条参考。

### 10.3 许可闸门（对 complexTTT 是决定性的）

- **AF3 权重条款**：仅非商业组织；不得再分发；**输出不得用于训练"与 AF3 类似的生物分子结构预测模型"（含蒸馏）**。README 至今仍写 "You may only use AlphaFold 3 model parameters if received directly from Google."
- ⇒ **如果 complexTTT 会拿生成的结构去训练/做 test-time training，AF3 在法律上出局。**
- **干净的替代**：**Boltz-2（MIT，代码+权重）**、**Chai-1 / OpenFold3 / Protenix（Apache 2.0）**、**AF2-Multimer 参数（CC BY 4.0）** —— 全都没有这条 derived-models 条款。
- ⚠️ 本地 A4500（20 GB）：AF3 官方只支持 A100/H100 80GB；最大体系 1HE8 = 1,107 token、1N8Z = 1,041，正压在 AF3 文档"V100 + unified memory 可到 1,280 token"那条线上。**Boltz-2（~11 GB）是本地更稳的选择。**

### 10.4 不要走的路

- ❌ **ESMFold + 甘氨酸 linker**：linker 是官方默认没错，但 Meta 从未验证过它做 hetero-complex，托管 API 直接拒绝，且没有任何 binding-DMS 工作用它。
- ❌ **Rosetta Flex ddG**：15 CPU-h/突变 ⇒ BindingGYM 全量 = **644 CPU-年**。
- ❌ **FoldX 全量**：210 s/突变 ⇒ 2.5 CPU-年，且只重排侧链、不给新骨架。

---

## 11. 陷阱清单

1. **`AF-<acc>-F1-model_v4` 已死** —— AFDB 现在是 v6，per-accession 端点只发当前版本。硬编码 `_v4` 会静默取不到。（v4 仍有 FTP 批量归档）
2. **AFDB 搜索 API 零命中返回 HTTP 404 而不是 `numFound: 0`** —— 自动扫覆盖率的脚本若把 404 当"端点坏了"，会系统性低估。这正是 workflow 里 survey agent 报 3/25 而 verify 报 8-9/25 的机制。
3. **AFDB complex 的质量阈值是 `ipSAE ≥ 0.6 AND pDockQ2 ≥ 0.23`，不是 ipTM** —— KRAS/RAF1 的 ipTM 0.824、ipSAE 0.631 都不错，但 pDockQ2 只有 0.0246，所以被挡在可检索层外。**"搜不到"不等于"不存在"。**
4. **AFDB complex 的 `.cif` 文件带 `.cif` 后缀但实际是 gzip 压缩的** —— 直接 grep 会静默找不到东西。
5. **P01892（HLA-A\*02:01）在 UniProt 已废弃**，并入 P04439（那是 A\*03:01 参考等位）—— 等位基因层面的结构在任何 accession 体系里都不可寻址。
6. **P0DTC2（SARS-CoV-2 spike）在 AFDB 核心库没有条目**，只有第三方 Viro3D 的 ColabFold 模型（`AF-0000000365840314`），且在 7.59 M 行 heterodimer 表里出现 **0 次**。
7. **行数吻合 ≠ 结构覆盖** —— §8 那个教训。任何"某数据集覆盖 BindingGYM N%"的说法，必须去清点结构文件本身。
8. **跨数据集比对抗体 DMS 时先看构建体边界** —— VH 域 vs 完整 Fab 重链会让序列/突变串交集直接变成 0，而它们其实是同一批分子。
9. **BindingGYM 悄悄做过的物种/旁系替换**（⚠️ agent，headers 已被剥所以从文件里读不出来）：1HE8/1LFD 把沉积的 **HRAS 换成 KRAS**；1LFD 把**大鼠** RALGDS 换成人；1PQ1 把**小鼠** Bcl-xL/Bim 换成人；5WER 把**小鼠 H-2Dd** 换成人 HLA-A\*02；3KZ0 把沉积的设计肽 "MB7" 换成 BIM BH3。
10. **paper 的 508,962 与 shipped 的 376,446 差在哪** —— ⚠️ `376,446 + 132,516（三个被砍的流感 assay）= 508,962`，精确闭合。三个流感 CSV 随包发布但不在 25 行索引里，且它们引用的 `3GBN_hm.pdb`/`4FQI_hm.pdb`/`4FQY_hm.pdb` **在 `structures/` 里不存在**（悬空引用）。GitHub issue #4 问的就是这个，2025-11-10 至今零回复。

---

## 12. 建议的下一步

1. **不要为了 BindingGYM 去下 AFDB 的 complex 资产。** 它在 WT 侧最多给你 6/22 个、构建体还对不上；BindingGYM 自带的 22 个实验骨架已经 100% 覆盖所有 DMS 位点，严格更好。
2. **如果需要 mutant complex 坐标**：先把 Zenodo v1 那 2,080 个 4D5_HER2 的 FoldX 结构拿下来（538 MB，映射键在 §7）。这是唯一免费的、且能直接对上 BindingGYM 变体的样本 —— 正好可以用来**校准"mutant backbone 到底带来多少额外信号"**，代价接近零。
3. **在扩大规模前，先用这 2,080 个做一次 gate 实验**：同一批 variant，(a) WT backbone + ProteinMPNN vs (b) FoldX mutant backbone + ProteinMPNN，比 per-assay Spearman。若 (b) 不显著优于 (a)，就不必再考虑给其余 24 个 assay 生成结构。这与 §10.2 里 AbBiBench leaderboard 的证据方向一致（逐 mutant 折叠反而更差）。
4. **要问 BindingGYM 作者要剩下 24 个 assay 的 FoldX 结构**（`luwei@aurekabio.com`；⚠️ 注意：论文四位作者的单位是 **Aureka Biotechnologies**，不是 Rice）。但 FoldX EULA 很可能挡住，见 §7。
5. **若最终决定自己生成**：用 **Boltz-2（MIT）或 Protenix（Apache 2.0）**，不要用 AF3 —— 许可条款直接卡 complexTTT 这类用法（§10.3），且本地 A4500 显存也更适配。
6. **AFDB 的 `nvda/msas/` 值得单独看一眼** —— 对 ProteinTTT/complexTTT 这条线，公开的官方 MSA 可能比结构本身更有用。

---

## 附录 A — 逐 assay 覆盖表

数据：`refs/bindinggym_structure_coverage.csv`（53 行 = 每个 (assay, chain) 一行）

| 覆盖维度 | 结果 |
|---|---|
| BindingGYM 自带 WT complex 覆盖 DMS 位点 | ✅ **2,220 / 2,220 = 100%**（25/25 assay） |
| AFDB complex 命中 WT UniProt 对 | ⚠️ 6 / 22 结构（7 / 25 assay），3 个网站可见 + 3 个仅 FTP |
| AFDB 命中 BindingGYM 的**精确 WT 链序列** | ⚠️ **0 / 47** |
| AFDB / ESM Atlas 覆盖 mutant | **0 / 376,446** |
| 任何公开资产覆盖 BindingGYM mutant complex | ✅ **2,080 / 376,446 = 0.55%**（仅 `4D5_HER2_fitness_1N8Z`，来自 Zenodo v1） |
| 含工程化链、accession 体系原理上够不到的 assay | ⚠️ 7–9 / 25 |

## 附录 B — 证据与工件

**本目录**
- `sh/audit_bindinggym_structure_coverage.py` — BindingGYM 结构覆盖审计（可复跑）
- `refs/bindinggym_structure_coverage.csv` — 逐 (assay, chain) 覆盖表
- `refs/abbibench_mutant_structure_inventory.csv` — AbBiBench 8.2 GB 包的逐体系结构计数
- `refs/abbibench_mutant_structure_listing.txt.gz` — 该包的完整 tar 清单（88,956 条）
- `refs/PROVENANCE.txt`

**外部关键 URL**
- BindingGYM Zenodo v1（**唯一的 mutant complex 资产**）：`https://zenodo.org/records/12200340`
- BindingGYM Zenodo v2（input.zip）：`https://zenodo.org/records/12514160`
- AFDB complex FTP：`https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvda/`
- AFDB complex API：`https://alphafold.ebi.ac.uk/api/complex/{UniProt}`
- AbBiBench 结构：`https://zenodo.org/records/16557372`；代码 `https://github.com/MSBMI-SAFE/AbBiBench`
- ESM Atlas（Biohub）：`https://registry.opendata.aws/biohub-esm-atlas/`
- AF3 权重：`https://storage.googleapis.com/alphafold3/af3.bin.zst`（代码 Apache 2.0，权重仍非商业 + 禁 derived models）

## 附录 C — 本次调研的元信息

- Workflow `predicted-complex-structure-asset-survey`，run `wf_fd88a803-b93`，**16 agent（8 维度 × survey→adversarial verify），16/16 完成，0 error，2,252,928 token，1,107 次工具调用，约 65 分钟**。
- Verify 阶段的 verdict 汇总：多数 CONFIRMED，但产生了 **6 条 REFUTED、11 条 OVERSTATED** 与大量 corrections —— 其中三条改变了结论：AFDB 覆盖从 3/25 上修到 8–9/25（survey 只查了搜索索引没查 FTP）、"BindingGYM 从未发布 mutant 结构"被推翻（Zenodo v1）、"SKEMPI 证明实验界没有 mutant complex"被推翻。
- **我自己又推翻了 verify 的一条**：AbBiBench 覆盖 BindingGYM 8.8% → 实际 **0%**（§8）。**这说明对抗验证也会漏，凡是"覆盖率"结论都要清点到文件级。**
- 完整 agent 输出：`~/.claude/projects/-home-guoj0f-repos-ProteinTTT--claude-worktrees-mutation-backbone-structure-analysis/88a5c184-.../subagents/workflows/wf_fd88a803-b93/journal.jsonl`

## 关联

- [`structure_pseudolabel_methods_survey_20260829-231404.md`](structure_pseudolabel_methods_survey_20260829-231404.md) —— 同 project 的下游方法调研：**要自己生成的话有哪些方法、代价多少、文献里实际怎么做**。它承接本篇 §10 与 §12，并**修正了 §12 的 gate 实验设计**：FoldX 不改 backbone，而 ProteinMPNN / ESM-IF1 / SaProt-3Di 全是 backbone-only consumer，所以「WT backbone vs FoldX mutant backbone」在构造上可能是同一个实验。
