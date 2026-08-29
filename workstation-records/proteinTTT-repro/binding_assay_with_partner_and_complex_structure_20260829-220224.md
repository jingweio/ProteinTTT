# ProteinGym binding assay × BindingGYM：补齐 partner sequence 与 complex structure

created 2026-08-29 22:02 · status: DONE（数据构建已完成并通过校验）

**结论：ProteinGym 的 13 个 binding assay 中，9 个可以从 BindingGYM 定位到 partner protein sequence
和 WT complex structure；4 个不能。** 构建脚本已跑通，9/9 全部通过三重坐标校验。

---

## 1. 哪些 binding assay 可以补齐？

配对不是按 assay 名字猜的，是**变体级实测**：用单点突变的 `(wt, mut)` 投票求坐标 offset，对齐后
要求交集上 Spearman ρ ≥ 0.90（低于此即判为不同 assay）。全量扫了 217 × 25 共 1036 个候选，只有
下面 9 对通过。

### ✅ 可用的 9 个

| ProteinGym binding assay | complex PDB | target 链 | **partner 链** | offset | 变体数 | 结构内可打分 | 触及界面 |
|---|---|---|---|---|---|---|---|
| `SPG1_STRSG_Olson_2014` | `1FCC_hm` | C (56) | **A: IgG-Fc, 206aa** | +226 | 536,962 | 100% | **87.2%** |
| `SPG1_STRSG_Wu_2016` | `1FCC2016_hm` | C (56) | **A: IgG-Fc, 206aa** | +226 | 149,360 | 100% | **100%** |
| `SPIKE_SARS2_Starr_2020_binding` | `6M0J` | E (194) | **A: ACE2, 597aa** | +332 | 3,802 | 96.5% | 23.8% |
| `ACE2_HUMAN_Chan_2020` | `6M17_BE` | B (748) | **E: SARS2-RBD, 183aa** | +20 | 2,223 | 98.3% | 36.5% |
| `Q53Z42_HUMAN_McShan_2019_binding-TAPBPR` | `5WER_hm` | A (274) | **C: TAPBPR, 370aa** | +25 | 3,344 | 100% | 24.0% |
| `CD19_HUMAN_Klesmith_2019_FMC_singles` | `7URV_hm` | C (255) | **D: FMC63 scFv, 242aa** | +22 | 3,761 | **79.6%** | 15.3% |
| `DLG4_RAT_McLaughlin_2012` | `1BE9_hm` | A (115) | B: CRIPT 肽, **5aa** | +300 | 1,576 | 100% | 39.8% |
| `YAP1_HUMAN_Araya_2012` | `1JMQ_hm` | A (46) | P: 肽, **10aa** | +164 | 10,075 | 100% | 84.2% |
| `B2L11_HUMAN_Dutta_2010_binding-Mcl-1` | `3KZ0_hm` | C (**23**) | A: Mcl-1, 150aa | +142 | 170 | 100% | 100% |

- **offset**：`ProteinGym 位点 = target 链位点 + offset`。
- **结构内可打分**：该变体的所有突变位点在 complex 里都有坐标且残基身份已知，占全部变体的比例。
- **触及界面**：在可打分变体中，至少有一个突变位点距 partner 重原子 ≤ 8 Å 的比例。

**按可用性分三档：**
- **主力（6 个）** —— 前 6 行。partner 是真实蛋白，加进去才对 structure-conditioned model 构成有效信息。
- **弱 partner（2 个）** —— `DLG4` / `YAP1` 的 partner 只有 5–10aa 的肽，"加 partner"几乎不改变输入，
  不适合作为主检验。
- **规模过小（1 个）** —— `B2L11` 只有 170 个变体、target 本身只是 23aa 的 BH3 肽。

**顺带可复用的对照：** `SPIKE_SARS2_Starr_2020_expression` 和 `Q53Z42_HUMAN_McShan_2019_expression`
与表中第 3、5 行是**同一变体集、同一结构**，只是 label 换成 expression。同一套 asset 直接复用，
天然构成「partner 提升的是 binding 特异性还是 folding」的判别对照。

### ❌ 不可用的 4 个

| ProteinGym binding assay | 原因 |
|---|---|
| `CCR5_HUMAN_Gill_2023` | BindingGYM 无此 target（只有 CXCR4） |
| `CP2C9_HUMAN_Amorosi_2021_activity` | BindingGYM 无此 target |
| `GCN4_YEAST_Staller_2018` | BindingGYM 无此 target |
| `RASK_HUMAN_Weng_2022_binding-DARPin_K55` | target 有（6 个 KRAS complex），但**没有 K55 这个 partner**，只有 K27 |

最后一个尤其要注意：KRAS 的 target 链和结构都是现成的，但 partner 不对。**给错 partner 比不给更糟**，
所以只能放弃，除非另找 DARPin K55 的复合物结构。

---

## 2. 构建 SOP

### 输入

| 资产 | 路径 |
|---|---|
| ProteinGym reference | `/home/guoj0f/repos/ProteinGym/reference_files/DMS_substitutions.csv` |
| ProteinGym DMS | `/home/guoj0f/share/ProteinGym/DMS_ProteinGym_substitutions/` |
| BindingGYM 索引 | `/home/guoj0f/share/BindingGYM/input/BindingGYM.csv` |
| BindingGYM DMS | `/home/guoj0f/share/BindingGYM/input/Binding_substitutions_DMS/` |
| BindingGYM 结构 | `/home/guoj0f/share/BindingGYM/input/structures/` |

### 运行

```bash
python workstation-records/proteinTTT-repro/sh/build_pg_binding_complex.py \
    --out <output_dir> [--min-rho 0.90] [--iface-cutoff 8.0]
```

单机 CPU，约 3 分钟，输出 42 MB。**不需要 GPU。**

### 四个步骤

**Step 1 — 定位配对（全量扫描，不预设名单）**
对全部 217 个 ProteinGym assay × 25 个 BindingGYM assay × 每条被突变链：
1. 只保留 BindingGYM 中「该链是唯一被突变链」的记录（跨链双突变会污染 key）；
2. 用单点突变的 `(wt, mut)` 投票求 offset —— 取得票最高的 `PG_pos − BG_pos`；
3. 按该 offset 映射后求变体交集，计算交集上的 Spearman ρ；
4. **ρ ≥ 0.90 才接受**。

> 第 4 步是关键。只看变体重合会误判：`PSD95_Tm2F_1BE9` 与 `DLG4_RAT_McLaughlin_2012` 的变体集
> **1576/1576 完全一致**，但 ρ 只有 0.435 —— 同一个 library 换了 peptide ligand，是**不同的 assay**。
> 同理 `BH3_Bcl-xL` (ρ=0.53)、6 个 KRAS assay (ρ=0.73–0.85) 也全部被这一步正确排除。

**Step 2 — 对齐 PDB 编号**
结构文件只有 ATOM 记录、**没有 SEQRES**。用残基身份一致性投票求 `shift`，使
`target 链位点 = pdb_resid − shift`。**9 个里有 5 个的 shift ≠ offset**（`1FCC` / `1FCC2016` shift=0
而 offset=226，`3KZ0_hm` 0 vs 142，`5WER_hm` 1 vs 25，`1JMQ_hm` 4 vs 164）—— 说明**不能假设 PDB 编号
等于任何一侧的序列编号**，两个映射必须独立求解。

**Step 3 — 三重校验（不通过就中止）**
1. **硬性**：ProteinGym 实际突变的**每一个位点**，其 WT 残基在 ProteinGym `target_seq`、
   BindingGYM 链序列、PDB ATOM 三处必须一致。
2. **记录**：整段映射窗口的一致率。不一致但落在非突变区的，记为 construct 差异（见 caveat）。
3. **标记**：BindingGYM 序列中标为 `X` 的位点身份未知，从可打分集合中剔除。

**Step 4 — 产出**
每个 assay 一个目录：

```
<PG_assay>/
├── complex.pdb        # WT complex，chain id 保持 BindingGYM 原样
├── sequences.fasta    # >target|chain_X|Naa  +  >partner|chain_Y|Maa
├── variants.csv       # mutant_PG, mutant_target_chain, DMS_score,
│                      # n_subs, in_structure, touches_interface
└── meta.json          # 两个 offset、ρ、各项计数、caveat 字段
manifest.csv           # 9 行汇总
_all_candidate_pairings.csv   # 1036 个候选（含被拒的），供复核
```

`variants.csv` 里 `mutant_PG` 是 ProteinGym 原始坐标（保证与官方 benchmark 可比），
`mutant_target_chain` 是结构/链坐标（喂给 structure-conditioned model 用）。

### 已知 caveat（都已写进 `meta.json`）

1. **BindingGYM 给的是 crystallisation construct 序列，不总是天然序列。**
   `1BE9` 的 target 链 idx 103–115（PG 位点 403–415）是构建体的 C 端延伸，与 PSD-95 天然序列
   完全不同；`1FCC` / `1FCC2016` 的 N 端各有 1–2 个残基不符。这些位点 ProteinGym 从不突变，
   所以不影响打分，但**若要按天然序列做任何比较，必须先排除这些区段**。
2. **未知残基 `X`。** CD19 target 有 37 个、TAPBPR partner 有 53 个、Mcl-1 partner 有 7 个。
   ProteinMPNN 的 alphabet 里 `X` 合法但不携带信息，解读结果时要单独处理。
3. **CD19 只有 79.6% 可打分。** PDB chain C 仅解出 218/255 残基，ProteinGym 的 272 个突变位点里
   有 54 个落在窗口外或未解析。跨 assay 汇总时这个 assay 的样本量口径与其他不同。
4. **配对不等于数值相同。** 9 对里只有 `DLG4`(ρ=1.000000) 数值完全 bit-identical；其余是同源
   study 的不同处理口径（如 `ACE2` ρ=0.944）。**label 一律以 ProteinGym 为准**，BindingGYM 只用来
   取 partner 序列和结构。

---

## 3. 复现

```bash
# 脚本
workstation-records/proteinTTT-repro/sh/build_pg_binding_complex.py
# 环境
/home/guoj0f/anaconda3/envs/proteingym-ttt   (pandas 2.x, numpy, scipy)
# 本次运行输出
9 assays, 42 MB, 全部通过 Step 3 校验
```
