# esmfold2_monomer_vs_complex_mpnn — experiment record

created 2026-08-30 22:18 · **status: RUNNING**

## 1. Goal

上一轮用 BindingGYM 晶体做的消融有两个硬伤：序列是**子窗口**（9%–93% 不等）、未解析处标 `X`，
导致 3 个 assay 只有 79.6%/96.5%/98.3% 的变体可打分，**消融不干净**。

本轮改用 **ESMFold2 预测结构**，从 ProteinGym 自己的 `target_seq` 折起 ——
坐标与突变位点天然 1:1 对齐，**覆盖率 100%，无 `X`，无 offset**。

测四个点：
- **(a)** ProteinGym leaderboard 上的 top-rank 模型
- **(b)** ProteinGym 官方的 ProteinMPNN（AF2 单体）
- **(c)** 我们：ProteinMPNN on **ESMFold2 预测的单体**
- **(d)** 我们：ProteinMPNN on **ESMFold2 预测的复合物**（target + **完整** partner）

---

## 2. Step (1) 完整 partner 序列的定位方式 ← 本实验的前置产物

**判定规则**（先分类再取序列）：

| 情形 | 处理 | 适用 |
|---|---|---|
| partner 是**被结晶截断的天然蛋白** | 取 UniProt canonical **全长** | 7 条 |
| partner 是**工程构建体**（scFv） | 构建体本身即完整分子，补回缺失 linker | CD19 |
| partner 是**短肽配体**（≤20 aa） | 肽即实验中的完整结合实体，原样保留 | YAP1 |

**核验方式（硬性）**：把 BindingGYM 晶体链按 `X` 切成片段，要求每一段都是候选 UniProt 全长的
**精确子串**；不精确匹配的再逐位算同一性并记录原因。

| PG assay | partner | 来源 | 全长 | 核验 |
|---|---|---|---|---|
| `ACE2_HUMAN_Chan_2020` | SARS-CoV-2 Spike | `P0DTC2` | 1273 | 1/1 段，**100%** |
| `SPIKE_SARS2_Starr_2020_binding` | Human ACE2 | `Q9BYF1` | 805 | 1/1 段，**100%** |
| `B2L11_..._Mcl-1` | Mcl-1 | `Q07820` | 350 | 2/2 段，**100%** |
| `DLG4_RAT_McLaughlin_2012` | Rat CRIPT | `Q792Q4` | 101 | 1/1 段，**100%**（BG 的 `KQTSV` 正是 CRIPT C 端 97–101） |
| `Q53Z42_..._TAPBPR` | TAPBPR | `Q9BX59` | 468 | **7/9 段**，181/317 残基 ⚠ |
| `SPG1_STRSG_Olson_2014` | IgG1 重链恒定区 | `P01857` | 399 | 0/1 精确段，但**同一性 96.6%**（7 处 E↔Q / D↔N，allotype 差异）⚠ |
| `SPG1_STRSG_Wu_2016` | 同上 | `P01857` | 399 | 同上 |
| `CD19_..._FMC_singles` | FMC63 scFv | 构建体 | 242 | 2/2 段 100%；**15 个连续 `X` 位于 VL/VH 之间，按 (G4S)₃ 补全** |
| `YAP1_HUMAN_Araya_2012` | PPxY 肽 | 肽 | 10 | 原样，无 `X` |

**两处需要注意的判断：**
- **`TAPBPR`**：9 段里 2 段（16 aa 与 120 aa）匹配不上人 TAPBPR，也匹配不上鼠 TAPBPR(0/9)、
  β2m、HLA-A。鉴于 7/9 段命中，认定 partner 身份是人 TAPBPR，**用 `Q9BX59` 全长**；
  不符段疑为构建体工程改造，已记录。
- **`IgG1`**：BG 片段对齐到 `P01857` 第 121 位（= Fc 的 CH2-CH3），121 个位点里 7 处差异全是
  E↔Q / D↔N —— 典型 allotype/脱酰胺模式，**身份判定成立**，取 canonical 全长。

**产物**（已 commit，可审计）：`refs/partner_sequences.fasta`、`refs/partner_resolution.csv`。

---

## 3. Design

| 条件 | 结构 | 链 | 覆盖 |
|---|---|---|---|
| **(b) official** | ProteinGym 自带 AF2 单体 | 1 | 100%（读官方表，不重跑） |
| **(c) esmfold_monomer** | ESMFold2-Fast 折 `target_seq` | A | **100%** |
| **(d) esmfold_complex** | ESMFold2-Fast 折 `target_seq` + 完整 partner | A+B | **100%** |

`(d) − (c)` = partner 的净效应，且**这次是在完全相同的 100% 变体集上做的配对比较**。
上一轮的 BindingGYM 晶体结果也会并入表中作横向参照。

**ESMFold2 配置**（遵循 SOP §9.4，复合物口径）：`ESMFold2-Fast`、`10 loops / 68 steps /
1 diffusion sample`、`seed=1`、`kernel=fused`、trunk fp32 + ESMC bf16。
单体条件用**同一套参数**，保证两侧唯一差异是有没有 partner。

**ProteinMPNN 配置**：与上一轮完全一致 —— `v_48_020`（md5 `698982b1…`）、`backbone_noise=0`、
每变体一个随机 decoding order、`seed=1`、主指标为 target 链限定的 `-mean NLL`。

### 规模与风险

| 条件 | L 范围 | 备注 |
|---|---|---|
| monomer | 198 – 1273 | 按 SOP §9.6 拟合，L=1273 约 29 GB |
| complex | 514 – **2078** | ⚠ `ACE2`/`SPIKE` 两个都是 L=2078，外推约 **56 GB** |

⚠ **L=2078 远超 SOP 已验证范围（最长 791）**，且开跑时 GPU 上另有他人约 29.6 GB。
脚本对单条失败只记录不中断（`*_failures.tsv`），必要时用 `--chunk-size 32` 重跑这两条。

### 已知会影响解读的点

1. **完整 partner ≠ 生物学正确的 partner。** `ACE2`×Spike 用的是全长 1273 aa spike，
   而 spike 的生物学单位是三聚体、真正结合 ACE2 的是 RBD。按"完整序列"的要求这么做，
   但这一条的结果要单独看。
2. **ESMFold 预测质量本身是变量。** `pLDDT`/`ipTM` 会一并记录；ipTM 低的复合物其界面近乎随机
   （SOP §9.4 实测：ipTM≈0.70 的体系换 seed 全复合物 CA-RMSD 就有 11.6–13.6 Å）。
3. **两个弱 partner**（`DLG4` 5aa 肽、`YAP1` 10aa 肽）仍不适合进主检验。

---

## 4. Run config

- GPU: A100 80GB @ 10.67.24.41 ｜ 启动前 free **51,420 MiB** / util 19%（他人 4 进程占 ~29.6 GB）
- env: `esmfold2`（torch 2.11.0+cu128 / transformers 4.57.6 / esm 3.3.0）+ `pgym-binding-partner-mpnn`
- 结构产出: `/data/guoj0f/share/our-predicted-structure/ProteinTTT-proteinTTT-proteinGYM-reproduce/ProteinGYM-bindingAssay-with-partener-and-complexStructure/`（SOP §2.4 约定）
- 启动脚本: `sh/esmfold2_monomer_vs_complex_mpnn_20260830-221855.sh`

## 5. Change log

- 22:18 完成 step (1) 的 partner 定位与核验，产出 `refs/partner_*`；写计划。

## 6. Results

（待填）
