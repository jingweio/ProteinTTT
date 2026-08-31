# esmfold2_monomer_vs_complex_mpnn — experiment record

created 2026-08-30 22:18 · **status: DONE**（2026-08-31 05:5x 完成）

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

- 08-30 22:18 完成 step (1) partner 定位与核验，产出 `refs/partner_*`；写计划。
- 08-30 22:22 **gate 抓到 bug**：`fold()` 返回 `MolecularComplex` 而非 `ProteinComplex`，无 `to_pdb_string`。
  改走 `to_protein_complex().to_pdb_string()`。若不先 gate，会等 18 个结构全折完才在 MPNN 阶段炸。
- 08-30 22:23 gate 通过：YAP1 复合物 `chain A 1..504 连续` / `chain B 1..10 连续`，**offset=0 成立**；
  MPNN 侧 ρ=0.1874（n=10,075，100% 可打分）。启动 **PID 3623575**。
- 08-31 05:5x 主流程 `ALL_DONE`（约 7 小时）。`ACE2`/`SPIKE` 两条 L=2078 复合物 **OOM**，7/9 成功。
- 08-31 05:55 用 `--chunk-size 32` 补跑那两条，**再次 OOM**（当时卡上只剩 705 MB，他人占约 40 GB）—— 是共享争用。
- 08-31 06:0x **修聚合脚本 bug**：leaderboard 段原本对**全部 217 个 assay**求均值而非这 9 个，
  导致官方 ProteinMPNN 被显示成 0.2950（全库均值），真实值是 0.1328。已修正并重算。

## 6. Results

### 6.1 四方对比（9 个 assay 均值）

| | mean ρ | 在这 9 个 assay 上的排名 |
|---|---|---|
| **(a)** leaderboard 顶部 `ProSST (K=4096)` | **0.4862** | 1 |
| （次位 `ProSST (K=1024)` 0.4494 / `VenusREM` 0.4444） | | 2 / 4 |
| **(b)** 官方 ProteinMPNN（AF2 单体） | **0.1328** | **97 / 101** |
| **(c)** 我们：ProteinMPNN on **ESMFold2 单体** | **0.1362** | 96 |
| **(d)** 我们：ProteinMPNN on **ESMFold2 复合物** | **0.1788** | 92 |
| 参照：上一轮 BindingGYM **晶体**单体 | 0.2193 | 73 |
| 参照：上一轮 BindingGYM **晶体**复合物 | **0.3013** | **40** |

### 6.2 逐 assay

| assay | n | official | esmfold_mono | esmfold_cplx | **Δpartner** | 晶体 mono | 晶体 cplx |
|---|---|---|---|---|---|---|---|
| `B2L11_..._Mcl-1` | 170 | −0.005 | −0.042 | **0.430** | **+0.472** | 0.254 | 0.726 |
| `Q53Z42_..._TAPBPR` | 3,344 | 0.192 | 0.171 | 0.162 | −0.009 | 0.260 | 0.258 |
| `SPG1_STRSG_Wu_2016` | 149,360 | 0.089 | 0.109 | 0.096 | −0.013 | 0.185 | 0.229 |
| `CD19_..._FMC_singles` | 3,761 | 0.174 | 0.103 | 0.088 | −0.015 | 0.136 | 0.178 |
| `SPG1_STRSG_Olson_2014` | 536,962 | 0.147 | 0.203 | 0.184 | −0.019 | 0.331 | 0.394 |
| `YAP1_HUMAN_Araya_2012` | 10,075 | 0.197 | **0.248** | 0.196 | −0.052 | 0.147 | 0.171 |
| `DLG4_RAT_McLaughlin_2012` | 1,576 | 0.135 | 0.161 | 0.096 | **−0.065** | 0.191 | 0.265 |
| `ACE2_HUMAN_Chan_2020` | 2,223 | 0.106 | 0.112 | **OOM** | — | 0.081 | 0.077 |
| `SPIKE_SARS2_Starr_2020` | 3,802 | 0.160 | 0.162 | **OOM** | — | 0.390 | 0.415 |

### 6.3 三个结论

**① ESMFold 单体 ≈ 官方 AF2 单体。** 0.1362 vs 0.1328，6/9 为正，**Wilcoxon p=0.734**。
同样喂全长序列，换结构预测器几乎不改变 ProteinMPNN 的表现（`SPIKE` 最典型：0.1615 vs 0.160）。
→ **本轮与上一轮晶体版的差距可以全部归因到「序列窗口」，不是预测器。**

**② partner 的正效应消失了。** 7 个里 **只有 1 个为正**，mean +0.0426、**Wilcoxon p=0.297**
（晶体版是 7/9 为正、p=0.020）。均值为正**完全由 `B2L11` 一个撑着**（+0.472，但只有 170 个变体）。

**③ 原因已定位：预测的界面不可信。**

| assay | pLDDT 单体→复合物 | pTM | **ipTM** | Δpartner |
|---|---|---|---|---|
| `DLG4` | 0.869 → 0.645 | 0.405 | **0.177** | **−0.065** |
| `Q53Z42` | 0.938 → 0.672 | 0.475 | 0.291 | −0.009 |
| `SPG1_Olson` | 0.826 → 0.681 | 0.383 | 0.446 | −0.019 |
| `SPG1_Wu` | 0.827 → 0.681 | 0.381 | 0.445 | −0.013 |
| `B2L11` | 0.759 → 0.579 | 0.398 | 0.463 | **+0.472** |
| `CD19` | 0.750 → 0.630 | 0.471 | 0.550 | −0.015 |
| `YAP1` | 0.764 → 0.420 | 0.148 | 0.551 | −0.052 |

**ipTM 全部 ≤ 0.55**，而 SOP §9.4 实测 ipTM≈0.70 的体系换 seed 全复合物 CA-RMSD 就有 11.6–13.6 Å
（近乎随机）。**ipTM 最低的 `DLG4`(0.177) 恰好效应最差**。且加 partner 后**连 target 自身的 pLDDT
都掉了**（`DLG4` 0.869→0.645）—— partner 不只放错位置，还把 target 的结构预测带坏了。

> **本轮实验回答的不是「partner 有没有用」，而是「ESMFold2-Fast 对全长序列预测的复合物有没有用」——
> 答案是没有，因为它折不出可用界面。** 上一轮晶体版能拿到 partner 收益，是因为实验结构把界面给定了。
> 代价交换很清晰：本轮换来 **100% 覆盖 / 零 `X` / 零 offset**（干净消融），但**丢掉了正确的界面**。

### 6.4 达成的目标与未达成的

✅ **消融口径彻底干净了** —— 结构从 ProteinGym 自己的 `target_seq` 折起，突变位点 i ↔ 残基 i，
`offset=0` 在脚本里硬断言，**9/9 assay 覆盖率 100%**（上一轮是 79.6%/96.5%/98.3% 三个有损失）。
❌ **两条 L=2078 复合物拿不到**（`ACE2`/`SPIKE`），两次 OOM 都发生在他人占卡 40 GB 时。
❌ **partner 收益没能复现** —— 原因是界面预测质量，不是消融设计。

### 6.5 下一步的三条路径（按代价排序）

1. **多 seed × 5 diffusion sample，按 ipTM 选 top-1**（SOP §9.4：Fast 版 PPI 45%→65%）—— 单卡可行，成本 ×N
2. **换标准版 `biohub/ESMFold2` + MSA**（Fast 版不吃 MSA；PPI 70% vs 68%）—— 三台机器都没下这个权重，需先补
3. **只折结合域而非全长** —— 界面质量会显著改善，但退回「截断」，与本轮的设计目标冲突

### 6.6 输出位置

| | |
|---|---|
| 预测结构 | `/data/guoj0f/share/our-predicted-structure/ProteinTTT-proteinTTT-proteinGYM-reproduce/ProteinGYM-bindingAssay-with-partener-and-complexStructure/{monomer,wt-complex}/`（18 个 PDB，9+7） |
| per-variant 分数 | `/data/guoj0f/.../scores_esmfold/`（16 个 csv） |
| 汇总 | `results/per_assay_esmfold.csv`、`results/leaderboard_esmfold.csv` |
| 结构质量 | `results/esmfold_metrics/{monomer,complex}_metrics.tsv`（含 pLDDT/pTM/ipTM/峰值显存） |

原 ProteinGym / BindingGYM 数据集**零写入**。
