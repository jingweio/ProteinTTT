# complex-mutant-structure-pred — ESMFold2-Fast mutant complex folding

（created 2026-08-30 00:33; status: **PLANNED** —— 计划待用户拍板，尚未启动生产运行）

## 1. Goal / hypothesis

用 `biohub/ESMFold2-Fast` 给 BindingGYM 的 mutant 折叠 **mutant complex structure**，回答两个问题：

- **(a) mutant backbone 相对 WT 变化大不大？single-site vs multi-site 是否不同？**
  假设：单序列 co-folding 模型对点突变高度不敏感（已有四篇独立测量支持，见
  [`structure_pseudolabel_methods_survey_20260829-231404.md`](structure_pseudolabel_methods_survey_20260829-231404.md) §3），
  所以 single-site 的 backbone deviation 很可能落在模型自身的 seed 噪声里；multi-site（k 最高到 21）才可能出信号。
- **(b) mut-vs-WT 的 backbone deviation 与 within-assay 的 DMS score 相关吗？**
  假设方向：deviation 越大 → binding 越差 → DMS score 越低（符号随各 assay 的 score 语义而定，需逐 assay 确认）。

> **前置结论（来自同目录两份 record）**：FoldX 那 2,080 个 mutant complex **不动 backbone**，
> 对 backbone-only scorer 在构造上不可见 —— 所以本任务不是"再验证一遍 FoldX"，而是第一次真正
> 拿到会动 backbone 的 mutant complex 坐标。

## 2. 🔴 Design & decision points（启动前需用户拍板的三条）

### D1. 必须先跑 NULL TEST，否则整个项目可能测的是噪声

ESMFold2 是 diffusion 模型，**同一条序列不同 seed 会给出不同结构**（`lm_dropout` 默认 0.3 正是
seed 间差异的来源）。所以：

> **mutant-vs-WT 的 backbone RMSD 只有在超过 seed-to-seed 噪声底时才有意义。**

Stage-0 必做：每个 assay 的 **WT 折 5 个 seed**，得到该 assay 的噪声底分布。若某 assay 的
mutant deviation 中位数 < 噪声底，该 assay 对目标 (a) 直接判负（这本身也是可发表的结论）。
成本 ≈ 0.4 GPU-h（24 个 WT × 5 seed）。

### D2. Config 选哪一档

论文 Fig 2E 数字化 + 源码核实（见 §6 证据）给出的成本结构：**5 个 diffusion sample 共享同一次 trunk
前向**，所以 samples 几乎免费，**seeds 才是真乘数**。

| | 配置 | 相对成本 | 说明 |
|---|---|---|---|
| A | 3 loops / 50 steps / 1×1 | 0.43× | 单体实测配置，复合物下未验证 |
| **B** | **10 loops / 68 steps / 1 seed × 1 sample** | **1.00×** | SOP §10.4 的复合物推荐参数，最小可用 |
| **C** | 10/68，1 seed × **5 samples** + ipTM top-1 | **1.21×** | ⭐ 性价比最高：只贵 21% 就拿到 5 选 1 |
| E | 10/68，5 seeds × 5 samples | 6.05× | 论文标准协议 |

> **本计划提议：config C。** 理由：(1) 只比 B 贵 21%；(2) ipTM top-1 能压掉一部分采样噪声，
> 这对目标 (a) 的信噪比是直接收益；(3) seed 预算改为只花在 D1 的 WT 噪声底上，那里才需要 seed 方差。
> ⚠️ **`fold()` 的默认值是 `num_loops=20, num_sampling_steps=200`，不是 3/50 也不是 10/68 —— 必须显式传参**，
> 否则白付 1.88×。

### D3. 每个 assay 抽样多少

全量 376,446 个 = **839 A100-h ≈ 单卡 35 天**，且对两个目标都没必要 —— within-assay Spearman
在 n=1,500 时 SE≈0.026，已经够判定相关性有无。

> **本计划提议：每 assay 分层抽样 1,500 个**（按突变点数 k 分层，保证 single 和各 multi 档都有足够样本），
> 24 个有效 assay 合计 **98 A100-h ≈ 单卡 4 天**。若某个 assay 出了信号，再对它加采到全量。

## 3. 🎯 Assay 排序（按目标对齐度，从最 aligned 开始跑）

排序依据两个分量各占一半：
- **目标 (a) 分量**：同一 assay 内是否**同时**有足量 single 和 multi（受控对照），以及 k 跨度。
- **目标 (b) 分量**：n 是否够算 per-assay Spearman，以及 DMS score 的动态范围（sd）。

| 序 | assay | L | n | single | multi | k_max | DMS sd | 对齐分 | s/结构 | 抽样1500 (h) | 累计 (h) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | `SARS2-RBD_ACE2_deltaKd_6M0J` | 791 | 21,872 | 3,134 | 18,737 | 10 | 1.86 | 1.0 | 23.1 | 9.6 | 9.6 |
| 2 | `hYAP65_peptide_FunctioncalScore_1JMQ` | 56 | 18,407 | 288 | 18,118 | 21 | 1.154 | 0.933 | 1.0 | 0.4 | 10.0 |
| 3 | `CD19_FMC63_Fitness_7URV` | 497 | 3,886 | 1,618 | 2,267 | 3 | 4.969 | 0.883 | 8.9 | 3.7 | 13.8 |
| 4 | `GB1_IgG-Fc_fitness_1FCC` | 262 | 92,891 | 1,045 | 91,845 | 2 | 1.008 | 0.785 | 3.1 | 1.3 | 15.0 |
| 5 | `CXCR4_CXCL12_enrich_8U4O` | 360 | 5,585 | 5,584 | 0 | 1 | 1.43 | 0.713 | 5.0 | 2.1 | 17.1 |
| 6 | `5A12_VEGF_fitness_4ZFF` | 528 | 29,981 | 54 | 29,926 | 9 | 0.962 | 0.709 | 10.0 | 4.2 | 21.3 |
| 7 | `4D5_HER2_fitness_1N8Z` | 1041 | 2,080 | 0 | 2,079 | 9 | 1.189 | 0.707 | 42.5 | 17.7 | 39.0 |
| 8 | `KRAS_RAF1_norfitness_6VJJ` | 245 | 12,677 | 1,188 | 11,488 | 2 | 0.508 | 0.701 | 2.8 | 1.2 | 40.2 |
| 9 | `KRAS_PICK3CG-RBD_norfitness_1HE8` | 1107 | 19,203 | 2,639 | 16,563 | 2 | 0.499 | 0.7 | 49.0 | 20.4 | 60.6 |
| 10 | `KRAS_DARPinK27_norfitness_5O2S` ⛔ | 321 | 19,533 | 2,507 | 17,025 | 2 | 0.458 | 0.693 | 4.2 | 1.8 | — |
| 11 | `KRAS_SOS1_norfitness_8BE4` | 643 | 19,425 | 2,503 | 16,921 | 2 | 0.457 | 0.693 | 14.9 | 6.2 | 66.8 |
| 12 | `KRAS_RALGDS-RBD_norfitness_1LFD` | 254 | 20,341 | 2,544 | 17,796 | 2 | 0.454 | 0.692 | 3.0 | 1.2 | 68.0 |
| 13 | `KRAS_RAF1-RBD_norfitness_6VJJ` | 245 | 23,162 | 2,825 | 20,336 | 2 | 0.392 | 0.682 | 2.8 | 1.2 | 69.2 |
| 14 | `ACE2_SARS2-RBD_enrich_6M17` | 931 | 2,186 | 2,185 | 0 | 1 | 1.086 | 0.656 | 33.0 | 13.8 | 83.0 |
| 15 | `HLA-A2_TAPBPR_meanscore_5WER` | 644 | 3,344 | 3,344 | 0 | 1 | 1.069 | 0.653 | 15.0 | 6.2 | 89.2 |
| 16 | `GB1_IgG-Fc_fitness_1FCC_2016` | 262 | 22,176 | 26 | 22,149 | 4 | 1.018 | 0.614 | 3.1 | 1.3 | 90.5 |
| 17 | `Z-domain_ZpA963_HL1_fitness_2M5A` | 116 | 2,904 | 24 | 2,879 | 6 | 0.618 | 0.579 | 1.4 | 0.6 | 91.1 |
| 18 | `Z-domain_ZSPA-1_LL2_fitness_1LP1` | 109 | 5,583 | 3 | 5,580 | 8 | 0.351 | 0.552 | 1.3 | 0.5 | 91.6 |
| 19 | `PSD95_CRIPT_1BE9` | 120 | 1,577 | 1,576 | 0 | 1 | 0.406 | 0.543 | 1.4 | 0.6 | 92.2 |
| 20 | `PSD95_Tm2F_1BE9` | 120 | 1,577 | 1,576 | 0 | 1 | 0.395 | 0.541 | 1.4 | 0.6 | 92.8 |
| 21 | `Z-domain_ZSPA-1_LL1_fitness_1LP1` | 109 | 45,476 | 3 | 45,473 | 9 | 0.14 | 0.534 | 1.3 | 0.5 | 93.3 |
| 22 | `BH3_Mcl-1_normed_3KZ0` | 173 | 518 | 170 | 347 | 5 | 0.492 | 0.487 | 1.9 | 0.3 | 93.6 |
| 23 | `Z-domain_ZpA963_HL2_fitness_2M5A` | 116 | 600 | 59 | 540 | 6 | 0.772 | 0.481 | 1.4 | 0.2 | 93.8 |
| 24 | `BH3_Bcl-xL_normed_1PQ1` | 229 | 518 | 170 | 347 | 5 | 0.319 | 0.459 | 2.6 | 0.4 | 94.2 |
| 25 | `5A12_Ang2_fitness_4ZFG` | 652 | 944 | 51 | 892 | 8 | 0.079 | 0.45 | 15.3 | 4.0 | 98.2 |
⛔ `KRAS_DARPinK27_norfitness_5O2S` **必须排除** —— 本 repo 已定案该 assay 装的是 SOS1 的数据（根因在 Nature 补充表）。

**排序解读**

- **第 1 名 `SARS2-RBD_ACE2_6M0J` 是目标 (a) 的最佳 assay**：k 从 1 到 10 连续覆盖、3,134 个 single + 18,737 个 multi、
  DMS sd 1.86。代价是 L=791（23.1 s/结构）。
- **第 2 名 `hYAP65_1JMQ` 是"最快见到答案"的那个**：L=56、1.0 s/结构、k 跨度最大（1–21）、
  抽样 1500 只要 **0.4 小时**。⚠️ 但它的 chain B 只有 10 aa（peptide），"backbone deviation"的语义与其他 assay 不同。
- **第 3 名 `CD19_FMC63_7URV`**：single/multi 最均衡（1,618 / 2,267），DMS 动态范围 4.97 为全库最大。
- **第 7 名 `4D5_HER2_1N8Z` 有独立价值**：这是唯一已有 2,080 个 FoldX mutant 结构的 assay，
  可做 **ESMFold2 mutant backbone vs FoldX mutant backbone vs WT** 的三方对照。但 L=1041（42.5 s/结构）且 **0 个 single**。
- **排在后面不代表没用**：`Z-domain_ZSPA-1_LL1`（n=45,476、k 到 9、L=109、全量只要 16 h）分低只因 DMS sd=0.14
  动态范围太窄，对目标 (b) 弱；但对目标 (a) 是便宜的大样本。

**建议执行顺序**（可随时终止，越早的越快出结果）：
1. **Stage 0**：24 个 WT × 5 seeds 噪声底（0.4 h）← 不做这个后面全部无法解读
2. **Stage 1**（快速见效，2.9 h）：hYAP65(0.4) → KRAS_RAF1_6VJJ(1.2) → GB1_1FCC(1.3)
3. **Stage 2**（决定性，15.4 h）：CD19_FMC63(3.7) → CXCR4(2.1) → SARS2-RBD_6M0J(9.6)
4. **Stage 3** 及以后：按上表顺序往下

## 4. Run config

| 项 | 值 |
|---|---|
| 机器 | workstation `10.67.24.41`，**A100 80GB PCIe** ×1，无调度器（`workstation-usage` skill） |
| 预检（2026-08-29 23:5x） | free 65,294 MiB / 81,920；util **0%**；⚠️ **另有 3 个他人 python 进程占 15,756 MiB**（未在算，不动它们） |
| 显存预算 | 权重 floor ~13.5 GB + activation（chunk=64，k≈14.3 KB/L²）→ L=1107 峰值 **~31 GB**。80 GB 宽裕 |
| conda env | `complex-mutant-structure-pred` @ `/data/guoj0f/miniconda3/envs/`（专属，勿复用） |
| torch | 驱动 535.230.02 → **CUDA 上限 12.2**，强制 `cu121` wheel（SOP §3.2 的 cu128 是 Ibex 口径，不可照抄） |
| esm 源码 | `/data/guoj0f/share/esm-latest/esm`（editable install），已从本地 rsync（28 MB） |
| 权重 | `HF_HOME=/data/guoj0f/share/hf_cache`：ESMFold2-Fast 721 MB + **ESMC-6B 24 GB**，已同步 ✅ |
| 输入数据 | `/data/guoj0f/share/BindingGYM/input`（1.3 GB，已在） |
| 产出 | `/data/guoj0f/complex-mutant-structure-pred/`（>5 GB 走 `/data`，skill §2） |
| ⚠️ 待办 | `ccd.pkl`（417 MB）本地与 workstation **都没有**，只在 Ibex 上。protein-only 是否需要它待验证 |

## 5. Change log

- 2026-08-30 00:33 计划创建（status PLANNED）。已完成：GPU 预检、权重同步（25 GB）、conda env 创建；pip install 进行中。

## 6. 成本估算的证据基础

- **锚点不是外推**：ESMFold2 论文 **Fig 2E** 有 published 的 latency-vs-length 曲线（ESMFold2-Fast，
  L=128–1024，H100，10 loops/200 steps）。数字化结果在 L=1024 给出 9.46 s，与正文写死的 "9.4 seconds" 吻合。
- 由该曲线换算到 A100 + 我们的精度路径（ESMC bf16 + trunk fp32，无 `torch.compile`）需乘一个 penalty，
  **该 penalty 是全案最大的单一不确定源，区间 1.5–3×** —— 所以下面的数字带区间而非点值。
- 全量 25 assay / 376,446 variants @ config B = **~840 A100-h [560–1,500]**。
- ⛔ **SOP §10.4 的"两个数量级"是错的**：真实值约 **14×**（高估约 8×）。原因：diffusion samples 共享 trunk；
  loops 只放大 trunk 项、steps 只放大 diffusion 项，三者不相乘。建议回填修订 SOP。

## 7. Results

（待填）

## 关联

- [`predicted_complex_structure_assets_20260829-000600.md`](predicted_complex_structure_assets_20260829-000600.md) —— 资产调研：现成的 mutant complex 只覆盖 0.55%
- [`structure_pseudolabel_methods_survey_20260829-231404.md`](structure_pseudolabel_methods_survey_20260829-231404.md) —— 方法调研：FoldX 不动 backbone；co-folding 对点突变不敏感的四篇证据
