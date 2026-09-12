# bindingGYM-zero-shot-benchmarking-evaluation — experiment record

> task: `bgym25_zeroshot_4model_benchmark` · created 2026-09-11 22:46 · **status: PLANNED**
> 锚点实验：[`workstation-records/BindingGYM-zero-shot-proteinMPNN/zeroshot_proteinmpnn_20260827-154500.md`](../../workstation-records/BindingGYM-zero-shot-proteinMPNN/zeroshot_proteinmpnn_20260827-154500.md)

> ## 📖 怎么读这篇（2026-09-12 16:56 更新）
>
> **只想看结论** → **§24.3（阶段性主结论，最新）**、§22.3、§17（侧链对比）、§12.1/§12.3/§12.4（MPNN 家族）。
> **想看设计为什么这么定** → §3（三族打分口径）、§4（组合库结构如何决定成本）。
> **想看踩过的坑** → §15（smoke 三轮）、§19（四个模型各自的 parser 差异）、
> **§21（一次差点写进结论的假结果 —— 建议优先看这条）**。
> **想看现在跑到哪** → §20。
>
> ⚠️ **结论有效性标注**
> - ✅ **已定论**：§12.1 的 5 个 MPNN run、§17 侧链对比、§22 ADFLIP、**§24 LASErMPNN(口径B) 与四模型主结论**
> - 🔄 **尚无有效结果**：StaB-ddG；LASErMPNN 的 **AR 口径**（5/25，见 §24.4）
> - ❌ **文档中不存在任何 LASErMPNN 的数值结论** —— 曾经算出的 0.3614 是新旧代码混合产物，已作废


## 1. Goal

把已复现的 **pretrained ProteinMPNN on BindingGYM zero-shot**（25 assay，Spearman 0.3914 vs 官方
0.3970）扩展成一张多模型 benchmark 表，纳入 **LigandMPNN / LASErMPNN / ADFLIP / StaB-ddG**。

核心问题：**更强的结构表征（侧链、全原子、配体感知）或更贴近热力学的目标（ddG），
能不能在 protein–protein binding DMS 上超过 backbone-only 的 ProteinMPNN？**

## 2. 调研结论 —— 四个模型各自能出什么（已实证，非推测）

| 模型 | ckpt | 打分入口 | 状态 |
|---|---|---|---|
| ProteinMPNN *(anchor)* | `v_48_020` | `_scores()` AR-NLL | ✅ 已跑完 |
| **LigandMPNN** | `ligandmpnn_v_32_020_25.pt` | `score.py` → `model.score()` / `model.single_aa_score()` | ✅ 开箱可用 |
| **LASErMPNN** | `laser_weights_0p1A_nothing_heldout.pt` | `utils/model.py:261 get_logits_for_score()` | ⚠️ API 在、**repo 内无人调用**，harness 要自写 |
| **ADFLIP** | `results/weights/ADFLIP_v1.pt` | `model(noisy_data, t)` → center 位 AA logits | ⚠️ 要写打分 wrapper |
| **StaB-ddG** | `model_ckpts/stability_finetuned.pt` | `run_stabddg.py` → binding ddG | ⚠️ 要推导 chain partition |

三处值得单独说明：

**(a) LigandMPNN 的"有/无侧链"就是一个 flag。** `--ligand_mpnn_use_side_chain_context 0/1`
控制是否把**固定残基的侧链原子**当作 context 喂进去，同一个 ckpt 即可。`model.score()` 是
**teacher-forced on 任意 `feature_dict["S"]`**，所以能直接套用我们已验证的 ProteinMPNN harness，
只换模型类与 ckpt。

**(b) ADFLIP 不需要训练新 head —— 与原计划的预期不同。**
`DiscreteFlow_AA.forward()` 里 `self.model(noisy_data, times)` **已经**在 center 位输出
amino-acid logits；`corrupt_data_by_sample()` 的 `keep_mask` 逻辑会在 mask 一个残基时
**同时移除它的侧链原子**、保留其余残基的身份+全部侧链坐标。
⇒ 把突变位点 mask 掉、其余喂 WT 全原子，forward 一次得到的就是
`p(aa_i | all-atom structure, rest-of-sequence)` —— 这**正是**所需的 structure–seq
compatibility likelihood。所谓 "header" 实为一个**打分协议 wrapper**（无新参数、无需训练）。

**(c) StaB-ddG 必须用 stage 2，这不只是偏好而是防泄漏。**
`model_ckpts/` 三个 ckpt 对应三阶段：`proteinmpnn.pt`(stage1) →
**`stability_finetuned.pt`(stage2, Megascale 稳定性)** → `stabddg.pt`(stage3, SKEMPI binding)。
stage3 在 SKEMPI 上 finetune 过，而 SKEMPI 与 BindingGYM 的复合物有重叠 ⇒ **用 stage3 就是 leakage**。
stage2 对 binding 是严格 zero-shot，是唯一正确选择。

## 3. 🔴 KEY DECISION —— 打分口径分三族，不可混谈

这是本实验最容易出错的地方：**四个模型并不共享同一个打分口径**，混在一张表里直接比大小会误导。

| 口径 | 定义 | 谁能出 |
|---|---|---|
| **A. AR-NLL** | autoregressive teacher-forced NLL，`Σ log p(aa_t \| bb, aa_{<t})`；BindingGYM 官方口径 | ProteinMPNN / LigandMPNN / LASErMPNN |
| **B. joint-masked marginal** | mask 掉该 variant 的**全部**突变位点（连侧链一起移除），其余残基保留身份+侧链，一次 forward 取各位点 logits，`Σ_j [log p(mt_j) − log p(wt_j)]` | **全部四个**（ADFLIP **只能**走这条） |
| **C. binding ddG** | 物理量，非 likelihood | StaB-ddG |

**决策：A 和 B 都跑。**
- A 保证与已有 anchor 严格同口径、可直接续表；
- **B 是唯一能把四个模型放进同一列的口径** —— 没有 B，ADFLIP 就无法与任何模型公平比较；
- C 单列，但因 BindingGYM 的 metric 本身是 rank-based（Spearman/AUC/MCC/NDCG/AP），
  ddG 仍可进同一张 Spearman 表，**只需注明机制不同**。

**B 不是偷懒的近似**：它是 ProteinGym masked-marginal 的标准做法（公认口径，非我们发明），
且 B 显式保留了 variant 内部的**位点集合**，只是假设这些位点在给定其余全原子 context 下条件独立
（即不建模突变位点**彼此之间**的 epistasis）。A 建模，B 不建模 —— 这个差异会如实写进结果表。

### 3.1 口径 B 的落地方式（P0 调研后的修正）

读了 LigandMPNN `model_utils.py` 后发现两件事，口径 B 的实现必须据此定：

- **不能直接用 `single_aa_score`** —— 它内部 `for idx in range(L)` 对**每个**位点各跑一次完整
  decoder（`4D5_HER2` 的 L=1041 就是 1041 次 / 每次调用），逐 variant 调用完全不可行。
- **但它给出了正确写法**：位点 i 的 logits 由 `order_mask[i]=0`（把 i 排到解码序最前）得到，
  条件于 `S_true` 的其余位点。把 `order_mask` 在**该 variant 的整组突变位点**上一起置 0，
  **一次 decoder pass** 就同时得到这 k 个位点的 log-probs，各自条件于"其余位点保持 WT"。

⇒ 口径 B 实现为 `joint_masked_score(feature_dict, pos_set)`，写在**我们自己的 wrapper 里**，
不改 vendor repo。成本 = **每个 distinct 位点组合一次 pass**（组合库内所有 variant 共享同一
masked 上下文）⇒ 全库 32,977 次，而非 376,446 次。

这同时让口径 B 与 ADFLIP 的天然口径**严格同构**（都是"mask 掉这组位点、条件于其余全部"），
四个模型因此真正可比。

> 顺带记录一个被否决的替代方案：把 `single_aa_score` 限制到突变位点、并条件于**mutant**
> 的其余位点（即 pseudo-likelihood，能通过条件捕捉 epistasis）。它更强，但成本是
> 每 variant k 次 pass（≈376k×3.5），且与 ADFLIP 不同构 —— 故作为可选的 follow-up，不进主表。

### 3.2 `_scores` 的确切定义（逐字复刻，勿改）

官方 `protein_mpnn_utils.py:39-47` 的 `_scores` **长度归一被注释掉了**：
```python
scores = torch.sum(loss * mask, dim=-1)   # / torch.sum(mask, dim=-1)  ← 注释掉的
```
即**未归一的 NLL 求和**。`design_score = -mean_over_M(_scores(S, log_probs, mask*chain_M*chain_M_pos))`，
`global_score` 用 `mask`。新模型一律复刻此定义，不得"顺手修正"成均值。

## 4. 数据规模 —— 组合库结构决定了成本

实测 25 assay：**376,446 variants，但只有 2,220 个突变位点、32,977 个 distinct 位点组合。**

BindingGYM **不是单点突变库**：25 个 assay 里只有 4 个是 ≥99% 单点
（`CXCR4`/`HLA-A2`/`ACE2_SARS2-RBD`/`PSD95`×2），其余多为**极少位点上的组合库**：

| assay | n | 突变位点数 | 单点占比 | max k |
|---|---|---|---|---|
| `Z-domain_ZSPA-1_LL1_1LP1` | 45,476 | **9** | 0.0% | 9 |
| `GB1_IgG-Fc_1FCC_2016` | 22,176 | **4** | 0.1% | 4 |
| `5A12_VEGF_4ZFF` | 29,981 | **9** | 0.2% | 9 |
| `4D5_HER2_1N8Z` | 2,080 | **9** | 0.0% | 9 |

两个后果：
1. **逐位点可加的 marginal 会废掉这些 assay** —— 组合库内所有 variant 共享同一位点集，
   可加分数退化成 9 个固定项的和。所以口径 B 必须是 **joint**-masked（整组一起 mask），不是逐位点相加。
2. **joint-masked 的成本按位点组合算，不按 variant 算** —— 组合库内所有 variant 的 masked 输入
   **完全相同**（masked 位点的 token 与侧链都被移除，与替换成哪个 AA 无关）。
   ⇒ **32,977 次 forward 覆盖全部 376,446 个 variant，11.4×** 便宜。这是 ADFLIP 可行的前提。

## 5. 模型矩阵（7 个 run）

| # | run_id | 模型 | 口径 | 关键设置 |
|---|---|---|---|---|
| 0 | `proteinmpnn_ar` | ProteinMPNN v_48_020 | A | 已有，直接引用 |
| 1 | `ligandmpnn_ar_nosc` | LigandMPNN v_32_020_25 | A | `use_side_chain_context=0` |
| 2 | `ligandmpnn_ar_sc` | 同上 | A | `use_side_chain_context=1` |
| 3 | `ligandmpnn_mm_sc` | 同上 | B | `single_aa_score`，`=1` |
| 4 | `lasermpnn_ar` | LASErMPNN 0p1A_nothing_heldout | A | `get_logits_for_score`，无 ligand |
| 5 | `adflip_mm` | ADFLIP_v1 | B | joint-masked wrapper |
| 6 | `stabddg_s2` | StaB-ddG stage2 | C | `stability_finetuned.pt`, `mc_samples=20` |

> ProteinMPNN 也补一个口径 B 的 run（`proteinmpnn_mm`），否则 B 列缺少 anchor —— 用 LigandMPNN
> repo 的 `--model_type protein_mpnn --single_aa_score 1` 跑，权重与 anchor 同为 `v_48_020`。

## 6. 🔴 产物规范（用户明确要求：per-mutant 分数必须落盘）

每个 run × 每个 assay **一个 csv**，与 anchor 那轮 `scores/seed1_M5/*.csv` 同构：

```
/ibex/user/guoj0f/bindingGYM-zs-benchmark/scores/{run_id}/{DMS_id}.csv
  列 = BindingGYM 原 csv 全部列 (mutant, DMS_score, ...) + score + seed + run_id
```
- **不做任何行过滤、不改变行序** —— 下游可直接按行 join 回原表。
- 逐 assay 指标另存 `results/per_assay_{run_id}.csv`；汇总表 `results/leaderboard.csv`。
- 大件留在 `/ibex/user/guoj0f/`，**只把 per-assay 指标与汇总表回流进 repo**（§2 gitignore 大件）。

## 7. 指标口径

**逐字复用** anchor 那轮已验证的 `refs/bindinggym_metrics.py`（移植自 BindingGYM 官方），
输出 Spearman / AUC / MCC / NDCG / AP 五项，25-assay 均值。不重新实现。

## 8. 执行分期（每个 job walltime ≤ 8 h）

> ⚠️ **已更正（2026-09-11 23:40）**：先前写的「1×a100 且 `--time ≤8h` 基本立刻起」是
> 从**别人**的 pending 作业预估里读出来的，**不适用于我们账号**。实测我们提交的
> 2h / 1×a100 job 被排到 **+21 小时**（est `2026-09-12T20:52`）。
> 原因：`sshare` 显示我们 **FairShare = 0.0997**（RawUsage 2,575,080 / NormShares 0.0256），
> `sprio` 的 997 分几乎全部来自 fairshare 项。
> ⇒ **短 walltime 仍然有帮助，但不足以抵消低 fairshare。** 两条应对：
>   1. **不需要 GPU 的一律走 CPU-only** —— `batch` 分区有 99 个 idle 节点 / 8,272 个 idle 核，
>      CPU job 几分钟就起（实测建 env 的 job 从 GPU 的 +21h 变成立刻 RUNNING）。
>   2. **用 `--dependency` 提前入队**，并让所有打分脚本**幂等**（输出已存在即跳过），
>      这样被 TIMEOUT 截断后重投即续跑。

| 期 | 内容 | 产出 |
|---|---|---|
| **P0** | 建 env（§9）+ 同步 BindingGYM 数据到 ibex + 3-assay smoke probe | 计时数据，定后续 walltime |
| **P1** | run 1/2/3 + `proteinmpnn_mm`（LigandMPNN repo，一个 env 全覆盖） | A/B 两列的 MPNN 家族 |
| **P2** | run 4 LASErMPNN（自写 harness） | A 列补齐 |
| **P3** | run 5 ADFLIP（joint-masked wrapper） | B 列补齐 |
| **P4** | run 6 StaB-ddG stage2（含 chain partition 推导） | C 列 |
| **P5** | 汇总 + 与 anchor 对照 | leaderboard |

## 9. Env 计划（§1b-0：env 先于任何项目代码）

| env | 用途 | 依据 |
|---|---|---|
| `bgym-official` *(已存在)* | 指标计算、数据解析 | py3.8.20 / torch 1.13.1+cu117 / np 1.24.4 —— anchor 那轮的口径来源 |
| `ligandmpnn-bgym` *(新建)* | run 1/2/3 + proteinmpnn_mm | LigandMPNN `requirements.txt`；py3.11 |
| `lasermpnn-bgym` *(新建)* | run 4 | repo 自带 `conda_env_cuda_12p4.yml` |
| `adflip-bgym` *(新建)* | run 5 | repo README：py3.10 + torch 2.1.0+cu121 + torch-cluster/scatter |
| `stabddg` *(已存在)* | run 6 | py3.10.20 / torch 2.6.0+cu124，StaB-ddG 专用 env |

CUDA：Ibex a100 节点驱动为 CUDA 12.x ⇒ 所有 torch wheel **不得跨到 cu13x**（§4 已记录的两次事故）。

## 10. 待确认的风险（会在 P0 后更新本节）

1. **LASErMPNN 吞吐未知** —— 等变图网络 + 全原子，可能比 ProteinMPNN 慢 10–30×。P0 probe 定量后
   若 A 口径不可行，退化为只出 B 口径并明确标注。
2. **LASErMPNN 的 chi teacher-forcing**：打分位点 i 的 logits 只依赖解码序在它之前的位点的
   seq+chi，**位点 i 自身的 chi 不参与** ⇒ 单点突变无问题；多点突变时先解码的突变位点会带着
   **WT 的 chi 配 mutant 的 aa**，是一处已知不一致，将如实记录。
3. **StaB-ddG 的 chain partition** BindingGYM.csv 未提供，需从 complex 结构 + 被突变链推导；
   25 个 assay 里 3 个是三链（`4D5_HER2` ABC、`5A12` AHL/CHL），划分需逐个核对。
4. **ADFLIP 的 time step t**：joint-masked 时只有极少位点被 mask，对应 flow 的 t≈1。
   P3 会在 3 个 assay 上扫 t 以确认取值，不盲取。

## 11. Change log

- 2026-09-11 22:46 · 建 project，完成四个 repo 的能力调研，写下本计划（status PLANNED）。
- 2026-09-11 23:0x · P0：数据/权重/repo 全部上 ibex 并校验；发现 login node 建 env 会被 Kill，改走 sbatch(job 51751154)；
- 2026-09-11 23:40 · 更正 §8 的排队判断：我方 FairShare 仅 0.0997，2h/a100 被排到 +21h。
  建 env 改走 CPU-only(立刻 RUNNING)；GPU 只留给真正要 GPU 的打分。
- 2026-09-11 23:45 · 冻结 chain partition(P4 阻塞项解除)；提交 P1 smoke(job 51753075, 依赖 51752817)。
- 2026-09-12 00:1x · smoke 三轮修复:setuptools<81 补 pkg_resources;heredoc 变量展开 bug;
- 2026-09-12 00:4x · smoke #5 gate 通过(两个 assay rho=1.0000 逐位复现 anchor)。
- 2026-09-12 05:0x · P1 全量 25/25;发现侧链 flag 无作用对象,加 --designed_chains mutated 重跑。
- 2026-09-12 13:4x · 侧链对比出结果(+0.0082, p=0.0158,含 4 个零差异内部对照,见 §17)。
- 2026-09-12 14:2x · 补记 §17-§20:吞吐/walltime 策略、P2-P4 的失败与修复、五类 parser 残基集合差异。
- 2026-09-12 16:5x · 抓到假结果:lasermpnn_jm 的 0.3614 是新旧代码混合产物(23/25 用修复前代码)。
  根因是幂等跳过不校验代码版本。已加 code_stamp、作废 LASErMPNN 全部结果并重跑;
  逐个查证其余 run 未受影响(§21)。
  ⚠️ 本次补记前,约 6 小时的进展只在 commit message 里、未进文档正文 —— 已纠正。
  探针纠正了 joint_masked 的解码序方向(原写法得到的是 backbone-only)。P1 五个 config 已提交。
  失败传播(坏 job 曾伪装成 COMPLETED);parse_atoms_with_zero_occupancy=True(BindingGYM 结构 occ 全为 0,
  且官方管线本就不按 occupancy 过滤 ⇒ 置 True 才与 anchor 一致)。
  据 LigandMPNN 源码修正口径 B 的实现为 joint_masked_score（1 pass/位点组合，而非 single_aa_score 的 L 次/次调用）。

## 12. Results

### 12.1 P1 —— MPNN 家族,25/25 全齐

| run_id | 口径 | Spearman | AUC | MCC | NDCG | AP |
|---|---|---|---|---|---|---|
| `proteinmpnn_mm` | B | **0.389911** | 0.685402 | 0.155203 | 0.719684 | 0.220039 |
| `proteinmpnn_ar` | A | **0.389889** | 0.686126 | 0.153173 | 0.720556 | 0.220086 |
| `ligandmpnn_ar_nosc` | A | 0.388257 | 0.686058 | 0.153354 | 0.721301 | 0.223911 |
| `ligandmpnn_ar_sc` | A | 0.388257 | 0.686058 | 0.153354 | 0.721301 | 0.223911 |
| `ligandmpnn_mm_sc` | B | 0.377051 | 0.682353 | 0.149206 | 0.718931 | 0.221173 |

参照：**anchor 那轮 0.391356**，**官方发布 0.396950**。
本次 `proteinmpnn_ar` = 0.389889，与 anchor 差 **−0.0015**（σ_combined 0.007–0.013 之内），
与官方差 −0.0071。差异来源是三路映射对 `3KZ0`/`4ZFF`/`4ZFG` 的处理与官方 parser 不同 + 解码序随机。

### 12.2 🔴 LigandMPNN 的「有/无侧链」在 BindingGYM 的链约定下**不可能有差别**

`ligandmpnn_ar_sc` 与 `ligandmpnn_ar_nosc` **五项指标、25 个 assay 全部逐位相同**。
根因在 `LigandMPNN/model_utils.py:1252`：

```python
mask_residues = input_features["chain_mask"]
xyz_37_m = xyz_37_m * (1 - mask_residues[:, :, None])   # 侧链只对 chain_mask==0 的残基生效
```

侧链原子**只有 fixed 残基（`chain_mask==0`）才作为 context**。而 BindingGYM 的 `chain_id`
把**所有链都列为 designed** ⇒ `chain_mask` 全 1 ⇒ `xyz_37_m*(1-1)=0` ⇒ **侧链 context 恒为零**。

⇒ 要真正测这个轴，必须把 **partner 链设为 fixed**。已统计：**21/25 个 assay 有一条从不被突变的
partner 链**可以设 fixed（4 个 Z-domain assay 两条链都突变，做不了）。
已加 `--designed_chains mutated` 并提交配对实验 `ligandmpnn_ar_scfix` / `ligandmpnn_ar_noscfix`
（job 51772043 / 51772047）。⚠️ 该 config 的 `chain_mask` 与 anchor 不同 ⇒ 解码序与
`design_score` 的打分范围都变了，**只能与自己的对照比，不能直接并进主表**。

### 12.3 口径 A vs 口径 B：**均值等价，但逐 assay 差异很大**

ProteinMPNN 上配对对照（同 ckpt、同 seed，唯一差别是口径）：

```
mean Δ = +0.0000   sd = 0.0446   正号 12/25   Wilcoxon p = 0.874
```

**均值上完全无法区分**，而口径 B 便宜约 **8×**（`proteinmpnn_mm` 17 min vs `ligandmpnn_ar_*` 2.5 h）。

⚠️ 但**不能据此说两个口径给出的是同一套分数**：配对差的 **sd = 0.0446 ≈ 2.4× 解码序噪声 σ(0.0184)**，
逐 assay 摆动很大（如 `Z-domain_ZpA963_HL1` 0.1839 → 0.3343，`GB1_IgG-Fc_2016` 0.4842 → 0.3892）。
是**均值相消**，不是逐 assay 一致。对单个 assay 下结论时必须注明用的是哪个口径。

### 12.4 LigandMPNN ≈ ProteinMPNN

配对检验（同口径 A、同 25 assay）：

```
proteinmpnn_ar 0.3899  vs  ligandmpnn_ar_nosc 0.3883
Δ = +0.0016   正号 17/25   Wilcoxon p = 0.3388   ⇒ 无显著差异
```
（口径 B 下：`proteinmpnn_mm` 0.3899 vs `ligandmpnn_mm_sc` 0.3771，Δ=+0.0129，
12/25，p = 0.3254 —— 同样不显著。）

BindingGYM 全是 protein–protein、**没有小分子配体**，
LigandMPNN 相对 ProteinMPNN 的增量能力（原子级配体 context）在这里没有用武之地 —— 结果符合预期。


## 13. P0 执行记录（2026-09-11）

**已完成：**
- BindingGYM 数据 → ibex 共享区 `/ibex/user/guoj0f/share/BindingGYM/input/`
  （624 MB / 51 文件，**双侧文件数核对 OK**：`Binding_substitutions_DMS` 28/28、`structures` 22/22；
  `msas/` 未同步 —— 结构类模型用不上）。
- 官方 harness `baselines/protein_mpnn/{compute_fitness_multi_pdb.py,protein_mpnn_utils.py}`
  与 `training/cache/v_48_020.pt` → 同一共享区，**ibex 侧 md5 `91d54c97a68bf551114f8c74c785e90f`
  与 anchor 那轮逐位一致**。
- 四个模型 repo → 按分支隔离的 `…/{branch}/ibex-records/{project}/models/`，
  **rsync 空跑四个全部 0 待传**。StaB-ddG 的 `output/`（stage3 SKEMPI-finetuned ckpt）
  **物理排除**，作为 leakage guard。
- LigandMPNN 权重下载并镜像到 `/ibex/user/guoj0f/share/model_params/LigandMPNN/`：
  `proteinmpnn_v_48_020.pt` md5 **`91d54c97…` = anchor 同一份权重**；
  `ligandmpnn_v_32_020_25.pt` `c2488988…`；`ligandmpnn_v_32_010_25.pt` `5cb0c045…`。

**踩到的坑（已记录，供后续 session）：**
- ⚠️ **login node 上建 conda env 会被 Kill。** `ligandmpnn-bgym` 在 login node 上
  `conda create` 时被 `Killed`（`ulimit` 显示 unlimited、节点有 265 GB available ⇒ 不是
  ulimit，而是 Ibex login node 的 process reaper）。**改用 sbatch 在计算节点建 env**，
  顺带在真 a100 上做 §4 要求的经验验证（`torch.cuda.is_available()` + 一次真 matmul）。
  `adflip-bgym` 侥幸在 login node 建成 —— 说明这是负载相关的间歇失败，更该走 sbatch。
- StaB-ddG 同步后 `examples/` 双侧差 8 个文件，查证为 `examples/list_of_mutations/output/`
  下的示例运行产物，被 `--exclude 'output'` 正确排除 ⇒ **预期行为，非缺漏**。

**env 状态：** `bgym-official`(已有,指标口径) / `stabddg`(已有) / `adflip-bgym`(已建) ；
`ligandmpnn-bgym`+`lasermpnn-bgym` 由 sbatch job **51751154** 建（含 CUDA 经验验证与 bytecode 预编译）。

## 14. P1 准备与 chain partition（2026-09-11 23:4x）

**chain partition 已冻结** —— `refs/chain_partition.tsv`，fingerprint md5 `068dae9cc3338368f0e89ebe39e752f4`
（生成环境 python 3.10.20 / numpy 2.2.6 / pandas 2.3.0，在 StaB-ddG 自己的 `stabddg` env 内算）。

22/25 是两链 ⇒ 平凡。3 个三链 assay 由重原子接触图（<5 Å）无歧义解开，
**且与 assay 命名语义独立吻合**（两条互不依赖的证据）：

| assay | 链(长度) | 最大接触对 | partition |
|---|---|---|---|
| `4D5_HER2_1N8Z` | A:214 B:220 C:607 | **A-B:942**（Fab 轻重链配对） | `AB_C` |
| `5A12_Ang2_4ZFG` | A:220 H:219 L:213 | **H-L:858** | `HL_A` |
| `5A12_VEGF_4ZFF` | C:96 H:219 L:213 | **H-L:912** | `HL_C` |

规则：2 链平凡；3 链取接触数最大的一对为 binder1，余者为 binder2。
运行时**只读此文件**，不重算（§1b-0）。

**P1 smoke 的三个 gate**（job 51753075，1h walltime，依赖 env build 51752817）：
1. `use_sequence` 的语义方向 —— 读 LigandMPNN 源码得到的方向与其 CLI help **相反**，
   用「扰动其余位点、看目标位点 log_probs 是否变化」实验定案，不靠推理。
2. **正确性 gate**：`score_bgym_mpnn.py` 用 `protein_mpnn` + `v_48_020` + AR + seed 1 + M=5
   跑 3 个小 assay，与 anchor 的逐 variant 分数对照。
   判据 `rho(mine, anchor) > 0.95`（两者只差解码序随机），且两侧 `rho_DMS` 落在
   per-assay σ≈0.018 内。anchor 对照数据已取到 `refs/anchor_scores/`（3 个 assay，共 2,695 行）。
3. 实测吞吐 → 定后续 job 的 walltime。

**smoke assay 选取**：`Z-domain_ZpA963_HL2_2M5A`(600, L=116)、`BH3_Mcl-1_3KZ0`(518, L=173)、
`PSD95_CRIPT_1BE9`(1577, L=120) —— 都小且快，1h walltime 足够，排队也最快。

## 15. P1 smoke 的三轮失败与修复（2026-09-11 23:5x ~ 00:1x）

smoke 连挂三次，每次都暴露一个会毒害正式结果的问题，值得逐条记下。

**#1（job 51753075，9 s，却报 COMPLETED）**
- `ModuleNotFoundError: pkg_resources` —— ProDy 依赖它，新版 setuptools 已移除；**三个新 env 全中**。
  修：三个 env 各装 `setuptools<81`（→ 80.10.2，ProDy 2.4.1/2.6.1 均可 import）。
- `FileNotFoundError: '/input/BindingGYM.csv'` —— sbatch 外层用了**未加引号**的 heredoc，
  内层 python 里的 `$BG`/`$OUT`/`$R` 在**写文件时**就被本地 shell 展开成空串。
  修：内层 python 抽成独立文件（`smoke_indices.py` / `compare_to_anchor.py`），参数全走 argv。
- ⚠️ **最该记的一条**：脚本用 `set -uo pipefail`（没有 `-e`），**失败的 job 报成 `COMPLETED`**。
  修：显式 `FAIL` 标志 + `exit $FAIL`，且 `compare_to_anchor.py` 在 `rho ≤ 0.95` 时 `exit 1`。
  不修的话，后续每个失败 job 都会显示成功，要等几小时拿到空结果才发现。

**#2（job 51753682，19 s，正确地 FAILED）** —— 失败传播已生效。
- `AttributeError: 'NoneType' object has no attribute 'select'`，出在
  LigandMPNN `data_utils.py:780` 的 `atoms = atoms.select("occupancy > 0")`。
- 根因：**BindingGYM 全部 22 个结构的 occupancy 都是 0.00**（多为同源模型 `*_hm.pdb`）
  ⇒ 该 select 选中 0 个原子、返回 `None`，下一个 `.select` 即崩。
- 修：`parse_atoms_with_zero_occupancy=True`。**这不是绕过，而是口径正确的选择** ——
  官方 BindingGYM 的 `parse_PDB` 根本不看 occupancy，anchor 那轮用的就是全部原子；
  置 True 才与 anchor 逐原子一致。
- 幸运之处：它是**崩溃**而非静默返回子集。否则我们会在残缺结构上打分而毫无察觉。
  已额外加 `assert pdict is not None and pdict["mask"].numel() > 0`。

**#3 ~ #5** —— 又两个 dtype/对齐问题，最后通过：
- `#3`：**`parse_PDB` 的残基集合与官方不同**。硬 assert 报 `BH3_Mcl-1_3KZ0 链A: 143 vs 150`。
  查证：该链编号 `172..321`（**跨度正好 150**），缺 `196–202` 共 7 个。
  ⇒ BindingGYM 的序列按**残基编号跨度**索引；官方 ProteinMPNN 的 `parse_PDB` **按编号补洞**
  （缺失位 `mask=0`，对 `_scores` 贡献 0），而 LigandMPNN 的 `parse_PDB` **只返回观测到的残基**。
  修：按 `(chain, 编号−该链最小编号)` 建 `pos_map`，未观测位不参与打分 —— 与 anchor 的 `mask=0` 等价。
  **没有这个 assert，就会按错位的序列打完全部 376,446 个 variant。**
- `#4`：`fd["S"]` 是 int32 而 token 张量默认 int64 → `index_put` dtype 不匹配；
  `#5`：`NLLLoss` 的 target 必须 int64（官方 `tied_featurize` 产出 int64，LigandMPNN 的 `featurize` 产出 int32）。

### smoke 结论（job 51754669，52 s，**gate 通过**）

**(1) `use_sequence` 的语义 —— LigandMPNN 的 CLI help 是错的**
```
use_sequence=True   max|Δlog_probs| = 0.000e+00   ⇒ 只看 backbone
use_sequence=False  max|Δlog_probs| = 9.271e-01   ⇒ 条件于其余序列
```
其 CLI help 写的是「1 = 用氨基酸序列信息，0 = 只用 backbone」，**方向相反**。
⚠️ **这个结果直接纠正了我自己实现里的一处方向错误**：`joint_masked_logprobs` 原本写成
`order_mask=ones, pos_idx=0`（把突变位点排到解码序**最前**）—— 那样得到的是 **backbone-only**
分数而非口径 B。已改为 `order_mask=zeros, pos_idx=1`（排到**最后** ⇒ 条件于其余全部）。
LASErMPNN 侧同一逻辑同步修正。**若无此探针，口径 B 的四个模型全部会是错的。**

**(2) 正确性 gate —— 通过，且两个 assay 逐位复现 anchor**

| assay | n | rho(mine, anchor) | rho_DMS 本次 | rho_DMS anchor |
|---|---|---|---|---|
| `PSD95_CRIPT_1BE9` | 1577 | **1.0000** | 0.3677 | 0.3677 |
| `Z-domain_ZpA963_HL2_2M5A` | 600 | **1.0000** | 0.3774 | 0.3774 |
| `BH3_Mcl-1_3KZ0` | 518 | 0.9889 | 0.6517 | 0.6514 |

判据是 `> 0.95`，实测两个 **1.0000**。`BH3_Mcl-1` 的 0.9889 正对应那个 7 残基缺口
（补洞方式的细微差异影响了邻居图），`rho_DMS` 只差 0.0003 ≈ σ/60，可接受且有解释。

**(3) 吞吐**：2,695 variants / 52 s（含模型加载与 CUDA 验证）⇒ 约 **35,000 (n·L)/s**，
比 anchor 的 13,000 快 2.7×（省掉了逐 assay 起子进程的开销）。
全库 Σn·L = 135,744,347 ⇒ **单个 config 约 65 min** ⇒ walltime 取 2 h。

### env 的两个补装（模块级 import 咬人）
建 env 时按「推理用不上」裁掉的包，其实在**模块级**被 import：
`lasermpnn-bgym` 缺 `logomaker`（`utils/helper_functions.py` 顶层 import）、
`adflip-bgym` 缺 `matplotlib`（`model/discrete_flow_aa.py` 顶层 import）。已补装并验证 import 通过
（LASErMPNN `aa_short_to_idx[A,R,C]=0,1,4` ✓；ADFLIP `<MASK>=1`, vocab=33 ✓）。
**教训：不要按「推理是否需要」去裁 repo 声明的依赖。**

> 通用教训：**沿用一个 vendor 的 `parse_PDB` 时，它的默认过滤条件可能与 benchmark 的
> 官方管线不同。** 这里 LigandMPNN 与 BindingGYM 在 occupancy 上的默认行为就不一致，
> 而 BindingGYM 的结构恰好落在分歧点上。

## 16. P1 全量已提交（2026-09-12）

gate 通过后提交 5 个 config，各一个 job，2 h walltime、幂等（输出已存在即跳过，被截断可重投续跑）：

| job | run_id | 模型 | ckpt | sc_ctx | 口径 |
|---|---|---|---|---|---|
| 51754950 | `proteinmpnn_ar` | ProteinMPNN | `v_48_020` | 0 | A |
| 51754951 | `ligandmpnn_ar_nosc` | LigandMPNN | `v_32_020_25` | 0 | A |
| 51754952 | `ligandmpnn_ar_sc` | LigandMPNN | `v_32_020_25` | **1** | A |
| 51755044 | `proteinmpnn_mm` | ProteinMPNN | `v_48_020` | 0 | **B** |
| 51755045 | `ligandmpnn_mm_sc` | LigandMPNN | `v_32_020_25` | **1** | **B** |

输出：`/ibex/user/guoj0f/bindingGYM-zs-benchmark/scores/{run_id}/{DMS_id}.csv`，
保留原 DMS csv 全部列与行序 + `design_score`/`global_score` + `seed` + `run_id`。

---

## 17. 🎯 LigandMPNN 的侧链对比 —— 结果（2026-09-12，25/25 完成）

§12.2 指出：照 BindingGYM 的链约定，侧链 flag **不可能有作用对象**。改用
`--designed_chains mutated`（把从不被突变的 partner 链设为 fixed）后，25/25 跑完：

| | Spearman | AUC | MCC | NDCG | AP |
|---|---|---|---|---|---|
| `ligandmpnn_ar_scfix`（+侧链） | **0.389560** | 0.688378 | 0.162368 | 0.726332 | 0.223533 |
| `ligandmpnn_ar_noscfix`（−侧链） | 0.382713 | 0.684321 | 0.155455 | 0.719661 | 0.220795 |
| Δ | **+0.0068** | +0.0041 | +0.0069 | +0.0067 | +0.0027 |

**五项指标全部偏向 +侧链。** 限定在 21 个**确实有 fixed partner 链**的 assay 上做配对检验：

```
+侧链 0.4081   −侧链 0.3999   mean Δ = +0.0082
sd 0.0134      正号 15/21      Wilcoxon p = 0.0158
```

✅ **内部对照（这条最关键）**：4 个两条链都被突变、**没有 fixed 链**的 Z-domain assay，
Δ = **+0.000000（逐位精确相同）**。这从实验上反证了 §12.2 定位的机制
（`model_utils.py:1252` 的 `xyz_37_m * (1 - chain_mask)`），而不只是读代码的推断。

逐 assay（21 个，按 Δ 降序摘录）：`KRAS_RAF1_6VJJ` +0.0355、`KRAS_SOS1_8BE4` +0.0293、
`KRAS_RAF1-RBD_6VJJ` +0.0229、`hYAP65_1JMQ` +0.0198、`4D5_HER2_1N8Z` +0.0194 …
负向的 6 个：`BH3_Bcl-xL_1PQ1` −0.0183、`KRAS_DARPinK27_5O2S` −0.0122、`BH3_Mcl-1_3KZ0` −0.0066、
`HLA-A2_TAPBPR_5WER` −0.0037、`5A12_Ang2_4ZFG` −0.0012、`5A12_VEGF_4ZFF` −0.0007。

⚠️ **口径说明**：该 config 的 `chain_mask` 与 anchor 不同（解码序与 `design_score` 的打分范围
都变了），**只能与自己的对照（`noscfix`）比，不能并进 §12.1 的主表**。

## 18. 吞吐实测与 walltime 策略（风险 #1 兑现）

| run | 2 h 内完成 | 外推全量 |
|---|---|---|
| `proteinmpnn_mm`（口径 B） | 23/25（17 min） | ~20 min |
| `ligandmpnn_ar_*`（口径 A） | 25/25 | ~2.5 h |
| `lasermpnn_jm`（口径 B） | 12/25（1 h 37） | ~3.5 h |
| **`lasermpnn_ar`（口径 A）** | **3/25** | **~16 h** |
| `stabddg` | 8/25 | ~6 h |

§10 的风险 #1（LASErMPNN 吞吐未知）**兑现了**：等变图网络 + 全原子，AR 口径逐 variant 一次
forward，全量约 16 h。处置：**优先保 `jm` 口径**（四模型唯一共同可比的那一列），
`ar` 口径排 4 个 slot、**跑到多少报多少并标注覆盖率**，不拿部分冒充全量。

**walltime 策略**：所有 run 改为 **`--dependency=afterany` 链式排多个 2 h slot**，
配合脚本幂等（输出已存在即跳过）⇒ 被 TIMEOUT 截断后自动续跑，不重算。

## 19. P2/P3/P4 的失败与修复（2026-09-12）

### 19.1 ADFLIP（两个 bug，第一个已修，第二个待修）

**(a) `Can't get attribute 'Config' on <module '__main__'>`** — `ADFLIP_v1.pt` 是用
`test/benchmark.py` 里**定义在 `__main__` 的 `Config` 类** pickle 的，unpickler 反序列化时
会去 `__main__` 找同名类。已在脚本里原样定义该类。

**(b) `size mismatch for model.layers.output.weight: [33,128] vs [20,128]`** — 我漏传了
`Zoidberg_GNN` 的三个构造参数 `number_ligand_atom` / `mpnn_cutoff` / **`output_dim`**，
其中 `output_dim` 决定输出词表大小（ckpt 是 33，默认退到 20）。
已逐字对齐 `benchmark.py:248-266`，并改为与其一致的**严格加载 `ckpt["model"]`**。
✅ 现在权重加载成功（日志 `权重严格加载成功; output_dim=33`）。

**(c) 残基集合不同（已修）** —— WT 校验全数失败：
```
4D5_HER2_1N8Z : ADFLIP 解析 1015 残基 vs BindingGYM 1041
5A12_Ang2_4ZFG: ADFLIP 解析  648 残基 vs BindingGYM  652
```
ADFLIP 同样丢掉 backbone 不完整的残基。**额外的麻烦**：`pdb2data` 只保留 ndarray 字段，
链**字母**丢失，只剩数值型 `chain_id`。
修法：**按链块切分 + 试所有排列取错配最少的指派**，再逐块 NW 比对建映射。
实测找到了非平凡指派 —— `5A12_Ang2` / `5A12_VEGF` 都是 `(1,2,0)`，即链顺序确实错位：
```
[map] 4D5_HER2_1N8Z : 解析 1015/1041, 链块指派 (0,1,2), 映射 1015 个位点
[map] 5A12_Ang2_4ZFG: 解析  648/652,  链块指派 (1,2,0), 映射  648 个位点
```

**(d) PyG 版本太新（已修）** —— 前三个修完后仍崩在：
```
ImportError: 'knn_graph' requires 'pyg-lib>=0.6.0'
  at model/zoidberg/zoidberg_GNN.py:172
```
实测：`PyG 2.8.0.post1`，`WITH_TORCH_CLUSTER = None`、`WITH_PYG_LIB = False`。
即 PyG 2.8 **去掉了 torch-cluster 后备路径**，`knn_graph` 改为强制要 `pyg-lib`。
而 `torch_cluster 1.6.3+pt21cu121` 装着且与 torch 2.1.0 匹配 —— 是 PyG 不用它。
根因：**ADFLIP 的 `requirements.txt` 里 `torch_geometric` 没有 pin 版本**，pip 抓了最新版。
修法：降到 `torch_geometric==2.6.1` ⇒ `WITH_TORCH_CLUSTER = True`，
并实测 `knn_graph(x, k=8)` 返回 `[2, 400]` 正常。
⇒ **这是「不 pin 版本」这一类坑的第二次出现**（第一次是 §15 的 `setuptools` / `pkg_resources`）。

### 19.2 LASErMPNN（残基集合 + 链顺序）

**(a) 残基数不等** —— `BH3_Bcl-xL_1PQ1: 解析 180 vs BindingGYM 229`；LASErMPNN 会丢掉
backbone 不完整的残基（它要算 chi 角）。已把等长 assert 换成 NW 比对映射。

**(b) 比对必须逐链做** —— 拼接后整体比对会在两条**同源**链之间错配。
`Z-domain_ZSPA-1_1LP1`（链 A = Z-SPA-1 affibody，链 B = Z-domain，二者同源）报
`[(85,'W','K'),(82,'F','Q'),(78,'K','N')]`。

**(c) 链顺序（已修）** —— 改逐链后错配位从 78/82/85 变成 **31/28/24，且残基对正好互换**
（`(31,'K','W')` vs 原 `(85,'W','K')`）⇒ **链 A/B 被对调**。
我用 `batch.chain_indices` 的唯一值顺序去对应 BindingGYM 的 `chain_id` 顺序，这个假设不成立。
修法：链字母改取自 LASErMPNN 自己的 **`data.residue_identifiers[i].chain_id`**（含真实字母）。
实测 `KRAS_SOS1_8BE4` 的解析顺序是 **S(440) 在前、R(165) 在后**，而 BindingGYM 的 `chain_id` 是 `RS`
—— 顺序确实无关。修后逐链比对：链 R 165/165、链 S 440/440，**零错配**。

### 19.3 StaB-ddG —— 突变编号约定（已修，旧结果已作废）

`stabddg/ppi_dataset.py:195` 用的是：
```python
mut_pos = int(mut[2:-1]) + mut_chain_offset - 1
assert mut_seq[mut_pos] == wt_aa
```
它把 SKEMPI 串里的数字当作 **`seq_chain_X` 内的 1-based 下标**，**不是 PDB 残基编号**。
我此前按残基编号生成，只有恰好对齐的 8 个 assay 通过，其余被它自己的 assert 拒掉。

修法：**不猜它的约定** —— 用 StaB-ddG 自己的 `parse_PDB` 取 `seq_chain_X`，
把 BindingGYM 的链序列 NW 比对上去，编号按比对到的下标生成，并断言比对后 WT 零错配。

⚠️ **此前那 8 个 assay 的结果用的是旧（错）编号，已整体作废重算** ——
移到 `/ibex/user/guoj0f/bindingGYM-zs-benchmark/stabddg_s2_OLD_wrong_numbering_1345`
（**不删除，留证**）。它们通过只是编号碰巧对齐，无法保证每个 variant 都对。

### 19.4 一个贯穿性的教训

到目前为止，**四个 repo 的 PDB parser 对「哪些残基算数」各有各的规矩**，没有一个与
BindingGYM 的序列约定天然对齐。已遇到五类：

| # | 现象 | 出处 |
|---|---|---|
| 1 | occupancy 全 0 ⇒ `select("occupancy>0")` 返回 None | LigandMPNN |
| 2 | 官方按残基编号**补洞**，vendor 只返回观测残基 | LigandMPNN vs BindingGYM |
| 3 | Kabat **insertion code** + 缺口并存 ⇒ 跨度 ≠ 序列长 | 4ZFF/4ZFG 抗体重链 |
| 4 | 丢掉 backbone 不完整的残基 | LASErMPNN / ADFLIP |
| 5 | 突变编号是**序列下标**而非残基编号 | StaB-ddG |

**通用对策**：一律「解析 → 逐链 NW 比对到 BindingGYM 参考序列 → 未映射位不参与打分」，
并对比对后的 WT 做零错配断言。**每一类都不会报错、数字也完全合理，只有硬校验能拦住。**

## 20. 当前状态快照（2026-09-12 16:56）

| run | 覆盖 | 状态 |
|---|---|---|
| `proteinmpnn_ar` / `proteinmpnn_mm` | 25/25 | ✅ 有效 |
| `ligandmpnn_ar_nosc` / `_sc` / `_mm_sc` | 25/25 | ✅ 有效 |
| `ligandmpnn_ar_scfix` / `_noscfix` | 25/25 | ✅ 有效，见 §17 |
| `lasermpnn_jm` / `lasermpnn_ar` | 0/25 | 🔄 **全部作废重跑**（§21） |
| `adflip_jm` | 8/25 | 🔄 四个 bug 全修完，**正在正常产出** |
| `stabddg_s2` | 7/25 | 🔄 编号已修（§19.3），重算中 |

**产物路径**
- per-variant 分数：`/ibex/user/guoj0f/bindingGYM-zs-benchmark/scores/{run_id}/{DMS_id}.csv`
- StaB-ddG：`/ibex/user/guoj0f/bindingGYM-zs-benchmark/stabddg_s2/{DMS_id}/output/`
- 作废留证：`scores/lasermpnn_*_STALE_prefix_*`、`stabddg_s2_OLD_wrong_numbering_*`
- 逐 assay 指标 + leaderboard（已回流进 repo）：`ibex-records/{project}/results/`

## 21. 🔴 一次差点写进结论的假结果 —— 幂等跳过 + 改代码 = 旧坏结果被静默保留

### 21.1 症状
`lasermpnn_jm` 跑出 25/25、job `COMPLETED`，聚合 Spearman = **0.361365**。
差点就作为「LASErMPNN 弱于 ProteinMPNN」写进结论。

### 21.2 是怎么被抓住的
汇报前算配对检验时，`Wilcoxon p = nan` —— 顺着 nan 查下去：

| assay | 现象 |
|---|---|
| `KRAS_SOS1_8BE4` | **19,425 个分数全部恰好为 0**，uniq=1，Spearman=NaN、AUC=0.5、MCC=0 |
| `KRAS_PICK3CG-RBD_1HE8` | 19,203 个里 **18,651 个为 0**（97%），Spearman −0.0136（ProteinMPNN 0.5055） |
| 对照 `proteinmpnn_mm` | 同两个 assay uniq=19,423 / 19,202，std≈2.8（正常） |

分数恒为 0 ⇒ 该 variant 的突变位点**一个都没映射上**（`diff` 为空）。

### 21.3 根因
**不是算法错。** 直接测比对函数：`KRAS_SOS1` 链 R 映射 165/165 零错配、链 S 440/440 零错配，
合计 605 —— 完全正常。

真正原因是**时间戳**：25 个 csv 里 **23 个写于 14:43 的链映射修复【之前】**，
只有 2 个 Z-domain 用了新代码。打分脚本是幂等的（输出已存在即跳过），
所以修复后的重跑**只补上了之前失败的 2 个，把 23 个坏的静默留下**，job 却报 `COMPLETED 25/25`。

### 21.4 持久性修复：code_stamp
仅仅重跑一次不够 —— 这个坑会在每次改脚本后复发。已给三个打分脚本加**代码版本戳**：

```python
def _code_stamp():                      # 脚本自身的 md5(前 12 位)
    return hashlib.md5(open(__file__,'rb').read()).hexdigest()[:12]

def _should_skip(out_csv, stamp):       # 跳过的判据从「文件存在」改为「戳一致」
    old = pd.read_csv(out_csv, nrows=1)
    return "code_stamp" in old.columns and str(old["code_stamp"][0]) == stamp
```
每个输出 csv 带 `code_stamp` 列；戳不一致就打印 `[restale]` 并重算。

### 21.5 波及范围核查（逐个查证，不假设）
- `lasermpnn_jm` / `lasermpnn_ar`：**全部作废**，移到 `scores/lasermpnn_*_STALE_prefix_*`（留证），已重跑。
- `stabddg_s2`：此前因编号问题已整体作废（§19.3），不叠加。
- **5 个主 MPNN run**：csv 全部早于 `score_bgym_mpnn.py` 的 05:09 修改。
  查证那次改动（加 `--designed_chains`）的 diff：`index` 分支那行
  `designed = [c for c in str(chain_ids)] if str(chain_ids) else sorted(set(letters))`
  与被替换的行**逐字相同** ⇒ 对这 5 个 run（都用 `index` 模式）**行为未变，结果有效**。
- `ligandmpnn_ar_scfix` / `_noscfix`：全部晚于该修改，有效。
  ⇒ **§17 的侧链结论不受影响。**

### 21.6 方法论教训
> **任何「跳过已有结果」的幂等机制，都必须把「产出它的代码版本」一起纳入判据。**
> 否则修 bug 之后的重跑会给出一个看起来完整、实则新旧混合的结果集，
> 而且 job 状态是 `COMPLETED` —— 没有任何信号提示你。

配套的第二条：**聚合层出现 `NaN` / `AUC=0.5` / `MCC=0` 时不要跳过，它们是退化预测的指纹。**
本次正是靠 `Wilcoxon p = nan` 才把这个问题揪出来。

## 22. ✅ ADFLIP 结果（25/25，2026-09-12 18:4x）

### 22.1 数据有效性（先验后报）
- 覆盖 **25/25**；`code_stamp` 取值集合 = `{'ece43ce4b11a'}`（**单一值** ⇒ 全部由同一版代码产出，
  不存在 §21 那种新旧混合）
- **零退化 assay**（无 `nunique<=1`、无 `std==0`）；逐 assay Spearman **无 NaN**

### 22.2 结果

| run | 口径 | Spearman | AUC | MCC | NDCG | AP |
|---|---|---|---|---|---|---|
| `adflip_jm` | B | **0.363033** | 0.671863 | 0.137245 | 0.702794 | 0.209901 |

配对对照（同口径 B、同 25 assay）：

```
ProteinMPNN 0.3899  vs  ADFLIP 0.3630   Δ=+0.0269  15/25  Wilcoxon p = 0.0275  ← 显著
LigandMPNN  0.3771  vs  ADFLIP 0.3630   Δ=+0.0140  11/25  Wilcoxon p = 0.9158  ← 不显著
```

### 22.3 科学结论

**ADFLIP 的 all-atom discrete flow matching，改造成 structure–seq compatibility likelihood 之后，
在 BindingGYM 的 protein–protein binding DMS 上并不优于 backbone-only 的 ProteinMPNN ——
反而显著更差（p = 0.0275）。** 与 LigandMPNN 则无显著差异。

这是一个**否定结论**，但它回答的正是立题时的核心问题（§1）：
「更强的结构表征（全原子、侧链、配体感知）能不能在 binding DMS 上超过 backbone-only？」
目前三条证据一致指向**不能**：
1. LigandMPNN ≈ ProteinMPNN（§12.4，p = 0.34 / 0.33，均不显著）
2. ADFLIP < ProteinMPNN（本节，p = 0.0275）
3. **唯一的例外是 §17**：LigandMPNN 的侧链 context 在 partner 链设为 fixed 后确有增益
   （+0.0082，p = 0.0158）—— 注意这里增益来自**partner 的侧链**，
   而不是「模型本身更全原子」。

⇒ 初步归纳：**在 binding DMS 上有用的不是「模型看得更细」，而是「模型看得到 partner」。**
这条推论还需 LASErMPNN 与 StaB-ddG 的结果来证实或否证。

### 22.4 逐 assay 的两极
ADFLIP 明显更好：`GB1_IgG-Fc_1FCC` +0.0702、`PSD95_CRIPT_1BE9` +0.0674。
ADFLIP 明显更差：`CD19_FMC63_7URV` −0.1686、`Z-domain_ZSPA-1_LL2` −0.1600、
`GB1_IgG-Fc_2016` −0.1465。
⚠️ 注意 `GB1_IgG-Fc` 的两个版本方向相反（+0.0702 vs −0.1465），同一复合物、同一 partition，
差别只在 DMS 数据版本 —— 提示逐 assay 差异有相当部分是 label 噪声而非模型能力。

## 23. StaB-ddG 的第三个坑：gap 字符不一致（已修）

新编号方案跑起来后，`4D5_HER2` / `5A12_*` 仍失败：
```
4D5_HER2_1N8Z 链C: 比对后 WT 不符,前 3 处 [(589, '-', 'X'), (588, '-', 'X'), ...]
```
**StaB-ddG 的 `parse_PDB` 用 `'-'` 表示按残基编号补洞的位置，而 BindingGYM 用 `'X'`。**
两者都是「未知/无坐标」，但我的校验只豁免了 `'X'` ⇒ 误判为错配。
已把 `{'X', '-'}` 一并视为未知位。

> 这是**同一个 BindingGYM 对齐问题的第六种表现形式**（§19.4 已列五种）。
> 补充进 §19.4 的表：**#6 gap 字符约定不同（`'-'` vs `'X'`）**。

## 24. ✅ LASErMPNN 结果 + 口径 B 的四模型完整对比（2026-09-12 20:4x）

### 24.1 有效性（先验后报）
`lasermpnn_jm` 重跑后：覆盖 **25/25**、`code_stamp` 单一值 `0ee856487461`、
**零退化 assay**、逐 assay Spearman **无 NaN**。
> 对比 §21 那次作废的版本：旧值 0.3614（新旧代码混合），**正确值 0.3757**，差 0.014。

### 24.2 口径 B —— 四个模型在同一口径、同 25 assay 上的完整对比

| 排名 | 模型 | Spearman | AUC | MCC | NDCG |
|---|---|---|---|---|---|
| 1 | **ProteinMPNN**（backbone-only） | **0.389911** | 0.685402 | 0.155203 | 0.719684 |
| 2 | LigandMPNN（+侧链 ctx） | 0.377051 | 0.682353 | 0.149206 | 0.718931 |
| 3 | LASErMPNN（全原子+rotamer） | 0.375715 | 0.676810 | 0.154669 | **0.726347** |
| 4 | ADFLIP（全原子 flow matching） | 0.363033 | 0.671863 | 0.137245 | 0.702794 |

**全部配对检验（Wilcoxon，同 25 assay）：**

| 对比 | Δ | 正号 | p | 显著? |
|---|---|---|---|---|
| ProteinMPNN vs **ADFLIP** | +0.0269 | 15/25 | **0.0275** | ✅ 显著 |
| ProteinMPNN vs LASErMPNN | +0.0142 | 11/25 | 0.6915 | ✗ |
| ProteinMPNN vs LigandMPNN | +0.0129 | 12/25 | 0.3254 | ✗ |
| LigandMPNN vs LASErMPNN | +0.0013 | 11/25 | 0.5602 | ✗ |
| LASErMPNN vs ADFLIP | +0.0127 | 11/25 | 0.5782 | ✗ |

### 24.3 🎯 阶段性主结论

**在 BindingGYM 的 protein–protein binding DMS 上，backbone-only 的 ProteinMPNN 排在第一，
没有任何一个「更全原子 / 更配体感知」的模型超过它。**
四个模型里唯一达到统计显著的差异是 **ProteinMPNN > ADFLIP（p = 0.0275）**，
其余两两之间都不显著 —— 即 LigandMPNN / LASErMPNN / ADFLIP 三者**与 ProteinMPNN 打平或更差**。

这正面回答了 §1 的立题问题，答案是**否定的**：
> 「更强的结构表征（侧链、全原子、配体感知）能不能在 binding DMS 上超过 backbone-only？」——**不能。**

**唯一有效的增量来自别处（§17）**：把 partner 链设为 fixed、让模型看到 **partner 的侧链**，
LigandMPNN 得到 +0.0082（15/21，p = 0.0158，五项指标全为正，且有 4 个零差异的内部对照）。

⇒ **归纳：在 binding DMS 上起作用的不是「模型把自己看得更细」，而是「模型看得到结合伙伴」。**
这条现在有两类证据支持（四模型横向打平 + 侧链轴纵向显著），待 StaB-ddG 补上第三类
（它是唯一显式建模 binding 的模型 —— 若它显著更好，则该归纳得到最强支持；若不然，需要重新解释）。

### 24.4 两个 caveat（必须随结论一起读）
1. **LASErMPNN 的 AR 口径尚未跑完**（5/25，全量约 16 h）。上表用的是口径 B。
   若 AR 口径给出不同排序，须以「两个口径都报」的形式呈现，不能只挑一个。
2. **逐 assay 差异有相当部分是 label 噪声**：`GB1_IgG-Fc` 的两个数据版本在 ADFLIP 上方向相反
   （+0.0702 vs −0.1465），同一复合物、同一 partition。**不要对单个 assay 的排序下强结论。**
