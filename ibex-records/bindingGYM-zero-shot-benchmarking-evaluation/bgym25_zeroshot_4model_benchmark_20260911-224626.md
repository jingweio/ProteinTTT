# bindingGYM-zero-shot-benchmarking-evaluation — experiment record

> task: `bgym25_zeroshot_4model_benchmark` · created 2026-09-11 22:46 · **status: PLANNED**
> 锚点实验：[`workstation-records/BindingGYM-zero-shot-proteinMPNN/zeroshot_proteinmpnn_20260827-154500.md`](../../workstation-records/BindingGYM-zero-shot-proteinMPNN/zeroshot_proteinmpnn_20260827-154500.md)

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

> 今天实测 Ibex：240 张 a100 仅 5 张 idle，但**等待时间几乎只取决于申请的 walltime** ——
> 1×a100 且 `--time ≤8h` 基本立刻起；22h 的要等 ~2.5 天。故一律切成 ≤8h 的块。

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
  据 LigandMPNN 源码修正口径 B 的实现为 joint_masked_score（1 pass/位点组合，而非 single_aa_score 的 L 次/次调用）。

## 12. Results

*(待填 —— 全部 run 完成后补 headline 表)*

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
