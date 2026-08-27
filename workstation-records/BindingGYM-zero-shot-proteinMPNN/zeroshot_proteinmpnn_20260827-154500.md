# BindingGYM-zero-shot-proteinMPNN — experiment record
（created 2026-08-27 15:45; status: **DONE** —— S1–S5 全部完成）

在 workstation A100 上复现 **BindingGYM 官方 zero-shot pretrained ProteinMPNN** 的评测结果。

---

## 1. Goal / hypothesis

**要复现的数字**（论文 Table 2，`refs/ProteinMPNN_zero_shot_metric.csv` 的 25-assay 未加权均值）：

| 指标 | 官方值 | 论文 Table 2 |
|---|---|---|
| **Spearman** | **0.396950** | 0.40 |
| AUC | 0.687909 | 0.69 |
| MCC | 0.153658 | 0.15 |
| NDCG | 0.722340 | 0.72 |
| AP | 0.218779 | 0.22 |
| UnbiasHit@10 | 0.296 | 0.30 |

论文 Table 6/11 给的官方误差棒：**Spearman 0.40 ± 0.03**（1000 次 bootstrap 重采样 assay 集合）。

### 🎯 本次复现的一个**可证伪预测**

上一次复现（Ibex, a100, 1:09:42，存于 `refs/prior_run_M1_per_assay.csv`）得到 **0.3840**，
比官方低 **0.0129**，当时归因于"随机解码顺序"。**我核出了更具体的原因**：

那次的 sbatch 脚本**没有传 `--num_seq_per_target`**，而 `compute_fitness_multi_pdb.py:295` 的
argparse 默认值是 **1**；官方 launcher（`modelzoo/proteinmpnn/run.py:57,76`）传的是 **5**。
⇒ **上次跑的是 M=1，官方是 M=5。**

支持这个解释的两条证据：
1. 我在 `BH3_Mcl-1` 上实测过 M 对**期望值**的影响（不只是方差）：
   M=1 **+0.6667** ± 0.0366 → M=5 **+0.6817** ± 0.0136 → M=20 **+0.6919** ± 0.0080
   —— **M 越小，期望值系统性偏低**（朝确定性极限单调上升）。
2. 上次的逐 assay 偏差 **17/25 为负**（`refs/prior_run_M1_per_assay.csv` 复算），
   方向与 M=1 偏低一致。

> **⇒ 预测：本次用官方 M=5，25-assay 均值应显著高于 0.3840、接近 0.3970。**
> 这条预测是本次实验的主要科学内容 —— 它把"复现"从"跑一遍看看"变成一个有方向的检验。

---

## 2. Design & decision points

### 2.1 打分协议（逐条对齐官方 launcher，harness 启动时 assert）

官方调用（`modelzoo/proteinmpnn/run.py:47-58`，逐字）：

```
compute_fitness_multi_pdb.py
  --model_location {checkpoint_folder}/v_48_020.pt
  --dms_index {idx}  --dms_mapping input/BindingGYM.csv
  --dms_input input/Binding_substitutions_DMS  --dms_output output
  --structure_folder input/structures
  --batch_size 8  --num_seq_per_target 5
```

| 项 | 值 | 依据 / 注意 |
|---|---|---|
| checkpoint | **`v_48_020.pt`（vanilla）** md5 `91d54c97a68b` | `run.py:25`。⛔ **不是** `StaB-ddG/model_ckpts/proteinmpnn.pt`（那是 **soluble** 变体，其 README:50 自陈；118/118 tensor 不同，cosine 0.0090） |
| **M（解码顺序数）** | **`--num_seq_per_target 5`** | 就是 `NUM_BATCHES`；5 份结构 clone 批在一次 forward 里，`randn_1` 形状 [5,L]，最后对 5 行取均值 |
| `--batch_size` | 传 8（**死参数**） | `BATCH_COPIES = args.batch_size` 全文件只出现一次、从不使用。为逐字对齐官方仍然传 8 |
| `--backbone_noise` | **0.00**（default，官方不传） | `:294` default 0.00；ckpt 里的 `noise_level=0.2` 只被打印。assert `model.features.augment_eps == 0.0` |
| 构造参数 | `hidden_dim=128, layers=3/3, k_neighbors=ck['num_edges']=48, ca_only=False, num_letters=21` | `:165` |
| 分数列 | **`global_score`** = `-1 * mean_over_M(Σ_i mask_i · NLL_i(mutant))` | 无 WT 项、无长度归一化（`_scores` 里 `/ torch.sum(mask,-1)` 被注释掉）。本 benchmark 上 `design_score ≡ global_score`（`chain_id` 覆盖全链、`fixed_chain_list` 空，25/25 已验证） |
| 结构输入 | WT 复合物，**只读 N/CA/C/O** | `parse_PDB` 的 `sidechain_atoms = ['N','CA','C','O']`（变量名有误导）。PDB 里 74.6% 的原子是侧链，**全部被丢弃** |
| 解码顺序缓存 | `randn_1_dic[POI]` —— **按 assay 缓存、该 assay 全部 variant 共用** | 刻意的方差控制：assay 内所有 variant 在同一因子分解下打分。⇒ 解码噪声是**共模扰动，不随 n 衰减** |

### 2.2 指标口径

用 `refs/bindinggym_metrics.py::bindinggym_metrics_one_assay` ——
**逐字移植自官方 `calc_metric.ipynb::calc_zero_shot_metric(top_test=False)`**（已在姊妹项目验证过）。

| 指标 | 定义 |
|---|---|
| Spearman | `df[label].rank().corr(df[pred].rank())` |
| AUC | `roc_auc_score(label > P90(label), pred)` —— 正类阈值是 **label 自己的 90 分位**，不是固定值 |
| MCC | 两边都二值化（pred 用**预测**的 90 分位） |
| NDCG | `ndcg_score(label.rank(), pred, k = n // 10)` |
| AP | 同 AUC 的正类定义 |

聚合：**先按 assay 算、再对 25 个 assay 取未加权均值**。

### 2.3 KEY DECISIONS

| 决策 | 选择 | 理由 |
|---|---|---|
| **M** | **5（官方）** | 唯一与 `refs/` 可比的设定。这也是本次要检验的那条预测 |
| **seed** | **显式传 5 个 seed（0–4），报 mean ± [min,max]** | 官方**从不传 `--seed`**，而 `if args.seed:` + `default=0` ⇒ 0 是 falsy ⇒ **每次运行都重随机**。所以 `refs/` 是**一次未设种的单抽**，逐 assay 逐位复现在原理上不可能。跑 5 个 seed 才能把"参考值落在我们分布内"变成可判定的判据 |
| **判据** | per-assay：参考值落入我们 5 seed 的 [min,max]，或 \|mean_ours − ref\| ≤ 2σ_combined<br>25-assay 均值：与 0.396950 比，容差由实测 σ 定 | 见 §2.4。**不用**"逐位相同"这种不可能达到的判据 |
| **env** | **专属 `bindinggym-zs-mpnn`**，不复用 `proteingym-ttt` | §10「一个任务一个专属 env，绝不 activate 属于另一个任务的 env」 |
| **版本 pin** | `numpy 1.24.4 / scipy 1.10.1 / sklearn 1.3.2 / pandas 2.0.3`（按 `BindingGYM.yml`）；**torch 用 2.4.1+cu121 而非官方的 1.13.1+cu117** | 前四个决定 metric 取值，必须对齐（§3-0）。torch 换新是为 A100 速度，属**已记录的 deviation**，靠"复现发布的 per-assay 值"这个指纹 guard 兜底 |
| **不做** | 不改 `randn_1` 缓存、不加 encoder 缓存、不做 WT 相减、不做热力学循环 | 全部会偏离官方口径。加速手段只能在 gate 通过之后另开 arm |

### 2.4 判据所需的 σ

本项目要自己测（不能沿用 ProteinGym 那边的数字）：

- 我在 `BH3_Mcl-1` 上用**官方代码路径 + vanilla ckpt + backbone_noise=0.00** 实测过
  M=5 的 σ ≈ **0.019–0.021**（两组独立 6-seed），M=20 的 σ ≈ 0.0093。
- σ_combined = √(σ_ours² + σ_ref²) ≈ **0.0225**（参考侧也是单抽，同样带 σ）⇒ per-assay 容差 ≈ **0.045**
- 25-assay 均值侧：**σ 不随 n 衰减**（共模），但会随 assay 数平均 ⇒ 用**实测的** per-assay σ 向量
  算出均值的 SE，再定容差。**本次的 S2 就是产出这个 σ 向量。**

---

## 3. Run config

- **GPU**：A100 80GB ×1 @ `10.67.24.41`（无调度器，`setsid nohup` 直接跑）
- **env**：`bindinggym-zs-mpnn`（`/data/guoj0f/miniconda3/envs/`）
- **代码**：`/data/guoj0f/share/BindingGYM/baselines/protein_mpnn/`（官方打分脚本 + utils，未改动）
- **数据**：`/data/guoj0f/share/BindingGYM/input/`（25 assay CSV + 22 结构 + 索引，md5 已核）
- **权重**：`/data/guoj0f/share/BindingGYM/training/cache/v_48_020.pt`（md5 `91d54c97a68b`，两侧一致）
- **产出**：per-assay 分数 CSV 写 `/data/guoj0f/BindingGYM-zero-shot-proteinMPNN/scores/seed{k}/`；
  只有指标汇总与本 md 回流进 git
- **幂等**：每 assay 一个输出 CSV，已存在则跳过 ⇒ 超时/中断直接重跑即续

### 分阶段与预算

| 阶段 | 内容 | 估时 |
|---|---|---|
| **S1** | 计时探针：3 个不同长度的 assay，M=1 与 M=5 各一次 | **< 15 min** |
| **S2** | **主跑**：25 assay × M=5 × seed 0 | 待 S1 定（见下） |
| **S3** | σ 测量：**10 个最便宜的 assay** × M=5 × seed 1–4 | 待 S1 定 |
| **S4** | 聚合 + 与 `refs/` 逐 assay 比对 + 与 `prior_run_M1` 对照 | CPU，分钟级 |

**预算锚点**：上次 M=1 全量 25 assay 在 a100 上 **1:09:42**（376,446 行，≈90 行/s）。
M=5 是**一次 batch=5 的 forward**（不是 5 次 forward），我在本 A100 上实测过同脚本的比值：
`L=173` M=5/M=20 → 35.6s/50.6s，`L=1107` → 61.2s/147.6s ⇒ **M=5 相对 M=1 大约 1.5–2.5×**。
⇒ **S2 估 1.7–3 h**；S3（10 个最便宜的 assay，实测占全量约 6% 的 n·L）× 4 seed 估 **0.5–1 h**。
**合计承诺 ≈ 4 GPU-h。** S1 跑完会把这两个数换成实测值再启动 S2。

---

## 4. Change log

- 2026-08-27 15:45 — 建 project、写 plan。就地化 `refs/`（官方参考值、已验证的 metric 移植、
  上次 M=1 的逐 assay 结果、assay 索引）。建专属 env `bindinggym-zs-mpnn`（按 `BindingGYM.yml`
  pin numpy/scipy/sklearn/pandas）。**尚未启动任何 GPU 任务。**

---

## 5. Results

### 5.1 S1 — 计时探针（A100 实测）

| assay | n | L_total | M=1 | M=5 | 比值 | M=5 s/variant |
|---|---|---|---|---|---|---|
| Z-domain_ZpA963_HL2_2M5A | 600 | 116 | 11 s | 13 s | 1.18× | 0.0217 |
| HLA-A2_TAPBPR_5WER | 3,344 | 644 | 54 s | 160 s | 2.96× | 0.0478 |
| 4D5_HER2_1N8Z | 2,080 | 1,041 | 48 s | 168 s | 3.50× | 0.0808 |

吞吐 ≈ **13,000 (n·L)/s**；M=5 相对 M=1 的开销随 L 增长（短序列被进程启动开销 ~11 s 主导）。
`ckpt md5` assert 通过（`91d54c97a68bf551114f8c74c785e90f` = 官方 vanilla v_48_020）。
Σn·L = 135,744,347 ⇒ S2 预测 2.9 h。

### 5.2 S2 — 全量 25 assay，M=5，seed 1 ✅

| 指标 | **本次 (M=5, seed 1)** | 官方发布 | Δ |
|---|---|---|---|
| **Spearman** | **0.391356** | 0.396950 | **−0.005594** |
| AUC | 0.687099 | 0.687947 | −0.000848 |
| MCC | 0.157233 | 0.153505 | +0.003728 |
| NDCG | 0.720946 | 0.722055 | −0.001109 |
| AP | 0.220265 | 0.219163 | +0.001102 |

**5 个指标里 4 个落在 ±0.004 以内**，Spearman 是残差最大的一项。

逐 assay：mean \|Δ\| = **0.0211**，max 0.1063，>0.05 的 **3/25**，Δ<0 的 **13/25**。
偏差最大：`Z-domain_ZSPA-1_LL1_1LP1` 0.2003 vs 0.3066（Δ=−0.1063）——
该 assay 是全库最难的一个（官方**微调后**也只有 0.0121），上次 M=1 复现在它上面是 0.0888。

### 5.3 那条预测：**被否证**

| | 25-assay mean Spearman | Δ vs 官方 |
|---|---|---|
| 官方发布（M=5，单次未设种） | 0.396950 | — |
| 上次复现（**M=1**，Ibex a100，未设种） | 0.384029 | −0.012921 |
| **本次复现（M=5，seed 1）** | **0.391356** | **−0.005594** |

聚合层面看似"弥合了 57%"，但那是**两次不受控的运行之间的比较**（不同机器、不同 env、
不同且未设种的 seed）。**受控的配对对照否证了它** —— 同 seed=1、同 assay、唯一差别是 M：

| assay | M=1 | M=5 | Δ |
|---|---|---|---|
| 4D5_HER2_1N8Z | 0.3261 | 0.3267 | +0.0006 |
| 5A12_Ang2_4ZFG | 0.1027 | 0.1093 | +0.0067 |
| BH3_Bcl-xL_1PQ1 | 0.6756 | 0.6533 | **−0.0223** |
| BH3_Mcl-1_3KZ0 | 0.6814 | 0.6514 | **−0.0300** |
| CD19_FMC63_7URV | 0.6119 | 0.6032 | −0.0087 |
| HLA-A2_TAPBPR_5WER | 0.4154 | 0.4116 | −0.0038 |
| PSD95_CRIPT_1BE9 | 0.3317 | 0.3677 | **+0.0360** |
| PSD95_Tm2F_1BE9 | 0.1789 | 0.1943 | +0.0154 |
| Z-ZSPA-1_LL2_1LP1 | 0.2387 | 0.2749 | **+0.0362** |
| Z-ZpA963_HL1_2M5A | 0.2007 | 0.1839 | −0.0168 |
| Z-ZpA963_HL2_2M5A | 0.3751 | 0.3774 | +0.0023 |
| hYAP65_1JMQ | 0.1256 | 0.0886 | **−0.0370** |
| **均值（12 assay）** | 0.3553 | 0.3535 | **−0.0018** |

```
配对差: mean −0.0018   sd 0.0233   正号 6/12
Wilcoxon signed-rank p = 0.733      paired t-test p = 0.797
```

⇒ **M=5 在本 benchmark 上并不系统性优于 M=1。** 上次那 −0.0129 的缺口就是 seed 波动，
**SOP 原本的归因（"残差来源是随机解码顺序"）是对的，我那个"更具体的解释"是错的。**

⚠️ 注意 BH3 上的方向甚至**反了**（M=1 的 0.6814 > M=5 的 0.6514），与我早前在同一 assay 上测到的
M=1 0.6667 < M=5 0.6817 相反 —— 那次用的是 **soluble 权重 + backbone_noise 0.1**，条件不同。
**M 的方向本身在不同条件下都不稳定**，配对差的 sd（0.0233）与 per-assay seed σ（0.0184）同阶。

### 5.4 S3 — seed 方差（5 seeds × 10 assay）

| assay | sd | range |
|---|---|---|
| 5A12_Ang2_4ZFG | 0.0044 | 0.0103 |
| BH3_Bcl-xL_1PQ1 | 0.0092 | 0.0208 |
| BH3_Mcl-1_3KZ0 | 0.0208 | 0.0557 |
| CD19_FMC63_7URV | 0.0171 | 0.0428 |
| PSD95_CRIPT_1BE9 | 0.0103 | 0.0280 |
| PSD95_Tm2F_1BE9 | 0.0061 | 0.0162 |
| Z-ZSPA-1_LL2_1LP1 | 0.0197 | 0.0465 |
| **Z-ZpA963_HL1_2M5A** | **0.0581** | **0.1380**（0.0646 → 0.2026） |
| Z-ZpA963_HL2_2M5A | 0.0303 | 0.0680 |
| hYAP65_1JMQ | 0.0230 | 0.0570 |

**per-assay sd：median 0.0184，max 0.0581。**
该中位数与我早前在 BH3 上独立测到的 M=5 σ ≈ 0.019–0.021 吻合（此处 BH3_Mcl-1 = 0.0208），
两次独立测量互证。

⚠️ **`Z-ZpA963_HL1` 极不稳定**：单靠解码顺序就能让 ρ 在 0.065–0.203 之间变化 3 倍。
任何基于该 assay 的结论都不可信。

### 5.5 复现是否成功：**是**

按 median per-assay σ = 0.0184 推 25-assay 均值 SE ≈ 0.0184/√25 = **0.0037**；
官方参考值同样是**单次抽样**、带同样的 σ ⇒ σ_combined ≈ **0.0052**。

⇒ **残差 −0.0056 ≈ 1.1 σ_combined**，且 5 个指标里 4 个落在 ±0.004 内。
**逐 assay 逐位复现在原理上不可能**（`if args.seed:` 使官方每次运行重随机），
本次结果与"参考值是一次未设种单抽"完全一致，**不构成 harness 有问题的证据**。

### 5.6 实际成本

| 阶段 | 内容 | 实测 |
|---|---|---|
| S1 | 计时探针（3 assay × M=1,5） | 7.6 min |
| S2 | 全量 25 assay × M=5 × seed 1 | **≈2.5 h**（预测 2.9 h） |
| S3 | 10 assay × seed 2–5 | ≈1.4 h |
| S5 | 10 assay × M=1 × seed 1（配对臂） | ≈9 min |
| **合计** | | **≈4.0 GPU-h** |

产出：`/data/guoj0f/BindingGYM-zero-shot-proteinMPNN/scores/{seed1_M5(25), seed1_M1(12), seed2-5_M5(10 each)}`，
聚合表 `per_assay_all_runs.csv`。
