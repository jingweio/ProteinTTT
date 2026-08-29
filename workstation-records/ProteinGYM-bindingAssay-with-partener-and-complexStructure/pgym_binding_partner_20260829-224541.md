# ProteinGYM-bindingAssay-with-partener-and-complexStructure — experiment record

created 2026-08-29 22:45 · **status: DONE**（2026-08-29 23:41 完成）

## 1. Goal

ProteinMPNN 在 ProteinGym binding 这一栏几乎垫底（9 个 assay 上官方均值 **0.133**）。假设是：
ProteinGym 只给它 **AF2 单体结构**，partner 缺席，所以它只能建模 folding 而非 binding。

本实验补上 partner，测三个点：
1. 用 BindingGYM 给 ProteinGym 的 9 个 binding assay 补 partner sequence + WT complex structure（临时数据集）；
2. 在该数据集上跑 pretrained ProteinMPNN zero-shot；
3. 与 ProteinGym 官方 102 个模型在同样这 9 个 assay 上的指标对比。

## 2. Design & key decisions

### 2.1 三个对照点

| 条件 | 结构 | partner | 作用 |
|---|---|---|---|
| **official** | ProteinGym AF2 单体 | 无 | 官方公布值，直接读表 |
| **monomer** | BindingGYM 晶体结构，**删掉 partner 链** | 无 | 隔离「晶体 vs AF2」的影响 |
| **complex** | 同一晶体结构，完整 | **有**（fixed context） | 待测条件 |

**为什么必须有 monomer 这一档：** official 与 complex 之间同时变了两件事（结构来源、partner 有无）。
只有 `complex − monomer` 才能把增益归因到 partner。这是本实验唯一有效的因果对照。

### 2.2 打分协议 —— 对齐官方

严格照搬 ProteinGym 官方 baseline（`proteingym/baselines/protein_mpnn/compute_fitness.py`）：
- checkpoint **v_48_020**（`num_edges=48, noise_level=0.2`，md5 `698982b1bda2b0d42e26538e64c93fda`）
- `backbone_noise=0.0`，**每个变体一个随机 decoding order**（官方 `num_seq_per_target=1`），seed=1
- score = `-mean NLL`

两点偏离，都是等价变换或已验证无影响：
- **批处理**：官方逐变体循环，我们把变体堆到 batch 维。每行仍有各自的 decoding order，数学等价，只为吞吐。
- **主指标用 target 链而非全局**：官方 `pmpnn_ll` 是全部残基的平均 NLL；complex 里这会把 partner 的
  NLL 掺进来稀释信号。主指标取 **target 链限定**（`mask*chain_M*chain_M_pos`），全局值也一并输出
  (`mpnn_global_ll`) 便于核对。monomer 条件下两者恒等。

### 2.3 不改动原始数据

builder 的输入根全部走环境变量且**只读**，输出只写 `--out`。原 ProteinGym / BindingGYM 目录零写入。
临时数据集落在 `/data/guoj0f/ProteinGYM-bindingAssay-with-partener-and-complexStructure/dataset`。

### 2.4 已知会影响解读的点

- **CD19 只有 79.6% 变体可打分**（PDB chain C 仅解出 218/255 残基），样本量口径与其他 assay 不同。
- **DLG4 / YAP1 的 partner 只有 5aa / 10aa 肽**，"加 partner"信息量本就很小，预期增益弱。
- **B2L11 只有 170 个变体**，Spearman 方差大。
- M=1 的 decoding order 随机性：本地重跑 B2L11 两次得 0.7236 / 0.7157（Δ≈0.008），远小于待测效应量级。

## 3. Run config

- GPU: A100 80GB @ 10.67.24.41 ｜ 启动前 free **65,294 MiB** / util 0%（另有 3 个他人进程占 15.7 GB）
- env: `pgym-binding-partner-mpnn`（torch 2.4.1+cu121 / numpy 1.24.4 / pandas 2.0.3 / scipy 1.10.1）
- 启动脚本: `workstation-records/ProteinGYM-bindingAssay-with-partener-and-complexStructure/sh/pgym_binding_partner_20260829-224541.sh`
- 规模: 9 assay × 2 条件 × 711,273 变体 ≈ 1.42M 次前向
- 显存: batch 自适应（`B = 24000 // L`），OOM 自动折半重试

## 4. Change log

- 22:45 本地 smoke test 通过（B2L11 / DLG4，坐标校验 OK），提交 `f65f464`、同步（ALIGNED 84 files）、启动 **PID 4011059**。
- 23:41 `ALL_DONE`，elapsed **56 分钟**，stderr 全空，9×2 = 18 份 score 全部通过坐标校验。
- 23:45 聚合脚本修一处 bug：官方表的 `Number of Mutants` 是 assay 元数据不是模型，之前被算进 leaderboard 排到了第 1。已剔除后重算。

## 5. Results

### 5.1 逐 assay（Spearman，target 链限定分数）

| assay | n | official | monomer | complex | **partner 效应** | vs official |
|---|---|---|---|---|---|---|
| `B2L11_HUMAN_Dutta_2010_binding-Mcl-1` | 170 | −0.005 | 0.254 | **0.726** | **+0.472** | +0.731 |
| `DLG4_RAT_McLaughlin_2012` | 1,576 | 0.135 | 0.191 | 0.265 | +0.074 | +0.130 |
| `SPG1_STRSG_Olson_2014` | 536,962 | 0.147 | 0.331 | **0.394** | +0.062 | +0.247 |
| `SPG1_STRSG_Wu_2016` | 149,360 | 0.089 | 0.185 | 0.229 | +0.045 | +0.140 |
| `CD19_HUMAN_Klesmith_2019_FMC_singles` | 2,995 | 0.174 | 0.136 | 0.178 | +0.042 | +0.004 |
| `SPIKE_SARS2_Starr_2020_binding` | 3,669 | 0.160 | 0.390 | **0.415** | +0.026 | +0.255 |
| `YAP1_HUMAN_Araya_2012` | 10,075 | 0.197 | 0.147 | 0.171 | +0.024 | −0.026 |
| `Q53Z42_HUMAN_McShan_2019_binding-TAPBPR` | 3,344 | 0.192 | 0.260 | 0.258 | −0.002 | +0.066 |
| `ACE2_HUMAN_Chan_2020` | 2,185 | 0.106 | 0.081 | 0.077 | −0.004 | −0.029 |
| **均值** | | **0.1328** | **0.2193** | **0.3013** | **+0.0819** | **+0.1685** |

### 5.2 增益从哪来 —— 结构和 partner 各占一半

```
official 0.1328 ──(结构：AF2全长 → 晶体结合域)──▶ monomer 0.2193 ──(加 partner)──▶ complex 0.3013
                        +0.0866  6/9  p=0.055                        +0.0819  7/9  p=0.020
```

**两个效应量级几乎相同。** 换句话说，把 ProteinMPNN 在 binding 上的分数从 0.133 抬到 0.301，
**一半功劳不是 partner，而是换了结构** —— AF2 预测的全长蛋白换成实验晶体里那段紧凑的结合域。

### 5.3 partner 效应的稳健性（这是本实验最需要谨慎的一点）

| 子集 | n | 均值 | 正号 | Wilcoxon p |
|---|---|---|---|---|
| 全部 9 个 | 9 | **+0.0819** | 7/9 | **0.020** |
| 去掉 `B2L11`（n=170，方差大） | 8 | +0.0332 | 6/8 | 0.039 |
| 去掉 5–10aa 肽 partner（`DLG4`/`YAP1`） | 7 | +0.0913 | 5/7 | 0.078 |
| **两者都去掉** | 6 | **+0.0280** | 4/6 | **0.156** |

**"全部 9 个"那个 +0.082 几乎完全由 `B2L11` 一个 assay 撑起来**（它一个就 +0.472，而且只有 170 个变体）。
去掉它之后效应降到 **+0.033**；再去掉两个弱 partner 后 **+0.028、p=0.156，不显著**。

诚实的结论：**方向是一致的（7/9 为正），但量级小，且在 n=9 的规模下经不起去掉单个离群点的检验。**

### 5.4 在这 9 个 assay 上的 leaderboard 排名（97 个官方模型 + 我们 2 个）

| | mean ρ | rank |
|---|---|---|
| ProSST (K=4096) | 0.4862 | 1 |
| VenusREM | 0.4444 | 4 |
| ESM-IF1 | 0.3598 | 14 |
| **[ours] MPNN_complex** | **0.3013** | **40** |
| **[ours] MPNN_monomer** | **0.2193** | **73** |
| ProteinMPNN（官方） | 0.1328 | **95** |

补上 partner + complex structure 让 ProteinMPNN 从 **95 名升到 40 名** —— 幅度很大，但**仍然进不了第一梯队**
（ProSST 0.486、ESM-IF1 0.360 都还在前面）。所以「给 partner 就能大幅提升」这个猜想
**方向对、幅度被高估了**：它把 ProteinMPNN 从倒数拉到中游，没有让它变得有竞争力。

### 5.5 caveats

1. **子集口径不一致（只影响 "vs official" 一列）。** 我们的分数只算结构内可打分的变体
   （`CD19` 79.6%、`SPIKE` 96.5%、`ACE2` 98.3%，其余 100%），官方是全部变体。
   `monomer` vs `complex` 是同一子集上的配对比较，**不受影响**。
2. **n=9 太小。** 所有 Wilcoxon 都在 9 个甚至 6 个样本上做的，p 值只能当参考。
3. **decoding order 随机性。** M=1，本地重跑 `B2L11` 得 0.7157/0.7236/0.7256（Δ≈0.01），
   远小于效应量级，但对 `Q53Z42`(−0.002)、`ACE2`(−0.004) 这种接近 0 的结果，符号不可信。
4. **`B2L11` 的 +0.472 要单独看。** 它的 target 是 23aa 的 BH3 肽 —— 单体几乎没有结构信息，
   加上 Mcl-1 后才第一次有了"结构"。这是极端情形，不能外推到一般的 binding assay。

### 5.6 输出位置

| | |
|---|---|
| 临时数据集 | `10.67.24.41:/data/guoj0f/ProteinGYM-bindingAssay-with-partener-and-complexStructure/dataset`（9 assay，各含 `complex.pdb` / `sequences.fasta` / `variants.csv` / `meta.json`） |
| per-variant 分数 | 同上 `…/scores/`（18 个 csv，含 `mpnn_design_ll` / `mpnn_global_ll`） |
| 汇总 | `results/per_assay.csv`、`results/leaderboard.csv` |
| 日志 | `pgym_binding_partner_20260829-224541.out`（gitignore，留在两侧） |

原 ProteinGym / BindingGYM 数据集**零写入**，已核对。
