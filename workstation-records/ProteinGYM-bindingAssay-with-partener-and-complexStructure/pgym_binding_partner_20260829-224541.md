# ProteinGYM-bindingAssay-with-partener-and-complexStructure — experiment record

created 2026-08-29 22:45 · **status: RUNNING**

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

- 20260829-224541: 本地 smoke test 通过（B2L11 / DLG4，坐标校验 OK），提交并同步，启动。

## 5. Results

（待填）
