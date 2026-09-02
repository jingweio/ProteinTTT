# ProteinMPNN 的 zero-shot 打分在 binding-site / 非 binding-site 上的分布差异
（project `binding-sites-analysis-pred` · 创建 2026-09-02 · status: **DONE**）

## 1. 目标 / 假设

**待检验的假设（用户提出）：** 真实 DMS_score 上，「碰 binding site」与「不碰」两组存在分布差异
（已由 `../binding-sites-analysis/BindingGYM_binding_sites_20260829.md` §5 建立）。
而预训练 ProteinMPNN **只建模折叠过程中 sequence–structure 的 compatibility**，
对「这个位点是不是界面」缺乏感知，因此它在这两组上的**相对分布差异应当与 ground-truth DMS 不一样**。
若成立 ⇒ 为 binding-site-aware 的 ProteinMPNN TTT（complexTTT）提供动机。

## 2. 是否需要重跑打分 —— **不需要，已有且已验证**

per-variant 分数**早已存在**（上一条分支的 4.0 GPU-h 产出），所以**没有启动新的 GPU 任务**：

- workstation `/data/guoj0f/BindingGYM-zero-shot-proteinMPNN/scores/seed1_M5/` —— **25/25 assay 齐全**，329 MB
- 本地此前只有**逐 assay 汇总**（`per_assay_all_runs.csv`），**没有** per-variant ⇒ 这次把 per-variant 拉下来了
- 其余 seed（2–5）只有 10 个 assay，是当时测噪声地板用的子集，不完整

**三道 gate 全部通过**（`p1_merge_scores.py`，任一不过就 assert 失败）：

| gate | 检验 | 结果 |
|---|---|---|
| (a) | 分数 csv 行数 == 官方 DMS csv 行数 | **25/25** |
| (b) | 按同样规则去掉 WT 行后，`DMS_score` 与 `variant_labels.parquet` **逐元素相等** ⇒ 位置对齐是**证明**的不是假设的 | **25/25**，376,424 行 |
| (c) | 从这批分数重算逐 assay Spearman，对比记录中的 seed1_M5 | max \|Δρ\| = **6.9e-17**，均值 **0.391356** vs 记录 **0.391356** |

另：`design_score ≡ global_score` 25/25（与既往结论一致）；官方口径用 `global_score` 且**不取负**，
即该列已定向为「越高越像天然序列」，与 DMS 同向（25/25 Spearman 为正）。

## 3. 两组分布（ProteinMPNN 打分）

![MPNN distribution by interface](fig_mpnn_distribution_by_interface.png)

*与 DMS 图**完全相同的布局与面板顺序**（按 δ_DMS 升序），因此可以逐格对照。
红=碰 binding site、蓝=不碰，竖线=各组中位数；标题下第二行给出 δ_DMS 作参照，
**红色 "SIGN FLIP" = MPNN 的 δ 与 DMS 的 δ 符号相反**。*

### Table P1  逐 assay：DMS 与 ProteinMPNN 的界面对比

| assay | n 碰 / 不碰 | δ **DMS** | δ **MPNN** | 符号 | OVL DMS | OVL MPNN | η² DMS | η² MPNN | ρ(MPNN, DMS) |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|
| CD19_FMC63_Fitness_7URV | 136 / 3,749 | -0.700 | **-0.465** | 一致 | 0.397 | 0.538 | 0.057 | 0.014 | 0.603 |
| PSD95_CRIPT_1BE9 | 285 / 1,291 | -0.573 | **-0.226** | 一致 | 0.512 | 0.730 | 0.158 | 0.022 | 0.367 |
| ACE2_SARS2-RBD_enrich_6M17 | 380 / 1,805 | -0.554 | **-0.211** | 一致 | 0.400 | 0.780 | 0.120 | 0.019 | 0.276 |
| KRAS_RALGDS-RBD_norfitness_1LFD | 5,534 / 14,806 | -0.467 | **-0.158** | 一致 | 0.617 | 0.860 | 0.128 | 0.013 | 0.587 |
| SARS2-RBD_ACE2_deltaKd_6M0J | 6,763 / 15,108 | -0.339 | **-0.198** | 一致 | 0.661 | 0.846 | 0.051 | 0.025 | 0.697 |
| KRAS_RAF1_norfitness_6VJJ | 4,939 / 7,737 | -0.317 | **+0.107** | **翻转** | 0.735 | 0.881 | 0.063 | 0.010 | 0.487 |
| KRAS_RAF1-RBD_norfitness_6VJJ | 5,953 / 17,208 | -0.312 | **+0.193** | **翻转** | 0.698 | 0.853 | 0.065 | 0.023 | 0.441 |
| KRAS_DARPinK27_norfitness_5O2S | 8,974 / 10,558 | -0.254 | **+0.197** | **翻转** | 0.780 | 0.841 | 0.051 | 0.034 | 0.403 |
| PSD95_Tm2F_1BE9 | 304 / 1,272 | -0.254 | **-0.047** | 一致 | 0.542 | 0.777 | 0.035 | 0.000 | 0.194 |
| GB1_IgG-Fc_fitness_1FCC | 58,502 / 34,388 | -0.232 | **+0.046** | **翻转** | 0.636 | 0.916 | 0.063 | 0.001 | 0.500 |
| KRAS_SOS1_norfitness_8BE4 | 9,949 / 9,475 | -0.217 | **+0.027** | **翻转** | 0.802 | 0.886 | 0.032 | 0.000 | 0.309 |
| HLA-A2_TAPBPR_meanscore_5WER | 412 / 2,932 | -0.215 | **-0.059** | 一致 | 0.725 | 0.791 | 0.018 | 0.002 | 0.412 |
| KRAS_PICK3CG-RBD_norfitness_1HE8 | 3,903 / 15,299 | -0.204 | **+0.178** | **翻转** | 0.697 | 0.859 | 0.019 | 0.016 | 0.502 |
| CXCR4_CXCL12_enrich_8U4O | 703 / 4,881 | -0.193 | **+0.105** | **翻转** | 0.826 | 0.841 | 0.011 | 0.002 | 0.201 |
| 5A12_Ang2_fitness_4ZFG | 819 / 124 | -0.036 | **-0.240** | 一致 | 0.766 | 0.681 | 0.000 | 0.020 | 0.107 |
| hYAP65_peptide_FunctioncalScore_1JMQ | 9,692 / 8,714 | -0.030 | **+0.041** | **翻转** | 0.952 | 0.879 | 0.000 | 0.000 | 0.089 |

**汇总（16 个可检验 assay，判据与 DMS 侧完全一致：两组各 ≥ 30）：**

| | δ<0 的 assay 数 | q<0.05 | \|δ\| 中位 | OVL 中位 | η² 中位 |
|---|---:|---:|---:|---:|---:|
| **DMS（measured）** | **16/16** | 15/16 | 0.254 | 0.697 | 0.051 |
| **ProteinMPNN（predicted）** | **8/16** | 14/16 | **0.168** | **0.844** | **0.013** |

## 4. 与 DMS 分布图的差异 —— 假设成立，而且比"更弱"更强

![DMS vs MPNN contrast](fig_dms_vs_mpnn_contrast.png)

1. **方向一致性崩了。** DMS 上 δ<0 是 **16/16**（碰界面一律更差）；MPNN 上只有 **8/16**，
   另 **8 个符号翻转** —— MPNN 认为碰界面的突变**更被容忍**。
2. **KRAS 家族最刺眼：6 个 assay 里 5 个符号翻转**（RAF1 +0.107、RAF1-RBD +0.193、
   DARPinK27 +0.197、PI3KCG +0.178、SOS1 +0.027），而 DMS 给的是 −0.20 ~ −0.47。
   这与既往「ProteinMPNN 在 BindingGYM 上基本 partner-blind」的观察同向且互相独立。
3. **效应量整体缩水约一半**：\|δ\| 中位 0.254 → 0.168，**14/16 个 assay 被衰减**，中位比值 **0.470**。
4. **重叠度显著上升**：OVL 中位 0.697 → **0.844**；η² 中位 0.051 → **0.013**（差 ~4×）。
5. **逐 assay 的界面敏感度互不相关**：ρ(δ_DMS, δ_MPNN) = **+0.376，p = 0.15，不显著**。
   即 MPNN 不只是"幅度小"，而是**排序都对不上** —— 它在哪些 assay 上对界面敏感，与真实情况无关。
6. **分布形状也不同**：DMS 普遍**双峰**（死变体堆在下限、功能变体在上限），
   MPNN 是平滑**单峰**近高斯 —— 它没有"死/活"这个模式的概念。

⇒ **假设成立。** 真实测量里存在的界面对比，在 ProteinMPNN 的预测里被削弱、被混淆、并在半数 assay 上被反转。

## 5. 机制：我试图用 burial 解释，**被自己的数据否掉了**

自然的机制猜想是：MPNN 的置信度跟随局部堆积密度，而这些界面（如 KRAS switch I）是**溶剂暴露的表面 loop**，
链内 burial 低 ⇒ MPNN 在那里本来就不确定 ⇒ 突变惩罚小 ⇒ δ 变正。用**删掉 partner 后的链内 Cβ 邻居数**（10 Å）检验：

### Table P2burial 检验（partner 已删除的链内 Cβ 邻居数，10 Å）

| assay | 库位点 site / non-site | burial site | burial non-site | Δburial | δ MPNN | δ MPNN (burial 分层后) | δ DMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| KRAS_RAF1_norfitness_6VJJ | 15 / 48 | 13.20 | 19.33 | -6.13 | +0.107 | +0.104 | -0.317 |
| KRAS_PICK3CG-RBD_norfitness_1HE8 | 16 / 148 | 12.62 | 18.47 | -5.84 | +0.178 | +0.176 | -0.204 |
| KRAS_DARPinK27_norfitness_5O2S | 26 / 137 | 12.35 | 17.86 | -5.52 | +0.197 | +0.187 | -0.254 |
| KRAS_RAF1-RBD_norfitness_6VJJ | 17 / 149 | 13.71 | 18.52 | -4.81 | +0.193 | +0.188 | -0.312 |
| KRAS_SOS1_norfitness_8BE4 | 35 / 128 | 13.17 | 17.52 | -4.34 | +0.027 | +0.020 | -0.217 |
| KRAS_RALGDS-RBD_norfitness_1LFD | 15 / 149 | 14.27 | 17.99 | -3.73 | -0.158 | -0.157 | -0.467 |
| 5A12_Ang2_fitness_4ZFG | 4 / 5 | 12.75 | 16.20 | -3.45 | -0.240 | +0.276 | -0.036 |
| CD19_FMC63_Fitness_7URV | 17 / 201 | 13.47 | 16.42 | -2.95 | -0.465 | -0.460 | -0.700 |
| SARS2-RBD_ACE2_deltaKd_6M0J | 21 / 173 | 14.52 | 16.20 | -1.68 | -0.198 | -0.198 | -0.339 |
| ACE2_SARS2-RBD_enrich_6M17 | 20 / 95 | 13.80 | 15.26 | -1.46 | -0.211 | -0.207 | -0.554 |
| CXCR4_CXCL12_enrich_8U4O | 37 / 258 | 14.30 | 15.51 | -1.21 | +0.105 | +0.116 | -0.193 |
| HLA-A2_TAPBPR_meanscore_5WER | 22 / 158 | 14.09 | 14.93 | -0.84 | -0.059 | -0.076 | -0.215 |
| hYAP65_peptide_FunctioncalScore_1JMQ | 11 / 23 | 12.36 | 12.26 | +0.10 | +0.041 | +0.041 | -0.030 |
| GB1_IgG-Fc_fitness_1FCC | 18 / 37 | 14.28 | 13.62 | +0.66 | +0.046 | +0.053 | -0.232 |
| PSD95_Tm2F_1BE9 | 16 / 67 | 17.38 | 16.70 | +0.67 | -0.047 | -0.051 | -0.254 |
| PSD95_CRIPT_1BE9 | 15 / 68 | 17.60 | 16.66 | +0.94 | -0.226 | -0.203 | -0.573 |

- **assay 级：** ρ(Δburial, δ_MPNN) = **−0.444, p = 0.085**（方向对，n=16 欠功效）；
  而 ρ(Δburial, δ_DMS) = **+0.024, p = 0.93** —— **burial 与真实的界面效应完全无关**。这个反差本身是干净的。
- **但逐 variant 的 burial 分层检验否掉了它**：在 5 个 burial 分层内重算 δ_MPNN，
  \|δ\| 中位 0.168 → **0.167**（保留 **0.993**）。**burial 匹配几乎没有削弱 MPNN 的界面对比。**

**为什么这个检验注定不够：** MPNN 的打分是在**完整复合物**上做的，它的 k=48 kNN 图**包含 partner 原子**。
所以「链内 burial」根本不是 MPNN 条件化的那个量。要分离"partner 带来的"与"自身结构带来的"，
必须真的把 partner 拿掉重新打分 —— 见 §6。

**结论：观察（§4）成立且强；机制（"只建模折叠 compatibility"）尚未被证明。** 这一条不要在论文里当成已证。

## 6. 决定性实验（**尚未跑**，建议下一步）

**monomer control：把 partner 链删掉，只用被突变链的结构重新 zero-shot 打分**，再算同样的 δ。

| 结果 | 解释 |
|---|---|
| δ 基本不变 | MPNN 的界面敏感度**全部来自被突变链自身的结构** ⇒ partner 是"看了但没用" ⇒ **complexTTT 的动机成立且强** |
| δ 明显变化 | MPNN 确实从 partner 提取了信息，只是提取得不好 ⇒ 动机要改写成"提取不充分"而非"感知不到" |

两种结果**都有用**，且这是唯一能把 §5 的空缺填上的实验。
**成本估算：** 全量 25 assay × M=5 × seed1 在本 A100 上实测 **≈2.5 h**（记录在上一分支）；
删掉 partner 后结构更小、kNN 图更少节点，预计 **≈1.5–2 GPU-h**。env `bindinggym-zs-mpnn` 已在 workstation 上。

## 7. Caveats

1. **只有 1 个 seed / M=5。** 既往实测 M=20 的逐 assay σ ≈ 0.008（Spearman 尺度）。
   本文的 δ 差异（0.25 vs 0.17，多个符号翻转）远大于该量级，但**δ 本身的 seed 噪声没有直接测过**。
   要加固就用 seed2–5 那 10 个 assay 做配对复算。
2. **9/25 个 assay 不可检验**（一侧子集 < 30），与 DMS 侧同一批，所以两侧的 16 个 assay 完全可比。
3. **界面定义、结构、映射全部沿用** `../binding-sites-analysis/`（D1：到从不被突变的 partner 链 ≤ 5 Å）。
   那边的 caveat 全部继承 —— 尤其 `4ZFF_CHL.pdb` 缺链、以及三个分数口径反常的 assay。
4. **MPNN 的 `global_score` 是 mask 上 NLL 的求和、无长度归一化** ⇒ 跨 assay 绝对值不可比。
   本文所有比较都在 assay 内做，未跨 assay 池化。
5. §5 的 burial 度量（链内 10 Å Cβ 邻居数）是**粗粒度代理**，它的失败**不能**证明 burial 无关，
   只能证明"这个代理解释不了"。

## 8. 复现

```
scripts/binding_sites_pred/
  p1_merge_scores.py   拉取的分数 → 三道 gate → variant_labels_with_mpnn.parquet
  p2_stats.py          DMS 与 MPNN 各自的两组统计（同一套估计量）
  p3_plot.py           MPNN 分布图（与 DMS 图同布局同顺序）
  p4_mechanism.py      burial 对比 + 逐 variant burial 分层检验
  p5_compare.py        对比图（δ 散点 + 配对效应量）
  p6_tables.py         本文 Table P1 / P2
```

- 分数来源：workstation `/data/guoj0f/BindingGYM-zero-shot-proteinMPNN/scores/seed1_M5/`
  （329 MB，**不入 repo**；`p1` 从 scratchpad 读，路径见脚本头）
- 界面标签来源：`../binding-sites-analysis/data/variant_labels.parquet`
- 产物：`data/{variant_labels_with_mpnn.parquet, stats_dms_vs_mpnn.csv, burial_contrast.csv,
  burial_stratified_delta.csv, merge_gates.csv}` + 两张图
- 纯 CPU，全流程约 4 分钟。**本次没有启动任何 GPU 任务。**

## 关联

- `../binding-sites-analysis/BindingGYM_binding_sites_20260829.md` —— 界面定义、DMS 侧的分布与统计（本文的对照臂）
- 上一分支 `proteinTTT-proteinGYM-reproduce` 的 `workstation-records/BindingGYM-zero-shot-proteinMPNN/`
  —— 这批分数是怎么跑出来的（协议、seed 语义、噪声地板）
