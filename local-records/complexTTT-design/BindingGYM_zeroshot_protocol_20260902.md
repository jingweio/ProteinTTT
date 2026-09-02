# BindingGYM zero-shot ProteinMPNN：评测契约、架构基底、与 complexTTT 的接入点

**分支** `BindingGYM-complexTTT-design` · **日期** 2026-09-02 · **状态** 口径已定案，设计未定案

**这份文档是什么**：complexTTT 全部设计所依赖的地基。凡是「BindingGYM zero-shot 到底怎么算的」
「ProteinMPNN 到底能表示什么」「TTT 能动哪里、动不了哪里」，以这份为准，不要回头重读 notebook。

**证据等级标注**：本文每条结论都标了来源。
- 🟢 **本人逐行读代码 / 亲自跑命令验证**
- 🔵 **14-agent 对抗验证 workflow 的产出，且我复核过关键数字**
- 🟡 **agent 产出，我未独立复核** —— 用之前请自己再确认一次

**方法学说明（必须知道）**：本文口径经过一轮 6 路独立读源 + 6 路对抗证伪
（1.64 M token）。**6 条 critical claim 里 4 条被推翻并修正**。下面记的全部是**修正后**的版本；
被推翻的原始说法保留在 §7，因为那 4 条正是最容易再犯一次的错。

---

## 1. 调用链与入口

```
modelzoo/proteinmpnn/run.sh → run.py                       # driver，遍历 BindingGYM.csv 的 25 个 index
  └─ baselines/protein_mpnn/compute_fitness_multi_pdb.py    # ← 真正的打分实现
       ckpt = training/cache/v_48_020.pt                    # vanilla ProteinMPNN, num_edges=48, noise_level=0.2
```

🟢 **这是原版 ProteinMPNN**（`forward(X,S,mask,chain_M,residue_idx,chain_encoding_all,randn)`），
与 `training/protein_mpnn_utils.py` 里那份改过的（`forward(data)` 收 PyG batch、带 logit-diff readout、
`masked_chains.sort()` 是激活的）**不是同一个类，绝不可混**。
zero-shot 依赖极轻：只要 `torch + pandas + 同目录的 protein_mpnn_utils.py`。

🟢 官方 driver 实际传的参数（`run.py:49-57`）：

```
--model_location {ckpt} --dms_index {0..24}
--dms_mapping ../../input/BindingGYM.csv
--dms_input   ../../input/Binding_substitutions_DMS
--structure_folder ../../input/structures
--dms_output  {dir_path}/output
--batch_size 8 --num_seq_per_target 5
```

| flag | 官方值 | 🟢 实际效果 |
|---|---|---|
| `--num_seq_per_target` | **5** | `NUM_BATCHES=5` ⇒ **M=5 个解码顺序**，脚本 default 是 1 |
| `--batch_size` | 8 | **死参数** —— `BATCH_COPIES = args.batch_size`（`:58`）全文件只赋值一次、从未被读 |
| `--seed` | **不传** | default 0 走 `if args.seed:` 的 **falsy** 分支 ⇒ `seed = np.random.randint(0,999)`。**官方分数用的解码顺序不可复现** |
| `--backbone_noise` | 不传 → 0.00 | 推理不加坐标噪声（ckpt 自带的 `noise_level=0.2` 只被 print，不生效） |
| 各种 `*_jsonl` | 不传 | 全 `None`；即便传 `chain_id_jsonl` 也会在 `:210-211` 被 per-assay 的 `chain_id` 覆盖 |

⚠️ 🟢 **ckpt 同名不同文件**（md5 亲验）：

```
91d54c97a68bf551114f8c74c785e90f  /home/guoj0f/repos/BindingGYM/training/cache/v_48_020.pt   ← 必须用这个
698982b1bda2b0d42e26538e64c93fda  /home/guoj0f/share/proteinmpnn/v_48_020.pt                 ← 另一个文件
```
🟡 权重最大逐元素差 **9.09**（agent 测的，我只核了 md5 不同）。

---

## 2. 打分函数（`compute_fitness_multi_pdb.py:187-282`）

```python
for (POI, chain_id), g in df.groupby(['POI','chain_id']):          # :187  每个 assay 一组
    pdb_dict = parse_PDB(pdb_file)                                 # :196  按 pdb_file 缓存，只解析一次
    designed = list(chain_id);  fixed = [c for c in all_chains if c not in designed]   # :204-209
    batch_clones = [deepcopy(pdb_dict)] * NUM_BATCHES              # :212  M 份完全相同的拷贝
    X, S, mask, chain_M, chain_M_pos, ... = tied_featurize(...)    # :214
    randn_1 = randn_1_dic.setdefault(POI, torch.randn(chain_M.shape))   # :216-220  [M, L]，按 POI 缓存

    for i in g.index:                                              # :225  逐 variant
        for c in chain_id:                                         # :231  按 chain_id 字符串顺序
            S[:, start:start+len] = mutated_sequence[c]            # :241  就地覆写全部 designed 链
        log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)  # :244
        design_score = -1 * _scores(S, log_probs, mask*chain_M*chain_M_pos).mean()   # :246,261
        global_score = -1 * _scores(S, log_probs, mask).mean()                        # :248,263
```

$$\boxed{\;\text{score}=-\frac{1}{M}\sum_{m=1}^{M}\ \sum_{j\,:\,\text{mask}_j=1}\mathrm{NLL}\!\left(S_j=\text{mut}_j \ \middle|\ X_{WT},\ S_{\text{mut}},\ \pi_m\right),\qquad M=5\;}$$

🟢 逐条性质：

- **无 WT 相减** —— 就是突变序列自身的自回归对数似然，没有 `score(mt) − score(wt)`。
- **无长度归一化** —— `_scores`（`protein_mpnn_utils.py:39-47`）里
  `torch.sum(loss*mask, -1)# / torch.sum(mask, -1)`，除法**被注释掉了**。是**求和**不是平均。
- **teacher-forced 一次 forward**，不是逐位采样。`model.eval()`（`:168`）+ `torch.no_grad()`（`:181`）。
- **方向**：越大 = 越像天然 = 与 `DMS_score` 同向。**评测端不取负**，直接算相关。
- **`design_score ≡ global_score`**（见 §3-①）。

### 2.1 `mask` 不是全 1 —— 这是最容易错的一条

🟢 `tied_featurize:409`：`mask = np.isfinite(np.sum(X,(2,3)))`。
**只有在 PDB 里有完整 N/CA/C/O 坐标的残基才进 loss。** 逐 assay 亲测：

| assay | mask=0 位点 | 占比 |
|---|---|---|
| BH3_Bcl-xL_normed_1PQ1 | 49 / 229 | **21.4%** |
| KRAS_PICK3CG-RBD_norfitness_1HE8 | 192 / 1107 | **17.3%** |
| CD19_FMC63_Fitness_7URV | 52 / 497 | 10.5% |
| HLA-A2_TAPBPR_meanscore_5WER | 62 / 644 | 9.6% |
| KRAS_SOS1_norfitness_8BE4 | 38 / 643 | 5.9% |
| BH3_Mcl-1_normed_3KZ0 | 7 / 173 | 4.0% |
| 4D5_HER2_fitness_1N8Z | 26 / 1041 | 2.5% |
| 5A12_VEGF_fitness_4ZFF | 8 / 528 | 1.5% |
| 5A12_Ang2_fitness_4ZFG | 4 / 652 | 0.6% |
| **合计** | **438 / 9931** | **4.41%（9/25 assay 受影响）** |

> **纳入口径**：上表只列了 mask=0 位点数 >0 的 9 个 assay；其余 16 个为 0，未列。
> 分母是「该 assay `chain_id` 全部链的残基数之和」。

这些位点：既不进 loss，也拿不到 kNN 边（`:944` 的 `D_adjust = D + (1-mask_2D)*D_max` 把它们推到最远），
且 **0 个 variant 突变在上面**。

⇒ **不改任何排序指标**（它们对每个 variant 贡献同一个常数）。
⇒ 但**任何绝对分数层面的 TTT 目标、跨 assay 诊断、「partner 贡献多少」的量化，都会对不上**。
   🟡 1HE8 的 partner（PIK3CG A 链）丢 192/941 = 20.4%、1PQ1 的 partner（Bcl-xL A 链）丢 49/196 = 25.0%。

### 2.2 解码顺序

🟢 `protein_mpnn_utils.py:1081-1083`：

```python
chain_M = chain_M * mask                                       # ← 有效掩码是 mask，不是 1
decoding_order = torch.argsort((chain_M + 0.0001) * torch.abs(randn))
```

- `chain_M ≡ 1`（§3-①）⇒ 解码顺序就是 `argsort(|randn|)`，是**跨整个复合物的均匀随机全排列，
  target 与 partner 残基完全交错**。
- `randn_1` 形状 `[M, L]`，**按 `POI` 缓存**（`:185` 定义在 `:187` 的 groupby 之外）⇒ 同一 assay
  的全部 variant 共用同一组 M 个排列；🟢 且 `POI` = 结构文件名 stem ⇒ **共用结构的 3 对 assay
  连解码顺序都共享**。
- `(chain_M+1e-4)` 是**乘性缩放不是硬 floor**：mask=0 的位点大概率但**不保证**排在最前。
  🟡 1PQ1（49/229 unresolved）2000 次抽样里 3.6% 出现 designed 位点插到前面。
- `forward` 暴露 `use_input_decoding_order=True, decoding_order=...` —— 🟢 **这是做 TTT vs baseline
  严格同序对照的干净钩子。**

---

## 3. 数据契约（全部 🟢 亲测 25/25）

### ① `chain_id` 覆盖 PDB 全部链 ⇒ `fixed_chain_list` 恒空

逐 assay 比对「PDB ATOM 记录里的链集合」vs「`chain_id`」：**0/25 存在 fixed 链**
（结构文件已被裁到只剩 `chain_id` 那几条，如 `4ZFF_CHL.pdb`、`6M17_BE.pdb`）。
⇒ `chain_M ≡ chain_M_pos ≡ 1` ⇒ **`design_score ≡ global_score`**。

⚠️ 这是**数据性质不是代码保证**。一旦把 partner 设成 fixed 链，两列立刻分家；
而 🔵 官方 score→metric 的胶水代码**不在 repo 里**（只有 git blob `994243c^:calc_metric.py:93`
证明 `pred_col='global_score'`）⇒ **官方 0.3970 对应哪一列不可知**。
这让「把 partner 固定住」这个改动比它看起来更危险。

### ② 逐链序列长度 25/25 精确吻合

用官方 `parse_PDB` 解析后逐链比对 `wildtype_sequence`：**25/25 全中，无一处 off-by-one**。
且 `chain_id` 字符串 25/25 都是**字母序**（与 `tied_featurize` 把 masked chain 排前面的假设一致）——
代码 `:241` 那句 `#assumes that S and S_input are alphabetically sorted` 是**真的有前提**。

### ③ target / partner 链：benchmark 里**没有这个字段**，只能从数据推

`BindingGYM.csv` 只有 6 列（`POI, DMS_id, DMS_filename, wildtype_sequence, chain_id, pdb_file`），
`POI` 是**结构文件名 stem** 不是链标识。哪条链被突变，只能扫 `mutant` 列推出来：

| 类别 | assay 数 | variant 数 | 说明 |
|---|---|---|---|
| **1 条 target + 1 条 partner** | **18** | **288,878 (76.7%)** | 符合「wt-target-seq + wt-partner-seq」的设定 |
| **2 条 target（抗体 H+L）+ 1 条抗原** | 3 | 33,005 (8.8%) | 4D5_HER2 / 5A12_Ang2 / 5A12_VEGF |
| **两条链都突变，无固定 partner** | **4** | **54,563 (14.5%)** | 4 个 Z-domain assay（2M5A ×2、1LP1 ×2） |

> **口径**：variant 数 = 该 assay CSV 的全部行数（**含 WT 行**）。
> 三类互斥且穷尽：18+3+4 = 25 个 assay，288,878+33,005+54,563 = **376,446** 行，与官方总数吻合。
> 逐 assay 的分类见 `data/complex_vs_single_delta.csv` 的 `cat` 列。

🟢 链字母与蛋白身份**没有稳定映射**：KRAS 在 6VJJ/5O2S 是 A、在 1HE8/1LFD 是 B、在 8BE4 是 R。

### ④ 22/25 有 WT 行，3 个没有

🟢 `Z-domain_ZSPA-1_LL1`、`Z-domain_ZSPA-1_LL2`、`HLA-A2_TAPBPR` **没有 WT 行**
（`mutant` 全空 dict 的那一行不存在）。任何依赖「WT 行必然存在」的 TTT 逻辑会在这 3 个上失败。

### ⑤ partner 残基在 score 里占比很大

🟢 逐 assay（`L` 用 `wildtype_sequence` 长度，**未扣 mask=0**，所以是上界）：

| assay | L_target | L_partner | partner 占比 | 库位点/全残基 |
|---|---|---|---|---|
| BH3_Mcl-1_3KZ0 | 23 | 150 | **86.7%** | 5.8% |
| KRAS_PICK3CG_1HE8 | 166 | 941 | **85.0%** | 14.8% |
| BH3_Bcl-xL_1PQ1 | 33 | 196 | 85.6% | 4.4% |
| SARS2-RBD_ACE2_6M0J | 194 | 597 | 75.5% | 24.5% |
| KRAS_SOS1_8BE4 | 168 | 475 | 73.9% | 25.4% |
| …（中位 33.7%，9/25 > 50%）… | | | | |
| PSD95_CRIPT_1BE9 | 115 | 5 | 4.2% | 69.2% |
| 4 个 Z-domain | 109–116 | **0** | **0** | 5.2–8.3% |

⇒ score = **巨大的近常数项 + 小方差项**。partner 的 log-prob 并非常数（解码顺序是全复合物随机排列，
平均约一半 partner 残基排在突变位点之后）—— **这是 partner 信息进入 score 的唯一通道**。

---

## 4. 架构基底：ProteinMPNN 到底能表示什么

### 4.1 结构特征（🟢 `ProteinFeatures.forward`，`protein_mpnn_utils.py:922-1020`）

- **k-NN 图基于 Ca 距离**，`k = num_edges = 48`（ckpt 提供，覆盖 `top_k=30` 的 default）。
- **边特征 416 维** = `num_positional_embeddings(16)` + `num_rbf(16) × 25`：
  25 组原子对（N/Ca/C/O/**虚拟 Cb** 的全部有序对）各 16 个 RBF bin，`D_min=2, D_max=22 Å`。
  虚拟 Cb 由 backbone 解析式构造：`Cb = -0.58273431·a + 0.56802827·b - 0.54067466·c + Ca`。
- `edge_embedding = Linear(416, 128, bias=False)` + `LayerNorm(128)`。
- `augment_eps` 的噪声加在 **k-NN 图构建之前**（`X = X + augment_eps*randn_like(X)` 是 forward 第一行）
  ⇒ 它会改变**图的连通性**，不只是边特征值。

### 4.2 ⭐ 「界面」在架构里只有 **16 个参数**

🟢 `PositionalEncodings.forward`：

```python
d = clip(offset + 32, 0, 64) * E_chains + (1 - E_chains) * 65
E = self.linear(one_hot(d, 66))            # linear: Linear(66, 16)
```

其中 `E_chains = (chain_labels[:,:,None] == chain_labels[:,None,:])`，即「同链 = 1 / 跨链 = 0」。

| 边类型 | `d` 取值 | 含义 |
|---|---|---|
| **同链** | `clip(相对序列偏移+32, 0, 64)` → 0..64 | 65 个 bin，编码相对序列位置（±32 截断） |
| **跨链** | **恒为 65** | **一个 bin，全部 inter-chain 边共用** |

⇒ **ProteinMPNN 对「这条边跨界面」的全部显式表示 = `linear.weight[:, 65]` 这一个 16 维向量、
16 个参数**，且**不区分是哪条链、界面多大、什么化学性质**。

**这条对 complexTTT 的意义**（我的解读，非实测）：
complex 相对 single-chain 的全部增益，只可能来自两处 ——
(a) partner 原子**占据了 k=48 的邻居槽位**并贡献 RBF 距离特征；
(b) 那**一个 16 维跨链 embedding**。
「让模型更好地刻画这个 WT 复合物界面」如果指的是**表示能力**，那可动的显式容量极小；
如果指的是**利用已有容量**，那战场在 (a)——即 k-NN 图里 partner 占了多少槽、以及编码器如何消化它们。

### 4.3 编码器与变体无关（🟢 从代码可判定，🟡 待实测确认 bit-identical）

`forward` 里 `h_V = torch.zeros(...)`，编码器只吃 `E, E_idx`（来自 `X, mask, residue_idx,
chain_encoding_all`）—— **全部与 `S` 无关**。序列只通过 `h_S = self.W_s(S)` 进入**解码器**。

⇒ **同一 assay 内，编码器输出对全部 variant 完全相同。** per-variant 的差异 100% 来自
解码器如何在这个固定 context 下读突变序列。
⇒ TTT 改编码器参数 = 改一个**所有 variant 共享的 context**；它能改排序，是因为解码器对不同 `S`
的响应随 context 非单调地变化 —— 但这是间接通道，不是直接给某个 variant 加分。

---

## 5. 指标与聚合（🟢 `calc_metric.ipynb` cell 2）

```python
label_bin = (df[label_col] > np.percentile(df[label_col].values, 90)) + 0    # 正类 = label 自身 90 分位
pred_bin  = (df[pred_col]  > np.percentile(df[pred_col].values,  90)) + 0    # MCC 预测侧也是自身 90 分位
Spearman = df[label_col].rank().corr(df[pred_col].rank())
AUC = roc_auc_score(label_bin, df[pred_col])
MCC = matthews_corrcoef(label_bin, pred_bin)
NDCG = ndcg_score(label.rank(), pred, k = df.shape[0] // 10)
AP  = average_precision_score(label_bin, df[pred_col])
# top_test=True（zero-shot 路径的默认）再加 TopHit/BottomHit/UnbiasHit @ {10,20,50,100}
```

- 🟢 `get_zero_shot_metric_df` 遍历 `BindingGYM.csv` 的 25 个 `DMS_id`，并
  **`assert df.shape[0] == orig_df.shape[0]`** ⇒ **WT 行必须保留，输出行数必须与原始 CSV 一致**。
- 🟢 **聚合 = 25 个 assay 的未加权算术平均**。对 `results/ProteinMPNN_zero_shot_metric.csv`（25×18）
  取列均值亲测得 **0.396950 / 0.687947 / 0.153505 / 0.722055 / 0.219163**，与论文
  `0.40 / 0.69 / 0.15 / 0.72 / 0.22` 逐位吻合。
- 🔵 **全部 metric 对预测的严格单调变换恒等不变**（因为二值化用 90 分位而非 `> WT`）。
  25 assay × 8 变换 × 18 指标实测 worst |Δ| = **0.0**。
  🟡 两条边界：在真实分数量级（sum-NLL，offset ≈ −1500）上直接 `np.exp` 会 underflow 塌成常数；
  `np.percentile` 的线性插值在 relative spread ≲1e-8 时会因 float64 舍入改变阈值后缀
  （真实 ProteinMPNN 是 ~1e-3~1e-4，离危险区 6 个数量级）。
- 🟢 `Binding_substitutions_DMS/` 有 **28** 个 CSV / 508,962 行，`BindingGYM.csv` 只登记 **25** 个 /
  **376,446** 行。多出的 3 个流感 assay 官方 pipeline 从不碰。
- 🟢 `inter_cluster` split：25 个 assay 落在 **14 个簇**（`training/cache/BindingGYM_cluster.tsv`），
  是唯一不泄漏测试 assay label 的切分。

---

## 6. 参考点与天花板

| 口径 | mean Spearman | 来源 |
|---|---|---|
| ProteinMPNN zero-shot（**complex**） | **0.396950** | 🟢 `results/ProteinMPNN_zero_shot_metric.csv` |
| ProteinMPNN zero-shot（**single chain**） | **0.356353** | 🟢 `results/ProteinMPNN_single_zero_shot_metric.csv` |
| 我们的复现（M=5、显式 seed） | 0.391356 | 🟢 memory，Δ=−0.0056 ≈ 0.4–0.8σ |
| ProteinMPNN **有标签** inter-cluster finetune | **0.4217** | 🔵 官方 |
| ProteinMPNN-R（随机初始化 + 同样微调） | 0.1585 | 🟡 H3-DDG ref 记录 |

🟢 **complex − single 的逐 assay 差**（Δ = complex − single，均值 **+0.040597**，**6/25 为负**）：

| 最正 5 个 | Δ | 最负 5 个 | Δ |
|---|---|---|---|
| Z-domain_ZSPA-1_LL1 | **+0.1474** | Z-domain_ZpA963_HL1 | **−0.0642** |
| GB1_IgG-Fc_1FCC | +0.1258 | 4D5_HER2_1N8Z | −0.0251 |
| PSD95_CRIPT_1BE9 | +0.1054 | KRAS_SOS1_8BE4 | −0.0154 |
| KRAS_RAF1-RBD_6VJJ | +0.1044 | ACE2_SARS2-RBD_6M17 | −0.0133 |
| GB1_IgG-Fc_1FCC_2016 | +0.0914 | KRAS_PICK3CG_1HE8 | −0.0097 |

> 完整 25 行见 `data/complex_vs_single_delta.csv`（本目录，与本文一同产出；列含
> `Spearman_{complex,single}` / `AUC_{complex,single}` / `delta_Spearman` / `delta_AUC` /
> `n` / `target` / `partner` / `cat`）。
> **口径**：Δ>0 表示「把 partner 放进图里」有帮助。上表只列了最正/最负各 5 个；
> 6 个负值的完整名单是 ZpA963_HL1(−0.0642)、4D5_HER2(−0.0251)、KRAS_SOS1(−0.0154)、
> ACE2_SARS2-RBD_6M17(−0.0133)、KRAS_PICK3CG_1HE8(−0.0097)、5A12_Ang2_4ZFG(−0.0020)。
> Δ 中位数 **+0.0308**（均值 +0.0406 被 ZSPA-1_LL1 的 +0.1474 拉高）。

⇒ **「建模 complex 值多少」有可直接对标的逐 assay 向量**，不必只拿一个 +0.0406 的均值。
这是 [partner-blind] 那条结论的第三条独立证据：**整个 complex-modelling 只值 +0.041，
而带标签的跨 assay 微调总共只值 +0.025。**

🔵 **噪声底**：per-assay 5-seed σ 中位 **0.0184**、最大 0.0581；
σ_official(25-assay 均值) ≈ 0.0249/√25 ≈ **0.0050**。**25-assay 均值要涨 >≈0.010 才算数。**

---

## 7. 四条被推翻的常见误述（我自己第一轮也说错了前三条）

| # | ❌ 错误说法 | ✅ 修正 |
|---|---|---|
| 1 | 「求和跑遍整个复合物的每一个残基」 | `mask = isfinite(backbone)`，438/9931 = 4.41% 不进 loss（§2.1）。不改排序，但改一切绝对分数层面的量 |
| 2 | 「zero-shot 全程没有任何 masking」 | 应为「**不对突变位点** masking」。`tied_featurize:254,288` 把未解析残基的 `'-'` 改写成 `'X'`(token 20)——**与 fine-tune 同一个 token**；且自回归解码顺序本身就是 masking。⇒ **ProteinMPNN 没有空闲的 mask token** |
| 3 | 「共享解码顺序，绕过会显著抬噪声（一个量级）」 | 🟡 实测 PSD95_CRIPT 全量：M=5 下 per-variant 重抽 randn 只让 Spearman 0.3789→0.3316（**−0.047，−12%**）；M=1 时才崩（0.3892→0.2372）。⇒ 代价主要来自 **M 小**而非共享与否。且**配对差不会完全抵消**：同组 200 order 下 `std(ΔNLL)` = 0.61–0.91 nats vs 单臂 2.59–2.76，只降到 **1/3–1/4** |
| 4 | 「官方报的是 `global_score`」 | 胶水代码不在 repo 里，**不可知**；当前两列恒等所以无所谓，但把 partner 设 fixed 后就变成真问题（§3-①） |

---

## 8. TTT 的接入点与陷阱

### 8.1 三处改动（其余保持 byte-identical）

🔵 复制成**同目录**的 `compute_fitness_multi_pdb_ttt.py`（必须同目录：`from protein_mpnn_utils import`
是裸导入）。只改：

**(a) `:216-220`** —— 把 `randn_1` 从全局 RNG 上摘下来：
```python
g_dec = torch.Generator(device=X.device); g_dec.manual_seed(stable_seed(POI, args.seed))
randn_1 = torch.randn(chain_M.shape, device=X.device, generator=g_dec)
```
⚠️ 这会改变与已复现 baseline 的数值 ⇒ **两臂都用这个改动，并重跑一次 baseline 作为新锚点。**

**(b) `:214`（tied_featurize）与 `:225`（variant 循环）之间**插入唯一的 hook：
```python
S_wt = S.clone()                      # 必须在 :241 就地覆写之前抓
with torch.enable_grad():
    model_ttt = ttt_adapt(model, X, S_wt, mask, chain_M*chain_M_pos,
                          residue_idx, chain_encoding_all, cfg)
model_ttt.eval(); model_ttt.features.augment_eps = 0.0
```

**(c) `:244`** `model(...)` → `model_ttt(...)`。

**一律不动**：`:245-249` 两次 `_scores`、`:261/263` 的 `-1*mean`、`:278-279` 两列、
`:281` `sort_index()`、`:282` `to_csv(index=False)`。

🔵 **验收测试**：`--ttt_steps 0` 时产出的 25 个 CSV 必须与 baseline **逐个 md5 相同**。

### 8.2 三个静默陷阱（全部 🟡 agent 实测，无报错无警告）

**V1 — CUDA RNG 污染 ⇒ 两臂不再配对。** `:217` 的 `randn_1` 在 **CUDA** 流上抽。
ProteinTTT 的 `ttt()`（`base.py:558`）调 `self.train()`，而 ProteinMPNN 每个 Enc/DecLayer 都带
`dropout=0.1` ⇒ 在 CUDA 上跑一次 dropout 就把 philox 流推走 ⇒ 5 个解码顺序全变。
agent 在 A4500 上的实测：
```
baseline                  [-0.2963,  2.6764, -0.1408]
after CPU  randn(1000)    [-0.2963,  2.6764, -0.1408]   same: True
after CUDA randn(1000)    [-0.6340, -0.6431, -0.6420]   same: False
after CUDA dropout(train) [-0.6340, -0.6431, -0.6420]   same: False
```
**解法就是 8.1(a) 的独立 Generator。**

**V2 — 忘了复位。** `torch.enable_grad()` 包住 TTT 后漏掉 `model.eval()`，或漏把
`model.features.augment_eps` 复位到 0（官方 finetune 正是 train 用 0.2 / eval 复位，
`training/main.py:414 / 456 / 531`）⇒ 每个 variant 每次 forward 重采坐标噪声。

**V3 — stale 输出复用。** `modelzoo/proteinmpnn/run.py:44-46` 见到 `./output/{DMS_id}.csv` 就跳过，
写入路径却是绝对的 ⇒ TTT 臂写进同一目录会**静默复用 baseline 的 CSV**。
**每臂必须独立 `--dms_output`。**

### 8.3 标签泄漏的真实入口

🔵 `:187` 的 `for (POI,chain_id),g in df.groupby(...)` 里，**`g` 就是含 `DMS_score` 整列的 variant 表**。
任何接收 `g` 的 hook 都把标签拿在手里。以下看起来无监督、实际是泄漏：
按 assay Spearman 早停或选步数 · per-assay 选 lr · 把预测的 top-decile 往标签的 90 分位对齐。
（**从 `mutated_sequence` 推哪条链是 target 不算泄漏。**）

🟡 **transductive 已经发生了，而且是 benchmark 自己干的**：`modelzoo/msa_BindingGYM.py:29-36` 的
`focus_chains` 是遍历整张 variant 表的 `mutant` 列推出来的 ⇒ 只要用了分发的 `.a2m`，
就已间接消费了 variant 列表。论文里必须写明。

### 8.4 合法 / 非法的 test-time 输入

**合法**：WT 复合物 PDB（仅 backbone N/CA/C/O 进模型）· 逐链 WT 序列（含 partner）·
`chain_id` / `POI` / `pdb_file` · 预训练 ckpt · variant 表除 `DMS_score` 外的列（transductive，需声明）·
22 个 `.a2m`（但见下）。

**非法**：`DMS_score` 任何形式、任何环节（含早停、模型选择、per-assay 归一化、阈值选择）。
🟢 官方 zero-shot 侧 `grep -rn "DMS_score" modelzoo/ --include=*.py` **为空**。

🟡 **MSA 基本没用**：22 个 `.a2m` 的 query = **仅被突变的 focus 链**按字母序拼接（22/22 长度吻合），
**partner 从不进 query** ⇒ 18/22 严格 target-side、零 partner 信息。
ProteinTTT 的 `msa=True` 路径在 BindingGYM 上拿不到任何 complex-level 进化信号。

---

## 9. 四条约束 complexTTT 立论的事实

**① 🟢 6/25 个 assay 的 test-time 输入逐字节相同 ⇒ 24% 的 headline 无 assay-specific 适配。**

md5(`chain_id` + `wildtype_sequence` + `pdb_file`) 实测 3/3 IDENTICAL：

| PDB | assay 对 | variant 数 | md5 |
|---|---|---|---|
| `1LP1.pdb` | ZSPA-1 LL1 / LL2 | 45,476 / 5,583 | `ef6a5622d50c` |
| `2M5A.pdb` | ZpA963 HL1 / HL2 | 2,904 / 600 | `8c2d04837271` |
| `6VJJ.pdb` | KRAS_RAF1 / RAF1-RBD | 12,677 / 23,162 | `284606deaaa7` |

**WT-only TTT 对这 6 个必然给出完全相同的 θ 和完全相同的分数。**
既是省 3 个 assay 算力的机会，**也是最诚实的自检：若增益全落在这 6 个上，那不是 assay-specific 适配。**

**② 🟢 complex-modelling 全部只值 +0.0406，且 6/25 为负**（§6）。

**③ 🔵 天花板很紧**：带标签的跨 assay 微调 headroom 只有 **+0.025**（0.4217 vs 0.3970）。
无标签、WT-only 的适配应默认落在这之内。噪声底见 §6。

**④ 🟡 没有 dev split，而且造不出来。** 25 个 assay 全是 headline；那 3 个未登记的流感 assay 的
`pdb_file`（`4FQI_hm` / `4FQY_hm` / `3GBN_hm`）在 `structures/` 里**全部 missing**、也没有 `.a2m`
⇒ **任何看着 BindingGYM Spearman 选出来的 TTT 超参都是 test-set tuning，必须在论文里明说。**

---

## 10. 尚未关闭的问题

1. 🟡 §7-3 的 M/共享-randn 实测只在 **PSD95_CRIPT_1BE9 一个 assay**（L=120）上做过，未推广。
2. 🟡 §2.1 的 mask=0 对「partner 贡献」的修正量只算了 1HE8 / 1PQ1 两个。
3. **未测**：score 的方差有多少来自 target 残基 / partner 残基 / 突变位点本身
   —— 这是决定 TTT 该往哪推的最关键分解，正在跑（见 `substrate_map_*.md`）。
4. **未定**：ProteinTTT 的 `_ttt_mask_token` 抽象在无 mask token 的 inverse-folding decoder 上如何定义。
5. 🟡 分发的结构不是 RCSB deposition（occupancy=0.00、B=0.00、含氢、author numbering 被改过）
   ⇒ **重新从 RCSB 下载 = 换资产**。未逐个核对。

---

## 附：本文的证据来源

- 🟢 部分：我在本 session 内逐行读 `/home/guoj0f/repos/BindingGYM` 的代码 + 在
  `~/anaconda3/envs/proteingym-ttt` 里跑的验证脚本。
- 🔵/🟡 部分：14-agent workflow `wf_98617ef3-936`（6 路独立读源 + 6 路对抗证伪 + completeness critic，
  1,639,470 token，470 次工具调用，57 分钟，0 error），journal 在
  `~/.claude/projects/.../subagents/workflows/wf_98617ef3-936/journal.jsonl`。
- 相关 memory：`bindinggym-zeroshot-protocol-and-ttt-hooks`、`complex-ttt-wt-likelihood-refuted`、
  `proteinmpnn-partner-blind-on-bindinggym`、`bindinggym-wt-anchor-per-assay`、
  `bindinggym-binding-sites-artifacts`。
