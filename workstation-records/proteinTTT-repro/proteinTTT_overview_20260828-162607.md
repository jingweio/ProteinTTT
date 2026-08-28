# ProteinTTT 技术要点速览

> 本文档内容全部对照本仓库代码（`proteinttt/base.py`、`proteinttt/models/*.py`、`scripts/config_*.yaml`）
> 与论文 *ProteinTTT* (ICLR 2026) 核对过。
> 同目录复现记录：`esm2_zeroshot_subs_20260824-225409.md`

---

# 第一部分 · 核心技术模块

**一句话：** 对**每一条待预测的蛋白**，用它**自己的序列**做几十步自监督微调，再用微调后的模型出预测；
换下一条蛋白前把权重**重置**回预训练状态。没有标签、没有验证集、没有跨样本共享。

## 1. 训练数据 = 单条 WT 序列 + 数据增强

每一步都从同一条序列现采一个 batch（`base.py:1029-1110`）：

**① 随机裁剪到 `crop_size = 1024`**
- 序列 ≥ 1024：每条样本独立随机取起点 → 同一 batch 内是不同的窗口
- 序列 < 1024：整条用上，不做 padding

**② 15% BERT 掩码（`mask_ratio = 0.15`）**
- 只在**非特殊 token** 上采样（跳过 `<cls>` / `<eos>` / `<pad>`）
- 数量 = `int(非特殊位置数 × 0.15)`

**③ 被选中位置按 80 / 10 / 10 处理**

| 概率 | 操作 | 配置项 |
|---|---|---|
| 80% | 替换为 `<mask>` | 1 − leave − replace |
| 10% | 替换为随机氨基酸 | `bert_replace_prob = 0.1` |
| 10% | **保持原样** | `bert_leave_prob = 0.1` |

> 这三条（裁剪长度、掩码率、80/10/10）**刻意与各模型的预训练设置一致** ——
> ESM2 / ESMFold / SaProt / ProSST 预训练就是这套，目的是让 TTT 不引入分布偏移。

**④ batch 内的随机性来自增强本身**：同一条序列复制 `batch_size` 份，各自独立采裁剪窗口与掩码位置。

## 2. 两种 loss —— 这是最容易被忽略的分叉点

代码里有两个实现（`base.py:1150` / `base.py:1191`），**用哪个由任务决定**：

### (a) `cross_entropy` —— 只算被掩码的位置

```python
loss = F.cross_entropy(logits[mask], targets[mask])   # 逐序列取均值，再跨 batch 取均值
```
纯粹的 **predict-mask**：只要求模型补回被遮住的 15%。

### (b) `unnormalized_cross_entropy` —— **完全不看 mask**

```python
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))   # 对 crop 内所有位置
```
对**裁剪窗口内的每一个位置**算 CE，因此同时包含两件事：

- **(a) predict-mask** —— 被遮住的 15%，真正的"猜"
- **(b) rebuild-unmask** —— 未被遮住的 85%，模型**看得见输入就是答案**，等于在做"抄写"

> ⚠️ 这两部分的梯度贡献严重不均。我们在复现中实测（ESM2 35M）：
> **掩码位置 loss ≈ 1.80，未掩码位置 loss ≈ 0.29，但后者数量是前者的 ~5.7 倍
> ⇒ 约 85% 的梯度来自"抄写已可见残基"。**
> 也就是说，fitness 任务上的 TTT 与其说是在做 MLM，不如说主要在把模型往
> **"记住这条特定序列"** 的方向推。

### 谁用哪个（代码实测）

| 任务 | 模型 | `loss_kind` |
|---|---|---|
| **fitness** | ESM2 35M/650M、SaProt 35M/650M、ProSST、ProGen2 | `unnormalized_cross_entropy` |
| **structure** | ESMFold、DPLM2 | `cross_entropy`（默认，只算 mask） |

ProGen2 是自回归模型，走 teacher forcing 的下一 token 预测（论文 Eq. 4），
且被强制要求 `loss_kind = unnormalized_cross_entropy`、`batch_size = 1`。

## 3. 微调范围：默认全量，装不下才上 LoRA

`base.py:768-801` 的逻辑是：**先冻结全部 → 解冻 trainable modules → 再冻结 frozen modules**。

**① 大多数模型全量微调（`lora_rank = 0`）** —— ESM2、SaProt、ProSST、ProGen2、DPLM2。

**② token embedding 一律冻结**（各模型覆写 `_ttt_get_frozen_modules`）：

```python
# esm2.py     → return [self.embed_tokens]
# prosst.py   → return [self.prosst.embeddings]
# esmfold.py  → return [self.esm.embed_tokens]
```

> **一个连带效果：** 这些模型的输出头与输入嵌入是**权重绑定**的
> （ESM2 `lm_head.weight ≡ embed_tokens.weight`，ProSST 同理）。
> 冻结 `embed_tokens` 因此**同时冻结了输出投影** —— 实际训练的只有 transformer 主干。
> （我们在复现中还发现上游 `_ttt_set_state` 的 reset 会打断这层绑定，已修复并加了回归测试。）

**③ 太大装不下的用 LoRA** —— 只有 **ESMFold（3B，底座是 ESM2-3B）**：

```yaml
# scripts/config_ESMFold_ProteinTTT.yaml
lora_rank: 8        lora_alpha: 32.0
```
- 注入目标：`lora_target_replace_module = "MultiheadAttention"`
- **只训练 `self.esm`（语言模型主干）** —— folding trunk / structure module **完全不动**
  （`esmfold.py:39-40`）。所以 ESMFold 的 TTT 本质是"只调 ESM2，让结构模块吃到更好的表征"。

## 4. 优化与步数选择

| 项 | 值 | 说明 |
|---|---|---|
| optimizer | **SGD，`momentum = 0.0`，`weight_decay = 0.0`** | 刻意最简，避免优化器状态跨蛋白泄漏 |
| 梯度累积 | `ags`（8–64） | 在小显存下放大有效 batch |
| 步数 | `steps = 30`（fitness 主流） | ProGen2 用 15，ESMFold-MSA 用 20 |
| lr | 随模型规模递减 | ESM2 35M `4e-4` → 650M `4e-5` → ProSST `1e-5` |

**best-step 选择（论文 §3）：** 单条蛋白没有验证集，所以先固定跑 T 步得到
Θ = {θ₀, θ₁, …, θ_T}，再取 `argmax_θ c(·)`，c 是置信度函数。

- **structure**：c = **pLDDT**（`esmfold.py:130` `confidence = plddt`）。
  注意候选集**包含 θ₀（未微调的原模型）** —— 所以 TTT 在结构任务上**有下界保护**：
  若所有步都不如原模型，就退回原模型。
- **fitness**：基类 `_ttt_eval_step` 返回 `confidence = None` ⇒ **没有可用的 c，直接取最后一步 θ_T**。
  **没有下界保护**，这是两个任务表现差异的一个直接来源。

**重置：** 换下一条蛋白前 `ttt_reset()` 回到 θ₀，蛋白之间完全独立。

---

# 第二部分 · 两个任务上的评估

## 1. Structure prediction

**Benchmark：** CAMEO（沿用 ESMFold 论文的 val/test 划分），**只取 ESMFold 预测置信度低的靶点**
（test 集 18 条）。指标 TM-score / LDDT，5 个随机种子。

| Method | TM-score ↑ | LDDT ↑ | ΔTM |
|---|---|---|---|
| ESMFold | 0.4649 | 0.5194 | — |
| ESMFold + MP（掩码预测 baseline） | 0.4862 ± 0.0043 | 0.5375 ± 0.0070 | +0.0213 |
| **ESMFold + ProteinTTT** | **0.5047 ± 0.0132** | **0.5478 ± 0.0058** | **+0.0398** |
| ESM3 | 0.3480 ± 0.0057 | 0.3723 ± 0.0055 | — |
| ESM3 + CoT | 0.3677 ± 0.0088 | 0.3835 ± 0.0024 | +0.0197 |
| **ESM3 + ProteinTTT** | **0.3954 ± 0.0067** | **0.4214 ± 0.0054** | **+0.0474** |
| DPLM2 Bit-based | 0.3701 ± 0.0102 | 0.4681 ± 0.0071 | — |
| DPLM2 + ProteinTTT | 0.3796 ± 0.0024 | 0.4742 ± 0.0093 | +0.0095 |
| HelixFold-Single | 0.4709 | 0.4758 | — |
| HelixFold-Single + ProteinTTT | 0.4839 ± 0.0045 | 0.4840 ± 0.0061 | +0.0130 |

**机制链条（论文 Fig. 3 的核心叙事）：**
```
TTT 降低 ESM2 对该序列的 perplexity  →  ESM2 表征变好  →  折叠模块给出更好的结构
```
7EBL_B 那个例子：perplexity 15.83 → 11.66，TM-score 0.29 → **0.92**（第 7 步）。

## 2. Protein fitness prediction

**Benchmark：** ProteinGym substitutions（217 个 DMS），5 个随机种子。
官方聚合方式：assay → (UniProt, 功能类别) → 功能类别 → 5 类取均值（**不是 217 个的算术平均**）。

| Model | baseline | + ProteinTTT | Δ |
|---|---|---|---|
| ESM2 (35M) | 0.3211 | 0.3407 ± 0.00014 | **+0.0196** |
| ProGen2-small (151M) | 0.3255 | 0.3591 ± 0.0002 | **+0.0336** |
| ProGen2-large (2.7B) | 0.3724 | 0.3817 ± 0.00158 | +0.0093 |
| SaProt (35M) | 0.4062 | 0.4106 ± 0.00004 | +0.0044 |
| ESM2 (650M) | 0.4139 | 0.4153 ± 0.00003 | +0.0014 |
| SaProt (650M) | 0.4569 | 0.4583 ± 0.00001 | +0.0014 |
| ProSST (K=2048) | 0.5068 | **0.5087** ± 0.00004 | +0.0019 |

**我们的复现结果**（workstation A100，官方聚合口径）：

| model | baseline 我们/论文 | +TTT 我们/论文 | 增益复现率 |
|---|---|---|---|
| ESM2 (35M) | 0.3208 / 0.3211 | 0.3397 / 0.3407 | **96%** |
| ESM2 (650M) | 0.4137 / 0.4139 | 0.4147 / 0.4153 | 70% |
| ProSST (K=2048) | 0.5069 / 0.5068 | **0.5087 / 0.5087** | **96%** |

## 3. 为什么 structure 上的收益远大于 fitness

五条原因，从强到弱：

**① 评测集选择方式完全不同 —— 这是最大的一条**
structure 只在 **ESMFold 置信度低的 18 条难靶点**上评；fitness 在 **全部 217 个 DMS** 上评。
前者是"专挑模型不会的"，后者含大量模型本来就做得好的样本，平均下来自然被稀释。

**② structure 有置信度做 best-step 选择，fitness 没有**
pLDDT 让结构任务能从 {θ₀ … θ_T} 里挑最好的一步，**且 θ₀ 在候选里 ⇒ 有下界保护**。
fitness 没有可用的置信度，只能取最后一步，**打坏了也退不回去**。
我们复现中确实观察到单个 assay 最差 **−0.1926**、最好 **+0.5137** 的巨大方差。

**③ 优化目标与下游读出的对齐程度不同**
- structure：TTT 降 perplexity → 表征变好 → 结构变好，**因果链直接**，pLDDT 还能验证它确实变好了。
- fitness：读出是**位点级 log-prob 之差**（`log p(mut) − log p(wt)`）。
  TTT 用的 `unnormalized_cross_entropy` 有 ~85% 梯度在"抄写已见残基"，
  它把整条序列的似然抬高，但**对排序（Spearman）而言，全局常数是不改变名次的**，
  真正起作用的只有那部分改变了**相对**分布的信号。目标与读出之间隔了一层。

**④ 基线饱和程度不同**
fitness 上增益与基线强度明显反相关：ESM2 35M（0.32）+0.0196，而 ESM2 650M（0.41）只有 +0.0014，
ProSST（0.51）+0.0019。论文也承认这可能是"大模型上 benchmark 已接近饱和"。
structure 那 18 条低置信靶点则**远未饱和**。

**⑤ 只需改善"表征"vs 需要改善"精细排序"**
结构任务只要表征整体更贴合这条序列即可；fitness 要在**同一条序列的上千个单点突变之间**排出正确次序，
这是更精细的要求，单序列自监督能提供的信息有限。

## 4. ProteinTTT 对 MSA 的消费与依赖

### (a) 主方法**不用** MSA

Eq. 2 的目标只吃**单条目标序列**。论文明确的动机：很多蛋白**没有同源序列**，
且搜同源**很慢**。这是 ProteinTTT 相对 MSA-based 方法的核心卖点。

### (b) 有 MSA 时的扩展：ProteinTTT_MSA（Eq. 5）

$$\mathcal{L}_{\mathrm{MSA}}(x;\theta)=\mathbb{E}_{x'\sim p_{\mathrm{MSA}}(x'|x)}\big[\mathcal{L}(x';\theta)\big]$$

做法很直接：**把同源序列当作额外的训练样本**，仍用同一套 15% 掩码 + 80/10/10，
只是采样源从"目标序列自身"扩成"目标序列 + 它的同源"。
代码里 `msa_sampling_strategy` 支持 `random` / `top` / `neighbors` / `cluster`（DBSCAN 聚类后轮转采样，
保证一个 batch 里同源来源多样）。

### (c) 效果：MSA 带来的增益远大于单序列

| Method | Avg. Spearman |
|---|---|
| ESM2 (650M) | 0.4139 |
| ESM2 + **ProteinTTT**（单序列） | 0.4153 (**+0.0014**) |
| ESM2 + **ProteinTTT_MSA** | **0.4299** (**+0.0160**) |
| MSA Transformer（预训练就吃 MSA） | 0.4319 |
| MSA Transformer + ProteinTTT | 0.4326 (+0.0007) |

**增益差 11 倍。** 一个纯单序列模型 + MSA-TTT（0.4299）已经逼近专门在 MSA 上预训练的
MSA Transformer（0.4319）。⇒ **fitness 任务上，TTT 的收益上限主要由"有没有进化信息"决定，
而不是由 TTT 这套机制本身决定。**

### (d) "TTT 主要帮低 MSA 深度的蛋白"—— 论文的说法，但它自己的表格并不完全支持

论文正文与 Table A4 的标题都说 *primarily improves performance on ... low MSA depth*。
按 Table A4 的数字逐一算 Δ：

| model | Low Δ | Medium Δ | High Δ |
|---|---|---|---|
| ESM2 (35M) | +0.0051 | **+0.0437** | +0.0088 |
| ProGen2-small | +0.0345 | **+0.0500** | +0.0152 |
| ProGen2-large | **+0.0123** | −0.0017 | +0.0062 |
| SaProt (35M) | +0.0019 | **+0.0051** | +0.0034 |
| ESM2 (650M) | +0.0017 | **+0.0063** | **−0.0078** |
| SaProt (650M) | **+0.0007** | −0.0001 | −0.0009 |
| ProSST | **+0.0078** | −0.0003 | +0.0001 |

**7 个模型里只有 3 个（ProGen2-large / SaProt 650M / ProSST）是 Low 最高**，
ESM2 35M 与 ProGen2-small 的最大增益都在 **Medium**，且 ESM2 650M 在 High 上是**负的**。
⇒ 这条结论应读作"**倾向于**在低/中深度上更有效"，而不是论文字面的强断言。

结构任务侧的同类观察更干净：BFVD 病毒蛋白上，**收益随可用同源数量增加而饱和**（论文 Fig. 5c），
方向与"TTT 对 OOD / 少同源蛋白最有用"一致。

---

## 小结：三条最值得记住的

1. **同一套增强（1024 裁剪 + 15% + 80/10/10）跨任务不变，但 loss 分叉：**
   structure 用只算 mask 的 `cross_entropy`，fitness 用**完全不看 mask** 的
   `unnormalized_cross_entropy`（~85% 梯度在抄写可见残基）。
2. **structure 的大幅优势主要来自评测集（只挑难靶点）与 pLDDT 提供的 best-step 下界保护**，
   而不只是机制本身更适合结构任务。fitness 侧没有置信度，只能取最后一步。
3. **fitness 上真正的天花板是进化信息**：ESM2 单序列 TTT +0.0014，换成 MSA-TTT +0.0160，差 11 倍。
