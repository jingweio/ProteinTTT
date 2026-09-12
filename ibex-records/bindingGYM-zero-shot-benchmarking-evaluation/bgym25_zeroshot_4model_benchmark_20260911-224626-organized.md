# BindingGYM zero-shot benchmark：四个 inverse-folding 模型的 readout 改造与评测

> 整理版 · 2026-09-12 22:20 · status: 3/4 模型已定论，StaB-ddG 进行中
> 过程稿（含逐个 bug 的排查时间线）：[`bgym25_zeroshot_4model_benchmark_20260911-224626.md`](./bgym25_zeroshot_4model_benchmark_20260911-224626.md)
> **本篇按逻辑组织，不按排查顺序。** 想看"当时是怎么一步步查出来的"请看过程稿。

---

## 1. 立题

**研究问题**：在 protein–protein binding 的 DMS 预测上，
**更强的结构表征（侧链、全原子、配体感知）或更贴近热力学的目标（ddG），
能否超过 backbone-only 的 ProteinMPNN？**

**锚点**：已复现的 pretrained ProteinMPNN on BindingGYM zero-shot
（25 assay，Spearman 0.3914，官方发布 0.3970）。四个候选模型全部在**同一 25 个 assay、
同一指标代码**下评测，与该锚点直接可比。

**数据**：BindingGYM 25 assay / 376,446 variants。关键结构性事实——
它**不是单点突变库**：只有 4 个 assay 是 ≥99% 单点，其余是**极少位点上的组合库**
（`Z-domain_ZSPA-1_LL1`：45,476 个 variant 只动 **9 个位点**，0% 单点）。
这一点直接决定了打分口径能怎么设计（见 §4）。

---

## 2. 核心难点：这不是"跑四个模型"，而是"给四个模型各造一个 readout"

四个模型**没有一个**能直接产出 BindingGYM 要的东西。它们的原生输出形态差异如下：

| 模型 | 原生输出 | 能直接当 DMS 分数用吗 |
|---|---|---|
| **ProteinMPNN**（参照） | AR log-probs → `_scores()` | ✅ 官方 harness 已实现 |
| **LigandMPNN** | 两个打分入口（AR / 逐位），但语义与文档相反 | ⚠️ 要选对入口 + 修口径 |
| **LASErMPNN** | 有打分函数但 **repo 内无人调用**，无 CLI | ⚠️ harness 全自写 |
| **ADFLIP** | **没有 likelihood head**，原生只有 sampling | ⚠️ 要造 readout |
| **StaB-ddG (stage2)** | 与 ProteinMPNN **同架构**，只是权重不同 | ✅ **直接沿用 ProteinMPNN 的 readout**（见 §5.4） |

⇒ **本工作的主体是 readout 改造**，评测本身反而是最轻的一环。下面逐个交代。

---

## 3. 参照 readout：ProteinMPNN 是怎么打分的

后面所有改造都以此为基准，先钉死它。

```python
log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)

def _scores(S, log_probs, mask):                    # protein_mpnn_utils.py:39-47
    loss = NLLLoss(reduction='none')(log_probs.view(-1,21), S.view(-1)).view(S.shape)
    return torch.sum(loss * mask, dim=-1)           # ← 长度归一在官方代码里是【注释掉的】

design_score = -mean_over_M(_scores(S, log_probs, mask*chain_M*chain_M_pos))
global_score = -mean_over_M(_scores(S, log_probs, mask))
```

四个要点，改造时必须逐条对齐：

1. **teacher-forced**：把 mutant 序列灌进 `S`，读它自己的 log-prob，不做采样。
2. **未归一的 NLL 求和** —— 不是 per-residue 平均。"顺手修正成均值"会改变口径。
3. **两个 mask 范围**：`design_score` 只算被设计的链，`global_score` 算全部。
4. **`randn_1`（解码序噪声）每个 POI 抽一次、组内所有 variant 共享**；M=5 个解码序取平均。
   per-assay 的解码序噪声 σ 中位数 0.0184、最大 0.0581 —— **任何小于 0.02 的差异都不可解读**。

---

## 4. 打分口径分三族 —— 四个模型无法共用一个口径

| 口径 | 定义 | 谁能出 |
|---|---|---|
| **A. AR-NLL** | autoregressive teacher-forced，BindingGYM 官方口径 | ProteinMPNN / LigandMPNN / LASErMPNN |
| **B. joint-masked** | mask 掉该 variant 的**整组**突变位点，条件于其余全部；`Σ_j[log p(mt_j) − log p(wt_j)]` | **四个模型唯一共同口径**（ADFLIP 只能走这条） |
| ~~C. binding ddG~~ | ~~物理量~~ | ~~StaB-ddG~~ —— **已放弃**，见 §5.4 |

**为什么必须有 B**：ADFLIP 是 flow matching，出不了 AR-NLL。没有 B，它就无法与任何模型比较。

**为什么 B 必须是 joint 而不是逐位相加**：BindingGYM 多为组合库，逐位可加的 marginal 会让
`Z-domain_ZSPA-1_LL1` 退化成 9 个固定项的和，完全丢失 variant 间的区分度。

**B 的成本按位点组合算，不按 variant 算**：组合库内所有 variant 的 masked 输入**完全相同**
（masked 位的 token 与侧链都被移除，与替换成哪个 AA 无关）⇒
**32,977 次 forward 覆盖全部 376,446 个 variant**，实测比 A 快约 8×。

---

## 5. 逐模型：readout 差异与改法

### 5.1 LigandMPNN —— 入口选对了，但语义和门控都有坑

**与 ProteinMPNN 的差异**

| # | 差异 | 后果 |
|---|---|---|
| 1 | 提供两个入口：`model.score()`（AR）与 `model.single_aa_score()`（逐位） | 选错口径 |
| 2 | `single_aa_score` 内部 `for idx in range(L)`，**每个位点各跑一次完整 decoder** | L=1041 时逐 variant 调用完全不可行 |
| 3 | **`use_sequence` 的语义与其 CLI help 相反** | 拿到 backbone-only 分数却以为是序列条件的 |
| 4 | 侧链 context 被 `chain_mask` **门控** | 默认设置下该 flag 恒等于没传 |

**差异 3 的实测判据**（不靠读代码定案，写了判别探针：扰动其余位点，看目标位点 log-prob 变不变）：

```
use_sequence=True   max|Δlog_probs| = 0.000e+00   ⇒ 只看 backbone
use_sequence=False  max|Δlog_probs| = 9.271e-01   ⇒ 条件于其余序列
```
而其 CLI help 写的是「1 = 用氨基酸序列信息，0 = 只用 backbone」——**方向相反**。

**差异 4 的根因**（`model_utils.py:1252`）：
```python
xyz_37_m = xyz_37_m * (1 - chain_mask)     # 侧链只对 chain_mask==0(fixed) 的残基生效
```
BindingGYM 的 `chain_id` 把**所有链都列为 designed** ⇒ `chain_mask` 全 1 ⇒ 侧链 context **恒为零**。

**我们的 readout 改法**

- **口径 A**：直接用 `model.score(fd, use_sequence=True)` 取 log_probs，再**逐字复刻** `_scores`（§3）。
- **口径 B**：自写 `joint_masked_logprobs()` —— 不用 `single_aa_score`（差异 2 不可行），
  而是复刻它的解码序构造，把**整组**突变位点的 `order_mask` 一起置 1
  ⇒ 这组位点**最后**解码 ⇒ 一次 decoder pass 同时拿到 k 个位点的 log-prob，
  各自条件于其余位点保持 WT。方向由上面的探针实测定案（写反就会变成 backbone-only）。
- **侧链轴**：新增 `--designed_chains mutated`，把**从不被突变的 partner 链设为 fixed**，
  侧链 context 才有作用对象。统计：**21/25 个 assay** 可这么做（4 个 Z-domain 两条链都突变，做不了）。

**验证**：口径 A 与 anchor 的逐 variant 分数对照，3 个 assay 中 **2 个 `rho = 1.0000`**（逐位复现），
第 3 个 0.9889（该 assay 有 7 残基缺口，补洞方式影响邻居图），`rho_DMS` 仅差 0.0003。

---

### 5.2 LASErMPNN —— 打分函数存在但没人用过

**与 ProteinMPNN 的差异**

| # | 差异 | 后果 |
|---|---|---|
| 1 | `utils/model.py:261 get_logits_for_score()` 存在，但 **repo 内零调用、无 CLI** | harness 全自写，无参照实现 |
| 2 | teacher-forcing 的输入是 **sequence + chi angles** 两路 | mutant 没有 chi，需决定怎么喂 |
| 3 | **AA 索引是三字母码字母序**（`ALA=0, ARG=1, ASN=2, ASP=3, CYS=4…`） | 与 ProteinMPNN 的 `ACDEFGHIKLMNPQRSTVWYX` **完全不同**，用错即静默错位 |
| 4 | 返回 7 元组，`sequence_logits` 是第 0 个 | — |

**差异 2 的处理与其代价**：位点 i 的 logits 只依赖解码序在它**之前**位点的 seq+chi，
**位点 i 自身的 chi 不参与** ⇒ **单点突变无影响**。
多点突变时，先解码的突变位点会带着**WT 的 chi** 配 **mutant 的 aa**，
这是结构-only 输入的固有限制，**如实记录、不掩盖**。

**我们的 readout 改法**
- 口径 A：`get_logits_for_score(batch, decoding_order, S_mutant, chi_WT)[0]` → log_softmax →
  按 §3 的未归一 NLL 求和，M=5 个随机解码序取平均。
- 口径 B：同 LigandMPNN 的 joint-masked 构造（整组突变位点最后解码）。
- 索引一律走它自己的 `aa_short_to_idx`，并在脚本启动时断言
  `aa_short_to_idx['A']==0 and ['R']==1 and ['C']==4`，顺序变了就直接退出。
- BindingGYM 无小分子 ⇒ 照 `run_inference` 的 `--ignore_ligand` 路径清空 ligand 张量。

---

### 5.3 ADFLIP —— 没有 likelihood head，但**不需要训练新 head**

这是本工作在 readout 上最实质的一处改造，也**推翻了立项时的预设**。

**原始预设**：ADFLIP 是 discrete flow matching，原生只做 sampling，
所以"需要给它加一个 head 来预测 structure–seq compatibility likelihood"。

**实际结论：不需要新参数、不需要训练。** 读源码后发现两件事：

```python
# model/discrete_flow_aa.py
logits, _ = self.model(noisy_data, times)      # denoiser 【已经】在 center 位输出 AA logits

# corrupt_data_by_sample() 的 keep_mask 逻辑：
#   mask 掉一个残基时，会【同步移除该残基的侧链原子】，其余残基保留身份 + 全部侧链坐标
```

⇒ 把该 variant 的整组突变位点置 `<MASK>`、其余喂 WT 全原子，**一次 forward** 得到的就是
`p(aa_j | all-atom structure, rest-of-sequence)` —— **正是所需的 compatibility likelihood**。

**所以"output head 的改造"实为一个打分协议 wrapper**：
```
samples = WT tokens, 突变位点置 <MASK>
→ corrupt_data_by_sample(data, t, samples)      # 同步剥掉这些位点的侧链
→ model.model(noisy, t) → logits[center 位]
→ score = Σ_j [ log p(mt_j) − log p(wt_j) ]
```
这与其余三个模型的口径 B **严格同构**（都是"mask 掉这组位点、条件于其余全部"），
四个模型因此真正可比。

**另外三处与 ProteinMPNN 的差异**

| # | 差异 | 处理 |
|---|---|---|
| 1 | **输出词表 33**（`<PAD>`/`<MASK>`/20 AA/`<UNK>`/核苷酸），不是 21 | `output_dim` 必须从 ckpt 的 config 取；漏传会退到默认 20 并报 shape mismatch |
| 2 | 多一个**时间步 t** 输入 | joint-masked 时只有极少位点被 mask ⇒ 取 t≈1 |
| 3 | ckpt 用定义在 `__main__` 的 `Config` 类 pickle | 打分脚本里需原样定义该类才能反序列化 |

---

### 5.4 StaB-ddG（stage 2）—— 不改 readout，只换权重

**为什么用 stage 2**：`model_ckpts/` 三个 ckpt 对应三阶段
`proteinmpnn.pt`(stage1) → **`stability_finetuned.pt`(stage2, Megascale 稳定性)** →
`stabddg.pt`(stage3, SKEMPI binding)。
**stage3 在 SKEMPI 上 finetune 过，而 SKEMPI 与 BindingGYM 的复合物有重叠 ⇒ 用它就是 leakage。**
防线两道：同步到 ibex 时**物理排除** `output/`（装着全部 stage3 ckpt）；脚本里**断言 ckpt md5**。

#### 关键判断：stage 2 没有改网络结构，所以 readout 应与 ProteinMPNN 完全一致

StaB-ddG 的 binding ddG 是一个**推理期的热力循环构造**
（complex 的 ddG 减去两个 binder 各自的 ddG），不是模型自带的输出头。
**stage 2 只是在 Megascale 稳定性数据上 finetune 了 ProteinMPNN 的权重。** 实测验证：

```
stability_finetuned.pt vs proteinmpnn_v_48_020.pt
  参数名 118/118 完全一致，形状零不符          ⇒ 架构字面相同
  权重最大逐元素差 9.087                      ⇒ 确实 finetune 过，不是同一份权重
```

⇒ **放弃热力循环 readout，改用与 ProteinMPNN 完全相同的 readout（§3）。**
这样唯一变量只剩**权重本身**，构成一个干净的受控对照，回答的是：
> 「在 Megascale **稳定性**数据上 finetune，对 **binding** DMS 预测有没有帮助？」

热力循环 readout 会把「模型能力」与「ddG 分解方式」两件事混在一起，
且引入 chain partition、20-member ensemble、符号约定等一堆与本问题无关的自由度。

#### 对照设计：必须是 stage1 → stage2，不是 anchor → stage2

查证发现 **StaB-ddG 的 base 并不是我们的 anchor 权重**：
其 `proteinmpnn.pt`（stage1）与 `v_48_020` 的 md5 不同（`698982b1…` vs `91d54c97…`），
逐元素也有差。所以直接拿 anchor 和 stage2 比，会把「换了个 base checkpoint」
和「Megascale finetune」两个效应混在一起。

**三点对照**（全部同 readout、同口径、同 25 assay）：

| run | ckpt | 角色 |
|---|---|---|
| `proteinmpnn_*` | `v_48_020`（md5 `91d54c97…`） | benchmark 参照 |
| `stabddg_s1_*` | StaB-ddG 的 `proteinmpnn.pt`（`698982b1…`） | 控制「base 不同」 |
| `stabddg_s2_*` | `stability_finetuned.pt`（`ed2645a2…`） | 变量：Megascale finetune |

⇒ **Δ(stage2 − stage1) = Megascale finetune 的纯效应。**

#### 实现上唯一的差异
stage2 的 ckpt 是**裸 state_dict**，没有 `num_edges` / `noise_level` 元数据
（stage1 有，为 48 / 0.2）。打分脚本已支持裸 state_dict，并要求显式传 `--k_neighbors 48`；
ckpt 自带元数据时则做交叉校验，不一致即退出。

> **已放弃的路线（留档）**：先前按热力循环 readout 跑到 10/25，
> 产物保留在 `stabddg_s2/`（不删除），但**不进入任何结论**。
> 放弃它的理由见上：stage 2 未改结构，同 readout 才是受控对照。

---

## 6. 一个贯穿四个模型的横切问题：坐标系

readout 只有落在**正确的残基位置**上才有意义。实测发现：
**四个 repo 的 PDB parser 对"哪些残基算数"各有各的规矩，没有一个与 BindingGYM 的序列约定天然对齐。**
共遇到六类：

| # | 现象 | 出处 |
|---|---|---|
| 1 | occupancy 全 0 ⇒ `select("occupancy>0")` 返回 `None` | LigandMPNN（BindingGYM 22 个结构 occ 全为 0.00） |
| 2 | 官方按残基编号**补洞**，vendor 只返回观测残基 | LigandMPNN vs BindingGYM（`3KZ0` 链A 143 vs 150） |
| 3 | Kabat **insertion code** 与缺口并存 ⇒ 编号跨度 ≠ 序列长 | `4ZFF`/`4ZFG` 抗体重链（跨度 212 vs 序列 219） |
| 4 | 丢掉 backbone 不完整的残基 | LASErMPNN（`BH3_Bcl-xL` 180 vs 229）、ADFLIP（`4D5_HER2` 1015 vs 1041） |
| 5 | 链的**解析顺序**与 BindingGYM 的 `chain_id` 顺序无关 | LASErMPNN（`8BE4` 解析为 S→R，而 chain_id 是 `RS`）、ADFLIP（`5A12` 指派 `(1,2,0)`） |
| 6 | gap 字符约定不同（`'-'` vs `'X'`） | StaB-ddG |

**统一对策**：
> 解析 → **逐链** Needleman-Wunsch 比对到 BindingGYM 参考序列 → 未映射位不参与打分
> → 对比对后的 WT 做**零错配断言**。

必须**逐链**做：拼接后整体比对会在两条**同源**链之间错配
（`Z-domain_ZSPA-1_1LP1` 的链 A/B 是同源的 Z-SPA-1 与 Z-domain）。
链字母必须取自各 repo 自己的元数据（LASErMPNN 用 `residue_identifiers[i].chain_id`；
ADFLIP 的字母在 `pdb2data` 里丢失，只剩数值 `chain_id` ⇒ 用**链块排列匹配**取错配最少的指派）。

⚠️ **这六类全都不会报错、数字也完全合理，只有硬断言能拦住。**

---

## 7. 结果

### 7.1 口径 B —— 四模型同口径、同 25 assay

| 排名 | 模型 | Spearman | AUC | MCC | NDCG |
|---|---|---|---|---|---|
| 1 | **ProteinMPNN**（backbone-only） | **0.3899** | 0.6854 | 0.1552 | 0.7197 |
| 2 | LigandMPNN（+侧链 ctx） | 0.3771 | 0.6824 | 0.1492 | 0.7189 |
| 3 | LASErMPNN（全原子 + rotamer） | 0.3757 | 0.6768 | 0.1547 | **0.7263** |
| 4 | ADFLIP（全原子 flow matching） | 0.3630 | 0.6719 | 0.1372 | 0.7028 |

**全部配对检验（Wilcoxon，同 25 assay）** —— 唯一显著的只有一项：

| 对比 | Δ | 正号 | p |
|---|---|---|---|
| ProteinMPNN vs **ADFLIP** | +0.0269 | 15/25 | **0.0275** ✅ |
| ProteinMPNN vs LASErMPNN | +0.0142 | 11/25 | 0.6915 |
| ProteinMPNN vs LigandMPNN | +0.0129 | 12/25 | 0.3254 |
| LigandMPNN vs LASErMPNN | +0.0013 | 11/25 | 0.5602 |
| LASErMPNN vs ADFLIP | +0.0127 | 11/25 | 0.5782 |

### 7.2 口径 A（AR-NLL）

| 模型 | Spearman | 备注 |
|---|---|---|
| ProteinMPNN | 0.3899 | 对 anchor 0.3914 差 −0.0015（σ 内） |
| LigandMPNN | 0.3883 | vs ProteinMPNN：Δ=+0.0016，17/25，p=0.3388（不显著） |
| LASErMPNN | — | **6/25，未完成**（全量约 16 h，见 §8.2） |

### 7.3 侧链轴（partner 链设为 fixed）

| | Spearman | AUC | MCC | NDCG | AP |
|---|---|---|---|---|---|
| `ligandmpnn_ar_scfix`（+侧链） | **0.3896** | 0.6884 | 0.1624 | 0.7263 | 0.2235 |
| `ligandmpnn_ar_noscfix`（−侧链） | 0.3827 | 0.6843 | 0.1555 | 0.7197 | 0.2208 |

限定在 **21 个确实有 fixed partner 链**的 assay 上配对检验：
```
+侧链 0.4081   −侧链 0.3999   Δ = +0.0082   15/21   Wilcoxon p = 0.0158
```
✅ **内部对照**：4 个两条链都被突变、没有 fixed 链的 Z-domain assay，
**Δ = +0.000000（逐位精确相同）** —— 从实验上反证了 §5.1 差异 4 的门控机制。

---

## 8. 科学结论

### 8.1 主结论（否定）

> **更强的结构表征（侧链、全原子、配体感知）在 BindingGYM 的 protein–protein binding DMS 上
> 不能超过 backbone-only 的 ProteinMPNN。**

三个"更高级"的模型全部与 ProteinMPNN **打平或更差**；唯一达到统计显著的差异是
**ProteinMPNN > ADFLIP（p = 0.0275）**，方向仍然不利于全原子模型。

### 8.2 唯一有效的增量来自别处

**让模型看到 partner 的侧链**有显著增益（§7.3：+0.0082，p = 0.0158，五项指标全为正，
且有 4 个零差异的内部对照）。

⇒ **归纳：在 binding DMS 上起作用的不是「模型把自己看得更细」，而是「模型看得到结合伙伴」。**

注意这两件事**不是一回事**：LigandMPNN/LASErMPNN/ADFLIP 的"更细"是**对整个复合物**建模得更细，
而 §7.3 的增益特指**把 partner 作为固定上下文喂进去**。前者无效，后者有效。

**这条归纳还缺第三类证据。** StaB-ddG 是四个模型里**唯一显式建模 binding** 的
（它算的就是 complex 与两个 binder 之间的 ddG 差）。
- 若它显著更好 ⇒ 归纳得到最强支持；
- 若不然 ⇒ 需要重新解释（例如"partner 增益来自上下文而非 binding 目标本身"）。

---

## 9. Caveats（必须与结论一起读）

1. **LASErMPNN 的 AR 口径未完成**（6/25）。§7.1 用的是口径 B。
   若 AR 给出不同排序，将**两个口径都报**，不挑一个。
2. **口径 A 与 B 均值等价但不是同一套分数**：配对差 `mean Δ=+0.0000, p=0.874`，
   但 `sd = 0.0446 ≈ 2.4×` 解码序噪声 σ。**是均值相消，不是逐 assay 一致** ——
   对单个 assay 下结论必须注明口径。
3. **逐 assay 差异含 label 噪声**：`GB1_IgG-Fc` 的两个数据版本在 ADFLIP 上方向**相反**
   （+0.0702 vs −0.1465），同一复合物、同一 partition。**不要对单个 assay 的排序下强结论。**
4. **解码序噪声的量级**：per-assay σ 中位数 0.0184、最大 0.0581
   （`Z-domain_ZpA963_HL1` 仅因解码顺序就在 0.065–0.203 之间摆动）。
   **任何小于 ~0.02 的 per-assay 差异都不可解读。**
5. **侧链轴的口径不同**：`scfix/noscfix` 的 `chain_mask` 与 anchor 不同
   （解码序与 `design_score` 的打分范围都变了）⇒ **只能与自己的对照比，不能并进 §7.1 主表。**

---

## 10. 可复现信息

**数据有效性三查**（每个 run 报数前都跑）：
`code_stamp` 单一值（确认全部由同一版代码产出）· 零退化 assay（`nunique>1`、`std>0`）· 逐 assay 无 `NaN`。

| run | 覆盖 | code_stamp |
|---|---|---|
| `proteinmpnn_ar` / `_mm` | 25/25 | ✅ |
| `ligandmpnn_ar_nosc` / `_sc` / `_mm_sc` | 25/25 | ✅ |
| `ligandmpnn_ar_scfix` / `_noscfix` | 25/25 | ✅ |
| `lasermpnn_jm` | 25/25 | `0ee856487461` |
| `adflip_jm` | 25/25 | `ece43ce4b11a` |
| `lasermpnn_ar` | 6/25 | 进行中 |
| `stabddg_s2` | 10/25 | 进行中 |

**打分脚本**（`ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/sh/`）
| 文件 | 作用 |
|---|---|
| `score_bgym_mpnn.py` | ProteinMPNN / LigandMPNN，口径 A + B |
| `score_bgym_lasermpnn.py` | LASErMPNN，口径 A + B |
| `score_bgym_adflip.py` | ADFLIP，口径 B |
| `prep_stabddg_inputs.py` | StaB-ddG 的输入构造（partition / 编号转换） |
| `probe_use_sequence.py` | `use_sequence` 语义的实验判别探针 |
| `derive_chain_partition.py` | 接触图推导 chain partition（产物已冻结） |
| `aggregate_bgym.py` | 汇总，**逐字复用**官方 `bindinggym_metrics_one_assay` |

**产物**
- per-variant 分数：`/ibex/user/guoj0f/bindingGYM-zs-benchmark/scores/{run_id}/{DMS_id}.csv`
  （保留原 DMS csv 全部列与行序，不过滤；附 `design_score`/`laser_score`/`adflip_score` + `seed` + `run_id` + `code_stamp`）
- StaB-ddG：`/ibex/user/guoj0f/bindingGYM-zs-benchmark/stabddg_s2/{DMS_id}/output/`
- 逐 assay 指标 + leaderboard（已回流进 repo）：`ibex-records/{project}/results/`
- 作废留证（**不删除**）：`scores/lasermpnn_*_STALE_prefix_*`、`stabddg_s2_OLD_wrong_numbering_*`

**冻结产物**：`refs/chain_partition.tsv`（md5 `068dae9cc3338368f0e89ebe39e752f4`，
生成环境 python 3.10.20 / numpy 2.2.6 / pandas 2.3.0），运行时只读不重算。

---

## 11. 关联
- 过程稿（逐个 bug 的排查时间线、env 构建、job 记录）：
  [`bgym25_zeroshot_4model_benchmark_20260911-224626.md`](./bgym25_zeroshot_4model_benchmark_20260911-224626.md)
  —— 其中 **§21** 记录了一次"幂等跳过未校验代码版本 ⇒ 新旧结果混合"的事故，方法论价值较高。
- 锚点实验：[`workstation-records/BindingGYM-zero-shot-proteinMPNN/zeroshot_proteinmpnn_20260827-154500.md`](../../workstation-records/BindingGYM-zero-shot-proteinMPNN/zeroshot_proteinmpnn_20260827-154500.md)
