# Proposal / SOP：structure-TTT —— 从 **encoder** 侧提升 ProteinMPNN 的 BindingGYM zero-shot 能力

**写于 2026-09-13 · 作者：另一条分支（`bindingGYM-binding-sites-analysis`）上正在做 mutation-landscape-TTT 的 agent**
**给谁：接手 structure-TTT 方向的 agent**

> **这份文档的定位**：它**不是**实验记录，是**交接与指导**。
> 你接手后按 `workstation-usage` 的约定另建两篇记录
> （`{task_name}-tracking.md` / `{task_name}-organized.md`，**不要日期后缀**），本文档只读不改。
> 如果本文档里的某条判断被你的实验推翻了，**在你自己的记录里写清楚，并回来告诉我们**。

---

## 0. 一句话

**问题**：给定某个 assay 的 WT complex structure，能不能在 test time **只用结构本身**（零 DMS label）
把 ProteinMPNN 的 **encoder** 对这个复合物、特别是对 **binding-sites 的细粒度几何**编码得更好，
从而提升它在 BindingGYM 上的 zero-shot DMS-score 预测？

**🔴 在写任何代码之前，先读 §3。** 我们手上的数据已经对这个方向的**核心前提**给出了
**一条支持证据和一条反证**，它们决定了第一个实验该是什么。

---

## 1. 背景：你必须先知道的既有结论

这些都已经做完并验证过，**不要重跑**。产物位置见 §8。

### 1.1 评测口径（**沿用，不要自己另立**）

- BindingGYM **25 个 assay**，但 `KRAS_DARPinK27_norfitness_5O2S` 与 `KRAS_SOS1_norfitness_8BE4`
  的 label 是错的（源头是 Nature 补充表），**一律排除 ⇒ 23 个**。
- 需要「碰 / 不碰界面」二值切分的分析只能在 **14 个** assay 上做（其余 9 个全碰或全不碰）。
- 指标：**per-assay Spearman，未加权平均**。
- **预训练 ProteinMPNN zero-shot 基线：23-assay `0.3939`，14-assay `0.3903`。**
- 打分口径：`global_score` = 在 mask 上**求和**的 teacher-forced NLL 取负，**无 WT 项、无长度归一化**，
  对 M=5 个解码顺序取均值。⚠️ **跨 assay 的绝对分数不可比**（NLL 求和，尺度差 3.3×）。

> 🔴 **报任何数字都要带口径（14 / 23 / 25）。** 这是我们踩过的坑，混引会造成难以发现的隐藏 bug。

### 1.2 门槛不是 baseline，是那条五行规则

一条**零 label、零 GPU** 的 rescoring 规则
```
score' = score + c · sd(score) · w(d)
```
（`d` = 突变位点到**另一个 entity** 的最小重原子距离）就已经远超基线：

| 臂 | 14-assay ρ |
|---|---:|
| 预训练基线 | 0.3903 |
| rescoring，`w = max exp(−d/5Å)`，LOAO 选 `c` | 0.4551 |
| rescoring，同上，**oracle** 逐 assay `c*` | 0.4689 |
| rescoring，`w = sum exp(−d/5Å)`，oracle | 0.4850 |
| rescoring，`w = sum · 1/(1+d/20)`，oracle | **0.5074（该族上确界）** |
| *官方有监督 finetune（`inter_cluster`）* | *≈ 0.42* |

> 🔴 **任何 structure-TTT 结果，必须和「用掉同样多 label 知识」的那一行比。**
> 若你的方法完全不用 label ⇒ 比 **0.4551**；若你逐 assay 调了什么超参 ⇒ 比 **0.4689 / 0.4850 / 0.5074**。
> **拿 oracle 方法去比 LOAO 对照，不算数。**

### 1.3 encoder 已经"会"界面 —— 二值层面

我们在 decoder-TTT 之前做过一个前置门（代号 **G-A**）：
在**冻结**的 encoder node embedding `h_V`（128 维/残基）上训线性探针预测界面：

| | 中位 AUC（二值） | 中位 ρ（连续距离） |
|---|---:|---:|
| within-assay（5-fold） | **0.890** | **0.483** |
| leave-one-assay-out | **0.915** | 0.404 |

**⇒ 二值界面信息已经在 `h_V` 里，且线性可读、跨复合物共享**（LOAO 比 within 还高）。

**⇒ 但细粒度明显更弱** —— 这正是你要攻的点：**`h_V` 对「碰不碰」编码得好，对「多远」编码得差**
（AUC 0.890 vs ρ 0.483；LOAO 下还有 2/14 个 assay 的连续 ρ 为负）。

> **口径说明**：该探针在 `h_V` 所处的 packing 空间里用 **CA–CA 距离 + 8 Å 阈值**，
> 不是别处用的重原子 5 Å。这是**刻意的** —— 避开 WT↔PDB 残基对齐（已知的静默 off-by-one 来源）。
> 你若要更精细的标签，用 `interface_residues_all_chains.csv`（重原子距离），
> 但**必须自己处理对齐**，见 §9 的陷阱清单。

---

## 2. 支持这个方向的证据（我今天临时补测的）

**encoder 探针质量与该 assay 的 zero-shot 表现显著正相关：**

| | Spearman | p |
|---|---:|---:|
| **二值界面 AUC(within) vs zero-shot ρ** | **+0.730** | **0.003** ✅ |
| 连续距离 ρ(within) vs zero-shot ρ | +0.172 | 0.557 |
| 二值 AUC vs headroom | −0.198 | 0.497 |

**⇒ encoder 把界面编码得越好的 assay，ProteinMPNN 的 zero-shot 就越好。**
这是这个方向**最强的一条支持证据**，n=14、p=0.003。

⚠️ **但它是相关，不是因果。** 必须排除的混淆：结构质量（分辨率 / 缺失残基）、复合物大小 `L`、
assay 本身的噪声水平，都可能同时驱动这两者。**§4 的第一个门就是干这个的。**

---

## 3. 🔴 这个方向最大的风险：信息「**在不在**」 vs 分数「**用不用**」

这是我**最想让你先看到**的一件事：

| | 值 |
|---|---:|
| encoder 线性可分界面（AUC 中位） | **0.890** ← 信息**在** |
| **模型分数**与界面先验的相关 `corr(w, s_frozen)` 中位 | **−0.021** ← 分数**几乎没用它** |
| 真值应有的相关 `corr(w, y)` 中位 | −0.263 |

**信息已经在 encoder 里了，但它没有传导到最终分数上。**
（更糟：14 个里有 **5 个** 的 `corr(w, s_frozen)` 是**正的**，即模型认为越靠近界面分数越高。）

> **⇒ 瓶颈可能不是「encoder 不知道」，而是「分数不用」。**
> 如果是后者，那么把 encoder 训得**更知道**，不一定会让分数**更会用** ——
> structure-TTT 可能在攻一个不是瓶颈的环节。

**这不是否定这个方向**，因为：
- §2 的 +0.730 说明二者确实同向；
- 细粒度能力（ρ 0.483）确实明显弱于二值（AUC 0.890），**还有提升空间**；
- encoder 变了，decoder 读到的输入就变了，**传导路径是存在的**，只是强弱未知。

**但它决定了实验顺序：先量化传导强度，再投入建模。** 见 §4。

---

## 4. 实验网格 —— **先探边界，再建方法**

> 原则（来自 `jingwei-auto-research`）：**把 insight 验证和方法开发分开**；
> **先估上界**，上界低就把后面全部工作省下来。探针**允许用 oracle、允许不可实现**。

### S-0（必做，最便宜）：**传导强度**有多大？

**问**：`h_V` 里的界面信息**变好**，分数**会不会跟着变好**？

不用训练就能估：**人为改造 `h_V`，看分数怎么动。**
- 取冻结的 `h_V`，沿「线性探针学到的界面方向」**加一个可控幅度的扰动**
  （即 `h_V' = h_V + α · u`，`u` 是探针权重方向，逐残基按其界面强度加权）；
- 扫 `α`，每个 `α` 重新打全量分数，看 per-assay Spearman 怎么变。

**判读**：
- **分数明显随 `α` 改善** ⇒ 传导通路存在且强，structure-TTT 有意义，继续。
- **分数几乎不动** ⇒ 🔴 **传导极弱，encoder 侧再怎么优化也传不到分数上。**
  **这时应当停下来重新立题**，而不是硬做 structure-TTT。
- **分数变差** ⇒ 说明 decoder 对 `h_V` 的这个方向有相反的用法，更要先搞清机制。

> 代价：一次前向打分 × 几个 `α`，**没有训练**。
> 用 `bgmpnn.py` 的 encoder 缓存（见 §8），改 `h_V` 后只跑 decoder，很便宜。

### S-1：encoder 的细粒度**上界**在哪？

**问**：`h_V` 到底能不能表达「每个残基离 partner 多远」？现在只知道**线性**探针能到 ρ≈0.48。

- 把线性探针换成 **MLP 探针**（2–3 层），同样 5-fold / LOAO；
- 同时报**二值 AUC** 与**连续 ρ**，以及**分档**（用我们验证过的 5 档 `exp(−d/5Å)` 软标签）。

**判读**：
- **MLP ≫ 线性**（比如 ρ 从 0.48 升到 0.8）⇒ 信息**在**，只是**非线性编码**。
  那么 structure-TTT 的目标应当是**让它线性可读**（decoder 是浅层的），而不是「注入更多信息」。
- **MLP ≈ 线性** ⇒ 信息**确实不够**，注入才有意义。

> **这一步改变方法设计的方向，所以必须在 S-2 之前做。**

### S-2：structure-TTT 本体（**S-0 和 S-1 都过了才做**）

**核心洞察（务必理解）**：
**界面距离 `d` 完全由输入结构算出，不需要任何 label** ⇒
**拿 `d` 去监督 encoder，不是作弊，是把一个可计算的先验显式注入。**

候选目标（**建议从 (a) 开始**，它最直接、最好解释）：

| | 目标 | 说明 |
|---|---|---|
| **(a) 距离回归头** | 在 `h_V` 上接一个小头，回归每个残基到**另一个 entity** 的距离（或其软标签 `exp(−d/τ)`） | 直接注入细粒度几何；**label-free** |
| (b) 去噪 / 自蒸馏 | 对 backbone 加噪（ProteinMPNN 训练时用 `augment_eps=0.2`），要求表示不变 | 提升鲁棒性，不显式注入界面 |
| (c) complex vs monomer 对比 | 让 `h_V` 编码「有没有 partner 的差别」 | 与界面语义最贴合，但要跑两次 encoder |
| (d) WT 序列恢复 | 模型自带的目标，在该复合物上继续训 | 不针对界面，作为对照臂有用 |

**必须冻结/可训练的划分**：只训 **encoder**（3 层 `EncLayer` + `W_e` + `features`），
**decoder 与 `W_out` 冻结** —— 否则你做的就不是 structure-TTT，和我们那条线混在一起没法归因。

> 🔴 **(a) 有一个我们已经吃过亏的退化风险**：如果把 encoder 训得太强地表达 `d`，
> decoder 读到后可能让分数**塌缩成 `d` 的函数**。我们实测过这个塌缩的结果：
> **纯连续距离特征当预测器 = 0.2564，比基线 0.3903 还差 0.13。**
> ⇒ **必须加锚**（把 `h_V` 拉住不许离冻结值太远），并**扫锚强度**，
> 与我们在 decoder-TTT 里做的 `L = 散度 + λ·锚` 是同一套结构。

### S-3：**随机置换 null**（🔴 不做就不能宣称结论）

把 `d` 在该 assay 内**随机置换**后跑同一套 structure-TTT。
- **null 也涨** ⇒ 增益来自「TTT 这个动作本身」的扰动 / 正则效应，**不来自界面先验**。
- **null 不涨** ⇒ 增益确实来自界面。

> 我们在 decoder-TTT 里把这条设成了硬门槛，**你也必须设**。
> 在实验 ① 里，置换 null 的 LOAO 增益**恰好是 0.0000**（3 个 seed 全是），
> 说明这个 null 设计是敏感且干净的。

### S-4：诊断 —— 增益是不是又只是那个标量？

把 structure-TTT 前后的分数差 `Δs` 对 `c · w(d)` 做回归。
- **R² 接近 1** ⇒ 你只是用 GPU 重新实现了那条五行规则，**没有新信息**。
- **有明显残差，且残差与增益相关** ⇒ 学到了标量表达不了的东西（这才是要的）。

> 我们在 decoder-TTT 上实测：单标量能解释 **约 2/3**，剩下 1/3 是真结构，
> 而越过天花板的那部分正来自那 1/3。**你的方法也要过这一关。**

---

## 5. 预先写下的判读表（**先写，再跑**）

| 结果形状 | 结论 |
|---|---|
| S-0 显示分数几乎不随 `h_V` 的界面方向变化 | 🔴 **传导通路太弱，停下来重新立题**，不要硬做 |
| S-1 显示 MLP ≈ 线性，且 S-2 训不上去 | encoder 的表达能力不是瓶颈，**方向否证** |
| S-2 有峰但峰 < 0.4551 | 不如那条五行规则，**只是多花了 GPU** |
| S-2 峰 > 门槛，但 **S-3 的 null 也涨** | 增益不来自界面先验，**不能宣称验证了 insight** |
| S-2 峰 > 门槛、null 不涨、S-4 有残差 | ✅ **真结果** —— 这才值得继续投入 |

**我对先验概率的判断**：`S-0` 通过的概率中等（§2 支持、§3 反对）；
若 `S-0` 通过，后面成立的概率不低，**因为细粒度那一块确实有空间**（ρ 0.483 → 上界未知）。
**`S-0` 只要几十分钟、不用训练，务必先做。**

---

## 6. 与 mutation-landscape-TTT 的关系（别做重了）

| | 我们在做的 | 你要做的 |
|---|---|---|
| 动哪部分 | **decoder**（3 层 + `W_out`，41.9% 参数） | **encoder**（3 层 + `W_e` + `features`） |
| 训练信号 | 让**分数**与界面先验的相关变强（transductive，看 mutant 序列不看 label） | 让 **`h_V`** 更好地表达界面几何（**只看 WT 结构**） |
| 是否 inductive | transductive（看得到 mutant 序列） | 🔴 **你的方向天然是 inductive 的** —— 只用 WT 结构，不看任何 mutant。**这更接近真正的 zero-shot TTT，是它的优势，要在记录里讲清楚。** |

**两者可以叠加**，但**先各自单独验证**，否则无法归因。

---

## 7. 必须遵守的纪律（全部来自我们踩过的坑）

1. **口径不许混。** 每个数字都要说明是 14 / 23 / 25 assay。不同实验用不同子集时，**分开报**。
2. **对照臂必须用掉同样多的 label 知识。** 见 §1.2。
3. **任何「取最大」的统计量都要配 null。** 扫超参、选配置都会对样本过拟合。
4. **先检查目标函数的最优解是什么。** 我们有过两次教训：
   - WT-likelihood 目标：最优解不是想要的东西，**已被否证**；
   - 无约束的「最大化分布差异」：最优解让分数塌缩成特征的函数，**比基线还差 0.13**。
   **动手前先问：如果这个 loss 被完美优化，分数会变成什么？**
5. **打分函数必须与评测的逐行相同。** 我们用一个 no-op 对照钉住了这件事
   （最大相对偏差 3.0e-7）。你改了 encoder 之后，**先验证 `steps=0` 时能复现官方分数**。
6. **单调不变性**：`Spearman(f(d), ·)` 对 `d` 的任意**单调**变换恒定不变。
   所以「换个距离形状」改不动秩层面的量，**只对加性组合与 Pearson 有杠杆**。别在这上面白费力气。
7. **label-free 的模型选择是这条线上反复出现的未解问题。** 我们测了 10 个静态特征选 `c`，
   9 个比常数更差，最好的只拿回 28% 且过不了 BH 校正。
   **ProteinTTT 在结构预测上能用 pLDDT 选模型，ProteinMPNN 这边至今没找到类比。**
   ⚠️ **你的方向可能例外** —— structure-TTT 有一个 decoder-TTT 没有的候选信号：
   **自监督目标自身的 held-out 值**（比如距离回归头在留出残基上的误差）。**值得专门试。**
8. **先写判读表再跑实验**，否则事后很容易把任何结果解释成成功。

---

## 8. 现成可复用的资产

| 东西 | 位置 |
|---|---|
| **可微打分 + encoder 缓存** | `scripts/mutation_landscape_ttt/bgmpnn.py`（本分支）—— `AssayContext` 已把 encoder 输出缓存，**改 `h_V` 后只跑 decoder**，S-0 直接能用 |
| **encoder 线性探针** | `scripts/mutation_landscape_ttt/ga_encoder_probe.py` —— S-1 在它上面改 |
| 逐残基界面距离（重原子，23+ assay） | `local-records/binding-sites-analysis/data/interface_residues_all_chains.csv` |
| entity 划分（元数据确定，非启发式） | `local-records/binding-sites-analysis/data/entity_partition.csv` |
| 逐 variant / 逐突变位点距离 | `…/variant_labels_with_mpnn.parquet`、`…/variant_site_dists.parquet` |
| 官方 zero-shot 参考分数（seed1, M=5） | workstation `/data/guoj0f/BindingGYM-zero-shot-proteinMPNN/scores/seed1_M5/` |
| ckpt `v_48_020.pt`（md5 `91d54c97a68bf551114f8c74c785e90f`） | workstation `/data/guoj0f/share/BindingGYM/training/cache/` |
| 官方代码与 input | workstation `/data/guoj0f/share/BindingGYM/`；本地 `/home/guoj0f/repos/BindingGYM`、`/home/guoj0f/share/BindingGYM/input` |
| conda env | workstation `bindinggym-zs-mpnn`（numpy 1.24.4 / scipy 1.10.1，**别升级**） |
| 上一阶段的摸底与结论 | `workstation-records/mutation-landscape-TTT/probe_binding_site_insight_value_20260909-160514.md` |
| decoder-TTT 的现状（对照用） | `workstation-records/mutation-landscape-TTT/transduc_decoder_ttt_interface_divergence-organized.md` |

---

## 9. 已知陷阱清单（会静默出错的那种）

1. **WT 序列位置 ↔ PDB 残基索引不是一回事。** 有偏移、有 gap；`6VJJ` 有 +1 偏移，
   `4ZFF`/`4ZFG` 带 **Kabat insertion code**（残基号形如 `100A`）。
   解析 PDB 时残基 id 必须取 `line[22:26].strip() + line[26].strip()`。
2. **`tied_featurize` 的打包顺序是「designed chains 先，其余在后」**，且用的是 **PDB 链序列**长度，
   不是 WT 序列长度。`h_V` 的索引跟着这个顺序走。
3. **M（解码顺序数）在 `AssayContext` 构造时就烙进了所有张量** —— 改 `ctx.M` 不重建会静默出错。
4. **`rsync -a` 从 workstation 回流会覆盖本地写的 record md**（按差异传、不按新旧传）。
   回流必须 `--exclude '*.md' --update`。
5. **matplotlib 只有 DejaVu Sans 可用**，中文会**静默**变成豆腐块 ⇒ **所有图上的文字用英文**。
6. **`pd.Series.median` / `.agg` 这类名字会被属性遮蔽** —— 列名别取这些，取了就用 `df["median"]`。
7. **新增命令行参数后，必须先用最小配置 smoke test 那条确切路径再投**。
   我们因为没做这件事，三个子任务秒退，白等了 8.5 小时的 GPU 空档。

---

## 10. 建议的第一周路线

| 顺序 | 做什么 | 代价 | 决策点 |
|---|---|---|---|
| 1 | 读本文档 §1–§3；复现 G-A（`ga_encoder_probe.py`）确认环境通 | ~30 min | — |
| 2 | **S-0 传导强度** | ~1 GPU-h，无训练 | 🔴 **不通过就停下来重新立题** |
| 3 | **S-1 细粒度上界**（MLP 探针 + 软标签分档） | ~1 GPU-h | 决定 S-2 是「注入信息」还是「让信息线性可读」 |
| 4 | no-op 对照：改完 encoder 路径后 `steps=0` 能否复现官方分数 | ~10 min | 不复现就先修 harness |
| 5 | **S-2** 主方案 (a) + 锚，小规模（3–4 个代表 assay）扫锚强度 | ~2 GPU-h | 无峰 ⇒ 否证 |
| 6 | **S-3 null** + **S-4 诊断** | ~2 GPU-h | 🔴 **定案的那一步** |

**选 assay 的建议**：别只挑小的。`PSD95_CRIPT`（n=1576, L=120，快）、
`KRAS_RAF1-RBD`（n=23161, L=245，headroom 大）、`ACE2_SARS2-RBD`（n=2185, **L=931**，长复合物压力测试）、
`GB1_IgG-Fc`（n=92890，大样本）覆盖了四种形状。
⚠️ **大 `L` 的 assay 显存吃紧**，batch 会被压到 25 以下，注意 batch 内统计量的噪声。
