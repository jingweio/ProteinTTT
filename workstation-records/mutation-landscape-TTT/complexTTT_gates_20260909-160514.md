# mutation-landscape-TTT — 实验记录

**created 2026-09-06 · 最近重构 2026-09-10 · status: `GATING`（设计未定稿，未启动任何 GPU 作业）**

> 本 project **唯一一篇**贯穿全生命周期的记录：计划先写、全程维护、结果往里追加。
> 新 gate / 实验**追加到 §6**，不要新开 md。

---

## 1. 目标 / 假设

**方法目标（complexTTT）**：给定当前 assay 的 **WT complex structure + WT target sequence + WT partner
sequence**，在**不使用该 assay 任何 mutant / DMS label** 的前提下，对预训练 ProteinMPNN 做
target-complex-specific 的 test-time customization，提升它在 BindingGYM 上的 zero-shot mutation ranking。

**待检验的 motivating 假设（用户提出）**：真值 DMS 上，`mutants_{碰 binding-site}` 与 `mutants_{不碰}`
的分布存在明显 divergence，而预训练 ProteinMPNN 建模不到这个先验；若能让模型感知到它，zero-shot 性能应有大幅提升。

**初步 implementation 方向（用户提出，尚未定稿）**：
1. 先做 **transductive** setting —— 模型可见 mutation variants，但不可见其 DMS score；
2. 主要动 **decoder** —— 假定 encoder 已把结构建模好，让 decoder 关注 binding-site 与突变后 DMS score 的相关性。

---

## 2. 🔴 口径定义：25 / 23 / 14 —— **引用任何数字前先确认是哪一个**

三个集合是**嵌套**的，每一层的剔除都有独立理由。由 `scripts/mutation_landscape_ttt/g1e_canonical14.py`
**从原始数据推导**（不是写死的名单）。

| 集合 | n | 怎么来的 |
|---|---:|---|
| **25** | 25 | BindingGYM 官方 benchmark 的全部 assay。**已发表数字（zero-shot 0.397、`inter_cluster` 0.42）都是这个口径。** |
| **23** | 23 | 25 减去 `KRAS_DARPinK27_5O2S`、`KRAS_SOS1_8BE4`。两者 label 逐值重复（19,227 个共享 variant，max\|Δ\|=0），根因在 Nature 补充表。**用户指定排除。** 见 `Sources/datasets/BindingGYM-issues/`。 |
| **14** | 14 | 23 再减去 9 个**二值界面分组无法定义**的 assay（见下表）。**本设计阶段的工作集**。 |

**从 23 掉到 14 的那 9 个，以及为什么：**

| assay | n(碰界面) | n(不碰) |
|---|---:|---:|
| `5A12_VEGF_fitness_4ZFF` | **0** | 29,980 |
| `4D5_HER2_fitness_1N8Z` | 2,076 | **3** |
| `BH3_Bcl-xL_normed_1PQ1` | 517 | **0** |
| `BH3_Mcl-1_normed_3KZ0` | 517 | **0** |
| `GB1_IgG-Fc_fitness_1FCC_2016` | 22,175 | **0** |
| `Z-domain_ZSPA-1_LL1_fitness_1LP1` | 45,476 | **0** |
| `Z-domain_ZSPA-1_LL2_fitness_1LP1` | 5,583 | **0** |
| `Z-domain_ZpA963_HL1_fitness_2M5A` | 2,903 | **0** |
| `Z-domain_ZpA963_HL2_fitness_2M5A` | 599 | **0** |

判据：**两侧都 ≥ 30**。这 9 个里一侧为空（或只有 3 个），**二值指示量恒定 ⇒ 二值方法的增益恒等于 0，无法定义**。

### 2.1 本阶段的决定（用户，2026-09-10）

**complexTTT 的设计阶段只在这 14 个 assay 上做** —— 能快速、准确地验证 insight。
拿到相对稳定的设计后再做 overall evaluation。**本文档因此只保留 14-assay 结果。**

> 23-assay 的结果（连续特征在 23/23 上都可用）不丢：数据在 `data/g1d_all23.csv`，
> 叙述在 git commit `c4140c1` / `e98f401`。等 overall evaluation 阶段再拉回来。

### 2.2 🔴 不可比的对照：已发表的 0.42

已发表的 `inter_cluster` 有监督 finetune ≈ **0.42 是 25-assay 口径**。
**它不能和本文的 14-assay 数字直接比。** 本文所有结论只用**同一批 14 个 assay 上我们自己算的基线**
（0.3903）作对照，报的是**相对增益**。跨口径的绝对比较留到 overall evaluation 阶段。

---

## 3. 设计与决策点

### 3.1 为什么先跑 gate 再写方案（已定）

既往有一条**已被实测否证**的近邻方案：字面版 WT-likelihood TTT 目标在同一套数据、同一个 ProteinMPNN 上
Spearman 随 lr 单调下降（0.705 → 0.694 → 0.648 → 0.388），失效机理是「目标函数最优解 = 打分公式自己的分母」。
当前想法**不是**那一条（监督信号是界面几何），但纪律适用：**先检查目标最优解、先跑 gate、再写生产代码。**

### 3.2 待用户拍板（阻塞中）

G1 的结果把门槛抬高了，且到达它**不需要 TTT**。complexTTT 该瞄什么必须重新定，候选见 §7。
**在此之前不写实现代码。**

---

## 4. Run config

- **G1（已完成）**：纯 CPU（本机），无 GPU，全流程 ≈ 3 min，无 PID。
- **后续 GPU 作业**：计划 A100 ×1 @ `10.67.24.41`，env `bindinggym-zs-mpnn`（已存在）。
  启动脚本放 `sh/`，PID 与预检显存在启动后填入 §5。

---

## 5. Change log

- **2026-09-06** G1 启动并完成（纯 CPU）。第一版交错 DP 有索引 bug，被 `ceiling ≥ baseline` 这条恒等式
  抓出（GB1 ceiling 0.3696 < baseline 0.5037）；已修，该判据现在是脚本里的 assert。
- **2026-09-09** 记录从 `local-records/` 迁到 `workstation-records/mutation-landscape-TTT/`，改为本模板。
- **2026-09-09** 更正措辞：硬编码 `c=−1` 不是「严格 zero-shot」（`c` 的符号有来源，见 §6.4）。
- **2026-09-09** G2（monomer control）原动机退役、降优先级 —— 它前置的那个 decoder loss 是
  binding-sites 分析之前臆想的 draft。
- **2026-09-10** **重构：** ① 全文改为 **14-assay 单一口径**，并把 25/23/14 的推导写进 §2；
  ② 按「先讲实验系统、再讲跑了哪些实验」重排 §6；③ 补 §6.5 讲清 DP 与 null；
  ④ 显式标出「已发表 0.42 是 25-assay 口径、不可直接比」（§2.2）—— 此前正是这类混用被用户抓出。

---

## 6. G1 — 「让 ProteinMPNN 感知 binding-site」值多少钱（**DONE**，纯 CPU，14 assay）

### 6.1 先讲实验系统：一个公式

**所有 G1 实验都在做同一件事 —— 对预训练 ProteinMPNN 的预测分数做一个由「离界面多近」驱动的平移：**

```
pred_score'(v)  =  pred_score(v)  +  c · sd(pred_score) · feature( d(v) )
```

| 符号 | 定义 | 能看到什么 |
|---|---|---|
| `pred_score(v)` | 预训练 ProteinMPNN 对 variant `v` 的 `global_score`（seed1 / M=5，在 WT 复合物上打分） | 模型输出 |
| `d(v)` | `v` **所有突变位点**到**另一个 entity** 的**最小**重原子距离（多点突变取最近的那一个） | **WT 复合物结构** —— label-free |
| `feature(·)` | 把距离映射成 0~1 的「在界面上的程度」。二值或 soft | 同上，label-free |
| `sd(pred_score)` | 该 assay 内**模型预测分数**的标准差 | **模型输出** —— label-free |
| `c` | **唯一的自由参数**，一个标量。决定平移的**强度与方向** | **取决于怎么选，见 §6.2** |

**`sd(pred_score)` 是干什么的 —— 一句话：单位换算，让同一个 `c` 在不同 assay 上强度相同。**
MPNN 的 `global_score` 是 mask 上 NLL 的**求和**、无长度归一化，所以尺度跨 assay 差很大
（14 个 assay 上 sd 从 **1.96 到 5.22，2.7×**；均值从 −152 到 −1829）。不除掉它，`c=−1` 在
`PSD95`（sd 2.25）上的实际位移只有在 `SARS2-RBD`（sd 5.22）上的 43%，`c` 就不可迁移。
**它只用模型输出，不碰任何 label。**

> ⚠️ 两处澄清（用户 2026-09-10 的理解需要修正）：
> **① `sd()` 不是「把两个 score 完全颠倒顺序的最小 offset」。** 要颠倒某两个特定 variant 的顺序，
> 所需 offset 就是它们的分数差本身，与 `sd` 无关。`sd` 只是**度量单位**。
> **② `c` 与 `sd()` 不是同一个作用。** `sd()` 是 label-free 的归一化常数；`c` 是**强度参数**，
> 而且**只有 oracle 那一档才用到测试 assay 的 label**（见 §6.2）。

### 6.2 在这个系统上跑了什么：**3 × 3 的实验网格**

**维度 A —— `feature` 怎么定：**

| id | feature | 含义 |
|---|---|---|
| **E1** | `1[d ≤ 5Å]` | **二值**：碰界面 = 1，否则 0 |
| **E2** | `exp(−d / 5Å)` | **soft**：接触处 ≈1、10Å ≈0.14、20Å ≈0.02，平滑衰减 |

**维度 B —— `c` 怎么选（决定这个数字能不能声称是 zero-shot）：**

| 档 | `c` 从哪来 | 用到了谁的 label | 可实现吗 |
|---|---|---|---|
| **oracle `c*`** | 用**该 assay 自己的真实 DMS** 挑最优 | **测试 assay 的 label** | ❌ 只作上界参考 |
| **LOAO 共享 `c`** | 只在**其它 13 个 assay** 上选，用到留出的那个上 | 其它 assay 的 label | ✅ 与官方 `inter_cluster` 同协议 |
| **硬编码 `c = −1`** | 完全不拟合 | 无（但符号有来源，见 §6.4） | ✅ 对测试 assay zero-shot |

**外加一个不属于这个公式的实验：**

| id | 是什么 | 为什么需要 |
|---|---|---|
| **E0** | **二值函数族的严格上界**（最优交错 DP + 随机标签 null） | E1 只是这个族里的**一个参数化形式**（加性）。E0 直接取**整个族的上确界** —— 回答「哪怕不用加性、任意方式使用二值标签，最多能值多少」。**详见 §6.5。** |

### 6.3 结果（14 assay，基线 **0.3903**，全部由 `g1e_canonical14.py` 现算）

| 实验 | `c` 档 | mean ρ | gain |
|---|---|---:|---:|
| — | 基线（无平移） | 0.3903 | — |
| **E1** 二值 | oracle `c*` | 0.4402 | +0.0498 |
| **E1** 二值 | **LOAO** | 0.4335 | **+0.0432** |
| **E1** 二值 | 硬编码 −1 | 0.4159 | +0.0256 |
| **E2** soft | oracle `c*` | 0.4689 | +0.0786 |
| **E2** soft | **LOAO** | **0.4551** | **+0.0648** |
| **E2** soft | 硬编码 −1 | 0.4459 | +0.0555 |

| E0（二值族严格上界，2000-variant 子样本） | mean ρ | gain vs 子样本基线 0.3895 |
|---|---:|---:|
| 真标签的最优交错 | 0.4497 | +0.0602 |
| **同尺寸随机标签**（null） | 0.4026 | +0.0131 |
| **净信息 = 真 − null** | — | **+0.0471** |

**四条读数：**

1. **E1 ≈ E0** —— 加性平移（+0.0498）基本吃满了二值族的上界（净 +0.0471）。
   ⇒ **不需要复杂重排序，一个标量偏移就够。** 设计问题塌缩成「这个标量怎么定」。
2. **E2 > E0** —— soft 特征的 oracle（0.4689）**高于二值族的上界**（0.4497）。这不矛盾：
   两个都碰界面但 `d` 不同的 variant 会拿到**不同**校正 ⇒ **组内次序也被改了** ⇒ soft 不属于二值族。
   **⇒ 「天花板」只对写明的函数族成立。** 更宽的族（任意 `g(pred_score, d)`）没有有意义的上界。
3. **soft 比二值多拿 +0.022**（LOAO 0.0648 vs 0.0432）⇒ **不要围绕二值界面标签设计。**
4. **`c` 不需要逐 target 估**：LOAO 拿到 oracle 的 **82%**（0.0648/0.0786）；
   `c*` 的符号在 E1 上 **14/14 为负**、E2 上 **13/14 为负**。
   ⇒ 框架内 TTT 还能贡献的只有那 **+0.014**（oracle 0.0786 − LOAO 0.0648）。

### 6.4 `c` 的符号从哪来 —— 必须说清楚

`c < 0` 编码的正是「**碰界面的突变更伤结合**」。它有两个来源：

- **(i) 生物物理先验** —— 突变界面残基损害结合，不需要任何数据；
- **(ii) 我们在 DMS 上确认过** —— 真值 δ<0 在 16/16 个可检验 assay 上成立（数据现状，见 §6.6 A1）。

所以「硬编码 `c=−1`」**对测试 assay 是 zero-shot**（没用它的任何 label），
但**不是「完全没有 DMS 知识介入」** —— 早先写成「严格 zero-shot」是**过强，已更正**。
**LOAO 那一档是把这件事做规范的版本。**

### 6.5 E0 里的两个东西到底是什么

#### (a) 最优交错 DP —— 二值族的上确界

**为什么二值标签的上界是可算的。** 想象两副牌：`A` = 碰界面的 variant 按模型分数排好，
`B` = 不碰的按模型分数排好。**任何「同组同校正」的方法都不能改变同一副牌内部的次序**
（同组成员拿到一样的位移，相对位置不动）。它唯一能做的是**改变两副牌怎么交错**。
所以「只用二值标签」这一族的上界 = **所有保持各自组内次序的合并方式中，与真值排序最吻合的那一个**。

**为什么要 DP（动态规划）。** 这样的合并共有 `C(n, n_A)` 种，天文数字。但最优解有子结构：
前 `i+j` 个位置的最优值只依赖 `(i, j)`。于是

```
dp[i][j] = 已从 A 放了 i 个、从 B 放了 j 个时,  Σ(位置 × 真值秩)  的最大值
转移:      下一个位置 (i+j+1) 从 A 取下一张, 或从 B 取下一张
答案:      dp[n_A][n_B]
```

复杂度 `O(n_A × n_B)`（实际在 2000-variant 子样本上跑，×5 次取均值）。

**为什么最大化 `Σ(位置 × 真值秩)` 就等于最大化 Spearman**：Spearman = Pearson(分配秩, 真值秩)；
分配秩**无论怎么合并都是 1..n 的一个排列**，均值方差固定，所以只有这个交叉项在变。

#### (b) 同尺寸随机标签的 null —— 为什么不能不做

**DP 是取最大值，所以它对样本会过拟合。** 哪怕标签毫无意义，它也总能挑出一个
「碰巧比按模型分数排序更吻合真值」的合并。所以 DP 的原始输出是**虚高**的。

**对照做法**：把 `z` **随机置换** —— 组大小完全不变、组员身份变得无意义 —— 跑**同一个 DP**。
它找到的增益就是「纯优化白拿的那部分」。

**真信息 = 真标签的上界 − 随机标签的上界 = 0.0602 − 0.0131 = +0.0471。**
（如果不做这个对照，就会把 +0.0602 当成界面标签的价值，虚高 28%。）

### 6.6 参照：本 project 之前就已经存在的**数据现状**（不是我们的实验结果）

来自 `../../local-records/binding-sites-overview/`，**口径是 16 个可检验 assay（未排除那两个）**：

| # | 现状 |
|---|---|
| A1 | 真值 DMS 上碰/不碰两组存在**方向一致**的差异（δ<0 在 **16/16**） |
| A2 | 但**效应量小**：OVL 中位 0.697、η² 中位 0.051 |
| A3 | **预训练 MPNN 建模不到**：δ<0 只在 8/16，另 8 个**符号翻转** |
| A4 | 逐 assay 界面敏感度与真值**不相关**（ρ = +0.376, p = 0.15） |
| A5 | **榜单成绩与界面区分无关**（Spearman(ρ, δ_MPNN) = +0.006, p = 0.98） |

### 6.7 Caveats

1. **这是 rescoring 结果，不是 modeling 结果。** 它不证明 MPNN「理解」了界面，只证明一个**手工加入的
   零参数结构特征**与 MPNN 互补。既往同向结论：零参数结构量（CB 邻居计数 ρ≈0.244、跨链最小距离 ρ≈0.260）
   本身就已打平 650M PLM ⇒ **「加了结构量后 ρ 涨了」不构成 binding-specific learning 的证据。**
2. **多点突变的聚合方式没调过。** `d` 目前取所有突变位点的**最小值**；求和/均值/加权可能更好，未探索。
3. **n = 14 个 assay**，`c` 只有一个自由度，LOAO 相对可靠，但样本量小。
4. **模型分数只有 1 个 seed / M=5。**
5. **E0 在 2000-variant 子样本上算**（DP 是 `O(n_A n_B)`，全量的 GB1 是 2e9 态），×5 次取均值；
   子样本基线 0.3895 与全量 0.3903 吻合。
6. **不可与已发表数字直接比**，见 §2.2。

---

## 7. 下一个决策点（阻塞中）

complexTTT 的定位候选（**门槛是 14-assay 上的 0.4551，不是 0.3903**）：

1. **先把 rescoring 基线做成正式结果** —— 零 label / 零 GPU / 零 TTT。需要补：多点突变的 `d` 聚合、
   多 seed、以及 overall evaluation（23 / 25 口径）。complexTTT 降为后续。
2. **重新瞄准那 +0.014** —— 坚持 TTT，目标改成「逐 target 估计 `c`」。诚实但天花板低。
3. **换一个 rescoring 做不到的读数** —— 如 partner-conditional specificity。需先确认距离特征在该读数上能不能被拉开。
4. ~~先跑 G2 再决定~~ —— **已退役**（原动机是 §5 里那个已废弃的 decoder loss）。

**用户 2026-09-09 的方向**：只围绕**已完成的 binding-sites 分析**（§6.6 的 A1–A5 + §6.3 的 G1）
把 complexTTT 细化到可落实可执行，不要再引入分析之前臆想的方案。

---

## 8. 复现

```
scripts/mutation_landscape_ttt/
  g1_ceiling.py       E0：最优交错 DP + 随机标签 null + 加性平移；带 ceiling≥baseline 的 assert
  g1b_shared_c.py     逐 assay c* vs 一个共享常数（二值）
  g1c_loao.py         LOAO + 四种 feature 形式的对比（14 assay 口径）
  g1d_all23.py        23-assay 口径（本阶段不用，留给 overall evaluation）
  g1e_canonical14.py  ★ 生成 §2 的集合推导与 §6.3 的权威结果表
```

- 输入：`../../local-records/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet`（376,424 × 12）
- 产物：`data/`（`g1e_canonical14.csv` 是 §6.3 的来源；其余为过程数据）
- 纯 CPU，约 3 分钟，无 GPU 依赖
- 界面定义与逐 variant 标签来源：`../../local-records/binding-sites-overview/`

## 关联

- `../../local-records/binding-sites-overview/BindingGYM_binding_sites_overview_20260903.md`
  —— 界面定义、真值与 MPNN 的分布对比（§6.6 的来源，本 project 的动机）
- `Sources/datasets/BindingGYM-issues/` —— 排除那两个 assay 的依据
- 已退役方向的原始记录：git commit `3df2466`（G2 的完整描述）
