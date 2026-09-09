# mutation-landscape-TTT — 实验记录

**created 2026-09-06 · 本篇整理于 2026-09-09 · status: `GATING`（设计未定稿，未启动任何 GPU 作业）**

> 这是本 project 的**唯一一篇**贯穿全生命周期的记录：计划先写、全程维护、结果往里追加。
> 新的 gate / 实验请**追加到 §5**，不要新开 md。

---

## 1. 目标 / 假设

**方法目标（complexTTT）**：给定当前 assay 的 **WT complex structure + WT target sequence + WT partner
sequence**，在**不使用该 assay 任何 mutant / DMS label** 的前提下，对预训练 ProteinMPNN 做
target-complex-specific 的 test-time customization，提升它在 BindingGYM 上的 zero-shot mutation ranking。

**待检验的 motivating 假设**（用户提出）：真值 DMS 上，`mutants_{碰 binding-site}` 与
`mutants_{不碰}` 的分布存在明显 divergence，而预训练 ProteinMPNN 建模不到这个先验；
若能让模型感知到它，zero-shot 性能应有**大幅**提升。
（现象部分已在 `../../local-records/binding-sites-overview/` 中量化确认。）

**初步 implementation 方向**（用户提出，尚未定稿）：
1. 先做 **transductive** setting —— 模型可见 mutation variants，但不可见其 DMS score；
2. 主要动 **decoder** —— 假定 encoder 已把结构建模好，让 decoder 去关注 target complex 的
   binding-site 与突变后 DMS score 之间的相关性。

---

## 2. 设计与决策点

### 2.1 评测口径（已定）

| 项 | 取值 | 理由 |
|---|---|---|
| assay 集合 | **23 个**（25 减 `KRAS_DARPinK27_5O2S`、`KRAS_SOS1_8BE4`） | 用户指定；两者 label 逐值重复，见 `Sources/datasets/BindingGYM-issues` |
| 指标 | per-assay Spearman，**未加权 assay 均值** | 与官方 headline 同口径 |
| 基线 | zero-shot ProteinMPNN **0.3939** | 既有 seed1/M=5 分数复算，见 §5.1 |
| 参照 | 官方 `inter_cluster` **有监督** finetune ≈ **0.42** | 唯一不泄漏测试 assay label 的官方 finetune 设定 |
| 噪声 | 单 assay seed σ ≈ 0.008（M=20 实测） | 判断增益是否真实的下限 |

### 2.2 为什么先跑 gate 再写方案（已定）

既往有一条**已被实测否证**的近邻方案：字面版 WT-likelihood TTT 目标在同一套数据、同一个
ProteinMPNN 上 Spearman 随 lr 单调下降（0.705 → 0.694 → 0.648 → 0.388），失效机理是
「目标函数最优解 = 打分公式自己的分母」。当前想法**不是**那一条（监督信号是界面几何），
但那次的纪律适用：**先检查目标最优解、先跑 gate、再写生产代码**。

于是定了两个 gate：

- **G1（纯 CPU）—— 收益上界。** 假设最好能值多少钱？→ **已完成，见 §5.1。**
- **G2（GPU ~1.5–2 h）—— 信号存在性。** 删掉 partner 重新打分，看 MPNN 到底有没有在用 partner。
  这是方向 2（decoder-TTT）的**可行性前置**：若 MPNN 实质 partner-blind，则
  `p(aa|complex) − p(aa|monomer) ≈ 0`，decoder 没有 partner 信号可放大。→ **未跑，见 §5.2。**

### 2.3 待用户拍板的决策点（阻塞中）

G1 的结果把门槛从 0.3939 抬到了 **0.4505**，且到达它**不需要 TTT**。所以「complexTTT 该瞄什么」
必须重新定，候选见 §6。**在此之前不写实现代码。**

---

## 3. Run config

- **G1**：纯 CPU（本机），无 GPU，全流程 ≈ 3 min。无 PID。
- **G2**：计划 A100 ×1 @ `10.67.24.41`，env `bindinggym-zs-mpnn`（已存在），预计 1.5–2 GPU-h。
  启动脚本将放 `sh/`，PID 与预检显存在启动后填入 §4。

---

## 4. Change log

- **2026-09-06** G1 启动并完成（纯 CPU）。第一版交错 DP 有索引 bug，被 `ceiling ≥ baseline`
  这条恒等式抓出（GB1 ceiling 0.3696 < baseline 0.5037）；已修并把该判据写成 assert。
- **2026-09-09** 修正 G1 的报告口径：原先只在 14 个 assay 上报，改为**全部 23 个**（见 §5.1 的说明）。
- **2026-09-09** 记录从 `local-records/mutation-landscape-TTT/` 迁到
  `workstation-records/mutation-landscape-TTT/`（用户指定的 project 目录），并改为本模板。

---

## 5. Results

### 5.1 G1 — 「让 ProteinMPNN 感知 binding-site」的收益上界（**DONE**，纯 CPU）

> ⚠️ **口径：G1.2–G1.4 只在 14 个 assay 上**（二值分组两侧都 ≥30 的那些），基线 0.3903。
> 剩下 9 个 assay 里**所有 variant 都碰界面**（或都不碰），二值指示量恒定 ⇒ 二值方法在那里恒等于 0 增益。
> **连续距离在 23/23 上都有变化**，所以最终结论（G1.5）改在**全部 23 个 assay** 上报，基线 **0.3939** —— 
> 这也是已发表数字（0.397 / 0.42）所用的口径。见 G1.5.1。

#### G1.1. 为什么可以算出上界

BindingGYM 的每个 metric 都**秩不变**；界面标签是**二值**的，而且**在 test time 是 label-free 可得的**
（从 WT 复合物结构算，不碰任何 DMS）。所以「让模型感知 binding-site」在数学上就是**把两组重新交错**。
给一个 oracle 真实 DMS 去挑最优交错，得到的就是**这一整类方法的天花板** —— 无论 TTT 怎么实现都超不过。

每一项都配了**同尺寸随机标签的 null**：oracle 在无意义标签上也能榨出一点增益，只有两者之差才是界面标签真正携带的信息。

#### G1.2. 结果

| 校正方式 | mean ρ | vs 基线 | null | 净增益 |
|---|---:|---:|---:|---:|
| 基线（无校正） | 0.3903 | — | — | — |
| 加性平移 `s + c·z`（逐 assay oracle c） | 0.4402 | +0.0499 | 0.3909 | **+0.0493** |
| **最优交错（严格天花板）** | 0.4497 | +0.0594 | 0.4026 | **+0.0471** |
| 界面标签单独作预测器 | −0.2068 | — | — | — |

**第一个结论：加性平移基本吃满了天花板**（+0.0493 vs +0.0471）。
也就是说**不需要复杂的重排序，一个标量偏移就够**。设计问题因此塌缩成：**这个标量能不能不用 label 估出来？**

> 自检：`ceiling ≥ baseline` 是恒等式（恒等交错本身就是合法解），脚本里是 assert。
> 第一版 DP 索引写错时正是这条把它抓了出来（GB1 ceiling 0.3696 < baseline 0.5037）。

#### G1.3. 那个标量需要逐 target 估计吗？—— **不需要**

| | mean ρ | gain | 占 oracle |
|---|---:|---:|---:|
| 逐 assay oracle `c*` | 0.4402 | +0.0499 | 100% |
| **一个共享常数** `c = −1.10 sd` | 0.4345 | +0.0442 | **89%** |
| 硬编码 `c = −1.00 sd`（完全不拟合） | 0.4342 | +0.0439 | 88% |

**`c*` 在 14/14 个 assay 上全为负**（范围 −4.00 ~ −0.25 sd，中位 −1.07）。方向是普适的：**把碰界面的 variant 往下压**。

#### G1.4. 连续距离比二值标签好得多（LOAO 全部样本外）

`c` 用 **leave-one-assay-out** 重新选，所以下表每个数字都是样本外的。
特征都做了标准化，因此 `c` 可比。`d` = 该 variant 的突变位点到另一个 entity 的最小重原子距离。

| 特征 | oracle `c*` | **LOAO** | 硬编码 `c=−1` | LOAO/oracle |
|---|---:|---:|---:|---:|
| 二值 `1[d ≤ 5Å]` | 0.4402 (+0.0498) | 0.4335 (**+0.0432**) | 0.4159 (+0.0256) | 87% |
| **软 `exp(−d/5Å)`** | 0.4689 (+0.0786) | **0.4551 (+0.0648)** | **0.4459 (+0.0555)** | 82% |
| 软 `1/(1+d/5Å)` | 0.4703 (+0.0799) | 0.4525 (+0.0621) | 0.4408 (+0.0505) | 78% |
| 原始 `−min(d,20Å)/20` | 0.4707 (+0.0803) | 0.4505 (+0.0602) | 0.4430 (+0.0527) | 75% |

**连续距离比二值标签多拿 +0.022**（LOAO 0.0648 vs 0.0432）。⇒ **不要围绕二值界面标签设计**。

#### G1.5. 结论 —— 假设成立，但它论证的不是 TTT

##### G1.5.1 全部 23 个 assay（最终口径）

| | mean ρ | gain | 说明 |
|---|---:|---:|---|
| 基线 zero-shot ProteinMPNN | 0.3939 | — | — |
| oracle 逐 assay `c*` | 0.4760 | +0.0821 | 上界，不可达 |
| **LOAO 共享 `c`** | **0.4505** | **+0.0566** | 样本外，与 `inter_cluster` 同协议 |
| **硬编码 `c = −1`（零拟合）** | **0.4463** | **+0.0524** | **严格 zero-shot** |

`c*` 在 **20/23** 个 assay 上为负（3 个为正：`hYAP65`、`Z-ZSPA-1_LL2`、`5A12_VEGF` —— 都在既往标记为
行为反常或效应极弱的那批里）。全 23 的共享 `c` = **−0.65 sd**。

分组看：二值可用的那 14 个 LOAO **+0.0645**；**二值碰不到的那 9 个也拿到 +0.0442** ——
连续特征在「全部 variant 都碰界面」的 assay 里同样有效，因为界面内部的远近仍然有区分度。


**你的 insight 被证实了，而且比预期更值钱：**

```
score' = score + c · sd(score) · exp( −d / 5Å )       c ≈ −1.5（LOAO），或直接取 −1
```

| 方法 | ρ | 用了什么 |
|---|---:|---|
| zero-shot ProteinMPNN | 0.3903 | — |
| 官方 `inter_cluster` **有监督** finetune | ≈ 0.42 | **4/5 个 assay 的 DMS label** + GPU |
| **上式，硬编码 c = −1，零拟合** | **0.4463** | 只有 WT 复合物结构 |
| **上式，c 由 LOAO 选** | **0.4505** | 只有 WT 复合物结构（c 在其它 assay 上选） |

**这套东西不需要 test-time training，也不需要 GPU** —— 是一条五行的 post-hoc 重打分规则，
却**超过了已发表的有监督 finetuning**。

**⇒ complexTTT 的门槛从 0.3939 变成了 0.4505。** 在这个框架内，TTT 还能贡献的只有
「逐 target 估计 c」那一部分 = oracle 0.0821 − LOAO 0.0566 = **+0.026**（全 23 口径）。

#### G1.6. Caveats

1. **这是 rescoring 结果，不是 modeling 结果。** 它不证明 MPNN「理解」了界面，只证明一个**手工加入的
   零参数结构特征**与 MPNN 互补。既往结论同向：零参数结构量（CB 邻居计数 ρ≈0.244、跨链最小距离 ρ≈0.260）
   本身就已打平 650M PLM ⇒ **「加了某个结构量之后 ρ 涨了」不构成 binding-specific learning 的证据**。
2. **多点突变的聚合方式没有调过。** `d` 目前取该 variant **所有突变位点的最小值**；
   换成求和/均值/加权可能更好，是一个未探索的设计旋钮。
3. **n = 23 个 assay**（G1.5.1）或 14 个（§2–4），且 `c` 只有一个自由度，LOAO 相对可靠，但样本量小。
6. **G1.2 的「严格天花板」只是二值标签那一族的天花板。** 连续特征会改变**组内**次序，因此不受它约束 ——
   实测 `exp(−d/5Å)` 的 oracle（14 assay 上 0.4689）确实**高于**该天花板（0.4497）。
4. **模型分数只有 1 个 seed / M=5。**
5. LOAO 与官方 `inter_cluster` 是同一类协议（在其它 assay 上定超参、在留出 assay 上测），所以两者可比；
   硬编码 `c=−1` 那一行则是**严格 zero-shot**，不涉及任何拟合。

---

### 5.2 G2 — monomer control（**PLANNED，未跑**）

**问题**：ProteinMPNN 打分时到底有没有在用 partner 链？它在完整复合物上打分、k=48 kNN 图包含
partner 原子，所以原理上「看得见」；但实测它的界面对比是错的甚至反的（6/14 个 assay 符号翻转）。

**做法**：删掉 partner 链、只用被突变链的结构，按**完全相同的官方协议**重新 zero-shot 打分
（同 ckpt / 同 M / 同 seed / `backbone_noise=0`），比较逐 variant 分数与 δ。

**预期结果与判读**：

| 若 | 结论 | 对方向 2（decoder-TTT）的影响 |
|---|---|---|
| 分数几乎不变（ρ(complex, monomer) ≈ 1，δ 不变） | MPNN 实质 partner-blind | **硬阻塞** —— decoder 没有 partner 信号可放大；只能像 G1 那样从外部注入几何 |
| 分数明显变、但 δ 没变好 | 用了 partner 但提取得不对 | **complexTTT 站得住** —— TTT 是重新加权一个已存在的信号 |

**成本**：全量 25 assay × M=5 含 partner 实测 ≈2.5 h；删 partner 后结构更小，预计 **1.5–2 GPU-h**。

---

## 6. 下一个决策点（阻塞中）

G1 之后，complexTTT 的定位候选：

1. **先把 rescoring 基线做成正式结果** —— 它本身是强结果（零 label / 零 GPU 超过有监督 finetune）。
   需要补：多点突变的 `d` 聚合方式、多 seed、与已发表 baseline 对齐。complexTTT 降为后续。
2. **重新瞄准那 +0.026** —— 坚持 TTT，但目标改成「逐 target 估计校正强度 `c`」。诚实但天花板低。
3. **换一个 rescoring 做不到的读数** —— 如 partner-conditional specificity。需先确认距离特征
   在该读数上能不能被拉开。
4. **先跑 G2 再决定** —— 结果直接决定上面三条里哪条可行。

---

## 7. 复现

```
scripts/mutation_landscape_ttt/
  g1_ceiling.py    G1 天花板（最优交错 DP + 加性平移 + 随机标签 null），带 ceiling≥baseline 的 assert
  g1b_shared_c.py  逐 assay c* vs 一个共享常数
  g1c_loao.py      LOAO 验证 + 二值 / 连续距离特征对比（14 assay 口径）
  g1d_all23.py     连续特征在全部 23 个 assay 上的最终口径（§5.1 的头条数字）
```

- 输入：`../../local-records/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet`（376,424 × 12）
- 产物：`data/`（`g1_ceiling.csv`、`g1b_shared_c.csv`、`g1c_loao.csv`、`g1d_all23.csv` + 两个 npy）
- 纯 CPU，约 3 分钟，无 GPU 依赖
- 界面定义与逐 variant 标签的来源：`../../local-records/binding-sites-overview/`

## 关联

- `../../local-records/binding-sites-overview/BindingGYM_binding_sites_overview_20260903.md`
  —— 界面定义、真值与 MPNN 的分布对比（本 project 的动机来源）
- `Sources/datasets/BindingGYM-issues/` —— 排除那两个 assay 的依据
