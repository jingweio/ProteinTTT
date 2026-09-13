# structure-TTT：从 **encoder** 侧提升 ProteinMPNN 的 BindingGYM zero-shot 能力

**写于 2026-09-13。** 本文档记录的是**另一条分支上已经做完的相关工作与已知事实**，
供 structure-TTT 方向参考。它不包含实验规划。

---

## 0. 问题

给定某个 assay 的 WT complex structure，能不能在 test time **只用结构本身**（零 DMS label）
把 ProteinMPNN 的 **encoder** 对这个复合物、特别是对 binding-sites 的**细粒度几何**编码得更好，
从而提升它在 BindingGYM 上的 zero-shot DMS-score 预测？

**效果预期**：只要**优于预训练 ProteinMPNN 的 zero-shot**即可。
这一侧的定位是**学一个更好的基础表征**，为后续的 mutation-landscape-TTT 提供更好的起点，
不需要直接去比那些更强的后处理方法。

---

## 1. 评测口径

### 1.1 基线与指标

- 指标：**per-assay Spearman（预测分数 vs DMS_score），再对 assay 取未加权平均**。
- **预训练 ProteinMPNN zero-shot 基线**：**23-assay `0.3939`**、**14-assay `0.3903`**。
- 打分口径（官方 `global_score`）：把 variant 的突变序列做 autoregressive teacher-forced，
  在 mask 上**求和** NLL 再取负，**无 WT 项、无长度归一化**，对 **M=5** 个解码顺序取均值。
  ⚠️ 因为是 NLL 求和且不做长度归一化，**跨 assay 的绝对分数不可比**（尺度差 3.3×）；
  只有 assay 内部的排序有意义。

### 1.2 25 / 23 / 14 分别是哪些 assay

**三套嵌套的 assay 集合**，报任何数字都必须说明用的是哪一套：

- **25** = BindingGYM 官方全集。
- **23** = 25 减去 **2 个 label 被污染的 assay**：`KRAS_DARPinK27_norfitness_5O2S`、
  `KRAS_SOS1_norfitness_8BE4`（根因在 Nature 补充表，不在 BindingGYM 本身；**一律排除**）。
- **14** = 23 中**「碰界面 / 不碰界面」两组都有足够样本（各 ≥ 30）**的那些。
  其余 9 个是**全碰或全不碰**，二值指示量恒定 ⇒ 任何基于二值界面标签的分析在它们上恒为 0，无法评。
  ⚠️ **连续距离在 23 个上都有变化**，所以只有**需要二值切分**的分析才受限于 14。

| # | DMS_id | n variants | 25 | 23 | 14 | 碰界面比例 |
|---:|---|---:|:--:|:--:|:--:|---:|
| 1 | `4D5_HER2_fitness_1N8Z` | 2,079 | ✅ | ✅ | — | 0.999 |
| 2 | `5A12_Ang2_fitness_4ZFG` | 943 | ✅ | ✅ | ✅ | 0.869 |
| 3 | `5A12_VEGF_fitness_4ZFF` | 29,980 | ✅ | ✅ | — | 0.000 |
| 4 | `ACE2_SARS2-RBD_enrich_6M17` | 2,185 | ✅ | ✅ | ✅ | 0.174 |
| 5 | `BH3_Bcl-xL_normed_1PQ1` | 517 | ✅ | ✅ | — | 1.000 |
| 6 | `BH3_Mcl-1_normed_3KZ0` | 517 | ✅ | ✅ | — | 1.000 |
| 7 | `CD19_FMC63_Fitness_7URV` | 3,885 | ✅ | ✅ | ✅ | 0.035 |
| 8 | `CXCR4_CXCL12_enrich_8U4O` | 5,584 | ✅ | ✅ | ✅ | 0.126 |
| 9 | `GB1_IgG-Fc_fitness_1FCC` | 92,890 | ✅ | ✅ | ✅ | 0.630 |
| 10 | `GB1_IgG-Fc_fitness_1FCC_2016` | 22,175 | ✅ | ✅ | — | 1.000 |
| 11 | `HLA-A2_TAPBPR_meanscore_5WER` | 3,344 | ✅ | ✅ | ✅ | 0.123 |
| 12 | `KRAS_DARPinK27_norfitness_5O2S` | 19,532 | ✅ | ❌ | — | 0.459 |
| 13 | `KRAS_PICK3CG-RBD_norfitness_1HE8` | 19,202 | ✅ | ✅ | ✅ | 0.203 |
| 14 | `KRAS_RAF1-RBD_norfitness_6VJJ` | 23,161 | ✅ | ✅ | ✅ | 0.257 |
| 15 | `KRAS_RAF1_norfitness_6VJJ` | 12,676 | ✅ | ✅ | ✅ | 0.390 |
| 16 | `KRAS_RALGDS-RBD_norfitness_1LFD` | 20,340 | ✅ | ✅ | ✅ | 0.272 |
| 17 | `KRAS_SOS1_norfitness_8BE4` | 19,424 | ✅ | ❌ | — | 0.512 |
| 18 | `PSD95_CRIPT_1BE9` | 1,576 | ✅ | ✅ | ✅ | 0.181 |
| 19 | `PSD95_Tm2F_1BE9` | 1,576 | ✅ | ✅ | ✅ | 0.193 |
| 20 | `SARS2-RBD_ACE2_deltaKd_6M0J` | 21,871 | ✅ | ✅ | ✅ | 0.309 |
| 21 | `Z-domain_ZSPA-1_LL1_fitness_1LP1` | 45,476 | ✅ | ✅ | — | 1.000 |
| 22 | `Z-domain_ZSPA-1_LL2_fitness_1LP1` | 5,583 | ✅ | ✅ | — | 1.000 |
| 23 | `Z-domain_ZpA963_HL1_fitness_2M5A` | 2,903 | ✅ | ✅ | — | 1.000 |
| 24 | `Z-domain_ZpA963_HL2_fitness_2M5A` | 599 | ✅ | ✅ | — | 1.000 |
| 25 | `hYAP65_peptide_FunctioncalScore_1JMQ` | 18,406 | ✅ | ✅ | ✅ | 0.527 |

**被排除出 14 的那 9 个，及其原因（碰 / 不碰 的样本数）：**

| DMS_id | 碰界面 | 不碰界面 |
|---|---:|---:|
| `4D5_HER2_fitness_1N8Z` | 2,076 | **3** |
| `5A12_VEGF_fitness_4ZFF` | **0** | 29,980 |
| `BH3_Bcl-xL_normed_1PQ1` | 517 | **0** |
| `BH3_Mcl-1_normed_3KZ0` | 517 | **0** |
| `GB1_IgG-Fc_fitness_1FCC_2016` | 22,175 | **0** |
| `Z-domain_ZSPA-1_LL1_fitness_1LP1` | 45,476 | **0** |
| `Z-domain_ZSPA-1_LL2_fitness_1LP1` | 5,583 | **0** |
| `Z-domain_ZpA963_HL1_fitness_2M5A` | 2,903 | **0** |
| `Z-domain_ZpA963_HL2_fitness_2M5A` | 599 | **0** |

> **这三套集合的完整推导与逐 assay 明细**（含每个 assay 的界面残基数、变体数、基线 Spearman）：
> `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-overview/BindingGYM_binding_sites_overview_20260903.md`
> 生成脚本：`/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/scripts/binding_sites_overview/o5_basis23.py`、
> `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/scripts/mutation_landscape_ttt/g1e_canonical14.py`

---

## 2. 已知事实：encoder 已经"会"界面 —— 但只在**二值**层面

### 2.1 实验是怎么做的

在 decoder-TTT 之前做过一个前置检查（代号 **G-A**）：

1. 对每个 assay，用**冻结**的 ProteinMPNN encoder 跑一遍它的 WT complex 结构，
   拿到每个残基的 node embedding `h_V`（**128 维 / 残基**）；
2. 在 `h_V` 上训一个**线性**模型（logistic regression / ridge），预测该残基的界面属性；
3. 两种读出各测一次：
   - **AUC（二值）**：logistic regression 预测「该残基是不是界面残基」，报 ROC-AUC。
   - **ρ（连续距离）**：ridge regression 预测「该残基到**另一个 entity** 的距离」这个**实数**，
     然后报**预测距离与真实距离之间的 Spearman 相关**。
     ⇒ **AUC 衡量「碰不碰」分得开不开；ρ 衡量「多远」估得准不准。**
4. 两种切分方式：
   - **within-assay（5-fold）**：在同一个 assay 的残基内部做 5 折交叉验证。
     回答「**对这个复合物**，信息在不在 `h_V` 里」。
   - **leave-one-assay-out（LOAO）**：拿其余 13 个 assay 的残基训练，在留出的那个 assay 上测。
     回答「这个编码是**跨复合物共享**的，还是每个结构各背一套」。

### 2.2 结果

**「中位」= 对 14 个 assay 各算一个值，再取这 14 个值的中位数。**

| | 中位 AUC（二值：碰 / 不碰） | 中位 ρ（连续：离 partner 多远） |
|---|---:|---:|
| **within-assay（5-fold）** | **0.890** | **0.483** |
| **leave-one-assay-out** | **0.915** | 0.404 |

**三条结论：**

1. **二值界面信息已经在 `h_V` 里，且线性可读。** AUC 0.890 远高于随机的 0.5。
2. **这个编码是跨复合物共享的** —— LOAO（0.915）比 within-assay（0.890）**还高**，
   因为 LOAO 用了 13 个 assay 的数据训练，样本多 13 倍。
   若编码是每个结构各背一套，LOAO 应当明显更差。
3. 🔴 **但细粒度明显更弱** —— `h_V` 对「碰不碰」编码得好（AUC 0.890），
   对「多远」编码得差（ρ 0.483），LOAO 下还有 **2/14 个 assay 的连续 ρ 为负**
   （`CD19_FMC63_Fitness_7URV`、`KRAS_PICK3CG-RBD_norfitness_1HE8`）。
   **这正是 structure-TTT 可能有空间的地方。**

### 2.3 这个探针用的界面定义，和别处**不一样**（口径说明）

**别处（所有变体级分析）用的定义**：残基 `r` 的任一**重原子**到**另一个 entity** 的任一重原子
的最小距离 ≤ **5.0 Å**，即为界面残基。entity 的划分**由元数据确定**（DMS_id 里写明了两个结合方，
不是靠结构启发式猜的）。

**G-A 这个探针用的是**：**CA–CA 距离 + 8 Å 阈值**。

**为什么要不一样 —— 这是刻意的：**

- `h_V` 的第 `i` 行对应的是 **ProteinMPNN 内部打包顺序**下的第 `i` 个残基
  （`tied_featurize` 按「designed chains 在前、其余在后」拼接，用的是 **PDB 链的序列**）。
- 而重原子界面标签（`interface_residues_all_chains.csv`）是按 **WT 序列位置**索引的。
- **这两套索引不是同一个东西** —— 中间隔着一次 WT 序列 ↔ PDB 残基的比对，
  有偏移、有 gap，而且是**已知会静默出错**的地方（见 §3 第 1 条）。
- G-A 要回答的问题是「**`h_V` 里有没有界面信息**」，**这个问题不需要精确到重原子**。
  所以直接在 `h_V` 自己的打包空间里、用 `ctx.X[:, :, 1]`（CA 坐标）算距离，
  **整个对齐环节被绕开了**，结果不可能因为 off-by-one 而错。
- 8 Å 而不是 5 Å：CA–CA 距离系统性地大于重原子最小距离（两个残基的重原子可以贴到 5 Å，
  而它们的 CA 仍相距 ~8 Å），8 Å 是一个粗略但合理的等价阈值。

**⇒ 如果你要用重原子级的精细标签**（比如做距离回归的监督信号），
可以用 `interface_residues_all_chains.csv`（里面有逐残基的重原子最小距离），
**但你必须自己把它对齐到 `h_V` 的打包索引上**，并自己验证对齐正确（见 §3 第 1、2 条）。

---

## 3. 支持这个方向的证据：encoder 编码得越好的 assay，zero-shot 就越好

### 3.1 具体测了什么

**这是一个跨 assay 的相关性分析，n = 14（上面 §1.2 表里标了 14 的那些）。**

对每个 assay，我们手上有两个独立得到的数：

| 量 | 怎么来的 | 含义 |
|---|---|---|
| **`AUC_within`** | §2 的 G-A 探针，within-assay 5-fold | **这个复合物的 encoder 表征，把界面残基分得有多开** |
| **`ρ_zeroshot`** | 预训练 ProteinMPNN 在该 assay 上的 per-assay Spearman | **这个 assay 的 zero-shot DMS 预测有多准** |

然后问：**这 14 对数之间有没有关系？** 即「encoder 把界面编码得好的那些 assay，
是不是恰好也是 ProteinMPNN 预测得准的那些」。用 Spearman 秩相关来测。

### 3.2 结果

| 探针指标 | 对照量 | Spearman | p |
|---|---|---:|---:|
| **`AUC_within`（二值界面）** | **`ρ_zeroshot`** | **+0.730** | **0.003** ✅ |
| `ρ_within`（连续距离） | `ρ_zeroshot` | +0.172 | 0.557 |
| `AUC_within` | headroom（该 assay 还剩多少提升空间） | −0.198 | 0.497 |

### 3.3 怎么解读

**正面**：
`+0.730 / p=0.003` 是一个**强且显著**的正相关（n=14）。
它说明：**encoder 对界面的编码质量，和 ProteinMPNN 在该 assay 上的 zero-shot 表现，是同向的。**
这是「改善 encoder 的界面表征 → 改善 zero-shot」这条因果链的**必要条件**，而且它成立了。

**但这是相关，不是因果。** 至少有三个混淆项必须警惕，它们可能**同时**驱动这两个量：

1. **结构质量** —— 分辨率高、缺失残基少的结构，encoder 编码得好是自然的，
   同时 ProteinMPNN 打分也更可靠。
2. **复合物大小 `L`** —— 大复合物的残基数多，探针训练样本多、AUC 可能偏高；
   同时 `L` 也影响打分的尺度与方差。
3. **assay 本身的噪声水平** —— DMS 实验噪声大的 assay，`ρ_zeroshot` 的上限本来就低，
   与 encoder 无关。

**⇒ 若要把这条证据从「相关」推进到「支持因果」**，最直接的办法是做**干预**：
不训练，直接人为改动冻结的 `h_V`（例如沿探针学到的界面方向加一个可控幅度的扰动），
重新打分，看 per-assay Spearman 是否随之变化。这只需要前向打分，不需要训练。

**反面参照**：`ρ_within`（连续距离）与 `ρ_zeroshot` 的相关只有 `+0.172, p=0.557`，**不显著**。
所以目前这条证据**只支持「二值界面编码质量」这一层**，
**细粒度距离编码与 zero-shot 表现之间的关系尚未被证实**。

---

## 4. 会静默出错的地方（纪律与陷阱）

**这一节全部来自实际踩过的坑 —— 它们的共同特征是：不报错、结果看起来合理、但已经错了。**

### 4.1 索引与对齐

1. **WT 序列位置 ↔ PDB 残基索引不是一回事。** 有偏移、有 gap。
   实测案例：`6VJJ` 有 **+1 偏移**；`4ZFF` / `4ZFG` 带 **Kabat insertion code**（残基号形如 `100A`）。
   解析 PDB 时残基 id 必须取 `line[22:26].strip() + line[26].strip()`，否则 insertion code 被吞掉。
2. **`tied_featurize` 的打包顺序是「designed chains 在前，其余在后」**，
   且用的是 **PDB 链序列**长度，不是 WT 序列长度。`h_V` 的行索引跟着这个顺序走。
3. **数据对齐后必须按文件数 / 行数核对**，不能只 `test -f`。
   曾经出现过 symlink 指向 worktree 外部、rsync 过去变成断链，而 `ls | wc -l` 只显示 1 的情况。

### 4.2 实验设计

4. **先检查目标函数的最优解是什么。** 这条线上已经有过两次教训：
   - 「让模型对 WT 序列的 likelihood 更高」这个 TTT 目标：最优解不是想要的东西，**已被否证**；
   - 无约束的「最大化两组分数分布差异」：最优解让分数**塌缩成界面特征的一个函数**，
     实测那个塌缩点是 **0.2564**，**比 0.3903 的基线还差 0.13**。
   **动手前先问：如果这个 loss 被完美优化，分数会变成什么？**
5. **任何「取最大」的统计量都必须配 null。** 扫超参、选配置、选最优 `c`，都会对样本过拟合。
   做法：把关键变量（如距离）随机置换后，跑**完全相同**的选择程序，把它的结果作为 null 报出来。
6. **判读标准要写成定量阈值，不要写成是非题。** 实测教训：预先写的是
   「置换 null 也涨 ⇒ 增益不来自先验」，结果 null 涨了、但只占总增益的 19% 且从不越线 ——
   二值判据在这里给不出结论。
7. **打分函数必须与评测用的逐行相同。** 改了模型路径之后，
   先验证「不训练时能否复现官方分数」。我们做过这个对照，最大相对偏差 **3.0e-7**（float32 累加量级）。
8. **口径不许混。** 每个数字都要说明是 14 / 23 / 25 assay；不同实验用不同子集时**分开报**。
   曾经把两次不同配置的运行拼在一起算均值，得出过错误的结论。
9. **单调不变性**：`Spearman(f(d), ·)` 对 `d` 的任意**单调**变换**恒定不变**。
   所以「换一个距离形状」（`exp(−d/τ)` → `1/(1+d/τ)` → `−d`）**改不动任何秩层面的量**，
   只对**加性组合**与 **Pearson** 有杠杆。别在这上面白费力气。
10. **label-free 的模型选择是这条线上反复出现的未解问题。**
    已测 10 个静态 WT 特征来预测最优超参，**9 个比用常数更差**，最好的只拿回 **28%** 的空间，
    且相关性**过不了 10 重检验的 BH 校正**。
    ProteinTTT 在结构预测上可以用 **pLDDT** 选最佳模型，**ProteinMPNN 这边至今没找到类比**。
    ⚠️ 但**只测过静态特征** —— 适配后模型**自身的行为**（自监督目标的 held-out 值、
    TTT 前后一致性、与独立预测器的 ensemble agreement）**从未测过**。

### 4.3 工程

11. **新增命令行参数后，必须先用最小配置 smoke test 那条确切路径再投长任务。**
    实测教训：没做这件事，三个子任务秒退，白等了 8.5 小时的 GPU 空档。
12. **M（解码顺序数）在 `AssayContext` 构造时就烙进了所有张量**（mask、randn 等）——
    只改 `ctx.M` 不重建会静默出错。
13. **`rsync -a` 从 workstation 回流会覆盖本地写的 record md**（它按差异传、不按新旧传）。
    回流必须 `--exclude '*.md' --update`。
14. **matplotlib 只有 DejaVu Sans 可用**，中文会**静默**变成豆腐块 ⇒ 图上的文字一律用英文。
15. **`pandas` 里 `median` / `agg` 这类名字会被属性遮蔽** —— `df.median` 返回的是方法不是列。
    列名别取这些，取了就用 `df["median"]`。
16. **stdout 会被缓冲** —— 长任务用 `python -u`，并让结果**逐步落盘**，
    否则跑几小时看不到任何进度，中途失败会丢掉已完成的部分。

---

## 附录 A：现成可复用的资产（绝对路径）

### A.1 代码（在 worktree `bindingGYM-binding-sites-analysis` 内）

| 东西 | 绝对路径 |
|---|---|
| **可微打分 + encoder 缓存**（`AssayContext` 已缓存 encoder 输出，改 `h_V` 后只需跑 decoder） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/scripts/mutation_landscape_ttt/bgmpnn.py` |
| **encoder 线性探针**（G-A，§2 的来源） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/scripts/mutation_landscape_ttt/ga_encoder_probe.py` |
| **no-op 对照**（验证打分函数与官方逐行一致） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/scripts/mutation_landscape_ttt/gb_noop_control.py` |
| 官方 `protein_mpnn_utils.py` 的 vendored 副本（md5 `56fc8e171b6d97dc9a048259f4eb3a77`） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/scripts/mutation_landscape_ttt/vendor/protein_mpnn_utils.py` |
| 界面计算流程（entity 划分、逐残基距离） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/scripts/binding_sites/` |

### A.2 数据产物（本地）

| 东西 | 绝对路径 |
|---|---|
| **逐残基**界面距离（重原子，全链，9,493 行） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-analysis/data/interface_residues_all_chains.csv` |
| **entity 划分**（由元数据确定，非结构启发式） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-analysis/data/entity_partition.csv` |
| 逐 variant 的界面标签 + MPNN zero-shot 分数（376,424 行） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet` |
| **逐突变位点**距离（1,173,273 对，可换任意聚合方式） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-analysis/data/variant_site_dists.parquet` |
| G-A 探针的逐 assay 结果（§2 的原始数据） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/workstation-records/mutation-landscape-TTT/data/ga_encoder_probe.csv` |
| 逐 assay 的 zero-shot ρ 与其他参照量 | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/workstation-records/mutation-landscape-TTT/data/t4_per_assay_bars.csv` |
| 14-assay 集合的推导与逐 assay 明细 | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/workstation-records/mutation-landscape-TTT/data/g1e_canonical14.csv` |

### A.3 BindingGYM 本体与模型权重

| 东西 | 绝对路径 |
|---|---|
| 官方代码仓库（评测口径与 split 的唯一权威） | `/home/guoj0f/repos/BindingGYM` |
| 官方 input（DMS csv + 结构 + 索引 `BindingGYM.csv`） | `/home/guoj0f/share/BindingGYM/input`（本地）<br>`/data/guoj0f/share/BindingGYM/input`（workstation） |
| 官方打分脚本 | `/data/guoj0f/share/BindingGYM/baselines/protein_mpnn/compute_fitness_multi_pdb.py`（workstation） |
| **ckpt `v_48_020.pt`**（md5 `91d54c97a68bf551114f8c74c785e90f`） | `/data/guoj0f/share/BindingGYM/training/cache/v_48_020.pt`（workstation） |
| 官方 zero-shot 参考分数（seed 1, M=5，329 MB，逐 variant） | `/data/guoj0f/BindingGYM-zero-shot-proteinMPNN/scores/seed1_M5/`（workstation） |
| conda env（numpy 1.24.4 / scipy 1.10.1，**别升级**） | workstation `bindinggym-zs-mpnn`（base 在 `/data/guoj0f/miniconda3`） |

### A.4 相关记录文档

| 内容 | 绝对路径 |
|---|---|
| binding-sites 定义、数据统计、真值与预测分布的逐 assay 对比 | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/binding-sites-overview/BindingGYM_binding_sites_overview_20260903.md` |
| 界面先验价值的摸底（含 rescoring 规则、`c` 的 label-free 选法否定结论） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/workstation-records/mutation-landscape-TTT/probe_binding_site_insight_value_20260909-160514.md` |
| decoder 侧 TTT 的现状与结论 | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/workstation-records/mutation-landscape-TTT/transduc_decoder_ttt_interface_divergence-organized.md` |
| decoder 侧 TTT 的操作日志（含全部踩坑细节） | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/workstation-records/mutation-landscape-TTT/transduc_decoder_ttt_interface_divergence-tracking.md` |
| BindingGYM 两个污染 assay 的调查 | `/home/guoj0f/repos/Sources/datasets/BindingGYM-issues` |
