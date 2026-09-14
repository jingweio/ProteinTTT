# Geometry-Counterfactual encoder-TTT：动机、评估与可执行方案

**project** `geometry-counterfactural-encoder-TTT` ｜ **首个 task** `g1_interface_sensitivity`
**开始** 2026-09-15 ｜ **最后更新** 2026-09-15
**状态：G-1 gate ✅ 通过（14 assay，29 秒）—— 信号存在且强；G-2（pull vs slide）在跑。建模尚未开始。**

| gate | job | 结论 |
|---|---|---|
| **G-1** | 51897603 | ✅ sanity `max\|gap\|=3.7e-04`；1 Å 平移 **+1.30 sd (13/14)**、5° 旋转 **+1.00 sd (13/14)** ⇒ **前提风险推翻** |
| **G-2** | 51897764 | ⏳ 分离「拉开」与「重新配对」 |

🔴 **G-1 同时暴露一个相反的问题：信号太强**（8 Å 下 `ACE2` **+45 sd**）⇒ negative 太易区分、学不到东西。
**可用的 hard negative 在 0.5–1 Å / 2–5°**，不是原文档设想的大幅扰动。

> **与 `structure-encoder-TTT` 的关系**：**两个独立 project**，同一分支同一 worktree，
> 按 ibex-usage §1c-4 用 `project_name` 区分。共用 `scripts/structure_encoder_ttt/` 下的
> `bgmpnn` / `bgpdb` 模块（只 import，不修改），记录、脚本、产出完全分开。
> **评测口径刻意保持一致（同一批 14 assay、同一基线 0.3903），以便横向对比效果的 notable 程度。**

---

## 1. 原始动机（用户 2026-09-15 提出）

来源：`/home/guoj0f/repos/Sources/complexTTT-doc/ComplexTTT_Geometry_Counterfactual_Motivation-cn.md`

核心主张：ComplexTTT 不应做成 **WT-complex memorization**，而应是
**test-time identification of the pair-specific structural constraints governing mutational effects**。

**Geometry-Counterfactual TTT**：从 WT complex 出发，对 **binding interface** 做局部结构扰动，
构造「interface-disrupted but globally plausible」的 structural negatives，然后要求

| 方向 | 约束 |
|---|---|
| **structure-side** | `Compat(X_AB, S_AB) > Compat(X̆_AB^(i), S_AB)` —— 固定 WT 序列，扰动结构 |
| **sequence-side** | `Compat(X_AB, S_AB) > Compat(X_AB, S̆_AB^(i))` —— 固定 WT 结构，用扰动结构诱导的序列作 negative |

## 2. 评估

### 2.1 为什么这个想法值得做

1. 🔴 **它引入新信息，而不是重排已有信息。** 这是它比 `structure-encoder-TTT` 强的根本之处：
   那条线的界面二分类目标只是把 encoder **已经看到**的几何换个方向摆，而 counterfactual
   **制造了 WT complex 本身不存在的数据点**。
2. **动机与本项目已有实测一致。** 「WT-likelihood ≠ binding-effect specialization」不是推测：
   BH3_Mcl-1 上 ρ 随 lr 单调下降 **0.705 → 0.694 → 0.648 → 0.388**，无操作窗口，
   根因是该目标的最优解正是打分公式自己的分母。
3. **contrastive 形式天然避开塌缩。** 本线两次失败（WT-likelihood、无约束散度）都是
   「最优解是退化解」，后者实测塌缩到 **0.2564**（比基线差 0.13）。contrast 的最优解是把
   positive 与 negative 分开，不存在这种退化。
4. **目标天然经过 decoder。** `Compat = log p(S|X)` 必须跑 decoder ⇒ 梯度作用在
   「decoder 看得见的东西」上。这恰是 `structure-encoder-TTT` 旧方案缺的一环。

### 2.2 三个必须处理的问题（评估时提出）

| # | 问题 | 处理 |
|---|---|---|
| **1** | **前提风险**：本项目有**两条独立证据**表明 ProteinMPNN 在 BindingGYM 上基本 partner-blind（KRAS 换 label 时 3 行只有 1 行对角线最高；全 25 assay 界面对比中真值 δ 16/16 为负而 MPNN 仅 8/16）。若模型本就不看 partner ⇒ 扰动无效果 ⇒ 整个方案无信号 | **G-1 gate**，零训练先测 |
| **2** | **`Compat` 未定义**，取 `log p_θ(S\|X)` 则 structure-side 可经由「抬高 `p(S_WT\|X_WT)`」优化 —— **那正是已被否证的 WT-likelihood** | 实验必须**分解增益**来自「抬高 positive」还是「压低 negative」 |
| **3** | **扰动方式决定 negative 质量**：随机扰坐标会产生 clash 与断裂，模型可用「结构不合理」这一 trivial 特征区分 | **只用刚体微动**（见 §3） |

---

## 3. 系统

```
              ┌── X_AB              (WT)                 ──┐
WT complex ───┤                                            ├──[encoder E_θ]──[decoder D_φ 冻结]──→ Compat(X,S)=log p_θ(S|X)
              └── X̆_AB^(i) = T_i ∘ X_AB   (entity B 刚体微动) ──┘
                     T_i：绕 entity B 自身质心旋转 θ° ＋ 平移 t Å
```

**为什么只用刚体微动**：它保持**每条链内部几何完全不变**（链内距离、二面角一律不动），
唯一改变的是 A–B 相对位姿 ⇒ **positive 与 negative 的唯一差异就是 interface geometry**，完美受控。

| 符号 | 定义 | 训练可见 |
|---|---|---|
| `T_i` | entity B 的刚体变换（平移 `t` Å / 绕自身质心旋转 `θ`°） | ✅ 纯几何，零 DMS label |
| `Compat(X,S)` | `log p_θ(S\|X)` ＝ **官方 BindingGYM 打分函数**（负 NLL 求和、无 WT 项、无长度归一） | — |
| `sd_mpnn` | 该 assay 内**预训练 MPNN 分数**的标准差（`refs/g1e_canonical14.csv`） | ✅ |
| `gap` | `Compat(X_WT,S_WT) − Compat(X̆,S_WT)`，并同时报 `gap / sd_mpnn` | — |
| `ρ` | per-assay Spearman → **14 assay** 未加权平均，基线 **0.3903** | ❌ 只在评测用 |

**为什么要 `gap / sd_mpnn`**：它回答「破坏界面相当于多少个**突变的标准差**」——
即这个扰动相对于 benchmark 真正在排序的那种变异有多大。`gap` 远小于 1 sd 意味着
把整个界面破坏掉还不如一个普通单点突变重要。

---

## 4. 实验网格

| id | 问题 | 做法 | 状态 |
|---|---|---|---|
| **G-1** 🔴 | **模型对 interface 扰动敏不敏感**（structure-side 信号是否存在） | 冻结模型，`t ∈ {0.5,1,2,4,8} Å`、`θ ∈ {2,5,10}°`，各 3 个随机方向，**只打 WT 序列** | **已提交 job 51897603，14 assay** |
| **G-1s** | **sanity**：整体刚体变换 | ProteinMPNN 只读距离 ⇒ 严格 SE(3) 不变 ⇒ **gap 必须 ≈0** | 含在 G-1 内 |
| **G-2** 🔴 | **拉开 vs 滑动**：gap 究竟来自「不再接触」还是「pair-specific 配对被改变」 | 新增「沿界面滑动」扰动 —— **保持接触面积、只改残基配对** | 待加（见 §6） |
| **G-3** | sequence-side 的 negative 有区别吗 | 对 `X̆` 采样 `S̆`，测与 `S_WT` 的 Hamming，**分界面/非界面残基报** | 待做 |
| **E-1** | structure-side TTT | margin loss ＋ 分数锚 | G-1/G-2 通过后 |

**递进**：G-1 确认信号存在 → G-2 确认信号来自 pair-specific 而非「接触与否」→ 才谈 TTT。

---

## 5. 判据

1. **G-1s sanity 必须过**：整体刚体变换的 `|gap|` 必须 ≈0。
   本地 smoke 实测 **1.2e-04**，而信号 gap 是 10+ nats ⇒ 扰动代码正确。
   ⚠️ 若这一项不为 0，说明是扰动代码错了，**不是模型敏感**。
2. **G-1 通过的条件**：在**仍然 plausible 的小扰动区间**（1–2 Å）有正的、随幅度单调的 gap。
   若只有大扰动（≥4 Å）才有 gap ⇒ 测到的是「两条链被拉开」，不是 pair-specific constraint。
3. **G-2 是 G-1 的必要补充**：只有滑动也产生 gap，才说明模型在建模 pair-specific 配对。

---

## 6. 🔴 已识别的混淆（G-1 之后必须分离）

**大幅扰动的 gap 主要来自「不再接触」，而非「pair-specific 配对被改变」。**

本地 smoke（A4500，仅 sanity check，不上报）已显示这个模式：

| 平移 | gap / sd |
|---:|---:|
| 0.5 Å | **−0.088**（无信号，甚至反向） |
| 1.0 Å | +0.256 |
| 2.0 Å | +2.166 |
| 8.0 Å | +4.526 |

8 Å 会把 entity B 整个拉开，界面接触大面积断开（界面判据是 5 Å）——
此时 gap 大是**平凡**的，不能作为「模型懂 pair-specific interaction」的证据。

**分离办法（G-2）**：增加**沿界面平面滑动**的扰动 —— 保持两链接触面积基本不变，
只改变具体的残基配对。
- 滑动也有 gap ⇒ 模型确实建模 pair-specific 关系 ⇒ counterfactual 有意义
- 只有拉开才有 gap ⇒ 模型只在测「接触与否」⇒ **negative 退化成「单体 vs 复合物」**，
  而那个对照 BindingGYM 官方已有（single-chain 表，25-assay 0.3564 vs 0.3970），不需要 TTT

---

## 7. 下一步（按能改变结论的程度排）

1. **G-2 拉开 vs 滑动** —— 它能证伪「这个方案在学 pair-specific constraint」这个核心主张。
2. **G-3 sequence-side 的信号强度** —— 若 `S̆ ≈ S_WT`，sequence-side 那一半目标恒满足、无梯度。
3. **E-1 建模** —— 且必须带上评估 §2.2 问题 2 的分解：增益来自抬高 positive（≈WT-likelihood，已否证）
   还是压低 negative（新东西）。
