# tracking — `geometry-counterfactural-encoder-TTT` / `g1_interface_sensitivity`

给自己查的操作日志。结论与框架看 `g1_interface_sensitivity-organized.md`。
**开始** 2026-09-15 ｜ **最后更新** 2026-09-15

## 1. 运行日志

| 日期 | 任务 | SLURM job | wall | 配置 | 产出 | 结果 |
|---|---|---|---|---|---|---|
| 2026-09-15 | **G-1 interface sensitivity** | **51897603** | **0:29** | `--M 5 --seed 1 --reps 3`，**14 assay 全量** | `data/g1_interface_sensitivity.csv` | ✅ **PASS，信号很强** |
| 2026-09-15 | **G-2 pull vs slide** | **51897764** | — | `--reps 5 --mags 0.5 1 2 4`，14 assay | `data/g2_pull_vs_slide.csv` | ⏳ |

**G-1 结果（14 assay，sanity `max|gap| = 3.662e-04` ⇒ 扰动代码正确）**

| 扰动 | gap (nats) | **gap / sd_mpnn** | 正的 assay |
|---|---:|---:|:--:|
| translate 0.5 Å | +0.904 | **+0.247** | 11/14 |
| translate 1.0 Å | +4.034 | **+1.303** | 13/14 |
| translate 2.0 Å | +13.524 | +4.615 | 14/14 |
| translate 4.0 Å | +24.048 | +8.623 | 14/14 |
| translate 8.0 Å | +38.793 | +14.030 | 14/14 |
| rotate 2° | +0.603 | +0.153 | 12/14 |
| rotate 5° | +3.346 | **+1.004** | 13/14 |
| rotate 10° | +9.584 | +3.208 | 14/14 |

✅ **前提风险被推翻**：在仍然 plausible 的小扰动区间（1 Å / 5°）就有一致信号，
不是只有把两条链拉开才有反应。

🔴 **但出现一个相反方向的新问题：信号太强。**
`ACE2_SARS2-RBD` 在 8 Å 下 **+45.0 sd**、`KRAS_RAF1` +26.9 sd。
对比学习里这意味着 **negative 太容易区分** —— 模型无需学任何精细约束就能分开，因而学不到东西。
⇒ **可用的 hard negative 在 0.5–1 Å / 2–5°**（+0.15 ~ +1.3 sd），**不是原文档设想的大幅扰动**。
逐 assay 差异也大（GB1 仅 +2.8 sd vs ACE2 +45.0 sd），建模时幅度可能需要**逐 assay 标定**。

### 本地 smoke（A4500，**仅 sanity check，永不上报** —— ibex-usage Notes）
2 个 assay、`--M 2 --reps 2`：sanity `max|gap| = 1.221e-04`（信号 gap 10+ nats）⇒ 扰动代码正确。
平移 gap/sd：0.5 Å **−0.088** ／ 1 Å +0.256 ／ 2 Å +2.166 ／ 4 Å +3.523 ／ 8 Å +4.526；
旋转 gap/sd：2° +0.338 ／ 5° +0.943 ／ 10° +2.044。

## 2. 决策记录

| 日期 | 决策 | 谁 | 理由 |
|---|---|---|---|
| 2026-09-15 | project 名 `geometry-counterfactural-encoder-TTT` | 用户 | — |
| 2026-09-15 | **在与 `structure-encoder-TTT` 相同的 14 assay 上做** | 用户 | 便于横向对比效果的 notable 程度 |
| 2026-09-15 | 与 `structure-encoder-TTT` **同分支同 worktree，用 project_name 区分** | agent（依 ibex-usage §1c-4） | skill 明确要求不为相关实验新建 worktree |
| 2026-09-15 | 扰动**只用刚体微动**，不用随机坐标噪声 | agent | 刚体保持链内几何完全不变 ⇒ 唯一变量是 A–B 相对位姿；随机噪声会产生 clash，模型可用「结构不合理」这个 trivial 特征区分 |
| 2026-09-15 | 先跑零训练 gate，不直接建模 | agent | 两条独立证据显示 MPNN 基本 partner-blind，若成立则整个方案无信号 |

## 3. 实现要点

1. **质心只用真实残基算**。`parse_PDB` 的缺口填充位坐标是 `(0,0,0)`，计入质心会把它拉偏。
   代码里用 `real & (mask>0)` 过滤后再求 mean。
2. **扰动后必须 `ctx._build_encoder_cache()`**，因为坐标变了 ⇒ `features()` 会重算 kNN 图（`E_idx`）。
   这是对的：扰动确实应当改变近邻关系。
3. **`randn` 不变** ⇒ 解码顺序与 WT 臂一致，gap 是配对量。
4. **每个 assay 跑完要把 `ctx.X` 复位并重建 cache**，否则下一个配置在已扰动的结构上叠加。
5. G-1 只打 **WT 序列一条**，不是全部 variant ⇒ 成本极低（14 assay × 11 配置 × 3 rep）。

## 4. 已知缺陷（读结果时必须带着）

1. 🔴 **大幅扰动的 gap 可能来自「不再接触」而非 pair-specific 配对被改变**。
   见 organized §6。**G-1 单独不足以支持「模型懂 pair-specific interaction」这个主张**，
   必须配 G-2（沿界面滑动）。
2. **`Compat` 取 `log p(S|X)` 时，structure-side 目标可经由抬高 `p(S_WT|X_WT)` 优化，
   而那正是已被否证的 WT-likelihood。** 建模阶段必须分解增益来源。
3. 0.5 Å 扰动在 smoke 里给出**负** gap ⇒ 极小扰动没有信号或方向不稳，
   报结论时不要把整条曲线当成单调的。

## 5. 资产与路径

| 东西 | 路径 |
|---|---|
| 本 project 代码 | `scripts/geometry_counterfactual_ttt/` |
| 共用模块（**只 import 不修改**） | `scripts/structure_encoder_ttt/{bgmpnn,bgpdb}.py` |
| 14-assay 名单与 `sd_mpnn` | `scripts/structure_encoder_ttt/refs/g1e_canonical14.csv` |
| entity 划分 | `scripts/structure_encoder_ttt/refs/entity_partition.csv` |
| 🔴 **14 assay 的权威定义** | `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis/local-records/structure-TTT/structure_ttt_encoder-proposal.md` §1.3 |
| 原始想法文档 | `/home/guoj0f/repos/Sources/complexTTT-doc/ComplexTTT_Geometry_Counterfactual_Motivation-cn.md` |

> **纪律（用户 2026-09-15 明确要求）**：每次用到「14 assay」都回上面那份 proposal §1.3 核验，
> 不凭记忆。2026-09-15 已核：proposal 标 ✅ 的 14 个与 `refs/g1e_canonical14.csv` **逐个吻合**。
