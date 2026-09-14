# tracking — `structure-encoder-TTT` / `encoder_ttt_interface_binary`

给自己查的操作日志。**不要求前后逻辑顺畅，要求细节查得到。**
结论与框架看 `encoder_ttt_interface_binary-organized.md`。

**开始** 2026-09-14 ｜ **最后更新** 2026-09-14 ｜ **当前阶段**：计划定稿，代码未写，未开跑。

---

## 1. 运行日志

| 日期 | 任务 | SLURM job | wall | 配置 | 产出 | 结果 |
|---|---|---|---|---|---|---|
| 2026-09-14 | **T0 no-op gate**（ibex 首个作业） | **51886003** | **2:04** | `--steps 0 --M 5 --seed 1 --batch 64`，3 个 pilot assay | `data/t0_noop_per_assay.csv` | ✅ **四项全过** |

**T0 结果（2026-09-14，node 驱动 570.86.15，A100-SXM4-80GB，MaxRSS 4.07 GB）**

| 检查 | 结果 |
|---|---|
| cu117 能否在该驱动上 launch kernel | ✅ `cuda check: 61479.859375`（真实 matmul，非版本号推断） |
| ckpt md5 | ✅ `load_model` 的断言通过 |
| `--steps 0` 分数逐行不变 | ✅ **`max\|Δscore\| = 0.000e+00`**（容差 1e-6） |
| baseline ρ vs 换平台前记录值 | ✅ **`max\|d\| = 0.0020`** |

| assay | ibex `rho_base` | 记录 `ref` | 差 |
|---|---:|---:|---:|
| `5A12_Ang2_fitness_4ZFG` | 0.1093 | 0.1074 | +0.0019 |
| `PSD95_CRIPT_1BE9` | 0.3677 | 0.3672 | +0.0005 |
| `CD19_FMC63_Fitness_7URV` | 0.6032 | 0.6032 | **0.0000** |

⚠️ **换平台后不是逐位可复现，只是在噪声内一致。** `CD19` 完全相同说明 `randn` 的生成是一致的；
另两个的 0.0019/0.0005 远在噪声底之内（per-assay seed σ 中位 **0.0184**），但**不为零**。
对本实验无影响（两臂共用同一 `randn`、内部配对），但**不可声称跨平台逐位复现**。

**成本标定（据此外推，替换 organized 附录 B 的先验估计）**：
`CD19` 3,886 variant / L=497 / M=5 耗 **78 s** ⇒ **≈20 ms/variant**。
14-assay 全量 227,652 variant ⇒ **≈1.3 GPU-h/臂**（原估 2.5，偏保守约 2×）。
内存 4.07 GB ⇒ `--mem=64G` 过量，后续作业降到 `32G` 以利排队。

| 2026-09-14 | ~~P1 / P1b~~ | 51886700 / 51887033 | 8:33 / 6:24 | **表征锚方案，已作废** | — | 见 §1.1 |
| 2026-09-15 | **E-1 λ 扫描（分数锚）** | **51897226** | **11:13** | lr=1e-4 steps=150, λ ∈ {0.1,1,10,100}, anchor_batch=32, anchor_M=1，3 pilot assay | `data/e1_lam*_per_assay.csv` | ✅ **四档全部 3/3 为正** |

**E-1 结果（分数锚，3 个 pilot assay，mean rho_base 0.3601）**

| λ | mean Δρ | wins | mean AP_norm before→after | 备注 |
|---:|---:|:--:|---|---|
| 0.1 | +0.0089 | 3/3 | 0.4640 → 0.9000 | 锚最松，表征动得最多 |
| 1 | +0.0090 | 3/3 | 0.4640 → 0.6284 | |
| 10 | +0.0082 | 3/3 | 0.4640 → 0.5510 | |
| **100** | **+0.0118** | **3/3** | 0.4640 → 0.5383 | **最好**，且是扫描边界 |

逐 assay（λ=100）：`5A12_Ang2` +0.0019 ／ `PSD95_CRIPT` +0.0287 ／ `CD19_FMC63` +0.0047。

- ✅ **12/12 个 (λ, assay) 组合全部为正**，与表征锚版（±0.0014，wins 1–2/3）形成明确对比。
- 🔴 **但未达显著**：λ=100 逐 assay sd=0.0146、sem=0.0084 ⇒ t≈1.4；按 assay 符号检验 p=0.125。**n=3 撑不起「显著」。**
- 🔴 **`AP_norm` 与 Δρ 反向**：λ 越大 `AP_norm` 越低（0.900→0.538）而 Δρ 反而最高
  ⇒ **增益不是来自「界面更可分」**，来源未知。
- λ=100 是扫描边界，**更大的 λ 未测**。

**⇒ 下一步优先做 null 对照，不是扫更多 λ**（按「按能改变结论的程度排」）：
置换界面标签后若也涨同量级，增益与界面先验无关，则「learn a better interface representation」这个叙事不成立。

### 1.1 已作废的方案：表征锚（2026-09-15 由用户决定重新设计）

**旧目标**：`L = L_probe + λ(‖h_V−h_V^frozen‖²/(L'·128) + ‖h_E−h_E^frozen‖²/(L'·K·128))`

| λ | mean Δρ（3 pilot assay） | mean AP_norm before→after |
|---:|---:|---|
| 1 | −0.0014 | 0.4640 → 1.0000 |
| 10 | +0.0000 | 0.4640 → 1.0000 |
| 100 | −0.0002 | 0.4640 → 0.9982 |
| 1000 | −0.0002 | — |

🔴 **这批结果不构成对 H 的检验，不要引用它下任何关于 H 的结论。**
原因（用户 2026-09-15 指出，我确认）：**encoder 在全部残基上训练，没有任何标签层面的留出**，
而可训练参数 907,776 对 `L` = 56–1107 个残基（`PSD95_CRIPT` 为 **7,500 : 1**）——
**记住标签是平凡的，不需要任何几何理解**。用独立探针重测得到 `AP_norm=1.0`，
恰恰说明标签信息已被烤进 `h_V` 本身；而探针的 5-fold 留出的是 `h_V` 的行、不是标签，
**结构上就检测不到 encoder 层面的记忆**。⇒ 无法区分「学到几何」与「记住标签」。

**保留价值**：① 打分口径与 no-op gate 的验证（T0）仍然有效；
② 一条方法论记录 —— 在无留出、参数远多于样本时，辅助表征目标可以被完美拟合而下游毫无变化。

### 本地 smoke（A4500，**仅 sanity check，永不上报** —— ibex-usage Notes）

2026-09-14，`proteingym-ttt` env（torch 2.4.1），`PSD95_CRIPT_1BE9`，`--limit` 截断：

| 配置 | 结果 |
|---|---|
| `--steps 0 --limit 300 --M 2` | **NO-OP GATE PASS，max\|Δscore\| = 0.000e+00** |
| 同上 | 标签的 `pi=0.167` 与 proposal 表里 PSD95_CRIPT 的界面占比**精确吻合** ⇒ 标签构建走对了 |
| `--steps 60 --lr 1e-4 --lam 1.0 --limit 400 --M 2` | 跑通；probe loss **0.1303 → 0.0020**（60 步），anchor_V 2.6e-3 / anchor_E 3.6e-4 |

🔴 **smoke 暴露的问题：λ=1 几乎不构成约束。** probe loss 两个数量级的下降 vs anchor 停在 1e-3/1e-4
⇒ 目标比 λ=1 所能约束的**容易拟合得多**（PSD95 只有 120 个残基，128 维特征 + 可训练 encoder，且无留出集）。
**P2 的 λ 扫描范围要整体上移**（原计划 `{0.1, 1, 10}` 恐怕整段偏小）。由 pilot 定，不在此处拍板。

---

## 2. 决策记录（谁定的、什么时候、为什么）

| 日期 | 决策 | 谁 | 理由 |
|---|---|---|---|
| 2026-09-14 | project 名 `structure-encoder-TTT` | 用户 | — |
| 2026-09-14 | 门槛只要 **> zero-shot**（14-assay 0.3903），不比 rescoring 族 | 用户（早前定，proposal §0 记载） | 本条线的定位是「学更好的基础表征」给下游用 |
| 2026-09-14 | **14 assay 全量跑，不做 variant 子采样** | 用户 | 评测是 inference，成本可接受；子采样会引入不必要的噪声 |
| 2026-09-14 | `L_probe` **先用二分类**，soft label 留到之后 | 用户 | 当前 `AP_norm`=0.484，「碰不碰」本身就没分干净 |
| 2026-09-14 | loss **必须处理正负样本不均衡** | 用户 | 否则只在 AUC 视角好看，而优化空间在 AP_norm |
| 2026-09-14 | **不单独做 H 的干预验证**，直接看 TTT 后的 `ρ` | 用户 | `q` 上升是构造性的（训练目标就是它），不构成证据 |
| 2026-09-14 | lr / optimizer / steps 由 agent 自主决定 | 用户授权 | 取值与理由见 organized 附录 A |
| 2026-09-14 | **撤回**原计划的 T1（partner 坐标加噪阶梯） | agent，用户质疑后 | 它干预的是「partner 几何信息量」而非 `q`，没有隔离出自变量；且 single-chain 端点与 partner-blind 结论已存在，属重复已知结论 |
| 2026-09-14 | **撤回**原计划的 T2（编辑 `h_V` 估上界） | agent | 形状上是 `s0_intervention` 的近亲，而后者被用户判定不可信、原因未记录。新方案让真实训练去动 `h_V`，绕开这个未知 |
| 2026-09-14 | single-chain 表的含义**不再追究** | 用户 | 所有实验都在 complex-level 做，用不到它 |
| 2026-09-14 | 只训 `encoder_layers`+`W_e`（907,776 / 54.7%），**冻结 `features`** | agent | `features` 是坐标→RBF 的几何入口，改它等于改结构表示的基底；界面编码由消息传递层决定。留 `--train_features` 开关默认关 |
| 2026-09-14 | **anchor 从只锚 `h_V` 改为同时锚 `h_V` 与 `h_E`** | 用户提问 → agent 查代码确认 | `L_probe` 只读 `h_V`，但 `EncLayer` 每层同时更新两者，且 `h_E` 有两条路径直达 decoder。只锚 node 则 `h_E` 可任意漂移而 loss 无感，而它的元素数是 `h_V` 的 **48 倍**（K=48）。**这是 review 抓出的第一个实质设计缺陷** |
| 2026-09-14 | 新增 organized §7「深入优化模型设计」，**仅为分析、不改当前实验计划** | agent（用户要求） | 从 edge 视角审视注入点；含实测诊断：entity 跨多链的 assay 仅 3/25、14-assay 里仅 1 个 ⇒ 「chain≠entity」缺口**降级为次要**（我原以为它是主要缺口） |
| 2026-09-14 | **执行平台 workstation → ibex**；记录目录 `workstation-records/` → `ibex-records/`；env 用已有的 `bgym-official` | 用户 | workstation 的 sshd 间歇不可用（§6.2）；ibex 上数据/ckpt 已就位且 md5 正确，`bgym-official` 的 numpy/scipy 与 workstation 那个 env 逐位相同 |
| 2026-09-14 | organized §7（edge 视角的深入优化）**暂缓，不进入本轮实验** | 用户 | review 后决定先推进既定计划 |
| 2026-09-15 | **锚从表征层面改到分数层面**：`λ‖s_θ−s_frozen‖²/var(s_frozen)` | 用户 | 表征锚约束太紧且是坏代理 —— 它罚「表征移动多远」而非「移动有没有抵达 decoder」；我们真正在乎的是分数排序。形式取自 decoder 侧已验证有效的目标 |
| 2026-09-15 | **删除 organized 中表征锚方案的全部实验结果**，从头开始 | 用户 | 那批结果无法区分「学到几何」与「记住标签」，说明不了什么。运行记录压缩留在 §1.1 |
| 2026-09-15 | **删除文档中 h_E（edge feature）的注入方案** | 用户 | 判断其价值不足，且干扰阅读。**保留** `h_E` 的实现要求（重建 cache），那是正确性问题不是方案 |
| 2026-09-15 | 若分数锚版仍无效，`L_probe` 改为对 `w(r)=exp(−d/5Å)` 的**回归** | 用户 | 二值只说「碰不碰」，soft label 保留「多近」；且连续距离在 23/23 assay 上都有变化 |

---

## 3. 实现要点（写代码时逐条对照）

### 3.1 必须做对的五件事

1. **有效残基过滤**：`ok = (mask > 0) & (不是 parse_PDB 的缺口填充位)`。
   填充位判定复用 `ga2_encoder_probe.py::pdb_chain_slots`（返回 `None` 即填充位）。
   `L_probe`、`L_anchor`、`π`、`w_pos` **全部只在 `ok` 上算**。
   实测非零填充位的 assay 只有 4 个：`KRAS_PICK3CG-RBD` 192/1107、`HLA-A2_TAPBPR` 62/644、
   `CD19_FMC63` 52/497、`5A12_Ang2` 4/652；其余 10 个为 0。
2. 🔴 **TTT 后必须重建整个 encoder cache，不能只 `set_h_V()`**。
   `AssayContext._build_encoder_cache()` 同时产出 `h_V`、`h_E`、`E_idx`，并由 `h_V`+`h_E`
   构造 `h_EXV_encoder` 与 `h_EXV_fw`。`set_h_V()` 只重建依赖 `h_V` 的那部分，**保留旧 `h_E`** ——
   它是为「编辑表征」的干预写的，对「重训 encoder」是错的。
   做法：用训好的权重、**同一个 `randn`**，重新构造 `AssayContext`。
   **代码依据**：`EncLayer.forward` 结尾 `h_E = self.norm3(h_E + self.dropout3(h_message))` 然后 `return h_V, h_E`；
   `ProteinMPNN.forward` 里 `h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)`（→ `bw` 项）与
   `h_EX_encoder = cat_neighbors_nodes(zeros_like(h_S), h_E, E_idx)`（→ `fw` 项）—— **`h_E` 两条路径都进 decoder**。
2b. 🔴 **锚必须与被锚的量同口径**（2026-09-15，锚改到分数层面后的实测教训）。
   评测分数是 **M=5 个解码顺序的平均**，而训练每步只采**一个**顺序。拿单序去锚 M 平均，
   `step 0` 的锚就不是 0 —— **实测 0.845**，整个训练被「模仿 M 平均」主导，
   probe loss 几乎不降（0.1303→0.1307），λ=1 与 λ=100 给出几乎相同的结果。
   **修法**：预先算 `(M, N)` 的**逐解码顺序**冻结分数（`score_all_per_m`），每步用**对应那一行**。
   修完实测 `step 0` 锚 = **3.48e-11**。
   ⇒ **判据固化**：`step 0` 的锚必须 ≈0，任何实现改动后都要重验这一条。
3b. **训练中的可微打分不能读缓存。** `ctx.score()` 用的是缓存的 `h_V`/`h_E`；训练时它们是可微的，
   必须用 `score_with()` 从当前 `h_V`/`h_E` **重建** `h_EXV_encoder`/`h_EXV_fw` 再跑 decoder，
   否则打的是预训练表征的分，梯度到不了 encoder，且**不会报错**。

### 3.2 数值与工程
- `python -u`，结果**逐 assay 落盘**，不要跑完才写。
- 图上文字一律英文（本机 matplotlib 只有 DejaVu Sans，中文静默变豆腐块）。
- `pandas` 里别用 `median` 之类会被方法名遮蔽的列名。
- 回流用 `rsync --exclude '*.md' --update`（`-a` 按差异传不按新旧传，会把本地写的 record 覆盖回旧版；
  2026-09-13 实测事故：574 行被覆盖成 345 行）。

## 4. 已知缺陷（读结果时必须带着）

0. 🔴 **P1 的零结果尚不可归因。** 我们从未验证过训练是否真的提高了 `h_V` 的界面可分性。smoke 里 probe loss 从 0.1303 降到 0.0020，但那是**训练集内**的值 —— 只有 120~500 个残基、128 维特征、head 与 encoder 联合优化、**没有留出集**。完全可能是 `h_V` 被推到一个让**那一个 head** 好分的方向，而非学到可泛化的界面几何。若如此，P1 测的是「无效干预 → zero-shot」，与 H 无关。**P1b（job 51887033）用独立 5-fold 探针在 TTT 前后各测一次 `AP_norm` 来判定归属。**
1. **P1/P2 是 test-set tuning**。BindingGYM 没有 dev split 也造不出来；pilot 的 3 个 assay
   包含在 M1 的 14 个里。缓解：主实验只用单一共享超参、不逐 assay 调。任何对外数字必须标注这一点。
2. **本任务不检验 H**。M1 成功只说明「这套做法有效」，不说明「encoder 界面表征 → zero-shot」这条因果链成立；
   M1 失败也无法区分 H 假 vs 实现/选参不好。
3. **`q` 的上升是构造性的**，不作为证据（用户明确指出）。
4. **界面只算在唯一一个 WT 复合物结构上**，22 个结构里 15 个是 `_hm` 同源模型，未建模突变引起的构象变化。
5. **`s0_intervention` 被判不可信的原因至今未知** —— 若新方案的**读数方式**（编辑 `h_V` 后重打分）
   与它共用了某个有问题的环节，会原样继承。目前只知道它的干预是 `h' = h + α(u·h)u`，
   对照为随机方向与 shuffle，按 induced Frobenius norm 匹配。

---

## 5. 资产与路径

### 5.1 直接复用的代码（在 `bindingGYM-binding-sites-analysis` worktree）
| 用途 | 路径 |
|---|---|
| 可微打分 + encoder 缓存（`AssayContext`） | `.../scripts/mutation_landscape_ttt/bgmpnn.py` |
| 现行探针（重原子 5 Å，已排除缺口）+ `pdb_chain_slots` | `.../scripts/mutation_landscape_ttt/ga2_encoder_probe.py` |
| no-op 对照（验证打分与官方逐行一致） | `.../scripts/mutation_landscape_ttt/gb_noop_control.py` |
| vendored `protein_mpnn_utils.py`（md5 `56fc8e171b6d97dc9a048259f4eb3a77`） | `.../scripts/mutation_landscape_ttt/vendor/` |

前缀 = `/home/guoj0f/repos/ProteinTTT/.claude/worktrees/bindingGYM-binding-sites-analysis`

> ⚠️ **本 worktree（`structure-encoder-TTT`）的 `workstation-records/` 与 `local-records/` 已于 2026-09-14 整体删除**
> （commit `b40eaf8`，用户要求）。上表所有文件的权威副本在 `bindingGYM-binding-sites-analysis` worktree；
> 按 ibex-usage §1c-3「数据就地化」，代码要**复制**进本 worktree，不要跨 worktree 引用。

### 5.2 数据与 env（**ibex**，2026-09-14 实测确认）
| 东西 | 路径 |
|---|---|
| BindingGYM input（`BindingGYM.csv` + 28 个 DMS csv + 22 个 structures） | `/ibex/user/guoj0f/share/BindingGYM/input` |
| ckpt `v_48_020.pt`（md5 **实测** `91d54c97a68bf551114f8c74c785e90f` ✅ 正确那份） | `/ibex/user/guoj0f/share/BindingGYM/training/cache/v_48_020.pt` |
| 官方打分脚本 | `/ibex/user/guoj0f/share/BindingGYM/baselines/protein_mpnn/compute_fitness_multi_pdb.py` |
| conda base | `/ibex/user/guoj0f/anaconda3` |
| **conda env** | **`bgym-official`** |
| 本分支的 ibex 代码目录（§1c-2 按分支隔离） | `/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/` |

**env 选型（2026-09-14）**：用 ibex 已有的 **`bgym-official`**，不新建。
实测 `py 3.8.20 / torch 1.13.1+cu117 / numpy 1.24.4 / scipy 1.10.1 / sklearn 1.3.2 / pandas 2.0.3`。
- **numpy 1.24.4 + scipy 1.10.1 与 workstation 上 `bindinggym-zs-mpnn` 逐位相同** ⇒ 数值环境一致，不引入新变量。
- sklearn 1.3.2 是 ibex-usage §1b-0 里**已验证等价**的两个版本之一（`1.2.1`/`1.3.2`）。
- 按 §4 env 粒度例外①（同一大类且 env 未变），**不为本 project 另建 env**。
- ⚠️ torch 1.13.1 是 **cu117（CUDA major 11）**。skill 禁止的是「新 runtime + 旧驱动」（cu13x 撞 12.x）；
  这里是**旧 runtime + 新驱动**，方向相反、属兼容侧 —— 但仍**必须在 a100 上实测** `torch.cuda.is_available()`
  ＋ 一次真实 kernel launch 才算数（见 §1 运行日志 T0）。

⚠️ **ibex 上没有** workstation 那份 `/data/guoj0f/BindingGYM-zero-shot-proteinMPNN/scores/seed1_M5/`
（官方 zero-shot 逐 variant 分数，329 MB）。**本任务不依赖它** —— baseline 臂必须用与 TTT 臂
**同一个 `randn`** 现算，才能配对（§3.1 第 3 条），旧分数的解码顺序对不上，本来也不能直接用。

### 5.3 参照数值
| 东西 | 路径 |
|---|---|
| 探针逐 assay 结果（`AP_norm` 等） | `.../workstation-records/mutation-landscape-TTT/data/ga2_encoder_probe.csv` |
| 各 assay 填充位计数 | `.../workstation-records/mutation-landscape-TTT/data/ga2_coverage.csv` |
| 14-assay 集合推导 + 逐 assay zero-shot ρ | `.../workstation-records/mutation-landscape-TTT/data/g1e_canonical14.csv` |

---

## 6. 平台与环境实况

**2026-09-14：本任务的执行平台从 workstation 改为 ibex**（用户决定），记录目录随之
`workstation-records/` → `ibex-records/`。

### 6.1 ibex a100 实况（2026-09-14 实测）

- **可调度节点上 a100 共 240 张，已分配 236 张，空闲 4 张**（98.3% 占用）；
  另有 1 节点 `drained`、1 节点 `reserved` 不可用。
- 队列：**81** 个 pending 作业申请 a100，176 个 running 占用。我自己无作业。
- 空闲的 4 张是 `gpu108-09-r` / `gpu108-23-r` / `gpu109-16-r` / `gpu202-02-r` **每节点各 1 张**的碎片
  ⇒ 对多卡作业无用，**对我们的单卡 + 短 walltime 作业恰好是最容易被 backfill 的形态**。
- 存储：`/ibex/user` 配额 **1.5 T，已用 268 G，余 1.3 T**。

### 6.2 为什么不用 workstation —— sshd 间歇性不可用（2026-09-14 实测）

同一天内 ssh 到 `10.67.24.41` 出现三种表现：① 首次预检**成功**；
② 随后 `Permission denied (publickey,password)`；③ 再后来直接超时。`ssh -v` 定位到真实原因：

```
debug1: identity file /home/guoj0f/.ssh/id_ed25519 type 3   <- key 正常读到
debug1: Connection established.                             <- TCP 已建立
Connection timed out during banner exchange                 <- sshd 未在超时内回 banner
```

`/dev/tcp/10.67.24.41/22` 可达 ⇒ **不是网络不通、不是 key 问题、不是本地沙箱**，
而是**远端 sshd 响应不过来**（56 核共享机，GPU util 0% 但 CPU/IO 可能被他人占满）。
它会让 `nohup` 启动、监控轮询、结果回流都间歇性失败 —— 这是改用 ibex 的直接原因之一。
（该机当时的另一项实况：`/home` 已 99%、只剩 96 GB。）

---

## 7. 口径变更史

| 日期 | 变更 | 说明 |
|---|---|---|
| 2026-09-14 | 本任务一律 **14 assay / 基线 0.3903** | 二值界面标签在另外 9 个 assay 上恒定（全碰或全不碰），无法评 |
| 2026-09-14 | 确认 **14 ⊂ 16**，差集恰为两个 label 污染的 KRAS assay | 16 = 25 中界面两侧各 ≥30 者；14 = 23 中同条件者。此关系此前未被任何文档写明，本次实测闭合 |

---

## 8. 复现

```bash
# 尚未有可运行脚本。计划中的入口：
#   scripts/structure_encoder_ttt/e1_encoder_ttt.py   （TTT + 重打分，单 assay）
#   ibex-records/structure-encoder-TTT/sh/{task}_{dt}.sh        （sbatch 脚本）
```
