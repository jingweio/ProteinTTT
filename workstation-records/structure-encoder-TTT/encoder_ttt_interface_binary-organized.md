# encoder-TTT（界面二分类目标）：在 test time 只用结构改进 ProteinMPNN 的 encoder

**project** `structure-encoder-TTT` ｜ **task** `encoder_ttt_interface_binary`
**开始** 2026-09-14 ｜ **最后更新** 2026-09-14
**状态：计划已定稿，等 review 后开跑。尚无任何结果。**

---

## 1. 问题与 hypothesis

给定某个 assay 的 **WT complex structure**，在 test time **只用结构本身（零 DMS label）**
把 ProteinMPNN 的 **encoder** 对 binding-sites 的几何编码得更好，从而提升 BindingGYM zero-shot。

**H**：把 `h_V` 中「界面 / 非界面」的可分性训上去，per-assay Spearman 会变好。

**效果门槛（用户定死）**：**只要优于预训练 zero-shot 即可** —— 14-assay **0.3903**。
本条线的定位是**学一个更好的基础表征**给下游 mutation-landscape-TTT 用，
**不要求**去比 rescoring 那一族更强的后处理（那边的门槛是 0.4689 / 0.4551，与此处无关）。

**起点证据**（来自 `structure_ttt_encoder-proposal.md`，非本次产出）：
冻结 encoder 的 `h_V` 上训线性探针，界面信息**线性可读但只读出一半** ——
`AP_norm,within` 均值 **0.484**（AUC 0.842）。**优化空间在 AP_norm，不在 AUC。**

---

## 2. 系统

```
                            ┌─→ h_V  (L×128)    ─┬─→ [probe head g_ψ] ──→ 界面 logits  ← 训练信号（零 DMS label）
WT complex X ─[encoder E_θ]─┤                    │
                            └─→ h_E  (L×K×128) ─┴─→ [decoder D_φ，全程冻结] ──→ score(v)  ← 只在评测用
                                 K = k_neighbors = 48                              │
                                                        per-assay Spearman → 14-assay 未加权平均 = ρ
```

🔴 **`encoder` 输出的是 `h_V` 和 `h_E` 两个量，而 probe 只看得见 `h_V`，decoder 两个都吃。**
`EncLayer.forward` 每层交替更新两者（先用 `(h_V,h_E)` 更新 `h_V`，再用**新的** `h_V` 更新 `h_E`），
所以**微调 encoder 必然同时改变两者，无法只动其一**。decoder 侧 `h_E` 走两条路径
（`h_ES` 带突变序列 embedding 进 `bw` 项；`h_EX_encoder` 序列位清零进 `fw` 项），`h_V` 也走两条
（decoder 的初始 node 状态；经 `h_EXV_encoder` 进 `fw` 项）。**这个不对称直接决定了 §2.2 的 anchor 怎么写。**

> **规模**：encoder 侧可训练参数 **907,776 / 1,660,485 = 54.7%**，比 decoder 侧（41.9%）还多 ——
> **容量不是这条线的瓶颈**。

### 2.1 变量与可见性

| 符号 | 定义 | **训练时可见吗** |
|---|---|---|
| `h_V` | encoder 的 **node** 输出，每残基 128 维；`L` = 复合物打包后的残基槽位数（56–1107） | — |
| `h_E` | encoder 的 **edge** 输出，`(L, K=48, 128)` —— **元素数是 `h_V` 的 48 倍**；probe 看不见它，decoder 看得见 | — |
| `y_r` | `1[ d(r) ≤ 5.0 Å ]`，`d(r)` = 残基 `r` 的重原子到**另一个 entity** 全部重原子的最小距离 | ✅ **可见** —— 由**结构 + 元数据 entity 划分**算出，**不含任何 DMS label** |
| `π` | 该 assay 的界面残基比例（0.038–0.339，均值 0.126） | ✅ 可见（同上） |
| `θ` | encoder 参数 | 被训练 |
| `ψ` | probe head 参数（线性 128→1） | 被训练 |
| `φ` | decoder 参数 | 🔒 **全程冻结** |
| `ρ` | per-assay Spearman(score, DMS_score) → 14 assay 未加权平均 | ❌ **只在评测用**，绝不进入训练 |

> 🔴 **`y_r` 对 DMS 而言是 label-free 的** —— 这是整条线成立的前提：
> 界面标签来自结构，把它当 test-time 监督信号不构成泄漏。

### 2.2 目标函数

```
L(θ,ψ) = L_probe + λ · L_anchor

L_probe  = −(1/L') Σ_r [ w_pos · y_r · log p_r + (1−y_r) · log(1−p_r) ]     p_r = σ( g_ψ(h_V,r) )
           w_pos = (1−π)/π                                  ← 类别不平衡校正，逐 assay 算
L_anchor = ‖ h_V(θ) − h_V^frozen ‖²_F / (L'·128)            ← node 项
         + ‖ h_E(θ) − h_E^frozen ‖²_F / (L'·K·128)          ← edge 项，K = 48
```

`L'` = 有效残基数（已剔除缺口填充位与 `mask==0`，见 §5.1）。
两项**各自按元素数归一**到「每元素平均平方偏移」这个同一尺度，所以先按 1:1 相加，
也使 λ 在 `L` 相差 20 倍的 assay 之间可比。是否需要额外的相对权重，pilot 里看。

> 🔴 **anchor 为什么必须覆盖 `h_E`**（2026-09-14 review 抓出的设计缺陷，原方案只写了 node 项）：
> `L_probe` 只读 `h_V`，所以训练信号**只直接作用于 node**；但 `h_E` 照样在变（§2 图下方），
> 而且**两条路径直达 decoder**。只锚 `h_V` 的话，优化可以在 `h_V` 几乎不动的情况下
> 把 `h_E` 改得面目全非，**而 anchor 完全察觉不到** —— 没人看管的那部分，元素数还是被看管部分的 **48 倍**。

**每个组件对应 hypothesis 的哪一环：**

| 组件 | 角色 | 不要它会怎样 |
|---|---|---|
| `L_probe` 二分类 | **直接把 `q` 往上推** —— 它就是 `AP_norm` 那个任务的训练版 | 没有驱动力 |
| **`w_pos=(1−π)/π`** | 让正负类总权重相等。界面是少数类（最低 3.8%），不加权则梯度被多数类主导，**AUC 好看而 AP_norm 不动** —— 而优化空间恰在 AP_norm | 训了等于没训 |
| `λ·L_anchor`（node 项） | 防塌缩。按 §5.2 的教训先问「完美优化会怎样」：`L_probe` 被完美优化会让 `h_V` 向 `y_r` 这一维对齐，极端情形塌缩成界面标签的函数，**decoder 拿到的信息反而变少**（同类塌缩实测过，落到 **0.2564**，比基线差 0.13） | 大概率塌缩 |
| **`λ·L_anchor`（edge 项）** | 看管 `h_E`。它**不在 probe 的视野里却直达 decoder**，且元素数是 `h_V` 的 **48 倍** | `h_E` 可以任意漂移而 loss 无感，增益/损伤都无法归因 |
| decoder 冻结 | 保证增益归因于 encoder 表征，而不是打分头重新拟合 | 无法归因 |

**`w_pos` 的实际量级**：π 从 0.038（`KRAS_PICK3CG-RBD`）到 0.339（`hYAP65`）
⇒ `w_pos` 从 **25.3** 到 **1.95**，跨 assay 差 **13 倍**。所以它必须逐 assay 算，不能取全局常数。

> **为什么先做二分类、不直接上 soft label**（用户 2026-09-14 定）：
> 当前瓶颈是 `AP_norm` 只有 0.484 ——「碰不碰」这件事本身就没分干净。
> soft label `w(r)=exp(−d/5Å)` 留到二分类的约束逻辑验证通过后再上（见 §6）。

---

## 3. 实验网格

| id | 目的 | 配置 | 范围 |
|---|---|---|---|
| **P0** | **no-op gate** | `steps=0`，走完整条 TTT 代码路径后重打分 | 3 个 pilot assay |
| **P1** | 定 lr × steps 的量级 | λ=1 固定，lr ∈ {3e-5, 1e-4, 3e-4} × steps ∈ {50, 150, 400} | 3 个 pilot assay |
| **P2** | 定 λ | P1 选出的 (lr, steps)，λ ∈ {0.1, 1, 10} | 3 个 pilot assay |
| **M1** | **主实验** | P1/P2 选出的**单一共享超参** | **14 assay 全量** |
| **M2** | **permutation null** | 与 M1 完全相同，但 `y_r` 在有效残基间随机置换（保持 π 不变），3 seed | **14 assay 全量** |

**pilot 选这 3 个 assay 的理由**（覆盖三个维度且最便宜）：

| assay | n variants | `L` | π | `ρ_zeroshot` | 填充位 |
|---|---:|---:|---:|---:|---:|
| `5A12_Ang2_fitness_4ZFG` | 943 | 652 | 0.056 | 0.107 | 4 |
| `PSD95_CRIPT_1BE9` | 1,576 | 120 | 0.167 | 0.367 | 0 |
| `CD19_FMC63_Fitness_7URV` | 3,885 | 497 | 0.092 | 0.603 | **52 (10.5%)** |

合计 6,404 个 variant。`CD19` 特意选进来 —— 它填充位占 10.5%，**顺便验证 §5.1 的过滤是否真的生效**。

**实验之间的递进关系**（说得出「因为上一个得到 X，所以下一个才问 Y」）：
P0 确认代码路径不改变分数 → P1 才有意义；P1 定了 lr/steps 的量级 → P2 才知道在哪个点上调 λ；
P1/P2 给出单一共享超参 → M1 才不是逐 assay 调参；M1 拿到增益 → M2 才需要回答「增益是否来自界面先验」。

---

## 4. 判据

1. **P0 必须过**：`steps=0` 时 14×M=5 的分数与 baseline **逐行相同**（相对误差 ≤1e-6）。不过就停。
2. **M1 的成败**：`ρ_M1 > 0.3903`（14-assay）。
   噪声底：per-assay seed σ 中位 **0.0184**，但**配对比较**下 gain 的跨 seed sd 只有 **0.0037**
   ⇒ M1 与 baseline 必须用**同一组解码顺序**（同一 `randn`），否则读不出 <0.02 的效应。
3. **M2 必须显著低于 M1**：若置换 null 也涨到同一量级，增益**不来自界面先验**，H 不成立。
   报告方式写成**定量占比**（null 增益 / 真实增益），不写成是非题。
4. **`q` 只作 sanity check**：TTT 后在新 `h_V` 上重训一个独立探针，确认 `AP_norm` 确实上升。
   ⚠️ **它不是 H 的证据** —— 训练目标就是它，上升是构造性的（用户 2026-09-14 明确指出）。

---

## 5. 已知会静默出错的地方（动手前必须处理）

### 5.1 索引与过滤
- 🔴 **`parse_PDB` 把晶体学缺口补成坐标 (0,0,0) 的假残基**，`mask` 置 0 但**在 `h_V` 里占着行**。
  实测最高 **17.3%**（`KRAS_PICK3CG-RBD`）。`L_probe` 与 anchor **都必须按 `mask` ＋「是否填充位」过滤**，
  否则会去拟合一批不存在的残基。
- `h_V` 的行序 = `tied_featurize` 的打包顺序（**designed chains 在前**，每条链按 PDB 残基顺序）。
  标签直接在 PDB 空间算，**不需要 WT 序列对齐**。

### 5.2 实验设计
- **先问「这个 loss 被完美优化会怎样」** —— 已在 §2.2 回答，结论是必须有 anchor。
- **任何「取最大」的量都要配 null** ⇒ M2 存在的理由。
- **`h_E` 也必须重建**（实现陷阱，与 §2.2 的 anchor 缺陷同源）：
  `AssayContext` 把 encoder 输出缓存为 `h_V` **和 `h_E`**，并由二者构造 `h_EXV_encoder` / `h_EXV_fw`；
  `score()` 里也直接用 `self.h_E` 拼 `h_ES`。现成的 `set_h_V()` **只替换 `h_V`** —— 它是为「编辑表征」
  的干预写的。**重训 encoder 后 `h_E` 同样改变**（`EncLayer` 每层 `return h_V, h_E`），
  只调 `set_h_V()` 会让 `h_E` 的**两条 decoder 路径全都沿用旧值**，且无任何报错。
  ⇒ TTT 后必须**用新权重重跑 `_build_encoder_cache`**（等价于用同一个 `randn` 重建 `AssayContext`）。
  > 同一件事的两个面：**评测侧**忘了重建 `h_E` ⇒ 读到的是旧表征；**训练侧**忘了锚 `h_E` ⇒ 它无人看管地漂移。
- **两臂必须共用解码顺序**：`AssayContext.__init__` 接受 `randn=`，显式传入同一个 `randn`。
  这同时绕开了「TTT 的 `model.train()` 触发 dropout、推走 CUDA philox 流导致两臂不配对」那个坑。
- **口径**：本文档所有数字一律 **14 assay，基线 0.3903**。不与 23/25 口径的数字并列。

---

## 6. 还不能说的 / 下一步（按「能改变结论的程度」排，不按代价排）

**现在还不能说的**：
- H 尚未被任何干预性证据检验过。§3 那条 `AP_norm,within` vs `ρ_zeroshot` 的相关
  （Spearman **+0.515, p=0.060**, n=14）**未达显著**，且 proposal 自列 3 个混淆项。
- 本任务**不单独验证 H**（用户 2026-09-14 决定）：直接看 TTT 后的 `ρ`。
  ⇒ 若 M1 成功，得到的是「这套做法有效」，**不是「H 被证实」**；若失败，也无法区分是 H 假还是实现/选参不好。

**下一步**：
1. **soft label 版本**（`w(r)=exp(−d/5Å)`，ridge/回归读出）—— 二分类通过后做，直接替换 `L_probe`。
2. λ 的 label-free 选法 —— 这是两条 TTT 线共同的瓶颈；本任务因门槛低而暂时回避（固定 λ），
   但要落地必须解决。已知：训练目标自身单调无峰（已排除）；下一个候选是 ESM2 ensemble agreement。
3. 扩到 23 assay —— 二值标签在另外 9 个上无法评（全碰或全不碰），需换成 soft label 才可行。

---

## 附录 A：超参的选择与理由

由我自主决定（用户 2026-09-14 授权），理由如下：

| 项 | 取值 | 理由 |
|---|---|---|
| **可训练参数** | **`encoder_layers` + `W_e`** = 907,776（**54.7%**），`features` 冻结 | 实测 total 1,660,485：`features` 54,576 / `W_e` 16,512 / `encoder_layers` 891,264 / decoder 侧 695,445（41.9%，与 decoder-TTT 记录逐位吻合 ⇒ 交叉验证通过）。**`features` 是坐标→RBF 的几何入口**，改它等于改结构信息的表示基底，风险高而收益不明；界面编码质量主要由做消息传递的 `encoder_layers` 决定。`--train_features` 作为 pilot 开关，默认关 |
| optimizer | **AdamW**，`weight_decay=0` | 短训；正则化由 `L_anchor` 显式承担，不要两套正则互相干扰 |
| lr | pilot 扫 **{3e-5, 1e-4, 3e-4}** | decoder-TTT 那侧只验证过 1e-4（3e-5/3e-4 未扫）。encoder 更靠近输入、扰动向下游传播更远，**先验上应更保守**，所以把 3e-5 纳进来 |
| steps | pilot 扫 **{50, 150, 400}** | 训练极便宜（单个 complex 的 encoder forward+backward），但**过训会塌缩**，所以扫的是「什么时候开始坏」而不是「多久收敛」 |
| λ | pilot 扫 **{0.1, 1, 10}** | 量级未知，先扫一个数量级跨度 |
| batch | 全部残基一次（1 complex） | `L ≤ 1107`，无需分批 |
| 精度 | **fp32** | 模型只有 ~1.7 M 参数，fp32 无压力；避免混精度在小模型上的数值噪声混进 <0.02 的效应里 |
| head 初始化 | **两阶段**：先在冻结 `h_V` 上把 `g_ψ` 训到收敛，再联合训 `θ,ψ` | 随机初始化的 head 会在最初若干步把乱梯度灌进 encoder |
| seed | 固定；`randn` 两臂共用 | 见 §5.2 |

**🔴 诚实性声明**：P1/P2 是**看着 3 个 assay 的 `ρ` 选超参**，而 BindingGYM **没有 dev split 也造不出来**
⇒ 这是 test-set tuning 的一种。缓解措施：① 主实验用**单一共享超参**，不逐 assay 调；
② pilot 的 3 个 assay 与 M1 的 14 个**重叠**，这一点必须在任何对外的数字旁标注。

## 附录 B：成本估算

- **TTT 训练**：单个 complex 的 encoder forward+backward，`L≤1107` ⇒ 每 assay 秒级。可忽略。
- **评测（主成本）**：decoder-only teacher-forced 打分，M=5。
  14 assay 全量 = **227,639** variant。官方口径 25 assay（376,424）完整 forward 是 4 GPU-h，
  本实现 encoder 已缓存 ⇒ 估 **≤2.5 GPU-h / 臂**，pilot 时实测校正。
- P0+P1+P2 ≈ 13 个配置 × 6,404 variant ≈ 83 K ⇒ 约 **1 GPU-h**
- M1 + M2(3 seed) = 4 × 227,639 ⇒ 约 **10 GPU-h**
- **合计 ≈ 11 GPU-h**（pilot 后校正；若单臂超过 3 GPU-h 则先缩 M2 的 seed 数）
