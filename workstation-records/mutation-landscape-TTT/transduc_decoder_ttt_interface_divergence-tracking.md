# transduc_decoder_ttt_interface_divergence — tracking log

**project `mutation-landscape-TTT` · task `transduc_decoder_ttt_interface_divergence` · 开始 2026-09-12**

> **这篇是什么**：任务推进的**操作日志** —— 跑了什么、怎么跑的、出了什么错、口径怎么改的。
> 不追求前后逻辑顺畅，**只追求细节查得到**。
> **结论看** [`transduc_decoder_ttt_interface_divergence-organized.md`](transduc_decoder_ttt_interface_divergence-organized.md)。
> 姊妹记录（上一阶段的摸底）：[`probe_binding_site_insight_value_20260909-160514.md`](probe_binding_site_insight_value_20260909-160514.md)

---

## 1. 运行日志

机器：workstation `10.67.24.41`，A100 80GB ×1，env `bindinggym-zs-mpnn`。
远端工作目录 `/home/guoj0f/repos/ProteinTTT/bindingGYM-binding-sites-analysis`。

| 日期 | 任务 | 配置 | PID | wall | 结果 |
|---|---|---|---|---|---|
| 09-12 22:18 | **实验 ①** 特征调优 `t3_feature_tuning.py` | 63 配置 × 161 `c`，14 assay，56 核 CPU | 3124841 | **16 min** | 最优 `sum+exp_tau5`，LOAO 0.4599 / oracle 0.4850 |
| 09-12 22:2x | **G-B** no-op 对照 `gb_noop_control.py` | `5A12_Ang2_4ZFG`, `PSD95_CRIPT_1BE9`，M=5 | — | ~3 min | **PASS**，最大相对偏差 3.0e-7 |
| 09-12 22:3x | **G-A** encoder 探针 `ga_encoder_probe.py` | 14 assay，线性探针 on 冻结 `h_V` | — | ~5 min | AUC 0.890 (within) / 0.915 (LOAO) |
| 09-12 22:34 | **E-1** λ 扫描 `e1_lambda_sweep.py` | 4 assay × 7 λ × 300 步，`--tag e1`，sweep M=5 | 3177492 | **154 min** | 见 organized §4 |
| 09-13 00:04 | **e1b** 小 assay 补充 | 5 assay × 7 λ，`max` 与 `sum` 两轮 | 3421080 | ~160 min | 补齐到 8 assay |
| 09-13 01:10 | **E-3** 第一次投 | — | 3597880 | **秒退** | ❌ 见 §2 bug 3/4，**白等 8.5 h** |
| 09-13 09:47 | **E-3** 重投 `e3_null_and_lamext_*.sh` | (a) ACE2 λ 延伸到 0.001；(b) 真实臂 3 assay；(c) null 臂 3 assay。全部 `--sweep_M 1`，λ∈[0.001,3] | 742602 | **~170 min** | ✅ `E3_DONE rc=0`。ACE2 −0.1192→−0.0648；null 只拿到真实增益的 **19%**、**0/3 越线**。见 organized §4.5 |

| 09-14 01:5x | **M-1** batch 显存实测 `m1_batch_memory.py` | 4 assay（L=120/497/652/931）× bs 26→768，真实训练步 | 804874 | ~6 min | ✅ 峰值 ≈ `7.2e-4 GiB × bs × L`；公式保守 **4.2×**。见 §3 缺陷 3 |

| 09-14 02:4x | **M-1b** 全 14 assay 显存标定 | bs 32→512，`--n_cap 700`（显存只依赖 `bs×L`，与 n 无关） | 967789 | ~9 min | ✅ 见 §1.2 |

| 09-14 03:09 | **E-4** 步数 × `sum` `e4_steps_and_sum_*.sh` | 3 assay × 8 λ × 3000 步，`--eval_at 300 1000 2000 3000`，`--agg sum` | 1043959 | **389 min**（GB1 11,445s / KRAS 8,319s / PSD95 3,556s） | ✅ `E4_DONE rc=0`。见 organized §4.6 |

**2026-09-14 09:40 起：workstation 上没有本任务的进程在跑，GPU util 0%。**

**日志与产物**：`workstation-records/mutation-landscape-TTT/{*.out,*.err,data/}`；
启动脚本 `sh/`；每份 `.out` 首行是 `[synced_commit]`，可溯源到 commit。

### 1.1 训练超参（全部运行一致，除非另注）

| 项 | 值 | 来源 |
|---|---|---|
| optimizer / lr | Adam / **1e-4** | 方案定的起点；3e-5 / 3e-4 **未扫** |
| steps | **300 固定** | ⚠️ 不是「训到收敛」，见 §3 已知缺陷 1 |
| early stopping | **无** | 方案要求：λ 是唯一旋钮 |
| batch | `clip(3e8/(L·48·256), 8, 128)`，再按 `w` 的**分位**切 5 档、每档有放回抽 `bs//档数` | 显存约束（常数未标定，M-1 实测保守 4.2×，见 §3 缺陷 3） |
| 可训练参数 | decoder 3 层 `DecLayer` + `W_out` = **695,445 / 1,660,485 (41.9%)** | 实测 |
| M（解码顺序） | 训练每步抽 1 个（整批共用）；评测 5 个（官方口径） | 09-14 起 `--sweep_eval_M` 只缩**读数**；旧的 `--sweep_M` 会连训练一起缩，已废弃 |
| seed | 1（与官方 zero-shot 同） | — |

**实际 batch 与等效 epoch**（同样是 300 步，曝光量差 **82 倍**）。
🔴 **09-14 更正**：原表把「每档 = `bs//5`」当成恒真，但 `w` 有并列时分位档会**塌**，
`CD19` 只有 4 档、`5A12` 只有 **2** 档 —— 这两行的每档数与等效 epoch 都算错了：

| assay | n | L | `bs_eff` | 非空档 | 每档 | 实际 batch | 等效 epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PSD95_CRIPT` / `PSD95_Tm2F` | 1576 | 120 | 128 | 5 | 25 | 125 | **23.79** |
| `KRAS_RAF1-RBD` | 23161 | 245 | 99 | 5 | 19 | 95 | 1.23 |
| `GB1_IgG-Fc` | 92890 | 262 | 93 | 5 | 18 | 90 | **0.29** |
| `CD19_FMC63` | 3885 | 497 | 49 | **4** | **12** | 48 | 3.71 |
| `HLA-A2_TAPBPR` | 3344 | 644 | 37 | 5 | 7 | 35 | 3.14 |
| `5A12_Ang2` | 943 | 652 | 37 | **2** | **18** | 36 | **11.45** |
| `ACE2_SARS2-RBD` | 2185 | 931 | 26 | 5 | **5** | 25 | 3.43 |

`5A12` 的档分布是 `[124, 0, 0, 0, 819]` —— 124 个变体每步被抽 18 个，300 步里每个平均出现
~43 次；`CD19` 是 `[776, 412, 0, 1920, 777]`，412 那档相对 1920 那档被**过采样 4.7×**。
**其余 6 个未跑的 assay 分位档都是满 5 档且均衡**（已核），这个问题只影响这两个。

### 1.2 显存标定（M-1 / M-1b，09-14 实测，全 14 assay）

真实训练步（forward + backward + Adam，只训 decoder 3 层 + `W_out`，fp32）。
**`torch.cuda.max_memory_reserved` ≈ `8.82e-4 GiB × bs × L`**（46 个测点中位，离散 8.21~9.01e-4；
偏低的那几个是 allocator 贴着卡顶被挤压的情形）。`max_memory_allocated` ≈ `7.1e-4 × bs × L`。
⇒ **可用 70 GiB 时 `bs_max ≈ 7.9e4 / L`**。

| assay | L | bs=128 | bs=256 | bs=512 | 上限@70GiB | 建议（留 10%） |
|---|---:|---|---|---|---:|---:|
| `hYAP65_peptide_1JMQ` | 56 | ✅ 6 | ✅ 13 | ✅ 25 | 1417 | 1275 |
| `PSD95_CRIPT` / `PSD95_Tm2F` | 120 | ✅ 14 | ✅ 27 | ✅ 54 | 661 | 595 |
| `KRAS_RAF1-RBD` / `KRAS_RAF1` | 245 | ✅ 28 | ✅ 55 | ❌ 111 | 323 | 291 |
| `KRAS_RALGDS-RBD` | 254 | ✅ 29 | ✅ 57 | ❌ 115 | 312 | 281 |
| `GB1_IgG-Fc` | 262 | ✅ 30 | ✅ 59 | ❌ 118 | 302 | 272 |
| `CXCR4_CXCL12` | 360 | ✅ 41 | ⚠️ 81（实测挤到 68 过了） | ❌ 163 | 220 | 198 |
| `CD19_FMC63` | 497 | ✅ 56 | ❌ 112 | ❌ 224 | 159 | 143 |
| `HLA-A2_TAPBPR` | 644 | ⚠️ 73（实测挤到 69 过了） | ❌ 145 | ❌ 291 | 123 | 110 |
| `5A12_Ang2` | 652 | ⚠️ 74（实测挤到 66 过了） | ❌ 147 | ❌ 294 | 121 | 109 |
| `SARS2-RBD_ACE2_6M0J` | 791 | ❌ 89（实测 OOM） | ❌ 179 | ❌ 357 | 100 | 90 |
| `ACE2_SARS2-RBD_6M17` | 931 | ❌ 105（实测 OOM） | ❌ 210 | ❌ 420 | 85 | 76 |
| `KRAS_PICK3CG-RBD` | 1107 | ❌ 125（实测 OOM） | ❌ 250 | ❌ 500 | 71 | **64** |

数字 = 预测的 peak reserved (GiB)。✅ ≤ 63 GiB（留 10%）／⚠️ 63~70／❌ > 70。
**⇒ 想让 14 个 assay 用同一个 batch，上限是 `bs = 64`**（受 `KRAS_PICK3CG`, L=1107 卡住）。
要统一到 128 必须把激活砍一半：bf16 autocast（~2×）或对 3 层 decoder 做 gradient checkpointing
（~3×，换约 30% 计算）。**两者都会改数值/时间口径，未做。**

小 batch 还更慢（每样本耗时，L=120）：bs=32 **3.5 ms** → 64~384 稳定 **~1.0 ms** → 512 起回升。

---

## 2. 踩过的坑与修法

1. **env 缺 `pyarrow`**（09-12）→ `pip install pyarrow`，装后断言 `numpy==1.24.4 / scipy==1.10.1` 未漂移。
2. **`tied_featurize()` 位置参数数错**（多传了一个 `None`）→ 签名是 8 个位置参 + `ca_only`。
3. **`--sweep_M 1` 时 baseline 循环仍按 `a.M=5`** → `IndexError`。改成 `range(sf_v.shape[1])`。
4. **`ctx.M = 5` 只改字段没重建 context** → `mask_bw` / `randn` 仍是 M=1 的，切片为空。
   **M 在构造时就烙进了所有张量，必须 `build()` 重建。**
   > 3 和 4 是同一次事故：加了 `--sweep_M` 却**没 smoke test 那条确切路径**就投了，三个子任务全部秒退。
   > **教训：新增参数后必须先用最小配置跑通那条路径。**
5. **`rsync -a` 回流把 record md 覆盖成旧版**（09-13）→ 574 行 → 345 行，从 git 恢复。
   `rsync -a` **按差异传、不按新旧传**，而 record 在本地写、远端那份是旧的。
   修法：`sh/pull_results.sh`（`--exclude '*.md'` + `--update` 双保险，原因写在脚本里）。
6. **`DataFrame.agg` 属性遮蔽**（列名叫 `agg`）→ 同 `.median` 那一类。列改名 `feat`。
7. **界面表存的是 3 位小数** → `t2_site_dists.py` 的 min 一致性断言容差放到 1e-3，实测最大偏差 5.0e-4 Å。
8. **stdout 缓冲**：E-1 跑了 154 min 一行进度都看不到 → 之后全部 `python -u` + 逐 λ 落盘。

---

## 3. 已知缺陷（影响解读，尚未修）

0. 🔴 **09-14 E-4 已回答「步数够不够」：够。** λ 在每个预算下重选之后，
   3000 步比 300 步平均 **−0.0112**（GB1 +0.0098 / KRAS −0.0440 / PSD95 +0.0006）。
   而且**覆盖率不是机制** —— `PSD95_CRIPT` 300 步就 100% 覆盖，却呈现与另外两个相同的
   λ×步数结构。**下面第 1 条的「混淆」仍然成立，但方向已知：λ 低于阈值时多跑步数变差、
   高于阈值时变好，3/3 assay 结构相同 ⇒ 两者是同一个旋钮。**
1. **`steps=300` 是固定预算，不是收敛。** 方案 §3.4(viii) 写的是「训到收敛，不做 early stopping」，
   理由是「λ 是唯一旋钮，否则 λ 与步数互相混淆」。**实现成了固定步数 ⇒ 两者现在是混淆的。**
   证据：`l_div` 逐步均值在 λ=0.03 下第 299 步仍是最负（−0.595）且还在下降。
   ⇒ **目前的 `λ*` 应读作「300 步预算下的最优锚强度」，不是最优锚强度本身。**
2. **等效 epoch 跨 assay 差 80 倍**（见 §1.1 表）⇒ 训练量不可比。
3. **大 `L` 的 assay batch 太小，而且是白给的**：`ACE2` 每档 5 个，而 `L_div` 是 batch 内
   Pearson ⇒ 估计噪声大。⚠️ `ACE2` 恰好是表现最差的那个，**这两件事可能不是巧合**。
   🔴 **09-14 M-1 实测**：峰值显存 ≈ `7.2e-4 GiB × bs × L`（4 个 L 从 120 到 931，系数一致到
   2%），保留量 ≈ `8.4e-4 × bs × L`。即这张卡的真实上限约 `bs ≈ 1.0e5 / L`，而公式给的是
   `2.4e4 / L` —— **保守 4.2 倍**。`ACE2` 实测能到 ~96（现在 26），`CD19` ~200（现在 49）。
   小 batch 还**更慢**：`PSD95` 每样本耗时 bs=26 是 2.46 ms，bs=64~384 稳定在 ~1.0 ms。
   🔴 **原先写的「用梯度累积把每档提回 ~26」是错的**：累积得到的是「k 个小 batch 各自
   Pearson 的均值」，**不是合并后那一个 Pearson** —— `L_div` 的估计质量不会因累积而改善。
   唯一的办法是**真的把 batch 提上去**（现已证明有 4.2× 余量）。
   ⚠️ 改 batch 会改等效 epoch ⇒ 与已跑的 8 个 assay 不可比，属**口径决定**，未擅自改。
   全 14 assay 的逐个可行性见 §1.2。
4. **`L_anchor / var(s_frozen)` 没达到设计目的**：该归一化本意是让 λ 跨 assay 可比，
   但实测 `λ*` 与 `|c*|` 的 Spearman 只有 **−0.296**（n=8，不显著）。
5. **~~`sweep_M`~~ 已于 09-14 废弃，但已跑的 8 个 assay 仍带着它的痕迹。**
   `sweep_M=1` 的 ρ 与 `sweep_M=5` 的不可直接比（同一 λ=0.03，ACE2 分别是 0.3649 / 0.3237），
   只有最后的 M=5 复评可比 ⇒ organized §4.3 的表限定「每个 assay 只用一次运行」。
   更要紧的是：`sweep_M` **同时**缩了训练可抽的顺序数，所以 `KRAS` / `GB1` / `PSD95_CRIPT` /
   `ACE2` 的 `λ*` 是在 **1-order 训练**下选出、却交给 **5-order 训练**去用的。新的
   `--sweep_eval_M` 只缩读数，这个混淆对**今后**的运行不再存在。
6. ~~**训练 M 从未作为超参扫过**~~ → 09-14 **定为 5**（`--sweep_eval_M` 不再改训练）。那 `GB1` 的 M=5 训练 0.5645 vs M=1 训练 0.5577 正是旧口径的代价。
7. **只有 3 个 assay 有 E-3 null 对照**（KRAS_RAF1-RBD / PSD95_CRIPT / GB1）。

---

## 4. 口径变更史

| 日期 | 变更 | 原因 |
|---|---|---|
| 09-12 | `λ` 从 LOAO 选改为**逐 assay 扫**（oracle） | 用户决定：摸底阶段不做跨 assay 选参 |
| 09-12 | 门槛从 LOAO 的 0.4551 → **oracle `c*`** | λ 是 oracle 选的，对照的 `c` 也必须是 oracle |
| 09-13 | 🔴 **更正**：门槛不是单一数字 | 我把 0.4850 写成「本阶段门槛」是错的 —— 那是 `sum+exp5` 的数，而 E-1 用的是 `max+exp5`（0.4689）。**TTT 用哪个 `w`，就比哪个 `w` 的 oracle 行。** |
| 09-13 | 记录该族 oracle 上确界 = **0.5074**（`sum + 1/(1+d/20)`） | 既然 `c` 和 `λ` 都 oracle，特征也应取 oracle 最优 |
| 09-13 | 🔴 **更正**：8-assay 均值 −0.0044 → **+0.0019** | 前者**混了两次不同 `sweep_M` 的运行**。改成逐 assay 只用一次运行，并纳入 ACE2 的延伸网格 |
| **09-14** | 🔴 **`--sweep_M` → `--sweep_eval_M`**：context 一律按 `--M` 建，训练永远从 5 个顺序里抽，只有**排序 λ 用的 ρ** 读 1 个顺序；`λ*` 定后同 seed 重训、按 M=5 出数 | 用户决定（选 λ 用 1、推理用 5，与 zero-shot 对齐）。旧参数**同时**缩训练与读数 ⇒ `λ*` 选在一个 regime、用在另一个。新写法还省掉一次 context 重建和一遍 frozen-score |
| **09-14** | 🔴 **分层抽样改为 batch 内无重复**（`torch.randint` → `torch.randperm[:per]`） | 用户决定。`L_div` 是 batch 内 Pearson，重复样本会在相关与锚的均值里各算两次。原先不是样本不够（n 943~92,890），只是调用更省事；代价实测为 `5A12` 每档每步 ~1.2 个重复、`PSD95` ~1.0、其余 <0.2。**档小于 `per` 时改为 assert**，不再静默缩 batch |
| **09-14** | **特征从 `max` 改为 `sum`**（用户决定：single/multi-mutation site 概念不同） | ⚠️ E-4 实测后**结论取决于口径**：比各自 `w` 的天花板是 `max` 赢（+0.0266 vs +0.0115），比「最好的便宜方法」是 `sum` 赢（+0.0028 vs +0.0091），绝对 ρ `sum` 高 +0.0063。**后一个口径才是部署问题，但它的门槛只取了 63 个配置里的 2 个，是下界** |
| 09-13 | 🔴 **判读表的写法要改**：二值 → 定量 | E-3 事先写的是「null 也涨 ⇒ 增益不来自界面」。**现实是定量的** —— null 涨了但只占 19%、从不越线。**以后判读表写阈值（如「净增益 > null 的 2 倍且 null 不越线」），不写是非题。** |

---

## 5. 脚本清单

| 脚本 | 作用 | 设备 |
|---|---|---|
| `bgmpnn.py` | 可微打分 + encoder 缓存；`load_model` 断言 ckpt md5 `91d54c97…` | GPU |
| `gb_noop_control.py` | **G-B**：与官方 `compute_fitness_multi_pdb.py` 逐行比对 | GPU |
| `ga_encoder_probe.py` | **G-A**：冻结 `h_V` 的线性探针 | GPU |
| `e1_lambda_sweep.py` | **E-1 / E-3**：λ 扫描；`--permute_d` 出 null 臂；`--sweep_eval_M` 省**读数**（不动训练） | GPU |
| `m1_batch_memory.py` | **M-1**：真实训练步下的峰值显存 vs batch，逐 assay 到 OOM | GPU |
| `d1_diagnostic.py` | **D-1**：`Δs` 对 `c·w` 回归 | CPU |
| `t1_loss_axis.py` | loss 的起点与终点 `corr(w,·)` | CPU |
| `t2_site_dists.py` | 逐突变位点距离（1,173,273 对），min 一致性有断言 | CPU |
| `t3_feature_tuning.py` | **实验 ①**：63 配置 × 161 `c` + 置换 null | CPU ×56 |
| `t4_per_assay_bars.py` | 逐 assay 的 zero-shot 与 oracle 天花板 | CPU |
| `t5_summary.py` | 汇总所有 TTT 运行 vs 天花板 | CPU |
| `sh/pull_results.sh` | 🔴 **唯一允许的回流方式** | — |

**依赖的数据资产**（全部在 repo 外或另一分支，见 memory `bindinggym-local-assets`）：
`/data/guoj0f/share/BindingGYM/`（官方代码 + input）、
`/data/guoj0f/BindingGYM-zero-shot-proteinMPNN/scores/seed1_M5/`（参考分数，329 MB）、
`v_48_020.pt`（ckpt）。

---

## 6. 逐 assay 原始数据

见 `data/`：`t5_summary.csv`（汇总）、`e1_lambda_sweep.csv`、`e1b_{max,sum}_lambda_sweep.csv`、
`e1c_lamext_lambda_sweep.csv`、`e3real_lambda_sweep{,_null}.csv`、`d1_diagnostic.csv`、
`t3_feature_tuning{,_null}.csv`、`t4_per_assay_bars.csv`、`ga_encoder_probe.csv`、`gb_noop.csv`。
逐 variant 预测在 `*_preds.npz`。
