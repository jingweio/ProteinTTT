# decoder_ttt_interface_divergence — tracking log

**project `mutation-landscape-TTT` · task `decoder_ttt_interface_divergence` · 开始 2026-09-12**

> **这篇是什么**：任务推进的**操作日志** —— 跑了什么、怎么跑的、出了什么错、口径怎么改的。
> 不追求前后逻辑顺畅，**只追求细节查得到**。
> **结论看** [`decoder_ttt_interface_divergence-organized.md`](decoder_ttt_interface_divergence-organized.md)。
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
| 09-13 09:47 | **E-3** 重投 `e3_null_and_lamext_*.sh` | (a) ACE2 λ 延伸到 0.001；(b) 真实臂 3 assay；(c) null 臂 3 assay。全部 `--sweep_M 1` | 742602 | 进行中 | ACE2 / KRAS / PSD95 真实臂已出 |

**日志与产物**：`workstation-records/mutation-landscape-TTT/{*.out,*.err,data/}`；
启动脚本 `sh/`；每份 `.out` 首行是 `[synced_commit]`，可溯源到 commit。

### 1.1 训练超参（全部运行一致，除非另注）

| 项 | 值 | 来源 |
|---|---|---|
| optimizer / lr | Adam / **1e-4** | 方案定的起点；3e-5 / 3e-4 **未扫** |
| steps | **300 固定** | ⚠️ 不是「训到收敛」，见 §3 已知缺陷 1 |
| early stopping | **无** | 方案要求：λ 是唯一旋钮 |
| batch | `clip(3e8/(L·48·256), 8, 128)`，再按距离分 **5 档等量抽** | 显存约束 |
| 可训练参数 | decoder 3 层 `DecLayer` + `W_out` = **695,445 / 1,660,485 (41.9%)** | 实测 |
| M（解码顺序） | 训练每步抽 1 个；评测 5 个（官方口径） | `--sweep_M` 控制扫描阶段 |
| seed | 1（与官方 zero-shot 同） | — |

**实际 batch 与等效 epoch**（同样是 300 步，曝光量差 **80 倍**）：

| assay | L | batch | 每档 | 等效 epoch |
|---|---:|---:|---:|---:|
| `PSD95_*` | 120 | 125 | 25 | **23.8** |
| `KRAS_RAF1-RBD` | 245 | 95 | 19 | 1.2 |
| `GB1_IgG-Fc` | 262 | 90 | 18 | **0.29** |
| `CD19_FMC63` | 497 | 45 | 9 | 3.5 |
| `HLA-A2` / `5A12` | 644/652 | 35 | 7 | ~1.5 |
| `ACE2_SARS2-RBD` | 931 | 25 | **5** | 3.4 |
| *（未跑）* `KRAS_PICK3CG` | 1107 | 20 | **4** | — |

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

1. **`steps=300` 是固定预算，不是收敛。** 方案 §3.4(viii) 写的是「训到收敛，不做 early stopping」，
   理由是「λ 是唯一旋钮，否则 λ 与步数互相混淆」。**实现成了固定步数 ⇒ 两者现在是混淆的。**
   证据：`l_div` 逐步均值在 λ=0.03 下第 299 步仍是最负（−0.595）且还在下降。
   ⇒ **目前的 `λ*` 应读作「300 步预算下的最优锚强度」，不是最优锚强度本身。**
2. **等效 epoch 跨 assay 差 80 倍**（见 §1.1 表）⇒ 训练量不可比。
3. **大 `L` 的 assay batch 太小**：`ACE2` 每档 5 个、`KRAS_PICK3CG` 每档 4 个，
   而 `L_div` 是 batch 内 Pearson ⇒ 估计噪声大。方案原本写的是 batch=128、每档 ~26。
   ⚠️ `ACE2` 恰好是表现最差的那个，**这两件事可能不是巧合**。
4. **`L_anchor / var(s_frozen)` 没达到设计目的**：该归一化本意是让 λ 跨 assay 可比，
   但实测 `λ*` 与 `|c*|` 的 Spearman 只有 **−0.296**（n=8，不显著）。
5. **`sweep_M=1` 的 ρ 与 `sweep_M=5` 的不可直接比**（同一 λ=0.03，ACE2 分别是 0.3649 / 0.3237）。
   只有最后的 M=5 复评可比。

---

## 4. 口径变更史

| 日期 | 变更 | 原因 |
|---|---|---|
| 09-12 | `λ` 从 LOAO 选改为**逐 assay 扫**（oracle） | 用户决定：摸底阶段不做跨 assay 选参 |
| 09-12 | 门槛从 LOAO 的 0.4551 → **oracle `c*`** | λ 是 oracle 选的，对照的 `c` 也必须是 oracle |
| 09-13 | 🔴 **更正**：门槛不是单一数字 | 我把 0.4850 写成「本阶段门槛」是错的 —— 那是 `sum+exp5` 的数，而 E-1 用的是 `max+exp5`（0.4689）。**TTT 用哪个 `w`，就比哪个 `w` 的 oracle 行。** |
| 09-13 | 记录该族 oracle 上确界 = **0.5074**（`sum + 1/(1+d/20)`） | 既然 `c` 和 `λ` 都 oracle，特征也应取 oracle 最优 |

---

## 5. 脚本清单

| 脚本 | 作用 | 设备 |
|---|---|---|
| `bgmpnn.py` | 可微打分 + encoder 缓存；`load_model` 断言 ckpt md5 `91d54c97…` | GPU |
| `gb_noop_control.py` | **G-B**：与官方 `compute_fitness_multi_pdb.py` 逐行比对 | GPU |
| `ga_encoder_probe.py` | **G-A**：冻结 `h_V` 的线性探针 | GPU |
| `e1_lambda_sweep.py` | **E-1 / E-3**：λ 扫描；`--permute_d` 出 null 臂；`--sweep_M` 省评测 | GPU |
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
