# tracking — `structure-encoder-TTT` / `encoder_ttt_interface_binary`

给自己查的操作日志。**不要求前后逻辑顺畅，要求细节查得到。**
结论与框架看 `encoder_ttt_interface_binary-organized.md`。

**开始** 2026-09-14 ｜ **最后更新** 2026-09-14 ｜ **当前阶段**：计划定稿，代码未写，未开跑。

---

## 1. 运行日志

| 日期 | 任务 | PID | wall | 配置 | 产出 | 结果 |
|---|---|---|---|---|---|---|
| — | 尚未开跑 | | | | | |

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
3. **两臂共用解码顺序**：`AssayContext.__init__(..., randn=)` 显式传入。
   顺带绕开「`model.train()` 触发 dropout → 推走 CUDA philox 流 → 两臂 randn 不同」这个坑。
4. **评测时 `model.eval()` + `augment_eps=0`**。TTT 用 `torch.enable_grad()` 包住后
   **必须复位这两项**，否则每个 variant 每次 forward 重采坐标噪声，无任何报错。
5. **每臂独立输出路径**。官方 runner 见到已存在的 csv 会跳过；写进同一目录会静默复用上一臂的结果。

### 3.2 数值与工程
- `python -u`，结果**逐 assay 落盘**，不要跑完才写。
- 图上文字一律英文（本机 matplotlib 只有 DejaVu Sans，中文静默变豆腐块）。
- `pandas` 里别用 `median` 之类会被方法名遮蔽的列名。
- 回流用 `rsync --exclude '*.md' --update`（`-a` 按差异传不按新旧传，会把本地写的 record 覆盖回旧版；
  2026-09-13 实测事故：574 行被覆盖成 345 行）。

---

## 4. 已知缺陷（读结果时必须带着）

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
> 按 workstation-usage §4-3「数据就地化」，代码要**复制**进本 worktree，不要跨 worktree 引用。

### 5.2 数据（workstation）
| 东西 | 路径 |
|---|---|
| BindingGYM input（DMS csv + 结构） | `/data/guoj0f/share/BindingGYM/input` |
| ckpt `v_48_020.pt`（md5 `91d54c97a68bf551114f8c74c785e90f`） | `/data/guoj0f/share/BindingGYM/training/cache/v_48_020.pt` |
| 官方 zero-shot 逐 variant 分数（seed1, M=5） | `/data/guoj0f/BindingGYM-zero-shot-proteinMPNN/scores/seed1_M5/` |
| conda env（numpy 1.24.4 / scipy 1.10.1，**别升级**） | `bindinggym-zs-mpnn` |

### 5.3 参照数值
| 东西 | 路径 |
|---|---|
| 探针逐 assay 结果（`AP_norm` 等） | `.../workstation-records/mutation-landscape-TTT/data/ga2_encoder_probe.csv` |
| 各 assay 填充位计数 | `.../workstation-records/mutation-landscape-TTT/data/ga2_coverage.csv` |
| 14-assay 集合推导 + 逐 assay zero-shot ρ | `.../workstation-records/mutation-landscape-TTT/data/g1e_canonical14.csv` |

---

## 6. 环境实况（2026-09-14 预检）

- A100 80GB PCIe，**free 65.7 GB**，util **0%**；另有 3 个他人进程占 15.3 GB（不动）。
- ⚠️ **`/home` 已 99%，只剩 96 GB**（workstation-usage skill 里记的 157 GB 已过期）。
  本任务产出是 csv/md，留 worktree 内即可；大件一律走 `/data`。
- `/data` 余 6.6 T。

### 6.1 🔴 SSH 间歇性不可用（2026-09-14 实测，未解决）

同一天内 ssh 出现三种表现：① 首次预检**成功**；② 随后 `Permission denied (publickey,password)`；
③ 再后来直接超时。`ssh -v` 定位到真实原因：

```
debug1: identity file /home/guoj0f/.ssh/id_ed25519 type 3   <- key 正常读到
debug1: Connection established.                             <- TCP 已建立
Connection timed out during banner exchange                 <- sshd 未在超时内回 banner
```

`/dev/tcp/10.67.24.41/22` 可达 ⇒ **不是网络不通、不是 key 问题、不是本地沙箱**，
而是**远端 sshd 响应不过来**（56 核共享机，GPU util 0% 但 CPU/IO 可能被他人占满）。
⚠️ **影响**：这会让 `nohup` 启动、监控轮询、结果回流都间歇性失败。
开跑前必须重测；长任务务必用 `nohup`/`tmux` 脱离 ssh 会话，**不要让任务依赖连接存活**。

- 远端 env 列表：`bgym-official` `bindinggym-zs-mpnn` `complex-mutant-structure-pred`
  `esmfold2` `h3ddg-reproduce` `pgym-binding-partner-mpnn` `proteingym-ttt`。

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
#   workstation-records/structure-encoder-TTT/sh/{task}_{dt}.sh  （批量 launcher）
```
