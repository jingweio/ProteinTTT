# complex-mutant-structure-pred — structure pseudo-label 方法调研 record

（created 2026-08-29 23:14; status: **DONE**；无 GPU 消耗，纯文献 / 工具调研 + 对抗性事实核查）

## 调研问题（用户两问）

- **Q2. FoldX 也是一个结构预估工具？它是不是非常慢？**
- **Q3. 现在用于打 structure-prediction 伪标签的方法，除了 ESMFold2 / AF3 / AF2-Multimer / FoldX 以外还有什么？**

> 前置：本篇的直接上游是同目录的资产调研 [`predicted_complex_structure_assets_20260829-000600.md`](predicted_complex_structure_assets_20260829-000600.md)，
> 它的结论是「BindingGYM 的 mutant complex 结构，公开资产只覆盖 0.55%（Zenodo v1 的 2,080 个 FoldX 结构，仅 `4D5_HER2_fitness_1N8Z`）」。
> 那篇问的是 **有没有现成的**；本篇问的是 **要自己生成的话有哪些方法、代价多少、文献里到底怎么做**。

---

## 0. 一页结论

1. **FoldX 不是 structure-prediction 工具。** 它是 empirical force field + rotamer 优化器，属于「在已有 backbone 上换残基 + 重排侧链」这一类，**没有从序列折叠的能力**。原文逐字：*Bioinformatics* 41(2):btaf064 (2025) — "FoldX is a software based on a rigid-solid approach where **the backbone doesn't change upon mutation**."

2. **「很慢」这个前提是反的。** 实测 210 CPU-s / mutation（StaB-ddG, ICML 2025 §Running time，FoldX v4.1）：比 GPU inverse folding 慢 ~1000×，但**比 Rosetta Flex ddG 快 ~250×**；对本任务的规模（376,446 variants ≈ 21,959 CPU-h）在 Ibex 上是一天量级。**卡住这条路的是 rigid backbone 与 license，不是速度。**

3. **必须把三件事拆开**，用户原问句里的四个工具横跨了其中两类：
   - **(a) co-folding / de-novo prediction** — AF2-Multimer / AF3 / Boltz / Chai-1 / Protenix / OpenFold3 / ESMFold(2)
   - **(b) fixed-backbone（或局部松弛）mutant building** — FoldX / Rosetta / EvoEF / FASPR / SCWRL / learned packers ← **「mutant complex」真正所在的类别**
   - **(c) structure token / embedding / 训练集伪标签** — ProSST / SaProt-3Di / ProstT5 / ESM-IF1 的 AFDB 蒸馏
   把 co-folding 跑在 mutant 序列上看似 (b)，实为重跑 (a) 然后落回 wild-type basin。

4. **⚠️ 对本项目最要紧的一条：那 2,080 个 FoldX 结构对 BindingGYM 的三个 structure-based baseline 在构造上不可见。**
   FoldX 不动 backbone；而 ProteinMPNN 只读 N/Cα/C/O + virtual Cβ、ESM-IF1 只读 N/Cα/C、Foldseek 3Di 由 Cα 与 virtual center 导出 —— **全是 backbone-only consumer**。
   §12 计划的 gate 实验「WT backbone + ProteinMPNN vs FoldX mutant backbone + ProteinMPNN」很可能是把 WT 实验重跑一遍，得到 null 是**数学上的必然**，不是关于生物学的证据。**下结论前必须先做 §5.1 的零算力实测。**

5. **文献里真正喂 mutant backbone 的方法，2023–2026 基本是零；唯一做过该对照的组报告它没用**（ThermoMPNN, PNAS 2024：PPV 未提升，Fireprot 上反而变差）。BindingGYM 自己已发表的三个 structure baseline，结构侧信息对 variant 而言都是常数。详见 §4。

6. **一处必须打折的既有引用**：AbBiBench leaderboard 上的「AF3 = −0.02, rank 17」在**论文表格里不存在**，只出现在 GitHub README，且仓库无 `models/AF3`。引用前需自行核对其 scoring 脚本。

---

## 1. 方法与证据等级

**做法：** 一个 11-agent 的 workflow —— 5 个 lens 并行调研（co-folding 预测器 / 快速与单序列折叠器 / fixed-backbone repacker / ddG 文献的实际做法 / 吞吐与 license），每个 lens 配 1 个**对抗性核查 agent** 逐条复查（工具是否真实存在、年份 / license / 速度数字是否被正确归因、是否把训练时间当成推理时间），最后 1 个 agent 合成。共产出 **113 条工具/方法条目**，全部带 verdict；核查 agent 另补报 **58 条 finder 漏项**。

**证据等级（沿用上游 record 的口径）：**

| 标记 | 含义 |
|---|---|
| **逐字引用** | 已从论文正文 / 官方文档 / 仓库文件抓到原文，可直接引用 |
| **verified** | 条目本身经独立核查通过（存在性、年份、license） |
| **CORRECTED** | 核查改正了 finder 的某个细节，正文写的是改正后的值 |
| **UNVERIFIED** | 核查明确无法证实 —— 正文保留但已标注，**不要当事实引用** |

**已知的核查局限：**
- FoldX 的 Academic License Agreement 全文在登录墙后，「禁止重分发生成的坐标」只有第三方转述（GearBind README），**官方条文未取到**。
- 部分 speed 数字是跨硬件比较（CPU-hours vs GPU-minutes），论文自身即如此表述，复述时不可说成同等硬件。
- 若干 license 与多链支持标注为 UNVERIFIED，落地前须自行确认（尤其 FAMPNN 的多链支持）。

---
## 2. Q2 — FoldX 是不是 structure-prediction 工具？它是不是「很慢」？

### 一句话结论

**不是。而且「很慢」这个前提是错的——至少在你关心的尺度上是错的。** FoldX 慢不慢取决于和谁比：相对 GPU 神经网络它慢 ~1000×，相对 Rosetta Flex ddG 它快 ~250×，相对 BindingGYM 这个任务本身它**完全跑得动**（全量 376,446 个 variant ≈ 22,000 CPU-hours，在 Ibex 上 1,024 核约一天）。真正卡住这条路的是两件别的事：**rigid backbone** 和 **license**。

### 它到底是什么

FoldX 是一个 **empirical force field + rotamer 优化器**，属于你分类里的 (b)，不是 (a)。

- 输入**必须**是已有的高分辨率 3D 坐标。给它一条序列，它什么都做不了——没有 fold prediction 机制，没有 docking。
- `BuildModel` 换掉指定位点的残基，重排该位点及邻居的 side-chain rotamer，**backbone 保持刚性**。原文逐字（Delgado J, Reche R, Cianferoni D, Orlando G, van der Kant R, Rousseau F, Schymkowitz J, Serrano L, "FoldX force field revisited, an improved version", *Bioinformatics* 41(2):btaf064, 2025-02-06）：

  > "FoldX is a software based on a rigid-solid approach where the backbone doesn't change upon mutation."

- 官方 wiki 对 `RepairPDB` 的表述同样明确：只重排 side chain，不动 backbone。做 backbone move 的模块（BackX）在官网上标为 "currently under development"。
- `AnalyseComplex` 算链间 interaction energy，这是 BindingGYM / GearBind 用它的原因。

**一个必须加的限定**（这是我在核对时发现、而通常被略过的）：BuildModel 为每个 mutant **同时输出一份配套的 repaired WT PDB**，原因是 "each mutation will move different neighbours"。所以「mutant 与 WT backbone 逐原子相同」这句话，严格说是**相对 FoldX repaired 的 WT reference** 成立，不是相对原始晶体结构。下面第 4 节会讲这为什么对你的 gate 实验很关键。

### 「慢」——拆成三个比较，结论各不相同

唯一可引用的实测 anchor 来自 StaB-ddG（Deng A, Householder K, Wu F, Thrun S, Garcia KC, Trippe B, "Predicting mutational effects on protein binding from folding energy", **ICML 2025**, arXiv:2507.05502），§Running time 逐字：

> "Flex ddG is the most computationally expensive of the three, requiring roughly 15 CPU hours per mutation. For FoldX, initial 'repair' steps are computed on the wild-type interface PDB followed by scoring of individual mutants. On our filtered SKEMPI binding dataset, the total compute time was roughly 260 CPU hours for 4451 mutants (210 seconds per mutation). For StaB-ddG, by contrast, predictions on the same dataset took 13 NVIDIA-5090 GPU-minutes with batched computation (0.2 seconds per mutation)."

（该文用的是 **FoldX v4.1**，不是 5.x。）

| 比较对象 | 倍数 | 说明 |
|---|---|---|
| vs StaB-ddG（batched GPU inverse folding） | FoldX 慢 **~1000×** | 210 CPU-s vs 0.2 GPU-s；**跨硬件比较**，论文自述为 "a 1000× speedup over FoldX on a single device" |
| vs Rosetta Flex ddG | FoldX 快 **~250×** | 15 CPU-h vs 210 s。注意 15 CPU-h 是 StaB-ddG 那套配置（Rosetta 3.8、35,000 backrub steps、10 models 平均），**是可调参数不是常数** |
| vs BindingGYM 任务本身 | **一点都不慢** | 见下 |

外推（算术已复核）：

- 2,080 variants → 2,080 × 210 s = **121 CPU-h**（单核 5 天；64 核 ~2 h）
- 376,446 variants → **21,959 CPU-h ≈ 915 CPU-days ≈ 2.5 CPU-years**
  - 256 核 → ~3.6 天；1,024 核 → **~21 h**
- CPU-bound、单线程、embarrassingly parallel → SLURM array job 线性扩展。本地 A4500 完全用不上。

**外推的四个已知漏洞，不要当精确值用：**
1. 210 s 是 SKEMPI 上 repair+scoring 的**聚合**协议耗时，RepairPDB 成本摊在少数几个 WT interface 上；BindingGYM 有 22–25 个不同 WT 结构，摊销结构不同。
2. **FoldX BuildModel 按 variant 计费，不按 mutation 计费**。BindingGYM 含组合文库式 assay，相对 SKEMPI（近乎全单点）的 per-variant 成本可能更**低**而非更高。
3. 分母本身没被验证过：376,446 里去重后真正需要构建的 unique (backbone, mutation-set) 有多少？没人查过。
4. 只评估了逐个 `BuildModel` 这条最贵的命令路径；FoldX 另有为饱和扫描设计的 `PositionScan` / `Pssm`，成本结构不同，**未评估**。

存储：376k × ~0.5 MB ≈ 190 GB 未压缩 / ~50 GB gzip（估计）。**真正的风险是 inode 不是字节**——37.6 万个小文件在 Lustre 上消耗的是元数据服务器，这类作业常被 inode 配额卡死而不是容量。要打包成 tar/HDF5/LMDB，或按 assay 分片。

### 对你最要紧的一点：rigid backbone × backbone-only consumer

这不是一个 speed 问题，是一个**特征可见性**问题：

| 环节 | 事实 |
|---|---|
| FoldX BuildModel | backbone 不动（官方原文，上引） |
| ProteinMPNN | 只读 N / Cα / C / O + 由主链推出的 virtual Cβ（Dauparas et al., *Science* 2022） |
| ESM-IF1 | 只读 N / Cα / C（连 O 都不读） |
| Foldseek 3Di（SaProt 消费的） | 由 Cα 与相对 N/Cα/Cβ 定义的 virtual center 算出——**backbone-derived** |
| BindingGYM 的 `protein_mpnn_utils.py` | `sidechain_atoms = ['N','CA','C','O']`（**来自你自己的源码审计，我未复核**） |

→ **BindingGYM 的三个 structure-based baseline 全是 backbone-only consumer，而 FoldX 从不改 backbone。** 那 2,080 个 Zenodo FoldX mutant complex 在 ProteinMPNN 眼里与配套 WT 近乎逐原子相同。你那个「mutant backbone 是否加信号」的实验，很可能**在构造上**就是把 WT 实验重跑了一遍——得到 null 是数学上的必然，不是关于生物学的证据。

**这一条我建议你在写下任何结论前先实测掉**，成本几分钟 CPU，见文末建议 #1。

### License：限制的是什么，以及我没能核实的部分

| 项 | 状态 |
|---|---|
| Academic 免费 | **CONFIRMED**——btaf064 的 Availability 节：'FoldX versions 4.1 and 5.1 are freely available for academics'；官网分 Academic / Evaluation / Commercial 三档 |
| 禁止重分发**软件** | 高可信（Academic License Agreement 全文在登录墙后，我读不到） |
| 禁止重分发**生成的坐标** | **只有第三方转述**。GearBind README 第 38 行逐字：'Note that we can not provide FoldX-generated HER2 and CR3022 mutant structures due to license restrictions.' 我在任何 FoldX 官方可读页面上找不到对应条文 |
| 「未经 CRG 书面同意不得向第三方发表 benchmark 结果」 | **UNVERIFIABLE**——这句在 evaluation license 里有近似表述，academic license 我读不到。这是本条 legal 分量最重的一句，却是唯一无法证实的 |

→ 如果这一点会影响你的发表，注册后自己读一遍原文；或者直接绕开（见建议 #7）。

---

## 3. Q3 — 除 ESMFold / AF3 / AF2-Multimer / FoldX 之外，还有什么

先把你那句「打 structure-prediction 伪标签」拆开——**这是整份答案的骨架**：

- **(a) co-folding / de-novo prediction**：从序列折出结构。AF2/AF3/Boltz/Chai/Protenix/OpenFold3/ESMFold(2)。
- **(b) fixed-backbone（或局部松弛）mutant building**：在已知 WT backbone 上换残基 + 重排侧链（± 局部极小化）。FoldX / Rosetta / EvoEF / SCWRL / FASPR / learned packers。**这才是「mutant complex」的所在类别。**
- **(c) structure token / embedding / 训练集伪标签**：不出坐标。ProSST / SaProt-3Di / ProstT5 / ESM3 tokens / ESM-IF1 的 AFDB 蒸馏。

**把 co-folding 跑在 mutant 序列上，看起来像 (b)，实际是重跑 (a)，然后落回 wild-type basin。** 这是全篇最重要的类别区分。

### (a) Co-folding prediction

| 名称 | 年份 | Complexes | Speed（标注来源） | License | 对 mutant 是否 sensible |
|---|---|---|---|---|---|
| **AlphaFold2-Multimer / ColabFold** | 2021-10 preprint；v2.3 = 2023-01；ColabFold *Nat Methods* 19:679 (2022) | yes（蛋白-蛋白） | ColabFold **verified**："close to 1,000 structures per day on a server with one GPU"（≈86 s/结构，**含 MSA，单体**）。complex 的 per-structure 秒数 **UNVERIFIED** | code Apache-2.0；**params CC BY 4.0**（README 逐字）——可商用、可重分发，最干净的 AF 血统 | **no**。MSA-driven，单点突变几乎不改 MSA。Lu et al. (bioRxiv 2024.05.25.595871) SKEMPI 子集：Δranking_score Pearson **0.21** |
| **AlphaFold 3** | *Nature* 630:493–500 (2024)；weights 2024-11 | yes（最广） | **verified**（官方 docs/performance.md，仅推理）：1024 tokens 62 s (A100-80GB) / 34 s (H100)；2048 275/144；4096 1434/774 | **最危险**。code Apache-2.0（2026-06-09 从 CC BY-NC-SA 改，commit 已核）；weights 仅 non-commercial 组织、**禁止对外分享**；output 禁止 "train machine learning models... for biomolecular structure prediction similar to AlphaFold 3"（含 distillation）；**且 "You must not share Output with any commercial organization"**，仅 scientific publication / open source release / journalism 三种例外 | 坐标 **no**（见下失效模式）；但 Δranking_score 作为**标量** Pearson **0.49**（Lu et al., 475 mutants / 42 complexes，且**丢弃了所有 ranking score < 0.8 的样本**），与 FoldX 的 0.49 打平。注意：这是 (c) 类标量，**喂不进 ProteinMPNN** |
| **Boltz-1 / 1x / 2** | 2024-11 / 2025-06 | yes | **UNVERIFIED**。Boltz-2 preprint 全文**没有任何秒数或 GPU 型号**；"~20 s/ligand on H100" 是两句话拼接的推断；34 s/sample on GH200 来自第三方（arXiv:2510.18870） | **MIT**，code + weights + 完整训练 pipeline，学术商用皆可。全表最干净 | 结构侧同 AF3。**affinity head 是 protein–ligand ONLY**：官方 docs 逐字 "Boltz only supports the computation of affinity of small molecules to protein targets"，且 "does not explicitly handle... multimeric binding partners"；ligand 上限 **56 atoms**（不是 50） |
| **Chai-1** | bioRxiv 2024.10.10.615955 | yes | **UNVERIFIED**。硬件：A100/H100/L40S 推荐，A10/A30 可跑小复合物，用户报告 RTX 4090 可行 | **Apache-2.0，code + weights**（v0.4.0, 2024-11 底 relicense；旧资料说「权重非商用」已过期） | 同族限制。但 **MSA-free 是默认行为**（README 逐字："By default, the model generates five sample predictions, and uses embeddings without MSAs or templates"）——做「MSA-free 是否更 mutation-sensitive」的实验，它是零配置成本的那个 |
| **Protenix v0.5 / Mini / v1 / v2** | 2024-11 repo；v1 **2026-02-05**（368M）；**v2 2026-04-08（464M）** | yes | **UNVERIFIED** | **Apache-2.0，code + weights**，学术+商用 | 同族。但**这是对 BindingGYM 最对口的一个**：v1 抗体-抗原 DockQ 成功率 **52.31% vs AF3 48.75%**；v2 README 自述 "clear gains on antibody-antigen structure prediction... absolute success rate gains of 9 to 13 percentage points over Protenix-v1"，且 "at only 5 seeds already exceeds the performance of Protenix-v1 at 1000 seeds"。另有与 TTT 直接相关的 inference-time scaling 声明 |
| **OpenFold3** | preview 2025-10；持续发版至 **v0.5.0 (2026-08-21)**；默认权重 OpenBind-0（cutoff 2025-06） | yes | **本表最好的同尺寸实测**（NVIDIA NIM, H100-80GB, 1 diffusion sample, ~4 templates/chain）：186 res 10.22 s；287 → 20.92；530 → 18.84；575 → 19.82；590 → 32.47；1286 → 37.17；1496 → 58.83。**显存 40–80 GB → 排除 20 GB A4500** | **Apache-2.0（含 weights）**。Zenodo 软件 DOI 10.5281/zenodo.19001000 | 同族。真正价值在 (c)：**完整训练数据公开**，含 AF3 那个 MGnify-based 13M-sequence distillation dataset 的复现，且**无 AF3 那条禁止训练模型的条款**。仓库名是 `aqlaboratory/openfold-3`，**连字符必需**（`openfold3` 是 404） |
| **ESMFold2 / ESMFold2-Fast** | 发布 2026-05-27；Candido et al., bioRxiv **10.64898/2026.06.03.729735** | **yes**（protein/DNA/RNA/ligand/covalent） | **verified 自 preprint 正文**："With 10 loops and 200 diffusion steps, ESMFold2 predicts a 1024 residue structure in **15.8 seconds**"；ESMFold2-Fast 同目标 **9.4 s**；**均为 single H100**；自称在该设置下比 AF3 快 1.3× | **MIT**（Chan Zuckerberg Biohub；`evolutionaryscale/esm` 现重定向到 `Biohub/esm`） | 「单序列可选 + 原生复合物 + MIT + 抗体界面不差」四者唯一交集。Foldbench antibody-antigen DockQ pass rate：**ESMFold2 单序列 50%±2%，+MSA 53%±2%，AF3+MSA 47%±2%，Fast（无 MSA）50%±2%**。**注意 data cutoff = Sept 2021**，BindingGYM 的 22 个 WT 结构几乎必然在训练集里 |
| ESMFold（原版） | *Science* 379:1123–1130 (2023) | **no**（单体；只有 poly-G linker hack，README 原文含拼写错误 "chains seprated by a ':'"） | **verified**："on a single NVIDIA V100 GPU... 384 residues in 14.2 s, six times faster than a single AlphaFold2 model... up to ~60×" | MIT；**repo 已 archived**（GitHub archived=true），实际部署走 HF `EsmForProteinFolding` | 单体 only，但**它是唯一有实测证据说「对突变更敏感」的**（见下） |
| RoseTTAFold2 / RFAA | 2023 / *Science* 384:eadl2528 (2024) | yes | UNVERIFIED | RF2 MIT；RFAA **BSD，licence 文本明确覆盖 weights** | **no**。RFAA 在 Masters et al. 里 WT baseline 最差（CDK2-ATP RMSD 2.2 Å vs AF3 0.2 Å），且同样 mutation-blind |
| HelixFold-Single | *Nat Mach Intell* 5:1087–1096 (2023) | **no**（单体） | **verified Table 2**（A100-40G, median s，按长度 [1,100]/(100,200]/(200,400]/(400,800]/(800,∞)）：MSA search 737.5/755.4/853.7/977.0/1203.8；AF2 766.1/795.8/908.3/1125.2/1611.2；**HFS 1.5/1.5/2.1/6.2/37.5**。**更正**：<100aa 时相对 AF2 是 **~511×** 不是 1000×，且随长度衰减到 800+ 时只剩 43× | code Apache-2.0 | 单体 only，对 binding 无用。列在这里因为它给了全文献最干净的 MSA-free 速度对照表 |
| OmegaFold | bioRxiv 2022.07.21.500999 | no | UNVERIFIED（A100-80GB 最长 4096 res） | Apache-2.0 | **仓库自 2022-12-12 无提交，实质停更**；至今无期刊版；优先级应低于 ESMFold |
| HelixFold3 | arXiv:2408.16975 (2024-08) | 报称 yes | UNVERIFIED | **非商用（code + weights）**——README 逐字 "available under the LICENSE for non-commercial use by individuals or non-commercial organizations only"。属于 AF3 那一档的受限区，不是模糊地带 | 无理由优先于 Boltz/Chai/Protenix/OpenFold3 |
| **抗体专用**：IgFold / ImmuneBuilder / ABodyBuilder3 | 2023 / 2023 / 2024 | **全部 no**（只出 Fv / VHH / TCR **本体**，不建 antibody-ANTIGEN complex） | IgFold "under 25 s"（硬件未注明）；ImmuneBuilder **~5 s on a Tesla P100**（Methods 逐字），CPU 5 核 <1 min；ABodyBuilder3 无 per-structure 数字 | IgFold **JHU 学术非商用**；ImmuneBuilder **BSD-3**；ABodyBuilder3 **Apache-2.0**，仓库是 `Exscientia/ABodyBuilder3`（**不是 OPIG，不是 BSD-3**——那是 ImmuneBuilder） | 对 BindingGYM 基本无用：没有 antigen 就没有界面，而 binding fitness 全在界面上。**更正一条被广泛转述的假信息**：ABodyBuilder3 论文并**没有**说因 license 限制而排除 IgFold（全文 "licen" 出现 0 次） |
| tFold-Ag | bioRxiv 2024.02.05.578892 | **yes**（唯一抗体专用且真出 Ab-Ag complex 的开源模型） | UNVERIFIED | **PolyForm Noncommercial License 1.0.0**（默认分支是 `master` 不是 `main`）——纯非商用，和 IgFold 同档 | 作者自己的 SAbDab-22H2-AbAg 表：AF-Multimer DockQ 0.158 / SR 18.2%；**AF3 0.257 / 32.3%**；tFold-Ag 0.217 / 28.3%。0.217 低于 acceptable 门槛 0.23，**且 AF3 在这张表上赢它**。常见误传的「DockQ +37%」是相对 AF-Multimer，不是相对 AF3 |

**不要规划的（存在但不可用）**：Chai-2（bioRxiv 10.1101/2025.07.05.663018，2025-07-06，是 zero-shot 抗体**设计**系统，无公开权重）；NeuralPLexer3（iambic-therapeutics 的 GitHub org 只有一个公开仓库 `np-bench`，**无模型权重**；且 arXiv 摘要里**没有** "seconds-fast" 这个速度声明）；Pearl（arXiv:2510.24670，专有；正确数字是 **Runs N' Poses +14.5% / PoseBusters +14.2%**，那个 3.6× 是 pocket-conditional cofolding 在 RMSD<1 Å 门槛下、在**专有内部靶点集**上的，不是 PoseBusters）；Vilya-2（arXiv:2607.25156，peptide/小分子界面，licence 未确认）；Umol（protein-**ligand** only，与 BindingGYM 无关；code Apache-2.0 + params CC BY 4.0）。

**顺带一个可用的效率件**：**Pairmixer**（arXiv:2510.18870，code `genesistherapeutics/pairmixer`，weights `huggingface.co/genesisml/pairmixer`）——drop-in 替换 Pairformer，对 Boltz-1 短序列 1.6×、长序列最高 4×。整份调研里它只被当成「别人跑时长的出处」引用，没人注意到它本身是可部署的。

#### (a) 类的共同失效模式：对点突变不敏感

四条独立测量：

1. **Feldman J, Brogi M, Skolnick J**, *Comput Struct Biotechnol J* 35 (2026), doi:10.34133/csbj.0142, PMID 42376644（preprint bioRxiv 10.64898/2026.02.25.708002）：200 个蛋白的对抗性突变研究，AF3 的预测 "remain invariant to mutations of up to **40% of residues** — including deliberately destabilizing substitutions — and to deletions of 10%"，含 fold-switching 蛋白；confidence metric 至多 35% 的时候能挑出最准的结构。
2. **Masters MR, Mahmoud AH, Lill MA**, *Nat Commun* 16:8854 (2025), doi:10.1038/s41467-025-63947-5, PMC12501370：测了 AF3 / RFAA / Chai-1 / Boltz-1 四家；把 binding site 每个残基都突变掉之后，CASF-2016（n=285）上 **42–52% 的 AF3 预测仍保持 ligand pose 不变**，对照组（不突变）是 78%。（**更正一条常见误引**：论文里 "the ATP molecule remains entirely within the binding site" 那句点名的是 **RFAA 和 Chai-1**，不是 Boltz-1。）
3. **DeltaDiff**（arXiv:2606.04452, 2026-06-03, Cai Y / Wang Y / Chen M, Purdue）：intro 把机制讲得最清楚——"When a mutant sequence differs from the wild type by only one residue, its MSA representation may remain highly similar to that of the wild type. As a result, the predicted mutant structure may remain artificially close to the wild-type structure"；实测 BBL D162N 上 "AlphaFold3 can only generate wild-type-like structures"。
4. **Pak MA et al.**, *PLoS ONE* 18(3):e0282689 (2023)：ΔpLDDT 与 ΔΔG 相关 **−0.17 ± 0.03**；"AlphaFold predictions are unlikely to be useful for ΔΔG predictions"。更早的 **Buel GR & Walters KJ**, *Nat Struct Mol Biol* 29(1):1–2 (2022)："largely unable to predict when a point mutation causes defective protein folding"。

**诚实的反面**（避免你在论文里过度声称）：**McBride JM, Polev K, Abdirasulov A, Reinharz V, Grzybowski BA, Tlusty T**, "AlphaFold2 Can Predict Single-Mutation Effects", *Phys Rev Lett* 131:218401 (2023)——用 effective strain 在 3,901 对结构上，AF2 **在群体平均意义上**确实捕捉到单突变效应的量级与范围。但那是 population statistic，不是逐 variant 精度，且不迁移到 PPI：Lu et al. 的 SKEMPI 表里 Effective Strain 只有 Pearson **0.18**。正确的表述是「co-folding 模型在**逐个 mutant 坐标**层面对突变不敏感」，不是「AF2 对突变一无所知」。

**唯一的正面线索**：同一篇 CSBJ 2026 逐字写 "**ESMFold exhibits greater, though still imperfect, mutational sensitivity**, suggesting a tighter coupling between sequence identity and predicted structure that may reflect differences in training objective rather than overall model quality"。注意作者自己的三重 hedge（"still imperfect"、"may reflect... rather than overall model quality"），**且他们测的是单体不是界面**。这指向 MSA 是罪魁——但 MSA-free co-folding 在 PPI 界面上是否 mutation-sensitive，**至今没有任何人测过**。这是一个真空，也是一个便宜的实验。

**一条对你更有用的正面结果（关于 WT 而不是 mutant）**：**Wee J & Wei G-W**, *J Chem Inf Model* 64(16):6676–6683 (2024), doi:10.1021/acs.jcim.4c00976（preprint arXiv:2406.03979, PMC11177964）：SKEMPI 2.0（317 complexes / 8,338 mutations）上，把实验 WT 复合物换成 **AF3 预测的 WT 复合物**，下游几乎无损——**Pearson 0.86 vs 0.88，RMSE +8.6%**。两条警告同样重要：**"there is little correlation between ipTM score and prediction RMSD"**（别拿 ipTM 当质量筛），以及 AF3 复合物对 intrinsically flexible region/domain 不可靠。→ **co-folding 的正确用途是补 WT 覆盖缺口，不是造 per-variant mutant。**

### (b) Fixed-backbone / 局部松弛的 mutant building

| 名称 | 年份 | Complexes | Speed | License | 对 mutant 是否 sensible |
|---|---|---|---|---|---|
| **FoldX BuildModel** | 2002 / 2005 / btaf064 2025 | yes（AnalyseComplex） | **210 s/mut**（verified，v4.1，SKEMPI） | 学术免费；输出重分发受限（**依据只有 GearBind README**，见 Q2） | 本职工作，但 **backbone 不动 → 对 ProteinMPNN 不可见** |
| **EvoEF / EvoEF2 BuildMutant** | 2019 / *Bioinformatics* 36(4):1135 (2020) | yes（ComputeBinding） | **UNVERIFIED**（作者动机是 "FoldX... rather slow"，无发表数字） | **两个仓库都带 MIT LICENSE 文件**（"The MIT License (MIT) Copyright (c) 2019, Xiaoqiang Huang"），但 README 第 125 行写 "EvoEF2 is free to academic users."——**仓库内部自相矛盾**，发表前建议直接问作者 | **FoldX 最接近的开源替代，且能绕开重分发死结**。注意：**做 ΔΔG 用 EvoEF1**（权重在热力学突变数据上优化），EvoEF2 的权重是为 sequence recapitulation 优化的 |
| **Rosetta Flex ddG** | *J Phys Chem B* 122(21):5389–5399 (2018) | yes（就是为 PPI interface 设计的） | **15 CPU-h/mut**（verified，但**是 StaB-ddG 那套配置**：Rosetta 3.8 / 35,000 backrub steps / 10 models 平均——**成本随 nstruct 线性缩放，不是常数**） | Rosetta non-commercial 免费；licence 只禁止分发**软件**，**未提及输出结构** | **backrub 真的动 backbone**。376k → 5.65M CPU-h ≈ 644 CPU-yr，不可行；**5,000-variant 子集 = 75,000 CPU-h，且降 nstruct 可再砍**。正确作者是 Barlow, Ó Conchúir, Thompson, Suresh, **Lucas, Heinonen**, Kortemme（**没有 Lyskov**） |
| **Rosetta cartesian_ddG** | Park et al., *JCTC* 12:6201 (2016) | 有 interface mode（*_bj.pdb 结合态 / *_aj.pdb 解离态），但官方文档自承 "**This has not been thoroughly benchmarked**" | UNVERIFIED | 同 Rosetta | **分类更正：它不是 fixed-backbone。** 官方文档逐字："the use of Cartesian-space refinement which allows **small local backbone movement** during the refinement procedure." **这是最便宜的、能真正产生 backbone 差异的经典方法**，此前被误分类掉了 |
| **PyRosetta FastRelax**（ThermoNet 式） | *PLoS Comput Biol* 16(11):e1008291 (2020) | yes | UNVERIFIED | 同 Rosetta（输出无限制） | 同样动 backbone，脚本可控。比 FoldX 是**更公平的检验**，但 cartesian_ddG 更便宜，应先试后者 |
| Rosetta ddg_monomer | *Proteins* 79(3):830–838 (2011) | **no**（单体 stability） | UNVERIFIED | 同上 | 对 binding 不适用；价值在它那条「固定 backbone 就够用」的早期结论——但该结论是在**单体 stability** 上得到的，往界面推需要额外论证 |
| **FASPR** | *Bioinformatics* 36(12):3758–3765 (2020) | yes（多链输入） | **verified**："FASPR achieved the highest speed for packing the 379 test protein structures in only **34.3 s**" ≈ **0.09 s/结构**（比默认 SCWRL4 快 26.77×）。→ 376k ≈ **9.4 CPU-h**。硬件未注明，且是单体 | **MIT** | 376k 尺度上最便宜的「只换侧链」方案，但**同样对 backbone-only consumer 不可见** |
| SCWRL4 | *Proteins* 77(4):778–795 (2009) | yes——主页逐字："SCWRL4 will converge on very large proteins or **protein complexes**... while SCWRL3 sometimes would not" | FRM 比 SCWRL3 慢 3–6×；绝对秒数 UNVERIFIED | 非营利免费（注册），营利需 Dunbrack/FCCC 授权。**主页对输出坐标一个字都没提——沉默不等于许可** | 同上 |
| **FlowPacker** | *Bioinformatics* 41(3):btaf010 (2025) | **yes，且这一类里唯一有界面证据的** | RTX 3060 上 "best RMSD-runtime tradeoff"，**论文未给秒数** | 论文 CC BY；仓库 LICENSE 未核 | SAbDab **104 个 antibody-antigen complex clusters**，跨链 edge 的相对位置编码固定为 32，"**despite being trained exclusively on monomeric proteins**"（zero-shot 迁移，不是训进去的）。做的是 side-chain **packing** 不是 design。**仍然不动 backbone** |
| **DiffPack** | NeurIPS 2023, arXiv:2306.01794 | UNVERIFIED | UNVERIFIED（相对：FlowPacker > DiffPack > AttnPacker > Rosetta） | **MIT**（2023-12 后无维护） | **对你的用法有致命问题**：FlowPacker 论文因发现 **data leakage（"generation quality depended on input side-chain coordinates"）** 而把它排除出主排名。你的用法正是「WT 结构 + mutant 序列」，输入里带 WT 侧链 → 产出被 WT 污染 |
| **AttnPacker** | *PNAS* 120(23):e2216438120 (2023) | UNVERIFIED | **verified 自附录**："reconstruct all side-chain atoms for all the CASP13 targets in **68 s**"（83 个 target ≈ 0.8 s/蛋白，**single RTX A6000**）。注意 "over 100×" 是 GPU 对 CPU 的跨硬件比较 | **仓库无 LICENSE 文件**（GitHub API: license = null）→ 默认保留一切权利。**这是比「未知」更糟的状态**，不能用于发布衍生数据 | 同上 |
| PIPPack | *Proteins* 92(10):1220–1233 (2024) | UNVERIFIED | 只有相对声明 | **MIT** | PMID 应为 **38790143**（38187664 指向的是 bioRxiv 预印本） |
| **MODELLER 10.8** | 1993–；**10.8 发布于 2025-11-06**，仍在活跃维护 | yes（template 是多链就能建多链） | UNVERIFIED | 学术免费需 key；**免费仅限 academic non-profit**——商业实体与政府研究机构（原文点名 NIH 与 US national labs）必须走 non-academic licensing。licence 限制的是 PROGRAM 与 IMPROVEMENTS，**输出坐标不在字面射程内**（这是解读不是条文） | 可用但没有理由优先：不给能量、也不为 ΔΔG 调过参 |
| **ProMod3** | SWISS-MODEL 的建模引擎 | yes（homo-/hetero-oligomer） | UNVERIFIED | **Apache-2.0**，`git.scicore.unibas.ch/schwede/ProMod3`，基于 OpenStructure | **更正一条常见误判**：SWISS-MODEL 不是「只有 web server」。引擎可本地部署批跑。真正的限制只是 public server 跑不了 376k |
| **MutateX** | *Brief Bioinform* 23(3):bbac074 (2022) | yes（摘要逐字 "multimeric assemblies"） | 取决于底层 FoldX | **GPL-3.0**，但**必须先装 FoldX，因此继承 FoldX 全部约束** | 如果最终要跑 FoldX 全量，别自己写调度器 |
| **OpenMM / PDBFixer** | *PLoS Comput Biol* 13(7):e1005659 (2017) | yes | UNVERIFIED | **MIT**，全表许可最干净 | 不是突变工具，是收尾工序。但**短程局部 MD 松弛是全表被整体漏掉的一整支**——它落在 FoldX（0 backbone 运动）与 Flex ddG（15 CPU-h）之间，GPU-bound，A4500 跑得动，是做 backbone gate 实验最自然的工具 |

**一条对「预测结构 + 再 repack」路线的负面证据**：**Vangaru S & Bhattacharya D**, "To pack or not to pack: revisiting protein side-chain packing in the post-AlphaFold era", *Brief Bioinform* 26(3):bbaf297 (2025), PMC12192453——评了 8 个 packer（SCWRL4 / Rosetta Packer / FASPR / DLPacker / AttnPacker / DiffPack / PIPPack / FlowPacker）。两条：(i) "the three most recent ones (FlowPacker, PIPPack, DiffPack) had substantially higher rotamer RRs than all other tools"；(ii) 更要紧的是——**在 AlphaFold 生成的 backbone 上，"the baseline AlphaFold-generated side-chains exhibit the best performance for almost all metrics compared with any of the PSCP methods"**，即再 repack 反而更差。**注意该研究只评 单链 CASP14/15 的 WT target，完全没有涉及 mutant。**

### (c) Structure token / embedding / 训练集伪标签

| 名称 | 年份 | Complexes | Speed | License | 对 mutant 是否 sensible |
|---|---|---|---|---|---|
| **ProSST** | NeurIPS 2024 | partially（逐残基量化，多链技术可行；**评测全在 ProteinGym 单体**） | 量化是 forward + 查表；376k 个 mutant **只需 25 次结构量化** → 成本 ≈ 0 | **CC-BY-NC-ND-4.0**——NonCommercial **且 NoDerivatives**。**ND 对 TTT 是硬伤**：fine-tune 产出的正是 derivative works | 打分公式里 "**Here, s is the structure token sequence of the wild type.**"（逐字）——即 ProSST **本身就是 WT-structure + mutant-sequence 范式**。**更正一个被广泛引用的数字**：Table A9 里 structure source 的干净同模型对比是 **AF2 0.504 vs ESMFold 0.471，spread = 0.033**，不是 0.11（0.438 是 MST 掩码训练的**另一个模型**，0.392 是无结构轨道的**又一个模型**） |
| **SaProt + Foldseek 3Di** | ICLR 2024 / *Nat Biotechnol* 42(2):243–246 | partially（3Di 是逐残基局部描述子，**不编码跨链接触**） | 毫秒级/结构；同样只需 25 次 | SaProt **MIT**（已核）；Foldseek GPLv3（likely，未逐字核） | 3Di 由 Cα + N/Cα/Cβ 定义的 virtual center 算出 → **FoldX 的侧链重排改变不了任何一个 3Di token**。BindingGYM 的做法是 WT 3Di + mutant 氨基酸（**你自己的审计**） |
| **ProstT5** | *NAR Genom Bioinform* 6(4):lqae150 (2024) | no（逐链） | 一次 T5 forward/序列，**估计 ~5 GPU-h 跑完 376k**（估计，非发表值） | **MIT** | **全表唯一能给 376k 个 mutant 各出一份不同 structural token、成本却接近零的方案**：mutant 序列 → mutant 3Di → 喂 SaProt。**天花板要说清楚**：它学的是 sequence→structure 映射，单点突变对输入扰动同样极小，输出 3Di 极可能与 WT 几乎一致。**当作 ~5 GPU-h 的廉价 ablation，不要当希望** |
| **ESM3 structure tokens** | *Science* 387:850–858 (2025) | UNVERIFIED | UNVERIFIED | **licence 已变更为 MIT**（Chan Zuckerberg Biohub 接手后，`Biohub/esm` README Licenses 节逐字 "These models are available under the MIT license"；HF `esm3-sm-open-v1` model card 正文亦写 MIT）。2024 年的 Cambrian Non-Commercial 说法**已过期**。**但 HF metadata 混乱**（该 model card 的 YAML frontmatter 没有 license 字段；ESMC-600M 同时挂 `mit` 和 `other`）→ **逐 checkpoint 当天再核一次** | VQ-VAE codebook 4096 + 700M all-atom decoder，理论上可在 token 空间做 mutant 条件生成再解出坐标。但仍是「模型的想象」，且 leakage 与规模都是问题 |
| **ESM-IF1 的 AFDB 蒸馏** | ICML 2022, PMLR 162:8946–8970 | 推理侧 yes（`--multichain-backbone`）；蒸馏训练集是单体 | 训练侧一次性（12M 次 AF2 预测） | facebookresearch/esm 的 LICENSE 是 **MIT**，README License 节只提 MIT + Atlas 数据 CC BY 4.0。**「ESM-IF1 权重是 CC-BY-NC 4.0」这条流传很广的说法我在仓库里找不到任何一手依据 → 标 UNVERIFIED，不要当成既定陷阱** | **这是「structure pseudo-label」最正统的用法，且与突变无关**：给 12M 条**天然序列**各配一个预测结构。README 逐字："trained with **12M protein structures predicted by AlphaFold2**"，51% native sequence recovery（埋藏残基 72%）。BindingGYM 的 ESM-IF1 baseline 则是 `coords_list = [coords] * len(seq_list)`（**你自己的审计**） |
| **OpenFold3 的公开训练数据** | 2026 | — | — | **Apache-2.0** | 含 AF3 那个 **MGnify-based 13M-sequence distillation dataset 的复现**。这是 (c) 类「用预测结构做训练集扩张」目前最大的、可自由使用的现成资产，且**不受 AF3 那条禁令约束** |

#### 一条信息论上的反驳，直接针对「打伪标签」这个类比

ESM-IF1 的 12M 伪标签之所以有效，是因为它们对应 **12M 条互不相同的天然序列**——伪标签带来的是真正的**结构多样性**。你的 376,446 个 mutant 结构来自 **22 个 backbone**，彼此近乎重复。即使全部生成成功、生成器完美，新增的结构信息量也就是 **22 个结构**的量级，其余全是冗余。**大规模伪标签在你这个设定下不成立。**

### 预计算库：没有 shortcut，而且是构造性的

| 库 | 规模 | Complexes | License | mutant 覆盖 |
|---|---|---|---|---|
| **AFDB** | 214M（2024 版 *NAR* 52(D1):D368–D375, doi:10.1093/nar/gkad1011）；2025 版 *NAR* 54(D1):D358–D362, doi:10.1093/nar/gkaf1226（online 2025-11-22） | 2026-03 起有复合物层（**此条来自你自己的记录，我未独立复核**） | CC BY 4.0 | **构造性零覆盖**：主键是 UniProt accession / canonical sequence，schema 里根本没有 variant 字段 |
| **ESM Atlas (Biohub 2026)** | 6.8B 序列 / ~1.1B 结构，由 ESMFold2 生成 | **no**（单体；此结论依据你自己的 Lance schema 审计） | **CC BY-SA 4.0**（AWS Registry of Open Data）——**不是 MIT**。MIT 适用于 HF 上的 ESMFold2 **权重**，不是这份**数据集**。share-alike 是 copyleft，衍生物可能被要求同等条款发布 | 零覆盖（天然/宏基因组序列，非设计点突变） |
| PDB-REDO | >180k entries（计数 UNVERIFIED） | yes | UNVERIFIED | 只 re-refine 已解出的晶体结构，不生成新结构 |
| SKEMPI 2.0 | 7,085 个 binding 突变 | yes | 学术公开 | **常见误读**：'structurally resolved' 指 **WT 复合物**在 PDB 里有结构，不是每个突变体有坐标 |

---

## 4. 最有决策价值的一条：文献里到底有多少方法真的喂 mutant backbone

**答案：2023–2026 年这条 SOTA 线上，基本是零；而且唯一做过这个对照的组，报告它没用。**

下面这些是**读了 dataset loader 源码或论文正文**得到的，不是从摘要推的：

**(a) 只用 WT 结构 + 序列改变 —— 主流，且在稳步扩大：**

| 方法 | 证据 |
|---|---|
| RDE-Network (ICLR 2023) | `rde/datasets/skempi.py` 只加载 `{pdbcode}.pdb`，然后 `aa_mut = data['aa'].clone(); aa_mut[seq_map[ch_rs_ic]] = one_to_index(mut['mt'])` |
| DiffAffinity / SidechainDiff (NeurIPS 2023) | 逐字同样的 loader 模式（lines 191–197） |
| Prompt-DDG (ICML 2024) | 同样模式，**外加** `{'type':'select_atom','resolution':'backbone+CB'}`——**主动丢掉所有侧链**。给它一份完美 repack 的 FoldX mutant，它会把 repacking 扔掉 |
| Light-DDG (ICLR 2025) | WT only；且报告在推理时给 WT 结构加高斯噪声会降到低于它自己的 sequence-only 变体，并指出 "the errors of existing structure prediction are mostly around 1Å" |
| **StaB-ddG (ICML 2025)** | 逐字："we use backbone structures derived from a single complex for all 6 terms. This choice reflects an assumption that **the backbone changes little upon binding and mutation**." Limitations 里承认 "does not model changes in the backbone upon a mutation"——列为 future work |
| **ThermoMPNN (PNAS 121(6):e2314853121, 2024)** | 逐字："we focus on ΔΔG° prediction conditioned on **only wild-type structural information**" |
| EvoIF (2025/2026) | 明写研究者 "assume that a small number of substitutions do not alter the protein's backbone structure"，设 X_wt = X_mt |
| **BindingGYM 自己的 baseline** | 论文说用 FoldX 为每个 mutant 生成结构；代码是 ProteinMPNN 每 assay `parse_PDB` 一次并缓存、只覆盖 `S`；ESM-IF1 `coords_list = [coords] * len(seq_list)`；SaProt 用 WT 3Di 配 mutant 氨基酸（**你自己的审计**）。→ **所有已发表的 BindingGYM 数字，结构侧信息对 variant 而言是常数** |

**(b) 真的构建 mutant 结构 —— 更老的一支，正在萎缩，且没有一个做过消融：**
GeoPPI (2021, FoldX BuildModel)、DDGPred (PNAS 2022，需要两份 PDB)、GearBind (*Nat Commun* 2024，FoldX RepairPDB+BuildModel)。GearBind 的消融（Fig. 2e）改的是架构（multi-relational graph、edge vs residue messaging、side-chain atoms），**从未把 FoldX mutant 换成 WT 做对照**。→ 它用 FoldX 是**继承的惯例，不是测出来的收益**。

**(c) per-mutant co-folding —— 在正经 ΔΔG 文献里基本缺席。**

**有人证明它有帮助吗？没有。一个组测了，结果是负面的：**

**ThermoMPNN (PNAS 2024) 逐字**：
> "We explored data augmentation strategies using modeled mutant structures to provide synthetic 'inverse mutation' training examples, which modestly improved results on Ssym-inverse (PCC = 0.63) but **did not improve the PPV of the model**... We also trained a Siamese variant of ThermoMPNN which takes both a wild-type and (modeled) mutant structure as input... boosted Ssym-inverse performance (PCC = 0.68), but only when both wild-type and mutant structures were provided... metrics for other datasets such as Fireprot (HF) **degraded** under these constraints (PCC = 0.56 vs. 0.65 and PPV = 33% vs. 56%)."

即：mutant 结构买到的是一个**对称性性质**，不是预测力，代价是泛化变差。他们最终发布的是 WT-only 模型。

**一条独立的复合物层面佐证（但要注意 scope）**：King J, Cornwall L, Nica AC, Day J, Sim A, Dalchau N, Wollman L, Meyers J (Synteny), "On fine-tuning Boltz-2 for protein-protein affinity prediction", arXiv:2512.06592 (2025-12-06, MLSB@NeurIPS 2025)。把 Boltz-2 的 affinity module 改造到 PPI 后：TCR3d Pearson **0.153** vs ESM2-650M **0.239**；PPB-affinity(filtered, 8,207 条) **0.338/0.357** vs ProtT5-PAD **0.48/0.51**、ESM2-650M-SC **0.47/0.48**。决定性的对照：**用真实实验结构重训得到 0.159，没有改善**，结论逐字 "suggesting that **structural quality is not the primary performance bottleneck**"。**Scope 警告：这篇跑的是跨复合物的 affinity regression，不是 mutational scan——它是有力的旁证，但不是和你的 DMS 设定同一个实验。**

**必须打的一个折扣（关于 AbBiBench）**：那份被广泛引用的 leaderboard（ProteinMPNN 0.30 rank 1 / ESM-IF1 0.28 / AntiFold 0.21 / Boltz-2 0.13 / FoldX 0.12 / **AF3 −0.02 rank 17**）——核查发现：**AbBiBench 的论文本身 benchmark 了 15 个模型，其表格里根本没有 AF3 / Boltz-2 / FoldX 这三行**（论文自己给的前三名是 0.27/0.27/0.19）；AF3 那行只存在于 GitHub README 的 leaderboard 上，仓库里**没有 models/AF3 目录、没有 run_af3.sh**；而论文正文里 AF3 是被用来预测**野生型**复合物的。→ **「AF3 per-mutant folding 得 −0.02」这个读法是推断，不是有据可查的事实。** 在论文里引用前，去看一眼他们的 scoring 脚本或发一封邮件。

---

## 5. 对本项目的具体建议

按成本从低到高。**我会花的**与**我不会花的**都写明。

#### 1. 零算力，今天就做：把「by construction」从引文推断变成实测事实

拿 Zenodo v1 的 2,080 对，逐对计算 backbone RMSD：

- **对照必须是配套的 `WT_1N8Z_hm_renumbered_{n}.pdb`，不是原始晶体** —— FoldX 为每个 mutant 输出自己的 repaired WT，因为 "each mutation will move different neighbours"。用错对照会得到假阳性差异。
- 先只算 **N/Cα/C**，再算 **含 O**。BindingGYM 的 ProteinMPNN 读 O，而 FoldX 可能在 N/Cα/C 刚性的前提下**重新理想化羰基 O**——如果 O 动了，ProteinMPNN 的分数**会**有微小变化，但那是 repacking artefact，不是 mutant-backbone 信号。

**这条决定你怎么写结论**：如果 N/Cα/C RMSD 全线 ≈ 0.000 Å，那么你此前的 null 结果**否证的是方法不是假设**，应该这样写——而且这本身就是一个比 null Spearman 表强得多的可发表陈述（"FoldX-built mutant complexes are provably invisible to backbone-only scorers"）。

#### 2. 零算力，第二件：换 consumer，而不是造结构

FoldX / EvoEF / FASPR / SCWRL / 所有 learned packer 产出的差异**全在侧链**。要让那 2,080 个已有结构第一次变得有信息量，必须换一个读侧链的 scorer：

- **LigandMPNN**（Dauparas et al., *Nat Methods* 2025）—— ProteinMPNN 官方谱系后继，已发期刊、原生多链、**自带 side-chain packing**。这是我的首选，而整份调研把 #1 建议押在了一篇预印本上却漏了它。
- **FAMPNN**（bioRxiv 10.1101/2025.02.13.637498，`github.com/richardshuai/fampnn`，**MIT**）—— 明确 "models both sequence identity and sidechain conformation"，支持 sequence-only 与 sequence-and-sidechain 两种 conditioning。**但 README 里没有任何多链/complex/interface 的表述 → 多链支持 UNVERIFIED，落地前必须先确认**，否则拿它去消费双链 mutant complex 会直接卡住。
- **RDE-Network / SidechainDiff**（NeurIPS 2023, arXiv:2310.19849, Liu S, Zhu T, Ren M, Yu C, Bu D, Zhang H；正式版收录于 NeurIPS 36, DOI 10.52202/075280-2128）—— interface-native，不要求你先造 mutant 结构，而是把突变位点的侧链构象分布当隐变量建模。licence 与 speed **UNVERIFIED**。

#### 3. ~1 GPU-h 的证伪 gate：co-folding 到底动不动 backbone

取 100 个 BindingGYM variant，跑 MSA-free 的 co-folding，量 **interface backbone RMSD(mutant, WT)** 的分布。**如果中位数 < 0.5 Å，整条 (a) 路当场关掉，省下 6,000+ GPU-h。**

候选（按优先级）：
- **ESMFold2-Fast**（MIT、原生复合物、9.4 s/1024-res on H100、纯单序列）——但注意 **data cutoff = Sept 2021**，你的 22 个 WT 复合物几乎必然在训练集里，所以这个实验只能回答「mutant 与 WT 差多少」，不能回答「预测得准不准」。
- **Chai-1**（Apache-2.0，MSA-free 是**默认**，零配置成本）。
- 顺带回答一个**全领域没人测过的问题**：MSA-free co-folding 在 PPI 界面上是否比 MSA-driven 更 mutation-sensitive。这是 CSBJ 2026 那条 ESMFold 发现的自然延伸，而且它测的是单体不是界面。

**A4500 (20 GB) 能不能跑**：ESMFold2 的官方数字是 H100，A4500 上会慢若干倍且大复合物可能 OOM；**OpenFold3 需要 40–80 GB，直接排除本地**；AF3 官方只给 A100/H100。→ 这一步要么用小复合物本地做，要么上 Ibex。

#### 4. 如果要真正检验「backbone 变化带不带信号」——别用 Flex ddG 当第一选择

正确的候选顺序（都动 backbone，成本递增）：
1. **短程局部 MD 松弛**（OpenMM，MIT，GPU-bound，A4500 跑得动）—— 突变位点 + 邻域，几十 ps。**整份调研把这一整支漏掉了**，而它恰好落在 FoldX（0 运动）与 Flex ddG（15 CPU-h）之间的成本真空里。
2. **Rosetta cartesian_ddG** —— 官方文档明写允许 small local backbone movement，比 Flex ddG 便宜得多，此前被误分类进 fixed-backbone 类而从未进入候选。
3. **Flex ddG** —— 最后才考虑，而且先跑一个低 nstruct 的版本看噪声；15 CPU-h 是可调参数不是常数。5,000-variant 子集的 75,000 CPU-h 可以再砍。

规模：几百个 variant 起步，不是 5,000，更不是 376k。

#### 5. 我不会花的算力

- **22,000 CPU-h 跑全量 FoldX** —— 输出对你现在的 scorer 不可见，且 licence 有摩擦。这条只有在建议 #2 拿到正面结果后才值得。
- **6,000+ GPU-h co-fold 376k 个 mutant** —— 四篇独立测量说结果会是 WT 的近似副本加噪声。**先花 1 GPU-h 证伪（建议 #3），别先花 6,000。**
- **任何基于 AF3 输出的东西**，只要你打算发布结构或用它训练模型 —— 输出条款禁止训练「similar to AlphaFold 3」的结构预测模型（含 distillation），**且禁止与任何商业组织分享 output**（例外只有 scientific publication / open source release / journalism）。字面上你训 binding-fitness predictor 大概率不落在第一条禁令内，但这是灰区，而且分发时要传递条款。Boltz (MIT)、Chai-1 (Apache-2.0)、Protenix (Apache-2.0)、OpenFold3 (Apache-2.0)、ESMFold2 (MIT)、AF2 params (CC BY 4.0) 都没有这个问题。

#### 6. 性价比更高的三件事（都比造 mutant 结构便宜）

- **提升那 22 个 WT 结构的质量**。ProSST 的干净同模型消融显示 structure source 值 **0.033 Spearman**（AF2 0.504 vs ESMFold 0.471）——比任何人报告过的 mutant-vs-WT 效应都大。具体动作：换 PDB-REDO 版本；以及**把 BindingGYM 剥掉的 HETATM 加回去**——你自己的记录里 6 个 KRAS 复合物缺 GDP/GNP，约占 benchmark 的 30%。这是一个有合理效应量、且几小时就能测的 WT-quality 缺陷。
- **ProstT5 mutant-3Di 消融**（~5 GPU-h 估计）—— 唯一能给 376k 个 mutant 各出一份**不同**结构表征、成本接近零的方案。我预期它会 collapse 到 WT，但一天就能证伪。
- **蒸馏标量而不是坐标**（RaSP 配方）。Blaabjerg LM et al., *eLife* 12:e82593 (2023)：训在 **Rosetta 生成的 ΔΔG 标签**上，比 Rosetta 快 480–1,036×，算了 ~**230 million** 稳定性变化（不是常被引成的 ~300M）。Zenodo v1 除了 2,080 个结构还带 **`.fxout` 能量表**——如果你真正想要的是「FoldX 的信号跑在神经网络速度上」，那些标量是更便宜、更有先例的目标，而且数字比坐标文件的 licence 钩子弱得多（**非法律意见**）。

#### 7. 如果最终要公开发布一批 mutant complex

避开 FoldX 和 AF3。可用的替代：
- **(b) 类**：**EvoEF**（MIT LICENSE 文件，但 README 说学术专用——**发布前问一句作者**）；**PyRosetta / cartesian_ddG**（licence 只限制软件，未提及输出坐标——**比 FoldX 干净得多，且这一步是零成本的规避**）；FASPR（MIT）；ProMod3（Apache-2.0）。
- **(a) 类**：Boltz（MIT）、Chai-1（Apache-2.0）、Protenix（Apache-2.0）、OpenFold3（Apache-2.0）、ESMFold2（MIT）、AF2 params（CC BY 4.0）。
- **不要用**：AttnPacker（**仓库无 LICENSE**）、ProSST 的衍生权重（**ND 条款**）、tFold（PolyForm Noncommercial）、IgFold（JHU 非商用）、HelixFold3（非商用）。

---

## 6. 剩余的不确定性（不掩盖）

**Licence 与法务**
- FoldX academic license 全文在登录墙后。「输出坐标不可重分发」只有 GearBind README 的第三方转述；「未经书面同意不得发表 benchmark 结果」我**读不到条文**。若影响发表，注册后自读。
- EvoEF 的 MIT LICENSE 文件与 README 的 "free to academic users" **自相矛盾**。
- ESM3 / ESMC 的 licence 是 **per-checkpoint** 的，且 HF metadata 本身混乱（frontmatter 缺 license 字段、同时挂 mit 和 other 标签）。用前当天再核。
- ESM-IF1 权重的「CC-BY-NC 4.0」说法我**找不到一手依据** → 应标 UNVERIFIED，而不是当成既定陷阱。
- ESM Atlas 数据集是 **CC BY-SA 4.0**（copyleft），与 ESMFold2 权重的 MIT 是两回事。

**数字与实测**
- Boltz / Chai-1 / Protenix / AF2-Multimer 在 BindingGYM 尺寸复合物上的 per-complex 秒数：**全部 UNVERIFIED**。Boltz-2 preprint 全文没有任何绝对运行时间。
- FASPR 的 0.09 s/结构、AttnPacker 的 0.8 s/蛋白、ImmuneBuilder 的 5 s——都是**单体**测的，硬件有的未注明。
- Flex ddG 的 15 CPU-h 与 FoldX 的 210 s 都是**单一来源**、且 configuration/dataset-dependent。
- 「210 s × 376,446」的四个漏洞见 Q2（摊销结构、per-variant 计费、去重后的真实分母、未评估 PositionScan）。
- Ibex 的 per-user 核数/GPU/array size/walltime 配额**没有被查**——决定日历时间的是这些上限，不是 CPU-hours 总量。

**证据链上的断点**
- **AbBiBench 的 AF3 / Boltz-2 / FoldX leaderboard 行**：scoring 程序在论文和仓库里都无据可查，且论文里 AF3 用于预测 **WT**。这是「refolding 更差」这个论点的头号证据，引用前必须自查。
- **FoldX RepairPDB 是否改动原始晶体的 backbone**：未测。这直接决定建议 #1 的对照怎么选。
- **MSA-free co-folding 在 PPI 界面上是否 mutation-sensitive**：全领域空白。
- **FAMPNN 的多链支持**：README 零文档。
- **AFDB 2026-03 复合物层**、**ESM Atlas 是单体库**：均来自你自己的审计，我这轮未独立复核。
- **BindingGYM 最大复合物的 token 数**（1HE8 ≈ 1,107、1N8Z ≈ 1,041）：来自你此前的记录，未独立复核；它决定显存能否落在 A4500 上。

**存在但太早期，不要投入（各自都被核实为真实存在）**
- **DeltaDiff**（arXiv:2606.04452, 2026-06-03，training-free、physics-guided，基座是 **Str2Str**）—— **唯一一个专门为「生成 mutant 结构」而设计的近期工作**，但只测了 3 个体系：Chignolin T8P（**10 残基**）、Novispirin G-10（**18 残基**）、BBL D162N（**50 残基**），全是小单体，**没有复合物**。值得读它的 intro 当权威引用，不要建在它上面。
- **BioEmu**（Lewis et al., *Science* 389(6761):eadv9817, 2025-08-14, PMID 40638710；`microsoft/bioemu` **MIT**）—— equilibrium ensemble emulator，"thousands of statistically independent structures per hour on a single GPU"，训练信号含实验 stability，是少数被设计成对突变敏感的生成器。**但复合物支持 UNVERIFIED / 很可能 no**，且 ensemble 怎么喂 ProteinMPNN 需要重新设计。

**未核实的线索（列出但不要据此行动）**：Twin Peaks（arXiv:2509.22950，第二条 structure-free 的 PPI affinity + mutation effect 结果）；DCFold（arXiv:2605.17899）；IntFold（arXiv:2507.02025）；OpenDDE（arXiv:2607.03787）；PPIformer / PPIRef（Bushuiev et al., ICLR 2024——StaB-ddG 引用它作为「DL 方法在 interface homology split 上不如 FoldX/Flex ddG」这一关键结论的来源）；Giulini et al., *Bioinformatics* 40(10):btae583 (2024)（测「Fv 预测 + information-driven docking 拼出 Ab-Ag 复合物」这一步的误差，正对抗体 Fv-only 家族的可用性问题）；AF2 "initial guess" 协议（Bennett NR et al., *Nat Commun* 2023, doi:10.1038/s41467-023-38328-5——用已知复合物 seeding AF2 而不是从头折，是「naive refold」与「什么都不做」之间的中间路线，CC BY 4.0 params，本轮未评估）。

**一个小但会影响你检索的事实**：bioRxiv 在 2026 年把 DOI 前缀从 `10.1101` 换成了 `10.64898`（ESMFold2 与 CSBJ 那篇 adversarial-mutation 的 preprint 都是新前缀）。用老前缀搜 2026 年的 preprint 会搜不到。
---

## 附录 A — 本次调研的元信息

| 项 | 值 |
|---|---|
| 执行方式 | Claude Code dynamic workflow，11 agents，0 error |
| workflow run id | `wf_79193fef-118`（task `w32538xc5`） |
| 结构 | Survey ×5（并行） → Verify ×5（每个 lens 一个对抗核查） → Synthesize ×1 |
| 5 个 lens | `cofolding` / `fast-folders` / `repackers` / `literature-practice` / `scale-and-cost` |
| 产出规模 | 113 条方法条目（全部带 verdict）+ 58 条 verifier 补报漏项 |
| 消耗 | 1,611,121 subagent tokens；740 次工具调用；elapsed 48.5 min |
| 原始产物 | `refs/pseudolabel_survey_lenses_20260829.json.gz`（5 个 lens 的完整 items + verdicts + missing） |
| agent 逐条输出 | `~/.claude/projects/-home-guoj0f-repos-ProteinTTT--claude-worktrees-bindingGYM-mutation-structure-analysis/b8973200-.../subagents/workflows/wf_79193fef-118/journal.jsonl`（注：该目录以 Claude Code 的 per-cwd project key 命名，即当前 worktree 名 `bindingGYM-mutation-structure-analysis`；worktree 一旦改名此路径即失效） |

## 关联

- [`predicted_complex_structure_assets_20260829-000600.md`](predicted_complex_structure_assets_20260829-000600.md) —— 同 project 的上游资产调研：**有没有现成的** mutant complex 结构（结论：公开资产只覆盖 0.55%）。本篇承接其 §10「如果要自己生成」与 §12「建议的下一步」，并**修正了 §12 的 gate 实验设计**（见本篇 §0.4 与 §5.1）。
