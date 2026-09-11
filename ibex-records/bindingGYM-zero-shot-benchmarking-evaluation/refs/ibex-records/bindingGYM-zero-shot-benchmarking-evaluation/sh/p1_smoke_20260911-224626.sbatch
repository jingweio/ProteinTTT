#!/bin/bash
#SBATCH --job-name=bgym_p1_smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/p1_smoke_20260911-224626_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/p1_smoke_20260911-224626_%j.err
# P1 smoke:在正式开跑前把三件事钉死
#   (1) use_sequence 的语义方向(读源码与 CLI help 矛盾,用实验定案)
#   (2) score_bgym_mpnn.py 能否复现 anchor 的逐 variant 分数(正确性 gate)
#   (3) 实测吞吐,用来定后续 job 的 walltime
# 短 walltime(1h) —— 我方 fairshare 仅 0.0997,长 walltime 会被排到 20h 以后。
set -uo pipefail
CB=/ibex/user/guoj0f/anaconda3
source $CB/etc/profile.d/conda.sh
conda activate ligandmpnn-bgym

SH=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/sh
BG=/ibex/user/guoj0f/share/BindingGYM
MP=/ibex/user/guoj0f/share/model_params/LigandMPNN
MODELS=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/models
OUT=/ibex/user/guoj0f/bindingGYM-zs-benchmark/scores/smoke_proteinmpnn_ar
mkdir -p $OUT

echo "=== §4 CUDA 经验验证(不靠版本号推断) ==="
nvidia-smi | grep -E 'Driver Version|CUDA Version'
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA 不可用"
n = torch.cuda.get_device_name(0); print("GPU:", n)
assert "A100" in n, n
a = torch.randn(4096, 4096, device="cuda"); print("matmul ok:", float((a@a).sum()))
PY
md5sum $MP/proteinmpnn_v_48_020.pt

echo; echo "=== (1) use_sequence 语义判别探针 ==="
python $SH/probe_use_sequence.py --repo $MODELS/LigandMPNN \
  --ckpt $MP/proteinmpnn_v_48_020.pt \
  --pdb $BG/input/structures/2M5A.pdb --idx 20 || echo "probe 失败(非致命,继续)"

echo; echo "=== (2)+(3) 三个小 assay:ProteinMPNN AR,seed=1 M=5 ==="
python - <<'PY' > /tmp/smoke_idx.txt
import pandas as pd
ids = pd.read_csv("/input/BindingGYM.csv")["DMS_id"].tolist()
for a in ["Z-domain_ZpA963_HL2_fitness_2M5A","BH3_Mcl-1_normed_3KZ0","PSD95_CRIPT_1BE9"]:
    print(ids.index(a))
PY
for i in $(cat /tmp/smoke_idx.txt); do
  T0=$(date +%s)
  python $SH/score_bgym_mpnn.py --repo $MODELS/LigandMPNN \
    --dms_mapping $BG/input/BindingGYM.csv \
    --dms_input $BG/input/Binding_substitutions_DMS \
    --structure_folder $BG/input/structures \
    --dms_index $i --dms_output $OUT \
    --checkpoint $MP/proteinmpnn_v_48_020.pt \
    --model_type protein_mpnn --protocol ar \
    --num_seq_per_target 5 --seed 1 --run_id smoke_proteinmpnn_ar
  echo "  idx=$i wall=$(($(date +%s)-T0))s"
done

echo; echo "=== 正确性 gate:与 anchor 的逐 variant 分数对照 ==="
python - <<'PY'
import pandas as pd, glob, os
from scipy.stats import spearmanr
OUT=""; A="/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/refs/anchor_scores"
print(f"{'assay':<38}{'n':>6}{'rho(mine,anchor)':>19}{'rho_DMS_mine':>14}{'rho_DMS_anchor':>16}")
for f in sorted(glob.glob(os.path.join(OUT,'*.csv'))):
    b=os.path.basename(f); af=os.path.join(A,b)
    if not os.path.exists(af): print(f"{b:<38}  (无 anchor 对照)"); continue
    m=pd.read_csv(f); a=pd.read_csv(af)
    assert len(m)==len(a), f"行数不一致 {len(m)} vs {len(a)}"
    r_pair=spearmanr(m.design_score, a.design_score).correlation
    r_m=spearmanr(m.DMS_score, m.design_score).correlation
    r_a=spearmanr(a.DMS_score, a.design_score).correlation
    print(f"{b[:-4]:<38}{len(m):>6}{r_pair:>19.4f}{r_m:>14.4f}{r_a:>16.4f}")
print("\n判据: rho(mine,anchor) 应 >0.95(仅差解码序随机);rho_DMS 两侧应在 per-assay sigma(~0.018)内")
PY
echo "P1_SMOKE_DONE"
