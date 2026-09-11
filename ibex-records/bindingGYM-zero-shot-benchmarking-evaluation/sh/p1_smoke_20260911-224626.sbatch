#!/bin/bash
#SBATCH --job-name=bgym_p1_smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/p1_smoke_20260911-224626_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/p1_smoke_20260911-224626_%j.err
set -uo pipefail
FAIL=0
CB=/ibex/user/guoj0f/anaconda3
source $CB/etc/profile.d/conda.sh
conda activate ligandmpnn-bgym

SH=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/sh
BG=/ibex/user/guoj0f/share/BindingGYM
MP=/ibex/user/guoj0f/share/model_params/LigandMPNN
MODELS=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/models
OUT=/ibex/user/guoj0f/bindingGYM-zs-benchmark/scores/smoke_proteinmpnn_ar
mkdir -p $OUT

echo "=== §4 CUDA 经验验证 ==="
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n,n;a=torch.randn(4096,4096,device='cuda');print('matmul ok:',float((a@a).sum()))" || FAIL=1
md5sum $MP/proteinmpnn_v_48_020.pt

echo; echo "=== (1) use_sequence 语义判别探针 ==="
python $SH/probe_use_sequence.py --repo $MODELS/LigandMPNN \
  --ckpt $MP/proteinmpnn_v_48_020.pt \
  --pdb $BG/input/structures/2M5A.pdb --idx 20 || { echo "PROBE_FAILED"; FAIL=1; }

echo; echo "=== (2)+(3) 三个小 assay:ProteinMPNN AR, seed=1 M=5 ==="
IDXS=$(python $SH/smoke_indices.py $BG/input/BindingGYM.csv \
        Z-domain_ZpA963_HL2_fitness_2M5A BH3_Mcl-1_normed_3KZ0 PSD95_CRIPT_1BE9)
echo "assay indices: $IDXS"
for i in $IDXS; do
  T0=$(date +%s)
  python $SH/score_bgym_mpnn.py --repo $MODELS/LigandMPNN \
    --dms_mapping $BG/input/BindingGYM.csv \
    --dms_input $BG/input/Binding_substitutions_DMS \
    --structure_folder $BG/input/structures \
    --dms_index $i --dms_output $OUT \
    --checkpoint $MP/proteinmpnn_v_48_020.pt \
    --model_type protein_mpnn --protocol ar \
    --num_seq_per_target 5 --seed 1 --run_id smoke_proteinmpnn_ar || FAIL=1
  echo "  idx=$i wall=$(($(date +%s)-T0))s"
done

echo; echo "=== 正确性 gate ==="
python $SH/compare_to_anchor.py $OUT /ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/refs/anchor_scores 0.95 || FAIL=1

echo "P1_SMOKE_DONE FAIL=$FAIL"
exit $FAIL
