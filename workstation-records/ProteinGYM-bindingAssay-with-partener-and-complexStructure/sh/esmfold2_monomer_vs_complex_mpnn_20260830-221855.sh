#!/bin/bash
set -euo pipefail
REPO=/home/guoj0f/repos/ProteinTTT/proteinTTT-proteinGYM-reproduce
PROJ=$REPO/workstation-records/ProteinGYM-bindingAssay-with-partener-and-complexStructure
DATA=/data/guoj0f/ProteinGYM-bindingAssay-with-partener-and-complexStructure
MANI=$DATA/manifests
PRED=/data/guoj0f/share/our-predicted-structure/ProteinTTT-proteinTTT-proteinGYM-reproduce/ProteinGYM-bindingAssay-with-partener-and-complexStructure
SC=$DATA/scores_esmfold
LOGD=$DATA/esmfold_logs

cd $REPO
echo "[synced_commit] $(cat .synced_commit 2>/dev/null | head -1)"
nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv,noheader
df -h /data | tail -1
mkdir -p $MANI $SC $LOGD

# ---------- STAGE 1: manifests (CPU) ----------
source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate pgym-binding-partner-mpnn
python $PROJ/sh/build_manifests.py --fasta $PROJ/refs/partner_sequences.fasta --out $MANI
echo "STAGE1_DONE"

# ---------- STAGE 2/3: ESMFold2-Fast (esmfold2 env) ----------
conda deactivate; conda activate esmfold2
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n"
export HF_HOME=/data/guoj0f/share/hf_cache
export HF_HUB_OFFLINE=1
export ESMCFOLD_CCD_PATH=/data/guoj0f/share/hf_cache/ccd.pkl

echo "=== STAGE 2: ESMFold2-Fast monomer ==="
python $PROJ/sh/esmfold2_predict.py --manifest $MANI/monomer.csv \
  --out-dir $PRED/ProteinGym-esmfold2-fast-predicted-monomer-structure \
  --log-dir $LOGD --num-loops 10 --num-sampling-steps 68 --seed 1 --kernel-backend fused
echo "STAGE2_DONE"

echo "=== STAGE 3: ESMFold2-Fast complex (target + full partner) ==="
python $PROJ/sh/esmfold2_predict.py --manifest $MANI/complex.csv \
  --out-dir $PRED/ProteinGym-esmfold2-fast-predicted-wt-complex-structure \
  --log-dir $LOGD --num-loops 10 --num-sampling-steps 68 --seed 1 --kernel-backend fused
echo "STAGE3_DONE"

# ---------- STAGE 4/5: ProteinMPNN zero-shot ----------
conda deactivate; conda activate pgym-binding-partner-mpnn
export MPNN_UTILS_DIR=$PROJ/sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DMS=/data/guoj0f/share/ProteinGym/DMS_ProteinGym_substitutions
ASSAYS="B2L11_HUMAN_Dutta_2010_binding-Mcl-1 DLG4_RAT_McLaughlin_2012 ACE2_HUMAN_Chan_2020 \
Q53Z42_HUMAN_McShan_2019_binding-TAPBPR CD19_HUMAN_Klesmith_2019_FMC_singles \
SPIKE_SARS2_Starr_2020_binding YAP1_HUMAN_Araya_2012 SPG1_STRSG_Wu_2016 SPG1_STRSG_Olson_2014"
for A in $ASSAYS; do
  for C in monomer complex; do
    [ "$C" = monomer ] && D=$PRED/ProteinGym-esmfold2-fast-predicted-monomer-structure \
                       || D=$PRED/ProteinGym-esmfold2-fast-predicted-wt-complex-structure
    [ -s "$SC/${A}__esmfold_${C}.csv" ] && { echo "[skip] $A/$C"; continue; }
    [ -s "$D/${A}.pdb" ] || { echo "[missing structure] $A/$C -- skipped"; continue; }
    echo "--- MPNN $A / esmfold_$C ---"
    python $PROJ/sh/score_mpnn_predicted.py --pdb $D/${A}.pdb --assay "$A" \
      --dms-dir $DMS --dms-file "${A}.csv" --condition "esmfold_${C}" \
      --checkpoint /data/guoj0f/share/proteinmpnn/v_48_020.pt --out $SC --seed 1
  done
done
echo "STAGE45_DONE"

# ---------- STAGE 6: aggregate ----------
python $PROJ/sh/aggregate_esmfold.py --scores $SC \
  --official $PROJ/refs/DMS_substitutions_Spearman_DMS_level.csv \
  --crystal-scores $DATA/scores --out $PROJ/results
echo "ALL_DONE"
