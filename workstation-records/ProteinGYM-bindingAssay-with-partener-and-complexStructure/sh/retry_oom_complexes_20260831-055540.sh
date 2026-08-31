#!/bin/bash
set -euo pipefail
REPO=/home/guoj0f/repos/ProteinTTT/proteinTTT-proteinGYM-reproduce
PROJ=$REPO/workstation-records/ProteinGYM-bindingAssay-with-partener-and-complexStructure
DATA=/data/guoj0f/ProteinGYM-bindingAssay-with-partener-and-complexStructure
PRED=/data/guoj0f/share/our-predicted-structure/ProteinTTT-proteinTTT-proteinGYM-reproduce/ProteinGYM-bindingAssay-with-partener-and-complexStructure
CDIR=$PRED/ProteinGym-esmfold2-fast-predicted-wt-complex-structure
SC=$DATA/scores_esmfold

cd $REPO; echo "[synced_commit] $(cat .synced_commit | head -1)"
nvidia-smi --query-gpu=memory.free --format=csv,noheader
source /data/guoj0f/miniconda3/etc/profile.d/conda.sh

# the two L=2078 complexes that OOM'd at full chunk; retry with the trunk chunked (SOP §4.2)
conda activate esmfold2
export HF_HOME=/data/guoj0f/share/hf_cache HF_HUB_OFFLINE=1
export ESMCFOLD_CCD_PATH=/data/guoj0f/share/hf_cache/ccd.pkl
head -1 $DATA/manifests/complex.csv > $DATA/manifests/oom2.csv
grep -E '^(ACE2_HUMAN_Chan_2020|SPIKE_SARS2_Starr_2020_binding),' $DATA/manifests/complex.csv >> $DATA/manifests/oom2.csv
python $PROJ/sh/esmfold2_predict.py --manifest $DATA/manifests/oom2.csv \
  --out-dir $CDIR --log-dir $DATA/esmfold_logs \
  --num-loops 10 --num-sampling-steps 68 --seed 1 --kernel-backend fused --chunk-size 32
echo "RETRY_FOLD_DONE"

conda deactivate; conda activate pgym-binding-partner-mpnn
export MPNN_UTILS_DIR=$PROJ/sh PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for A in ACE2_HUMAN_Chan_2020 SPIKE_SARS2_Starr_2020_binding; do
  [ -s "$SC/${A}__esmfold_complex.csv" ] && { echo "[skip] $A"; continue; }
  [ -s "$CDIR/${A}.pdb" ] || { echo "[still missing] $A"; continue; }
  python $PROJ/sh/score_mpnn_predicted.py --pdb $CDIR/${A}.pdb --assay "$A" \
    --dms-dir /data/guoj0f/share/ProteinGym/DMS_ProteinGym_substitutions --dms-file "${A}.csv" \
    --condition esmfold_complex --checkpoint /data/guoj0f/share/proteinmpnn/v_48_020.pt --out $SC --seed 1
done
echo "RETRY_MPNN_DONE"

python $PROJ/sh/aggregate_esmfold.py --scores $SC \
  --official $PROJ/refs/DMS_substitutions_Spearman_DMS_level.csv \
  --crystal-scores $DATA/scores --out $PROJ/results
echo "ALL_DONE"
