#!/bin/bash
# S1 -- ProteinTTT customization + rescoring over ProteinGym DMS substitutions.
#
#   usage: s1_ttt_esm2_<dt>.sh <esm2_t12_35M_UR50D|esm2_t33_650M_UR50D> "<seed list>" [extra args...]
#   e.g.   s1_ttt_esm2_<dt>.sh esm2_t12_35M_UR50D "0 1 2 3 4"
#
# Resumable: assays whose output CSV already exists are skipped, so the script can
# be re-launched after an interruption without redoing finished work.
set -euo pipefail
MODEL="${1:?model}"
SEEDS="${2:?seed list, e.g. \"0 1 2 3 4\"}"
shift 2

source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate proteingym-ttt

REPO=/home/guoj0f/repos/ProteinTTT/proteinTTT-proteinGYM-reproduce
cd "$REPO"
echo "[synced_commit] $(head -1 .synced_commit 2>/dev/null)"

# single A100 -- do NOT set CUDA_VISIBLE_DEVICES
python -c "import torch; n=torch.cuda.get_device_name(0); print('GPU:', n); assert 'A100' in n, n"

export PYTHONPATH="$REPO"
export TORCH_HOME=/data/guoj0f/share/torch_hub
export HF_HOME=/data/guoj0f/share/hf_cache
OUT=/data/guoj0f/proteingym-ttt-repro/scores
DMS=/data/guoj0f/share/ProteinGym/DMS_ProteinGym_substitutions
REF="$REPO/workstation-records/proteingym-ttt-repro/refs/DMS_substitutions.csv"
df -h /home /data | tail -2

for S in $SEEDS; do
  echo "================ TTT $MODEL seed $S  ($(date '+%F %T')) ================"
  python scripts/eval_proteingym.py \
    --model "$MODEL" --mode ttt --seed "$S" \
    --dms_reference "$REF" --dms_dir "$DMS" --out_dir "$OUT" \
    --score_batch_size 16 "$@"
done
echo "S1_DONE $MODEL seeds=$SEEDS"
