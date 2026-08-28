#!/bin/bash
# S0 -- harness validation gate.  No TTT: score all 217 ProteinGym DMS substitution
# assays with plain ESM2 (35M and 650M) and compare per-assay Spearman against
# ProteinGym's published leaderboard values.  S1 (TTT) does not run unless this passes.
set -euo pipefail
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
OUT=/data/guoj0f/proteinTTT-repro/scores
DMS=/data/guoj0f/share/ProteinGym/DMS_ProteinGym_substitutions
REF="$REPO/workstation-records/proteinTTT-repro/refs/DMS_substitutions.csv"
df -h /home /data | tail -2

for M in esm2_t12_35M_UR50D esm2_t33_650M_UR50D; do
  echo "================ BASELINE $M ================"
  python scripts/eval_proteingym.py \
    --model "$M" --mode baseline \
    --dms_reference "$REF" --dms_dir "$DMS" --out_dir "$OUT" \
    --score_batch_size 16
done

echo "================ AGGREGATE + VALIDATE ================"
python scripts/aggregate_proteingym.py \
  --score_dir "$OUT/esm2_t12_35M_UR50D__baseline" --column score_pre_ttt \
  --dms_reference "$REF" \
  --published "$REPO/workstation-records/proteinTTT-repro/refs/DMS_substitutions_Spearman_DMS_level.csv" \
  --published_column "ESM2 (35M)" \
  --out_json "$OUT/agg_esm2_35M_baseline.json"
python scripts/aggregate_proteingym.py \
  --score_dir "$OUT/esm2_t33_650M_UR50D__baseline" --column score_pre_ttt \
  --dms_reference "$REF" \
  --published "$REPO/workstation-records/proteinTTT-repro/refs/DMS_substitutions_Spearman_DMS_level.csv" \
  --published_column "ESM2 (650M)" \
  --out_json "$OUT/agg_esm2_650M_baseline.json"
echo "S0_DONE"
