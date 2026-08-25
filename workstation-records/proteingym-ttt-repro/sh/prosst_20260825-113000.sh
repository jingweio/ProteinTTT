#!/bin/bash
# ProSST (K=2048) on ProteinGym DMS substitutions -- baseline and + ProteinTTT.
#   usage: prosst_<dt>.sh baseline            # ~minutes: one forward pass per assay
#          prosst_<dt>.sh ttt "0 1 2 3 4"
set -uo pipefail
MODE="${1:?baseline|ttt}"
SEEDS="${2:-0}"
shift 1; [ $# -gt 0 ] && shift 1 || true

source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate proteingym-ttt
REPO=/home/guoj0f/repos/ProteinTTT/proteinTTT-proteinGYM-reproduce
cd "$REPO"
echo "[synced_commit] $(head -1 .synced_commit 2>/dev/null)"
python -c "import torch; n=torch.cuda.get_device_name(0); print('GPU:', n); assert 'A100' in n, n"

export PYTHONPATH="$REPO"
export HF_HOME=/data/guoj0f/share/hf_cache
export TORCH_HOME=/data/guoj0f/share/torch_hub
OUT=/data/guoj0f/proteingym-ttt-repro/scores
REF="$REPO/workstation-records/proteingym-ttt-repro/refs/DMS_substitutions.csv"
COMMON=(--dms_reference "$REF"
        --dms_dir /data/guoj0f/share/ProteinGym/DMS_ProteinGym_substitutions
        --prosst_dir /data/guoj0f/share/ProSST
        --out_dir "$OUT")
df -h /home /data | tail -2

if [ "$MODE" = "baseline" ]; then
  python scripts/eval_proteingym_prosst.py --mode baseline "${COMMON[@]}" "$@"
else
  for S in $SEEDS; do
    echo "================ ProSST TTT seed $S  ($(date '+%F %T')) ================"
    python scripts/eval_proteingym_prosst.py --mode ttt --seed "$S" "${COMMON[@]}" "$@"
  done
fi
echo "PROSST_DONE $MODE seeds=$SEEDS"
