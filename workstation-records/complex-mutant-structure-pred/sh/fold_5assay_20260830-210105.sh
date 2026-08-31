#!/bin/bash
set -euo pipefail
source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate esmfold2                                   # SOP §2.3 共享 env

export HF_HOME=/data/guoj0f/share/hf_cache
export ESMCFOLD_CCD_PATH="$HF_HOME/ccd.pkl"
export HF_HUB_OFFLINE=1

REPO=/home/guoj0f/repos/ProteinTTT/bindingGYM-mutation-structure-analysis
PROJ=$REPO/workstation-records/complex-mutant-structure-pred
OUT=/data/guoj0f/share/our-predicted-structure/ProteinTTT-bindingGYM-mutation-structure-analysis/complex-mutant-structure-pred

echo "[synced_commit] $(head -1 $REPO/.synced_commit 2>/dev/null)"
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n"
df -h /data | tail -1

python $PROJ/sh/fold_complexes.py \
  --manifest $PROJ/refs/fold_manifest.csv.gz \
  --out-root $OUT \
  --log-dir $PROJ/results/fold_logs \
  --num-loops 10 --num-sampling-steps 68 --num-diffusion-samples 1 \
  --kernel-backend fused
