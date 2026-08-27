#!/bin/bash
# BindingGYM 官方 zero-shot pretrained ProteinMPNN —— 逐字对齐 modelzoo/proteinmpnn/run.py:47-58
#
#   usage: zeroshot_mpnn_<dt>.sh <SEED> <M> [ASSAY_IDX_CSV]
#     SEED : 必须非 0 —— compute_fitness_multi_pdb.py:26 是 `if args.seed:`，0 是 falsy 会退回随机
#     M    : --num_seq_per_target。官方 = 5。（上一次复现漏传此参数，用了 default=1，低了 0.0129）
#     ASSAY_IDX_CSV : 可选，如 "0,3,17"；省略则跑全部 25 个
#
# 幂等：每 assay 一个输出 csv，已存在则跳过 ⇒ 中断后重跑即续。
set -uo pipefail
SEED="${1:?seed (must be non-zero)}"
M="${2:?num_seq_per_target}"
IDXS="${3:-}"
[ "$SEED" = "0" ] && { echo "FATAL: seed 0 会被当成 falsy 而退回随机，请用非 0 值"; exit 1; }

source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate bindinggym-zs-mpnn

BG=/data/guoj0f/share/BindingGYM
OUT=/data/guoj0f/BindingGYM-zero-shot-proteinMPNN/scores/seed${SEED}_M${M}
mkdir -p "$OUT"
cd "$BG/baselines/protein_mpnn"

echo "[synced_commit] $(head -1 /home/guoj0f/repos/ProteinTTT/proteinTTT-proteinGYM-reproduce/.synced_commit 2>/dev/null)"
# 单卡 —— 不要设 CUDA_VISIBLE_DEVICES
python -c "import torch; n=torch.cuda.get_device_name(0); print('GPU:', n); assert 'A100' in n, n"
python - <<'PY'
import hashlib
p='/data/guoj0f/share/BindingGYM/training/cache/v_48_020.pt'
h=hashlib.md5(open(p,'rb').read()).hexdigest()
print('ckpt md5:', h)
assert h == '91d54c97a68bf551114f8c74c785e90f', f'checkpoint 不是官方 vanilla v_48_020: {h}'
PY
df -h /home /data | tail -2
echo "seed=$SEED  M=$M  out=$OUT"

mapfile -t IDS < <(python -c "
import pandas as pd; print('\n'.join(pd.read_csv('$BG/input/BindingGYM.csv')['DMS_id']))")
echo "assays in index: ${#IDS[@]}"

if [ -n "$IDXS" ]; then RANGE=$(echo "$IDXS" | tr ',' ' '); else RANGE=$(seq 0 $((${#IDS[@]}-1))); fi

for i in $RANGE; do
  ID="${IDS[$i]}"
  if [ -s "$OUT/${ID}.csv" ]; then echo "[$i] $ID -- exists, skip"; continue; fi
  T0=$(date +%s)
  python compute_fitness_multi_pdb.py \
    --model_location "$BG/training/cache/v_48_020.pt" \
    --dms_index "$i" \
    --dms_mapping "$BG/input/BindingGYM.csv" \
    --dms_input "$BG/input/Binding_substitutions_DMS" \
    --dms_output "$OUT" \
    --structure_folder "$BG/input/structures" \
    --batch_size 8 \
    --num_seq_per_target "$M" \
    --seed "$SEED" \
    --suppress_print 1
  RC=$?
  T1=$(date +%s)
  N=$(python -c "import pandas as pd;print(len(pd.read_csv('$OUT/${ID}.csv')))" 2>/dev/null || echo NA)
  echo "[$i] $ID  rc=$RC  wall=$((T1-T0))s  rows=$N"
done

echo "scored: $(ls "$OUT"/*.csv 2>/dev/null | wc -l) / ${#IDS[@]}"
echo "ZS_DONE seed=$SEED M=$M"
