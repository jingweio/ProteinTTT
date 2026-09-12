#!/bin/bash
#SBATCH --job-name=bgym_p1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/p1_run_%x_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/p1_run_%x_%j.err
# 用法: sbatch --job-name=<run_id> --export=ALL,RUN_ID=...,MODEL_TYPE=...,CKPT=...,SC=...,PROTO=... p1_run.sbatch
# 幂等:每 assay 一个输出 csv,已存在即跳过 ⇒ 被 TIMEOUT 截断后重投即续跑。
# walltime 只要 2h —— 我方 FairShare 仅 0.0997,长 walltime 会被排到 20h 以后。
set -uo pipefail
CB=/ibex/user/guoj0f/anaconda3
source $CB/etc/profile.d/conda.sh
conda activate ligandmpnn-bgym

SH=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/sh
BG=/ibex/user/guoj0f/share/BindingGYM
MP=/ibex/user/guoj0f/share/model_params/LigandMPNN
MODELS=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/models
OUT=/ibex/user/guoj0f/bindingGYM-zs-benchmark/scores/${RUN_ID}
mkdir -p $OUT

python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n,n" || exit 1
echo "run_id=${RUN_ID} model=${MODEL_TYPE} ckpt=$(basename ${CKPT}) sc=${SC} protocol=${PROTO}"
md5sum ${CKPT}

N=$(python -c "import pandas as pd;print(len(pd.read_csv('$BG/input/BindingGYM.csv')))")
FAIL=0
for i in $(seq 0 $((N-1))); do
  python $SH/score_bgym_mpnn.py --repo $MODELS/LigandMPNN \
    --dms_mapping $BG/input/BindingGYM.csv \
    --dms_input $BG/input/Binding_substitutions_DMS \
    --structure_folder $BG/input/structures \
    --dms_index $i --dms_output $OUT \
    --checkpoint ${CKPT} --model_type ${MODEL_TYPE} \
    --use_side_chain_context ${SC} --protocol ${PROTO} \
    --designed_chains ${DCH:-index} --k_neighbors ${K:-0} \
    --num_seq_per_target 5 --seed 1 --run_id ${RUN_ID} || FAIL=1
done
echo "scored: $(ls $OUT/*.csv 2>/dev/null | wc -l) / $N"
echo "P1_RUN_DONE run_id=${RUN_ID} FAIL=$FAIL"
exit $FAIL
