#!/bin/bash
#SBATCH --job-name=bgym_p23
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/p23_%x_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/p23_%x_%j.err
# 用法: sbatch --job-name=<run_id> --export=ALL,WHICH=laser|adflip,RUN_ID=...,PROTO=ar|jm
# 幂等 + 1h walltime(fairshare 低,短 job 才排得上);被截断重投即续跑。
set -uo pipefail
CB=/ibex/user/guoj0f/anaconda3
source $CB/etc/profile.d/conda.sh
SH=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/sh
BG=/ibex/user/guoj0f/share/BindingGYM
MODELS=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/models
OUT=/ibex/user/guoj0f/bindingGYM-zs-benchmark/scores/${RUN_ID}
mkdir -p $OUT
N=25; FAIL=0

if [ "${WHICH}" = "laser" ]; then
  conda activate lasermpnn-bgym
  python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n,n" || exit 1
  for i in $(seq 0 $((N-1))); do
    python $SH/score_bgym_lasermpnn.py --repo $MODELS \
      --weights $MODELS/LASErMPNN/model_weights/laser_weights_0p1A_nothing_heldout.pt \
      --dms_mapping $BG/input/BindingGYM.csv --dms_input $BG/input/Binding_substitutions_DMS \
      --structure_folder $BG/input/structures --dms_index $i --dms_output $OUT \
      --protocol ${PROTO} --num_decoding_orders 5 --seed 1 --run_id ${RUN_ID} || FAIL=1
  done
else
  conda activate adflip-bgym
  cd $MODELS/ADFLIP
  python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n,n" || exit 1
  for i in $(seq 0 $((N-1))); do
    python $SH/score_bgym_adflip.py --repo $MODELS/ADFLIP \
      --ckpt $MODELS/ADFLIP/results/weights/ADFLIP_v1.pt \
      --dms_mapping $BG/input/BindingGYM.csv --dms_input $BG/input/Binding_substitutions_DMS \
      --structure_folder $BG/input/structures --dms_index $i --dms_output $OUT \
      --t 1.0 --run_id ${RUN_ID} || FAIL=1
  done
fi
echo "scored: $(ls $OUT/*.csv 2>/dev/null | wc -l) / $N"
echo "P23_DONE run_id=${RUN_ID} FAIL=$FAIL"
exit $FAIL
