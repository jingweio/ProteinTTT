#!/bin/bash
#SBATCH --job-name=p1_lambda_scan
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/p1_lambda_scan_20260914-160145_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/p1_lambda_scan_20260914-160145_%j.err

set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate bgym-official
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT
echo "[synced_commit] $(head -1 .synced_commit)"
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n,n"
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/scripts/structure_encoder_ttt

# P1: fix lr and steps, sweep lambda. Order swapped from the original plan because the local
# smoke showed lambda=1 barely restrains the objective (probe loss fell two orders of
# magnitude while the anchor stayed at 1e-3) -- picking lr/steps under an ineffective anchor
# would just select whichever config trains least.
for LAM in 1 10 100 1000; do
  echo "================ lambda = $LAM ================"
  python -u e1_encoder_ttt.py \
    --assays 5A12_Ang2_fitness_4ZFG PSD95_CRIPT_1BE9 CD19_FMC63_Fitness_7URV \
    --lr 1e-4 --steps 150 --lam $LAM --head_steps 300 \
    --M 5 --seed 1 --batch 64 --tag p1_lam$LAM
done
echo "EXIT=$?"
