#!/bin/bash
#SBATCH --job-name=e1_lambda_scan
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/e1_lambda_scan_20260915-003505_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/e1_lambda_scan_20260915-003505_%j.err

set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate bgym-official
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT
echo "[synced_commit] $(head -1 .synced_commit)"
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n,n"
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/scripts/structure_encoder_ttt

# E-1: anchor is now on the score. Sweep lambda; the old representation-anchor scale
# does not carry over. step-0 anchor must read ~0 in every arm (see tracking 3.1/2b).
for LAM in 0.1 1 10 100; do
  echo "================ lambda = $LAM ================"
  python -u e1_encoder_ttt.py \
    --assays 5A12_Ang2_fitness_4ZFG PSD95_CRIPT_1BE9 CD19_FMC63_Fitness_7URV \
    --lr 1e-4 --steps 150 --lam $LAM --head_steps 300 \
    --anchor_batch 32 --anchor_M 1 \
    --M 5 --seed 1 --batch 64 --probe_qa --tag e1_lam$LAM
done
echo "EXIT=$?"
