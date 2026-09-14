#!/bin/bash
#SBATCH --job-name=e1e_lr_zero
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/e1e_lr_zero_20260915-013602_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/e1e_lr_zero_20260915-013602_%j.err
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate bgym-official
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT
echo "[synced_commit] $(head -1 .synced_commit)"
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n,n"
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/scripts/structure_encoder_ttt
A="5A12_Ang2_fitness_4ZFG PSD95_CRIPT_1BE9 CD19_FMC63_Fitness_7URV"
B="--steps 150 --lam 100 --head_steps 300 --anchor_batch 32 --anchor_M 1 --M 5 --seed 1 --batch 64 --weight_decay 0"

# arm1 kept probe and decay off yet still moved rho +0.0108, so neither explains it. The
# remaining suspect is Adam itself: the anchor is 3.5e-11 at step 0, and Adam divides out the
# gradient magnitude, so a consistent direction in numerical noise still yields lr-sized steps.
echo "======== lr = 0: the optimizer cannot move anything. delta MUST be 0 ========"
python -u e1_encoder_ttt.py --assays $A $B --lr 0    --probe_weight 0 --tag e1e_lr0
echo "======== lr = 1e-6: two orders down; if Adam normalises, the gain should persist ========"
python -u e1_encoder_ttt.py --assays $A $B --lr 1e-6 --probe_weight 0 --tag e1e_lr1e6
echo "EXIT=$?"
