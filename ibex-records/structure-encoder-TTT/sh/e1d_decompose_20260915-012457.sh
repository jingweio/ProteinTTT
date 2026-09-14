#!/bin/bash
#SBATCH --job-name=e1d_decompose
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/e1d_decompose_20260915-012457_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/e1d_decompose_20260915-012457_%j.err
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate bgym-official
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT
echo "[synced_commit] $(head -1 .synced_commit)"
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n,n"
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/scripts/structure_encoder_ttt
A="5A12_Ang2_fitness_4ZFG PSD95_CRIPT_1BE9 CD19_FMC63_Fitness_7URV"
C="--lr 1e-4 --steps 150 --lam 100 --head_steps 300 --anchor_batch 32 --anchor_M 1 --M 5 --seed 1 --batch 64"

# Decompose the +0.0118. Arm 2 already ran as job 51897820 (+0.0115) under the accidental decay.
echo "======== arm1: nothing at all (probe off, decay off) -> must be ~0 ========"
python -u e1_encoder_ttt.py --assays $A $C --probe_weight 0 --weight_decay 0    --tag e1d_none
echo "======== arm3: the method as documented (probe on, decay off) ========"
python -u e1_encoder_ttt.py --assays $A $C --probe_weight 1 --weight_decay 0    --tag e1d_method
echo "======== arm4: decay only, confirming arm2 with the switch explicit ========"
python -u e1_encoder_ttt.py --assays $A $C --probe_weight 0 --weight_decay 0.01 --tag e1d_decayonly
echo "EXIT=$?"
