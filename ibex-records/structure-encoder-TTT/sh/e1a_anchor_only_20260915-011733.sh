#!/bin/bash
#SBATCH --job-name=e1a_anchor_only
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/e1a_anchor_only_20260915-011733_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/e1a_anchor_only_20260915-011733_%j.err
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate bgym-official
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT
echo "[synced_commit] $(head -1 .synced_commit)"
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n,n"
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/scripts/structure_encoder_ttt

# E-1a: the harshest control. The permutation null already took 88% of the gain, so ask
# whether ANY probe signal is needed: set probe_weight=0, leaving only the score anchor.
# If rho still rises ~+0.01, the gain comes from training-plus-anchor alone and has nothing
# to do with supervision of any kind. Expected under the honest hypothesis: delta = 0.
python -u e1_encoder_ttt.py \
  --assays 5A12_Ang2_fitness_4ZFG PSD95_CRIPT_1BE9 CD19_FMC63_Fitness_7URV \
  --lr 1e-4 --steps 150 --lam 100 --head_steps 300 --probe_weight 0 \
  --anchor_batch 32 --anchor_M 1 --M 5 --seed 1 --batch 64 --tag e1a_anchoronly
echo "EXIT=$?"
