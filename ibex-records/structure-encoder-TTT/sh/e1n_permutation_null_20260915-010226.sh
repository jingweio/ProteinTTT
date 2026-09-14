#!/bin/bash
#SBATCH --job-name=e1n_perm_null
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/e1n_permutation_null_20260915-010226_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/e1n_permutation_null_20260915-010226_%j.err
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate bgym-official
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT
echo "[synced_commit] $(head -1 .synced_commit)"
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n,n"
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/scripts/structure_encoder_ttt

# E-1n: permutation null at E-1's best setting (lambda=100). Interface labels are shuffled
# among VALID slots only, so pi and w_pos are preserved and the only thing destroyed is which
# residues are the interface. If the null buys the same +0.01, the gain is not the prior.
for PS in 0 1 2; do
  echo "================ permute seed = $PS ================"
  python -u e1_encoder_ttt.py \
    --assays 5A12_Ang2_fitness_4ZFG PSD95_CRIPT_1BE9 CD19_FMC63_Fitness_7URV \
    --lr 1e-4 --steps 150 --lam 100 --head_steps 300 \
    --anchor_batch 32 --anchor_M 1 --permute_labels $PS \
    --M 5 --seed 1 --batch 64 --tag e1n_perm$PS
done
echo "EXIT=$?"
