#!/bin/bash
#SBATCH --job-name=p1b_probe_qa
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/p1b_probe_qa_20260914-161405_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/p1b_probe_qa_20260914-161405_%j.err

set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate bgym-official
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT
echo "[synced_commit] $(head -1 .synced_commit)"
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n,n"
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/scripts/structure_encoder_ttt

# P1b: did the training actually raise the interface separability of h_V?
# P1 found no lambda that helps, but that only bears on H if the training did what it claims.
# An INDEPENDENT 5-fold probe before and after decides whether the null result is attributable
# to the hypothesis or merely to an ineffective intervention.
for LAM in 1 10 100; do
  echo "================ lambda = $LAM ================"
  python -u e1_encoder_ttt.py \
    --assays 5A12_Ang2_fitness_4ZFG PSD95_CRIPT_1BE9 CD19_FMC63_Fitness_7URV \
    --lr 1e-4 --steps 150 --lam $LAM --head_steps 300 \
    --M 5 --seed 1 --batch 64 --probe_qa --tag p1b_lam$LAM
done
echo "EXIT=$?"
