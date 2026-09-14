#!/bin/bash
#SBATCH --job-name=t0_noop_gate
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/t0_noop_gate_20260914-155239_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/structure-encoder-TTT/t0_noop_gate_20260914-155239_%j.err

set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate bgym-official

cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT
echo "[synced_commit] $(head -1 .synced_commit)"
echo "[dirty]         $(tail -n +2 .synced_commit | head -5)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# torch 1.13.1+cu117 on this node's driver: prove a real kernel launch, not just a version string
python -c "import torch,sys; n=torch.cuda.get_device_name(0); print('GPU:',n,'| torch',torch.__version__,'| cuda',torch.version.cuda); assert torch.cuda.is_available(); assert 'A100' in n, n"

cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/scripts/structure_encoder_ttt
# T0: steps=0 must leave every score bit-identical, and the baseline rho must match
# the values recorded in refs/g1e_canonical14.csv (env + data + code path all verified at once)
python -u e1_encoder_ttt.py \
  --assays 5A12_Ang2_fitness_4ZFG PSD95_CRIPT_1BE9 CD19_FMC63_Fitness_7URV \
  --steps 0 --head_steps 50 --M 5 --seed 1 --batch 64 \
  --tag t0_noop
echo "EXIT=$?"
