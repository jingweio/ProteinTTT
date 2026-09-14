#!/bin/bash
#SBATCH --job-name=g1_iface_sens
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/geometry-counterfactural-encoder-TTT/g1_interface_sensitivity_20260915-010521_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/geometry-counterfactural-encoder-TTT/g1_interface_sensitivity_20260915-010521_%j.err
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate bgym-official
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT
echo "[synced_commit] $(head -1 .synced_commit)"
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n,n"
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/scripts/geometry_counterfactual_ttt

# G-1 gate on the SAME 14 assays as structure-encoder-TTT, for a like-for-like comparison.
# The list is taken from refs/g1e_canonical14.csv, verified identical to the 14 marked in
# .../bindingGYM-binding-sites-analysis/local-records/structure-TTT/structure_ttt_encoder-proposal.md §1.3
python -u g1_interface_sensitivity.py --M 5 --seed 1 --reps 3 --tag g1
echo "EXIT=$?"
