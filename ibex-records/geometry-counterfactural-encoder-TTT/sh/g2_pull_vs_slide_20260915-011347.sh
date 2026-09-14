#!/bin/bash
#SBATCH --job-name=g2_pull_slide
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/geometry-counterfactural-encoder-TTT/g2_pull_vs_slide_20260915-011347_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/ibex-records/geometry-counterfactural-encoder-TTT/g2_pull_vs_slide_20260915-011347_%j.err
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate bgym-official
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT
echo "[synced_commit] $(head -1 .synced_commit)"
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n,n"
cd /ibex/user/guoj0f/ProteinTTT/structure-encoder-TTT/scripts/geometry_counterfactual_ttt
# G-2 on the same 14 assays (verified against proposal 1.3). Small magnitudes first: G-1 showed
# 2 A+ already gives 4.6+ sd, which is too easy a negative to learn anything from.
python -u g2_pull_vs_slide.py --M 5 --seed 1 --reps 5 --mags 0.5 1.0 2.0 4.0 --tag g2
echo "EXIT=$?"
