#!/bin/bash
set -uo pipefail
source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate bindinggym-zs-mpnn
cd /home/guoj0f/repos/ProteinTTT/bindingGYM-binding-sites-analysis
echo "[synced_commit] $(head -1 .synced_commit)"
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n"
A="PSD95_CRIPT_1BE9 PSD95_Tm2F_1BE9 5A12_Ang2_fitness_4ZFG HLA-A2_TAPBPR_meanscore_5WER CD19_FMC63_Fitness_7URV CXCR4_CXCL12_enrich_8U4O ACE2_SARS2-RBD_enrich_6M17"
# alpha swept wide: one direction out of 128 carries ~1% of the variance, so even full
# ablation (alpha = -1) is only a ~10% edit to h_V. If a large amplification still does not
# move the ranking, that is the answer.
time python -u scripts/mutation_landscape_ttt/s0_intervention.py --assays $A \
  --alphas -8 -4 -2 -1 -0.5 0.5 1 2 4 8 --seeds 0 1 2 --tag s0
echo "S0_DONE rc=$?"
