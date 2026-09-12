#!/bin/bash
set -uo pipefail
source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate bindinggym-zs-mpnn
cd /home/guoj0f/repos/ProteinTTT/bindingGYM-binding-sites-analysis
echo "[synced_commit] $(head -1 .synced_commit)"
A="PSD95_CRIPT_1BE9 PSD95_Tm2F_1BE9 5A12_Ang2_fitness_4ZFG HLA-A2_TAPBPR_meanscore_5WER CD19_FMC63_Fitness_7URV"
# same TTT, two priors: the untuned max feature and the tuned sum feature from experiment 1
python -u scripts/mutation_landscape_ttt/e1_lambda_sweep.py --assays $A --steps 300 --agg max --tag e1b_max
python -u scripts/mutation_landscape_ttt/e1_lambda_sweep.py --assays $A --steps 300 --agg sum --tag e1b_sum
echo "E1B_DONE rc=$?"
