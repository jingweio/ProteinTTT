#!/bin/bash
set -uo pipefail
source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate bindinggym-zs-mpnn
cd /home/guoj0f/repos/ProteinTTT/bindingGYM-binding-sites-analysis
echo "[synced_commit] $(head -1 .synced_commit)"
LAMS="0.001 0.003 0.01 0.03 0.1 0.3 1.0 3.0"
A3="KRAS_RAF1-RBD_norfitness_6VJJ PSD95_CRIPT_1BE9 GB1_IgG-Fc_fitness_1FCC"

# (a) ACE2's best lambda WAS the smallest one tested and the curve was still rising there,
#     so its apparent failure may be an unconverged sweep rather than a real one.
python -u scripts/mutation_landscape_ttt/e1_lambda_sweep.py \
  --assays ACE2_SARS2-RBD_enrich_6M17 --lams $LAMS --steps 300 --sweep_M 1 --tag e1c_lamext

# (b) the matched real arm: same grid, same sweep_M, so it is comparable to the null
python -u scripts/mutation_landscape_ttt/e1_lambda_sweep.py \
  --assays $A3 --lams $LAMS --steps 300 --sweep_M 1 --tag e3real

# (c) E-3: permuted-d null. Without it a gain cannot be attributed to the interface prior
#     rather than to TTT perturbing the model at all.
python -u scripts/mutation_landscape_ttt/e1_lambda_sweep.py \
  --assays $A3 --lams $LAMS --steps 300 --sweep_M 1 --permute_d --tag e3real
echo "E3_DONE rc=$?"
