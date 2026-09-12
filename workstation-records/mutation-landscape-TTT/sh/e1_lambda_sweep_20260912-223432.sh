#!/bin/bash
set -uo pipefail
source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate bindinggym-zs-mpnn
cd /home/guoj0f/repos/ProteinTTT/bindingGYM-binding-sites-analysis
echo "[synced_commit] $(head -1 .synced_commit)"
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n"
ASSAYS="PSD95_CRIPT_1BE9 KRAS_RAF1-RBD_norfitness_6VJJ GB1_IgG-Fc_fitness_1FCC ACE2_SARS2-RBD_enrich_6M17"
time python scripts/mutation_landscape_ttt/e1_lambda_sweep.py --assays $ASSAYS --steps 300 --tag e1
echo "E1_DONE rc=$?"
