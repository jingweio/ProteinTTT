#!/bin/bash
set -uo pipefail
source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate bindinggym-zs-mpnn
cd /home/guoj0f/repos/ProteinTTT/bindingGYM-binding-sites-analysis
echo "[synced_commit] $(head -1 .synced_commit)"
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n"
time python scripts/mutation_landscape_ttt/e1_lambda_sweep.py \
  --assays GB1_IgG-Fc_fitness_1FCC KRAS_RAF1-RBD_norfitness_6VJJ PSD95_CRIPT_1BE9 \
  --agg sum --lams 0.001 0.003 0.01 0.03 0.1 0.3 1 3 \
  --steps 3000 --eval_at 300 1000 2000 3000 --tag e4_sum_steps
echo "E4_DONE rc=$?"
