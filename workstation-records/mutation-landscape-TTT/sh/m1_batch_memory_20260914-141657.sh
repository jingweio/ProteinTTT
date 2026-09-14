#!/bin/bash
set -uo pipefail
source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate bindinggym-zs-mpnn
cd /home/guoj0f/repos/ProteinTTT/bindingGYM-binding-sites-analysis
echo "[synced_commit] $(head -1 .synced_commit)"
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n"
time python scripts/mutation_landscape_ttt/m1_batch_memory.py \
  --assays PSD95_CRIPT_1BE9 CD19_FMC63_Fitness_7URV 5A12_Ang2_fitness_4ZFG ACE2_SARS2-RBD_enrich_6M17 \
  --bs 26 64 128 256 384 512 768 --steps 3
echo "M1_DONE rc=$?"
