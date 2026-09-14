#!/bin/bash
set -uo pipefail
source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate bindinggym-zs-mpnn
cd /home/guoj0f/repos/ProteinTTT/bindingGYM-binding-sites-analysis
echo "[synced_commit] $(head -1 .synced_commit)"
python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n"
time python scripts/mutation_landscape_ttt/m1_batch_memory.py \
  --assays 5A12_Ang2_fitness_4ZFG ACE2_SARS2-RBD_enrich_6M17 CD19_FMC63_Fitness_7URV CXCR4_CXCL12_enrich_8U4O GB1_IgG-Fc_fitness_1FCC HLA-A2_TAPBPR_meanscore_5WER KRAS_PICK3CG-RBD_norfitness_1HE8 KRAS_RAF1-RBD_norfitness_6VJJ KRAS_RAF1_norfitness_6VJJ KRAS_RALGDS-RBD_norfitness_1LFD PSD95_CRIPT_1BE9 PSD95_Tm2F_1BE9 SARS2-RBD_ACE2_deltaKd_6M0J hYAP65_peptide_FunctioncalScore_1JMQ --bs 32 64 128 256 512 --steps 2 --n_cap 700
echo "M1B_DONE rc=$?"
