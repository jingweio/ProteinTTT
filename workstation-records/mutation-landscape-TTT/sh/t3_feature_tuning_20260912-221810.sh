#!/bin/bash
set -euo pipefail
source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate bindinggym-zs-mpnn

cd /home/guoj0f/repos/ProteinTTT/bindingGYM-binding-sites-analysis
echo "[synced_commit] $(head -1 .synced_commit)"
python -c "import numpy,scipy,pandas;print('numpy',numpy.__version__,'scipy',scipy.__version__,'pandas',pandas.__version__)"
df -h /home | tail -1

# CPU only -- no GPU needed for experiment 1
time python scripts/mutation_landscape_ttt/t3_feature_tuning.py
echo "EXIT=$?"
