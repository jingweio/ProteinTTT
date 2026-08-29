#!/bin/bash
set -euo pipefail
source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate pgym-binding-partner-mpnn

REPO=/home/guoj0f/repos/ProteinTTT/proteinTTT-proteinGYM-reproduce
PROJ=$REPO/workstation-records/ProteinGYM-bindingAssay-with-partener-and-complexStructure
DATA=/data/guoj0f/ProteinGYM-bindingAssay-with-partener-and-complexStructure
DS=$DATA/dataset
SC=$DATA/scores

cd $REPO
echo "[synced_commit] $(cat .synced_commit 2>/dev/null | head -1)"
python -c "import torch; n=torch.cuda.get_device_name(0); print('GPU:', n); assert 'A100' in n, n"
python -c "import torch,numpy,pandas,scipy; print('VERSIONS', torch.__version__, numpy.__version__, pandas.__version__, scipy.__version__)"
md5sum /data/guoj0f/share/proteinmpnn/v_48_020.pt
df -h /home /data | tail -2

export MPNN_UTILS_DIR=$PROJ/sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# read-only input roots; the original ProteinGym / BindingGYM trees are never written to
export PG_REF=$PROJ/refs/DMS_substitutions.csv
export PG_DIR=/data/guoj0f/share/ProteinGym/DMS_ProteinGym_substitutions
export BG_IDX=/data/guoj0f/share/BindingGYM/input/BindingGYM.csv
export BG_DIR=/data/guoj0f/share/BindingGYM/input/Binding_substitutions_DMS
export BG_PDB=/data/guoj0f/share/BindingGYM/input/structures

echo "=== STAGE 1: build the temporary dataset ==="
mkdir -p $DS $SC
python $PROJ/sh/build_pg_binding_complex.py --out $DS
echo "STAGE1_DONE"

echo "=== STAGE 2: ProteinMPNN zero-shot, complex vs monomer ==="
ASSAYS="B2L11_HUMAN_Dutta_2010_binding-Mcl-1 DLG4_RAT_McLaughlin_2012 ACE2_HUMAN_Chan_2020 \
Q53Z42_HUMAN_McShan_2019_binding-TAPBPR CD19_HUMAN_Klesmith_2019_FMC_singles \
SPIKE_SARS2_Starr_2020_binding YAP1_HUMAN_Araya_2012 SPG1_STRSG_Wu_2016 SPG1_STRSG_Olson_2014"
for A in $ASSAYS; do
  for C in complex monomer; do
    if [ -s "$SC/${A}__${C}.csv" ]; then echo "[skip] $A/$C"; continue; fi
    echo "--- $A / $C ---"
    python $PROJ/sh/score_mpnn_complex.py --dataset $DS --assay "$A" --condition $C \
      --checkpoint /data/guoj0f/share/proteinmpnn/v_48_020.pt --out $SC --seed 1
  done
done
echo "STAGE2_DONE"

echo "=== STAGE 3: aggregate vs ProteinGym official baselines ==="
python $PROJ/sh/aggregate_results.py --scores $SC \
  --official $PROJ/refs/DMS_substitutions_Spearman_DMS_level.csv --out $PROJ/results
echo "ALL_DONE"
