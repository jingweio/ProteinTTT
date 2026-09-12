#!/bin/bash
#SBATCH --job-name=bgym_p4_stabddg
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/p4_stabddg_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/p4_stabddg_%j.err
# StaB-ddG **stage 2**(Megascale 稳定性 finetune)在 BindingGYM 上的 zero-shot binding ddG。
# 🔴 绝不能用 stage3(stabddg.pt) —— 它在 SKEMPI 上 finetune 过,与 BindingGYM 复合物重叠 = leakage。
# 幂等:每 assay 一个输出目录,已有结果即跳过。
set -uo pipefail
CB=/ibex/user/guoj0f/anaconda3
source $CB/etc/profile.d/conda.sh
conda activate stabddg

SH=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/sh
BG=/ibex/user/guoj0f/share/BindingGYM
SD=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/models/StaB-ddG
export STABDDG_REPO=$SD
export PYTHONPATH=$SD:${PYTHONPATH:-}
OUT=/ibex/user/guoj0f/bindingGYM-zs-benchmark/stabddg_s2
mkdir -p $OUT

python -c "import torch;n=torch.cuda.get_device_name(0);print('GPU:',n);assert 'A100' in n,n" || exit 1

CKPT=$SD/model_ckpts/stability_finetuned.pt
echo "=== 🔴 stage 守卫:必须是 stage2 (Megascale),不是 stage3 (SKEMPI) ==="
python - <<'PY'
import hashlib, os, sys
p = os.environ["STABDDG_REPO"] + "/model_ckpts/stability_finetuned.pt"
h = hashlib.md5(open(p,'rb').read()).hexdigest()
print("ckpt md5:", h)
assert h == "ed2645a2dc380932e50bfafff0fd5a36", f"不是 stage2 的 stability_finetuned.pt: {h}"
bad = os.environ["STABDDG_REPO"] + "/model_ckpts/stabddg.pt"
if os.path.exists(bad):
    print("注意:仓库里仍有 stage3 的 stabddg.pt(md5 b4dc0067...),本脚本不会用它")
PY

N=25; FAIL=0
for i in $(seq 0 $((N-1))); do
  ID=$(python -c "import pandas as pd;print(pd.read_csv('$BG/input/BindingGYM.csv').DMS_id[$i])")
  if [ -s "$OUT/$ID/output/output.csv" ]; then echo "[skip] $ID"; continue; fi
  python $SH/prep_stabddg_inputs.py \
    --dms_mapping $BG/input/BindingGYM.csv --dms_input $BG/input/Binding_substitutions_DMS \
    --structure_folder $BG/input/structures \
    --partition_tsv /ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/refs/chain_partition.tsv \
    --dms_index $i --out_root $OUT || { FAIL=1; continue; }
  T0=$(date +%s)
  python $SD/run_stabddg.py --checkpoint $CKPT \
    --csv_path $OUT/$ID/muts.csv --pdb_dir $OUT/$ID/pdbs \
    --output_dir $OUT/$ID/output --mc_samples 20 --batch_size 10000 --seed 1 \
    || { echo "  [fail] $ID"; FAIL=1; continue; }
  echo "  [done] $ID wall=$(($(date +%s)-T0))s"
done
echo "done assays: $(ls -d $OUT/*/output 2>/dev/null | wc -l) / $N"
echo "P4_DONE FAIL=$FAIL"
exit $FAIL
