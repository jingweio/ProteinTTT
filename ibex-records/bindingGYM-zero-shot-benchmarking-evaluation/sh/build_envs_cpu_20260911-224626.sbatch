#!/bin/bash
#SBATCH --job-name=bgym_envbuild_cpu
#SBATCH --cpus-per-task=16

#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/build_envs_cpu_20260911-224626_%j.out
#SBATCH --error=/ibex/user/guoj0f/ProteinTTT/proteinTTT-proteinGYM-reproduce/ibex-records/bindingGYM-zero-shot-benchmarking-evaluation/build_envs_cpu_20260911-224626_%j.err
# 在计算节点建 env —— login node 的 process reaper 会 Kill 掉 conda solver(实测 2026-09-11)。
# 顺带在真 a100 上做 §4 要求的 CUDA 经验验证(不靠版本号推断)。
set -uo pipefail
CB=/ibex/user/guoj0f/anaconda3
source $CB/etc/profile.d/conda.sh
conda config --set solver libmamba 2>/dev/null || true

echo "=== 节点 CUDA 预检(§4:先探驱动,wheel 不得跨 CUDA major) ==="
echo '(CPU-only job:CUDA 经验验证拆到后续 GPU job 的 preflight)'

mk(){  # mk <env> <py> <pip-cmds...>
  local E=$1 PY=$2; shift 2
  # 🔴 验包不验名:login node 的 reaper 在 pip 阶段 Kill 会留下 site-packages 为空的 env 壳,
  #    只查 env 名会把半成品当成已建好而跳过(2026-09-11 实测 lasermpnn-bgym pkgs=0)。
  if conda env list | grep -q "^$E "; then
    if $CB/envs/$E/bin/python -c "import torch" 2>/dev/null; then
      echo "[OK] $E 已存在且可用,跳过"; return 0
    fi
    echo "[WARN] $E 是半成品(torch 无法 import),移除重建"
    conda env remove -n $E -y >/dev/null 2>&1
  fi
  conda create -y -n $E python=$PY || { echo "[FAIL] $E create"; return 1; }
  conda activate $E || return 1
  local rc=0
  while [ $# -gt 0 ]; do eval "$1" || rc=1; shift; done
  conda deactivate
  [ $rc -eq 0 ] && echo "[OK] $E built" || echo "[FAIL] $E pip"
  return $rc
}

mk ligandmpnn-bgym 3.11 \
  "pip install -q torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121" \
  "pip install -q numpy==1.23.5 scipy==1.12.0 ProDy==2.4.1 biopython==1.79 ml-collections==0.1.1 dm-tree==0.1.8 pandas"

mk lasermpnn-bgym 3.11 \
  "pip install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124" \
  "pip install -q torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html" \
  "pip install -q numpy scipy pandas ProDy rdkit scikit-learn h5py matplotlib tqdm pdbecif"

echo "=== §4 经验验证:真的能在 a100 上跑 kernel 吗 ==="
for E in ligandmpnn-bgym adflip-bgym lasermpnn-bgym; do
  echo "--- $E ---"
  conda activate $E 2>/dev/null || { echo "  无法激活"; continue; }
  python - <<'PY'
import sys, torch
print(f"  py {sys.version.split()[0]}  torch {torch.__version__}  cuda_build {torch.version.cuda}")
print("  cuda_available(CPU节点上必为False,不作判据):", torch.cuda.is_available())

a = torch.randn(512, 512); print("  cpu matmul ok:", float((a@a).sum()))
import numpy; print("  numpy", numpy.__version__)
PY
  # §4:预编译 bytecode,避开 Lustre 冷 .pyc 的 import race
  SP=$(python -c "import site;print(site.getsitepackages()[0])" 2>/dev/null)
  [ -n "$SP" ] && python -m compileall -q $SP/torch >/dev/null 2>&1
  conda deactivate
done
echo "BUILD_ENVS_DONE"
