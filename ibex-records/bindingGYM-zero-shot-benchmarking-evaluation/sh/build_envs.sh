#!/bin/bash
# 建三个新 env(ligandmpnn-bgym / adflip-bgym / lasermpnn-bgym)。
# §1b-0:env 必须先于任何项目代码存在。§4:torch wheel 不得跨 CUDA major(节点驱动为 12.x)。
set -uo pipefail
CB=/ibex/user/guoj0f/anaconda3
source $CB/etc/profile.d/conda.sh

ok(){ echo "[OK]  $*"; }; bad(){ echo "[FAIL] $*"; }

build_ligandmpnn(){
  local E=ligandmpnn-bgym
  conda env list | grep -q "^$E " && { ok "$E 已存在,跳过"; return 0; }
  conda create -y -n $E python=3.11 || return 1
  conda activate $E
  # LigandMPNN requirements.txt 钉 torch==2.2.1(cu121);pandas 是我们 harness 需要的
  pip install -q torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121 || return 1
  pip install -q numpy==1.23.5 scipy==1.12.0 ProDy==2.4.1 biopython==1.79 \
                 ml-collections==0.1.1 dm-tree==0.1.8 pandas || return 1
  conda deactivate; ok "$E built"
}

build_adflip(){
  local E=adflip-bgym
  conda env list | grep -q "^$E " && { ok "$E 已存在,跳过"; return 0; }
  conda create -y -n $E python=3.10 || return 1
  conda activate $E
  pip install -q torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121 || return 1
  pip install -q torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html || return 1
  pip install -q numpy==1.24.0 pandas==2.2.3 prody==2.6.1 scipy==1.14.1 biopython==1.84 \
                 pyyaml==6.0.2 tqdm==4.66.5 ema-pytorch==0.4.8 torch_geometric \
                 sortedcontainers==2.4.0 hydra-core omegaconf || return 1
  conda deactivate; ok "$E built"
}

build_lasermpnn(){
  local E=lasermpnn-bgym
  conda env list | grep -q "^$E " && { ok "$E 已存在,跳过"; return 0; }
  # 不用 repo 的 conda yml:它同时拉 conda-forge+defaults+pyg+pytorch 四个源,solver 行为不可控。
  # 改为 pip 装 torch + 与之精确匹配的 pyg wheel。
  conda create -y -n $E python=3.11 || return 1
  conda activate $E
  pip install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 || return 1
  pip install -q torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html || return 1
  pip install -q numpy scipy pandas ProDy rdkit scikit-learn h5py matplotlib tqdm pdbecif || return 1
  conda deactivate; ok "$E built"
}

verify(){
  local E=$1
  conda activate $E 2>/dev/null || { bad "$E 无法激活"; return 1; }
  python - <<'PY'
import sys, torch
print(f"  python {sys.version.split()[0]}  torch {torch.__version__}  cuda_build {torch.version.cuda}")
try:
    import numpy; print(f"  numpy {numpy.__version__}")
except Exception as e: print("  numpy MISSING", e)
PY
  conda deactivate
}

echo "########## build ##########"
build_ligandmpnn || bad ligandmpnn-bgym
build_adflip     || bad adflip-bgym
build_lasermpnn  || bad lasermpnn-bgym
echo "########## verify ##########"
for e in ligandmpnn-bgym adflip-bgym lasermpnn-bgym; do echo "--- $e ---"; verify $e; done
echo "BUILD_ENVS_DONE"
