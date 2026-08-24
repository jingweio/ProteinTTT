#!/bin/bash
# Full S0-fixup + S1 queue for the ProteinGym reproduction.  Single shared A100,
# so everything is strictly serial and each stage waits for the GPU to be free.
#
# Stage 0 re-scores only the 5 assays whose MSA region is a strict sub-region of
# target_seq; the harness now truncates to that region the way ProteinGym's
# compute_fitness.py does, which is what makes the remaining per-assay Spearman
# match the published leaderboard.
set -uo pipefail
L=/data/guoj0f/proteingym-ttt-repro/logs
D=/home/guoj0f/repos/ProteinTTT/proteinTTT-proteinGYM-reproduce
S1="$D/workstation-records/proteingym-ttt-repro/sh/s1_ttt_esm2_20260824-231500.sh"
SUBREGION=A0A140D2T1_ZIKV_Sourisseau_2019,KCNH2_HUMAN_Kozek_2020,POLG_CXB3N_Mattenberger_2021,POLG_HCVJF_Qi_2014,SCN5A_HUMAN_Glazer_2019

wait_for_free () { while pgrep -f "[e]val_proteingym.py" >/dev/null; do sleep 30; done; }

source /data/guoj0f/miniconda3/etc/profile.d/conda.sh
conda activate proteingym-ttt
cd "$D"
export PYTHONPATH="$D" TORCH_HOME=/data/guoj0f/share/torch_hub
OUT=/data/guoj0f/proteingym-ttt-repro/scores
DMS=/data/guoj0f/share/ProteinGym/DMS_ProteinGym_substitutions
REF="$D/workstation-records/proteingym-ttt-repro/refs/DMS_substitutions.csv"

echo "[queue] $(date '+%F %T') stage 0: re-score the 5 MSA-sub-region assays"
for M in esm2_t12_35M_UR50D esm2_t33_650M_UR50D; do
  wait_for_free
  python scripts/eval_proteingym.py --model "$M" --mode baseline \
    --dms_reference "$REF" --dms_dir "$DMS" --out_dir "$OUT" \
    --score_batch_size 16 --assays "$SUBREGION" --overwrite
done

echo "[queue] $(date '+%F %T') stage 1: S1 35M seed 0"
wait_for_free
bash "$S1" esm2_t12_35M_UR50D "0" --pre_score every > $L/s1_35M_seed0.out 2> $L/s1_35M_seed0.err

echo "[queue] $(date '+%F %T') stage 2: S1 650M seed 0"
wait_for_free
bash "$S1" esm2_t33_650M_UR50D "0" --pre_score first > $L/s1_650M_seed0.out 2> $L/s1_650M_seed0.err

echo "[queue] $(date '+%F %T') stage 3: S1 35M seeds 1-4"
wait_for_free
bash "$S1" esm2_t12_35M_UR50D "1 2 3 4" --pre_score every > $L/s1_35M_seed1234.out 2> $L/s1_35M_seed1234.err

echo "[queue] $(date '+%F %T') QUEUE_DONE"
