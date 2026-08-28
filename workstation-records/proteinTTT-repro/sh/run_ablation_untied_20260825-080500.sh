#!/bin/bash
# Ablation: how much of our residual gap to the paper is explained by the weight-tying fix?
#
# Runs ESM2-35M seed 0 exactly as the shipped code behaves -- ttt_reset() leaves
# lm_head.weight untied, so assays #2..217 additionally train the output projection.
# Everything else (seed derivation, scoring, MSA-region convention) is identical to the
# fixed run, so the difference isolates the fix.
#
# Motivation: our per-category TTT gains differ from the paper with a consistent sign in
# BOTH model sizes -- Activity -0.0051/-0.0066, Stability +0.0032/+0.0070 -- while
# Binding/Expression/OrganismalFitness match to ~0.001. The tying fix is the only known
# methodological difference between our run and the shipped code.
set -uo pipefail
L=/data/guoj0f/proteinTTT-repro/logs
D=/home/guoj0f/repos/ProteinTTT/proteinTTT-proteinGYM-reproduce
while pgrep -f "[e]val_proteingym.py" >/dev/null; do sleep 60; done
echo "[ablation] $(date '+%F %T') starting 35M seed 0, tie restoration DISABLED"
bash "$D/workstation-records/proteinTTT-repro/sh/s1_ttt_esm2_20260824-231500.sh" \
  esm2_t12_35M_UR50D "0" --pre_score every --no_tie_restore \
  > $L/ablation_untied_35M_seed0.out 2> $L/ablation_untied_35M_seed0.err
echo "[ablation] $(date '+%F %T') ABLATION_DONE"
