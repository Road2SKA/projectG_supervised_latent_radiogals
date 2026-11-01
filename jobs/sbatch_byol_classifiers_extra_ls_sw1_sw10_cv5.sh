#!/bin/bash
#SBATCH --job-name=byol_clf_extra_ls_sw1_sw10_cv5
#SBATCH --array=0-9
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

# LR/KNN/RF (run_downstream_classifiers.py) and MLP (run_downstream_mlp_es.py)
# for extra label sets — sw=1.0 and sw=10.0, 5-fold CV.
# Mirrors sbatch_byol_classifiers_extra_ls_cv5.sh for the two missing sw values.
#
# Label sets: classical_pure, initial_binary, full_binary
# (initial_pure already covered by sbatch_byol_classifiers_sw1_sw10_cv5.sh)
#
# Array layout: task_id = sw_idx*5 + fold
#   sw_idx 0 → sw=1.0,  tasks 0-4
#   sw_idx 1 → sw=10.0, tasks 5-9

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p outputs/logs
export PYTHONUNBUFFERED=1

SW_VALUES=("1.0" "10.0")
SW_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
SW=${SW_VALUES[$SW_IDX]}
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f1"

LABEL_SETS=("classical_pure" "initial_binary" "full_binary")

echo "=== byol_clf_extra_ls_sw1_sw10_cv5  sw=${SW}  fold=${FOLD}  task=${SLURM_ARRAY_TASK_ID} — $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"

for LS in "${LABEL_SETS[@]}"; do
    echo "--- label-set: ${LS} ---"
    FORCE_ARG=""
    [ "${LS}" = "initial_binary" ] && FORCE_ARG="--force"

    python scripts/run_downstream_classifiers.py \
        --outputs-root outputs/byol_runs \
        --run-glob     "${RUN_NAME}" \
        --feature-type projections \
        --label-set    "${LS}" \
        --n-estimators 200 \
        --workers      8 \
        --seed         2 \
        --data-seed    2 \
        --cv-fold      "${FOLD}" \
        ${FORCE_ARG}

    python scripts/run_downstream_mlp_es.py \
        --outputs-root outputs/byol_runs \
        --run-glob     "${RUN_NAME}" \
        --feature-type projections \
        --label-set    "${LS}" \
        --seed         2 \
        --data-seed    2 \
        --cv-fold      "${FOLD}" \
        ${FORCE_ARG}
done

echo "=== Done — $(date) ==="
