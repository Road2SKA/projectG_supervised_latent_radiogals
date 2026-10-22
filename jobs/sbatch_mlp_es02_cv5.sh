#!/bin/bash
#SBATCH --job-name=mlp_es02_cv5
#SBATCH --array=0-19
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

# MLP classifiers with early stopping (val_fraction=0.2) — 5-fold CV, 4 sw values.
# Runs both v1 (sklearn, val=0.2) and v2 (PyTorch, val=0.2, patience=20, max 300 ep).
# Saves to mlp_es02_*.json and mlp_v2_*_es02_*.json — does NOT overwrite existing results.
#
# Array layout: task_id = sw_idx * 5 + fold
#   sw_idx 0 → sw=0.0,  tasks 0-4
#   sw_idx 1 → sw=0.05, tasks 5-9
#   sw_idx 2 → sw=0.1,  tasks 10-14
#   sw_idx 3 → sw=0.5,  tasks 15-19
# CPU only (no --gres).

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p outputs/logs
export PYTHONUNBUFFERED=1

SW_VALUES=("0.0" "0.05" "0.1" "0.5")
SW_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
SW=${SW_VALUES[$SW_IDX]}
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f1"

echo "=== mlp_es02_cv5  sw=${SW}  fold=${FOLD}  task=${SLURM_ARRAY_TASK_ID} — $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-4}"

python scripts/run_downstream_mlp_es.py \
    --outputs-root outputs/byol_runs \
    --run-glob     "${RUN_NAME}" \
    --feature-type projections \
    --label-set    initial_pure \
    --seed         2 \
    --data-seed    2 \
    --cv-fold      "${FOLD}"

echo "=== Done — $(date) ==="
