#!/bin/bash
#SBATCH --job-name=mlp_es02_sw1_sw10_cv5
#SBATCH --array=0-9
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

# MLP classifiers with early stopping — sw=1.0 and sw=10.0, 5-fold CV.
# Mirrors sbatch_mlp_es02_cv5.sh for the two missing sw values.
# Does NOT overwrite existing results.
#
# Array layout: task_id = sw_idx * 5 + fold
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

echo "=== mlp_es02_sw1_sw10_cv5  sw=${SW}  fold=${FOLD}  task=${SLURM_ARRAY_TASK_ID} — $(date) ==="
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
