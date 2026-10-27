#!/bin/bash
#SBATCH --job-name=mlp_frac_cv5
#SBATCH --array=0-59
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

# MLP classifiers (early stopping) for f<1 runs — fills in the f-sweep in
# the classification notebook.  sbatch_mlp_es02_cv5.sh already covers f=1.
#
# 12 runs × 5 CV folds = 60 tasks.  task_id = run_idx*5 + fold_idx
#
# Run layout:
#   idx 0:  sw=0.0  f=0.05
#   idx 1:  sw=0.0  f=0.1
#   idx 2:  sw=0.0  f=0.25
#   idx 3:  sw=0.0  f=0.5
#   idx 4:  sw=0.05 f=0.05
#   idx 5:  sw=0.05 f=0.1
#   idx 6:  sw=0.05 f=0.25
#   idx 7:  sw=0.05 f=0.5
#   idx 8:  sw=0.1  f=0.05
#   idx 9:  sw=0.1  f=0.1
#   idx 10: sw=0.1  f=0.25
#   idx 11: sw=0.1  f=0.5

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p outputs/logs
export PYTHONUNBUFFERED=1

RUN_NAMES=(
    "pd128_qext_v1_wd1e-3_lrconst_sw0.0_f0.05"
    "pd128_qext_v1_wd1e-3_lrconst_sw0.0_f0.1"
    "pd128_qext_v1_wd1e-3_lrconst_sw0.0_f0.25"
    "pd128_qext_v1_wd1e-3_lrconst_sw0.0_f0.5"
    "pd128_qext_v1_wd1e-3_lrconst_sw0.05_f0.05"
    "pd128_qext_v1_wd1e-3_lrconst_sw0.05_f0.1"
    "pd128_qext_v1_wd1e-3_lrconst_sw0.05_f0.25"
    "pd128_qext_v1_wd1e-3_lrconst_sw0.05_f0.5"
    "pd128_qext_v1_wd1e-3_lrconst_sw0.1_f0.05"
    "pd128_qext_v1_wd1e-3_lrconst_sw0.1_f0.1"
    "pd128_qext_v1_wd1e-3_lrconst_sw0.1_f0.25"
    "pd128_qext_v1_wd1e-3_lrconst_sw0.1_f0.5"
)

RUN_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
RUN_NAME="${RUN_NAMES[$RUN_IDX]}"

echo "=== mlp_frac_cv5  run=${RUN_NAME}  fold=${FOLD}  task=${SLURM_ARRAY_TASK_ID} — $(date) ==="
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
