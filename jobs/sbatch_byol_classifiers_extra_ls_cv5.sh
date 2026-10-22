#!/bin/bash
#SBATCH --job-name=byol_clf_extra_ls_cv5
#SBATCH --array=0-19
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

# Runs LR/KNN/RF (run_downstream_classifiers.py) and MLP-v2 (run_downstream_mlp_es.py)
# for additional label sets on the 4 sw-sweep f=1 runs × 5 CV folds.
#
# Label sets: classical_pure, initial_binary, full, full_individual
# (initial_pure is already covered by sbatch_byol_classifiers_cv5.sh)
#
# Array: 4 runs × 5 folds = 20 tasks  (task_id = run_idx*5 + fold_idx)

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"

mkdir -p outputs/logs

RUN_NAMES=(
    "pd128_qext_v1_wd1e-3_lrconst_sw0.0_f1"
    "pd128_qext_v1_wd1e-3_lrconst_sw0.05_f1"
    "pd128_qext_v1_wd1e-3_lrconst_sw0.1_f1"
    "pd128_qext_v1_wd1e-3_lrconst_sw0.5_f1"
)

LABEL_SETS=("classical_pure" "initial_binary" "full" "full_individual")

RUN_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
RUN_NAME="${RUN_NAMES[$RUN_IDX]}"

echo "Starting extra-LS classifiers (CV) — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "run=${RUN_NAME}  fold=${FOLD}  data_seed=2  seed=2"

for LS in "${LABEL_SETS[@]}"; do
    echo "--- label-set: ${LS} ---"
    python scripts/run_downstream_classifiers.py \
        --outputs-root outputs/byol_runs \
        --run-glob     "${RUN_NAME}" \
        --feature-type projections \
        --label-set    "${LS}" \
        --n-estimators 200 \
        --workers      8 \
        --seed         2 \
        --data-seed    2 \
        --cv-fold      "${FOLD}"

    python scripts/run_downstream_mlp_es.py \
        --outputs-root outputs/byol_runs \
        --run-glob     "${RUN_NAME}" \
        --feature-type projections \
        --label-set    "${LS}" \
        --seed         2 \
        --data-seed    2 \
        --cv-fold      "${FOLD}"
done

echo "Done — $(date)"
