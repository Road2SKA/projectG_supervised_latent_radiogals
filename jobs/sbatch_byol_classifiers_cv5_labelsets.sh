#!/bin/bash
#SBATCH --job-name=byol_clf_cv5_ls
#SBATCH --array=0-879
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"

mkdir -p outputs/logs

# 11 run configs × 16 label sets × 5 folds = 880 tasks
N_LS=16
N_FOLDS=5
RUN_IDX=$((SLURM_ARRAY_TASK_ID / (N_LS * N_FOLDS)))
LS_IDX=$(( (SLURM_ARRAY_TASK_ID / N_FOLDS) % N_LS ))
FOLD=$((SLURM_ARRAY_TASK_ID % N_FOLDS))

SW_VALUES=("0.0"  "0.05" "0.05" "0.05" "0.05" "0.05" "0.1"  "0.1"  "0.1"  "0.1"  "0.1")
F_VALUES=( "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0")
LABEL_SETS=(
    "initial_pure"          "classical_pure"        "morphology_pure"
    "environment_pure"      "full_pure"
    "derived"               "interest_tier"         "interest_binary"
    "initial"               "morphology"            "full"
    "initial_individual"    "classical_individual"  "morphology_individual"
    "environment_individual" "full_individual"
)

SW=${SW_VALUES[$RUN_IDX]}
F=${F_VALUES[$RUN_IDX]}
F_STR="${F%.0}"   # strip trailing .0 so f=1.0 → f1 (matches dir names)
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f${F_STR}"
LS=${LABEL_SETS[$LS_IDX]}

echo "Starting BYOL classifiers (CV, label sets) — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "run=${RUN_NAME}  label_set=${LS}  fold=${FOLD}  data_seed=2  training_seed=2"

python scripts/train_byol_classifiers.py \
    --outputs-root outputs/byol_runs/byol_runs \
    --run-glob     "${RUN_NAME}" \
    --feature-type projections \
    --label-set    "${LS}" \
    --n-estimators 200 \
    --workers      8 \
    --seed         2 \
    --data-seed    2 \
    --cv-fold      "${FOLD}"

echo "Done — $(date)"
