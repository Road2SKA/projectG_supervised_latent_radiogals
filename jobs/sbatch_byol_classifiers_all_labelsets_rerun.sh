#!/bin/bash
#SBATCH --job-name=byol_clf_all_ls_rerun
#SBATCH --array=1,7,12,17,23,28,33,39,44,49,55,60
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

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"

mkdir -p outputs/logs

# Targets the 3 binary label sets (n_classes=2) that crashed with the
# label_binarize IndexError, across all 4 seeds — 12 tasks total.
#   LS_IDX  1 → classical_pure       (n=2)
#   LS_IDX  7 → interest_binary      (n=2)
#   LS_IDX 12 → classical_individual (n=2)
N_LS=16
SEEDS=(3 4 5 6)
LABEL_SETS=(
    "initial_pure"          "classical_pure"        "morphology_pure"
    "environment_pure"      "full_pure"
    "derived"               "interest_tier"         "interest_binary"
    "initial"               "morphology"            "full"
    "initial_individual"    "classical_individual"  "morphology_individual"
    "environment_individual" "full_individual"
)

SEED_IDX=$((SLURM_ARRAY_TASK_ID / N_LS))
LS_IDX=$((SLURM_ARRAY_TASK_ID % N_LS))

SEED=${SEEDS[$SEED_IDX]}
LS=${LABEL_SETS[$LS_IDX]}

echo "Starting BYOL classifiers (rerun binary label sets) — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "seed=${SEED}  data_seed=2  label_set=${LS}"

python scripts/run_downstream_classifiers.py \
    --outputs-root outputs/byol_runs/byol_runs \
    --run-glob     "pd128_*_f1" \
    --feature-type projections \
    --label-set    "${LS}" \
    --n-estimators 200 \
    --workers      8 \
    --seed         "${SEED}" \
    --data-seed    2

echo "Done — $(date)"
