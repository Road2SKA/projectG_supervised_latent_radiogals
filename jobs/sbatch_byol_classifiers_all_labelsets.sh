#!/bin/bash
#SBATCH --job-name=byol_clf_all_ls
#SBATCH --array=0-63
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

# ── Array layout: 4 seeds × 16 label sets = 64 tasks ─────────────────────────
# Seeds 3-6 with data_seed=2 (off-diagonal dirs: data_seed_2/training_seed_N).
# run-glob matches all sw values for f=1 in one shot.
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

echo "Starting BYOL classifiers (all label sets) — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "seed=${SEED}  data_seed=2  label_set=${LS}"

python scripts/train_byol_classifiers.py \
    --outputs-root outputs/byol_runs/byol_runs \
    --run-glob     "pd128_*_f1" \
    --feature-type projections \
    --label-set    "${LS}" \
    --n-estimators 200 \
    --workers      8 \
    --seed         "${SEED}" \
    --data-seed    2

echo "Done — $(date)"
