#!/bin/bash
#SBATCH --job-name=byol_prot_cv5
#SBATCH --array=0-10
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"

mkdir -p outputs/logs

# Same 11 combinations as sbatch_byol_cv5.sh
SW_VALUES=("0.0"  "0.05" "0.05" "0.05" "0.05" "0.05" "0.1"  "0.1"  "0.1"  "0.1"  "0.1")
F_VALUES=( "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0")

i=$SLURM_ARRAY_TASK_ID
SW=${SW_VALUES[$i]}
F=${F_VALUES[$i]}
F_STR="${F%.0}"   # strip trailing .0 so f=1.0 → f1 (matches dir names)
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f${F_STR}"

echo "Starting BYOL proteges (CV) — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "run=${RUN_NAME}  data_seed=2  training_seed=2"

for FOLD in 0 1 2 3 4; do
    echo "--- fold ${FOLD} ---"

    SEED_DIR="outputs/byol_runs/byol_runs/${RUN_NAME}/data_seed_2/training_seed_2/cross_val_${FOLD}"
    if [ ! -d "${SEED_DIR}" ]; then
        echo "Skipping missing: ${SEED_DIR}"
        continue
    fi

    python scripts/run_protege_scoring.py \
        --outputs-root outputs \
        --run-glob     "byol_runs/byol_runs/${RUN_NAME}" \
        --byol-seed    2 \
        --byol-data-seed 2 \
        --workers      8 \
        --steps        100 \
        --cv-fold      "${FOLD}"
done

echo "Done — $(date)"
