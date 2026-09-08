#!/bin/bash
#SBATCH --job-name=noise_rob_cv5
#SBATCH --array=0-10
#SBATCH --account=sk036
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT
export PYTHONUNBUFFERED=1

mkdir -p outputs/logs

# Same 11 combinations as sbatch_byol_cv5.sh
SW_VALUES=("0.0"  "0.05" "0.05" "0.05" "0.05" "0.05" "0.1"  "0.1"  "0.1"  "0.1"  "0.1")
F_VALUES=( "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0")

i=$SLURM_ARRAY_TASK_ID
SW=${SW_VALUES[$i]}
F=${F_VALUES[$i]}
F_STR="${F%.0}"   # strip trailing .0 so f=1.0 → f1 (matches dir names)
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f${F_STR}"

echo "START: $(date)"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "run=${RUN_NAME}  data_seed=2  training_seed=2"

for FOLD in 0 1 2 3 4; do
    echo "════════════════════════════════════════════════════════"
    echo "Fold ${FOLD}  run=${RUN_NAME}"
    echo "════════════════════════════════════════════════════════"

    SEED_DIR="outputs/byol_runs/byol_runs/${RUN_NAME}/data_seed_2/training_seed_2/cross_val_${FOLD}"
    if [ ! -d "${SEED_DIR}" ]; then
        echo "Skipping missing: ${SEED_DIR}"
        continue
    fi

    OUT_JSON="${SEED_DIR}/data/anomaly/noise_robustness.json"
    if [ -f "${OUT_JSON}" ]; then
        echo "Skipping already complete: ${OUT_JSON}"
        continue
    fi

    python scripts/compute_noise_robustness.py \
        --byol-run  "byol_runs/${RUN_NAME}" \
        --data_seed 2 \
        --seed      2 \
        --cv-fold   "${FOLD}"

    echo "  Done: $(date)"
done

echo "END: $(date)"
