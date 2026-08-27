#!/bin/bash
#SBATCH --job-name=noise_robustness
#SBATCH --account=sk036
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x_%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"
echo "Node: ${SLURMD_NODENAME:-local}"

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT
export PYTHONUNBUFFERED=1

# =============================================================================
# CONFIGURATION
# Diagonal condition: data_seed == training_seed (same seed for data split
# and model training). Add --force to overwrite existing outputs.
# =============================================================================
SW_VALUES=(0.0 0.05 0.1 0.5)
SEEDS=(2 3 4 5 6)
# =============================================================================

for SW in "${SW_VALUES[@]}"; do
    RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f1"
    for SEED in "${SEEDS[@]}"; do
        SEED_DIR="outputs/byol_runs/${RUN_NAME}/data_seed_${SEED}/training_seed_${SEED}"
        if [ ! -d "${SEED_DIR}" ]; then
            echo "Skipping ${RUN_NAME} seed=${SEED}: directory not found"
            continue
        fi
        echo "════════════════════════════════════════════════════════"
        echo "Run: ${RUN_NAME}  data_seed=${SEED}  training_seed=${SEED}"
        echo "════════════════════════════════════════════════════════"
        python scripts/compute_noise_robustness.py \
            --byol-run  "${RUN_NAME}" \
            --data_seed "${SEED}" \
            --seed      "${SEED}"
        echo "  Done: $(date)"
    done
done

echo "END: $(date)"
