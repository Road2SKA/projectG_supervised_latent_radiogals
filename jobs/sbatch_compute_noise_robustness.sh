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
# =============================================================================
DATA_SEED=2
# Add --force to overwrite existing outputs
# =============================================================================

for RUN_DIR in outputs/byol_runs/pd128_*_f1; do
    RUN_NAME="$(basename "${RUN_DIR}")"
    for SEED_DIR in "${RUN_DIR}"/seed*; do
        SEED="$(basename "${SEED_DIR}" | sed 's/seed//')"
        echo "════════════════════════════════════════════════════════"
        echo "Run: ${RUN_NAME}  seed: ${SEED}"
        echo "════════════════════════════════════════════════════════"
        python scripts/compute_noise_robustness.py \
            --byol-run  "${RUN_NAME}" \
            --data_seed "${DATA_SEED}" \
            --seed      "${SEED}" \
            --force
        echo "  Done: $(date)"
    done
done

echo "END: $(date)"
