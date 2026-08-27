#!/bin/bash
#SBATCH --job-name=baselines_dataseed
#SBATCH --account=sk036
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
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
# Compute Ellipses GP + blob_doh complexity baselines for each diagonal seed.
# Outputs: outputs/anomaly_baselines/baselines_{seed}.json
# Add --force to overwrite existing outputs.
# =============================================================================
SEEDS=(2 3 4 5 6)
# =============================================================================

for SEED in "${SEEDS[@]}"; do
    echo "════════════════════════════════════════════════════════"
    echo "data_seed=${SEED}"
    echo "════════════════════════════════════════════════════════"
    python scripts/compute_baselines.py --data-seed "${SEED}"
    echo "  Done: $(date)"
done

echo "END: $(date)"
