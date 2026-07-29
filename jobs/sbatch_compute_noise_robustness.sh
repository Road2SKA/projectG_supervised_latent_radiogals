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
DATA_SEED=42
SEED=42
# Add --force to overwrite existing outputs
# =============================================================================

python scripts/compute_noise_robustness.py \
    --data_seed "${DATA_SEED}" \
    --seed      "${SEED}" \
    --force
echo "END: $(date)"
