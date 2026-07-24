#!/bin/bash
#SBATCH --job-name=noise_sweep
#SBATCH --account=sk036
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"
echo "Node: ${SLURMD_NODENAME:-local}"

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT
export PYTHONUNBUFFERED=1

# =============================================================================
# CONFIGURATION
# Set NOISE_RUN_DIR to the specific BYOL run directory to sweep.
# Leave empty to use the first directory matched by --run-glob under OUTPUTS_ROOT.
# =============================================================================
OUTPUTS_ROOT="outputs/byol_runs"
NOISE_RUN_DIR=""
RUN_GLOB="enb0_*"
# =============================================================================

NOISE_SWEEP_ARGS=(
    --noise-sweep
    --outputs-root  "${OUTPUTS_ROOT}"
    --run-glob      "${RUN_GLOB}"
    --noise-sigmas  0.0 0.25 0.5 1.0 2.0
    --noise-rng-seed 42
    --pca
)

if [ -n "${NOISE_RUN_DIR}" ]; then
    NOISE_SWEEP_ARGS+=(--noise-run-dir "${NOISE_RUN_DIR}")
fi

python scripts/train_byol_proteges.py "${NOISE_SWEEP_ARGS[@]}"

echo "END: $(date)"
