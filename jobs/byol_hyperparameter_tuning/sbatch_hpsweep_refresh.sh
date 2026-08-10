#!/bin/bash
#SBATCH --job-name=hpsweep_refresh
#SBATCH --account=sk036
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"
REPO_ROOT=/users/mbredber/p3_SUPLAT
cd "${REPO_ROOT}"
source /users/mbredber/p3_SUPLAT/.venv/bin/activate

RUNS_ROOT="outputs/hyperparameter_tuning/byol_runs"

# Refresh row.json for every run dir found under byol_runs/
for RUN_DIR in "${RUNS_ROOT}"/*/; do
    RUN_DIR="${RUN_DIR%/}"   # strip trailing slash
    # Skip if no checkpoint (not a real run dir)
    [ -f "${RUN_DIR}/byol_model_best.pt" ] || continue
    echo "Refreshing: $(basename ${RUN_DIR})"
    python scripts/hyperparameter_sweep/collate_hpsweep.py --mode row --run-dir "${RUN_DIR}"
done

echo ""
echo "Aggregating..."
python scripts/hyperparameter_sweep/collate_hpsweep.py \
    --mode aggregate \
    --sweep-root "${RUNS_ROOT}"

echo "END: $(date)"
