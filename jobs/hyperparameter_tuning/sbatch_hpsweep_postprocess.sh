#!/bin/bash
#SBATCH --job-name=hpsweep_postprocess
#SBATCH --account=sk036
#SBATCH --array=0-31
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)  TASK_ID=${SLURM_ARRAY_TASK_ID}"
REPO_ROOT=/users/mbredber/p3_SUPLAT
cd "${REPO_ROOT}"
source /users/mbredber/p3_SUPLAT/.venv/bin/activate

# =============================================================================
# GRID (must match sbatch_hpsweep.sh)
# =============================================================================
PD_TAGS=("pd128" "pd512")
WD_TAGS=("wd1e-4" "wd1e-3")
LR_TAGS=("lrconst" "lrstep")
AUG_TAGS=("qext" "qsmp")
VIC_TAGS=("v0" "v1")

i=${SLURM_ARRAY_TASK_ID}
pd_i=$(( i / 16 ))
aug_i=$(( (i % 16) / 8 ))
vic_i=$(( (i % 8) / 4 ))
wd_i=$(( (i % 4) / 2 ))
lr_i=$(( i % 2 ))

RUNS_ROOT="outputs/hyperparameter_tuning/byol_runs"

RUN_NAME="${PD_TAGS[$pd_i]}_${AUG_TAGS[$aug_i]}_${VIC_TAGS[$vic_i]}_${WD_TAGS[$wd_i]}_${LR_TAGS[$lr_i]}"

RUN_DIR=$(ls -dt "${RUNS_ROOT}/${RUN_NAME}_"* 2>/dev/null | head -1)

if [ -z "${RUN_DIR}" ]; then
    echo "ERROR: No run dir found matching ${RUNS_ROOT}/${RUN_NAME}_*"
    exit 1
fi

echo "Run: ${RUN_NAME}  ->  ${RUN_DIR}"

# =============================================================================
# PROTEGE GP + LR CLASSIFIER + STATUS.JSON + ROW.JSON
# (all handled by run_hpsweep_postprocess.py)
# =============================================================================
python scripts/run_hpsweep_postprocess.py \
    --run-dir "${RUN_DIR}" \
    --epsilon 2.0 \
    --steps   100

echo ""
echo "END: $(date)"
