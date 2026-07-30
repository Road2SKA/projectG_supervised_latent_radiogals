#!/bin/bash
#SBATCH --job-name=byol_hpsweep
#SBATCH --account=sk036
#SBATCH --array=0-31
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)  TASK_ID=${SLURM_ARRAY_TASK_ID}"
REPO_ROOT=/users/mbredber/p3_SUPLAT
cd "${REPO_ROOT}"
source /users/mbredber/p3_SUPLAT/.venv/bin/activate

# =============================================================================
# GRID DEFINITION  (32 = 2^5)
# =============================================================================
PD_VALUES=(128 512)
PD_TAGS=("pd128" "pd512")

WD_FLOATS=("0.0001" "0.001")
WD_TAGS=("wd1e-4" "wd1e-3")

LR_SCHEDULES=("constant" "step")
LR_TAGS=("lrconst" "lrstep")

AUG_VALUES=("quart_ext" "quart")
AUG_TAGS=("qext" "qsmp")

VIC_TAGS=("v0" "v1")

# Index arithmetic:  i = 0..31
#   pd_i  = i / 16          (0=pd128, 1=pd512)
#   aug_i = (i % 16) / 8    (0=quart_ext, 1=quart)
#   vic_i = (i % 8) / 4     (0=vic_off, 1=vic_on)
#   wd_i  = (i % 4) / 2     (0=wd1e-4, 1=wd1e-3)
#   lr_i  = i % 2           (0=constant, 1=step)
i=${SLURM_ARRAY_TASK_ID}
pd_i=$(( i / 16 ))
aug_i=$(( (i % 16) / 8 ))
vic_i=$(( (i % 8) / 4 ))
wd_i=$(( (i % 4) / 2 ))
lr_i=$(( i % 2 ))

SWEEP_ROOT="outputs/hyperparameter_tuning"
RUNS_ROOT="${SWEEP_ROOT}/byol_runs"

RUN_NAME="${PD_TAGS[$pd_i]}_${AUG_TAGS[$aug_i]}_${VIC_TAGS[$vic_i]}_${WD_TAGS[$wd_i]}_${LR_TAGS[$lr_i]}"

echo "Run: ${RUN_NAME}  (pd=${PD_VALUES[$pd_i]}  aug=${AUG_VALUES[$aug_i]}  vic_i=${vic_i}  wd=${WD_FLOATS[$wd_i]}  lr_sched=${LR_SCHEDULES[$lr_i]})"

# =============================================================================
# VICReg flags
# =============================================================================
if [ $vic_i -eq 0 ]; then
    VIC_FLAGS="--vicreg-var-weight 0.0 --vicreg-cov-weight 0.0"
else
    VIC_FLAGS="--vicreg-var-weight 2.0 --vicreg-cov-weight 0.2 --vicreg-gamma 0.25"
fi

# =============================================================================
# TRAINING
# =============================================================================
python scripts/train_byol.py \
    --run-name            "${RUN_NAME}" \
    --output-dir          "${SWEEP_ROOT}" \
    --model-type          efficientnet-b0 \
    --projector           mlp \
    --projection-dim      "${PD_VALUES[$pd_i]}" \
    --augmentation        "${AUG_VALUES[$aug_i]}" \
    --supervision-weight  0.0 \
    --f-label             1.0 \
    --ema-decay           0.996 \
    --weighting           closest \
    --batch-size          512 \
    --epochs              300 \
    --seed                1 \
    --data-seed           1 \
    --lr                  3e-4 \
    --lr-schedule         "${LR_SCHEDULES[$lr_i]}" \
    --weight-decay        "${WD_FLOATS[$wd_i]}" \
    --num-workers         4 \
    ${VIC_FLAGS}

# =============================================================================
# POST-TRAINING PROBES
# =============================================================================
RUN_DIR=$(ls -dt "${RUNS_ROOT}/${RUN_NAME}_"* 2>/dev/null | head -1)

if [ -z "${RUN_DIR}" ]; then
    echo "ERROR: Could not find run dir matching ${RUNS_ROOT}/${RUN_NAME}_*"
    exit 1
fi

echo "Run dir: ${RUN_DIR}"

python scripts/run_hpsweep_classifiers.py --run-dir "${RUN_DIR}"
python scripts/collate_hpsweep.py --mode row --run-dir "${RUN_DIR}"

echo ""
echo "END: $(date)"
