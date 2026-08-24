#!/bin/bash
#SBATCH --job-name=byollr_gen
#SBATCH --account=sk036
#SBATCH --array=2-6           # task ID = BYOL seed; one task per seed
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT
export PYTHONUNBUFFERED=1

TRAINING_SEED=${SLURM_ARRAY_TASK_ID}   # BYOL training seed
DATA_SEED=2

BYOL_RUNS_ROOT="outputs/byol_runs"
RUN_DIR="outputs/supervised_baseline_classifiers/gen_sweep"  # same root as ENB0
GEN_VARIANT="initial"

SW_VALS=(0.0 0.1 0.5)

CONFIGS=(
    "initial_pure   none   0.0"
)

for SW in "${SW_VALS[@]}"; do
    BYOL_RUN_DIR="${BYOL_RUNS_ROOT}/pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f1/data_seed_${DATA_SEED}/training_seed_${TRAINING_SEED}"
    GEN_DIR="${BYOL_RUN_DIR}/data/generative"
    NAME_SUFFIX="_sw${SW}_s${TRAINING_SEED}"

    echo "════════════════════════════════════════════════════════"
    echo "SW=${SW}  TRAINING_SEED=${TRAINING_SEED}"
    echo "BYOL run dir : ${BYOL_RUN_DIR}"
    echo "Gen dir      : ${GEN_DIR}"
    echo "════════════════════════════════════════════════════════"

    for cfg in "${CONFIGS[@]}"; do
        read -r LS CWM CWS <<< "$cfg"

        if [ "${CWM}" = "none" ] || [ "${CWM}" = "None" ]; then
            CW_TAG="cwNone"
        else
            CW_STR_PART=$([ "${CWS}" != "1.0" ] && echo "${CWS}" || echo "")
            CW_TAG="cw${CWM}${CW_STR_PART}"
        fi

        OUT_DIR="${RUN_DIR}/with_generative/byollr_${LS}_${CW_TAG}${NAME_SUFFIX}"

        echo "  label_set=${LS}  cw_tag=${CW_TAG}  → ${OUT_DIR}"

        python scripts/train_byol_lr_gen_aug.py \
            --byol_run_dir  "${BYOL_RUN_DIR}" \
            --gen_dir       "${GEN_DIR}" \
            --gen_variant   "${GEN_VARIANT}" \
            --out_dir       "${OUT_DIR}" \
            --label_set     "${LS}" \
            --data_seed     "${DATA_SEED}" \
            --gen_fracs     0.0 0.5 1.0 2.0 5.0 \
            --n_runs        3 \
            --seed          42 \
            --lr_c          1.0

        echo "  Done: $(date)"
    done
done

echo "END: $(date)"
