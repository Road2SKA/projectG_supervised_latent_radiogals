#!/bin/bash
#SBATCH --job-name=byollr_sw_fl_sweep
#SBATCH --account=sk036
#SBATCH --array=0-4
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

# sw in {0.05, 0.1}, f_label in {0.05, 0.1, 0.25, 0.5}
# data_seed in {2,3,4,5,6}, training_seed = data_seed + 1
DATA_SEEDS=(2 3 4 5 6)
TRAINING_SEEDS=(3 4 5 6 7)

DATA_SEED=${DATA_SEEDS[$SLURM_ARRAY_TASK_ID]}
TRAINING_SEED=${TRAINING_SEEDS[$SLURM_ARRAY_TASK_ID]}

BYOL_RUNS_ROOT="outputs/byol_runs"
RUN_DIR="outputs/supervised_baseline_classifiers/gen_sweep"
GEN_VARIANT="initial"

SW_VALS=(0.05 0.1)
FL_VALS=(0.05 0.1 0.25 0.5)
FL_TAGS=(f0.05 f0.1 f0.25 f0.5)

CONFIGS=(
    "initial_pure   none   0.0"
)

echo "data_seed=${DATA_SEED}  training_seed=${TRAINING_SEED}"

for SW in "${SW_VALS[@]}"; do
    for fl_i in 0 1 2 3; do
        FL=${FL_VALS[$fl_i]}
        FL_TAG=${FL_TAGS[$fl_i]}

        BYOL_RUN_DIR="${BYOL_RUNS_ROOT}/pd128_qext_v1_wd1e-3_lrconst_sw${SW}_${FL_TAG}/data_seed_${DATA_SEED}/training_seed_${TRAINING_SEED}"
        GEN_DIR="${BYOL_RUN_DIR}/data/generative"
        NAME_SUFFIX="_sw${SW}_${FL_TAG}_s${TRAINING_SEED}"

        if [ ! -d "${BYOL_RUN_DIR}" ]; then
            echo "Skipping missing: ${BYOL_RUN_DIR}"
            continue
        fi

        echo "════════════════════════════════════════════════════════"
        echo "SW=${SW}  FL=${FL}  DATA_SEED=${DATA_SEED}  TRAINING_SEED=${TRAINING_SEED}"
        echo "BYOL run dir : ${BYOL_RUN_DIR}"
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
done

echo "END: $(date)"
