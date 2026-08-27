#!/bin/bash
#SBATCH --job-name=gen_dataseed
#SBATCH --account=sk036
#SBATCH --array=0-8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT
export PYTHONUNBUFFERED=1

# All (data_seed, training_seed) pairs:
#   data_seed=2: training seeds 2-6 (existing runs)
#   data_seed=N: training seed N    (new diagonal runs, N in {3,4,5,6})
DATA_SEEDS=(2 2 2 2 2 3 4 5 6)
TRAINING_SEEDS=(2 3 4 5 6 3 4 5 6)

DATA_SEED=${DATA_SEEDS[$SLURM_ARRAY_TASK_ID]}
SEED=${TRAINING_SEEDS[$SLURM_ARRAY_TASK_ID]}

IMAGES="data/preprocessed/lotss/images_filtered.npy"
LABEL="initial"

echo "data_seed=${DATA_SEED}  training_seed=${SEED}"

for RUN_DIR in \
    outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.0_f1 \
    outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.05_f1 \
    outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.1_f1 \
    outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.5_f1 \
    outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw1.0_f1; do

    OUT_DIR="${RUN_DIR}/data_seed_${DATA_SEED}/training_seed_${SEED}/data/generative"

    if [ ! -d "${RUN_DIR}/data_seed_${DATA_SEED}/training_seed_${SEED}" ]; then
        echo "Skipping missing: ${RUN_DIR}/data_seed_${DATA_SEED}/training_seed_${SEED}"
        continue
    fi

    if [ -f "${OUT_DIR}/decoder_${LABEL}.pt" ] && [ -f "${OUT_DIR}/nsf_${LABEL}.pt" ]; then
        echo "Skipping (already done): ${OUT_DIR}"
        continue
    fi

    echo "════════════════════════════════════════════════════════"
    echo "Run: $(basename ${RUN_DIR})  data_seed=${DATA_SEED}  training_seed=${SEED}"
    echo "════════════════════════════════════════════════════════"

    python scripts/train_generative.py \
        --base-dir            "${RUN_DIR}" \
        --images-path         "${IMAGES}" \
        --label-subset        "${LABEL}" \
        --seed                "${SEED}" \
        --data-seed           "${DATA_SEED}" \
        --decoder-type        flow \
        --decoder-batch-size  64 \
        --base-ch             32 \
        --decoder-epochs      300 \
        --decoder-patience    40 \
        --flow-epochs         200 \
        --flow-patience       20

    echo "  Done: $(date)"
done

echo "END: $(date)"
