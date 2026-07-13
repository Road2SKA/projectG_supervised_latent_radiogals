#!/bin/bash
#SBATCH --job-name=bclass_gen
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT
export PYTHONUNBUFFERED=1

# =============================================================================
# Edit these before submitting
# =============================================================================
BYOL_RUN="outputs/byol_runs/enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f1_sw0.05_augquart_ext_20260709_2203"
RUN_DIR="outputs/supervised_baseline_classifiers/with_generative/enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f1_sw0.05_augquart_ext_20260709_2203"
DATA_DIR="data/preprocessed/lotss"
GEN_DIR="${BYOL_RUN}/data/generative"
LABEL_SET="initial_pure"
N_RUNS=5
EPOCHS=100
BATCH_SIZE=256
LR=1e-4
PATIENCE=15
SEED=42
DATA_SEED=42
NUM_WORKERS=4
# =============================================================================

mkdir -p "${RUN_DIR}"

python scripts/train_baseline_classifiers.py \
    --run_dir      "${RUN_DIR}" \
    --byol_run_dir "${BYOL_RUN}" \
    --data_dir     "${DATA_DIR}" \
    --model        enb0 \
    --label_set    "${LABEL_SET}" \
    --gen_dir      "${GEN_DIR}" \
    --n_runs       "${N_RUNS}" \
    --epochs       "${EPOCHS}" \
    --batch_size   "${BATCH_SIZE}" \
    --lr           "${LR}" \
    --patience     "${PATIENCE}" \
    --seed         "${SEED}" \
    --data_seed    "${DATA_SEED}" \
    --num_workers  "${NUM_WORKERS}" \
    --force

echo "END: $(date)"
