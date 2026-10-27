#!/bin/bash
#SBATCH --job-name=lineareval_frac_cv5
#SBATCH --array=0-59
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

# LinearEval (frozen backbone) label-fraction curve — 3 sw x 4 f x 5 folds = 60 tasks.
# Trains a linear head on labelled_train_idx_f<f>.npy (f-fraction labelled data only).
# Results written to:
#   outputs/classifiers/finetuned_frac_cv5/  (for Section 6 loading)
#   AND the BYOL run dir cross_val_K/data/classifiers/finetuning/ (via --byol_run_dir)
#
# Array layout: task_id = sw_idx*20 + f_idx*5 + fold_idx
#   sw in {0.0, 0.05, 0.1},  f in {0.05, 0.1, 0.25, 0.5}
#
# sw=0.0 reuses the f=1 checkpoint (supervision_weight=0 => model is f-independent).

set -euo pipefail
VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p outputs/logs
export PYTHONUNBUFFERED=1

SW_VALUES=("0.0" "0.05" "0.1")
F_VALUES=("0.05" "0.1" "0.25" "0.5")

SW_IDX=$(( SLURM_ARRAY_TASK_ID / 20 ))
F_IDX=$(( (SLURM_ARRAY_TASK_ID % 20) / 5 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))

SW=${SW_VALUES[$SW_IDX]}
F=${F_VALUES[$F_IDX]}
F_STR="${F%.0}"

RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f${F_STR}"
BYOL_RUN_DIR="outputs/byol_runs/${RUN_NAME}/data_seed_2/training_seed_2"

# Checkpoint: sw=0.0 runs reuse the f=1 model (sw=0 => labels have no effect on training)
if [ "${SW}" = "0.0" ]; then
    SW0_F1_DIR="outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.0_f1/data_seed_2/training_seed_2"
    FOLD_CKPT="${SW0_F1_DIR}/cross_val_${FOLD}/byol_model_best.pt"
    SHARED_CKPT="${SW0_F1_DIR}/byol_model_best.pt"
else
    FOLD_CKPT="${BYOL_RUN_DIR}/cross_val_${FOLD}/byol_model_best.pt"
    SHARED_CKPT="${BYOL_RUN_DIR}/byol_model_best.pt"
fi

if [ -f "${FOLD_CKPT}" ]; then
    BYOL_CKPT="${FOLD_CKPT}"
elif [ -f "${SHARED_CKPT}" ]; then
    BYOL_CKPT="${SHARED_CKPT}"
else
    echo "ERROR: no checkpoint found at ${FOLD_CKPT} or ${SHARED_CKPT}"
    exit 1
fi

echo "=== lineareval_frac_cv5  sw=${SW}  f=${F}  fold=${FOLD}  task=${SLURM_ARRAY_TASK_ID} -- $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "Checkpoint: ${BYOL_CKPT}"

python scripts/train_baseline_classifier.py \
    --run_dir        outputs/classifiers/finetuned_frac_cv5 \
    --run_name       "lineareval_frac_sw${SW}_f${F_STR}_fold${FOLD}" \
    --data_dir       data/preprocessed/lotss \
    --model          enb0 \
    --label_set      initial_pure \
    --epochs         20 \
    --batch_size     64 \
    --lr             1e-3 \
    --seed           2 \
    --data_seed      2 \
    --cv_fold        "${FOLD}" \
    --n_runs         1 \
    --num_workers    8 \
    --training_mode  linear \
    --f_label        "${F}" \
    --byol_run_dir   "${BYOL_RUN_DIR}" \
    --byol_checkpoint "${BYOL_CKPT}" \
    --force

echo "=== Done -- $(date) ==="
