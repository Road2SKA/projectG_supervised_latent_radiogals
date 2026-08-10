#!/bin/bash
#SBATCH --job-name=ft_reg_sweep4
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

# Sweep 4: broader LR range (1e-1 to 1e-3), mode 2 only.
# Uses sw=0 BYOL run to decouple HP selection from the models being evaluated.
# Note: only data_seed=2 exists, so test-set overfitting cannot be fully avoided.

MODE=2
BYOL_SEED=3
MODEL_PATH="outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.0_f1/seed${BYOL_SEED}"

LRS=(1e-1 1e-2 1e-3)
WDS=(3e-1 1e-1 3e-2)
DROPOUTS=(0.0 0.2 0.4)

source .venv/bin/activate

for LR in "${LRS[@]}"; do
  for WD in "${WDS[@]}"; do
    for DO in "${DROPOUTS[@]}"; do
      RUN_NAME="mode${MODE}_lr${LR}_wd${WD}_do${DO}_ep40"
      echo "=== ${RUN_NAME} ==="
      python scripts/train_finetuning.py \
        --model-path="${MODEL_PATH}" \
        --training-mode=${MODE} \
        --label-set=initial_pure \
        --epochs=40 \
        --lr=${LR} \
        --weight-decay=${WD} \
        --dropout=${DO} \
        --n-runs=3 \
        --num-workers=4 \
        --augmentation=quart_ext \
        --run-name="${RUN_NAME}"
    done
  done
done
