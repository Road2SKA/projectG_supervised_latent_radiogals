#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

source .venv/bin/activate

MODEL_PATH="outputs/byol_runs/enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f1_sw0.05_augquart_ext_20260709_2203"

for F_LABEL in 0.01 0.05 0.1 0.25 0.5 1.0; do
    echo "==============================="
    echo "Running f_label=${F_LABEL}"
    echo "==============================="
    python scripts/train_finetuning.py \
        --model-path "${MODEL_PATH}" \
        --f-label "${F_LABEL}" \
        --run-name "fl${F_LABEL}" \
        --training-mode 3 \
        --lr 1e-4 \
        --epochs 10 \
        --n-runs 10 \
        --augmentation quart \
        --batch-size 256 \
        --weight-decay 1e-4
done
