#!/bin/bash
#SBATCH --job-name=byol_finetune
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"
cd /users/mbredber/p3_SUPLAT
source /users/mbredber/p3_SUPLAT/.venv/bin/activate

# ── Target BYOL checkpoint (gamma=0.25, augquart_ext, sw=0.05) ───────────────
BEST_RUN=outputs/byol_runs/enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f1_sw0.05_augquart_ext_20260709_2203

# ── Shared hyperparameters ────────────────────────────────────────────────────
EPOCHS=100      # early stopping (patience=15) will terminate before this in practice
PATIENCE=15
LR=1e-4
WD=1e-4
N_RUNS=10
NUM_WORKERS=4

# Mode 1: frozen encoder + projector, train linear head only
# Equivalent to a linear probe on 128-dim projector output (vs probe_encodings.py on 1280-dim encoder)
echo ""
echo "=== Mode 1: frozen encoder + projector, linear head only ==="
python scripts/train_finetuning.py \
    --model-path=${BEST_RUN} \
    --training-mode=1 \
    --epochs=${EPOCHS} \
    --patience=${PATIENCE} \
    --lr=${LR} \
    --weight-decay=${WD} \
    --n-runs=${N_RUNS} \
    --num-workers=${NUM_WORKERS} \
    --augmentation=quart_ext \
    --run-name=mode1_lr${LR}_ep${EPOCHS}

# Mode 2: frozen encoder, fine-tune projector + linear head
echo ""
echo "=== Mode 2: frozen encoder, fine-tune projector + head ==="
python scripts/train_finetuning.py \
    --model-path=${BEST_RUN} \
    --training-mode=2 \
    --epochs=${EPOCHS} \
    --patience=${PATIENCE} \
    --lr=${LR} \
    --weight-decay=${WD} \
    --n-runs=${N_RUNS} \
    --num-workers=${NUM_WORKERS} \
    --augmentation=quart_ext \
    --run-name=mode2_lr${LR}_ep${EPOCHS}

echo ""
echo "END: $(date)"
