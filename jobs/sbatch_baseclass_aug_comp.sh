#!/bin/bash
#SBATCH --job-name=aug_comp
#SBATCH --account=sk036
#SBATCH --array=2-6
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

source .venv/bin/activate
cd /users/mbredber/p3_SUPLAT

SEED=${SLURM_ARRAY_TASK_ID}
DATA_SEED=2
RUN_DIR="outputs/supervised_baseline_classifiers/aug_comp"
DATA_DIR="data/preprocessed/lotss"
SW_VALS=(0.0 0.1 0.5)

# Arguments shared by all three modes. All evaluated on data_seed=2 test set.
# --skip_sweep disables the label-fraction sweep (not needed here).
COMMON="--model enb0 --label_set initial_pure --data_dir $DATA_DIR
        --epochs 50 --batch_size 256 --lr 3e-5
        --data_seed $DATA_SEED --n_runs 3 --num_workers 4
        --gen_variant initial --skip_sweep --seed $SEED"

# ── Mode 1: real images, NO classical augmentation ───────────────────────────
python scripts/train_baseline_classifier.py $COMMON \
    --run_dir "$RUN_DIR" \
    --run_name "enb0_initial_pure_noaug_s${SEED}" \
    --no_augmentation

# ── Mode 2: real images, WITH classical augmentation (quart_ext) ─────────────
python scripts/train_baseline_classifier.py $COMMON \
    --run_dir "$RUN_DIR" \
    --run_name "enb0_initial_pure_classicaug_s${SEED}"

# ── Mode 3: real + generated images, NO classical augmentation ───────────────
# Sweeps over BYOL style-weight values and gen fractions.
# Each gen_frac adds a frac_X.XX/ subdir to the same run dir.
GEN_FRACS=(1.0 5.0)
for SW in "${SW_VALS[@]}"; do
    GEN_DIR="outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f1/data_seed_${DATA_SEED}/training_seed_${SEED}/data/generative"
    for GF in "${GEN_FRACS[@]}"; do
        python scripts/train_baseline_classifier.py $COMMON \
            --run_dir "$RUN_DIR" \
            --run_name "enb0_initial_pure_genaug_noaug_sw${SW}_s${SEED}" \
            --gen_dir "$GEN_DIR" \
            --gen_frac "$GF" \
            --no_augmentation
    done
done
