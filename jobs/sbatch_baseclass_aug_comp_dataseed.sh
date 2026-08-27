#!/bin/bash
#SBATCH --job-name=aug_comp_dataseed
#SBATCH --account=sk036
#SBATCH --array=0-8
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

# All (data_seed, training_seed) pairs:
#   data_seed=2: training seeds 2-6 (existing runs)
#   data_seed=N: training seed N    (new diagonal runs, N in {3,4,5,6})
DATA_SEEDS=(2 2 2 2 2 3 4 5 6)
TRAINING_SEEDS=(2 3 4 5 6 3 4 5 6)

DATA_SEED=${DATA_SEEDS[$SLURM_ARRAY_TASK_ID]}
SEED=${TRAINING_SEEDS[$SLURM_ARRAY_TASK_ID]}

RUN_DIR="outputs/supervised_baseline_classifiers/aug_comp"
DATA_DIR="data/preprocessed/lotss"
SW_VALS=(0.0 0.05 0.1 0.5 1.0)

echo "data_seed=${DATA_SEED}  training_seed=${SEED}"

COMMON="--model enb0 --label_set initial_pure --data_dir $DATA_DIR
        --epochs 50 --batch_size 256 --lr 3e-5
        --data_seed $DATA_SEED --n_runs 3 --num_workers 4
        --gen_variant initial --skip_sweep --seed $SEED"

# ── Mode 1: real images, NO classical augmentation ───────────────────────────
python scripts/train_baseline_classifier.py $COMMON \
    --run_dir "$RUN_DIR" \
    --run_name "enb0_initial_pure_noaug_s${SEED}_ds${DATA_SEED}" \
    --no_augmentation

# ── Mode 2: real images, WITH classical augmentation (quart_ext) ─────────────
python scripts/train_baseline_classifier.py $COMMON \
    --run_dir "$RUN_DIR" \
    --run_name "enb0_initial_pure_classicaug_s${SEED}_ds${DATA_SEED}"

# ── Mode 3: real + generated images, NO classical augmentation ───────────────
GEN_FRACS=(1.0 5.0)
for SW in "${SW_VALS[@]}"; do
    GEN_DIR="outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f1/data_seed_${DATA_SEED}/training_seed_${SEED}/data/generative"

    if [ ! -d "${GEN_DIR}" ]; then
        echo "Skipping gen-aug sw=${SW}: generator not found at ${GEN_DIR}"
        continue
    fi

    for GF in "${GEN_FRACS[@]}"; do
        python scripts/train_baseline_classifier.py $COMMON \
            --run_dir "$RUN_DIR" \
            --run_name "enb0_initial_pure_genaug_noaug_sw${SW}_s${SEED}_ds${DATA_SEED}" \
            --gen_dir "$GEN_DIR" \
            --gen_frac "$GF" \
            --no_augmentation
    done
done
