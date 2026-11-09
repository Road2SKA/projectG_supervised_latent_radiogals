#!/bin/bash
#SBATCH --job-name=noise_robustness
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --partition=normal
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
python /users/mbredber/p3_SUPLAT/scripts/noise_robustness_classifiers.py \
    --outputs_root /users/mbredber/p3_SUPLAT/outputs \
    --run_glob "byol_runs/pd128_*_f1" \
    --out_dir /users/mbredber/p3_SUPLAT/outputs/noise_robustness \
    --sw_vals 0.0 0.1 1.0 10.0 \
    --label_sets initial_pure \
    --feat_sigmas 0.0 0.25 0.5 1.0 2.0 \
    --classifiers lr mlp \
    --force
