#!/bin/bash
#SBATCH --job-name=noise_robustness
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/noise_robustness_%j.out

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
python /users/mbredber/p3_SUPLAT/scripts/noise_robustness_classifiers.py \
    --outputs_root /users/mbredber/p3_SUPLAT/outputs \
    --run_glob "byol_runs/pd128_*_f1" \
    --out_dir /users/mbredber/p3_SUPLAT/outputs/noise_robustness \
    --sw_vals 0.0 0.1 1.0 10.0 \
    --label_sets initial_pure initial_binary \
    --feat_sigmas 0.0 0.25 0.5 1.0 2.0 \
    --label_noise 0.0 0.05 0.1 0.2 0.4 \
    --classifiers lr mlp
