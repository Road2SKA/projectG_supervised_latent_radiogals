#!/bin/bash
#SBATCH --job-name=noise_rob_image_label
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
mkdir -p /users/mbredber/p3_SUPLAT/outputs/logs

echo "Running noise_robustness_image_label.py — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"

python /users/mbredber/p3_SUPLAT/scripts/noise_robustness_image_label.py \
    --data_dir     /users/mbredber/p3_SUPLAT/data/preprocessed/lotss/ \
    --outputs_root /users/mbredber/p3_SUPLAT/outputs \
    --out_dir      /users/mbredber/p3_SUPLAT/outputs/noise_robustness \
    --label_set    initial_pure \
    --sw_vals      0.0 0.05 0.1 \
    --img_sigmas   0.0 0.1 0.25 0.5 1.0 2.0 \
    --blur_sigmas  0.0 1.0 2.0 4.0 6.0 10.0 \
    --occlusion_sizes 0.0 10.0 20.0 30.0 45.0 60.0 \
    --data_seed    2 \
    --training_seed 2 \
    --batch_size   256 \
    --seed         42 \
    --force

echo "Done — $(date)"
