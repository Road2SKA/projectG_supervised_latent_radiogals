#!/bin/bash
#SBATCH --job-name=noise_robustness
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
mkdir -p /users/mbredber/p3_SUPLAT/outputs/logs

echo "Running noise_robustness_classifiers.py — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"

python /users/mbredber/p3_SUPLAT/scripts/noise_robustness_classifiers.py \
    --outputs_root /users/mbredber/p3_SUPLAT/outputs \
    --run_glob "byol_runs/pd128_*_f1" \
    --out_dir /users/mbredber/p3_SUPLAT/outputs/noise_robustness \
    --sw_vals 0.0 0.05 0.1 0.5 1.0 10.0 \
    --label_sets initial_pure initial_binary \
    --feat_sigmas 0.0 0.25 0.5 1.0 2.0 \
    --label_noise 0.0 0.05 0.1 0.2 0.4 0.6 0.8 \
    --classifiers lr mlp

echo "Done — $(date)"
