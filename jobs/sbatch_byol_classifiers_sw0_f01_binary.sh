#!/bin/bash
#SBATCH --job-name=byol_clf_sw0_f01_bin
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT
mkdir -p outputs/logs

RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw0.0_f0.1"

echo "Starting BYOL classifiers (non-CV, initial_individual) — sw=0 f=0.1 — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"

python scripts/train_byol_classifiers.py \
    --outputs-root outputs/byol_runs/byol_runs \
    --run-glob     "${RUN_NAME}" \
    --feature-type projections \
    --label-set    initial_individual \
    --n-estimators 200 \
    --workers      8 \
    --seed         2 \
    --data-seed    2

echo ""
echo "Done — $(date)"
