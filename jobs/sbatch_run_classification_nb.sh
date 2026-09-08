#!/bin/bash
#SBATCH --job-name=classification_nb
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT/notebooks
mkdir -p /users/mbredber/p3_SUPLAT/outputs/logs

echo "Running classification.ipynb — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"

python -m jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=3600 \
    classification.ipynb

echo "Done — $(date)"
