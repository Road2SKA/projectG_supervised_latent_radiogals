#!/bin/bash
#SBATCH --job-name=hpsweep_aggregate
#SBATCH --account=sk036
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"
cd /users/mbredber/p3_SUPLAT
source /users/mbredber/p3_SUPLAT/.venv/bin/activate

python scripts/collate_hpsweep.py \
    --mode aggregate \
    --sweep-root outputs/hyperparameter_tuning/byol_runs

echo "END: $(date)"
