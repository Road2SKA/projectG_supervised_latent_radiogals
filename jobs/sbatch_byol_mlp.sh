#!/bin/bash
#SBATCH --job-name=byol_mlp
#SBATCH --array=0-3
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"

mkdir -p outputs/logs

# Seeds 3-6 with data_seed=2 (off-diagonal dirs: data_seed_2/training_seed_N).
# run-glob matches all sw values for f=1 in one shot.
SEEDS=(3 4 5 6)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Starting BYOL MLP classifier — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "seed=${SEED}  data_seed=2  label_set=initial_pure"

python scripts/run_downstream_mlp_classifier.py \
    --outputs-root outputs/byol_runs/byol_runs \
    --run-glob     "pd128_*_f1" \
    --feature-type projections \
    --label-set    initial_pure \
    --hidden-layers 256,128 \
    --max-iter     500 \
    --lr           1e-3 \
    --seed         "${SEED}" \
    --data-seed    2

echo "Done — $(date)"
