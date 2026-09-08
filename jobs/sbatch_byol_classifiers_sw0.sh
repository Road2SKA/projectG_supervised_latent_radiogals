#!/bin/bash
#SBATCH --job-name=byol_clf_sw0
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT
mkdir -p outputs/logs

RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw0.0_f1"

echo "Starting BYOL downstream classifiers — sw=0 f=1 — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"

for FOLD in 0 1 2 3 4; do
    echo ""
    echo "--- fold ${FOLD} ---"
    for LS in initial_pure initial_individual; do
        echo "  ${LS}"
        python scripts/train_byol_classifiers.py \
            --outputs-root outputs/byol_runs/byol_runs \
            --run-glob     "${RUN_NAME}" \
            --feature-type projections \
            --label-set    "${LS}" \
            --n-estimators 200 \
            --workers      8 \
            --seed         2 \
            --data-seed    2 \
            --cv-fold      "${FOLD}"
    done
done

echo ""
echo "Done — $(date)"
