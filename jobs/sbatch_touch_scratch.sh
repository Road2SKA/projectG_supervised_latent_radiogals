#!/bin/bash
#SBATCH --job-name=touch_scratch
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=02:00:00
#SBATCH --account=sk036

# Touches all files in the p3_SUPLAT scratch directory to reset the
# auto-purge timer, then resubmits itself to run again in 13 days.
# Submit once with: sbatch jobs/sbatch_touch_scratch.sh

set -euo pipefail

SCRATCH=/capstor/scratch/cscs/mbredber/p3_SUPLAT
SELF=/users/mbredber/p3_SUPLAT/jobs/sbatch_touch_scratch.sh

echo "Touching scratch files — $(date)"
find "${SCRATCH}" -exec touch {} +
echo "Done touching — $(date)"

echo "Rescheduling in 13 days..."
sbatch --begin=now+13days "${SELF}"
echo "Resubmitted OK."
