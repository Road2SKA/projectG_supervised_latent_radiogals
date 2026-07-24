#!/bin/bash
#SBATCH --job-name=rbyol_classifiers
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"

mkdir -p outputs/logs

# =============================================================================
# CONFIGURATION — keep BYOL_RUN_DIR in sync with run_classifiers.sh so both
# scripts evaluate on the same train/test split.
# Set to "" to sweep ALL matching run directories instead.
# =============================================================================
BYOL_RUN_DIR=""
# =============================================================================

if [ -n "$BYOL_RUN_DIR" ]; then
    RUN_GLOB="$(basename "$BYOL_RUN_DIR")"
else
    RUN_GLOB="enb0_*"
fi

echo "Starting BYOL classifiers sweep (all CW variants, --force) — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "Run glob: ${RUN_GLOB}"

# ── Class weighting variants to (re)train ─────────────────────────────────────
# Each entry: "label_set mode strength"  ("None" mode = cwNone)
VARIANTS=(
    "full            None          1.0"  # cwNone
    "full            all           0.3"  # cwall0.3
    "full            all           1.0"  # cwall
    "initial         None          1.0"  # cwNone
    "initial         initial       0.3"  # cwinitial
    "initial         initial       1.0"  # cwinitial_pure
    "initial_pure    None          0.0"  # cwNone
    "initial_pure    initial_pure  0.3"  # cwinitial_pure0.3
    "initial_pure    initial_pure  1.0"  # cwinitial_pure
    "initial_binary  None          0.0"  # cwNone, element-wise accuracy
    "initial_binary  initial_binary  0.3"  # cwinitial_binary0.3
    "initial_binary  initial_binary  1.0"  # cwinitial_binary
)

for VARIANT in "${VARIANTS[@]}"; do
    LS=$(echo      "$VARIANT" | awk '{print $1}')
    CW_MODE=$(echo "$VARIANT" | awk '{print $2}')
    CW_STR=$(echo  "$VARIANT" | awk '{print $3}')
    echo ""
    echo "── label_set=${LS}  cw_mode=${CW_MODE}  strength=${CW_STR} ──────────────────────"
    python scripts/train_byol_classifiers.py \
        --outputs-root          outputs/byol_runs \
        --run-glob              "${RUN_GLOB}" \
        --feature-type          projections \
        --label-set             "${LS}" \
        --n-estimators          200 \
        --workers               8 \
        --class-weight-mode     "${CW_MODE}" \
        --class-weight-strength "${CW_STR}"
done

echo ""
echo "Done — $(date)"
