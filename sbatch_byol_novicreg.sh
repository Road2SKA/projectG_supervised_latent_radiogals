#!/bin/bash
#SBATCH --job-name=byol_novicreg
#SBATCH --output=outputs/logs/byol_novicreg-%j.out
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -e
cd /users/mbredber/p3_SUPLAT
source .venv/bin/activate

OUTDIR="outputs/byol_runs"
COMMON="--output-dir $OUTDIR --epochs 300 --weight-decay 1e-4 --lr-schedule constant --label-type full"
COMMON="$COMMON --vicreg-var-weight 0.0 --vicreg-cov-weight 0.0"

echo "===== No-VicReg BYOL sweep ====="
echo "Node: $(hostname)  GPUs: $(nvidia-smi --list-gpus | wc -l)"
echo "Started: $(date)"

# --- mlp_pd128 + closest (current architecture), sweep sw ---
for SW in 0.1 0.5 1.0; do
    NAME="enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar0_cov0_gamma1.0_f1_sw${SW}"
    echo ""
    echo ">>> $NAME"
    python scripts/train_byol.py $COMMON \
        --projector mlp --projection-dim 128 \
        --weighting closest \
        --supervision-weight $SW \
        --run-name "$NAME"
done

# --- none projector + ponderate (best early-June config) ---
NAME="enb0_none_pond_lrconst_wd1e-4_lfull_ema0.996_vicregvar0_cov0_gamma1.0_f1_sw10"
echo ""
echo ">>> $NAME"
python scripts/train_byol.py $COMMON \
    --projector none \
    --weighting ponderate \
    --supervision-weight 10.0 \
    --run-name "$NAME"

echo ""
echo "===== Done: $(date) ====="
