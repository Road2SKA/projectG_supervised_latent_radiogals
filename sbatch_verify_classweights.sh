#!/bin/bash
#SBATCH --job-name=verify_classweights
#SBATCH --output=outputs/logs/verify_classweights-%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -e
cd /users/mbredber/p3_SUPLAT
source .venv/bin/activate

echo "===== Class-weight verification ====="
echo "Node: $(hostname)"

# 1. Import smoke test
echo ""
echo "--- 1. Import smoke test ---"
python -c "
from suplat.utils.class_weights import compute_class_weights, compute_sample_weights
print('OK: compute_class_weights and compute_sample_weights imported')
"

# 2. Unit tests (manual, no pytest)
echo ""
echo "--- 2. Manual unit tests ---"
python tests/test_class_weights.py 2>&1 || python -c "
import sys, types
# Run the test file as a script by importing and calling functions
sys.path.insert(0, 'src')
exec(open('tests/test_class_weights.py').read())
test_uniform_class_weights_mode_none()
test_uniform_class_weights_strength_zero()
test_uniform_sample_weights_mode_none()
test_uniform_sample_weights_strength_zero()
test_class_weights_upweights_rare_classes()
test_class_weights_score_raises()
test_class_weights_unknown_mode_raises()
test_sample_weights_pure_set_mean_one()
test_sample_weights_pure_set_equal_mass()
test_sample_weights_impure_raises()
test_sample_weights_score_upweights_rare_tiers()
print('All unit tests PASSED')
"

# 3. train_byol.py — score mode
echo ""
echo "--- 3. train_byol.py: score mode (5 epochs, subsample) ---"
python scripts/train_byol.py \
    --epochs 5 --subsample 500 \
    --class-weight-mode score --class-weight-strength 1.0 \
    --run-name verify_cw_score \
    --no-plot-umap --no-plot-history --no-metrics

# 4. train_byol.py — label-set mode
echo ""
echo "--- 4. train_byol.py: initial label-set mode (5 epochs, subsample) ---"
python scripts/train_byol.py \
    --epochs 5 --subsample 500 \
    --class-weight-mode initial --class-weight-strength 0.5 \
    --run-name verify_cw_initial \
    --no-plot-umap --no-plot-history --no-metrics

echo ""
echo "===== All verification checks completed ====="
