#!/bin/bash
#SBATCH --job-name=data_prep
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:05:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

# =============================================================================
# DATA DIRECTORY SETUP (run once after cloning)
# =============================================================================
# Processed crops, metadata CSVs, and the lotss arrays are committed to the
# repo — no setup needed for those. Only raw FITS files live on scratch and
# are accessed via a symlink.
#
# To set up the raw symlink on a new machine:
#
#   SCRATCH=/capstor/scratch/cscs/$USER   # adjust to your scratch path
#   REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
#
#   mkdir -p "$SCRATCH/p3_SUPLAT_data_raw"
#   ln -s "$SCRATCH/p3_SUPLAT_data_raw" "$REPO_ROOT/data/raw"
#
# Verify with:
#   ls -la "$REPO_ROOT/data/"
# =============================================================================

echo "START: $(date)"
REPO_ROOT=$SLURM_SUBMIT_DIR
cd "$REPO_ROOT"

source .venv/bin/activate

# 1. Download + preprocess MGCLS (skip if already done)
if [ -f data/metadata/mgcls_crops.csv ]; then
    echo "=== MGCLS already processed — skipping ==="
else
    echo "=== Downloading MGCLS data ==="
    python scripts/data/download_mgcls.py data/raw/mgcls_fits

    echo "=== Preprocessing MGCLS data ==="
    python scripts/data/preprocess_mgcls.py \
        --fits_dir   data/raw/mgcls_fits \
        --output_dir data/preprocessed/mgcls_20k \
        --meta_csv   data/metadata/mgcls_crops.csv
fi

# 2. Download + preprocess MiraBest (skip if already done)
if [ -f data/metadata/mirabest_labels.csv ]; then
    echo "=== MiraBest already processed — skipping ==="
else
    echo "=== Downloading MiraBest data ==="
    python scripts/data/download_mirabest.py

    echo "=== Preprocessing MiraBest data ==="
    python scripts/data/preprocess_mirabest.py \
        --root       data/raw/mirabest \
        --output_dir data/preprocessed/mirabest \
        --labels_csv data/metadata/mirabest_labels.csv
fi

# 3. Preprocess RadioGalaxyDataset (FIRST) — requires galaxy_data_h5.h5 to be present
if [ -f data/metadata/first_labels.csv ]; then
    echo "=== RadioGalaxyDataset already processed — skipping ==="
else
    echo "=== Preprocessing RadioGalaxyDataset (FIRST) ==="
    python scripts/data/preprocess_first.py \
        --h5_path    data/raw/RadioGalaxyDataset/firstgalaxydata/galaxy_data_h5.h5 \
        --output_dir data/preprocessed/first \
        --labels_csv data/metadata/first_labels.csv
fi

# 4. Download + preprocess MIGHTEE (skip if already done)
if [ -f data/metadata/mightee_crops.csv ] && [ -n "$(ls -A data/preprocessed/mightee 2>/dev/null)" ]; then
    echo "=== MIGHTEE already processed — skipping ==="
else
    echo "=== Downloading MIGHTEE data ==="
    python scripts/data/download_mightee.py data/raw/mightee_fits

    echo "=== Preprocessing MIGHTEE data ==="
    python scripts/data/preprocess_mightee.py \
        --input_dir  data/raw/mightee_fits \
        --output_dir data/preprocessed/mightee \
        --meta_csv   data/metadata/mightee_crops.csv
fi

# 5. Create MGCLS-5k subsample (skip if already done)
if [ -f data/metadata/mgcls_5k_crops.csv ]; then
    echo "=== MGCLS-5k already created — skipping ==="
else
    echo "=== Creating MGCLS-5k subsample ==="
    python scripts/data/subsample_mgcls_5k.py \
        --meta_csv   data/metadata/mgcls_crops.csv \
        --output_csv data/metadata/mgcls_5k_crops.csv \
        --source_dir data/preprocessed/mgcls_20k \
        --output_dir data/preprocessed/mgcls_5k
fi

echo "END: $(date)"
