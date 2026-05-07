"""
mgcls_prep.py

Converts raw MGCLS enhanced FITS.gz files into 89x89 pixel .npy crops
ready for the SUPLAT BYOL encoder.

Pipeline per field:
  1. Decompress .fits.gz  → in-memory (no disk temp file)
  2. Read FITS data + WCS
  3. Compute per-field contrast-stretch percentiles (P2, P98)
  4. Tile the image into overlapping crops (same logic as mgcls_data_prep.py)
  5. Reject crops with >40% NaN pixels
  6. Apply contrast-stretch (rescale_intensity to [0, 1])
  7. Resize from 256x256 → 89x89 (anti-aliased bicubic)
  8. Save each crop as   processed/<field>_crop_<i>.npy
  9. Append one row per crop to metadata CSV

Usage:
  python mgcls_prep.py --fits_dir data/raw/mgcls_fits/5pln_cubes \
                       --output_dir data/processed/mgcls_20k \
                       --meta_csv data/metadata/mgcls_crops.csv

  # Dry-run a single field to test:
  python mgcls_prep.py --fits_dir data/raw/mgcls_fits/5pln_cubes \
                       --output_dir data/processed/mgcls_test \
                       --meta_csv /tmp/test_meta.csv \
                       --limit 1
"""

import os
import gzip
import glob
import shutil
import argparse
import tempfile

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from skimage.exposure import rescale_intensity
from skimage.transform import resize   # bicubic resize with anti-aliasing

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CROP_SHAPE   = (256, 256)   # native crop size (paper uses 256x256)
OUTPUT_SHAPE = (89, 89)     # target size for BYOL encoder
PERCENT_NAN  = 0.40         # reject crops with more than 40% NaN pixels
P_LOW        = 2            # contrast-stretch lower percentile
P_HIGH       = 98           # contrast-stretch upper percentile (MGCLS uses 98)

# ---------------------------------------------------------------------------
# Step 1: Decompress .fits.gz to a temporary file and open it
# ---------------------------------------------------------------------------

def open_fits_gz(path):
    """
    Decompresses a .fits.gz file into a temporary .fits file and opens it.

    Why a temp file and not in-memory?
    astropy's fits.open() needs a seekable file-like object. gzip.open()
    returns a streaming object that is not seekable, so we write to a
    NamedTemporaryFile first and then open that.

    Parameters
    ----------
    path : str
        Path to a .fits.gz file.

    Returns
    -------
    astropy.io.fits.HDUList
        Opened FITS object. Caller is responsible for closing it.
    str
        Path to the temp file (caller must delete after closing HDUList).
    """
    # NamedTemporaryFile creates a file that auto-cleans on close,
    # but we set delete=False so we control when it's removed.
    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    with gzip.open(path, "rb") as gz_in:
        shutil.copyfileobj(gz_in, tmp)   # stream copy, memory-efficient
    tmp.close()
    hdul = fits.open(tmp.name, memmap=False)
    return hdul, tmp.name


# ---------------------------------------------------------------------------
# Step 2: Extract 2D image array and WCS from HDUList
# ---------------------------------------------------------------------------

def extract_image(hdul):
    """
    Returns a 2D numpy array and its WCS from an MGCLS HDUList.

    MGCLS enhanced images are 5-plane cubes (5pln), so the raw array
    shape is (5, 1, H, W) or similar. We take plane 0 and squeeze
    all length-1 axes to get a clean 2D array.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList

    Returns
    -------
    wcs : astropy.wcs.WCS
    arr : np.ndarray, shape (H, W), dtype float32
    """
    header = hdul[0].header
    wcs    = WCS(header)
    arr    = hdul[0].data

    # Handle cube shapes: take first plane, squeeze extra dims
    if arr.ndim == 4:
        arr = arr[0, 0, :, :]
    elif arr.ndim == 3:
        arr = arr[0, :, :]
    # arr should now be 2D

    return wcs, arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3: Per-field contrast-stretch percentiles
# ---------------------------------------------------------------------------

def compute_stretch_params(arr):
    """
    Compute contrast-stretch parameters from the full field image.

    The paper scales by the P2 and P98 of the *entire field*, not each
    individual crop. This preserves relative brightness between crops
    from the same field (important near the primary beam edge).

    NaN pixels are excluded from the percentile calculation.

    Parameters
    ----------
    arr : np.ndarray, 2D

    Returns
    -------
    p_low, p_high : float
    """
    valid = arr[np.isfinite(arr)]   # exclude NaN and Inf
    p_low  = np.percentile(valid, P_LOW)
    p_high = np.percentile(valid, P_HIGH)
    return float(p_low), float(p_high)


# ---------------------------------------------------------------------------
# Step 4-5: Tile image into crops and reject NaN-heavy ones
# ---------------------------------------------------------------------------

def calc_stride(dim, crop_dim, n_crops):
    """
    Calculate stride so that n_crops fit in dim with minimal overlap.

    This mirrors calc_stride() in mgcls_data_prep.py.
    overlap = (n_crops * crop_dim - dim) / (n_crops - 1)
    stride  = crop_dim - overlap
    """
    ideal  = crop_dim * n_crops
    rest   = ideal - dim
    margin = rest // (n_crops - 1) if n_crops > 1 else 0
    return crop_dim - margin


def tile_image(wcs, arr, crop_shape=CROP_SHAPE):
    """
    Tiles the 2D field array into Cutout2D crops.

    Strategy (from the paper):
      - Find the largest integer number of crops that fits, then set
        stride to minimise overlap (at most ~10 pixels each side).
      - Reject crops where >PERCENT_NAN pixels are NaN.
      - Reject crops that don't have exactly crop_shape (edge crops).

    Parameters
    ----------
    wcs : astropy.wcs.WCS  (celestial, 2D)
    arr : np.ndarray       (H, W)
    crop_shape : tuple     (height, width)

    Returns
    -------
    list of np.ndarray   each of shape crop_shape, dtype float32
    """
    h, w = arr.shape
    ch, cw = crop_shape

    # Number of crops that fit; stride minimises overlap
    n_x = w // cw
    n_y = h // ch
    stride_x = calc_stride(w, cw, n_x + 1)
    stride_y = calc_stride(h, ch, n_y + 1)

    crops = []
    bad   = 0
    y = 0
    while y < h:
        x = 0
        while x < w:
            # Centre of this crop in pixel coords
            cx = x + cw / 2
            cy = y + ch / 2

            # Convert pixel centre to sky coords, then make cutout
            # (Cutout2D handles the WCS properly)
            try:
                sky_pos = wcs.celestial.pixel_to_world(cx, cy)
                cutout  = Cutout2D(arr, sky_pos, crop_shape,
                                   mode="partial", wcs=wcs.celestial)
            except Exception:
                x += stride_x
                continue

            data = cutout.data

            # Skip partial edge crops
            if data.shape != crop_shape:
                x += stride_x
                continue

            # Compute NaN fraction
            nan_mask  = ~np.isfinite(data)
            nan_frac  = nan_mask.sum() / data.size

            if nan_frac <= PERCENT_NAN:
                crops.append(data)
            else:
                bad += 1

            x += stride_x
        y += stride_y

    if bad > 0:
        print(f"    Rejected {bad} crops (>{PERCENT_NAN*100:.0f}% NaN)")
    return crops


# ---------------------------------------------------------------------------
# Step 6-7: Contrast-stretch then resize
# ---------------------------------------------------------------------------

def preprocess_crop(crop, p_low, p_high):
    """
    Apply contrast-stretch then resize a single crop.

    Steps:
      1. Replace NaNs with p_low (treat as background noise)
      2. Contrast-stretch to [0, 1] using field-level percentiles
      3. Resize 256x256 → 89x89 with bicubic interpolation + anti-aliasing

    Parameters
    ----------
    crop   : np.ndarray, shape (256, 256)
    p_low  : float  (P2 of the field)
    p_high : float  (P98 of the field)

    Returns
    -------
    np.ndarray, shape (89, 89), float32, values in [0, 1]
    """
    # Replace NaN with background level before stretching
    crop = np.where(np.isfinite(crop), crop, p_low)

    # Contrast-stretch: maps [p_low, p_high] → [0, 1]
    # Values outside this range are clipped to 0 or 1
    stretched = rescale_intensity(crop,
                                  in_range=(p_low, p_high),
                                  out_range=(0.0, 1.0))

    # Resize: order=3 is bicubic, anti_aliasing prevents aliasing artefacts
    # when downsampling (important here: 256 → 89 is a significant reduction)
    resized = resize(stretched, OUTPUT_SHAPE,
                     order=3,
                     anti_aliasing=True,
                     preserve_range=True)

    return resized.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 8-9: Process one field and write outputs
# ---------------------------------------------------------------------------

def process_field(fits_gz_path, output_dir, rows):
    """
    Full pipeline for one MGCLS field:
      open → extract → stretch params → tile → preprocess → save.

    Appends one dict per saved crop to `rows` (for metadata CSV).

    Parameters
    ----------
    fits_gz_path : str
    output_dir   : str
    rows         : list   (mutated in place)
    """
    field_name = os.path.basename(fits_gz_path).replace(".fits.gz", "")
    print(f"  Processing: {field_name}")

    # --- Open and extract ---
    try:
        hdul, tmp_path = open_fits_gz(fits_gz_path)
    except Exception as e:
        print(f"    ERROR opening {fits_gz_path}: {e}")
        return

    try:
        wcs, arr = extract_image(hdul)
    except Exception as e:
        print(f"    ERROR extracting image: {e}")
        hdul.close()
        os.remove(tmp_path)
        return
    finally:
        hdul.close()
        os.remove(tmp_path)   # clean up temp decompressed file

    # --- Per-field contrast-stretch params ---
    p_low, p_high = compute_stretch_params(arr)

    # --- Tile ---
    crops = tile_image(wcs, arr)
    if not crops:
        print(f"    WARNING: no valid crops for {field_name}")
        return
    print(f"    {len(crops)} valid crops")

    # --- Preprocess and save ---
    for i, crop in enumerate(crops):
        processed = preprocess_crop(crop, p_low, p_high)

        fname = f"{field_name}_crop_{i:04d}.npy"
        np.save(os.path.join(output_dir, fname), processed)

        # Metadata row: used later for source-count regression labels
        rows.append({
            "filename":   fname,
            "field":      field_name,
            "crop_index": i,
            "p_low":      p_low,
            "p_high":     p_high,
            # source_count filled in later by cross-matching with catalog
            "source_count": np.nan,
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fits_dir, output_dir, meta_csv, limit=None):
    """
    Process all .fits.gz files in fits_dir.

    Parameters
    ----------
    fits_dir   : str   directory containing .fits.gz files
    output_dir : str   where to save .npy crops
    meta_csv   : str   path to output metadata CSV
    limit      : int   if set, only process this many fields (for testing)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(meta_csv), exist_ok=True)

    # Glob for all fits.gz files; sort for reproducibility
    fits_files = sorted(glob.glob(os.path.join(fits_dir, "*.fits.gz")))
    if not fits_files:
        print(f"No .fits.gz files found in {fits_dir}")
        return

    if limit:
        fits_files = fits_files[:limit]

    print(f"Found {len(fits_files)} field(s) to process.\n")

    rows = []   # accumulates one dict per crop
    for i, path in enumerate(fits_files):
        print(f"[{i+1}/{len(fits_files)}]", end=" ")
        process_field(path, output_dir, rows)

    # Write metadata CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(meta_csv, index=False)
        print(f"\nSaved {len(rows)} crop entries → {meta_csv}")
    else:
        print("No crops saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MGCLS FITS.gz → 89x89 .npy crops")
    parser.add_argument("--fits_dir",
                        default="data/raw/mgcls_fits/5pln_cubes",
                        help="Dir with .fits.gz files "
                             "(default: data/raw/mgcls_fits/5pln_cubes)")
    parser.add_argument("--output_dir",
                        default="data/processed/mgcls_20k",
                        help="Where to save .npy crops "
                             "(default: data/processed/mgcls_20k)")
    parser.add_argument("--meta_csv",
                        default="data/metadata/mgcls_crops.csv",
                        help="Path for crop metadata CSV "
                             "(default: data/metadata/mgcls_crops.csv)")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Only process this many fields (for testing)")
    args = parser.parse_args()

    main(args.fits_dir, args.output_dir, args.meta_csv, args.limit)