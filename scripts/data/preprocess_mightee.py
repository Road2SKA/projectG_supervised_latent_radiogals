"""
preprocess_mightee.py

Converts MIGHTEE DR1 FITS images into 89x89 .npy crops for SUPLAT.

Two output datasets are produced:
  1. mightee      — unlabelled wide-field crops (for UMAP, like MGCLS)
  2. mightee_fr   — labelled FRI/FRII crops centred on catalogue sources
                    (for classifier evaluation) — requires a separate
                    hand-labelled CSV from Scaife (2023)

Pipeline for mightee (unlabelled):
    Per field (COSMOS, XMMLSS):
      1. Read FITS, extract 2D Stokes-I array + WCS
      2. Compute per-field P2/P99.9 contrast-stretch params
         (Lastufka et al. use P99.9 for MIGHTEE due to spurious bright pixels)
      3. Tile into 256x256 crops, reject >40% NaN
      4. Contrast-stretch → resize 256→89 → save .npy
      5. Append row to metadata CSV

Pipeline for mightee_fr (labelled):
    1. Load hand-labelled CSV (source RA/Dec + FRI/FRII label)
    2. For each source, make a 150x150 crop centred on its coordinates
    3. Contrast-stretch → resize 150→89 → save .npy
    4. Save labels CSV

Usage:
    # Unlabelled wide-field crops (mightee)
    python preprocess_mightee.py

    # Also produce labelled mightee_fr crops (requires label CSV)
    python preprocess_mightee.py --fr_labels path/to/mightee_fr_labels.csv
"""

import os
import argparse

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from skimage.exposure import rescale_intensity
from skimage.transform import resize

# ---------------------------------------------------------------------------
# Constants — match Lastufka et al. Table 1 and Section 2.4
# ---------------------------------------------------------------------------
CROP_SHAPE_MIGHTEE    = (256, 256)   # wide-field crops (same as MGCLS)
CROP_SHAPE_FR         = (150, 150)   # postage-stamp crops for MIGHTEE_FR
OUTPUT_SHAPE          = (89, 89)
PERCENT_NAN           = 0.40
P_LOW                 = 2
P_HIGH_MIGHTEE        = 99.9         # paper uses 99.9th for MIGHTEE (not 98th)
                                     # due to a few spurious bright pixels

# FITS files to process for the unlabelled dataset
MIGHTEE_FIELDS = [
    "MIGHTEE_Continuum_DR1_COSMOS_8p9arcsec_I_v1.1.fits",
    "MIGHTEE_Continuum_DR1_XMMLSS_8p9arcsec_I_v1.1.fits",
]

# ---------------------------------------------------------------------------
# Shared helpers (same logic as mgcls_prep.py)
# ---------------------------------------------------------------------------

def read_fits(fits_path):
    """
    Read a MIGHTEE FITS file and return (wcs, 2D array).

    MIGHTEE images may have shape (1,1,H,W) or (H,W) depending on version.
    We squeeze to 2D in all cases.
    """
    with fits.open(fits_path, memmap=True) as hdul:
        header = hdul[0].header
        wcs    = WCS(header)
        data   = hdul[0].data.squeeze().astype(np.float32)
    return wcs, data


def compute_stretch_params(arr, p_high=P_HIGH_MIGHTEE):
    """Per-field contrast-stretch percentiles, NaN-safe."""
    valid  = arr[np.isfinite(arr)]
    p_low  = np.percentile(valid, P_LOW)
    p_high = np.percentile(valid, p_high)
    return float(p_low), float(p_high)


def calc_stride(dim, crop_dim, n_crops):
    """Stride that fits n_crops with minimal overlap."""
    ideal  = crop_dim * n_crops
    margin = (ideal - dim) // max(n_crops - 1, 1)
    return crop_dim - margin


def preprocess_crop(crop, p_low, p_high):
    """
    Replace NaN → contrast-stretch → resize → float32.

    Parameters
    ----------
    crop   : np.ndarray, shape (H, W)
    p_low  : float
    p_high : float

    Returns
    -------
    np.ndarray, shape OUTPUT_SHAPE, dtype float32
    """
    crop = np.where(np.isfinite(crop), crop, p_low)
    stretched = rescale_intensity(crop,
                                  in_range=(p_low, p_high),
                                  out_range=(0.0, 1.0))
    resized = resize(stretched, OUTPUT_SHAPE,
                     order=3, anti_aliasing=True, preserve_range=True)
    return resized.astype(np.float32)


# ---------------------------------------------------------------------------
# Unlabelled wide-field crops (mightee dataset)
# ---------------------------------------------------------------------------

def tile_field(wcs, arr, crop_shape):
    """
    Tile a field into crops, rejecting those with >PERCENT_NAN NaN pixels.
    Same strategy as mgcls_prep.py.
    """
    h, w  = arr.shape
    ch, cw = crop_shape
    n_x   = w // cw
    n_y   = h // ch
    sx    = calc_stride(w, cw, n_x + 1)
    sy    = calc_stride(h, ch, n_y + 1)

    crops = []
    bad   = 0
    y = 0
    while y < h:
        x = 0
        while x < w:
            cx, cy = x + cw / 2, y + ch / 2
            try:
                sky = wcs.celestial.pixel_to_world(cx, cy)
                cut = Cutout2D(arr, sky, crop_shape,
                               mode="partial", wcs=wcs.celestial)
            except Exception:
                x += sx
                continue

            data = cut.data
            if data.shape != crop_shape:
                x += sx
                continue

            nan_frac = (~np.isfinite(data)).sum() / data.size
            if nan_frac <= PERCENT_NAN:
                crops.append(data)
            else:
                bad += 1
            x += sx
        y += sy

    print(f"    {bad} crops rejected (>{PERCENT_NAN*100:.0f}% NaN)")
    return crops


def process_mightee_field(fits_path, output_dir, rows):
    """Process one MIGHTEE field into unlabelled 89x89 crops."""
    field_name = os.path.basename(fits_path).replace(".fits", "")
    print(f"  Processing: {field_name}")

    try:
        wcs, arr = read_fits(fits_path)
    except Exception as e:
        print(f"    ERROR reading FITS: {e}")
        return

    p_low, p_high = compute_stretch_params(arr, p_high=P_HIGH_MIGHTEE)
    crops         = tile_field(wcs, arr, CROP_SHAPE_MIGHTEE)
    print(f"    {len(crops)} valid crops")

    for i, crop in enumerate(crops):
        processed = preprocess_crop(crop, p_low, p_high)
        fname     = f"{field_name}_crop_{i:04d}.npy"
        np.save(os.path.join(output_dir, fname), processed)
        rows.append({
            "filename": fname,
            "field":    field_name,
            "p_low":    p_low,
            "p_high":   p_high,
        })


def run_mightee(input_dir, output_dir, meta_csv):
    """Produce the unlabelled mightee dataset."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(meta_csv), exist_ok=True)

    rows = []
    for fname in MIGHTEE_FIELDS:
        fits_path = os.path.join(input_dir, fname)
        if not os.path.exists(fits_path):
            print(f"  WARNING: {fname} not found — skipping")
            continue
        process_mightee_field(fits_path, output_dir, rows)

    df = pd.DataFrame(rows)
    df.to_csv(meta_csv, index=False)
    print(f"\nmightee: {len(rows)} crops saved → {meta_csv}")


# ---------------------------------------------------------------------------
# Labelled postage-stamp crops (mightee_fr dataset)
# ---------------------------------------------------------------------------

def process_mightee_fr(input_dir, fr_labels_csv, output_dir, labels_csv):
    """
    Produce the labelled mightee_fr dataset.

    The hand-labelled CSV (Scaife 2023) must have at minimum:
        RA, DEC, label   (label: 0=FRI, 1=FRII)

    For each source, we find the COSMOS or XMMLSS field that covers its
    coordinates, extract a 150x150 crop centred on (RA, DEC), then
    preprocess and save.
    """
    if not os.path.exists(fr_labels_csv):
        print(f"  mightee_fr labels CSV not found: {fr_labels_csv}")
        print("  Skipping mightee_fr — obtain from Scaife (2023).")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(labels_csv), exist_ok=True)

    label_df = pd.read_csv(fr_labels_csv)
    print(f"  {len(label_df)} labelled sources")

    # Load both field images once
    fields = {}
    for fname in MIGHTEE_FIELDS:
        fpath = os.path.join(input_dir, fname)
        if os.path.exists(fpath):
            field_key = "COSMOS" if "COSMOS" in fname else "XMMLSS"
            print(f"  Loading {field_key}...")
            fields[field_key] = read_fits(fpath)

    rows = []
    for i, row in label_df.iterrows():
        ra, dec   = row["RA"], row["DEC"]
        label     = int(row["label"])

        # Try each field — use the one whose WCS covers this position
        saved = False
        for field_key, (wcs, arr) in fields.items():
            p_low, p_high = compute_stretch_params(arr, p_high=P_HIGH_MIGHTEE)
            try:
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                sky = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                cut = Cutout2D(arr, sky, CROP_SHAPE_FR,
                               mode="strict", wcs=wcs.celestial)
            except Exception:
                continue   # source not in this field

            processed = preprocess_crop(cut.data, p_low, p_high)
            fname_out = f"fr_{i:04d}.npy"
            np.save(os.path.join(output_dir, fname_out), processed)
            rows.append({
                "filename":  fname_out,
                "split":     "train",     # MIGHTEE_FR has no canonical split
                "label":     label,
                "label_str": "FRI" if label == 0 else "FRII",
                "field":     field_key,
                "ra":        ra,
                "dec":       dec,
            })
            saved = True
            break

        if not saved:
            print(f"    WARNING: source {i} (RA={ra}, DEC={dec}) "
                  f"not found in any field — skipping")

    df = pd.DataFrame(rows)
    df.to_csv(labels_csv, index=False)
    fri_n  = (df["label"] == 0).sum()
    frii_n = (df["label"] == 1).sum()
    print(f"\nmightee_fr: {len(rows)} crops saved "
          f"(FRI={fri_n}, FRII={frii_n}) → {labels_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir, meta_csv,
         fr_labels_csv, fr_output_dir, fr_labels_out):

    print("=== MIGHTEE unlabelled crops ===")
    run_mightee(input_dir, output_dir, meta_csv)

    if fr_labels_csv:
        print("\n=== MIGHTEE_FR labelled crops ===")
        process_mightee_fr(input_dir, fr_labels_csv,
                           fr_output_dir, fr_labels_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess MIGHTEE DR1 FITS → 89x89 .npy crops"
    )
    parser.add_argument("--input_dir",
                        default="data/raw/mightee",
                        help="Dir containing MIGHTEE .fits files "
                             "(default: data/raw/mightee)")
    parser.add_argument("--output_dir",
                        default="data/preprocessed/mightee",
                        help="Output dir for unlabelled crops "
                             "(default: data/preprocessed/mightee)")
    parser.add_argument("--meta_csv",
                        default="data/metadata/mightee_crops.csv",
                        help="Metadata CSV for unlabelled crops "
                             "(default: data/metadata/mightee_crops.csv)")
    parser.add_argument("--fr_labels",
                        default=None,
                        dest="fr_labels_csv",
                        help="Path to hand-labelled MIGHTEE_FR CSV "
                             "(Scaife 2023). If omitted, mightee_fr "
                             "is skipped.")
    parser.add_argument("--fr_output_dir",
                        default="data/preprocessed/mightee_fr",
                        help="Output dir for labelled FR crops "
                             "(default: data/preprocessed/mightee_fr)")
    parser.add_argument("--fr_labels_out",
                        default="data/metadata/mightee_fr_labels.csv",
                        help="Labels CSV for FR crops "
                             "(default: data/metadata/mightee_fr_labels.csv)")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.meta_csv,
         args.fr_labels_csv, args.fr_output_dir, args.fr_labels_out)