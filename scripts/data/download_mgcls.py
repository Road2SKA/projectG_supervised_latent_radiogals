#!/usr/bin/env python3
"""
download_mgcls.py

Downloads all .fits.gz images AND the compact source catalogue from the
MGCLS DR1 archive.

What gets downloaded:
  <OUTPUT_DIR>/
    5pln_cubes/                       ← one Stokes-I Farcsec cube per field
      Abell_133_...fits.gz
      ...

  Only 5pln_cubes/ is downloaded — not the 15arcsec/7arcsec
  multi-polarisation products in the same archive.
    Table2_MGCLS_compactcat_DR1.fits  ← ~626k compact sources, needed for
                                         source-count labels in mgcls_prep.py

How the image listing works:
  The archive is an S3-compatible store. We query its listing API:
      GET https://archive-gw-1.kat.ac.za/public/?prefix=<path>
  This returns XML with <Contents> entries. If <IsTruncated>true</IsTruncated>,
  there are more pages — we follow <NextMarker> to get the next batch.

The catalogue is a single direct URL — no listing needed.
"""

import os
import sys
import requests
import xml.etree.ElementTree as ET  # Standard library XML parser

# ---------------------------------------------------------------------------
# Configuration — edit these if needed
# ---------------------------------------------------------------------------
BASE_URL    = "https://archive-gw-1.kat.ac.za"
BUCKET      = "public"
KEY_PREFIX  = "repository/10.48479/7epd-w356/data/enhanced_products/5pln_cubes/"
OUTPUT_DIR  = "data/raw/mgcls_fits"

# Direct URL for the compact source catalogue (single file, no listing needed)
CATALOG_URL  = (
    "https://archive-gw-1.kat.ac.za/public/repository/"
    "10.48479/7epd-w356/data/Table2_MGCLS_compactcat_DR1.fits"
)
CATALOG_NAME = "Table2_MGCLS_compactcat_DR1.fits"

# S3 XML namespace (S3 wraps all tags in this namespace)
NS = "http://s3.amazonaws.com/doc/2006-03-01/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ---------------------------------------------------------------------------
# Step 1: Collect all image file keys via paginated S3 XML listing
# ---------------------------------------------------------------------------
def get_all_keys(prefix):
    """
    Queries the S3 listing API repeatedly until all pages are retrieved.
    Returns a list of all object keys (file paths) under the given prefix.

    S3 pagination works like this:
      - Each response may contain <IsTruncated>true</IsTruncated>
      - If truncated, <NextMarker> gives the key to start the next page from
      - We pass that as ?marker=<NextMarker> in the next request
    """
    keys = []
    marker = None   # No marker on first request
    page = 0

    while True:
        page += 1

        params = {"prefix": prefix}
        if marker:
            params["marker"] = marker

        url = f"{BASE_URL}/{BUCKET}/"
        print(f"  Fetching listing page {page} ...", end=" ", flush=True)

        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        print(f"HTTP {r.status_code}")

        if r.status_code != 200:
            print(f"ERROR: Unexpected status {r.status_code}")
            print(r.text[:500])
            break

        root = ET.fromstring(r.text)

        for content in root.findall(f"{{{NS}}}Contents"):
            key = content.find(f"{{{NS}}}Key").text
            # Keep only Stokes-I at full (Farcsec) resolution — the product
            # used by Lastufka et al. Each field has 8 files in 5pln_cubes/:
            #   pol_I_15arcsec, pol_I_Farcsec   ← I at two resolutions
            #   pol_Q/U/V_15arcsec, pol_Q/U/V_Farcsec  ← other Stokes params
            # We want pol_I_Farcsec only (~115 files, one per cluster field).
            if key.endswith(".fits.gz") and "pol_I_Farcsec" in key:
                keys.append(key)

        is_truncated = root.find(f"{{{NS}}}IsTruncated").text.lower()
        if is_truncated == "true":
            marker_el = root.find(f"{{{NS}}}NextMarker")
            marker = marker_el.text if marker_el is not None else (keys[-1] if keys else None)
        else:
            break

    return keys

# ---------------------------------------------------------------------------
# Step 2: Download a single file — works for both image keys and direct URLs
# ---------------------------------------------------------------------------
def download_file(url, local_path, label):
    """
    Downloads a file from url to local_path.
    Skips if the file already exists (safe to re-run after interruption).

    Parameters
    ----------
    url        : str   full download URL
    local_path : str   destination path on disk
    label      : str   short description printed in progress output
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        print(f"  SKIP (exists): {label}")
        return

    print(f"  Downloading: {label}")

    try:
        r = requests.get(url, headers=HEADERS, stream=True, timeout=60)
        r.raise_for_status()  # raises on 4xx/5xx

        total_mb   = int(r.headers.get("Content-Length", 0)) / 1e6
        chunk_size = 1024 * 1024   # 1 MB chunks
        downloaded = 0

        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                pct = downloaded / (total_mb * 1e6) * 100 if total_mb else 0
                print(f"    {downloaded/1e6:.1f}/{total_mb:.1f} MB ({pct:.0f}%)",
                      end="\r", flush=True)

        print(f"    Done: {downloaded/1e6:.1f} MB                    ")

    except Exception as e:
        print(f"    ERROR: {e} — removing partial file.")
        if os.path.exists(local_path):
            os.remove(local_path)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Allow overriding output dir from command line:
    #   python download_mgcls.py /path/to/output
    if len(sys.argv) > 1:
        OUTPUT_DIR = sys.argv[1]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # --- Part A: Download compact source catalogue (single file) ---
    print("=== Compact source catalogue ===")
    catalog_path = os.path.join(OUTPUT_DIR, CATALOG_NAME)
    download_file(CATALOG_URL, catalog_path, CATALOG_NAME)

    # --- Part B: Download enhanced image FITS.gz files ---
    print("\n=== Enhanced image cubes ===")
    print("Fetching file listing...")
    keys = get_all_keys(KEY_PREFIX)
    print(f"\nFound {len(keys)} .fits.gz files.\n")

    if not keys:
        print("No image files found — check the prefix or your connection.")
        sys.exit(1)

    for i, key in enumerate(keys, start=1):
        # key = "repository/.../5pln_cubes/Abell_133_...fits.gz"
        # rel_path strips the prefix → "5pln_cubes/Abell_133_...fits.gz"
        rel_path   = key.replace(KEY_PREFIX, "")
        local_path = os.path.join(OUTPUT_DIR, rel_path)
        file_url   = f"{BASE_URL}/{BUCKET}/{key}"
        print(f"[{i}/{len(keys)}]", end=" ")
        download_file(file_url, local_path, rel_path)

    print(f"\nAll done. Files saved to: {OUTPUT_DIR}")
    print("If the script was interrupted, re-run to resume — existing files are skipped.")

    # --- Part C: Remove unwanted files from a previous broad download ---
    # If you previously ran this script before the pol_I_Farcsec filter was
    # added, use --cleanup to delete the Q/U/V and 15arcsec files.
    if "--cleanup" in sys.argv:
        cube_dir = os.path.join(OUTPUT_DIR, "5pln_cubes")
        removed  = 0
        if os.path.isdir(cube_dir):
            for fname in os.listdir(cube_dir):
                if not fname.endswith(".fits.gz"):
                    continue
                # Delete anything that is NOT the Stokes-I Farcsec product
                if "pol_I_Farcsec" not in fname:
                    fpath = os.path.join(cube_dir, fname)
                    os.remove(fpath)
                    print(f"  REMOVED: {fname}")
                    removed += 1
        print(f"\nCleanup done: {removed} files removed.")