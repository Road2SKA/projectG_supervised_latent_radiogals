#!/usr/bin/env python3
"""
download_mightee.py

Downloads MIGHTEE DR1 continuum images and source catalogues
needed for the SUPLAT project.

What gets downloaded:
    data/raw/mightee/
        MIGHTEE_Continuum_DR1_COSMOS_8p9arcsec_I_v1.1.fits     ← ~GB, COSMOS field
        MIGHTEE_Continuum_DR1_COSMOS_8p9arcsec_I_v1.1_FinalCatalogue.srl.fits
        MIGHTEE_Continuum_DR1_XMMLSS_8p9arcsec_I_v1.1.fits     ← ~GB, XMM-LSS field
        MIGHTEE_Continuum_DR1_XMMLSS_8p9arcsec_I_v1.1_FinalCatalogue.srl.fits

Why COSMOS and XMMLSS at 8.9":
    Lastufka et al. (2024) use COSMOS and XMMLSS from the MIGHTEE Early
    Science release at 8.6" and 8.2" resolution. The DR1 equivalent is
    the 8.9" product — the closest match in this release. CDFS-DEEP was
    not used in the paper and is skipped here.

Why the source catalogues:
    The .srl.fits catalogue is used to generate source-count labels for
    each crop (analogous to the MGCLS compact source catalogue), and
    to define crop centres for the MIGHTEE_FR labelled subset.

Usage:
    python download_mightee.py                        # default output dir
    python download_mightee.py /path/to/output/dir   # custom output dir
"""

import os
import sys
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL   = "https://archive-gw-1.kat.ac.za/public/repository/10.48479/7msw-r692/data"
OUTPUT_DIR = "data/raw/mightee"

# Files to download: (filename, description)
# Stokes-I total intensity images + source-level catalogues (.srl)
# for COSMOS and XMMLSS at 8.9" resolution.
FILES = [
    (
        "MIGHTEE_Continuum_DR1_COSMOS_8p9arcsec_I_v1.1.fits",
        "COSMOS field, total intensity, 8.9\"",
    ),
    (
        "MIGHTEE_Continuum_DR1_COSMOS_8p9arcsec_I_v1.1_FinalCatalogue.srl.fits",
        "COSMOS field, source catalogue, 8.9\"",
    ),
    (
        "MIGHTEE_Continuum_DR1_XMMLSS_8p9arcsec_I_v1.1.fits",
        "XMM-LSS field, total intensity, 8.9\"",
    ),
    (
        "MIGHTEE_Continuum_DR1_XMMLSS_8p9arcsec_I_v1.1_FinalCatalogue.srl.fits",
        "XMM-LSS field, source catalogue, 8.9\"",
    ),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ---------------------------------------------------------------------------
# Download function (same pattern as download_mgcls.py)
# ---------------------------------------------------------------------------

def download_file(url, local_path, label):
    """
    Downloads a file from url to local_path.
    Skips if already exists — safe to re-run after interruption.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        print(f"  SKIP (exists): {label}")
        return

    print(f"  Downloading: {label}")

    try:
        r = requests.get(url, headers=HEADERS, stream=True, timeout=120)
        r.raise_for_status()

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
    if len(sys.argv) > 1:
        OUTPUT_DIR = sys.argv[1]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")
    print(f"Downloading {len(FILES)} file(s) from MIGHTEE DR1...\n")

    for filename, description in FILES:
        url        = f"{BASE_URL}/{filename}"
        local_path = os.path.join(OUTPUT_DIR, filename)
        print(f"[{FILES.index((filename, description))+1}/{len(FILES)}] {description}")
        download_file(url, local_path, filename)

    print(f"\nAll done. Files saved to: {OUTPUT_DIR}")
    print("Re-run at any time to resume — existing files are skipped.")