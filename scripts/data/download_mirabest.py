"""Download MiraBest dataset using MiraBest_full auto-download."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from MiraBest import MiraBest_full

ROOT = "data/raw/mirabest"
print(f"Downloading MiraBest to {ROOT} ...")
MiraBest_full(root=ROOT, download=True)
print("Done.")
