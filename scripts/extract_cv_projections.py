"""
Extract BYOL projections for CV folds using an existing trained model.

For runs where --cross-val was NOT used during training (e.g. sw=0.0_f0.1),
this script re-uses the trained byol_model_best.pt to produce the per-fold
projection files that run_downstream_classifiers.py --cv-fold K expects.

For sw=0.0, labels never affect BYOL training, so the single trained model
is equivalent to per-fold models (which would have converged identically).

Output per fold K:
  <run_dir>/cross_val_K/data/byol/labelled_train_projections.npy
  <run_dir>/cross_val_K/data/byol/test_projections.npy
  <run_dir>/cross_val_K/data/byol/unlabelled_train_projections.npy

Usage:
  python scripts/extract_cv_projections.py \
      --run-dir outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.0_f0.1/data_seed_2/training_seed_2 \
      --data-dir data/preprocessed/lotss \
      --splits-dir outputs/data_splits/2 \
      --f-label 0.1 \
      --augmentation quart_ext \
      [--folds 0 1 2 3 4]  (default: all 5)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from suplat.data.augmentations import get_augmentation
from suplat.data.data_samplers import UnlabelledBYOLDataset
from suplat.models.byol_models import BYOLEfficientNetB0
from suplat.trainer.trainer import extract_embeddings_from_loader, extract_raw_embeddings_from_loader

# ── args ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--run-dir",    required=True, type=Path,
                help="Path to training_seed_N dir (outputs written here)")
ap.add_argument("--checkpoint", default=None, type=Path,
                help="Override checkpoint path (default: run_dir/byol_model_best.pt). "
                     "Useful when reusing a sw=0.0 model for a different f-label run.")
ap.add_argument("--data-dir",  default="data/preprocessed/lotss", type=Path,
                help="Dir with images_filtered.npy and labels_filtered.npy")
ap.add_argument("--splits-dir", required=True, type=Path,
                help="Dir containing cross_val_K/ subdirs (outputs/data_splits/<seed>)")
ap.add_argument("--f-label",   default=0.1, type=float,
                help="Label fraction used when saving split indices (default: 0.1)")
ap.add_argument("--augmentation", default="quart_ext",
                help="Augmentation profile (default: quart_ext)")
ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                help="Which fold indices to process (default: 0 1 2 3 4)")
ap.add_argument("--batch-size", type=int, default=256)
ap.add_argument("--workers",    type=int, default=4)
ap.add_argument("--force",      action="store_true",
                help="Re-extract even if output files already exist")
args = ap.parse_args()

# ── device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── load data ─────────────────────────────────────────────────────────────────
images_path = args.data_dir / "images_filtered.npy"
labels_path = args.data_dir / "labels_filtered.npy"
print(f"Loading images from {images_path} ...")
images = np.load(images_path).astype(np.float32) / 255.0
labels = np.load(labels_path)
print(f"  images: {images.shape}  labels: {labels.shape}")

# ── augmentation (same as training) ──────────────────────────────────────────
aug = get_augmentation(args.augmentation)

# ── determine checkpoint mode ─────────────────────────────────────────────────
# Single checkpoint: trained without --cross-val (e.g. sw=0.0_f0.1)
# Per-fold checkpoints: trained with --cross-val (sw>0 runs); checkpoint lives
#   at run_dir/cross_val_K/byol_model_best.pt, loaded fresh for each fold.
_shared_ckpt_path = args.checkpoint if args.checkpoint else args.run_dir / "byol_model_best.pt"
_per_fold_mode = not _shared_ckpt_path.exists()

if _per_fold_mode:
    print(f"\nPer-fold checkpoint mode (no shared {_shared_ckpt_path.name} found).")
    model = None  # loaded per fold below
else:
    print(f"\nLoading model from {_shared_ckpt_path} ...")
    ckpt = torch.load(_shared_ckpt_path, map_location="cpu")
    cfg  = ckpt["config"]
    model_type     = cfg["model_type"]
    projection_dim = cfg["projection_dim"]
    hidden_dim     = cfg["hidden_dim"]
    projector_type = cfg.get("projector", "mlp")
    print(f"  model_type={model_type}  projection_dim={projection_dim}  "
          f"hidden_dim={hidden_dim}  projector={projector_type}")
    if model_type == "efficientnet-b0":
        model = BYOLEfficientNetB0(
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            bn_momentum=0.1,
            feature_compression_mode=projector_type,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. "
                         "Add the appropriate branch from train_byol.py.")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("  Model loaded OK.")

# ── f_label string for index filenames ───────────────────────────────────────
f_val = args.f_label
f_str = str(int(f_val)) if f_val == int(f_val) else str(f_val)


def _make_loader(img_subset):
    """DataLoader for a subset of images (no augmentation for x1)."""
    ds = UnlabelledBYOLDataset(img_subset, transform=aug)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.workers, pin_memory=(device.type == "cuda"))


# ── per-fold extraction ───────────────────────────────────────────────────────
for fold_idx in args.folds:
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}")
    print(f"{'='*60}")

    fold_dir    = args.run_dir / f"cross_val_{fold_idx}"
    byol_dir    = fold_dir / "data" / "byol"

    if _per_fold_mode:
        ckpt_path = fold_dir / "byol_model_best.pt"
        print(f"  Loading fold checkpoint: {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg  = ckpt["config"]
        model_type     = cfg["model_type"]
        projection_dim = cfg["projection_dim"]
        hidden_dim     = cfg["hidden_dim"]
        projector_type = cfg.get("projector", "mlp")
        print(f"  model_type={model_type}  projection_dim={projection_dim}  "
              f"hidden_dim={hidden_dim}  projector={projector_type}")
        if model_type == "efficientnet-b0":
            model = BYOLEfficientNetB0(
                projection_dim=projection_dim,
                hidden_dim=hidden_dim,
                bn_momentum=0.1,
                feature_compression_mode=projector_type,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}.")
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()
        print("  Model loaded OK.")
    splits_dir  = args.splits_dir / f"cross_val_{fold_idx}"

    out_lab  = byol_dir / "labelled_train_projections.npy"
    out_test = byol_dir / "test_projections.npy"
    out_unlab = byol_dir / "unlabelled_train_projections.npy"

    if not args.force and out_lab.exists() and out_test.exists():
        print(f"  Already done — skipping (use --force to re-extract).")
        # NOTE: this skip also bypasses encoding extraction (lines below).
        # If projections exist but encodings don't, use --force to regenerate.
        continue

    # Verify splits exist
    lab_idx_path   = splits_dir / f"labelled_train_idx_f{f_str}.npy"
    unlab_idx_path = splits_dir / f"unlabelled_train_idx_f{f_str}.npy"
    test_idx_path  = splits_dir / "test_idx.npy"
    for p in [lab_idx_path, unlab_idx_path, test_idx_path]:
        if not p.exists():
            print(f"  ERROR: missing split file: {p}")
            sys.exit(1)

    lab_idx   = np.load(lab_idx_path)
    unlab_idx = np.load(unlab_idx_path)
    test_idx  = np.load(test_idx_path)

    print(f"  labelled train: {len(lab_idx)}  "
          f"unlabelled train: {len(unlab_idx)}  "
          f"test: {len(test_idx)}")

    byol_dir.mkdir(parents=True, exist_ok=True)

    # ── labelled train projections ────────────────────────────────────────────
    print("  Extracting labelled train projections ...")
    lab_loader = _make_loader(images[lab_idx])
    lab_proj   = extract_embeddings_from_loader(model, lab_loader, model_type, device)
    np.save(out_lab, lab_proj)
    print(f"    Saved {lab_proj.shape} → {out_lab}")

    # ── test projections ──────────────────────────────────────────────────────
    print("  Extracting test projections ...")
    test_loader = _make_loader(images[test_idx])
    test_proj   = extract_embeddings_from_loader(model, test_loader, model_type, device)
    np.save(out_test, test_proj)
    print(f"    Saved {test_proj.shape} → {out_test}")

    # ── unlabelled train projections ──────────────────────────────────────────
    if len(unlab_idx) == 0:
        print("  Skipping unlabelled train projections (f_label=1.0, no unlabelled sources).")
    else:
        print("  Extracting unlabelled train projections ...")
        unlab_loader = _make_loader(images[unlab_idx])
        unlab_proj   = extract_embeddings_from_loader(model, unlab_loader, model_type, device)
        np.save(out_unlab, unlab_proj)
        print(f"    Saved {unlab_proj.shape} → {out_unlab}")

    # ── also save raw encodings (before projector) ────────────────────────────
    enc_lab_path   = byol_dir / "labelled_train_encodings.npy"
    enc_test_path  = byol_dir / "test_encodings.npy"
    enc_unlab_path = byol_dir / "unlabelled_train_encodings.npy"
    if args.force or not enc_lab_path.exists():
        print("  Extracting labelled train encodings ...")
        enc_lab = extract_raw_embeddings_from_loader(model, lab_loader, device)
        np.save(enc_lab_path, enc_lab)
        print(f"    Saved {enc_lab.shape} → {enc_lab_path}")
    if args.force or not enc_test_path.exists():
        print("  Extracting test encodings ...")
        enc_test = extract_raw_embeddings_from_loader(model, test_loader, device)
        np.save(enc_test_path, enc_test)
        print(f"    Saved {enc_test.shape} → {enc_test_path}")
    if len(unlab_idx) > 0 and (args.force or not enc_unlab_path.exists()):
        print("  Extracting unlabelled train encodings ...")
        enc_unlab = extract_raw_embeddings_from_loader(model, unlab_loader, device)
        np.save(enc_unlab_path, enc_unlab)
        print(f"    Saved {enc_unlab.shape} → {enc_unlab_path}")

print(f"\nDone.")
