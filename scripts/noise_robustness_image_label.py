"""Noise robustness sweep — image and resolution noise for deep models.

Three model families:
  - byol_mlp   : sklearn MLP on BYOL projections (re-extracted from corrupted images)
  - finetuned  : full PyTorch finetuned model (lineareval_byol_sw_cv5_sw{sw}_fold{k})
  - supervised : full PyTorch supervised model (retrain_sup_noes_cv5_fold{k})

Image and resolution noise both follow a "train clean / test noisy"
paradigm: the model/classifier is fit once on clean train data, then
evaluated against a corrupted test set (noisy images or blurred images).
Resolution noise simulates PSF/angular-resolution degradation via Gaussian
blur of the image (kernel std in pixels), rather than additive per-pixel noise.

Outputs
-------
  <out_dir>/image_noise/{model_type}_sw{sw}_fold{k}_sigma{s}.json
  <out_dir>/resolution_noise/{model_type}_sw{sw}_fold{k}_blur{b}.json
  <out_dir>/occlusion/{model_type}_sw{sw}_fold{k}_patch{p}.json
  For supervised (no sw): sw value is "none" in filename.

Each JSON: model_type, sw, fold, label_set, noise_type, noise_level,
           f1_macro, f1_per_class, accuracy, class_names.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_downstream_mlp_es import build_mlp, train_mlp_es, predict_mlp

from suplat.data.label_sets import ALL_CLASS_NAMES, LABEL_SETS, apply_label_set
from suplat.models.byol_models import BYOLEfficientNetB0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred, class_names):
    return {
        "f1_macro":     float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "accuracy":     float(accuracy_score(y_true, y_pred)),
        "class_names":  class_names,
    }


def _blur_images(images_arr, sigma_px):
    """Gaussian-blur each (H, W) image independently; sigma_px in pixels."""
    if sigma_px == 0.0:
        return images_arr
    return np.stack([
        gaussian_filter(im, sigma=sigma_px) for im in images_arr
    ]).astype(np.float32)


def _occlude_images(images_arr, patch_px, border_px=10):
    """Replace a centered (patch_px, patch_px) square with each image's own
    local background level (median of its outer `border_px`-wide ring).

    Works on either the [0, 1] or [0, 255] scale since the median is
    scale-equivariant — call with whichever array matches the model's
    expected input range.
    """
    if patch_px <= 0:
        return images_arr
    n, h, w = images_arr.shape
    patch_px = int(patch_px)
    border_mask = np.ones((h, w), dtype=bool)
    border_mask[border_px:h - border_px, border_px:w - border_px] = False
    y0 = max((h - patch_px) // 2, 0)
    x0 = max((w - patch_px) // 2, 0)
    y1 = min(y0 + patch_px, h)
    x1 = min(x0 + patch_px, w)
    result = images_arr.copy()
    for i in range(n):
        bg = np.median(images_arr[i][border_mask])
        result[i, y0:y1, x0:x1] = bg
    return result.astype(np.float32)


def _images_to_tensor(images_arr, device):
    """(N, H, W) float32 → (N, 1, H, W) torch tensor on device."""
    t = torch.from_numpy(images_arr[:, None, :, :].astype(np.float32))
    return t.to(device)


@torch.no_grad()
def _extract_projections(byol_model, images_arr, device, batch_size=256):
    """Run BYOL online_encoder + online_projector over images_arr.

    Args:
        images_arr: (N, H, W) float32 numpy array in [0, 1].
    Returns:
        (N, proj_dim) float32 numpy array.
    """
    byol_model.eval()
    byol_model.to(device)
    results = []
    n = len(images_arr)
    for start in range(0, n, batch_size):
        batch = images_arr[start:start + batch_size]
        t = _images_to_tensor(batch, device)
        enc  = byol_model.online_encoder(t)   # (B, 1280)
        proj = byol_model.online_projector(enc)  # (B, 128)
        results.append(proj.cpu().numpy())
    return np.concatenate(results, axis=0).astype(np.float32)


@torch.no_grad()
def _run_model_logits(model, images_arr, device, batch_size=256):
    """Run full model over images_arr, return (N, n_classes) numpy logits."""
    model.eval()
    model.to(device)
    results = []
    n = len(images_arr)
    for start in range(0, n, batch_size):
        batch = images_arr[start:start + batch_size]
        t = _images_to_tensor(batch, device)
        logits = model(t)
        results.append(logits.cpu().numpy())
    return np.concatenate(results, axis=0).astype(np.float32)


def _load_byol_model(byol_runs_root, sw, fold_k, data_seed, training_seed, device):
    """Load BYOL model for given sw and fold."""
    run_name  = f"pd128_qext_v1_wd1e-3_lrconst_sw{sw}_f1"
    ckpt_path = (byol_runs_root / run_name
                 / f"data_seed_{data_seed}"
                 / f"training_seed_{training_seed}"
                 / f"cross_val_{fold_k}"
                 / "byol_model_best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"BYOL checkpoint not found: {ckpt_path}")

    model = BYOLEfficientNetB0(
        projection_dim=128,
        hidden_dim=4096,
        feature_compression_mode="mlp",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Checkpoint is a training dict; model weights live under 'model_state_dict'
    state = ckpt.get("model_state_dict", ckpt)
    # Strip _orig_mod. prefix added by torch.compile
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def _load_full_model(pt_path, device, backbone_bn_train=False):
    """Load a torch.save(model, ...) full-model file.

    backbone_bn_train: if True, sets the backbone (model[0]) to train mode so
      BatchNorm uses batch statistics instead of stale running stats.  Use this
      for lineareval/finetuned models whose backbone was frozen during training
      (and therefore has BYOL-pretraining BN statistics, not data statistics).
    """
    model = torch.load(pt_path, map_location=device, weights_only=False)
    model.eval()
    if backbone_bn_train:
        # Set only BatchNorm layers to train mode so they use batch statistics.
        # Dropout stays in eval mode → deterministic, consistent features.
        for _m in model[0].modules():
            if isinstance(_m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                torch.nn.BatchNorm3d)):
                _m.train()
    return model


def _sw_str(sw):
    """Format sw float for filenames: 0.0 → '0.0', etc."""
    return str(float(sw))


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Noise robustness for image/label noise — deep models."
    )
    parser.add_argument("--data_dir",    default="data/preprocessed/lotss/")
    parser.add_argument("--outputs_root", required=True)
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--label_set",   default="initial_pure")
    parser.add_argument("--sw_vals",     nargs="+", type=float, default=[0.0, 0.05, 0.1])
    parser.add_argument("--img_sigmas",  nargs="+", type=float,
                        default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--blur_sigmas", nargs="+", type=float,
                        default=[0.0, 1.0, 2.0, 4.0, 6.0, 10.0],
                        help="Gaussian blur kernel std, in pixels (images are 89x89).")
    parser.add_argument("--occlusion_sizes", nargs="+", type=float,
                        default=[0.0, 10.0, 20.0, 30.0, 45.0, 60.0],
                        help="Centered square occlusion patch side length, in "
                             "pixels (images are 89x89). Occluded region is "
                             "filled with each image's own local background "
                             "level (median of its outer border ring).")
    parser.add_argument("--data_seed",   type=int, default=2)
    parser.add_argument("--training_seed", type=int, default=2)
    parser.add_argument("--n_folds",     type=int, default=5)
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--device",      default=None,
                        help="'cpu' or 'cuda'. Auto-detects if not set.")
    parser.add_argument("--force",       action="store_true",
                        help="Overwrite existing JSON output files.")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}", flush=True)

    outputs_root   = Path(args.outputs_root)
    out_dir        = Path(args.out_dir)
    img_noise_dir  = out_dir / "image_noise"
    res_noise_dir  = out_dir / "resolution_noise"
    occ_noise_dir  = out_dir / "occlusion"
    img_noise_dir.mkdir(parents=True, exist_ok=True)
    res_noise_dir.mkdir(parents=True, exist_ok=True)
    occ_noise_dir.mkdir(parents=True, exist_ok=True)

    byol_runs_root   = outputs_root / "byol_runs"
    finetuned_root   = outputs_root / "classifiers" / "finetuned_sw_cv5" / "without_generative"
    supervised_root  = outputs_root / "classifiers" / "supervised_noes_cv5" / "without_generative"
    splits_base      = outputs_root / "data_splits" / str(args.data_seed)

    # ── Load images and labels once ───────────────────────────────────────
    data_dir = Path(args.data_dir)
    print("Loading images...", flush=True)
    images = np.load(data_dir / "images_filtered.npy").astype(np.float32) / 255.0
    labels = np.load(data_dir / "labels_filtered.npy")
    img_std = float(images.std())
    # Classifier models (finetuned, supervised) were trained with images as uint8→float32
    # (no /255), so their BN running stats are calibrated for the [0, 255] scale.
    img_std_255 = img_std * 255.0
    print(f"  images shape={images.shape}, img_std={img_std:.4f} (×255→{img_std_255:.2f})", flush=True)

    # ── Determine class names ─────────────────────────────────────────────
    _pure_sets = {"classical_pure", "initial_pure", "morphology_pure",
                  "environment_pure", "full_pure"}
    is_multiclass = args.label_set in _pure_sets
    if not is_multiclass:
        raise ValueError(f"Only multiclass (pure) label sets are supported. "
                         f"Got: {args.label_set!r}")
    _base_ls   = args.label_set
    class_names = [ALL_CLASS_NAMES[i] for i in LABEL_SETS[_base_ls]]

    rng = np.random.default_rng(args.seed)

    for fold_k in range(args.n_folds):
        splits_dir = splits_base / f"cross_val_{fold_k}"
        train_idx  = np.load(splits_dir / "labelled_train_idx_f1.npy")
        test_idx   = np.load(splits_dir / "test_idx.npy")

        X_tr_raw = images[train_idx]   # (n_tr, H, W)
        X_te_raw = images[test_idx]    # (n_te, H, W)

        y_tr_full = labels[train_idx]
        y_te_full = labels[test_idx]

        y_tr_sub, tr_mask = apply_label_set(y_tr_full, args.label_set)
        y_te_sub, te_mask = apply_label_set(y_te_full, args.label_set)

        X_tr_raw = X_tr_raw[tr_mask]
        X_te_raw = X_te_raw[te_mask]
        y_tr_int = y_tr_sub.argmax(axis=1)
        y_te_int = y_te_sub.argmax(axis=1)
        # [0, 255] versions for finetuned/supervised (trained without /255 normalisation)
        X_tr_raw_255 = X_tr_raw * 255.0
        X_te_raw_255 = X_te_raw * 255.0

        print(f"\n── Fold {fold_k}: n_tr={len(X_tr_raw)}, n_te={len(X_te_raw)} ──",
              flush=True)

        # ==================================================================
        # BYOL MLP
        # ==================================================================
        for sw in args.sw_vals:
            sw_s = _sw_str(sw)
            print(f"  [byol_mlp] sw={sw_s}", flush=True)

            try:
                byol = _load_byol_model(
                    byol_runs_root, sw_s, fold_k,
                    args.data_seed, args.training_seed, device,
                )
            except FileNotFoundError as e:
                print(f"    [skip] {e}", flush=True)
                continue

            # Clean projections
            X_tr_proj = _extract_projections(byol, X_tr_raw, device, args.batch_size)
            X_te_proj = _extract_projections(byol, X_te_raw, device, args.batch_size)

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr_proj)

            # Train MLP on clean projections — same arch/hyperparams as mlp_v2_256_wd1e-3_es02
            mlp_clean, *_ = train_mlp_es(X_tr_sc, y_tr_int, [256], 1e-3, args.seed)

            # ── Image noise ─────────────────────────────────────────────
            for sigma in args.img_sigmas:
                _out = img_noise_dir / f"byol_mlp_sw{sw_s}_fold{fold_k}_sigma{sigma}.json"
                if _out.exists() and not args.force:
                    continue
                if sigma == 0.0:
                    X_te_noisy_proj = X_te_proj
                else:
                    noise = rng.standard_normal(X_te_raw.shape).astype(np.float32)
                    X_te_noisy = X_te_raw + sigma * img_std * noise
                    X_te_noisy_proj = _extract_projections(
                        byol, X_te_noisy, device, args.batch_size
                    )
                X_te_noisy_sc = scaler.transform(X_te_noisy_proj)
                preds, _ = predict_mlp(mlp_clean, X_te_noisy_sc)
                result = {
                    "model_type": "byol_mlp",
                    "sw": sw,
                    "fold": fold_k,
                    "label_set": args.label_set,
                    "noise_type": "image",
                    "noise_level": sigma,
                    **_metrics(y_te_int, preds, class_names),
                }
                with open(_out, "w") as fh:
                    json.dump(result, fh)
            print(f"    image noise done", flush=True)

            # ── Resolution noise (Gaussian PSF blur) ─────────────────────
            for blur_sigma in args.blur_sigmas:
                _out = res_noise_dir / f"byol_mlp_sw{sw_s}_fold{fold_k}_blur{blur_sigma}.json"
                if _out.exists() and not args.force:
                    continue
                X_te_blur      = _blur_images(X_te_raw, blur_sigma)
                X_te_blur_proj = _extract_projections(byol, X_te_blur, device, args.batch_size)
                X_te_blur_sc   = scaler.transform(X_te_blur_proj)
                preds, _ = predict_mlp(mlp_clean, X_te_blur_sc)
                result = {
                    "model_type": "byol_mlp",
                    "sw": sw,
                    "fold": fold_k,
                    "label_set": args.label_set,
                    "noise_type": "resolution",
                    "noise_level": blur_sigma,
                    **_metrics(y_te_int, preds, class_names),
                }
                with open(_out, "w") as fh:
                    json.dump(result, fh)
            print(f"    resolution noise done", flush=True)

            # ── Occlusion (centered patch, local-background fill) ────────
            for patch_px in args.occlusion_sizes:
                _out = occ_noise_dir / f"byol_mlp_sw{sw_s}_fold{fold_k}_patch{patch_px}.json"
                if _out.exists() and not args.force:
                    continue
                X_te_occ      = _occlude_images(X_te_raw, patch_px)
                X_te_occ_proj = _extract_projections(byol, X_te_occ, device, args.batch_size)
                X_te_occ_sc   = scaler.transform(X_te_occ_proj)
                preds, _ = predict_mlp(mlp_clean, X_te_occ_sc)
                result = {
                    "model_type": "byol_mlp",
                    "sw": sw,
                    "fold": fold_k,
                    "label_set": args.label_set,
                    "noise_type": "occlusion",
                    "noise_level": patch_px,
                    **_metrics(y_te_int, preds, class_names),
                }
                with open(_out, "w") as fh:
                    json.dump(result, fh)
            print(f"    occlusion done", flush=True)

        # ==================================================================
        # Finetuned
        # ==================================================================
        for sw in args.sw_vals:
            sw_s     = _sw_str(sw)
            run_dir  = finetuned_root / f"lineareval_byol_sw_cv5_sw{sw_s}_fold{fold_k}"
            pt_path  = run_dir / "model_best_full.pt"
            if not pt_path.exists():
                print(f"  [skip finetuned] not found: {pt_path}", flush=True)
                continue

            print(f"  [finetuned] sw={sw_s}", flush=True)
            model = _load_full_model(pt_path, device, backbone_bn_train=True)

            # ── Image noise ─────────────────────────────────────────────
            for sigma in args.img_sigmas:
                _out = img_noise_dir / f"finetuned_sw{sw_s}_fold{fold_k}_sigma{sigma}.json"
                if _out.exists() and not args.force:
                    continue
                if sigma == 0.0:
                    X_te_input = X_te_raw_255
                else:
                    noise = rng.standard_normal(X_te_raw_255.shape).astype(np.float32)
                    X_te_input = X_te_raw_255 + sigma * img_std_255 * noise
                logits = _run_model_logits(model, X_te_input, device, args.batch_size)
                preds  = logits.argmax(axis=1)
                result = {
                    "model_type": "finetuned",
                    "sw": sw,
                    "fold": fold_k,
                    "label_set": args.label_set,
                    "noise_type": "image",
                    "noise_level": sigma,
                    **_metrics(y_te_int, preds, class_names),
                }
                with open(_out, "w") as fh:
                    json.dump(result, fh)
            print(f"    image noise done", flush=True)

            # ── Resolution noise (Gaussian PSF blur) ─────────────────────
            for blur_sigma in args.blur_sigmas:
                _out = res_noise_dir / f"finetuned_sw{sw_s}_fold{fold_k}_blur{blur_sigma}.json"
                if _out.exists() and not args.force:
                    continue
                X_te_blur   = _blur_images(X_te_raw_255, blur_sigma)
                logits_blur = _run_model_logits(model, X_te_blur, device, args.batch_size)
                preds       = logits_blur.argmax(axis=1)
                result = {
                    "model_type": "finetuned",
                    "sw": sw,
                    "fold": fold_k,
                    "label_set": args.label_set,
                    "noise_type": "resolution",
                    "noise_level": blur_sigma,
                    **_metrics(y_te_int, preds, class_names),
                }
                with open(_out, "w") as fh:
                    json.dump(result, fh)
            print(f"    resolution noise done", flush=True)

            # ── Occlusion (centered patch, local-background fill) ────────
            for patch_px in args.occlusion_sizes:
                _out = occ_noise_dir / f"finetuned_sw{sw_s}_fold{fold_k}_patch{patch_px}.json"
                if _out.exists() and not args.force:
                    continue
                X_te_occ    = _occlude_images(X_te_raw_255, patch_px)
                logits_occ  = _run_model_logits(model, X_te_occ, device, args.batch_size)
                preds       = logits_occ.argmax(axis=1)
                result = {
                    "model_type": "finetuned",
                    "sw": sw,
                    "fold": fold_k,
                    "label_set": args.label_set,
                    "noise_type": "occlusion",
                    "noise_level": patch_px,
                    **_metrics(y_te_int, preds, class_names),
                }
                with open(_out, "w") as fh:
                    json.dump(result, fh)
            print(f"    occlusion done", flush=True)

        # ==================================================================
        # Supervised (no sw dimension)
        # ==================================================================
        sup_run_dir = supervised_root / f"retrain_sup_noes_cv5_fold{fold_k}"
        sup_pt_path = sup_run_dir / "model_best_full.pt"
        if not sup_pt_path.exists():
            print(f"  [skip supervised] not found: {sup_pt_path}", flush=True)
        else:
            print(f"  [supervised] fold={fold_k}", flush=True)
            sup_model = _load_full_model(sup_pt_path, device)

            # ── Image noise ─────────────────────────────────────────────
            for sigma in args.img_sigmas:
                _out = img_noise_dir / f"supervised_swnone_fold{fold_k}_sigma{sigma}.json"
                if _out.exists() and not args.force:
                    continue
                if sigma == 0.0:
                    X_te_input = X_te_raw_255
                else:
                    noise = rng.standard_normal(X_te_raw_255.shape).astype(np.float32)
                    X_te_input = X_te_raw_255 + sigma * img_std_255 * noise
                logits = _run_model_logits(sup_model, X_te_input, device, args.batch_size)
                preds  = logits.argmax(axis=1)
                result = {
                    "model_type": "supervised",
                    "sw": None,
                    "fold": fold_k,
                    "label_set": args.label_set,
                    "noise_type": "image",
                    "noise_level": sigma,
                    **_metrics(y_te_int, preds, class_names),
                }
                with open(_out, "w") as fh:
                    json.dump(result, fh)
            print(f"    image noise done", flush=True)

            # ── Resolution noise (Gaussian PSF blur) ─────────────────────
            for blur_sigma in args.blur_sigmas:
                _out = res_noise_dir / f"supervised_swnone_fold{fold_k}_blur{blur_sigma}.json"
                if _out.exists() and not args.force:
                    continue
                X_te_blur   = _blur_images(X_te_raw_255, blur_sigma)
                logits_blur = _run_model_logits(sup_model, X_te_blur, device, args.batch_size)
                preds       = logits_blur.argmax(axis=1)
                result = {
                    "model_type": "supervised",
                    "sw": None,
                    "fold": fold_k,
                    "label_set": args.label_set,
                    "noise_type": "resolution",
                    "noise_level": blur_sigma,
                    **_metrics(y_te_int, preds, class_names),
                }
                with open(_out, "w") as fh:
                    json.dump(result, fh)
            print(f"    resolution noise done", flush=True)

            # ── Occlusion (centered patch, local-background fill) ────────
            for patch_px in args.occlusion_sizes:
                _out = occ_noise_dir / f"supervised_swnone_fold{fold_k}_patch{patch_px}.json"
                if _out.exists() and not args.force:
                    continue
                X_te_occ    = _occlude_images(X_te_raw_255, patch_px)
                logits_occ  = _run_model_logits(sup_model, X_te_occ, device, args.batch_size)
                preds       = logits_occ.argmax(axis=1)
                result = {
                    "model_type": "supervised",
                    "sw": None,
                    "fold": fold_k,
                    "label_set": args.label_set,
                    "noise_type": "occlusion",
                    "noise_level": patch_px,
                    **_metrics(y_te_int, preds, class_names),
                }
                with open(_out, "w") as fh:
                    json.dump(result, fh)
            print(f"    occlusion done", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
