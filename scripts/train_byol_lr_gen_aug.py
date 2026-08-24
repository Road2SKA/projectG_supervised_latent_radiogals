"""
train_byol_lr_gen_aug.py

Train a logistic regression on top of frozen BYOL projections augmented with
generated images, sweeping over gen_frac values.

For each gen_frac, n_gen = gen_frac * n_real generated images are embedded
with the frozen BYOL encoder and added to the real labelled training set.
A StandardScaler + LogisticRegression is then trained and evaluated on the
real test set.

Output layout (compatible with the gen-aug notebook cell):
  <out_dir>/
    config.log
    frac_0.00/run{i}/results.json
    frac_0.00/run{i}/test_probs.npy
    frac_0.00/run{i}/test_labels.npy
    frac_0.00/run{i}/test_labels_20.npy
    frac_0.50/run{i}/...
    frac_1.00/run{i}/...
    frac_2.00/run{i}/...

Usage:
  python scripts/train_byol_lr_gen_aug.py \\
    --byol_run_dir outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.05_f1/seed2 \\
    --gen_dir      outputs/generative/run_xxx \\
    --data_dir     outputs/data \\
    --out_dir      outputs/baselines/with_generative/byollr_initial_pure_cwNone_gen_initial_pure_sw0.05_s2 \\
    --label_set    initial_pure
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from suplat.models.byol_models import (
    BYOLEfficient,
    BYOLEfficientNetB0,
    BYOLEncoder,
    BYOLOriginal,
    BYOLPretrainedBackbone,
    PCAProjection,
    create_convnext_tiny_backbone,
    create_resnet18_backbone,
    create_resnet50_backbone,
)
from suplat.models.generative_models import FlowMatchingUNet


# ── Label-set definitions (shared with train_byol_classifiers.py) ────────────

ALL_CLASS_NAMES = [
    'FRI', 'FRII', 'Hybrids', 'Spirals', 'Relaxed doubles',
    'C-curv', 'S-curv', 'Misalign', 'Wings', 'X-shaped',
    'Straight jets', 'Multi hotspots', 'Cont. jets', 'Banding',
    'One-sided', 'Restarted', 'Cluster', 'Merger', 'Diffuse', 'Unknown',
]

LABEL_SETS = {
    "initial":          [0, 1, 2, 3, 4],
    "initial_pure":     [0, 1, 2, 3, 4],
    "classical":        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "classical_pure":   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "morphology":       [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "morphology_pure":  [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "environment":      [0, 1, 16, 17, 18, 19],
    "environment_pure": [0, 1, 16, 17, 18, 19],
    "full":             list(range(20)),
    "full_pure":        list(range(20)),
}

_PURE_SETS = {
    "initial_pure", "classical_pure", "morphology_pure",
    "environment_pure", "full_pure",
}


# ── BYOL model loading ────────────────────────────────────────────────────────

def load_byol_model(checkpoint_path: Path, device: torch.device):
    """Load BYOL model from checkpoint; return (model, model_type, config)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    model_type  = cfg.get("model_type", "convnet")
    proj_dim    = cfg.get("projection_dim", 256)
    hidden_dim  = cfg.get("hidden_dim", 4096)
    encoder_dim = cfg.get("encoder_dim", 512)
    fcm         = cfg.get("feature_compression_mode") or cfg.get("projector", "pca")

    if model_type == "efficientnet-b0":
        model = BYOLEfficientNetB0(
            projection_dim=proj_dim, hidden_dim=hidden_dim,
            feature_compression_mode=fcm,
        )
    elif model_type == "convnet":
        model = BYOLEfficient(
            encoder_dim=encoder_dim,
            projection_dim=proj_dim,
            hidden_dim=hidden_dim,
        )
    elif model_type == "resnet18":
        backbone, _ = create_resnet18_backbone()
        model = BYOLPretrainedBackbone(
            backbone, encoder_dim=encoder_dim,
            projection_dim=proj_dim, hidden_dim=hidden_dim,
            feature_compression_mode=fcm,
        )
    elif model_type == "resnet50":
        backbone, _ = create_resnet50_backbone()
        model = BYOLPretrainedBackbone(
            backbone, encoder_dim=encoder_dim,
            projection_dim=proj_dim, hidden_dim=hidden_dim,
            feature_compression_mode=fcm,
        )
    elif model_type == "convnext-tiny":
        backbone, _ = create_convnext_tiny_backbone()
        model = BYOLPretrainedBackbone(
            backbone, encoder_dim=encoder_dim,
            projection_dim=proj_dim, hidden_dim=hidden_dim,
            feature_compression_mode=fcm,
        )
    else:
        enc = BYOLEncoder()
        model = BYOLOriginal(
            enc, image_size=89,
            projection_size=proj_dim,
            projection_hidden_size=hidden_dim,
        )

    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # PyTorch excludes None-registered buffers from state_dict(), so load_state_dict
    # silently skips PCA buffers (mean_, active_mask_, components_) that were fitted
    # during training. Restore them manually from the raw checkpoint state dict.
    sd = ckpt["model_state_dict"]
    for proj_attr in ("online_projector", "target_projector"):
        submod = getattr(model, proj_attr, None)
        if not isinstance(submod, PCAProjection):
            continue
        for buf_name in ("mean_", "active_mask_", "components_"):
            key = f"{proj_attr}.{buf_name}"
            if key in sd and sd[key] is not None:
                submod.register_buffer(buf_name, sd[key].to(device))

    model.to(device).eval()
    return model, model_type, cfg


@torch.no_grad()
def embed_images(model, model_type: str, images_float: np.ndarray,
                 device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Project (N, H, W) float32 [0,1] images into BYOL projection space.

    Uses online_encoder -> online_projector, matching how labelled_train_projections
    and test_projections are computed during BYOL training.
    """
    all_proj = []
    for i in range(0, len(images_float), batch_size):
        batch = torch.from_numpy(images_float[i:i + batch_size]).unsqueeze(1).float().to(device)
        if model_type in ("convnet", "efficientnet-b0", "resnet18", "resnet50", "convnext-tiny"):
            proj = model.online_projector(model.online_encoder(batch))
        else:
            proj, _ = model(batch, return_embedding=True, return_projection=True)
        all_proj.append(proj.float().cpu().numpy())
    return np.vstack(all_proj)


# ── Label-set helpers ─────────────────────────────────────────────────────────

def apply_label_set(y_full: np.ndarray, label_set: str):
    """Apply label-set + pure filtering to a (N, >=20) label array.

    Returns (y_filtered, mask) where y_filtered has shape (N_kept, n_cols)
    and mask is a boolean array of length N.
    """
    cols = LABEL_SETS[label_set]
    y    = y_full[:, cols]

    if label_set in _PURE_SETS:
        n_init = len(cols) if label_set == "full_pure" else 5
        mask   = y_full[:, :n_init].sum(axis=1) == 1
    else:
        mask = np.ones(len(y_full), dtype=bool)

    return y[mask], mask


# ── Metric computation ────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob, class_names, is_multiclass) -> dict:
    n_cls = len(class_names)
    f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    if is_multiclass:
        y_bin = label_binarize(y_true, classes=list(range(n_cls)))
        try:
            auc = float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            auc = float("nan")
    else:
        aucs = [
            float(roc_auc_score(y_true[:, i], y_prob[:, i]))
            for i in range(n_cls)
            if len(np.unique(y_true[:, i])) >= 2
        ]
        auc = float(np.mean(aucs)) if aucs else float("nan")

    return {"f1_macro": f1, "auc_macro": auc, "accuracy": acc, "recall_macro": rec}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LR gen-aug sweep on frozen BYOL projections."
    )
    parser.add_argument("--byol_run_dir", required=True, type=Path,
                        help="BYOL seed dir (e.g. .../seed2/). "
                             "Must contain byol_model_best.pt and data/byol/.")
    parser.add_argument("--gen_dir", required=True, type=Path,
                        help="Generative model dir (decoder_<variant>.pt, nsf_<variant>.pt).")
    parser.add_argument("--gen_variant", default="all",
                        choices=["all", "full", "initial", "initial_pure"],
                        help="Generator variant to load (default: all).")
    parser.add_argument("--out_dir", required=True, type=Path,
                        help="Output directory "
                             "(e.g. baselines_dir/with_generative/byollr_...).")
    parser.add_argument("--label_set", default="initial_pure",
                        help="Label set for classification (default: initial_pure).")
    parser.add_argument("--f_label", type=float, default=None,
                        help="Label fraction from BYOL training "
                             "(determines labelled_train_labels_f*.npy to load). "
                             "Default: read from checkpoint config.")
    parser.add_argument("--data_seed", type=int, default=None,
                        help="Data split seed (default: read from checkpoint config).")
    parser.add_argument("--gen_fracs", type=float, nargs="+", default=None,
                        help="Gen fractions to sweep "
                             "(default: 0.0 0.5 1.0 2.0 5.0).")
    parser.add_argument("--n_runs", type=int, default=3,
                        help="Repeated runs per fraction (default: 3).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42).")
    parser.add_argument("--lr_c", type=float, default=1.0,
                        help="LR inverse regularisation strength (default: 1.0).")
    parser.add_argument("--feature_type", default="projections",
                        choices=["projections", "encodings"],
                        help="Feature type to load (default: projections).")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for encoder inference (default: 256).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if cached result exists.")
    args = parser.parse_args()

    if args.gen_variant == "full":
        args.gen_variant = "all"

    _ls_base = args.label_set
    if _ls_base not in LABEL_SETS:
        parser.error(f"Unknown label_set: {args.label_set!r}")

    GEN_FRACS = args.gen_fracs if args.gen_fracs is not None else [0.0, 0.5, 1.0, 2.0, 5.0]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load BYOL model ───────────────────────────────────────────────────────
    ckpt_path = args.byol_run_dir / "byol_model_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading BYOL checkpoint: {ckpt_path}")
    byol_model, model_type, byol_cfg = load_byol_model(ckpt_path, device)
    print(f"  model_type : {model_type}")

    # ── Resolve data_seed and f_label from checkpoint config ──────────────────
    data_seed = (args.data_seed if args.data_seed is not None
                 else int(byol_cfg.get("data_seed", 2)))
    f_label   = (args.f_label if args.f_label is not None
                 else float(byol_cfg.get("f_label", 1.0)))
    _f_str    = str(int(f_label)) if f_label == int(f_label) else str(f_label)
    print(f"  data_seed  : {data_seed}")
    print(f"  f_label    : {f_label}  (f_str={_f_str!r})")

    # ── Locate splits_dir (walk up from byol_run_dir.parent = run_dir) ───────
    _search = args.byol_run_dir.parent
    for _ in range(5):
        if (_search / "data_splits").is_dir():
            break
        _search = _search.parent
    splits_dir = _search / "data_splits" / str(data_seed)
    if not splits_dir.exists():
        raise FileNotFoundError(f"data_splits dir not found: {splits_dir}")
    print(f"  splits_dir : {splits_dir}")

    # ── Load pre-computed projections ─────────────────────────────────────────
    byol_data_dir   = args.byol_run_dir / "data" / "byol"
    train_feat_path = byol_data_dir / f"labelled_train_{args.feature_type}.npy"
    test_feat_path  = byol_data_dir / f"test_{args.feature_type}.npy"

    if not train_feat_path.exists():
        raise FileNotFoundError(f"Missing: {train_feat_path}")
    if not test_feat_path.exists():
        raise FileNotFoundError(f"Missing: {test_feat_path}")

    X_train_raw = np.load(train_feat_path).astype(np.float32)
    X_test_raw  = np.load(test_feat_path).astype(np.float32)
    print(f"Train projections : {X_train_raw.shape}")
    print(f"Test  projections : {X_test_raw.shape}")

    # ── Load labels ───────────────────────────────────────────────────────────
    run_lab_labels_path = byol_data_dir / "labelled_train_labels.npy"
    lab_labels_path     = splits_dir / f"labelled_train_labels_f{_f_str}.npy"

    if run_lab_labels_path.exists():
        y_train_full = np.load(run_lab_labels_path)
    elif lab_labels_path.exists():
        y_train_full = np.load(lab_labels_path)
    else:
        raise FileNotFoundError(
            f"Labelled train labels not found:\n  {run_lab_labels_path}\n  {lab_labels_path}"
        )

    y_test_full = np.load(splits_dir / "test_labels.npy")
    y_test_20   = y_test_full[:, :20].copy()

    print(f"Train labels : {y_train_full.shape}")
    print(f"Test  labels : {y_test_full.shape}")

    # ── Apply label set + pure filtering ──────────────────────────────────────
    label_cols    = LABEL_SETS[_ls_base]
    class_names   = [ALL_CLASS_NAMES[i] for i in label_cols]
    n_classes     = len(class_names)
    is_multiclass = args.label_set in _PURE_SETS

    y_train_ls, train_mask = apply_label_set(y_train_full, args.label_set)
    y_test_ls,  test_mask  = apply_label_set(y_test_full,  args.label_set)
    X_train            = X_train_raw[train_mask]
    X_test             = X_test_raw[test_mask]
    y_test_20_filtered = y_test_20[test_mask]

    print(f"Label set '{args.label_set}': "
          f"train={len(X_train)}  test={len(X_test)}  classes={n_classes}")

    if is_multiclass:
        y_train = y_train_ls.argmax(axis=1)
        y_test  = y_test_ls.argmax(axis=1)
    else:
        y_train = y_train_ls
        y_test  = y_test_ls

    # Initial labels (5-class) for NSF conditioning — from unfiltered train set
    _tr_initial = y_train_full[train_mask][:, :5].astype(np.float32)

    # ── Load generative model ─────────────────────────────────────────────────
    import zuko

    print(f"\nLoading generator from {args.gen_dir}  (variant={args.gen_variant})")
    _dec_file = args.gen_dir / f"decoder_{args.gen_variant}.pt"
    _nsf_file = args.gen_dir / f"nsf_{args.gen_variant}.pt"

    _dec_ckpt = torch.load(_dec_file, map_location="cpu", weights_only=False)
    _decoder  = FlowMatchingUNet(z_dim=_dec_ckpt["feat_dim"], base_ch=_dec_ckpt["base_ch"])
    _decoder.load_state_dict(_dec_ckpt["model_state_dict"])
    _decoder.to(device).eval()

    _nsf_ckpt = torch.load(_nsf_file, map_location="cpu", weights_only=False)
    _nsf      = zuko.flows.NSF(
        features=_nsf_ckpt["feat_dim"],
        context=_nsf_ckpt["n_labels"],
        transforms=_nsf_ckpt["n_transforms"],
    )
    _nsf.load_state_dict(_nsf_ckpt["model_state_dict"])
    _nsf.to(device).eval()

    print(f"  NSF context dim : {_nsf_ckpt['n_labels']}")
    print(f"  Decoder feat_dim: {_dec_ckpt['feat_dim']}")

    # ── Output dir ────────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {args.out_dir}")

    cfg_path = args.out_dir / "config.log"
    if not cfg_path.exists() or args.force:
        with open(cfg_path, "w") as _f:
            for k, v in sorted(vars(args).items()):
                _f.write(f"{k}: {v}\n")
            # Extra keys used by notebook helpers
            _f.write(f"byol_run_dir: {args.byol_run_dir}\n")
            _f.write(f"gen_dir: {args.gen_dir}\n")
            _f.write(f"data_seed_resolved: {data_seed}\n")
            _f.write(f"f_label_resolved: {f_label}\n")

    # ── Gen-frac sweep ────────────────────────────────────────────────────────
    _GEN_BS = 64
    n_real  = len(X_train)

    print(f"\n{'='*60}")
    print(f"Gen-aug LR sweep  (label_set={args.label_set}, "
          f"n_real={n_real}, n_runs={args.n_runs})")
    print(f"Gen fracs: {GEN_FRACS}")
    print(f"{'='*60}")

    for gfrac in GEN_FRACS:
        frac_dir = args.out_dir / f"frac_{gfrac:.2f}"
        frac_dir.mkdir(exist_ok=True)
        n_gen = int(gfrac * n_real)
        print(f"\nfrac={gfrac}  n_real={n_real}  n_gen={n_gen}")

        for ri in range(1, args.n_runs + 1):
            run_seed = args.seed + (ri - 1)
            run_dir  = frac_dir / f"run{ri}"
            run_dir.mkdir(exist_ok=True)
            res_path = run_dir / "results.json"

            if res_path.exists() and not args.force:
                print(f"  run={ri}: cached — skipping.")
                continue

            torch.manual_seed(run_seed)
            np.random.seed(run_seed)

            # ── Generate and embed images ─────────────────────────────────────
            if n_gen > 0:
                _cond_idx = np.resize(np.arange(n_real), n_gen)
                _cond     = torch.tensor(_tr_initial[_cond_idx], dtype=torch.float32)

                _gen_imgs_list = []
                _gen_lbs_list  = []
                with torch.no_grad():
                    for b0 in range(0, n_gen, _GEN_BS):
                        b1   = min(b0 + _GEN_BS, n_gen)
                        cy   = _cond[b0:b1].to(device)
                        z    = _nsf(cy).sample()
                        xgen = _decoder.sample(z)              # (B, 1, H, W) float [0,1]
                        _gen_imgs_list.append(xgen.squeeze(1).float().cpu().numpy())
                        _gen_lbs_list.append(y_train[_cond_idx[b0:b1]])

                _gen_imgs = np.concatenate(_gen_imgs_list, axis=0)  # (n_gen, H, W)
                _gen_lbs  = np.concatenate(_gen_lbs_list,  axis=0)

                print(f"  run={ri}: embedding {n_gen} generated images...", flush=True)
                _gen_proj = embed_images(byol_model, model_type, _gen_imgs, device, args.batch_size)

                X_aug = np.concatenate([X_train, _gen_proj], axis=0)
                y_aug = np.concatenate([y_train, _gen_lbs],  axis=0)
            else:
                # frac=0: real data only (baseline point)
                X_aug = X_train.copy()
                y_aug = y_train.copy()

            # ── Fit scaler + LR ───────────────────────────────────────────────
            scaler    = StandardScaler()
            X_aug_sc  = scaler.fit_transform(X_aug)
            X_test_sc = scaler.transform(X_test)

            if is_multiclass:
                clf = LogisticRegression(
                    max_iter=1000, C=args.lr_c, random_state=run_seed
                )
                clf.fit(X_aug_sc, y_aug)
                y_pred = clf.predict(X_test_sc)
                y_prob = clf.predict_proba(X_test_sc)
                # Ensure columns align with [0..n_classes-1]
                if y_prob.shape[1] != n_classes:
                    _full_p = np.zeros((len(X_test_sc), n_classes), dtype=np.float64)
                    for _j, _c in enumerate(clf.classes_):
                        _full_p[:, int(_c)] = y_prob[:, _j]
                    y_prob = _full_p
            else:
                from sklearn.multioutput import MultiOutputClassifier
                clf = MultiOutputClassifier(
                    LogisticRegression(max_iter=1000, C=args.lr_c, random_state=run_seed)
                )
                clf.fit(X_aug_sc, y_aug)
                y_pred = clf.predict(X_test_sc)

                def _pos_prob(est, X):
                    p = est.predict_proba(X)
                    return p[:, 1] if p.shape[1] > 1 else p[:, 0]

                y_prob = np.stack(
                    [_pos_prob(est, X_test_sc) for est in clf.estimators_], axis=1
                )

            # ── Evaluate + save ───────────────────────────────────────────────
            results = compute_metrics(y_test, y_pred, y_prob, class_names, is_multiclass)
            results.update({
                "gen_frac": gfrac, "n_real": n_real, "n_gen": n_gen,
                "label_set": args.label_set, "run": ri,
            })

            print(f"  run={ri}: F1={results['f1_macro']:.4f}  "
                  f"AUC={results['auc_macro']:.4f}  "
                  f"Rec={results['recall_macro']:.4f}  "
                  f"Acc={results['accuracy']:.4f}", flush=True)

            with open(res_path, "w") as _f:
                json.dump(results, _f, indent=2)
            np.save(run_dir / "test_probs.npy",     y_prob.astype(np.float32))
            np.save(run_dir / "test_labels.npy",    y_test)
            np.save(run_dir / "test_labels_20.npy", y_test_20_filtered)

    print("\nDone.")


if __name__ == "__main__":
    main()
