#!/usr/bin/env python3
"""
extract_embeddings.py

Extract frozen BYOL encoder embeddings for one or more eval datasets.

For each dataset, saves:
    {output_dir}/{dataset}_embeddings.npy   — (N, D) float32
    {output_dir}/{dataset}_labels.npy       — (N,)   int  (-1 = unlabelled)

Usage:
    python scripts/extract_embeddings.py \\
        --datasets mirabest radio_galaxy_dataset mgcls_5k \\
        --checkpoint outputs/runs/run_id/byol_model_best.pt \\
        --output_dir outputs/embeddings/run_id \\
        --root /users/mbredber/p3_SUPLAT
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from suplat.data.eval_dataset import EvalDataset
from suplat.models.byol_models import (
    BYOLEfficient,
    BYOLEfficientNetB0,
    BYOLEncoder,
    BYOLOriginal,
    BYOLPretrainedBackbone,
    create_convnext_tiny_backbone,
    create_resnet18_backbone,
    create_resnet50_backbone,
)


def load_model(checkpoint_path: Path, device: torch.device):
    """Reconstruct BYOL model from checkpoint and return in eval mode."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    model_type  = cfg.get("model_type", "convnet")
    proj_dim    = cfg.get("projection_dim", 256)
    hidden_dim  = cfg.get("hidden_dim", 4096)
    encoder_dim = cfg.get("encoder_dim", 512)
    fcm         = cfg.get("feature_compression_mode", "pca")

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
    model.to(device)
    model.eval()
    return model, model_type


@torch.no_grad()
def extract(model, model_type: str, loader: DataLoader,
            device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    all_emb, all_lbl = [], []
    for imgs, labels in tqdm(loader, desc="  extracting", leave=False):
        imgs = imgs.to(device)
        if model_type in ("convnet", "efficientnet-b0", "resnet18",
                          "resnet50", "convnext-tiny"):
            emb = model.online_encoder(imgs)
        else:
            emb, _ = model(imgs, return_embedding=True, return_projection=True)
        all_emb.append(emb.cpu().float().numpy())
        all_lbl.append(labels.numpy())
    return np.vstack(all_emb), np.concatenate(all_lbl)


def main():
    ap = argparse.ArgumentParser(
        description="Extract BYOL encoder embeddings for eval datasets"
    )
    ap.add_argument("--datasets",    nargs="+", required=True,
                    help="Dataset names from DATASET_REGISTRY")
    ap.add_argument("--checkpoint",  type=Path,  required=True,
                    help="Path to byol_model_best.pt checkpoint")
    ap.add_argument("--output_dir",  type=Path,  required=True,
                    help="Directory to save embeddings and labels")
    ap.add_argument("--root",        type=Path,  default=ROOT,
                    help="Project root (default: repo root)")
    ap.add_argument("--batch_size",  type=int,   default=256)
    ap.add_argument("--num_workers", type=int,   default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model, model_type = load_model(args.checkpoint, device)
    print(f"Model type: {model_type}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in args.datasets:
        print(f"\nDataset: {ds_name}")
        try:
            ds = EvalDataset(ds_name, root=args.root)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        if len(ds) == 0:
            print("  SKIP: empty dataset")
            continue

        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        emb, lbl = extract(model, model_type, loader, device)

        out_emb = args.output_dir / f"{ds_name}_embeddings.npy"
        out_lbl = args.output_dir / f"{ds_name}_labels.npy"
        np.save(out_emb, emb)
        np.save(out_lbl, lbl)
        print(f"  Saved {emb.shape} embeddings -> {out_emb}")
        print(f"  Saved {lbl.shape} labels     -> {out_lbl}")


if __name__ == "__main__":
    main()
