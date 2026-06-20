#!/usr/bin/env python3
"""
Train the generative model (FlowMatchingUNet decoder + conditional NSF) from
pre-extracted BYOL projections saved by train_byol.py.

Outputs are written to <base-dir>/generative/<decoder-type>_<label-subset>_<timestamp>/
  decoder.pt  — decoder checkpoint + training curves
  nsf.pt      — NSF checkpoint + training curves
  config.json — all CLI args

To load in the notebook:
    ckpt = torch.load("...generative/flow_initial_YYYYMMDD_HHMM/decoder.pt")
    decoder = FlowMatchingUNet(z_dim=ckpt["feat_dim"], base_ch=ckpt["base_ch"])
    decoder.load_state_dict(ckpt["model_state_dict"])

Usage:
    python scripts/train_generative.py \\
        --base-dir  outputs/run_cnxt_pca_pond_cosine_wd_20260424_1125 \\
        --images-path data/preprocessed/lotss/images_filtered.npy \\
        --label-subset initial \\
        --decoder-type flow \\
        --decoder-batch-size 64
"""

import argparse
import copy
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zuko
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Label subset definitions ──────────────────────────────────────────────────

ALL_CLASS_NAMES = [
    "FRI", "FRII", "Hybrids", "Spirals", "Relaxed doubles",
    "C-curv", "S-curv", "Misalign", "Wings", "X-shaped",
    "Straight jets", "Multi hotspots", "Cont. jets", "Banding",
    "One-sided", "Restarted",
    "Cluster", "Merger", "Diffuse", "Unknown",
]

LABEL_SUBSETS = {
    "classical":      [0, 1],
    "classical_pure": [0, 1],
    "initial":        list(range(0, 5)),
    "initial_pure":   list(range(0, 5)),
    "environment":    list(range(16, 20)),
    "morphology":     list(range(5, 15)),
    "all":            list(range(0, 20)),
    "pure":           list(range(0, 20)),
}

DERIVED_CLASS_NAMES = [
    "Pure hybrid", "FR hybrid", "Curved FRI", "Curved FRII", "Straight+multi-HS",
]


def _make_derived(y):
    c = lambda i: y[:, i].astype(bool)
    return np.stack([
        (c(2) & ~c(0) & ~c(1)).astype(np.float32),
        (c(2) & (c(0) | c(1))).astype(np.float32),
        (c(0) & (c(5) | c(6))).astype(np.float32),
        (c(1) & (c(5) | c(6))).astype(np.float32),
        (c(10) & c(11)).astype(np.float32),
    ], axis=1)


# ── Model definitions ─────────────────────────────────────────────────────────

class ProjectionDecoder(nn.Module):
    """FC decoder: z → 89×89.  Fast baseline, tends to produce blurry outputs."""
    def __init__(self, proj_dim=256, dropout=0.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(proj_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
            nn.Linear(1024, 256 * 5 * 5), nn.BatchNorm1d(256 * 5 * 5), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
        )
        def up(ic, oc):
            layers = [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout * 0.5))
            return nn.Sequential(*layers)
        self.up1 = up(256, 128); self.up2 = up(128, 64)
        self.up3 = up(64,  32);  self.up4 = up(32,  16)
        self.out_conv = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 5, 5)
        x = self.up1(x); x = self.up2(x); x = self.up3(x); x = self.up4(x)
        x = F.interpolate(x, size=(89, 89), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.out_conv(x))


class _SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        x     = t[:, None] * freqs[None]
        return torch.cat([x.sin(), x.cos()], dim=-1)


class _CondResBlock(nn.Module):
    """Residual conv block with GroupNorm + AdaIN-style scale/shift."""
    def __init__(self, channels, cond_dim):
        super().__init__()
        g = min(8, channels)
        self.norm1 = nn.GroupNorm(g, channels)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.cond  = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x, cond):
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h) * (1 + scale[..., None, None]) + shift[..., None, None])
        return x + self.conv2(h)


class FlowMatchingUNet(nn.Module):
    """
    Flow-matching velocity network  v_θ(x_t, t, z) → velocity field.
    Training loss: MSE(v_θ(x_t, t, z), x_real − x_noise)
    Inference:     Euler ODE from N(0,1) over flow_n_steps steps.
    """
    IMG_SIZE = 89

    def __init__(self, z_dim, t_dim=128, base_ch=32):
        super().__init__()
        C        = base_ch
        cond_dim = z_dim + t_dim

        self.t_embed = nn.Sequential(
            _SinusoidalEmbedding(t_dim),
            nn.Linear(t_dim, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.in_conv = nn.Conv2d(1, C, 3, padding=1)
        self.res_e1  = _CondResBlock(C,    cond_dim)
        self.down1   = nn.Conv2d(C,   C*2, 3, stride=2, padding=1)
        self.res_e2  = _CondResBlock(C*2,  cond_dim)
        self.down2   = nn.Conv2d(C*2, C*4, 3, stride=2, padding=1)
        self.res_e3  = _CondResBlock(C*4,  cond_dim)
        self.down3   = nn.Conv2d(C*4, C*8, 3, stride=2, padding=1)
        self.res_m1  = _CondResBlock(C*8, cond_dim)
        self.res_m2  = _CondResBlock(C*8, cond_dim)
        self.up3     = nn.Conv2d(C*8, C*4, 3, padding=1)
        self.res_d3  = _CondResBlock(C*8, cond_dim)
        self.up2     = nn.Conv2d(C*8, C*2, 3, padding=1)
        self.res_d2  = _CondResBlock(C*4, cond_dim)
        self.up1     = nn.Conv2d(C*4, C,   3, padding=1)
        self.res_d1  = _CondResBlock(C*2, cond_dim)
        self.out_norm = nn.GroupNorm(min(8, C*2), C*2)
        self.out_conv = nn.Conv2d(C*2, 1, 3, padding=1)

    def forward(self, x_t, t, z):
        cond = torch.cat([z, self.t_embed(t)], dim=-1)
        h    = F.silu(self.in_conv(x_t))
        h1   = self.res_e1(h, cond)
        h2   = self.res_e2(F.silu(self.down1(h1)), cond)
        h3   = self.res_e3(F.silu(self.down2(h2)), cond)
        h    = F.silu(self.down3(h3))
        h    = self.res_m1(h, cond)
        h    = self.res_m2(h, cond)
        h    = F.silu(self.up3(F.interpolate(h, size=h3.shape[-2:], mode='nearest')))
        h    = self.res_d3(torch.cat([h, h3], dim=1), cond)
        h    = F.silu(self.up2(F.interpolate(h, size=h2.shape[-2:], mode='nearest')))
        h    = self.res_d2(torch.cat([h, h2], dim=1), cond)
        h    = F.silu(self.up1(F.interpolate(h, size=h1.shape[-2:], mode='nearest')))
        h    = self.res_d1(torch.cat([h, h1], dim=1), cond)
        return self.out_conv(F.silu(self.out_norm(h)))

    @torch.no_grad()
    def sample(self, z, n_steps=20):
        self.eval()
        x  = torch.randn(z.shape[0], 1, self.IMG_SIZE, self.IMG_SIZE, device=z.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((z.shape[0],), i * dt, device=z.device)
            x = x + self(x, t, z) * dt
        return torch.clamp(x, 0.0, 1.0)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train decoder + NSF generative model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-dir",      required=True,
                   help="BYOL run output directory (must contain *_projections.npy)")
    p.add_argument("--images-path",   required=True,
                   help="Path to images_filtered.npy  (N, 89, 89) uint8")
    p.add_argument("--label-subset",  default="initial",
                   choices=list(LABEL_SUBSETS) + ["derived"],
                   help="Label subset to condition NSF on")
    p.add_argument("--use-projections",    dest="use_projections",
                   action="store_true",  default=True,
                   help="Use PCA/MLP projections as decoder input (default)")
    p.add_argument("--no-use-projections", dest="use_projections",
                   action="store_false",
                   help="Use raw encoder features as decoder input")
    p.add_argument("--decoder-type",  default="flow", choices=["flow", "cnn"])

    # Decoder
    g = p.add_argument_group("decoder training")
    g.add_argument("--decoder-epochs",       type=int,   default=300)
    g.add_argument("--decoder-patience",     type=int,   default=40)
    g.add_argument("--decoder-lr",           type=float, default=1e-4)
    g.add_argument("--decoder-batch-size",   type=int,   default=64,
                   help="Reduce if OOM; 64 works on ~12 GB GPU for flow decoder")
    g.add_argument("--decoder-weight-decay", type=float, default=1e-4)
    g.add_argument("--decoder-dropout",      type=float, default=0.3,
                   help="CNN decoder only")
    g.add_argument("--decoder-loss-alpha",   type=float, default=0.7,
                   help="MSE weight for cnn decoder; (1-α) → L1")
    g.add_argument("--base-ch",              type=int,   default=32,
                   help="FlowMatchingUNet base channels; use 16 for lower memory footprint")

    # NSF
    g = p.add_argument_group("NSF training")
    g.add_argument("--flow-epochs",       type=int,   default=200)
    g.add_argument("--flow-patience",     type=int,   default=20)
    g.add_argument("--flow-lr",           type=float, default=3e-4)
    g.add_argument("--flow-batch-size",   type=int,   default=256)
    g.add_argument("--flow-n-transforms", type=int,   default=8)
    g.add_argument("--flow-weight-decay", type=float, default=1e-5)
    g.add_argument("--flow-feat-noise",   type=float, default=0.05)
    g.add_argument("--flow-n-steps",      type=int,   default=20,
                   help="Euler integration steps at inference")

    # Misc
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--run-name",     default="",
                   help="Override output directory suffix (default: <type>_<label>)")
    p.add_argument("--catalogue",    default=None,
                   help="Path to catalogue split sidecar JSON (e.g. "
                        "catalogues/lotss_initial_all_split.json).  When "
                        "provided (or auto-detected), projections are loaded "
                        "from <base-dir>/projections/lotss_*.npy using the "
                        "catalogue 64/16/20 train/val/test split — matching "
                        "classification.ipynb.  Falls back to legacy "
                        "train/val_projections.npy if not found.")
    p.add_argument("--skip-decoder", action="store_true",
                   help="Skip decoder training")
    p.add_argument("--skip-nsf",     action="store_true",
                   help="Skip NSF training")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def _find_catalogue_sidecar(base_dir: Path, catalogue_arg=None):
    """Return path to a catalogue split sidecar JSON, or None if not found.

    Search order:
      1. Explicit --catalogue argument (YAML or JSON accepted).
      2. Auto-detect: look for *_split.json next to known YAML files in
         <root>/catalogues/, where root is derived from base_dir.
    """
    import json as _json

    if catalogue_arg is not None:
        p = Path(catalogue_arg)
        if p.suffix == ".yaml":
            # Derive sidecar path from YAML name
            sidecar = p.parent / (p.stem + "_split.json")
        else:
            sidecar = p
        if sidecar.exists():
            return sidecar
        print(f"  WARNING: --catalogue path not found: {sidecar}")
        return None

    # Auto-detect: walk up from base_dir to find catalogues/ directory
    for parent in [base_dir] + list(base_dir.parents):
        cat_dir = parent / "catalogues"
        if cat_dir.is_dir():
            # Prefer lotss_initial_all_split.json if present
            preferred = cat_dir / "lotss_initial_all_split.json"
            if preferred.exists():
                return preferred
            # Fall back to any *_split.json
            sidecars = sorted(cat_dir.glob("*_split.json"))
            if sidecars:
                return sidecars[0]
    return None


def load_data(args):
    import json as _json

    base_dir = Path(args.base_dir)

    # ── Try catalogue-based projections first ─────────────────────────────────
    cat_proj_dir = base_dir / "projections"
    cat_proj_file = cat_proj_dir / "lotss_projections.npy"
    cat_split_file = cat_proj_dir / "lotss_splits.npy"
    sidecar_path = _find_catalogue_sidecar(base_dir, getattr(args, "catalogue", None))

    use_catalogue = (cat_proj_file.exists() and cat_split_file.exists()
                     and sidecar_path is not None)

    if use_catalogue:
        print(f"Loading catalogue-aligned projections…")
        print(f"  Projections : {cat_proj_file}")
        print(f"  Sidecar     : {sidecar_path}")

        _all_z     = np.load(cat_proj_file).astype(np.float32)
        _split_ids = np.load(cat_split_file)

        with open(sidecar_path) as _f:
            _sidecar = _json.load(_f)
        # Dataset key is the first non-seed key
        _ds_key  = next(k for k in _sidecar if k != "dataset_seed")
        train_idx = np.array(_sidecar[_ds_key]["train"])
        val_idx   = np.array(_sidecar[_ds_key]["val"])
        print(f"  Catalogue split: train={len(train_idx)} | "
              f"val={len(np.array(_sidecar[_ds_key]['val']))} | "
              f"test={len(np.array(_sidecar[_ds_key]['test']))}")

        train_z = _all_z[_split_ids == 0]
        val_z   = _all_z[_split_ids == 1]

        # Full 20-dim labels from images_path directory
        lbl_path = Path(args.images_path).parent / "labels_filtered.npy"
        if not lbl_path.exists():
            raise FileNotFoundError(
                f"labels_filtered.npy not found next to images_filtered.npy: {lbl_path}"
            )
        _all_lbl = np.load(lbl_path).astype(np.float32)
        train_lbl = _all_lbl[train_idx]
        val_lbl   = _all_lbl[val_idx]

    else:
        # ── Legacy fallback: train/val_projections.npy ────────────────────────
        proj_dir = (base_dir / "data"       if (base_dir / "data").is_dir()
                    else base_dir / "embeddings" if (base_dir / "embeddings").is_dir()
                    else base_dir)
        print("Loading projections (legacy path)…")
        print(f"  Projections dir: {proj_dir}")
        train_z   = np.load(proj_dir / "train_projections.npy").astype(np.float32)
        val_z     = np.load(proj_dir / "val_projections.npy").astype(np.float32)
        train_lbl = np.load(proj_dir / "train_labels.npy").astype(np.float32)
        val_lbl   = np.load(proj_dir / "val_labels.npy").astype(np.float32)

        def _find_idx_dir(*dirs):
            for d in dirs:
                if d is not None and d.exists() and all(
                    (d / f).exists() for f in ("train_idx.npy", "val_idx.npy")
                ):
                    return d
            return None

        idx_dir = _find_idx_dir(proj_dir)
        if idx_dir is not None:
            train_idx = np.load(idx_dir / "train_idx.npy")
            val_idx   = np.load(idx_dir / "val_idx.npy")
            print(f"  Loaded split indices from {idx_dir.name}/")
        else:
            from sklearn.model_selection import train_test_split as _tts
            print("  Index files not found — replicating 70/15/15 split")
            _all_idx = np.arange(np.load(args.images_path, mmap_mode="r").shape[0])
            train_idx, temp_idx = _tts(_all_idx, test_size=0.30, random_state=args.seed)
            val_idx, _          = _tts(temp_idx, test_size=0.50, random_state=args.seed)

    # ── Label subset ──────────────────────────────────────────────────────────
    label_subset = args.label_subset
    if label_subset == "derived":
        train_y     = _make_derived(train_lbl)
        val_y       = _make_derived(val_lbl)
        label_names = DERIVED_CLASS_NAMES
    else:
        label_idx   = LABEL_SUBSETS[label_subset]
        train_y     = train_lbl[:, label_idx]
        val_y       = val_lbl[:, label_idx]
        label_names = [ALL_CLASS_NAMES[i] for i in label_idx]

    # ── Pure filtering ────────────────────────────────────────────────────────
    keep_train = np.ones(len(train_lbl), dtype=bool)
    keep_val   = np.ones(len(val_lbl),   dtype=bool)
    if label_subset == "pure":
        keep_train = train_lbl.sum(axis=1) == 1
        keep_val   = val_lbl.sum(axis=1)   == 1
    elif label_subset in ("classical_pure", "initial_pure"):
        keep_train = train_lbl[:, 0:5].sum(axis=1) == 1
        keep_val   = val_lbl[:, 0:5].sum(axis=1)   == 1

    # ── Image tensors ─────────────────────────────────────────────────────────
    print("Loading images…")
    all_images = np.load(args.images_path)  # (N, 89, 89) uint8

    train_imgs = torch.from_numpy(
        all_images[train_idx[:len(keep_train)]].astype(np.float32) / 255.0
    ).unsqueeze(1)
    val_imgs = torch.from_numpy(
        all_images[val_idx[:len(keep_val)]].astype(np.float32) / 255.0
    ).unsqueeze(1)

    train_feats = torch.from_numpy(train_z)
    val_feats   = torch.from_numpy(val_z)

    # Apply filter mask
    train_imgs  = train_imgs[keep_train]
    val_imgs    = val_imgs[keep_val]
    train_feats = train_feats[keep_train]
    val_feats   = val_feats[keep_val]
    train_y     = train_y[keep_train]
    val_y       = val_y[keep_val]

    print(f"  Split (after filter): train={len(train_feats)} | val={len(val_feats)}")
    print(f"  Feature dim: {train_feats.shape[1]} | n_labels: {train_y.shape[1]} {label_names}")

    return train_feats, val_feats, train_imgs, val_imgs, train_y, val_y, label_names


# ── Training ──────────────────────────────────────────────────────────────────

def train_decoder(args, train_feats, val_feats, train_imgs, val_imgs, device, out_dir):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feat_dim = train_feats.shape[1]
    print(f"\n── Decoder ({args.decoder_type}) ──────────────────────────────────────────")
    print(f"   feat_dim={feat_dim}  base_ch={args.base_ch}  bs={args.decoder_batch_size}")
    print(f"   epochs={args.decoder_epochs}  patience={args.decoder_patience}  lr={args.decoder_lr}")

    if args.decoder_type == "flow":
        decoder = FlowMatchingUNet(z_dim=feat_dim, base_ch=args.base_ch).to(device)
    else:
        decoder = ProjectionDecoder(emb_dim=feat_dim, dropout=args.decoder_dropout).to(device)
    print(f"   params: {sum(p.numel() for p in decoder.parameters()):,}")

    opt   = torch.optim.Adam(decoder.parameters(), lr=args.decoder_lr,
                             weight_decay=args.decoder_weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.decoder_epochs)

    bs           = args.decoder_batch_size
    train_loader = DataLoader(TensorDataset(train_feats, train_imgs),
                              batch_size=bs, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TensorDataset(val_feats,   val_imgs),
                              batch_size=bs, shuffle=False)

    def flow_loss(z_b, img_b):
        t     = torch.rand(img_b.shape[0], device=device)
        noise = torch.randn_like(img_b)
        x_t   = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * img_b
        return F.mse_loss(decoder(x_t, t, z_b), img_b - noise)

    def cnn_loss(z_b, img_b):
        pred = decoder(z_b)
        return (args.decoder_loss_alpha * F.mse_loss(pred, img_b)
                + (1 - args.decoder_loss_alpha) * F.l1_loss(pred, img_b))

    compute_loss = flow_loss if args.decoder_type == "flow" else cnn_loss

    best_val, best_state, best_epoch, patience_ctr = float("inf"), None, 0, 0
    train_losses, val_losses = [], []

    for epoch in range(args.decoder_epochs):
        decoder.train()
        tr_loss = 0.0
        for z, img in train_loader:
            z, img = z.to(device), img.to(device)
            loss = compute_loss(z, img)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        decoder.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for z, img in val_loader:
                z, img = z.to(device), img.to(device)
                vl_loss += compute_loss(z, img).item()
        vl_loss /= len(val_loader)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        sched.step()

        if vl_loss < best_val:
            best_val, best_epoch, patience_ctr = vl_loss, epoch, 0
            best_state = copy.deepcopy(decoder.state_dict())
        else:
            patience_ctr += 1
            if patience_ctr >= args.decoder_patience:
                print(f"   Early stopping at epoch {epoch + 1} (patience={args.decoder_patience})")
                break

        if (epoch + 1) % 20 == 0:
            print(f"   E{epoch + 1:4d}  train={tr_loss:.4f}  val={vl_loss:.4f}")

    print(f"   Best val={best_val:.4f} at epoch {best_epoch + 1}")
    decoder.load_state_dict(best_state)

    ckpt_path = out_dir / "decoder.pt"
    torch.save({
        "model_state_dict": best_state,
        "decoder_type":     args.decoder_type,
        "feat_dim":         feat_dim,
        "base_ch":          args.base_ch,
        "epoch":            best_epoch,
        "best_val_loss":    best_val,
        "train_losses":     train_losses,
        "val_losses":       val_losses,
    }, ckpt_path)
    print(f"   Saved → {ckpt_path}")
    return decoder


def train_nsf(args, train_feats, val_feats, train_y, val_y, device, out_dir):
    torch.manual_seed(args.seed)

    feat_dim = train_feats.shape[1]
    n_labels = train_y.shape[1] if hasattr(train_y, "shape") else train_y.size(1)

    print(f"\n── NSF ─────────────────────────────────────────────────────────────")
    print(f"   feat_dim={feat_dim}  n_labels={n_labels}  transforms={args.flow_n_transforms}")
    print(f"   bs={args.flow_batch_size}  epochs={args.flow_epochs}  patience={args.flow_patience}  lr={args.flow_lr}")

    flow = zuko.flows.NSF(
        features=feat_dim,
        context=n_labels,
        transforms=args.flow_n_transforms,
        randperm=True,
    ).to(device)
    print(f"   params: {sum(p.numel() for p in flow.parameters() if p.requires_grad):,}")

    train_y_t = train_y if torch.is_tensor(train_y) else torch.from_numpy(train_y)
    val_y_t   = val_y   if torch.is_tensor(val_y)   else torch.from_numpy(val_y)

    opt   = torch.optim.Adam(flow.parameters(), lr=args.flow_lr,
                             weight_decay=args.flow_weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.flow_epochs)

    bs           = args.flow_batch_size
    train_loader = DataLoader(TensorDataset(train_feats, train_y_t),
                              batch_size=bs, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TensorDataset(val_feats,   val_y_t),
                              batch_size=bs, shuffle=False)

    best_val, best_state, best_epoch, patience_ctr = float("inf"), None, 0, 0
    train_nlls, val_nlls = [], []

    for epoch in range(args.flow_epochs):
        flow.train()
        tr_nll = 0.0
        for z_b, y_b in train_loader:
            z_b, y_b = z_b.to(device), y_b.to(device)
            if args.flow_feat_noise > 0:
                z_b = z_b + torch.randn_like(z_b) * args.flow_feat_noise
            loss = -flow(y_b).log_prob(z_b).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tr_nll += loss.item()
        tr_nll /= len(train_loader)

        flow.eval()
        vl_nll = 0.0
        with torch.no_grad():
            for z_b, y_b in val_loader:
                z_b, y_b = z_b.to(device), y_b.to(device)
                vl_nll += (-flow(y_b).log_prob(z_b).mean()).item()
        vl_nll /= len(val_loader)

        train_nlls.append(tr_nll)
        val_nlls.append(vl_nll)
        sched.step()

        if vl_nll < best_val:
            best_val, best_epoch, patience_ctr = vl_nll, epoch, 0
            best_state = copy.deepcopy(flow.state_dict())
        else:
            patience_ctr += 1
            if patience_ctr >= args.flow_patience:
                print(f"   Early stopping at epoch {epoch + 1} (patience={args.flow_patience})")
                break

        if (epoch + 1) % 20 == 0:
            print(f"   E{epoch + 1:4d}  train={tr_nll:.4f}  val={vl_nll:.4f}")

    print(f"   Best val={best_val:.4f} at epoch {best_epoch + 1}")
    flow.load_state_dict(best_state)

    ckpt_path = out_dir / "nsf.pt"
    torch.save({
        "model_state_dict": best_state,
        "feat_dim":         feat_dim,
        "n_labels":         n_labels,
        "n_transforms":     args.flow_n_transforms,
        "epoch":            best_epoch,
        "best_val_nll":     best_val,
        "train_nlls":       train_nlls,
        "val_nlls":         val_nlls,
    }, ckpt_path)
    print(f"   Saved → {ckpt_path}")
    return flow


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")

    ts      = datetime.now().strftime("%Y%m%d_%H%M")
    suffix  = args.run_name or f"{args.decoder_type}_{args.label_subset}"
    out_dir = Path(args.base_dir) / "generative" / f"{suffix}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    (train_feats, val_feats,
     train_imgs, val_imgs,
     train_y, val_y, label_names) = load_data(args)

    if not args.skip_decoder:
        train_decoder(args, train_feats, val_feats, train_imgs, val_imgs, device, out_dir)
    else:
        print("Skipping decoder training (--skip-decoder)")

    if not args.skip_nsf:
        train_nsf(args, train_feats, val_feats, train_y, val_y, device, out_dir)
    else:
        print("Skipping NSF training (--skip-nsf)")

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
