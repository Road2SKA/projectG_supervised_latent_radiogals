#!/usr/bin/env python3
"""
apply_vicreg_patch.py

Adds VICReg variance/covariance anti-collapse regularisation (plus a RankMe
effective-rank diagnostic) to the SUPLAT BYOL pipeline.

Run from the repo root:
    python apply_vicreg_patch.py            # apply
    python apply_vicreg_patch.py --check    # dry-run: verify anchors, change nothing

Design notes
------------
* Pure string replacement with an assert on every anchor's occurrence count, so
  the script fails loudly rather than silently mis-patching.
* Idempotent: re-running detects the marker `# [VICReg]` and skips files already
  patched.
* VICReg loss weights default to 0.0 -> behaviour is identical to the current
  code unless you pass --vicreg-var-weight / --vicreg-cov-weight. RankMe logging
  is always on (cheap) so baseline runs gain the diagnostic for free.
"""

import argparse
import sys
from pathlib import Path

CHECK_ONLY = False


def _edit(path: Path, old: str, new: str, *, count=1, label=""):
    """Replace `old` with `new` in `path`, asserting it occurs exactly `count` times."""
    text = path.read_text()
    found = text.count(old)
    if found != count:
        raise SystemExit(
            f"ABORT [{path.name}] anchor not found exactly {count}x (found {found}): {label}\n"
            f"  ---\n  {old.strip().splitlines()[0][:90]}\n  ---"
        )
    if CHECK_ONLY:
        print(f"  ok  [{path.name}] {label} ({found}x)")
        return
    path.write_text(text.replace(old, new))
    print(f"  patched [{path.name}] {label}")


def already_patched(path: Path) -> bool:
    return path.exists() and "# [VICReg]" in path.read_text()


# ===========================================================================
# 1) trainer.py  — append vicreg_var_cov_loss() and effective_rank()
# ===========================================================================
TRAINER_FUNCS = '''

# =============================================================================
# VICReg ANTI-COLLAPSE + RankMe DIAGNOSTIC                       # [VICReg]
# =============================================================================

def vicreg_var_cov_loss(z, gamma=1.0, eps=1e-4):
    """
    VICReg variance + covariance regularisation on a batch of projections.

    This is the anti-collapse term: it does NOT pull samples together (that is
    L_aug's and L_friend's job). It only insists that, across the batch, every
    projection dimension carries spread (variance) and that dimensions are
    mutually decorrelated (covariance). Together these keep the representational
    space high-rank, which is exactly what the downstream PROTEGE GP needs.

    Reference: Bardes, Ponce & LeCun (2022), "VICReg".

    Args:
        z:     (B, D) batch of projections (online branch, NOT normalised).
               B must be > 1 for the covariance term to be meaningful.
        gamma: target per-dimension standard deviation. The variance hinge only
               penalises dimensions whose std falls *below* gamma; dimensions
               already above gamma incur no penalty.
        eps:   numerical floor inside the sqrt to avoid a divide-by-zero gradient
               when a dimension's variance is ~0.

    Returns:
        (var_loss, cov_loss): two scalar tensors. Caller weights and sums them.
    """
    B, D = z.shape

    # --- Variance term --------------------------------------------------------
    # Per-dimension standard deviation across the batch. The +eps under the sqrt
    # keeps the gradient finite at zero variance (a fully collapsed dimension).
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)        # (D,)
    # Hinge: only dimensions below the target gamma are pushed up. relu() zeroes
    # the penalty for healthy dimensions so we never *inflate* variance needlessly.
    var_loss = torch.mean(F.relu(gamma - std))

    # --- Covariance term ------------------------------------------------------
    # Centre the batch, form the DxD covariance matrix, and penalise the squared
    # off-diagonal entries (the on-diagonal entries are variances, handled above).
    z_centered = z - z.mean(dim=0, keepdim=True)               # (B, D)
    cov = (z_centered.T @ z_centered) / max(B - 1, 1)          # (D, D)
    # Sum of squared off-diagonals, normalised by D (as in the VICReg paper).
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / D

    return var_loss, cov_loss


@torch.no_grad()
def effective_rank(z, eps=1e-7):
    """
    RankMe effective rank of a batch of projections (Garrido et al. 2023).

    Rather than thresholding cumulative variance at some arbitrary cut (e.g. 95%),
    RankMe treats the singular-value spectrum as a probability distribution and
    returns exp(entropy). Intuitively: "how many dimensions are *effectively* in
    use". A fully collapsed space -> ~1; a space using D dimensions equally -> D.
    One smooth, comparable number per epoch to watch collapse happen (or not).

    Args:
        z:   (B, D) batch of projections.
        eps: floor on the normalised singular values before taking the log.

    Returns:
        float effective rank.
    """
    # Singular values of the batch. We do not centre here so the metric matches
    # the raw geometry the GP will see downstream.
    s = torch.linalg.svdvals(z.float())                        # (min(B, D),)
    # Normalise to a probability distribution p_i = s_i / sum(s).
    p = s / (s.sum() + eps)
    # Shannon entropy of the spectrum, then exponentiate -> effective rank.
    entropy = -(p * torch.log(p + eps)).sum()
    return float(torch.exp(entropy).item())
'''


def patch_trainer(root: Path):
    path = root / "src/suplat/trainer/trainer.py"
    if already_patched(path):
        print(f"  skip [{path.name}] already patched"); return
    if CHECK_ONLY:
        print(f"  ok  [{path.name}] will append VICReg + RankMe functions"); return
    path.write_text(path.read_text().rstrip() + "\n" + TRAINER_FUNCS)
    print(f"  patched [{path.name}] appended vicreg_var_cov_loss + effective_rank")


# ===========================================================================
# 2) trainer/__init__.py — export the two new functions
# ===========================================================================
def patch_trainer_init(root: Path):
    path = root / "src/suplat/trainer/__init__.py"
    if already_patched(path):
        print(f"  skip [{path.name}] already patched"); return
    _edit(
        path,
        'from .trainer import get_ema_decay, get_warmup_lr, get_supervision_weight, byol_loss, extract_embeddings_from_loader\n\n'
        '__all__ = ["get_ema_decay", "get_warmup_lr", "get_supervision_weight", "byol_loss", "extract_embeddings_from_loader"]',
        '# [VICReg] export anti-collapse helpers\n'
        'from .trainer import (\n'
        '    get_ema_decay, get_warmup_lr, get_supervision_weight, byol_loss,\n'
        '    extract_embeddings_from_loader, vicreg_var_cov_loss, effective_rank,\n'
        ')\n\n'
        '__all__ = [\n'
        '    "get_ema_decay", "get_warmup_lr", "get_supervision_weight", "byol_loss",\n'
        '    "extract_embeddings_from_loader", "vicreg_var_cov_loss", "effective_rank",\n'
        ']',
        label="exports",
    )


# ===========================================================================
# 3) byol_models.py — forward() gains return_online_proj kwarg (x3 classes)
# ===========================================================================
def patch_models(root: Path):
    path = root / "src/suplat/models/byol_models.py"
    if already_patched(path):
        print(f"  skip [{path.name}] already patched"); return
    _edit(
        path,
        "    def forward(self, x1, x2):",
        "    def forward(self, x1, x2, return_online_proj=False):  # [VICReg]",
        count=3, label="forward signatures",
    )
    _edit(
        path,
        "        return online_pred_1, online_pred_2, target_proj_1, target_proj_2",
        "        # [VICReg] optionally expose online projections for VICReg regularisation\n"
        "        if return_online_proj:\n"
        "            return (online_pred_1, online_pred_2, target_proj_1, target_proj_2,\n"
        "                    online_proj_1, online_proj_2)\n"
        "        return online_pred_1, online_pred_2, target_proj_1, target_proj_2",
        count=3, label="forward returns",
    )


# ===========================================================================
# 4) train_byol.py — CLI flags, loss term, logging, checkpoint, run name
# ===========================================================================
def patch_train(root: Path):
    path = root / "scripts/train_byol.py"
    if already_patched(path):
        print(f"  skip [{path.name}] already patched"); return

    # 4a) import the two new helpers
    _edit(
        path,
        "from suplat.trainer.trainer import byol_loss, get_warmup_lr, get_supervision_weight, extract_embeddings_from_loader",
        "from suplat.trainer.trainer import (byol_loss, get_warmup_lr, get_supervision_weight,\n"
        "                                    extract_embeddings_from_loader,\n"
        "                                    vicreg_var_cov_loss, effective_rank)  # [VICReg]",
        label="import",
    )

    # 4b) argparse flags (inserted just before parse)
    _edit(
        path,
        "    return ap.parse_args()",
        "    # [VICReg] anti-collapse regularisation (default 0.0 = off; behaviour unchanged)\n"
        '    ap.add_argument("--vicreg-var-weight", type=float, default=0.0,\n'
        '                    help="Weight on VICReg variance hinge (try ~25 for mlp/none projector).")\n'
        '    ap.add_argument("--vicreg-cov-weight", type=float, default=0.0,\n'
        '                    help="Weight on VICReg covariance penalty (try ~1 for mlp/none projector).")\n'
        '    ap.add_argument("--vicreg-gamma", type=float, default=1.0,\n'
        '                    help="Target per-dimension std for the VICReg variance hinge (default 1.0).")\n\n'
        "    return ap.parse_args()",
        label="argparse flags",
    )

    # 4c) constants
    _edit(
        path,
        'USE_CURRICULUM = SUPERVISION_WEIGHT_SCHEDULE != "constant"',
        'USE_CURRICULUM = SUPERVISION_WEIGHT_SCHEDULE != "constant"\n'
        "# [VICReg] anti-collapse weights\n"
        "VICREG_VAR_WEIGHT = args.vicreg_var_weight\n"
        "VICREG_COV_WEIGHT = args.vicreg_cov_weight\n"
        "VICREG_GAMMA      = args.vicreg_gamma\n"
        "if (VICREG_VAR_WEIGHT > 0 or VICREG_COV_WEIGHT > 0) and PROJECTOR == 'pca':\n"
        '    print("WARNING: VICReg weights > 0 with --projector pca. PCA outputs are already "\n'
        '          "decorrelated and have descending variances; VICReg is intended for "\n'
        '          "mlp/none projectors. Proceeding anyway.")',
        label="constants",
    )

    # 4d) run-name tag
    _edit(
        path,
        '    RUN_ID += f"_{MODEL_TYPE}_proj{PROJECTOR}_w{args.weighting}_f{F_LABEL}"',
        '    RUN_ID += f"_{MODEL_TYPE}_proj{PROJECTOR}_w{args.weighting}_f{F_LABEL}"\n'
        "    if args.vicreg_var_weight > 0 or args.vicreg_cov_weight > 0:  # [VICReg]\n"
        '        RUN_ID += f"_vic{args.vicreg_var_weight}-{args.vicreg_cov_weight}"',
        label="run name",
    )

    # 4e) history dict keys
    _edit(
        path,
        "    history = {\n"
        "        'train_loss': [],\n"
        "        'train_aug_loss': [],\n"
        "        'train_friend_loss': [],\n"
        "        'monitor_val_loss': [],\n"
        "        'lr': [],\n"
        "        'supervision_schedule': [],\n"
        "    }",
        "    history = {\n"
        "        'train_loss': [],\n"
        "        'train_aug_loss': [],\n"
        "        'train_friend_loss': [],\n"
        "        'train_vic_var': [],          # [VICReg]\n"
        "        'train_vic_cov': [],          # [VICReg]\n"
        "        'effective_rank': [],         # [VICReg] RankMe diagnostic\n"
        "        'monitor_val_loss': [],\n"
        "        'lr': [],\n"
        "        'supervision_schedule': [],\n"
        "    }",
        label="history dict",
    )

    # 4f) per-epoch accumulators
    _edit(
        path,
        "        train_loss = 0.0\n"
        "        train_aug_loss = 0.0\n"
        "        train_friend_loss = 0.0\n"
        "        train_friend_batches = 0",
        "        train_loss = 0.0\n"
        "        train_aug_loss = 0.0\n"
        "        train_friend_loss = 0.0\n"
        "        train_friend_batches = 0\n"
        "        train_vic_var = 0.0           # [VICReg]\n"
        "        train_vic_cov = 0.0           # [VICReg]\n"
        "        _rank_buffer = []             # [VICReg] accumulate online projections for RankMe",
        label="accumulators",
    )

    # 4g) L_aug forward -> capture online projections
    _edit(
        path,
        "            # L_aug: ALL samples\n"
        "            pred1_t, pred2_t, proj1_t, proj2_t = fold_model(x1, x1_aug)\n"
        "            loss_trans = byol_loss(pred1_t, pred2_t, proj1_t, proj2_t)\n"
        "            loss = loss_trans",
        "            # L_aug: ALL samples\n"
        "            # [VICReg] also retrieve the online projections (oproj*) for regularisation\n"
        "            pred1_t, pred2_t, proj1_t, proj2_t, oproj1_t, oproj2_t = fold_model(\n"
        "                x1, x1_aug, return_online_proj=True)\n"
        "            loss_trans = byol_loss(pred1_t, pred2_t, proj1_t, proj2_t)\n"
        "            loss = loss_trans",
        label="L_aug forward",
    )

    # 4h) add VICReg term AFTER the (1+sw) normalisation, plus RankMe buffer
    _edit(
        path,
        "                loss = (loss_trans + current_supervision_weight * loss_friend) / (1 + current_supervision_weight)\n"
        "                train_friend_loss += loss_friend.item()\n"
        "                train_friend_batches += 1",
        "                loss = (loss_trans + current_supervision_weight * loss_friend) / (1 + current_supervision_weight)\n"
        "                train_friend_loss += loss_friend.item()\n"
        "                train_friend_batches += 1\n"
        "\n"
        "            # [VICReg] anti-collapse term, added OUTSIDE the (1+sw) normalisation so\n"
        "            # its strength does not shrink as the supervision weight grows.\n"
        "            if VICREG_VAR_WEIGHT > 0 or VICREG_COV_WEIGHT > 0:\n"
        "                _v1, _c1 = vicreg_var_cov_loss(oproj1_t, gamma=VICREG_GAMMA)\n"
        "                _v2, _c2 = vicreg_var_cov_loss(oproj2_t, gamma=VICREG_GAMMA)\n"
        "                _vic_var = 0.5 * (_v1 + _v2)\n"
        "                _vic_cov = 0.5 * (_c1 + _c2)\n"
        "                loss = loss + VICREG_VAR_WEIGHT * _vic_var + VICREG_COV_WEIGHT * _vic_cov\n"
        "                train_vic_var += _vic_var.item()\n"
        "                train_vic_cov += _vic_cov.item()\n"
        "\n"
        "            # [VICReg] accumulate projections for the per-epoch RankMe metric\n"
        "            if len(_rank_buffer) * x1.size(0) < 4096:\n"
        "                _rank_buffer.append(oproj1_t.detach().float().cpu())",
        label="VICReg loss + RankMe buffer",
    )

    # 4i) epoch averages + RankMe
    _edit(
        path,
        "        avg_train_loss = train_loss / len(train_loader)\n"
        "        avg_train_aug_loss = train_aug_loss / len(train_loader)\n"
        "        avg_train_friend_loss = train_friend_loss / train_friend_batches if train_friend_batches > 0 else 0.0",
        "        avg_train_loss = train_loss / len(train_loader)\n"
        "        avg_train_aug_loss = train_aug_loss / len(train_loader)\n"
        "        avg_train_friend_loss = train_friend_loss / train_friend_batches if train_friend_batches > 0 else 0.0\n"
        "        # [VICReg] epoch means + RankMe effective rank of the online projections\n"
        "        avg_vic_var = train_vic_var / len(train_loader)\n"
        "        avg_vic_cov = train_vic_cov / len(train_loader)\n"
        "        epoch_rank = (effective_rank(torch.cat(_rank_buffer, dim=0))\n"
        "                      if _rank_buffer else float('nan'))",
        label="epoch averages",
    )

    # 4j) history append
    _edit(
        path,
        "        history['train_loss'].append(avg_train_loss)\n"
        "        history['train_aug_loss'].append(avg_train_aug_loss)\n"
        "        history['train_friend_loss'].append(avg_train_friend_loss)\n"
        "        history['monitor_val_loss'].append(avg_monitor_loss)\n"
        "        history['lr'].append(current_lr)\n"
        "        history['supervision_schedule'].append(current_supervision_weight)",
        "        history['train_loss'].append(avg_train_loss)\n"
        "        history['train_aug_loss'].append(avg_train_aug_loss)\n"
        "        history['train_friend_loss'].append(avg_train_friend_loss)\n"
        "        history['train_vic_var'].append(avg_vic_var)              # [VICReg]\n"
        "        history['train_vic_cov'].append(avg_vic_cov)              # [VICReg]\n"
        "        history['effective_rank'].append(epoch_rank)             # [VICReg]\n"
        "        history['monitor_val_loss'].append(avg_monitor_loss)\n"
        "        history['lr'].append(current_lr)\n"
        "        history['supervision_schedule'].append(current_supervision_weight)",
        label="history append",
    )

    # 4k) print line
    _edit(
        path,
        '        print(f"Epoch {epoch+1:>4}/{NUM_EPOCHS} | {_loss_str}{mon_str} | lr: {current_lr:.2e}{sup_str}")',
        '        # [VICReg] append VICReg means + RankMe to the per-epoch log line\n'
        '        _vic_str = (f" | vic_var: {avg_vic_var:.4f} vic_cov: {avg_vic_cov:.4f}"\n'
        '                    if (VICREG_VAR_WEIGHT > 0 or VICREG_COV_WEIGHT > 0) else "")\n'
        '        _rank_str = f" | rank: {epoch_rank:.1f}" if epoch_rank == epoch_rank else ""\n'
        '        print(f"Epoch {epoch+1:>4}/{NUM_EPOCHS} | {_loss_str}{mon_str} | lr: {current_lr:.2e}{sup_str}{_vic_str}{_rank_str}")',
        label="print line",
    )

    # 4l) checkpoint config
    _edit(
        path,
        "            'supervision_weight': SUPERVISION_WEIGHT,\n"
        "            'supervision_weight_schedule': SUPERVISION_WEIGHT_SCHEDULE,\n"
        "            'projector': PROJECTOR,\n"
        "            'label_type': args.label_type,",
        "            'supervision_weight': SUPERVISION_WEIGHT,\n"
        "            'supervision_weight_schedule': SUPERVISION_WEIGHT_SCHEDULE,\n"
        "            'projector': PROJECTOR,\n"
        "            'vicreg_var_weight': VICREG_VAR_WEIGHT,   # [VICReg]\n"
        "            'vicreg_cov_weight': VICREG_COV_WEIGHT,   # [VICReg]\n"
        "            'vicreg_gamma': VICREG_GAMMA,             # [VICReg]\n"
        "            'label_type': args.label_type,",
        label="checkpoint config",
    )


def main():
    global CHECK_ONLY
    ap = argparse.ArgumentParser(description="Apply VICReg + RankMe patch to SUPLAT")
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="Repo root (default: current directory)")
    ap.add_argument("--check", action="store_true",
                    help="Dry-run: verify all anchors exist, change nothing")
    args = ap.parse_args()
    CHECK_ONLY = args.check

    root = args.root
    must_exist = root / "scripts/train_byol.py"
    if not must_exist.exists():
        raise SystemExit(f"ABORT: {must_exist} not found. Run from the repo root or pass --root.")

    print(f"{'CHECK' if CHECK_ONLY else 'APPLY'} VICReg patch  (root={root.resolve()})\n")
    patch_trainer(root)
    patch_trainer_init(root)
    patch_models(root)
    patch_train(root)
    print("\nDone." + ("  (no files changed)" if CHECK_ONLY else ""))


if __name__ == "__main__":
    main()