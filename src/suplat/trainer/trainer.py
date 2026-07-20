import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_ema_decay(
    epoch: int,
    num_epochs: int,
    schedule: str = 'constant',
    base_decay: float = 0.996,
    start_decay: float = 0.996,
    end_decay: float = 0.9999,
) -> float:
    """Compute EMA decay rate for current epoch based on schedule."""
    if schedule == 'constant':
        return base_decay
    elif schedule == 'cosine':
        progress = epoch / max(num_epochs - 1, 1)
        return end_decay - (end_decay - start_decay) * (np.cos(np.pi * progress) + 1) / 2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def get_warmup_lr(
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
) -> float:
    """Compute learning rate during warmup phase."""
    if epoch >= warmup_epochs or warmup_epochs == 0:
        return base_lr
    else:
        return base_lr * (epoch + 1) / warmup_epochs


def get_supervision_weight(
    epoch: int,
    num_epochs: int,
    schedule: str = 'constant',
    base_weight: float = 1.0,
    start_weight: float = 0.0,
    end_weight: float = 1.0,
) -> float:
    """Compute supervision weight for current epoch based on curriculum schedule."""
    if schedule == 'constant':
        return base_weight
    elif schedule == 'linear':
        progress = epoch / max(num_epochs - 1, 1)
        return start_weight + (end_weight - start_weight) * progress
    elif schedule == 'cosine':
        progress = epoch / max(num_epochs - 1, 1)
        return start_weight + (end_weight - start_weight) * (1 - np.cos(np.pi * progress)) / 2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

def byol_loss(
    online_pred_1: torch.Tensor,
    online_pred_2: torch.Tensor,
    target_proj_1: torch.Tensor,
    target_proj_2: torch.Tensor,
) -> torch.Tensor:    
    """BYOL loss: normalized Mean Squared Error."""
    # L2 normalize all vectors
    online_pred_1 = F.normalize(online_pred_1, dim=-1, p=2)
    online_pred_2 = F.normalize(online_pred_2, dim=-1, p=2)
    target_proj_1 = F.normalize(target_proj_1, dim=-1, p=2)
    target_proj_2 = F.normalize(target_proj_2, dim=-1, p=2)
    
    # Compute symmetric MSE loss
    loss_1 = (2 - 2 * (online_pred_1 * target_proj_2).sum(dim=-1)).mean()
    loss_2 = (2 - 2 * (online_pred_2 * target_proj_1).sum(dim=-1)).mean()
    
    return loss_1 + loss_2

# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================

def extract_embeddings_from_loader(model, dataloader, model_type, device, max_batches=None):
    """
    Extract projections from a DataLoader using the trained model.

    Args:
        model:        Trained BYOL model
        dataloader:   DataLoader yielding (x1, x1_trans, x2_friend, _) tuples
        model_type:   'efficientnet-b0', 'convnet', or 'original'
        device:       torch.device to run inference on
        max_batches:  Limit number of batches (None = all)

    Returns:
        projections: (N, D) array of projected embeddings
    """
    model.eval()
    all_projections = []

    with torch.no_grad():
        for batch_idx, (x1, x1_trans, x2_friend, _) in enumerate(tqdm(dataloader, desc="Extracting")):
            if max_batches and batch_idx >= max_batches:
                break

            x1 = x1.to(device)

            if model_type in ("convnet", "efficientnet-b0", "resnet18", "resnet50", "convnext-tiny"):
                representation = model.online_encoder(x1)
                projection = model.online_projector(representation)
            else:  # original
                projection, _ = model(x1, return_embedding=True, return_projection=True)

            all_projections.append(projection.cpu().numpy())

    return np.vstack(all_projections)


def extract_raw_embeddings_from_loader(model, dataloader, device, max_batches=None):
    """Extract raw encoder representations (before projector) for all samples."""
    model.eval()
    all_repr = []
    with torch.no_grad():
        for batch_idx, (x1, *_) in enumerate(tqdm(dataloader, desc="Extracting raw")):
            if max_batches and batch_idx >= max_batches:
                break
            all_repr.append(model.online_encoder(x1.to(device)).cpu().numpy())
    return np.vstack(all_repr)


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


# =============================================================================
# TIER-WEIGHTED BYOL LOSS                                        # [TierW]
# =============================================================================

# =============================================================================
# I-JEPA LOSS
# =============================================================================

def jepa_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    I-JEPA prediction loss: MSE between predictor outputs and normalised
    stop-gradient target encoder tokens (Assran et al. 2023, §3.3).

    Fix C: apply per-token LayerNorm to the target patch embeddings before
    computing the MSE.  This removes the constant-shift degree of freedom
    that otherwise makes the zero-loss collapse solution reachable: without
    normalisation, gradient descent can minimise the loss by collapsing all
    representations to a single constant vector (predictions ≈ targets ≈ mean),
    giving MSE → 0 without learning anything useful.  Normalising each target
    token to zero mean and unit variance forces the predictor to match the
    *direction and shape* of the target distribution, not just its mean.

    Args:
        predictions : (B, n_patches, D) — predictor output (not normalised).
        targets     : (B, n_patches, D) — target encoder outputs (no grad).

    Returns:
        Scalar MSE loss averaged over batch, patches, and feature dimensions.
    """
    # Normalise each target token independently over its D-dimensional feature
    # axis.  Predictions are left un-normalised so the predictor must learn to
    # produce correctly-scaled, correctly-directed embeddings.
    targets = F.layer_norm(targets, (targets.size(-1),))
    return F.mse_loss(predictions, targets)


def byol_loss_weighted(
    online_pred_1: torch.Tensor,
    online_pred_2: torch.Tensor,
    target_proj_1: torch.Tensor,
    target_proj_2: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Tier-weighted BYOL loss.

    Computes per-sample symmetric MSE (same formula as byol_loss) then takes a
    weighted mean so that high-tier (rare/interesting) pairs contribute more
    gradient than low-tier (common FRI/FRII) ones.

    Args:
        online_pred_1/2: predictor outputs for the two views, shape (B, D).
        target_proj_1/2: target-network projections (stop-gradient), shape (B, D).
        weights:         per-sample weights, shape (B,). Caller is responsible for
                         normalising (e.g. divide by mean so the expected magnitude
                         matches the unweighted loss).

    Returns:
        Weighted scalar loss.
    """
    p1 = F.normalize(online_pred_1, dim=-1, p=2)
    p2 = F.normalize(online_pred_2, dim=-1, p=2)
    z1 = F.normalize(target_proj_1, dim=-1, p=2)
    z2 = F.normalize(target_proj_2, dim=-1, p=2)
    # Per-sample symmetric MSE: same expression as byol_loss but without .mean()
    per_sample = (2 - 2 * (p1 * z2).sum(dim=-1)) + (2 - 2 * (p2 * z1).sum(dim=-1))
    w = weights / weights.sum().clamp(min=1e-6)
    return (w * per_sample).sum()
