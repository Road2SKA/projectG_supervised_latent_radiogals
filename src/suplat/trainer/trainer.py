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

def byol_loss(
    online_pred_1: torch.Tensor,
    online_pred_2: torch.Tensor,
    target_proj_1: torch.Tensor,
    target_proj_2: torch.Tensor,
) -> torch.Tensor:    
    """BYOL loss for efficient model: normalized Mean Squared Error."""
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
        model_type:   'efficient' or 'original'
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

            if model_type == "efficient":
                representation = model.online_encoder(x1)
                projection = model.online_projector(representation)
            else:  # original
                projection, _ = model(x1, return_embedding=True, return_projection=True)

            all_projections.append(projection.cpu().numpy())

    return np.vstack(all_projections)
