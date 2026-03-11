import torch
import torchvision.transforms as T


# =============================================================================
# CUSTOM TRANSFORMS
# =============================================================================

class GaussianNoise:
    """Add Gaussian noise to a tensor image."""
    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std

    def __repr__(self) -> str:
        return f"GaussianNoise(std={self.std})"


class IntensityScaling:
    """Randomly scale pixel intensities by a factor in [1-scale, 1+scale]."""
    def __init__(self, scale: float = 0.2):
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.scale
        return x * factor

    def __repr__(self) -> str:
        return f"IntensityScaling(scale={self.scale})"


# =============================================================================
# AUGMENTATION PIPELINES
# =============================================================================

def get_augmentation(name: str) -> T.Compose:
    """
    Return a named augmentation pipeline.

    Args:
        name: 'standard' or 'extended'

    Returns:
        T.Compose pipeline
    """
    if name == "standard":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180),
        ])
    elif name == "extended":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180),
            GaussianNoise(std=0.05),
            IntensityScaling(scale=0.2),
        ])
    else:
        raise ValueError(f"Unknown augmentation: '{name}'. Choose from: standard, extended")
