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


class QuarterTurnRotation:
    """Rotate by a randomly chosen quarter turn: 0, 90, 180, or 270 degrees."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(x, k, dims=[-2, -1])

    def __repr__(self) -> str:
        return "QuarterTurnRotation()"


# =============================================================================
# AUGMENTATION PIPELINES
# =============================================================================

def get_augmentation(name: str) -> T.Compose:
    """
    Return a named augmentation pipeline.

    Args:
        name: 'quart', 'cont', 'quart_ext', or 'cont_ext'
            quart    — flip + quarter-turn rotation (lossless)
            cont     — flip + continuous rotation
            quart_ext — crop + flip + quarter-turn rotation + noise + intensity scaling
            cont_ext  — crop + flip + continuous rotation + noise + intensity scaling

    Returns:
        T.Compose pipeline
    """
    if name == "quart":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            QuarterTurnRotation(),
        ])
    elif name == "cont":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180),
        ])
    elif name == "quart_ext":
        return T.Compose([
            T.RandomResizedCrop(89, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            QuarterTurnRotation(),
            GaussianNoise(std=0.05),
            IntensityScaling(scale=0.2),
        ])
    elif name == "cont_ext":
        return T.Compose([
            T.RandomResizedCrop(89, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180),
            GaussianNoise(std=0.05),
            IntensityScaling(scale=0.2),
        ])
    else:
        raise ValueError(f"Unknown augmentation: '{name}'. Choose from: quart, cont, quart_ext, cont_ext")
