import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from functools import wraps
from torchvision import models

# =============================================================================
# HELPER FUNCTIONS (for original model)
# =============================================================================
'''
def default(val, def_val):
    """Return default value if val is None"""
    return def_val if val is None else val
'''
def flatten(t):
    """Flatten tensor to (batch_size, features)"""
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    """Decorator for singleton pattern - creates instance once and caches it"""
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance
            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    """Get device of first parameter in module"""
    return next(module.parameters()).device

def set_requires_grad(model, val):
    """Set requires_grad for all parameters in model"""
    for p in model.parameters():
        p.requires_grad = val


# =============================================================================
# EMA CLASS (for original model)
# =============================================================================

class EMA():
    """Exponential moving average updater"""
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        """Update EMA: new_avg = beta * old + (1 - beta) * new"""
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    """Update all parameters in ma_model using EMA from current_model"""
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# =============================================================================
# MLP HEADS
# =============================================================================

def MLP(dim, projection_size, hidden_size=4096, bn_momentum=0.1):
    """Standard MLP projection head for BYOL"""
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size, momentum=bn_momentum),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

# =============================================================================
# BYOL ARCHITECTURES
# =============================================================================

def create_efficientnet_b0_backbone(num_channels=1, img_size=89, dropout_rate=0.2):
    """
    EfficientNet-B0 backbone for single-channel radio images.
    Outputs 1280-dim representation (vs 512 from BYOLEncoder).
    Initialised with ImageNet weights; first conv adapted to num_channels;
    classifier head removed (replaced with Dropout if dropout_rate > 0).
    """
    # Load with ImageNet weights
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace first conv: 3-channel RGB → num_channels
    original_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        num_channels,
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False
    )
    # Initialise by averaging the pretrained RGB weights across channels
    with torch.no_grad():
        new_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
    model.features[0][0] = new_conv

    # Replace classifier with Dropout
    model.classifier = nn.Dropout(p=dropout_rate)

    return model

def create_resnet50_backbone(num_channels=1, dropout_rate=0.2):
    """ResNet-50 backbone for single-channel images. Outputs 2048-dim."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    orig = model.conv1
    new_conv = nn.Conv2d(num_channels, orig.out_channels,
                         kernel_size=orig.kernel_size, stride=orig.stride,
                         padding=orig.padding, bias=False)
    with torch.no_grad():
        new_conv.weight.copy_(orig.weight.mean(dim=1, keepdim=True))
    model.conv1 = new_conv
    model.fc = nn.Dropout(p=dropout_rate)
    return model, 2048

def create_resnet18_backbone(num_channels=1, dropout_rate=0.2):
    """ResNet-18 backbone for single-channel images. Outputs 512-dim."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    orig = model.conv1
    new_conv = nn.Conv2d(num_channels, orig.out_channels,
                         kernel_size=orig.kernel_size, stride=orig.stride,
                         padding=orig.padding, bias=False)
    with torch.no_grad():
        new_conv.weight.copy_(orig.weight.mean(dim=1, keepdim=True))
    model.conv1 = new_conv
    model.fc = nn.Dropout(p=dropout_rate)
    return model, 512

def create_convnext_tiny_backbone(num_channels=1, dropout_rate=0.2):
    """ConvNeXt-Tiny backbone for single-channel images. Outputs 768-dim."""
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    orig = model.features[0][0]
    new_conv = nn.Conv2d(num_channels, orig.out_channels,
                         kernel_size=orig.kernel_size, stride=orig.stride,
                         padding=orig.padding, bias=orig.bias is not None)
    with torch.no_grad():
        new_conv.weight.copy_(orig.weight.mean(dim=1, keepdim=True))
        if orig.bias is not None:
            new_conv.bias.copy_(orig.bias)
    model.features[0][0] = new_conv
    model.classifier = nn.Dropout(p=dropout_rate)
    return model, 768

# Shared encoder architecture
class BYOLEncoder(nn.Module):
    """
    ResNet-style encoder (f_θ in BYOL paper) for 89x89 greyscale images.
    Outputs 512-dimensional representation (y_θ in BYOL vocabulary).
    """
    def __init__(self, bn_momentum=0.1, dropout_rate=0.2):
        super().__init__()

        # Initial conv: 89x89 -> 45x45
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)

        # Residual blocks: 45x45 -> 23x23 -> 12x12 -> 6x6
        self.layer1 = self._make_layer(64, 128, stride=2, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(128, 256, stride=2, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(256, 512, stride=2, bn_momentum=bn_momentum)

        # Global pooling: 6x6 -> 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)

    def _make_layer(self, in_channels, out_channels, stride=1, bn_momentum=0.1):
        """Create a residual block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global pooling - output: representation y (batch, 512)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        return x


# Efficient model components
class ProjectionHead(nn.Module):
    """
    MLP projection head (g_θ in BYOL paper).
    Projects representation y to projection z.
    """
    def __init__(self, in_dim=512, hidden_dim=4096, out_dim=256, bn_momentum=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class PredictionHead(nn.Module):
    """
    MLP prediction head (q_θ in BYOL paper).
    Predicts target projection from online projection.
    Only exists in online network (asymmetry prevents collapse).
    """
    def __init__(self, in_dim=256, hidden_dim=4096, out_dim=256, bn_momentum=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class PCAProjection(nn.Module):
    """
    Fixed (non-trainable) PCA projection.
    Reduces encoder representations to the minimum number of principal components
    needed to explain `variance_threshold` of the total variance.
    Must be fitted on encoder outputs via fit() before use.
    """
    def __init__(self, variance_threshold=0.95):
        super().__init__()
        self.variance_threshold = variance_threshold
        # Pre-register as None so load_state_dict can populate them from a checkpoint
        self.register_buffer('mean_', None)
        self.register_buffer('active_mask_', None)
        self.register_buffer('components_', None)

    def fit(self, X: torch.Tensor):
        """
        Compute PCA on representations X of shape (N, D).
        Selects fewest components needed to reach variance_threshold.
        """
        X = X.detach().float()
        mean = X.mean(dim=0)
        X_centered = X - mean
        # Drop zero-variance features to avoid LAPACK SLASCL degenerate-matrix error
        active = X_centered.var(dim=0) > 1e-8
        n_dropped = int((~active).sum().item())
        if n_dropped > 0:
            print(f"  [PCA] Dropped {n_dropped}/{active.numel()} zero-variance features")
        self.register_buffer('mean_', mean)
        self.register_buffer('active_mask_', active)
        _, S, Vh = torch.linalg.svd(X_centered[:, active], full_matrices=False)
        cumvar = (S ** 2).cumsum(0) / (S ** 2).sum()
        n_components = int((cumvar < self.variance_threshold).sum().item()) + 1
        self.register_buffer('components_', Vh[:n_components])  # (n_components, n_active)

    @property
    def out_dim(self) -> int:
        if self.components_ is None:
            raise RuntimeError("PCAProjection has not been fitted yet")
        return self.components_.shape[0]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # PyTorch cannot copy_() into None-registered buffers, so assign them
        # directly before the normal load path runs.
        for name in ('mean_', 'active_mask_', 'components_'):
            key = prefix + name
            if key in state_dict and state_dict[key] is not None:
                self._buffers[name] = state_dict[key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, x):
        if self.components_ is None:
            raise RuntimeError("PCAProjection must be fitted before use; call fit_pca() first")
        return (x.float() - self.mean_)[:, self.active_mask_] @ self.components_.T


class BYOLEfficientNetB0(nn.Module):
    """
    BYOL model using EfficientNet-B0 as backbone.

    Architecture:
    - Encoder f_θ: EfficientNet-B0 (1-channel input) → 1280-dim representation
    - Projector g_θ: representation → compressed projection (feature_compression_mode)
    - Predictor q_θ: projection → prediction [online only]

    feature_compression_mode:
        'pca'  (default) — PCA keeping 95% variance; call fit_pca() before training
        'mlp'            — learned MLP projection head (projection_dim output)
        'none'           — no compression; predictor operates on raw 1280-dim repr
    """
    def __init__(self, projection_dim=256, hidden_dim=4096, bn_momentum=0.1,
                 feature_compression_mode='pca', dropout_rate=0.2):
        super().__init__()
        encoder_dim = 1280
        self.feature_compression_mode = feature_compression_mode

        self.online_encoder = create_efficientnet_b0_backbone(num_channels=1, dropout_rate=dropout_rate)

        if feature_compression_mode == 'mlp':
            self.online_projector = ProjectionHead(
                in_dim=encoder_dim, hidden_dim=hidden_dim,
                out_dim=projection_dim, bn_momentum=bn_momentum
            )
            self.online_predictor = PredictionHead(
                in_dim=projection_dim, hidden_dim=hidden_dim,
                out_dim=projection_dim, bn_momentum=bn_momentum
            )
        elif feature_compression_mode == 'pca':
            self.online_projector = PCAProjection(variance_threshold=0.95)
            self.online_predictor = None  # built in fit_pca() once output dim is known
            self._predictor_config = (hidden_dim, bn_momentum)
        elif feature_compression_mode == 'none':
            self.online_projector = nn.Identity()
            self.online_predictor = PredictionHead(
                in_dim=encoder_dim, hidden_dim=hidden_dim,
                out_dim=encoder_dim, bn_momentum=bn_momentum
            )
        else:
            raise ValueError(f"Unknown feature_compression_mode: {feature_compression_mode!r}. "
                             "Choose from 'pca', 'mlp', 'none'.")

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def fit_pca(self, encoder_outputs: torch.Tensor):
        """
        Fit PCA projectors on encoder outputs and build the predictor.
        Must be called before training when feature_compression_mode='pca'.

        Args:
            encoder_outputs: (N, 1280) tensor of encoder representations,
                             typically collected over one pass through the dataset.
        """
        assert self.feature_compression_mode == 'pca', \
            "fit_pca() should only be called when feature_compression_mode='pca'"
        self.online_projector.fit(encoder_outputs)
        # Target gets the same fixed PCA transform
        self.target_projector = copy.deepcopy(self.online_projector)
        for param in self.target_projector.parameters():
            param.requires_grad = False
        # Build predictor now that PCA output dim is known
        hidden_dim, bn_momentum = self._predictor_config
        pca_dim = self.online_projector.out_dim
        self.online_predictor = PredictionHead(
            in_dim=pca_dim, hidden_dim=hidden_dim,
            out_dim=pca_dim, bn_momentum=bn_momentum
        )

    def forward(self, x1, x2):
        if self.feature_compression_mode == 'pca' and self.online_predictor is None:
            raise RuntimeError("Call fit_pca() before the first forward pass")

        online_repr_1 = self.online_encoder(x1)
        online_repr_2 = self.online_encoder(x2)

        online_proj_1 = self.online_projector(online_repr_1)
        online_proj_2 = self.online_projector(online_repr_2)

        online_pred_1 = self.online_predictor(online_proj_1)
        online_pred_2 = self.online_predictor(online_proj_2)

        with torch.no_grad():
            target_repr_1 = self.target_encoder(x1)
            target_repr_2 = self.target_encoder(x2)

            target_proj_1 = self.target_projector(target_repr_1)
            target_proj_2 = self.target_projector(target_repr_2)

        return online_pred_1, online_pred_2, target_proj_1, target_proj_2

    @torch.no_grad()
    def update_target_network(self, momentum=0.996):
        for online_params, target_params in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_params.data = (
                momentum * target_params.data + (1 - momentum) * online_params.data
            )

        # PCA projector is fixed — no EMA update needed
        if self.feature_compression_mode != 'pca':
            for online_params, target_params in zip(
                self.online_projector.parameters(), self.target_projector.parameters()
            ):
                target_params.data = (
                    momentum * target_params.data + (1 - momentum) * online_params.data
                )


class BYOLPretrainedBackbone(nn.Module):
    """
    Generic BYOL wrapper for any pretrained torchvision backbone.
    Mirrors BYOLEfficientNetB0 exactly; accepts backbone and encoder_dim as arguments.
    """
    def __init__(self, backbone, encoder_dim, projection_dim=256, hidden_dim=4096,
                 bn_momentum=0.1, feature_compression_mode='pca'):
        super().__init__()
        self.feature_compression_mode = feature_compression_mode

        self.online_encoder = backbone

        if feature_compression_mode == 'mlp':
            self.online_projector = ProjectionHead(
                in_dim=encoder_dim, hidden_dim=hidden_dim,
                out_dim=projection_dim, bn_momentum=bn_momentum
            )
            self.online_predictor = PredictionHead(
                in_dim=projection_dim, hidden_dim=hidden_dim,
                out_dim=projection_dim, bn_momentum=bn_momentum
            )
        elif feature_compression_mode == 'pca':
            self.online_projector = PCAProjection(variance_threshold=0.95)
            self.online_predictor = None
            self._predictor_config = (hidden_dim, bn_momentum)
        elif feature_compression_mode == 'none':
            self.online_projector = nn.Identity()
            self.online_predictor = PredictionHead(
                in_dim=encoder_dim, hidden_dim=hidden_dim,
                out_dim=encoder_dim, bn_momentum=bn_momentum
            )
        else:
            raise ValueError(f"Unknown feature_compression_mode: {feature_compression_mode!r}. "
                             "Choose from 'pca', 'mlp', 'none'.")

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def fit_pca(self, encoder_outputs: torch.Tensor):
        assert self.feature_compression_mode == 'pca', \
            "fit_pca() should only be called when feature_compression_mode='pca'"
        self.online_projector.fit(encoder_outputs)
        self.target_projector = copy.deepcopy(self.online_projector)
        for param in self.target_projector.parameters():
            param.requires_grad = False
        hidden_dim, bn_momentum = self._predictor_config
        pca_dim = self.online_projector.out_dim
        self.online_predictor = PredictionHead(
            in_dim=pca_dim, hidden_dim=hidden_dim,
            out_dim=pca_dim, bn_momentum=bn_momentum
        )

    def forward(self, x1, x2):
        if self.feature_compression_mode == 'pca' and self.online_predictor is None:
            raise RuntimeError("Call fit_pca() before the first forward pass")

        online_repr_1 = self.online_encoder(x1)
        online_repr_2 = self.online_encoder(x2)

        online_proj_1 = self.online_projector(online_repr_1)
        online_proj_2 = self.online_projector(online_repr_2)

        online_pred_1 = self.online_predictor(online_proj_1)
        online_pred_2 = self.online_predictor(online_proj_2)

        with torch.no_grad():
            target_repr_1 = self.target_encoder(x1)
            target_repr_2 = self.target_encoder(x2)

            target_proj_1 = self.target_projector(target_repr_1)
            target_proj_2 = self.target_projector(target_repr_2)

        return online_pred_1, online_pred_2, target_proj_1, target_proj_2

    @torch.no_grad()
    def update_target_network(self, momentum=0.996):
        for online_params, target_params in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_params.data = (
                momentum * target_params.data + (1 - momentum) * online_params.data
            )

        if self.feature_compression_mode != 'pca':
            for online_params, target_params in zip(
                self.online_projector.parameters(), self.target_projector.parameters()
            ):
                target_params.data = (
                    momentum * target_params.data + (1 - momentum) * online_params.data
                )


class BYOLEfficient(nn.Module):
    """
    Efficient BYOL model (Document 2 style).

    Architecture:
    - Encoder f_θ: input → representation y (512-dim)
    - Projector g_θ: representation y → compressed projection (feature_compression_mode)
    - Predictor q_θ: projection → prediction [online only]

    feature_compression_mode:
        'pca'  (default) — PCA keeping 95% variance; call fit_pca() before training
        'mlp'            — learned MLP projection head (projection_dim output)
        'none'           — no compression; predictor operates on raw 512-dim repr
    """
    def __init__(self, encoder_dim=512, projection_dim=256, hidden_dim=4096, bn_momentum=0.1,
                 feature_compression_mode='pca', dropout_rate=0.2):
        super().__init__()
        self.feature_compression_mode = feature_compression_mode

        # Online network: encoder → projector → predictor
        self.online_encoder = BYOLEncoder(bn_momentum=bn_momentum, dropout_rate=dropout_rate)

        if feature_compression_mode == 'mlp':
            self.online_projector = ProjectionHead(
                in_dim=encoder_dim, hidden_dim=hidden_dim,
                out_dim=projection_dim, bn_momentum=bn_momentum
            )
            self.online_predictor = PredictionHead(
                in_dim=projection_dim, hidden_dim=hidden_dim,
                out_dim=projection_dim, bn_momentum=bn_momentum
            )
        elif feature_compression_mode == 'pca':
            self.online_projector = PCAProjection(variance_threshold=0.95)
            self.online_predictor = None  # built in fit_pca() once output dim is known
            self._predictor_config = (hidden_dim, bn_momentum)
        elif feature_compression_mode == 'none':
            self.online_projector = nn.Identity()
            self.online_predictor = PredictionHead(
                in_dim=encoder_dim, hidden_dim=hidden_dim,
                out_dim=encoder_dim, bn_momentum=bn_momentum
            )
        else:
            raise ValueError(f"Unknown feature_compression_mode: {feature_compression_mode!r}. "
                             "Choose from 'pca', 'mlp', 'none'.")

        # Target network: encoder → projector (no predictor!)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Freeze target network parameters (updated via EMA only)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def fit_pca(self, encoder_outputs: torch.Tensor):
        """
        Fit PCA projectors on encoder outputs and build the predictor.
        Must be called before training when feature_compression_mode='pca'.

        Args:
            encoder_outputs: (N, 512) tensor of encoder representations,
                             typically collected over one pass through the dataset.
        """
        assert self.feature_compression_mode == 'pca', \
            "fit_pca() should only be called when feature_compression_mode='pca'"
        self.online_projector.fit(encoder_outputs)
        # Target gets the same fixed PCA transform
        self.target_projector = copy.deepcopy(self.online_projector)
        for param in self.target_projector.parameters():
            param.requires_grad = False
        # Build predictor now that PCA output dim is known
        hidden_dim, bn_momentum = self._predictor_config
        pca_dim = self.online_projector.out_dim
        self.online_predictor = PredictionHead(
            in_dim=pca_dim, hidden_dim=hidden_dim,
            out_dim=pca_dim, bn_momentum=bn_momentum
        )

    def forward(self, x1, x2):
        """
        Forward pass for two augmented views.

        Returns:
            online_pred_1: prediction from view 1
            online_pred_2: prediction from view 2
            target_proj_1: projection from view 1 (detached)
            target_proj_2: projection from view 2 (detached)
        """
        if self.feature_compression_mode == 'pca' and self.online_predictor is None:
            raise RuntimeError("Call fit_pca() before the first forward pass")

        # Online network forward pass
        online_repr_1 = self.online_encoder(x1)           # (batch, 512)
        online_repr_2 = self.online_encoder(x2)           # (batch, 512)

        online_proj_1 = self.online_projector(online_repr_1)
        online_proj_2 = self.online_projector(online_repr_2)

        online_pred_1 = self.online_predictor(online_proj_1)
        online_pred_2 = self.online_predictor(online_proj_2)

        # Target network forward pass (no gradients)
        with torch.no_grad():
            target_repr_1 = self.target_encoder(x1)
            target_repr_2 = self.target_encoder(x2)

            target_proj_1 = self.target_projector(target_repr_1)
            target_proj_2 = self.target_projector(target_repr_2)

        return online_pred_1, online_pred_2, target_proj_1, target_proj_2

    @torch.no_grad()
    def update_target_network(self, momentum=0.996):
        """
        Exponential moving average (EMA) update of target network.

        Target parameters: θ_target = m * θ_target + (1 - m) * θ_online
        """
        for online_params, target_params in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_params.data = (
                momentum * target_params.data + (1 - momentum) * online_params.data
            )

        # PCA projector is fixed — no EMA update needed
        if self.feature_compression_mode != 'pca':
            for online_params, target_params in zip(
                self.online_projector.parameters(), self.target_projector.parameters()
            ):
                target_params.data = (
                    momentum * target_params.data + (1 - momentum) * online_params.data
                )


# Original (snippet-style) model components
class NetWrapper(nn.Module):
    """
    Wrapper for base network following snippet pattern.
    Manages projection head and representation extraction.
    """
    def __init__(self, net, projection_size, projection_hidden_size, bn_momentum=0.1):
        super().__init__()
        self.net = net
        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.bn_momentum = bn_momentum

    @singleton('projector')
    def _get_projector(self, hidden):
        """Create projector on first forward pass (singleton pattern)"""
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size, self.bn_momentum)
        return projector.to(hidden)

    def get_representation(self, x):
        """Extract features from encoder"""
        return self.net(x)

    def forward(self, x, return_projection=True):
        """
        Forward pass through encoder and optionally projector.

        Args:
            x: Input tensor
            return_projection: If True, return (projection, representation)
                             If False, return representation only
        """
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        # Get or create projector (singleton pattern ensures single creation)
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


class BYOLOriginal(nn.Module):
    """
    Original BYOL model (Document 3 style, snippet-based).
    Uses NetWrapper with singleton pattern and EMA class.
    """
    def __init__(
        self,
        net,
        image_size,
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99,
        use_momentum=True,
        bn_momentum=0.1
    ):
        super().__init__()
        self.net = net

        # Online encoder with projection head
        self.online_encoder = NetWrapper(
            net,
            projection_size,
            projection_hidden_size,
            bn_momentum=bn_momentum
        )

        # Target encoder (EMA of online)
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        # Predictor (only for online network)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size, bn_momentum)

        # Move to correct device
        device = get_module_device(net)
        self.to(device)

        # Initialize singleton parameters with mock forward pass
        self.forward(torch.randn(2, 1, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        """Create target encoder as deepcopy of online encoder (singleton)"""
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        """Reset target encoder (forces recreation on next forward)"""
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        """Update target network using EMA"""
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, x, return_embedding=False, return_projection=True):
        """
        Forward pass for BYOL.

        Args:
            x: Input batch (concatenated pair [view1; view2] for training)
            return_embedding: If True, return embeddings instead of loss
            return_projection: If True (with return_embedding), return projections too
        """
        # If requesting embeddings, just return from online encoder
        if return_embedding:
            return self.online_encoder(x, return_projection=return_projection)

        # Training mode: x should be concatenated pair [view1; view2]
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        # Forward through online network
        online_projections, _ = self.online_encoder(x)
        online_predictions = self.online_predictor(online_projections)

        # Split predictions into two views
        online_pred_one, online_pred_two = online_predictions.chunk(2, dim=0)

        # Forward through target network (no gradients)
        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_projections, _ = target_encoder(x)
            target_projections = target_projections.detach()

            target_proj_one, target_proj_two = target_projections.chunk(2, dim=0)

        # Compute symmetric loss
        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()

# =============================================================================
# LOSS FUNCTION (for original model)
# =============================================================================

def loss_fn(x, y):
    """
    BYOL loss: normalized MSE between prediction and target
    Returns per-sample loss (not reduced)
    """
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)