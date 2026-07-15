"""
Generative model components: FlowMatchingUNet decoder with sinusoidal
time embedding and conditional residual blocks.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
