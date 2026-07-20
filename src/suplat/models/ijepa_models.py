"""
I-JEPA model components.

Architecture (ViT-Tiny defaults):
  - PatchEmbed : Conv2d(1, D, 8, 8) → (B, 144, D)
  - IJEPAViT   : PatchEmbed + pos_embed + N × TransformerBlock, D=192
  - JEPAPredictor : narrow ViT (D_pred=96) that predicts target tokens from context
  - IJEPAModel : online_encoder + target_encoder (EMA) + predictor

References: Assran et al. 2023, "Self-Supervised Learning from Images with a
            Joint-Embedding Predictive Architecture" (I-JEPA).
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    """Standard pre-norm transformer block (attention + MLP)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention (self)
        x_n = self.norm1(x)
        attn_out, _ = self.attn(x_n, x_n, x_n)
        x = x + attn_out
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Linear patch embedding via a strided Conv2d.

    Input  : (B, in_chans, img_size, img_size)
    Output : (B, n_patches, embed_dim)
    """

    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 192,
    ):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size {img_size} must be divisible by patch_size {patch_size}"
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, D, H', W') → (B, n_patches, D)
        return self.proj(x).flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# IJEPAViT — context / target encoder
# ---------------------------------------------------------------------------

class IJEPAViT(nn.Module):
    """
    Lightweight ViT encoder for I-JEPA.

    Default (ViT-Tiny): D=192, depth=9, heads=3, mlp_ratio=4.
    No [CLS] token; returns all patch tokens.

    Parameters
    ----------
    img_size : int
        Spatial size of (square) input images. Must be divisible by patch_size.
    patch_size : int
        Patch side length in pixels.
    in_chans : int
        Number of input channels.
    embed_dim : int
        Token embedding dimension (D).
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        MLP hidden-dim multiplier.
    """

    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 192,
        depth: int = 9,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.n_patches   = self.patch_embed.n_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(
            *[_TransformerBlock(embed_dim, num_heads, mlp_ratio)
              for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, patch_ids: list = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, C, H, W)
        patch_ids : list[int] or None
            If given, return only tokens at these positions.

        Returns
        -------
        tokens : Tensor (B, n_patches, D) or (B, |patch_ids|, D)
        """
        tokens = self.patch_embed(x) + self.pos_embed          # (B, N, D)
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        if patch_ids is not None:
            tokens = tokens[:, patch_ids, :]
        return tokens

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool all patch tokens to a single vector (B, D).
        Used for linear probe and downstream eval.
        """
        return self.forward(x).mean(dim=1)                      # (B, D)


# ---------------------------------------------------------------------------
# JEPAPredictor — narrow ViT that predicts target tokens from context
# ---------------------------------------------------------------------------

class JEPAPredictor(nn.Module):
    """
    Narrow ViT predictor following I-JEPA §3.3.

    Takes context tokens from the online encoder (with their positions) plus
    learnable mask tokens at target positions, runs them through a shallow ViT,
    and outputs predicted embeddings at the target positions.

    Parameters
    ----------
    encoder_dim : int
        Dimension D of the (full) encoder tokens.
    pred_dim : int
        Internal dimension of the predictor (narrower than encoder_dim).
    depth : int
        Number of transformer blocks in the predictor.
    num_heads : int
        Attention heads.
    n_patches : int
        Total number of patches in the grid (144 for 96×96 / 8×8).
    """

    def __init__(
        self,
        encoder_dim: int = 192,
        pred_dim: int = 96,
        depth: int = 6,
        num_heads: int = 3,
        n_patches: int = 144,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.pred_dim    = pred_dim
        self.n_patches   = n_patches

        # Project encoder tokens into predictor space
        self.input_proj  = nn.Linear(encoder_dim, pred_dim)
        # Learnable mask token (replaces target positions)
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Positional embeddings in predictor space (one per patch in the full grid)
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches, pred_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(
            *[_TransformerBlock(pred_dim, num_heads, mlp_ratio=4.0)
              for _ in range(depth)]
        )
        self.norm        = nn.LayerNorm(pred_dim)
        # Project back to encoder dimension
        self.output_proj = nn.Linear(pred_dim, encoder_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_ids: list,
        target_ids_list: list,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        context_tokens : Tensor (B, n_ctx, D)
        context_ids    : list[int] — positions of context tokens in the grid
        target_ids_list : list[list[int]] — one list per target block

        Returns
        -------
        predictions : Tensor (B, n_tgt_total, D)
            Predicted embeddings at all target positions
            (blocks concatenated in order).
        """
        B = context_tokens.size(0)
        device = context_tokens.device

        # Collect all target patch ids (keeping order, no dedup)
        all_target_ids = [i for block in target_ids_list for i in block]
        n_tgt = len(all_target_ids)

        # --- Context: project + add positional embedding ---------------------
        ctx = self.input_proj(context_tokens)                       # (B, n_ctx, P)
        ctx_pos = self.pos_embed[:, context_ids, :]                 # (1, n_ctx, P)
        ctx = ctx + ctx_pos

        # --- Target: mask tokens + positional embedding ----------------------
        tgt = self.mask_token.expand(B, n_tgt, -1)                 # (B, n_tgt, P)
        tgt_pos = self.pos_embed[:, all_target_ids, :]              # (1, n_tgt, P)
        tgt = tgt + tgt_pos

        # --- Concatenate and run through transformer -------------------------
        # Context tokens first, then target (mask) tokens
        seq = torch.cat([ctx, tgt], dim=1)                         # (B, n_ctx+n_tgt, P)
        seq = self.blocks(seq)
        seq = self.norm(seq)

        # Extract the target positions (last n_tgt tokens) and project back
        n_ctx = ctx.size(1)
        pred = seq[:, n_ctx:, :]                                    # (B, n_tgt, P)
        return self.output_proj(pred)                               # (B, n_tgt, D)


# ---------------------------------------------------------------------------
# IJEPAModel — online encoder + EMA target + predictor
# ---------------------------------------------------------------------------

class IJEPAModel(nn.Module):
    """
    Full I-JEPA model.

    Attributes
    ----------
    online_encoder : IJEPAViT
    target_encoder : IJEPAViT (frozen EMA copy)
    predictor      : JEPAPredictor
    """

    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 192,
        depth: int = 9,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        pred_dim: int = 96,
        pred_depth: int = 6,
        pred_heads: int = 3,
    ):
        super().__init__()
        vit_kwargs = dict(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.online_encoder = IJEPAViT(**vit_kwargs)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        n_patches = self.online_encoder.n_patches
        self.predictor = JEPAPredictor(
            encoder_dim=embed_dim,
            pred_dim=pred_dim,
            depth=pred_depth,
            num_heads=pred_heads,
            n_patches=n_patches,
        )

    def forward(
        self,
        x: torch.Tensor,
        context_ids: list,
        target_ids_list: list,
    ) -> tuple:
        """
        Parameters
        ----------
        x               : Tensor (B, C, H, W) — padded images
        context_ids     : list[int]
        target_ids_list : list[list[int]]

        Returns
        -------
        predictions    : Tensor (B, n_tgt_total, D)
        target_tokens  : Tensor (B, n_tgt_total, D) — stop-gradient
        """
        # Online encoder: get context tokens
        all_online = self.online_encoder(x)                     # (B, 144, D)
        ctx_tokens  = all_online[:, context_ids, :]             # (B, n_ctx, D)

        # Predictor: predict target tokens from context
        predictions = self.predictor(ctx_tokens, context_ids, target_ids_list)

        # Target encoder: get target tokens (no gradient)
        all_target_ids = [i for block in target_ids_list for i in block]
        with torch.no_grad():
            all_target = self.target_encoder(x)                 # (B, 144, D)
            target_tokens = all_target[:, all_target_ids, :]    # (B, n_tgt, D)

        return predictions, target_tokens

    @torch.no_grad()
    def update_target_network(self, momentum: float = 0.996):
        """EMA update: θ_target ← m·θ_target + (1−m)·θ_online."""
        for online_p, target_p in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_p.data = (
                momentum * target_p.data + (1.0 - momentum) * online_p.data
            )
