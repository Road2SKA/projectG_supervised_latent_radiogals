"""
I-JEPA mask sampling and dataset.

Implements the multi-block mask strategy from Assran et al. 2023.
Grid: 12×12 = 144 patches (96×96 image, 8×8 patches).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MultiBlockMaskSampler:
    """
    Samples I-JEPA-style context / target masks on a 2-D patch grid.

    Follows §4 defaults from Assran et al. 2023:
      - n_target_blocks=4, each scale 0.15–0.20, aspect 0.75–1.5
      - 1 context block, scale 0.85–1.0, aspect 0.75–1.5,
        with all target positions removed afterward.

    Usage::

        sampler = MultiBlockMaskSampler()
        ctx_ids, tgt_ids_list = sampler.sample()
        # ctx_ids          : list[int], length varies
        # tgt_ids_list     : list[list[int]], length n_target_blocks
    """

    def __init__(
        self,
        grid_size: tuple = (12, 12),
        n_target_blocks: int = 4,
        target_scale: tuple = (0.15, 0.20),
        target_aspect: tuple = (0.75, 1.5),
        context_scale: tuple = (0.85, 1.0),
        context_aspect: tuple = (0.75, 1.5),
    ):
        self.H, self.W = grid_size
        self.n_patches = self.H * self.W
        self.n_target_blocks = n_target_blocks
        self.target_scale = target_scale
        self.target_aspect = target_aspect
        self.context_scale = context_scale
        self.context_aspect = context_aspect

    def _sample_block(self, scale_range: tuple, aspect_range: tuple) -> list:
        """Return a list of patch indices for one randomly sampled rectangle."""
        scale = np.random.uniform(*scale_range)
        aspect = np.random.uniform(*aspect_range)

        # Number of patches, then derive height / width
        n = max(1, int(round(self.n_patches * scale)))
        h = max(1, int(round(np.sqrt(n / aspect))))
        w = max(1, int(round(np.sqrt(n * aspect))))
        h = min(h, self.H)
        w = min(w, self.W)

        # Random top-left corner
        r = np.random.randint(0, max(1, self.H - h + 1))
        c = np.random.randint(0, max(1, self.W - w + 1))

        return [
            (r + dr) * self.W + (c + dc)
            for dr in range(h)
            for dc in range(w)
        ]

    def sample(self) -> tuple:
        """
        Sample one set of context + target masks.

        Returns
        -------
        context_ids : list[int]
            Patch indices that form the context (targets removed).
        target_ids_list : list[list[int]]
            One list of patch indices per target block.
        """
        # --- Target blocks ---------------------------------------------------
        all_target_ids: set = set()
        target_ids_list: list = []
        for _ in range(self.n_target_blocks):
            ids = self._sample_block(self.target_scale, self.target_aspect)
            target_ids_list.append(ids)
            all_target_ids.update(ids)

        # --- Context block (target positions removed) ------------------------
        ctx_ids_raw = self._sample_block(self.context_scale, self.context_aspect)
        context_ids = [i for i in ctx_ids_raw if i not in all_target_ids]

        # Fallback: if context is completely consumed by targets, use complement
        if len(context_ids) == 0:
            context_ids = sorted(set(range(self.n_patches)) - all_target_ids)

        return context_ids, target_ids_list


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IJEPADataset(Dataset):
    """
    Minimal dataset for I-JEPA training.

    Returns ``(img_padded, context_ids, target_ids_list)`` per sample:
      - ``img_padded`` : float32 tensor (1, 96, 96) — reflect-padded from 89×89.
      - ``context_ids`` : list[int] — patch indices for context region.
      - ``target_ids_list`` : list[list[int]] — one list per target block.

    A new mask is sampled independently for every call to ``__getitem__``.
    No labels or friend images are used.
    """

    def __init__(self, img_data: np.ndarray, sampler: MultiBlockMaskSampler = None):
        """
        Parameters
        ----------
        img_data : np.ndarray
            Float32 images of shape (N, 89, 89) in [0, 1].
        sampler : MultiBlockMaskSampler or None
            Mask sampler instance.  Defaults to ``MultiBlockMaskSampler()``.
        """
        self.img_data = img_data
        self.sampler = sampler if sampler is not None else MultiBlockMaskSampler()

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.img_data[idx]).unsqueeze(0).float()  # (1, 89, 89)
        # Reflect-pad 89 → 96: 7 pixels total, split 3+4 each dim
        img = F.pad(img, (3, 4, 3, 4), mode='reflect')                   # (1, 96, 96)
        context_ids, target_ids_list = self.sampler.sample()
        return img, context_ids, target_ids_list


def jepa_collate_fn(batch: list) -> tuple:
    """
    Collate for IJEPADataset.

    Stacks images into a batch tensor (B, 1, 96, 96) and uses the first
    sample's mask for the whole batch (mask diversity comes from different
    batches; within-batch identical masks are fine for efficiency).

    Returns
    -------
    imgs : Tensor (B, 1, 96, 96)
    context_ids : list[int]
    target_ids_list : list[list[int]]
    """
    imgs = torch.stack([item[0] for item in batch])
    context_ids = batch[0][1]
    target_ids_list = batch[0][2]
    return imgs, context_ids, target_ids_list
