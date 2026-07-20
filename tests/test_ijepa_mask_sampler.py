"""Unit tests for MultiBlockMaskSampler and IJEPADataset."""

import numpy as np
import pytest

from suplat.data.ijepa_mask_sampler import MultiBlockMaskSampler, IJEPADataset


GRID = (12, 12)
N_PATCHES = 144


@pytest.fixture
def sampler():
    return MultiBlockMaskSampler(grid_size=GRID)


def test_context_nonempty(sampler):
    ctx, _ = sampler.sample()
    assert len(ctx) > 0, "Context must not be empty"


def test_all_target_blocks_nonempty(sampler):
    _, tgt_list = sampler.sample()
    for i, block in enumerate(tgt_list):
        assert len(block) > 0, f"Target block {i} must not be empty"


def test_n_target_blocks(sampler):
    _, tgt_list = sampler.sample()
    assert len(tgt_list) == sampler.n_target_blocks


def test_no_overlap_context_targets(sampler):
    ctx, tgt_list = sampler.sample()
    ctx_set = set(ctx)
    for i, block in enumerate(tgt_list):
        overlap = ctx_set & set(block)
        assert len(overlap) == 0, f"Context overlaps with target block {i}: {overlap}"


def test_indices_in_range(sampler):
    ctx, tgt_list = sampler.sample()
    # Target blocks may overlap — collect unique indices
    all_ids = list(set(ctx) | {i for block in tgt_list for i in block})
    assert all(0 <= i < N_PATCHES for i in all_ids), \
        "All patch indices must be in [0, 143]"


def test_coverage_not_exceeded(sampler):
    """Unique patch indices used (ctx ∪ targets) must not exceed grid size."""
    ctx, tgt_list = sampler.sample()
    # Target blocks may overlap each other — count unique indices only.
    all_unique = set(ctx) | {i for block in tgt_list for i in block}
    assert len(all_unique) <= N_PATCHES, \
        f"Unique tokens ({len(all_unique)}) exceeds grid size ({N_PATCHES})"


def test_reproducibility_different_calls(sampler):
    """Two independent calls should (almost always) give different masks."""
    results = [sampler.sample() for _ in range(20)]
    contexts = [tuple(r[0]) for r in results]
    # Not all identical (extremely unlikely to get 20 identical samples)
    assert len(set(contexts)) > 1, "Sampler appears to always return the same mask"


def test_dataset_shape():
    images = np.random.rand(10, 89, 89).astype(np.float32)
    ds = IJEPADataset(images)
    img, ctx, tgt_list = ds[0]
    assert img.shape == (1, 96, 96), f"Expected (1,96,96), got {img.shape}"
    assert len(ctx) > 0
    assert len(tgt_list) > 0


def test_dataset_pad_range():
    """Padded values should be in a sane range (reflect pad, same image data)."""
    images = np.random.rand(5, 89, 89).astype(np.float32)
    ds = IJEPADataset(images)
    for i in range(len(ds)):
        img, _, _ = ds[i]
        assert img.min() >= 0.0
        assert img.max() <= 1.0


@pytest.mark.parametrize("n_runs", [100])
def test_sampler_stress(sampler, n_runs):
    """Run sampler many times; check all invariants hold."""
    for _ in range(n_runs):
        ctx, tgt_list = sampler.sample()
        assert len(ctx) > 0
        ctx_set = set(ctx)
        for block in tgt_list:
            assert len(block) > 0
            assert len(ctx_set & set(block)) == 0
        # Target blocks may overlap each other; check unique indices only
        all_unique = list(set(ctx) | {i for b in tgt_list for i in b})
        assert all(0 <= i < N_PATCHES for i in all_unique)
