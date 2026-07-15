"""Unit tests for suplat.utils.class_weights."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from suplat.utils.class_weights import (
    LABEL_SETS,
    _LABEL_COL_IDX,
    compute_class_weights,
    compute_sample_weights,
)

# Column indices for label sets used in tests
_INITIAL_COLS = [_LABEL_COL_IDX[c] for c in LABEL_SETS["initial"]]  # 0-4
_N_INITIAL = len(_INITIAL_COLS)


def _make_labels(n_per_class, cols, n_total=20, rng=None):
    """Build (N, n_total) label matrix with n_per_class[c] positives in cols."""
    if rng is None:
        rng = np.random.default_rng(0)
    rows = []
    for c_idx, c_col in enumerate(cols):
        for _ in range(n_per_class[c_idx]):
            row = np.zeros(n_total, dtype=np.int64)
            row[c_col] = 1
            rows.append(row)
    return np.array(rows, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# Uniform fallbacks
# ──────────────────────────────────────────────────────────────────────────────

def test_uniform_class_weights_mode_none():
    labels = np.zeros((50, 20), dtype=np.int64)
    alpha = compute_class_weights(labels, None, 1.0)
    assert alpha.shape == (20,)
    assert np.all(alpha == 1.0)


def test_uniform_class_weights_strength_zero():
    labels = np.zeros((50, 20), dtype=np.int64)
    alpha = compute_class_weights(labels, "initial", 0.0)
    assert np.all(alpha == 1.0)


def test_uniform_sample_weights_mode_none():
    labels = np.zeros((50, 20), dtype=np.int64)
    w = compute_sample_weights(labels, None, 1.0)
    assert w.shape == (50,)
    assert np.all(w == 1.0)


def test_uniform_sample_weights_strength_zero():
    labels = np.zeros((50, 20), dtype=np.int64)
    w = compute_sample_weights(labels, "score", 0.0)
    assert np.all(w == 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# compute_class_weights: upweights rare classes
# ──────────────────────────────────────────────────────────────────────────────

def test_class_weights_upweights_rare_classes():
    # Build a label matrix with 2 common classes (80 each) and 1 rare (10),
    # all from the 'initial' set (5 classes; use indices 0, 1, 2 from the set).
    # Use pure rows so each row has exactly one positive.
    cols = _INITIAL_COLS  # [0, 1, 2, 3, 4]
    n_per = [80, 80, 10, 0, 0]
    labels = _make_labels(n_per, cols)

    alpha = compute_class_weights(labels, "initial", 1.0)

    # Only initial columns are non-zero
    for i in range(20):
        if i in cols:
            assert alpha[i] > 0, f"col {i} should be non-zero"
        else:
            assert alpha[i] == 0.0, f"col {i} outside set should be 0"

    # Rare class (col index 2 in LABEL_COLS = 'hybrid') must have higher alpha
    # than common classes (col 0 = 'fri', col 1 = 'frii')
    alpha_common = alpha[cols[0]]   # fri
    alpha_rare = alpha[cols[2]]     # hybrid (10 samples)
    assert alpha_rare > alpha_common, "rare class should be upweighted"

    # At strength=1, positive mass per class should be approximately equal:
    # mass_c = n_c * alpha_c = n_c * (mean_n / n_c) = mean_n (constant)
    n_c = np.array(n_per, dtype=float)
    mean_n = n_c[n_c > 0].mean()
    alpha_set = alpha[cols]
    mass = n_c * alpha_set
    # Only check classes with > 0 samples
    nonzero = n_c > 0
    assert np.allclose(mass[nonzero], mean_n, rtol=0.01), \
        f"positive mass should be equal at strength=1, got {mass}"


def test_class_weights_score_raises():
    labels = np.zeros((10, 20), dtype=np.int64)
    with pytest.raises(ValueError, match="score"):
        compute_class_weights(labels, "score", 1.0)


def test_class_weights_unknown_mode_raises():
    labels = np.zeros((10, 20), dtype=np.int64)
    with pytest.raises(ValueError, match="Unknown"):
        compute_class_weights(labels, "nonsense_mode", 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# compute_sample_weights: pure label set
# ──────────────────────────────────────────────────────────────────────────────

def test_sample_weights_pure_set_mean_one():
    # Pure label matrix: 80 FRI, 80 FRII, 10 hybrid (initial cols 0,1,2)
    cols = _INITIAL_COLS
    n_per = [80, 80, 10, 0, 0]
    labels = _make_labels(n_per, cols)
    # Drop zero-count classes to avoid pure-purity issue (rows with 0 positives)
    labels = labels[labels[:, cols].sum(axis=1) > 0]

    w = compute_sample_weights(labels, "initial", 1.0)
    assert w.shape[0] == len(labels)
    assert np.isclose(w.mean(), 1.0, rtol=1e-4), f"mean(w) should be 1, got {w.mean()}"


def test_sample_weights_pure_set_equal_mass():
    cols = _INITIAL_COLS
    n_per = [80, 80, 10, 0, 0]
    labels = _make_labels(n_per, cols)
    labels = labels[labels[:, cols].sum(axis=1) > 0]

    w = compute_sample_weights(labels, "initial", 1.0)

    # Each class contributes equally to the weighted sum (equal effective mass)
    masses = []
    for c_idx, c_col in enumerate(cols[:3]):   # first 3 have samples
        mask = labels[:, c_col] == 1
        masses.append(w[mask].sum())
    assert np.allclose(masses, masses[0], rtol=0.01), \
        f"weighted class mass should be equal, got {masses}"


def test_sample_weights_impure_raises():
    # Rows with 2 positives in initial columns -> should raise
    labels = np.zeros((10, 20), dtype=np.int64)
    labels[:, 0] = 1   # fri
    labels[:, 1] = 1   # frii  (all rows have 2 positives in initial cols)
    with pytest.raises(ValueError, match="pure label set"):
        compute_sample_weights(labels, "initial", 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# compute_sample_weights: score mode
# ──────────────────────────────────────────────────────────────────────────────

def test_sample_weights_score_upweights_rare_tiers():
    # Build labels where some samples are tier-4 (xshaped) and rest are tier-1 (fri)
    xshaped_col = _LABEL_COL_IDX["xshaped"]
    fri_col = _LABEL_COL_IDX["fri"]
    labels = np.zeros((100, 20), dtype=np.int64)
    labels[:10, xshaped_col] = 1   # tier-4 (score 4)
    labels[10:, fri_col] = 1       # tier-1 (score 1)

    w = compute_sample_weights(labels, "score", 1.0)
    assert w[:10].mean() > w[10:].mean(), "tier-4 samples should be upweighted"
    assert np.isclose(w.mean(), 1.0, rtol=0.05)
