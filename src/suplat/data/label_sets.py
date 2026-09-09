"""Central definition of label sets, class names, and helpers.

All training scripts and notebooks should import from here rather than
duplicating these definitions locally.
"""

import numpy as np

ALL_CLASS_NAMES = [
    'FRI', 'FRII', 'Hybrids', 'Spirals', 'Relaxed doubles',
    'C-curv', 'S-curv', 'Misalign', 'Wings', 'X-shaped',
    'Straight jets', 'Multi hotspots', 'Cont. jets', 'Banding',
    'One-sided', 'Restarted', 'Cluster', 'Merger', 'Diffuse', 'Unknown',
]

LABEL_COLS = [
    "fri", "frii", "hybrid", "spiral", "relaxed",
    "cshaped", "sshaped", "misaligned", "wings", "xshaped",
    "straight", "multihotspots", "continuous", "banding", "onesided",
    "restarted", "cluster", "merger", "diffuse", "unknown",
]

DERIVED_CLASS_NAMES = [
    'Pure hybrid',        # col2 & ~col0 & ~col1
    'FR hybrid',          # col2 & (col0 | col1)
    'Curved FRI',         # col0 & (col5 | col6)
    'Curved FRII',        # col1 & (col5 | col6)
    'Straight+multi-HS',  # col10 & col11
]

INTEREST_TIER_CLASS_NAMES = ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
INTEREST_BINARY_CLASS_NAMES = ['Common (tier 1–2)', 'Rare (tier 3–4)']

# Interest-tier column indices (mirrors TIERS in class_weights.py)
_LCI = {c: i for i, c in enumerate(LABEL_COLS)}
_TIER4_COLS = [_LCI[c] for c in ("xshaped", "unknown", "cluster", "merger")]
_TIER3_COLS = [_LCI[c] for c in ("diffuse", "sshaped", "spiral")]
_TIER2_COLS = [_LCI[c] for c in ("restarted", "onesided", "banding", "cshaped",
                                   "wings", "misaligned", "multihotspots", "relaxed")]

LABEL_SETS = {
    # base sets
    "classical":              [0, 1],
    "initial":                list(range(0, 5)),
    "morphology":             list(range(5, 16)),
    "environment":            list(range(16, 20)),
    "full":                   list(range(0, 20)),
    "derived":                None,
    # interest-tier sets (computed labels, not column slices)
    "interest_tier":          None,
    "interest_binary":        None,
    # _pure variants: same columns, row-filtered to single-positive
    "classical_pure":         [0, 1],
    "initial_pure":           list(range(0, 5)),
    "morphology_pure":        list(range(5, 16)),
    "environment_pure":       list(range(16, 20)),
    "full_pure":              list(range(0, 20)),
    # _individual variants: same columns, element-wise eval (no row filter)
    "classical_individual":   [0, 1],
    "initial_individual":     list(range(0, 5)),
    "morphology_individual":  list(range(5, 16)),
    "environment_individual": list(range(16, 20)),
    "full_individual":        list(range(0, 20)),
}


def _interest_score(y: np.ndarray) -> np.ndarray:
    """Per-source interest score (1–4): highest applicable tier across all labels."""
    score = np.ones(len(y), dtype=np.int64)
    score[y[:, _TIER2_COLS].any(axis=1)] = 2
    score[y[:, _TIER3_COLS].any(axis=1)] = 3
    score[y[:, _TIER4_COLS].any(axis=1)] = 4
    return score


def make_interest_tier(y: np.ndarray) -> np.ndarray:
    """(N, 4) one-hot: each source's interest tier (1→col0, 2→col1, 3→col2, 4→col3)."""
    score = _interest_score(y)
    out   = np.zeros((len(y), 4), dtype=np.int64)
    out[np.arange(len(y)), score - 1] = 1
    return out


def make_interest_binary(y: np.ndarray) -> np.ndarray:
    """(N, 2) one-hot: common (tier 1–2) → col0, rare (tier 3–4) → col1."""
    score = _interest_score(y)
    out   = np.zeros((len(y), 2), dtype=np.int64)
    out[score <= 2, 0] = 1
    out[score >= 3, 1] = 1
    return out


def make_derived(y: np.ndarray) -> np.ndarray:
    """Compute 5 derived class labels from the 20 raw labels."""
    c = lambda i: y[:, i].astype(bool)
    return np.stack([
        ( c(2) & ~c(0) & ~c(1)).astype(np.int64),
        ( c(2) &  (c(0) | c(1))).astype(np.int64),
        ( c(0) &  (c(5) | c(6))).astype(np.int64),
        ( c(1) &  (c(5) | c(6))).astype(np.int64),
        (c(10) &   c(11)).astype(np.int64),
    ], axis=1)


def apply_label_set(labels_20: np.ndarray, label_set: str):
    """Apply column selection and pure-source row filtering.

    Returns (labels_sub, row_mask). For non-pure label sets row_mask is all-True.
    """
    n        = len(labels_20)
    row_mask = np.ones(n, dtype=bool)

    if label_set == "derived":
        return make_derived(labels_20), row_mask
    if label_set == "interest_tier":
        return make_interest_tier(labels_20), row_mask
    if label_set == "interest_binary":
        return make_interest_binary(labels_20), row_mask

    if label_set == "classical_pure":
        fri_frii = labels_20[:, 0:2]
        rest     = labels_20[:, 2:5]
        row_mask = (fri_frii.sum(axis=1) == 1) & (rest.sum(axis=1) == 0)
    elif label_set == "initial_pure":
        initial  = labels_20[:, 0:5]
        row_mask = initial.sum(axis=1) == 1
    elif label_set == "morphology_pure":
        morph    = labels_20[:, 5:16]
        row_mask = morph.sum(axis=1) == 1
    elif label_set == "environment_pure":
        row_mask = labels_20[:, 16:20].sum(axis=1) == 1
    elif label_set == "full_pure":
        row_mask = labels_20[:, :20].sum(axis=1) == 1

    _base      = label_set[:-11] if label_set.endswith('_individual') else label_set
    cols       = LABEL_SETS[_base]
    labels_sub = labels_20[row_mask][:, cols]
    return labels_sub.astype(np.int64), row_mask


def class_names_for(label_set: str) -> list:
    """Return display names for the classes in a label set."""
    if label_set == "derived":
        return list(DERIVED_CLASS_NAMES)
    if label_set == "interest_tier":
        return list(INTEREST_TIER_CLASS_NAMES)
    if label_set == "interest_binary":
        return list(INTEREST_BINARY_CLASS_NAMES)
    cols = LABEL_SETS[label_set]
    return [ALL_CLASS_NAMES[i] for i in cols]
