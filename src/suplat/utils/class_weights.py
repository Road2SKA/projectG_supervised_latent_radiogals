import numpy as np

LABEL_COLS = [
    "fri", "frii", "hybrid", "spiral", "relaxed",
    "cshaped", "sshaped", "misaligned", "wings", "xshaped",
    "straight", "multihotspots", "continuous", "banding", "onesided",
    "restarted", "cluster", "merger", "diffuse", "unknown",
]

SCORE_4 = ["xshaped", "unknown", "cluster", "merger"]
SCORE_3 = ["diffuse", "sshaped", "spiral"]
SCORE_2 = ["restarted", "onesided", "banding", "cshaped", "wings", "misaligned", "multihotspots", "relaxed"]
SCORE_1 = ["fri", "frii", "hybrid", "straight", "continuous"]
TIERS   = [(4, SCORE_4), (3, SCORE_3), (2, SCORE_2), (1, SCORE_1)]

LABEL_SETS = {
    "classical":   ["fri", "frii"],
    "initial":     ["fri", "frii", "hybrid", "spiral", "relaxed"],
    "morphology":  ["cshaped", "sshaped", "misaligned", "wings", "xshaped",
                    "straight", "multihotspots", "continuous", "banding", "onesided", "restarted"],
    "environment": ["cluster", "merger", "diffuse", "unknown"],
    "all":         LABEL_COLS,
}

_LABEL_COL_IDX = {c: i for i, c in enumerate(LABEL_COLS)}

_VALID_MODES = {"score"} | set(LABEL_SETS.keys())


def compute_class_weights(labels, mode, strength):
    """Per-class alpha vector for multi-label neural scripts.

    Parameters
    ----------
    labels   : np.ndarray, shape (N, 20), in LABEL_COLS order
    mode     : None | 'initial' | 'morphology' | 'environment' | 'classical' | 'all'
               'score' is NOT supported — score is a whole-sample property, not per-class.
    strength : float; 0.0 → uniform weights regardless of mode

    Returns
    -------
    np.ndarray, shape (20,), float32
        alpha[c] > 0 for columns in the selected set.
        alpha[c] = 0.0 for all other columns — those columns carry no weighting signal
        and are excluded from the loss when alpha is used as a class mask.
    """
    if mode is None or strength == 0.0:
        return np.ones(20, dtype=np.float32)

    if mode == 'score':
        raise ValueError(
            "'score' is not a valid mode for compute_class_weights: score is a "
            "whole-sample property, not an individual class property. "
            "Use compute_sample_weights for score-based weighting."
        )
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown class-weight mode '{mode}'. "
            f"Valid: {sorted(_VALID_MODES - {'score'})}"
        )

    ci = [_LABEL_COL_IDX[c] for c in LABEL_SETS[mode]]
    n_c = labels[:, ci].sum(axis=0).astype(np.float32)   # positive counts per class
    mean_n = n_c.mean()
    alpha = np.zeros(20, dtype=np.float32)
    alpha[ci] = (mean_n / np.maximum(n_c, 1)) ** strength
    # Columns outside the selected set are 0.0: they carry no weighting signal.
    # Callers that use alpha as a class mask will exclude those class terms from the loss.
    return alpha


def compute_sample_weights(labels, mode, strength):
    """Compute per-sample loss weights.

    For multi-label neural scripts use compute_class_weights instead.
    This function is valid for:
      - mode='score'      : any label matrix (tier-based, whole-sample property)
      - mode=<label_set>  : only pure label sets where each row has exactly one
                            positive in the selected columns.

    Parameters
    ----------
    labels   : np.ndarray, shape (N, 20), in LABEL_COLS order
    mode     : None | 'score' | 'initial' | 'morphology' | 'environment' | 'classical' | 'all'
    strength : float; 0.0 → uniform weights regardless of mode

    Returns
    -------
    np.ndarray, shape (N,), float32, mean ≈ 1.0
    """
    N = len(labels)
    if mode is None or strength == 0.0:
        return np.ones(N, dtype=np.float32)

    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown class-weight mode '{mode}'. Valid: {sorted(_VALID_MODES)}")

    if mode == 'score':
        raw = np.ones(N, dtype=np.float32)
        for score_val, cols in reversed(TIERS):
            ci = [_LABEL_COL_IDX[c] for c in cols]
            raw[labels[:, ci].any(axis=1)] = float(score_val)
        raw_norm = raw / raw.mean()
        w = (1.0 + strength * (raw_norm - 1.0)).clip(min=0.0).astype(np.float32)
        return w

    # Label-set mode: requires a pure label set (exactly one positive per row in the set)
    cols = LABEL_SETS[mode]
    ci = [_LABEL_COL_IDX[c] for c in cols]
    Y_sub = labels[:, ci]
    row_sums = Y_sub.sum(axis=1)
    if not np.all(row_sums == 1):
        raise ValueError(
            f"per-sample weighting requires a pure label set (exactly one "
            f"positive per sample); mode '{mode}' has rows with "
            f"{sorted(set(row_sums.tolist()))} positives. Use a *_pure "
            f"label set, or use per-class weighting for the multi-label case."
        )
    class_idx = Y_sub.argmax(axis=1)      # one class per sample
    n_c = np.bincount(class_idx, minlength=len(ci)).astype(np.float32)
    mean_n = n_c.mean()
    raw = (mean_n / np.maximum(n_c[class_idx], 1)) ** strength
    w = raw / raw.mean()
    return w.astype(np.float32)
