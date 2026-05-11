"""
src/suplat/data/catalogue.py

Catalogue system for reproducible multi-dataset BYOL training and evaluation.

A catalogue YAML specifies which datasets to use, what labels to expose,
and what fraction of train-split labels the model sees. The catalogue
materialises into concrete train/val/test index splits, saved as a sidecar
JSON to guarantee reproducibility across runs.

Usage:
    from suplat.data.catalogue import Catalogue

    cat  = Catalogue.from_yaml("catalogues/multi_all.yaml")
    mat  = cat.materialise(root=".")

    # For BYOL training
    all_images, labelled_df = mat.get_byol_data()

    # For classifier training / evaluation
    splits = mat.get_split_datasets()   # {'train': ..., 'val': ..., 'test': ...}

    # Metadata
    print(mat.n_labelled_train)
    print(mat.label_names)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Label alias definitions
# ---------------------------------------------------------------------------

# LoTSS column indices (0-based) for each alias
LOTSS_ALIAS = {
    "full":           slice(0, 20),
    "native":         slice(0, 20),
    "none":           slice(0, 20),
    "initial":        slice(0, 5),
    "classical":      slice(0, 2),
    "classical_pure":    "classical_pure",    # special: filter rows
    "initial_pure":      "initial_pure",      # special: filter rows
    "morphology":        slice(5, 16),
    "morphology_pure":   "morphology_pure",   # special: filter rows
    "environment":       slice(16, 20),
    "environment_pure":  "environment_pure",  # special: filter rows
    "derived":           "derived",           # special: computed from label combinations
}

LOTSS_LABEL_NAMES = {
    "full":        [f"col_{i}" for i in range(20)],
    "native":      [f"col_{i}" for i in range(20)],
    "none":        [f"col_{i}" for i in range(20)],
    "initial":     ["FRI", "FRII", "Hybrid", "Spiral", "RelaxedDouble"],
    "classical":   ["FRI", "FRII"],
    "classical_pure":   ["FRI", "FRII"],
    "initial_pure":     ["FRI", "FRII", "Hybrid", "Spiral", "RelaxedDouble"],
    "morphology":       ['C-curv', 'S-curv', 'Misalign', 'Wings', 'X-shaped',
                         'StraightJets', 'MultiHS', 'ContJets', 'Banding', 'OneSided', 'Restarted'],
    "morphology_pure":  ['C-curv', 'S-curv', 'Misalign', 'Wings', 'X-shaped',
                         'StraightJets', 'MultiHS', 'ContJets', 'Banding', 'OneSided', 'Restarted'],
    "environment":      ["Cluster", "Merger", "Diffuse", "Unknown"],
    "environment_pure": ["Cluster", "Merger", "Diffuse", "Unknown"],
    "derived":          [f"derived_{i}" for i in range(5)],
}

# Datasets that are always unlabelled — never contribute supervised signal
UNLABELLED_DATASETS = {"mgcls_20k", "mgcls_5k", "mightee", "mightee_fr"}

# CSV-based datasets (use EvalDataset / labels.csv)
CSV_DATASETS = {"mirabest", "radio_galaxy_dataset", "mightee_fr"}

# Split ratios
TRAIN_RATIO = 0.64
VAL_RATIO   = 0.16
# TEST_RATIO  = 0.20  (remainder)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CatalogueEntry:
    """One dataset entry within a catalogue."""
    dataset:  str
    labels:   str    # label alias
    f_labels: float  # fraction of train labels exposed (0.0–1.0)


@dataclass
class SplitView:
    """
    A single train/val/test partition for one labelled dataset.

    images : np.ndarray, shape (N, 89, 89)
    labels : np.ndarray, shape (N,) or (N, D) depending on label type
    indices: np.ndarray  original row indices (for sidecar tracking)
    """
    images:  np.ndarray
    labels:  np.ndarray
    indices: np.ndarray
    dataset: str
    split:   str      # "train" | "val" | "test"


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def compute_split(
    n: int,
    seed: int,
    existing_test: Optional[np.ndarray] = None,
    existing_val:  Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 64/16/20 train/val/test index split for n items.

    If existing_test is provided (e.g. canonical test split from a CSV),
    those indices become the test set. The remainder is split 80/20 for
    train/val from the remaining data.

    Parameters
    ----------
    n             : int   total number of samples
    seed          : int   random seed
    existing_test : optional array of test indices (pre-defined)
    existing_val  : optional array of val indices (pre-defined)

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray of int indices
    """
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)

    if existing_test is not None:
        test_idx      = np.asarray(existing_test)
        remaining_idx = np.setdiff1d(all_idx, test_idx)
    else:
        # Random 20% test split
        n_test    = max(1, round(n * (1 - TRAIN_RATIO - VAL_RATIO)))
        test_idx  = rng.choice(all_idx, size=n_test, replace=False)
        remaining_idx = np.setdiff1d(all_idx, test_idx)

    if existing_val is not None:
        val_idx   = np.asarray(existing_val)
        train_idx = np.setdiff1d(remaining_idx, val_idx)
    else:
        # From remaining, 80% train / 20% val  (= 64% / 16% overall)
        n_val     = max(1, round(len(remaining_idx) * VAL_RATIO /
                                 (TRAIN_RATIO + VAL_RATIO)))
        val_idx   = rng.choice(remaining_idx, size=n_val, replace=False)
        train_idx = np.setdiff1d(remaining_idx, val_idx)

    return (np.sort(train_idx),
            np.sort(val_idx),
            np.sort(test_idx))


def save_split_sidecar(path: str, dataset_seed: int,
                       splits: Dict[str, Dict]) -> None:
    """Save split indices to a JSON sidecar file."""
    payload = {"dataset_seed": dataset_seed}
    for name, s in splits.items():
        payload[name] = {
            k: v.tolist() for k, v in s.items()
        }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_split_sidecar(path: str) -> Tuple[int, Dict[str, Dict]]:
    """Load split indices from a JSON sidecar file."""
    with open(path) as f:
        payload = json.load(f)
    seed   = payload.pop("dataset_seed")
    splits = {
        name: {k: np.array(v) for k, v in s.items()}
        for name, s in payload.items()
    }
    return seed, splits


# ---------------------------------------------------------------------------
# Label processing helpers
# ---------------------------------------------------------------------------

def _apply_label_type_lotss(
    images: np.ndarray,
    raw_labels: np.ndarray,
    label_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply label alias to LoTSS 20-dim multi-hot label array.

    For "pure" aliases, rows are filtered to keep only unambiguous samples.
    For slice aliases, columns are selected, no row filtering.

    Parameters
    ----------
    images     : (N, 89, 89)
    raw_labels : (N, 20) multi-hot
    label_type : alias string

    Returns
    -------
    images_out, labels_out : filtered/sliced arrays
    """
    if label_type in ("full", "native", "none"):
        return images, raw_labels

    alias = LOTSS_ALIAS.get(label_type)
    if alias is None:
        raise ValueError(f"Unknown LoTSS label alias: '{label_type}'")

    if label_type == "classical_pure":
        # Keep rows where exactly one of FRI(0) or FRII(1) is 1,
        # and hybrid(2), spiral(3), relaxed(4) are all 0
        fri  = raw_labels[:, 0]
        frii = raw_labels[:, 1]
        rest = raw_labels[:, 2:5].sum(axis=1)
        mask = ((fri + frii == 1) & (rest == 0))
        labels_out = np.where(frii[mask] == 1, 1, 0).astype(np.int64)
        return images[mask], labels_out

    if label_type == "initial_pure":
        # Keep rows where exactly one of the 5 initial labels is set
        initial = raw_labels[:, :5]
        mask    = (initial.sum(axis=1) == 1)
        labels_out = initial[mask].argmax(axis=1).astype(np.int64)
        return images[mask], labels_out

    if label_type == "morphology_pure":
        # Keep rows where exactly one of the 11 morphology cols (5-15) is set
        morph = raw_labels[:, 5:16]
        mask  = (morph.sum(axis=1) == 1)
        labels_out = morph[mask].argmax(axis=1).astype(np.int64)
        return images[mask], labels_out

    if label_type == "environment_pure":
        # Keep rows where exactly one of the 4 environment cols (16-19) is set
        env  = raw_labels[:, 16:20]
        mask = (env.sum(axis=1) == 1)
        labels_out = env[mask].argmax(axis=1).astype(np.int64)
        return images[mask], labels_out

    if label_type == "derived":
        raise NotImplementedError(
            "The 'derived' label type is computed from combinations of other labels "
            "and is not stored as columns in the raw array. "
            "Use 'full' labels and compute derived labels in your training code."
        )

    # Slice alias — no row filtering
    return images, raw_labels[:, alias]


def _apply_label_type_csv(
    df: pd.DataFrame,
    label_type: str,
    dataset: str,
) -> pd.DataFrame:
    """
    Apply label alias to a CSV-based dataset (mirabest, radio_galaxy_dataset).

    For "classical" alias on radio_galaxy_dataset: filters to FRI/FRII only.
    For mirabest "classical" is a no-op (already FRI/FRII).

    Returns filtered df with "label" column containing final integer labels.
    """
    if label_type in ("full", "native", "none", "classical_pure"):
        # All aliases map to the existing integer label column
        return df

    if label_type == "classical" and dataset == "radio_galaxy_dataset":
        # Keep only FRI (0) and FRII (1)
        df = df[df["label"].isin([0, 1])].copy()
        return df

    # For all other aliases, return as-is (mirabest is always binary)
    return df


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_lotss(root: str, label_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load LoTSS images and labels from monolithic .npy files.

    Expected files:
        <root>/data/preprocessed/lotss/images_filtered.npy   (N, 89, 89)
        <root>/data/preprocessed/lotss/labels_filtered.npy   (N, 20)

    Returns images, labels (filtered/sliced by label_type).
    """
    img_path = os.path.join(root, "data", "preprocessed", "lotss", "images_filtered.npy")
    lbl_path = os.path.join(root, "data", "preprocessed", "lotss", "labels_filtered.npy")

    images     = np.load(img_path, mmap_mode="r")
    raw_labels = np.load(lbl_path, mmap_mode="r")

    images, labels = _apply_label_type_lotss(
        np.array(images), np.array(raw_labels), label_type
    )
    return images, labels


def _load_csv_dataset(
    root: str,
    dataset: str,
    label_type: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load a CSV-based dataset (mirabest, radio_galaxy_dataset).

    Returns images, labels, df (for split column access).
    """
    from suplat.data.eval_dataset import DATASET_REGISTRY

    proc_dir, csv_path = DATASET_REGISTRY[dataset]
    proc_dir = os.path.join(root, proc_dir)
    csv_path = os.path.join(root, csv_path)

    df = pd.read_csv(csv_path)
    df = _apply_label_type_csv(df, label_type, dataset)

    images = np.stack([
        np.load(os.path.join(proc_dir, fname))
        for fname in df["filename"]
    ])
    labels = df["label"].values.astype(np.int64)
    return images, labels, df


def _load_unlabelled(root: str, dataset: str) -> np.ndarray:
    """
    Load images for an unlabelled dataset (no labels returned).
    """
    from suplat.data.eval_dataset import DATASET_REGISTRY
    import glob

    proc_dir, csv_path = DATASET_REGISTRY[dataset]
    proc_dir = os.path.join(root, proc_dir)

    if csv_path is not None:
        df    = pd.read_csv(os.path.join(root, csv_path))
        files = [os.path.join(proc_dir, f) for f in df["filename"]]
    else:
        files = sorted(glob.glob(os.path.join(proc_dir, "*.npy")))

    images = np.stack([np.load(f) for f in files])
    return images


# ---------------------------------------------------------------------------
# MaterialisedCatalogue
# ---------------------------------------------------------------------------

class MaterialisedCatalogue:
    """
    A fully resolved catalogue with concrete train/val/test splits.

    Created by Catalogue.materialise() — do not instantiate directly.
    """

    def __init__(self):
        # Per-labelled-dataset data
        self._labelled: Dict[str, Dict] = {}
        # Per-unlabelled-dataset images
        self._unlabelled: Dict[str, np.ndarray] = {}
        # Split indices per labelled dataset
        self._splits: Dict[str, Dict[str, np.ndarray]] = {}
        # Catalogue-level metadata
        self.dataset_seed: int = 42
        self._entries: List[CatalogueEntry] = []

    # --- Public API ---

    @property
    def n_labelled_train(self) -> int:
        """Total number of labelled training samples across all datasets."""
        total = 0
        for name, data in self._labelled.items():
            f = data["f_labels"]
            n_train = len(self._splits[name]["train"])
            total += round(f * n_train)
        return total

    @property
    def label_names(self) -> List[str]:
        """Label names for the first labelled dataset (all should match)."""
        for name, data in self._labelled.items():
            return data["label_names"]
        return []

    def get_byol_data(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Return all images and a DataFrame of labelled training samples.

        The DataFrame has columns: filename (or index), image_idx, label,
        dataset. Only the f_labels fraction of train-split indices is
        included — the rest are treated as unlabelled by BYOL.

        Returns
        -------
        all_images     : np.ndarray  (N_total, 89, 89)
        labelled_df    : pd.DataFrame  labelled train samples
        """
        all_image_blocks = []
        labelled_rows    = []
        image_offset     = 0

        # Labelled datasets first
        for name, data in self._labelled.items():
            images     = data["images"]
            labels     = data["labels"]
            f          = data["f_labels"]
            train_idx  = self._splits[name]["train"]

            all_image_blocks.append(images)

            # Subsample f_labels fraction of train indices
            rng      = np.random.default_rng(self.dataset_seed)
            n_expose = max(0, round(f * len(train_idx)))
            if n_expose > 0:
                exposed = rng.choice(train_idx, size=n_expose, replace=False)
                for idx in exposed:
                    labelled_rows.append({
                        "image_idx": image_offset + int(idx),
                        "label":     labels[idx],
                        "dataset":   name,
                    })

            image_offset += len(images)

        # Unlabelled datasets
        for name, images in self._unlabelled.items():
            all_image_blocks.append(images)
            image_offset += len(images)

        all_images  = np.concatenate(all_image_blocks, axis=0)
        labelled_df = pd.DataFrame(labelled_rows)
        return all_images, labelled_df

    def get_split_datasets(self) -> Dict[str, List[SplitView]]:
        """
        Return train/val/test SplitView lists for all labelled datasets.

        Unlabelled datasets are excluded.

        Returns
        -------
        dict with keys "train", "val", "test", each a list of SplitView
        (one per labelled dataset).
        """
        result = {"train": [], "val": [], "test": []}

        for name, data in self._labelled.items():
            images = data["images"]
            labels = data["labels"]

            for split_name, idx in self._splits[name].items():
                result[split_name].append(SplitView(
                    images=images[idx],
                    labels=labels[idx],
                    indices=idx,
                    dataset=name,
                    split=split_name,
                ))

        return result


# ---------------------------------------------------------------------------
# Catalogue (the main class)
# ---------------------------------------------------------------------------

class Catalogue:
    """
    Describes a multi-dataset training configuration.

    Load from YAML:
        cat = Catalogue.from_yaml("catalogues/multi_all.yaml")

    Then materialise to get concrete splits:
        mat = cat.materialise(root=".")
    """

    def __init__(
        self,
        name:         str,
        entries:      List[CatalogueEntry],
        dataset_seed: int  = 42,
        cval:         bool = False,
    ):
        self.name         = name
        self.entries      = entries
        self.dataset_seed = dataset_seed
        self.cval         = cval

    @classmethod
    def from_yaml(cls, path: str) -> "Catalogue":
        """
        Load a catalogue from a YAML file.

        Expected format:
            name: multi_all
            datasets:  [lotss, mirabest, mgcls_20k]
            labels:    [full,  native,   none]
            f_labels:  [1.0,   1.0,      0.0]
            dataset_seed: 42
            cval: false
        """
        with open(path) as f:
            cfg = yaml.safe_load(f)

        datasets = cfg["datasets"]
        labels   = cfg["labels"]
        f_labels = cfg.get("f_labels", [1.0] * len(datasets))

        # Allow scalar f_labels applied to all datasets
        if isinstance(f_labels, (int, float)):
            f_labels = [float(f_labels)] * len(datasets)

        if not (len(datasets) == len(labels) == len(f_labels)):
            raise ValueError(
                "datasets, labels, and f_labels must have the same length"
            )

        entries = [
            CatalogueEntry(dataset=d, labels=l, f_labels=float(f))
            for d, l, f in zip(datasets, labels, f_labels)
        ]

        return cls(
            name         = cfg.get("name", os.path.splitext(
                               os.path.basename(path))[0]),
            entries      = entries,
            dataset_seed = cfg.get("dataset_seed", 42),
            cval         = cfg.get("cval", False),
        )

    def materialise(
        self,
        root:      str = ".",
        split_dir: Optional[str] = None,
    ) -> MaterialisedCatalogue:
        """
        Load all datasets, compute or reload splits, return MaterialisedCatalogue.

        The split sidecar is saved as:
            <split_dir>/<catalogue_name>_split.json
        Default split_dir is the directory containing the catalogue YAML,
        or the current directory if not specified.

        Parameters
        ----------
        root      : project root (all paths resolved relative to this)
        split_dir : where to store/load the sidecar JSON
        """
        if split_dir is None:
            split_dir = "catalogues"
        os.makedirs(os.path.join(root, split_dir), exist_ok=True)

        sidecar_path = os.path.join(
            root, split_dir, f"{self.name}_split.json"
        )

        mat              = MaterialisedCatalogue()
        mat.dataset_seed = self.dataset_seed
        mat._entries     = self.entries

        # --- Load data for each entry ---
        labelled_splits_to_compute: Dict[str, int] = {}  # name → n_samples

        # Track pre-existing splits from CSV datasets
        existing_splits: Dict[str, Dict[str, np.ndarray]] = {}

        for entry in self.entries:
            name = entry.dataset

            if name in UNLABELLED_DATASETS:
                print(f"  [{name}] loading unlabelled images...")
                images = _load_unlabelled(root, name)
                mat._unlabelled[name] = images
                print(f"    {len(images)} images")

            elif name == "lotss":
                print(f"  [lotss] loading with labels='{entry.labels}'...")
                images, labels = _load_lotss(root, entry.labels)
                label_names    = LOTSS_LABEL_NAMES.get(entry.labels, [])
                mat._labelled[name] = {
                    "images":      images,
                    "labels":      labels,
                    "f_labels":    entry.f_labels,
                    "label_names": label_names,
                }
                labelled_splits_to_compute[name] = len(images)
                print(f"    {len(images)} images, "
                      f"{labels.shape[-1] if labels.ndim > 1 else 1}-dim labels")

            elif name in CSV_DATASETS or name == "radio_galaxy_dataset":
                print(f"  [{name}] loading with labels='{entry.labels}'...")
                images, labels, df = _load_csv_dataset(root, name, entry.labels)
                label_names = ["FRI", "FRII"] if "classical" in entry.labels \
                              else ["FRI", "FRII", "Compact", "Bent"] \
                              if name == "radio_galaxy_dataset" else ["FRI", "FRII"]

                mat._labelled[name] = {
                    "images":      images,
                    "labels":      labels,
                    "f_labels":    entry.f_labels,
                    "label_names": label_names,
                }
                print(f"    {len(images)} images")

                # Extract pre-existing test/val splits from CSV
                if "split" in df.columns:
                    test_mask = df["split"] == "test"
                    val_mask  = df["split"] == "valid"
                    existing_splits[name] = {}
                    if test_mask.any():
                        existing_splits[name]["test"] = \
                            np.where(test_mask.values)[0]
                    if val_mask.any():
                        existing_splits[name]["val"] = \
                            np.where(val_mask.values)[0]

                labelled_splits_to_compute[name] = len(images)

            else:
                raise ValueError(f"Unknown dataset: '{name}'")

        # --- Load or compute splits ---
        if os.path.exists(sidecar_path):
            saved_seed, saved_splits = load_split_sidecar(sidecar_path)
            if saved_seed != self.dataset_seed:
                raise RuntimeError(
                    f"Sidecar seed ({saved_seed}) does not match "
                    f"catalogue seed ({self.dataset_seed}). "
                    f"Delete {sidecar_path} to recompute."
                )
            print(f"  Loaded split sidecar: {sidecar_path}")
            mat._splits = saved_splits
        else:
            print("  Computing new splits...")
            for name, n in labelled_splits_to_compute.items():
                ex = existing_splits.get(name, {})
                train_idx, val_idx, test_idx = compute_split(
                    n,
                    seed          = self.dataset_seed,
                    existing_test = ex.get("test"),
                    existing_val  = ex.get("val"),
                )
                mat._splits[name] = {
                    "train": train_idx,
                    "val":   val_idx,
                    "test":  test_idx,
                }
                print(f"    {name}: train={len(train_idx)}, "
                      f"val={len(val_idx)}, test={len(test_idx)}")
            save_split_sidecar(sidecar_path, self.dataset_seed, mat._splits)
            print(f"  Split sidecar saved: {sidecar_path}")

        return mat