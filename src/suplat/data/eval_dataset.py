"""
eval_dataset.py

Unified Dataset for evaluating BYOL encoder generalisation across
multiple radio-galaxy datasets.

Supports:
  - mgcls_20k            : unlabelled 89x89 crops (label = -1)
  - mgcls_5k             : random 5k subset of mgcls_20k (label = -1)
  - mightee              : unlabelled MIGHTEE DR1 crops (label = -1)
  - mightee_fr           : labelled MIGHTEE FRI/FRII crops
  - mirabest             : labelled MiraBest FRI/FRII crops
  - radio_galaxy_dataset : labelled RadioGalaxyDataset (FIRST) crops

Usage:
    from suplat.data.eval_dataset import EvalDataset
    ds = EvalDataset('mirabest', root='/path/to/project')
    img, label = ds[0]   # img: (1,89,89) float32, label: int (-1 if unlabelled)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Registry: dataset_name -> (processed_dir_relative, labels_csv_relative or None)
DATASET_REGISTRY = {
    "mgcls_20k":            ("data/processed/mgcls_20k",            "data/metadata/mgcls_crops.csv"),
    "mgcls_5k":             ("data/processed/mgcls_5k",             "data/metadata/mgcls_5k_crops.csv"),
    "mightee":              ("data/processed/mightee",              None),
    "mightee_fr":           ("data/processed/mightee_fr",           "data/metadata/mightee_fr_labels.csv"),
    "mirabest":             ("data/processed/mirabest",             "data/metadata/mirabest_labels.csv"),
    "radio_galaxy_dataset": ("data/processed/first",                "data/metadata/radio_galaxy_dataset_labels.csv"),
}


class EvalDataset(Dataset):
    """
    Unified evaluation dataset for BYOL encoder generalisation study.

    Parameters
    ----------
    name : str
        One of the keys in DATASET_REGISTRY.
    root : str or Path
        Project root directory. Paths in DATASET_REGISTRY are relative to this.
    split : str or None
        If provided, filter to rows where the CSV column 'split' equals this
        value.  Ignored for unlabelled datasets (no labels CSV).
    transform : callable or None
        Optional transform applied to the (1, H, W) float tensor.
    """

    def __init__(
        self,
        name: str,
        root: str | Path = ".",
        split: str | None = None,
        transform=None,
    ):
        if name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{name}'. "
                f"Choose from: {sorted(DATASET_REGISTRY)}"
            )

        self.name = name
        self.root = Path(root)
        self.split = split
        self.transform = transform

        data_dir_rel, labels_csv_rel = DATASET_REGISTRY[name]
        self.data_dir = self.root / data_dir_rel

        if labels_csv_rel is not None:
            labels_csv = self.root / labels_csv_rel
            df = pd.read_csv(labels_csv)
            if split is not None and "split" in df.columns:
                df = df[df["split"] == split].reset_index(drop=True)
            self.filenames = df["filename"].tolist()
            self.labels = (
                df["label"].tolist() if "label" in df.columns else [-1] * len(df)
            )
        else:
            # Unlabelled: discover .npy files from the directory
            self.filenames = sorted(p.name for p in self.data_dir.glob("*.npy"))
            self.labels = [-1] * len(self.filenames)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        img = np.load(self.data_dir / self.filenames[idx])
        img = torch.from_numpy(img).unsqueeze(0).float()  # (1, H, W)
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])

    def __repr__(self) -> str:
        n_labelled = sum(l != -1 for l in self.labels)
        return (
            f"EvalDataset(name={self.name!r}, n={len(self)}, "
            f"labelled={n_labelled}, split={self.split!r})"
        )
