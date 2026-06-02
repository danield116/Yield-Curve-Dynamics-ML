"""DataLoader builders for train/val/test splits.

Readable, notebook-style wrappers around Dataset classes.
"""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from data.datasets import (
    YieldCurveDataset,
    YieldCurveWindowDataset,
    load_processed_splits_as_tensors,
)


@dataclass
class SplitLoaders:
    """Container for three DataLoaders."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def build_pointwise_dataloaders(
    processed_dir="data/processed",
    batch_size=128,
    num_workers=0,
    scaled=True,
):
    """Create pointwise loaders for Stage A reconstruction models."""
    splits = load_processed_splits_as_tensors(
        processed_dir=processed_dir,
        scaled=scaled,
        dtype=torch.float32,
    )
    return SplitLoaders(
        train=DataLoader(
            YieldCurveDataset(splits.train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        val=DataLoader(
            YieldCurveDataset(splits.val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        test=DataLoader(
            YieldCurveDataset(splits.test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )


def build_window_dataloaders(
    processed_dir="data/processed",
    batch_size=128,
    lookback=21,
    horizon=1,
    num_workers=0,
    scaled=True,
):
    """Create sliding-window loaders for Stage B dynamics models."""
    splits = load_processed_splits_as_tensors(processed_dir=processed_dir, scaled=scaled, dtype=torch.float32)
    return SplitLoaders(
        train=DataLoader(
            YieldCurveWindowDataset(splits.train, lookback=lookback, horizon=horizon),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        val=DataLoader(
            YieldCurveWindowDataset(splits.val, lookback=lookback, horizon=horizon),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        test=DataLoader(
            YieldCurveWindowDataset(splits.test, lookback=lookback, horizon=horizon),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )
