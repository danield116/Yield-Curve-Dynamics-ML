"""PyTorch dataset into tensors for yield-curve training.

"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _read_frame(path):
    """Load CSV/Parquet file into a date-indexed DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


@dataclass
class SplitTensors:
    """one tensor per split
    - train: [T_train, N_tenors]
    - val:   [T_val, N_tenors]
    - test:  [T_test, N_tenors]
    """

    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor


def load_processed_splits_as_tensors(
    processed_dir="data/processed",
    scaled=True,
    dtype=torch.float32,
):
    """Load preprocessed train/val/test files and return tensors."""
    processed_path = Path(processed_dir)
    suffix = "scaled" if scaled else "raw"

    train_df = _read_frame(processed_path / f"train_{suffix}.csv")
    val_df = _read_frame(processed_path / f"val_{suffix}.csv")
    test_df = _read_frame(processed_path / f"test_{suffix}.csv")

    return SplitTensors(
        train=torch.tensor(train_df.to_numpy(dtype=np.float32), dtype=dtype),
        val=torch.tensor(val_df.to_numpy(dtype=np.float32), dtype=dtype),
        test=torch.tensor(test_df.to_numpy(dtype=np.float32), dtype=dtype),
    )


class YieldCurveDataset(Dataset):
    """Pointwise dataset for reconstruction training.

    Each item is one curve vector x_t with shape [N_tenors].
    """

    def __init__(self, curves):
        if curves.ndim != 2:
            raise ValueError(f"Expected 2D tensor [T, N], got shape {tuple(curves.shape)}.")
        self.curves = curves

    def __len__(self):
        return self.curves.shape[0]

    def __getitem__(self, idx):
        return self.curves[idx]


class YieldCurveWindowDataset(Dataset):
    """Sliding-window dataset for dynamics models.

    For each item:
    - x_hist: [lookback, N_tenors]
    - y_fut:  [horizon, N_tenors]
    """

    def __init__(self, curves, lookback=21, horizon=1):
        if curves.ndim != 2:
            raise ValueError(f"Expected 2D tensor [T, N], got shape {tuple(curves.shape)}.")
        if lookback <= 0 or horizon <= 0:
            raise ValueError("lookback and horizon must be positive integers.")

        self.curves = curves
        self.lookback = lookback
        self.horizon = horizon
        self.num_samples = curves.shape[0] - lookback - horizon + 1
        if self.num_samples <= 0:
            raise ValueError(
                "Not enough time points for requested lookback/horizon. "
                f"T={curves.shape[0]}, lookback={lookback}, horizon={horizon}."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx
        mid = idx + self.lookback
        end = mid + self.horizon
        return {
            "x_hist": self.curves[start:mid],  # [lookback, N]
            "y_fut": self.curves[mid:end],  # [horizon, N]
        }


class LatentWindowDataset(Dataset):
    """Sliding-window dataset over latent paths (+ curves for LevelScript decode).

    Returns:
    - z_hist: [lookback, latent_dim]
    - y_hist: [lookback, N_tenors]
    - z_fut:  [horizon, latent_dim]
    - y_fut:  [horizon, N_tenors]  (targets for forecast horizons)
    """

    def __init__(self, latents, curves, lookback=21, horizon=1):
        if latents.shape[0] != curves.shape[0]:
            raise ValueError("Latent and curve time dimensions must match.")
        if lookback <= 0 or horizon <= 0:
            raise ValueError("lookback and horizon must be positive integers.")

        self.latents = latents
        self.curves = curves
        self.lookback = lookback
        self.horizon = horizon
        self.num_samples = latents.shape[0] - lookback - horizon + 1
        if self.num_samples <= 0:
            raise ValueError(
                "Not enough time points for requested lookback/horizon. "
                f"T={latents.shape[0]}, lookback={lookback}, horizon={horizon}."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx
        mid = idx + self.lookback
        end = mid + self.horizon
        return {
            "z_hist": self.latents[start:mid],
            "y_hist": self.curves[start:mid],
            "z_fut": self.latents[mid:end],
            "y_fut": self.curves[mid:end],
        }


def load_latent_splits(latent_dir):
    """Load train/val/test latent tensors saved by Stage A."""
    latent_path = Path(latent_dir)
    return SplitTensors(
        train=torch.load(latent_path / "train_latents.pt", map_location="cpu"),
        val=torch.load(latent_path / "val_latents.pt", map_location="cpu"),
        test=torch.load(latent_path / "test_latents.pt", map_location="cpu"),
    )


def maybe_add_levelscript_condition(curves, level_tenor_index=3):
    """Return (shape_curves, level_cond) for CVAE-style conditioning.

    - shape_curves: [T, N_tenors]
    - level_cond:   [T, 1]
    """
    if not (0 <= level_tenor_index < curves.shape[1]):
        raise ValueError(f"level_tenor_index out of range for N={curves.shape[1]}.")

    # level = one tenor (default: 1Y if standard ordering)
    level = curves[:, [level_tenor_index]]
    shape = curves - level
    return shape, level
