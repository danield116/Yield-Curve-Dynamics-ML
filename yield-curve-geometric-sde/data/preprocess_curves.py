"""Preprocess yield curves for train/val/test and model ingestion."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class SplitData:
    """Container for chronological splits.

    Shapes:
    - train: [T_train, N_tenors]
    - val:   [T_val, N_tenors]
    - test:  [T_test, N_tenors]
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def align_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Align maturities and fill missing values with robust policy."""
    # TODO: sort index, reindex to business days, fill gaps (ffill + interpolation).
    return df.copy()


def chronological_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> SplitData:
    """Chronological split without leakage."""
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return SplitData(train=train, val=val, test=test)


def robust_scale(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Scale by train-only robust statistics (median/IQR style)."""
    # TODO: Replace simple placeholder with robust scaler object.
    median = train.median(axis=0)
    iqr = (train.quantile(0.75) - train.quantile(0.25)).replace(0.0, 1.0)
    return {
        "train": (train - median) / iqr,
        "val": (val - median) / iqr,
        "test": (test - median) / iqr,
    }


def apply_levelscript(curves: np.ndarray, level_tenor_index: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Optional decomposition into shape + level.

    Inputs:
    - curves shape: [T, N_tenors]
    Returns:
    - shape component: [T, N_tenors]
    - level component: [T, 1]
    """
    level = curves[:, [level_tenor_index]]
    shape = curves - level
    return shape, level
