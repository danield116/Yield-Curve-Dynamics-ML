"""Preprocess yield curves for train/val/test and model ingestion."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
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
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    # Reindex to business-day calendar to create a consistent panel.
    full_bday_index = pd.date_range(start=out.index.min(), end=out.index.max(), freq="B")
    out = out.reindex(full_bday_index)

    # Fill gaps with time interpolation first, then directional carries.
    out = out.interpolate(method="time", limit_direction="both")
    out = out.ffill().bfill()
    return out


def chronological_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> SplitData:
    """Chronological split without leakage."""
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return SplitData(train=train, val=val, test=test)


def robust_scale(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
    """Scale by train-only robust statistics (median/IQR style)."""
    median = train.median(axis=0)
    iqr = (train.quantile(0.75) - train.quantile(0.25)).replace(0.0, 1.0)
    scaled = {
        "train": (train - median) / iqr,
        "val": (val - median) / iqr,
        "test": (test - median) / iqr,
    }
    stats = {"median": median, "iqr": iqr}
    return scaled, stats


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


def _save_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=True)
    else:
        df.to_csv(path, index=True)


def _print_summary(name: str, df: pd.DataFrame) -> None:
    print(f"=== {name} ===")
    print(f"Rows: {len(df):,} | Cols: {df.shape[1]}")
    if len(df) > 0:
        print(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Any NA: {df.isna().any().any()}")


def main() -> None:
    """CLI for cleaning/splitting/scaling raw FRED data."""
    parser = argparse.ArgumentParser(description="Preprocess yield-curve panel.")
    parser.add_argument("--input-path", default="data/raw/fred_yields.csv", help="Input raw data path.")
    parser.add_argument("--output-dir", default="data/processed", help="Folder for processed outputs.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Chronological training ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Chronological validation ratio.")
    parser.add_argument("--levelscript", action="store_true", help="Save shape/level decompositions.")
    parser.add_argument(
        "--level-tenor-index",
        type=int,
        default=3,
        help="Column index for level tenor (default 3 -> 1Y if standard tenor ordering).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        raw_df = pd.read_parquet(input_path)
    else:
        raw_df = pd.read_csv(input_path, index_col=0, parse_dates=True)

    clean_df = align_and_impute(raw_df)
    split = chronological_split(clean_df, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    scaled, stats = robust_scale(split.train, split.val, split.test)

    output_dir = Path(args.output_dir)
    _save_frame(clean_df, output_dir / "curves_clean.csv")
    _save_frame(split.train, output_dir / "train_raw.csv")
    _save_frame(split.val, output_dir / "val_raw.csv")
    _save_frame(split.test, output_dir / "test_raw.csv")
    _save_frame(scaled["train"], output_dir / "train_scaled.csv")
    _save_frame(scaled["val"], output_dir / "val_scaled.csv")
    _save_frame(scaled["test"], output_dir / "test_scaled.csv")

    stats_payload = {
        "median": {k: float(v) for k, v in stats["median"].to_dict().items()},
        "iqr": {k: float(v) for k, v in stats["iqr"].to_dict().items()},
    }
    (output_dir / "scaler_stats.json").write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    if args.levelscript:
        for split_name, split_df in [("train", split.train), ("val", split.val), ("test", split.test)]:
            shape, level = apply_levelscript(split_df.to_numpy(), level_tenor_index=args.level_tenor_index)
            np.save(output_dir / f"{split_name}_shape.npy", shape)
            np.save(output_dir / f"{split_name}_level.npy", level)

    _print_summary("RAW INPUT", raw_df)
    _print_summary("CLEAN PANEL", clean_df)
    _print_summary("TRAIN RAW", split.train)
    _print_summary("VAL RAW", split.val)
    _print_summary("TEST RAW", split.test)
    print(f"\nSaved processed outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
