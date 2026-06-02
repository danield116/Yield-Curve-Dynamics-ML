"""Quick visual diagnostics for raw/clean train-val-test panels."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_frame(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def plot_split_boundaries(train_df, val_df, test_df, save_path):
    """Plot all tenors with vertical lines at split boundaries."""
    fig, ax = plt.subplots(figsize=(14, 6))

    concat_df = pd.concat([train_df, val_df, test_df], axis=0)
    for col in concat_df.columns:
        ax.plot(concat_df.index, concat_df[col], alpha=0.5, linewidth=1.0)

    split_1 = train_df.index[-1]
    split_2 = val_df.index[-1]
    ax.axvline(split_1, color="black", linestyle="--", linewidth=1.5, label="train/val split")
    ax.axvline(split_2, color="red", linestyle="--", linewidth=1.5, label="val/test split")
    ax.set_title("Yield Curves with Chronological Split Boundaries")
    ax.set_xlabel("Date")
    ax.set_ylabel("Yield (scaled or raw)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize train/val/test split boundaries.")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory with train_*.csv split files.")
    parser.add_argument(
        "--suffix",
        choices=["raw", "scaled"],
        default="raw",
        help="Which split files to plot (train_raw/val_raw/test_raw or scaled versions).",
    )
    parser.add_argument("--output-path", default="reports/figures/split_boundaries.png", help="PNG output path.")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    train_df = _read_frame(processed_dir / f"train_{args.suffix}.csv")
    val_df = _read_frame(processed_dir / f"val_{args.suffix}.csv")
    test_df = _read_frame(processed_dir / f"test_{args.suffix}.csv")

    plot_split_boundaries(train_df, val_df, test_df, save_path=Path(args.output_path))
    print(f"Saved split visualization to: {Path(args.output_path).resolve()}")


if __name__ == "__main__":
    main()
