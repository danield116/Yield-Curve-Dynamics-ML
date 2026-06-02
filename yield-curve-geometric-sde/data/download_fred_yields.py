"""Download and persist daily yield-curve data from FRED.

Pseudocode-only scaffold:
- configure API connection,
- fetch tenor series,
- align by date index,
- save to `data/raw/`.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd


FRED_SERIES_MAP: Dict[str, str] = {
    # TODO: Validate exact FRED tickers for all tenors.
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}


def fetch_fred_yields(tenors: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Return raw yield panel indexed by date.

    Expected output shape: [num_dates, num_tenors].
    """
    # TODO:
    # 1) Initialize fred client (fredapi / fallback source).
    # 2) Download each tenor series.
    # 3) Merge on calendar date.
    # 4) Return DataFrame with columns in `tenors` order.
    return pd.DataFrame()


def save_raw_data(df: pd.DataFrame, output_path: Path) -> None:
    """Persist downloaded panel to CSV/Parquet."""
    # TODO: create parent dirs, choose file format, write atomically.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_csv(output_path, index=True)


def main() -> None:
    """Minimal entrypoint for command-line usage."""
    # TODO: parse args for start/end/output path.
    tenors = list(FRED_SERIES_MAP.keys())
    df = fetch_fred_yields(tenors=tenors, start_date="1990-01-01", end_date="2026-01-01")
    save_raw_data(df, Path("data/raw/fred_yields.csv"))


if __name__ == "__main__":
    main()
