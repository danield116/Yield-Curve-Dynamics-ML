"""Download and persist daily yield-curve data from FRED.

This module uses FRED's public CSV endpoint (no API key required):
https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>
"""

import argparse
from pathlib import Path

import pandas as pd


# Earliest date where all MVP tenors (incl. 1M) are available on FRED.
DEFAULT_START_DATE = "2001-07-01"


FRED_SERIES_MAP = {
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


def _normalize_fred_columns(series_df, series_id):
    """Map FRED CSV headers to a stable date + value schema.

    FRED has used both ``DATE`` and ``observation_date`` for the date column.
    """
    columns = {col.strip(): col for col in series_df.columns}
    normalized = series_df.rename(columns={orig: key for key, orig in columns.items()})

    date_col = None
    for candidate in ("observation_date", "DATE", "date"):
        if candidate in normalized.columns:
            date_col = candidate
            break

    value_col = series_id if series_id in normalized.columns else None
    if value_col is None:
        # Some exports use a generic label when only one series is requested.
        non_date_cols = [col for col in normalized.columns if col != date_col]
        if len(non_date_cols) == 1:
            value_col = non_date_cols[0]

    if date_col is None or value_col is None:
        raise ValueError(
            f"Unexpected FRED response shape for series '{series_id}'. "
            f"Columns received: {list(series_df.columns)}"
        )

    out = normalized[[date_col, value_col]].copy()
    out = out.rename(columns={date_col: "date", value_col: series_id})
    return out


def _fetch_single_series(series_id, start_date, end_date):
    """Fetch one FRED time series from the public CSV endpoint.

    Returns:
    - DataFrame with columns: ["date", series_id].
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    series_df = pd.read_csv(url)
    series_df = _normalize_fred_columns(series_df, series_id)

    series_df["date"] = pd.to_datetime(series_df["date"])
    series_df[series_id] = pd.to_numeric(series_df[series_id], errors="coerce")
    series_df = series_df[(series_df["date"] >= start_date) & (series_df["date"] <= end_date)]
    return series_df[["date", series_id]]


def fetch_fred_yields(tenors, start_date, end_date):
    """Return raw yield panel indexed by date.

    Expected output shape: [num_dates, num_tenors], yields in percent units.
    """
    unknown_tenors = [tenor for tenor in tenors if tenor not in FRED_SERIES_MAP]
    if unknown_tenors:
        raise ValueError(f"Unknown tenor(s): {unknown_tenors}. Supported: {list(FRED_SERIES_MAP)}")

    merged = None
    for tenor in tenors:
        series_id = FRED_SERIES_MAP[tenor]
        series_df = _fetch_single_series(series_id=series_id, start_date=start_date, end_date=end_date)
        series_df = series_df.rename(columns={series_id: tenor})
        merged = series_df if merged is None else merged.merge(series_df, on="date", how="outer")

    assert merged is not None
    merged = merged.sort_values("date").set_index("date")
    merged = merged[tenors]
    return merged


def save_raw_data(df, output_path):
    """Persist downloaded panel to CSV/Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        if output_path.suffix.lower() == ".parquet":
            df.to_parquet(output_path, index=True)
        else:
            df.to_csv(output_path, index=True)
    else:
        raise ValueError("Downloaded DataFrame is empty. Check date range and ticker availability.")


def print_data_summary(df):
    """Print quick diagnostics for visual correctness checks."""
    missing_by_tenor = df.isna().mean().sort_values(ascending=False)
    print("=== FRED DOWNLOAD SUMMARY ===")
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    if len(df) > 0:
        print(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print("\nMissing fraction by tenor:")
    print(missing_by_tenor.to_string())
    print("\nHead:")
    print(df.head(5).to_string())
    print("\nTail:")
    print(df.tail(5).to_string())


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Download U.S. Treasury yields from FRED.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument(
        "--tenors",
        nargs="+",
        default=list(FRED_SERIES_MAP.keys()),
        help="Tenors to fetch, e.g. 1M 3M 6M 1Y 2Y 3Y 5Y 7Y 10Y 20Y 30Y.",
    )
    parser.add_argument("--output-path", default="data/raw/fred_yields.csv", help="Output CSV/Parquet path.")
    args = parser.parse_args()

    df = fetch_fred_yields(tenors=args.tenors, start_date=args.start_date, end_date=args.end_date)
    save_raw_data(df, Path(args.output_path))
    print_data_summary(df)
    print(f"\nSaved raw data to: {Path(args.output_path).resolve()}")


if __name__ == "__main__":
    main()
