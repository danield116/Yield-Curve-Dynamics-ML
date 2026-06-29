"""Forecast/reconstruction metrics for yield curves."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEGMENT_SLICES = {
    "short_end": slice(0, 4),
    "belly": slice(4, 8),
    "long_end": slice(8, 11),
}


def rmse_by_tenor(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return RMSE per tenor, shape [N_tenors]."""
    return np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))


def mae_by_tenor(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return MAE per tenor, shape [N_tenors]."""
    return np.abs(y_true - y_pred).mean(axis=0)


def mean_curve_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Aggregate RMSE over full curve and time."""
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def segment_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segment_slices: dict | None = None,
) -> dict:
    """Compute short-end / belly / long-end error summaries."""
    segment_slices = segment_slices or DEFAULT_SEGMENT_SLICES
    out = {}
    for name, sl in segment_slices.items():
        out[name] = float(np.sqrt(((y_true[:, sl] - y_pred[:, sl]) ** 2).mean()))
    return out


def latent_rmse(z_true: np.ndarray, z_pred: np.ndarray) -> float:
    return float(np.sqrt(((z_true - z_pred) ** 2).mean()))


def build_scorecard(rows: list[dict], tenors: list[str] | None = None) -> pd.DataFrame:
    """Flatten evaluation rows into a comparison table."""
    flat_rows = []
    for row in rows:
        item = dict(row)
        segments = item.pop("segments", {}) or {}
        arbitrage = item.pop("arbitrage", {}) or {}
        for key, value in segments.items():
            item[f"seg_{key}"] = value
        for key, value in arbitrage.items():
            item[f"arb_{key}"] = value
        rmse_by_tenor = item.pop("rmse_by_tenor", None)
        flat_rows.append(item)
        if tenors is not None and rmse_by_tenor is not None:
            for idx, tenor in enumerate(tenors):
                if len(rmse_by_tenor) > idx:
                    flat_rows[-1][f"rmse_{tenor}"] = float(rmse_by_tenor[idx])

    frame = pd.DataFrame(flat_rows)
    sort_cols = [c for c in ["horizon", "model"] if c in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return frame


def save_scorecard(rows: list[dict], output_dir: str | Path, tenors: list[str] | None = None) -> Path:
    """Write comparison CSV + JSON summaries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable = []
    for row in rows:
        item = dict(row)
        for key, value in list(item.items()):
            if isinstance(value, np.ndarray):
                item[key] = value.tolist()
        serializable.append(item)

    json_path = output_dir / "scorecard.json"
    json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    frame = build_scorecard(serializable, tenors=tenors)
    csv_path = output_dir / "scorecard.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path
