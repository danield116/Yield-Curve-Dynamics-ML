"""Visualization helpers for curves, forecasts, and training history."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(
    history_path: str | Path,
    title: str | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot train/val loss curves from a Stage A or Stage B history JSON file."""
    history_path = Path(history_path)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train"]["loss"] for row in history]
    val_loss = [row["val"]["loss"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title or history_path.stem)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_curve_snapshot(
    curve: np.ndarray,
    tenors: list[str],
    title: str = "Yield curve snapshot",
    output_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tenors, curve, marker="o")
    ax.set_xlabel("Tenor")
    ax.set_ylabel("Yield (scaled)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_forecast_fan(
    forecast_paths: np.ndarray,
    tenors: list[str],
    true_curve: np.ndarray | None = None,
    title: str = "Forecast fan",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot percentile bands across simulated paths [N_paths, N_tenors]."""
    fig, ax = plt.subplots(figsize=(8, 4))
    qs = [5, 25, 50, 75, 95]
    percentiles = np.percentile(forecast_paths, qs, axis=0)
    ax.fill_between(
        range(len(tenors)),
        percentiles[0],
        percentiles[-1],
        alpha=0.2,
        label="5-95%",
    )
    ax.fill_between(
        range(len(tenors)),
        percentiles[1],
        percentiles[-2],
        alpha=0.3,
        label="25-75%",
    )
    ax.plot(percentiles[2], marker="o", label="median")
    if true_curve is not None:
        ax.plot(true_curve, marker="x", linestyle="--", label="true")
    ax.set_xticks(range(len(tenors)))
    ax.set_xticklabels(tenors, rotation=45)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_scorecard_bar(
    scorecard_rows: list[dict],
    metric: str = "curve_rmse",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart comparing models at a fixed horizon."""
    rows = [r for r in scorecard_rows if r.get("horizon", 0) > 0]
    if not rows:
        rows = scorecard_rows
    models = [r["model"] for r in rows]
    values = [r[metric] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(models, values)
    ax.set_ylabel(metric)
    ax.set_title(f"Model comparison ({metric})")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_horizon_curve(
    scorecard_rows: list[dict],
    model: str,
    metric: str = "curve_rmse",
    output_path: str | Path | None = None,
) -> plt.Figure:
    rows = sorted([r for r in scorecard_rows if r["model"] == model], key=lambda r: r["horizon"])
    horizons = [r["horizon"] for r in rows]
    values = [r[metric] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(horizons, values, marker="o")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel(metric)
    ax.set_title(f"{model}: {metric} vs horizon")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig
