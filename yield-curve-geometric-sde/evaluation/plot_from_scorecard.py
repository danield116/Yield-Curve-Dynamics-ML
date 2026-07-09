"""Regenerate comparison figures from a saved scorecard CSV (no checkpoints required)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import constraint_ablation_summary, save_scorecard
from visualization.plot_curves import plot_horizon_curve, plot_scorecard_bar

STAGE_B_ABLATIONS = ["sde_only", "sde_pde", "sde_jacobian", "sde_both"]


def load_scorecard_rows(csv_path: Path) -> list[dict]:
    frame = pd.read_csv(csv_path)
    rows = []
    for record in frame.to_dict(orient="records"):
        row = {k: (None if pd.isna(v) else v) for k, v in record.items()}
        arb = {}
        for key, value in list(row.items()):
            if key.startswith("arb_") and value is not None:
                arb[key.replace("arb_", "", 1)] = value
        if arb:
            row["arbitrage"] = arb
        rows.append(row)
    return rows


def save_plots_from_scorecard(rows: list[dict], output_dir: Path, config: dict) -> Path:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = config.get("evaluation", {})
    horizons = eval_cfg.get("horizons", [1, 5, 21, 63])
    primary_h = horizons[0] if horizons else 1

    h_rows = [r for r in rows if r.get("horizon") == primary_h and r.get("horizon", 0) > 0]
    if h_rows:
        plot_scorecard_bar(
            h_rows,
            metric="curve_rmse",
            output_path=figures_dir / f"curve_rmse_by_model_h{primary_h}.png",
        )

    constraint_rows = [
        r
        for r in rows
        if str(r.get("model", "")).startswith("stage_b_") and r.get("horizon") == primary_h
    ]
    for metric_key, file_stub in [
        ("manifold_off_manifold_rmse", "manifold_off_manifold_rmse"),
        ("manifold_correction_gain", "manifold_correction_gain"),
        ("tangent_move_residual_rmse", "tangent_move_residual_rmse"),
        ("arb_forward_smoothness", "arb_forward_smoothness"),
        ("arb_discount_monotonicity_violations", "arb_discount_monotonicity_violations"),
        ("arb_scenario_stability", "arb_scenario_stability"),
    ]:
        plot_rows = []
        for row in constraint_rows:
            if metric_key.startswith("arb_"):
                arb = row.get("arbitrage") or {}
                inner_key = metric_key.replace("arb_", "", 1)
                if inner_key in arb:
                    plot_rows.append({**row, metric_key: arb[inner_key]})
            elif metric_key in row and row[metric_key] is not None:
                plot_rows.append(row)
        if plot_rows:
            plot_scorecard_bar(
                plot_rows,
                metric=metric_key,
                output_path=figures_dir / f"{file_stub}_stage_b_h{primary_h}.png",
            )

    for ablation in STAGE_B_ABLATIONS:
        model_name = f"stage_b_{ablation}"
        if any(r["model"] == model_name for r in rows):
            plot_horizon_curve(
                rows,
                model=model_name,
                metric="curve_rmse",
                output_path=figures_dir / f"{model_name}_rmse_vs_horizon.png",
            )

    return figures_dir


def main():
    parser = argparse.ArgumentParser(
        description="Plot paper figures from a scorecard CSV (no model retrain needed)."
    )
    parser.add_argument(
        "--scorecard",
        default=str(PROJECT_ROOT / "reports" / "comparison" / "paper_best_scorecard.csv"),
        help="Input scorecard CSV.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "default.yaml"),
        help="Config for horizons / tenor labels.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "reports" / "comparison" / "paper_best_figures"),
        help="Directory for PNG outputs.",
    )
    args = parser.parse_args()

    csv_path = Path(args.scorecard)
    if not csv_path.exists():
        raise FileNotFoundError(f"Scorecard not found: {csv_path}")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = load_scorecard_rows(csv_path)
    rows = [r for r in rows if not str(r.get("model", "")).endswith("_hard")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tenors = config.get("data", {}).get("tenors")
    summary_horizons = config.get("evaluation", {}).get("constraint_summary_horizons", [1, 5, 21])
    save_scorecard(rows, output_dir, tenors=tenors, constraint_summary_horizons=summary_horizons)
    figures_dir = save_plots_from_scorecard(rows, output_dir, config)

    print(f"Loaded scorecard: {csv_path.resolve()}")
    print(f"Saved figures to: {figures_dir.resolve()}")
    print(f"Scorecard copy: {(output_dir / 'scorecard.csv').resolve()}")
    for summary_h in summary_horizons:
        summary = constraint_ablation_summary(pd.read_csv(output_dir / "scorecard.csv"), horizon=summary_h)
        if not summary.empty:
            print(f"\nConstraint ablation @ h={summary_h}:")
            print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
