"""Evaluate trained Stage A/B models and baselines on held-out splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.nelson_siegel import rolling_forecast_nss
from baselines.pca_var import rolling_forecast_pca_var
from data.dataloaders import build_latent_window_dataloaders, build_pointwise_dataloaders
from evaluation.arbitrage_diagnostics import curve_arbitrage_metrics
from evaluation.metrics import (
    latent_rmse,
    mean_curve_rmse,
    mae_by_tenor,
    rmse_by_tenor,
    save_scorecard,
    segment_errors,
)
from models.neural_sde import LatentNeuralSDE
from training.manifold_ops import make_manifold_ops
from training.train_stage_a import (
    build_stage_a_model,
    forward_model,
    load_config,
    prepare_batch,
)
from training.train_stage_b import ABLATION_PRESETS, load_stage_a_checkpoint


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_stage_b_checkpoint(config, ablation: str, device, checkpoint_path=None):
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    if checkpoint_path is None:
        checkpoint_path = training_cfg.get(
            "checkpoint_dir_stage_b",
            "reports/checkpoints/stage_b",
        )
        checkpoint_path = Path(checkpoint_path) / f"stage_b_{ablation}_best.pt"
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage B checkpoint not found: {checkpoint_path}")

    sde = LatentNeuralSDE(latent_dim=int(model_cfg.get("latent_dim", 3))).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sde.load_state_dict(checkpoint["sde_state_dict"])
    sde.eval()
    return sde, checkpoint


@torch.no_grad()
def evaluate_stage_a_reconstruction(config, split="test", device=None) -> dict:
    device = device or get_device()
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    variant = model_cfg.get("stage_a_variant", "student_t_cvae")
    ckpt_path = Path(training_cfg.get("checkpoint_dir", "reports/checkpoints/stage_a")) / f"stage_a_{variant}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage A checkpoint not found: {ckpt_path}")

    model = build_stage_a_model(config).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loaders = build_pointwise_dataloaders(
        processed_dir=training_cfg.get("processed_dir", "data/processed"),
        batch_size=int(training_cfg.get("batch_size", 128)),
        scaled=True,
    )
    loader = {"train": loaders.train, "val": loaders.val, "test": loaders.test}[split]

    use_levelscript = bool(data_cfg.get("use_levelscript", False))
    level_tenor_index = int(data_cfg.get("level_tenor_index", 3))
    variant = model_cfg.get("stage_a_variant", "student_t_cvae")

    y_true_all = []
    y_pred_all = []

    for curves in loader:
        curves = curves.to(device)
        x, cond, level = prepare_batch(curves, use_levelscript, level_tenor_index)
        outputs = forward_model(model, x, cond, variant)
        x_hat = outputs[0]
        if use_levelscript:
            y_true = (x + level).cpu().numpy()
            y_pred = (x_hat + level).cpu().numpy()
        else:
            y_true = x.cpu().numpy()
            y_pred = x_hat.cpu().numpy()
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    arb = curve_arbitrage_metrics(y_pred)

    return {
        "model": f"stage_a_{variant}",
        "split": split,
        "horizon": 0,
        "curve_rmse": mean_curve_rmse(y_true, y_pred),
        "curve_mae_mean": float(mae_by_tenor(y_true, y_pred).mean()),
        "rmse_by_tenor": rmse_by_tenor(y_true, y_pred),
        "segments": segment_errors(y_true, y_pred),
        "arbitrage": arb,
        "best_val_loss": float(checkpoint.get("best_val_loss", np.nan)),
    }


@torch.no_grad()
def evaluate_stage_b_forecast(
    config,
    ablation: str = "sde_only",
    split: str = "test",
    horizon: int = 1,
    device=None,
) -> dict:
    device = device or get_device()
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    use_levelscript = bool(data_cfg.get("use_levelscript", False))
    level_tenor_index = int(data_cfg.get("level_tenor_index", 3))
    lookback = int(training_cfg.get("lookback", 21))
    dt = float(training_cfg.get("dt", 1.0))

    loaders = build_latent_window_dataloaders(
        latent_dir=training_cfg.get("latent_dir", "reports/latents/stage_a"),
        processed_dir=training_cfg.get("processed_dir", "data/processed"),
        batch_size=int(training_cfg.get("batch_size", 128)),
        lookback=lookback,
        horizon=horizon,
        scaled=True,
    )
    loader = {"train": loaders.train, "val": loaders.val, "test": loaders.test}[split]

    stage_a = load_stage_a_checkpoint(config, device)
    sde, checkpoint = load_stage_b_checkpoint(config, ablation, device)

    y_true_all = []
    y_pred_all = []
    z_true_all = []
    z_pred_all = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        z0 = batch["z_hist"][:, -1, :]
        z = z0
        for _ in range(horizon):
            z = z + sde.drift_p(z) * dt

        level = None
        if use_levelscript:
            level = batch["y_fut"][:, 0, level_tenor_index : level_tenor_index + 1]

        _, decode_fn = make_manifold_ops(stage_a, level, use_levelscript, level_tenor_index)
        y_pred = decode_fn(z)
        y_true = batch["y_fut"][:, horizon - 1, :]
        z_true = batch["z_fut"][:, horizon - 1, :]

        y_true_all.append(y_true.cpu().numpy())
        y_pred_all.append(y_pred.cpu().numpy())
        z_true_all.append(z_true.cpu().numpy())
        z_pred_all.append(z.cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    z_true = np.concatenate(z_true_all, axis=0)
    z_pred = np.concatenate(z_pred_all, axis=0)
    arb = curve_arbitrage_metrics(y_pred)

    return {
        "model": f"stage_b_{ablation}",
        "split": split,
        "horizon": horizon,
        "curve_rmse": mean_curve_rmse(y_true, y_pred),
        "curve_mae_mean": float(mae_by_tenor(y_true, y_pred).mean()),
        "latent_rmse": latent_rmse(z_true, z_pred),
        "rmse_by_tenor": rmse_by_tenor(y_true, y_pred),
        "segments": segment_errors(y_true, y_pred),
        "arbitrage": arb,
        "best_val_loss": float(checkpoint.get("best_val_loss", np.nan)),
    }


def evaluate_baseline(
    config,
    baseline_name: str,
    split: str = "test",
    horizon: int = 1,
) -> dict:
    training_cfg = config.get("training", {})
    processed_dir = training_cfg.get("processed_dir", "data/processed")
    lookback = int(training_cfg.get("lookback", 21))

    from data.datasets import load_processed_splits_as_tensors

    splits = load_processed_splits_as_tensors(processed_dir=processed_dir, scaled=True)
    train = splits.train.numpy()
    test = splits.test.numpy() if split == "test" else splits.val.numpy()

    if baseline_name == "pca_var":
        y_true, y_pred = rolling_forecast_pca_var(train, test, lookback=lookback, horizon=horizon)
    elif baseline_name in ("nss", "nelson_siegel", "nelson_siegel_svensson"):
        y_true, y_pred = rolling_forecast_nss(train, test, lookback=lookback, horizon=horizon)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    arb = curve_arbitrage_metrics(y_pred)
    return {
        "model": "nss" if baseline_name in ("nelson_siegel", "nelson_siegel_svensson") else baseline_name,
        "split": split,
        "horizon": horizon,
        "curve_rmse": mean_curve_rmse(y_true, y_pred),
        "curve_mae_mean": float(mae_by_tenor(y_true, y_pred).mean()),
        "rmse_by_tenor": rmse_by_tenor(y_true, y_pred),
        "segments": segment_errors(y_true, y_pred),
        "arbitrage": arb,
        "best_val_loss": np.nan,
    }


def evaluate_run(
    config,
    *,
    split: str = "test",
    horizons: list[int] | None = None,
    ablations: list[str] | None = None,
    baselines: list[str] | None = None,
    include_stage_a: bool = True,
    device=None,
) -> list[dict]:
    eval_cfg = config.get("evaluation", {})
    horizons = horizons or eval_cfg.get("horizons", [1, 5, 21, 63])
    ablations = ablations or list(ABLATION_PRESETS.keys())
    baselines = baselines if baselines is not None else eval_cfg.get("baselines", ["pca_var", "nss"])
    device = device or get_device()

    rows = []
    if include_stage_a:
        rows.append(evaluate_stage_a_reconstruction(config, split=split, device=device))

    for ablation in ablations:
        ckpt = Path(config.get("training", {}).get("checkpoint_dir_stage_b", "reports/checkpoints/stage_b"))
        if not (ckpt / f"stage_b_{ablation}_best.pt").exists():
            print(f"Skipping missing Stage B checkpoint: {ablation}")
            continue
        for horizon in horizons:
            try:
                rows.append(
                    evaluate_stage_b_forecast(
                        config,
                        ablation=ablation,
                        split=split,
                        horizon=horizon,
                        device=device,
                    )
                )
            except Exception as exc:
                print(f"Stage B {ablation} h={horizon} failed: {exc}")

    for baseline_name in baselines:
        for horizon in horizons:
            try:
                rows.append(evaluate_baseline(config, baseline_name, split=split, horizon=horizon))
            except Exception as exc:
                print(f"Baseline {baseline_name} h={horizon} failed: {exc}")

    return rows


def save_evaluation_plots(rows: list[dict], output_dir: str | Path, config: dict) -> Path:
    """Write comparison bar/horizon plots under output_dir/figures."""
    from training.train_stage_b import ABLATION_PRESETS
    from visualization.plot_curves import plot_horizon_curve, plot_scorecard_bar, plot_training_history

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = config.get("evaluation", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    horizons = eval_cfg.get("horizons", [1, 5, 21, 63])
    primary_h = horizons[0] if horizons else 1

    h_rows = [r for r in rows if r.get("horizon") == primary_h and r.get("horizon", 0) > 0]
    if not h_rows:
        h_rows = [r for r in rows if r.get("horizon", 0) > 0]
    if h_rows:
        plot_scorecard_bar(
            h_rows,
            metric="curve_rmse",
            output_path=figures_dir / f"curve_rmse_by_model_h{primary_h}.png",
        )

    for ablation in ABLATION_PRESETS:
        model_name = f"stage_b_{ablation}"
        if any(r["model"] == model_name for r in rows):
            plot_horizon_curve(
                rows,
                model=model_name,
                metric="curve_rmse",
                output_path=figures_dir / f"{model_name}_rmse_vs_horizon.png",
            )

    variant = model_cfg.get("stage_a_variant", "student_t_cvae")
    stage_a_hist = Path(training_cfg.get("checkpoint_dir", "reports/checkpoints/stage_a")) / (
        f"stage_a_{variant}_history.json"
    )
    if stage_a_hist.exists():
        plot_training_history(stage_a_hist, output_path=figures_dir / "stage_a_val_loss.png")

    stage_b_dir = Path(training_cfg.get("checkpoint_dir_stage_b", "reports/checkpoints/stage_b"))
    for hist_path in sorted(stage_b_dir.glob("stage_b_*_history.json")):
        plot_training_history(hist_path, output_path=figures_dir / f"{hist_path.stem}.png")

    return figures_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained yield-curve models.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="reports/comparison")
    parser.add_argument("--skip-stage-a", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    tenors = config.get("data", {}).get("tenors")
    rows = evaluate_run(config, split=args.split, include_stage_a=not args.skip_stage_a)
    output_dir = Path(args.output_dir)
    csv_path = save_scorecard(rows, output_dir, tenors=tenors)
    figures_dir = save_evaluation_plots(rows, output_dir, config)

    summary = []
    for row in rows:
        summary.append(
            {
                "model": row["model"],
                "horizon": row["horizon"],
                "curve_rmse": row["curve_rmse"],
                "latent_rmse": row.get("latent_rmse"),
                "segments": row.get("segments"),
                "arbitrage": row.get("arbitrage"),
            }
        )
    print(json.dumps(summary, indent=2))
    print(f"Saved scorecard to: {csv_path.resolve()}")
    print(f"Saved plots to: {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
