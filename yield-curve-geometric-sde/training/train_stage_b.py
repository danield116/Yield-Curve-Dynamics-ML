"""Stage B training entrypoint: latent Neural SDE + constraint ablations."""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constraints.no_arbitrage_pde import total_constraint_loss
from data.dataloaders import build_latent_window_dataloaders
from training.manifold_ops import linearized_curve_forecast, make_manifold_ops
from training.sde_utils import build_latent_neural_sde, build_sde_input, roll_latent_forecast
from training.train_stage_a import build_stage_a_model, load_config, set_seed, get_device


ABLATION_PRESETS = {
    "sde_only": {"use_pde": False, "use_jacobian": False, "use_diag": False},
    "sde_pde": {"use_pde": True, "use_jacobian": False, "use_diag": True},
    "sde_jacobian": {"use_pde": False, "use_jacobian": True, "use_diag": False},
    "sde_both": {"use_pde": True, "use_jacobian": True, "use_diag": True},
}


def resolve_ablation_flags(config, ablation_override=None):
    """Map ablation name to constraint toggles."""
    constraints_cfg = config.get("constraints", {})
    ablation = ablation_override or constraints_cfg.get("ablation", "sde_only")

    if ablation not in ABLATION_PRESETS:
        raise ValueError(f"Unknown ablation: {ablation!r}. Expected one of {list(ABLATION_PRESETS)}")

    flags = ABLATION_PRESETS[ablation].copy()
    flags["ablation"] = ablation
    flags["lambda_pde"] = float(constraints_cfg.get("pde_penalty_weight", 0.1))
    flags["lambda_diag"] = float(constraints_cfg.get("pde_penalty_weight", 0.1))
    flags["lambda_jac"] = float(constraints_cfg.get("jacobian_projection_weight", 0.1))
    flags["projection_method"] = constraints_cfg.get("projection_method", "reencode")
    flags["include_hessian"] = bool(constraints_cfg.get("include_hessian", False))
    return flags


def load_stage_a_checkpoint(config, device, checkpoint_path=None):
    """Load frozen Stage A model from checkpoint."""
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    if checkpoint_path is None:
        variant = model_cfg.get("stage_a_variant", "student_t_cvae")
        checkpoint_path = training_cfg.get(
            "stage_a_checkpoint",
            f"reports/checkpoints/stage_a/stage_a_{variant}_best.pt",
        )

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Stage A checkpoint not found: {ckpt_path}. Run training/train_stage_a.py first."
        )

    stage_a = build_stage_a_model(config).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    stage_a.load_state_dict(checkpoint["model_state_dict"])
    stage_a.eval()
    for param in stage_a.parameters():
        param.requires_grad = False

    print(f"Loaded Stage A checkpoint: {ckpt_path.resolve()}")
    return stage_a


def load_stage_b_checkpoint(config, device, ablation: str, checkpoint_path=None):
    """Load trained Stage B SDE checkpoint."""
    training_cfg = config.get("training", {})
    if checkpoint_path is None:
        checkpoint_path = Path(training_cfg.get("checkpoint_dir_stage_b", "reports/checkpoints/stage_b"))
        checkpoint_path = checkpoint_path / f"stage_b_{ablation}_best.pt"
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage B checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = checkpoint.get("config", config)
    sde = build_latent_neural_sde(ckpt_config).to(device)
    sde.load_state_dict(checkpoint["sde_state_dict"])
    sde.eval()
    return sde, checkpoint


def make_full_curve_decoder(stage_a, level, use_levelscript):
    """Return decode(z)->full_curve callable for constraint modules."""
    _, decode_fn = make_manifold_ops(stage_a, level, use_levelscript)
    return decode_fn


def compute_stage_b_loss(batch, sde, stage_a, config, ablation_flags, dt=1.0, forecast_horizon=1):
    """Composite Stage B loss: latent fit + curve fit + optional constraints."""
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    use_levelscript = bool(data_cfg.get("use_levelscript", False))
    level_tenor_index = int(data_cfg.get("level_tenor_index", 3))
    history_steps = int(training_cfg.get("latent_history_steps", 5))
    latent_weight = float(training_cfg.get("latent_fit_weight", 0.05))
    curve_weight = float(training_cfg.get("curve_loss_weight", 1.0))

    z_hist = batch["z_hist"]
    z_fut = batch["z_fut"]
    y_fut = batch["y_fut"]
    y_hist = batch["y_hist"]
    forecast_horizon = max(1, int(forecast_horizon))

    z_window = z_hist
    z_t = z_window[:, -1, :]
    for _ in range(forecast_horizon):
        sde_input = build_sde_input(z_window, history_steps)
        z_t = z_t + sde.drift_p(sde_input) * dt
        z_window = torch.cat([z_window[:, 1:, :], z_t.unsqueeze(1)], dim=1)

    z_target = z_fut[:, forecast_horizon - 1, :]
    fit_loss = F.mse_loss(z_t, z_target)

    level = None
    if use_levelscript:
        level = y_hist[:, -1, level_tenor_index : level_tenor_index + 1]

    decode_fn = make_full_curve_decoder(stage_a, level, use_levelscript)
    y_true = y_fut[:, forecast_horizon - 1, :]
    y_pred_curve = decode_fn(z_t)
    curve_loss = F.mse_loss(y_pred_curve, y_true)

    loss = latent_weight * fit_loss + curve_weight * curve_loss

    metrics = {
        "loss": loss.item(),
        "fit": fit_loss.item(),
        "curve_loss": curve_loss.item(),
        "constraint": 0.0,
        "latent_rmse": math.sqrt(fit_loss.detach().item()),
        "curve_rmse": math.sqrt(curve_loss.detach().item()),
        "curve_mae": F.l1_loss(y_pred_curve.detach(), y_true).item(),
    }

    if ablation_flags["use_pde"] or ablation_flags["use_jacobian"] or ablation_flags["use_diag"]:
        encode_fn, _ = make_manifold_ops(
            stage_a, level, use_levelscript, level_tenor_index=level_tenor_index
        )
        z_start = z_hist[:, -1, :]
        y_constraint = linearized_curve_forecast(decode_fn, z_start, z_t)
        sde_input = build_sde_input(z_hist, history_steps)
        mu_q = sde.drift_q(sde_input)
        sigma = sde.diffusion(sde_input)

        constraint_loss = total_constraint_loss(
            y=y_constraint,
            z=z_start,
            decoder=decode_fn,
            encoder=stage_a,
            encode_fn=encode_fn,
            mu_q=mu_q,
            sigma=sigma,
            lambda_pde=ablation_flags["lambda_pde"],
            lambda_diag=ablation_flags["lambda_diag"],
            lambda_jac=ablation_flags["lambda_jac"],
            use_pde=ablation_flags["use_pde"],
            use_diag=ablation_flags["use_diag"],
            use_jacobian=ablation_flags["use_jacobian"],
            projection_method=ablation_flags["projection_method"],
            include_hessian=ablation_flags["include_hessian"],
        )
        loss = loss + constraint_loss
        metrics["constraint"] = constraint_loss.item()
        metrics["loss"] = loss.item()

    return loss, metrics


def run_epoch(sde, stage_a, loader, optimizer, device, config, ablation_flags, dt, train=True):
    if train:
        sde.train()
    else:
        sde.eval()

    training_cfg = config.get("training", {})
    train_horizons = [int(h) for h in training_cfg.get("train_horizons", [1])]
    val_horizon = int(training_cfg.get("val_horizon", 1))

    totals = {
        "loss": 0.0,
        "fit": 0.0,
        "curve_loss": 0.0,
        "constraint": 0.0,
        "latent_rmse": 0.0,
        "curve_rmse": 0.0,
        "curve_mae": 0.0,
    }
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        if train:
            optimizer.zero_grad()
            forecast_horizon = int(np.random.choice(train_horizons))
        else:
            forecast_horizon = val_horizon

        loss, metrics = compute_stage_b_loss(
            batch=batch,
            sde=sde,
            stage_a=stage_a,
            config=config,
            ablation_flags=ablation_flags,
            dt=dt,
            forecast_horizon=forecast_horizon,
        )

        if train:
            loss.backward()
            optimizer.step()

        for key in totals:
            totals[key] += metrics[key]
        n_batches += 1

    for key in totals:
        totals[key] /= max(n_batches, 1)
    return totals


@torch.no_grad()
def save_forecast_paths(sde, stage_a, loader, device, config, output_path, dt=1.0, max_batches=20):
    """Simulate short latent paths and decode to yield curves."""
    data_cfg = config.get("data", {})
    use_levelscript = bool(data_cfg.get("use_levelscript", False))
    level_tenor_index = int(data_cfg.get("level_tenor_index", 3))

    sde.eval()
    stage_a.eval()

    z_paths = []
    y_paths = []
    z_true_paths = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        z_hist = batch["z_hist"]
        horizon = batch["z_fut"].shape[1]

        z_window = z_hist
        z_sim = [z_window[:, -1, :]]
        for _ in range(horizon):
            sde_input = build_sde_input(z_window, sde.history_steps)
            z = z_window[:, -1, :] + sde.drift_p(sde_input) * dt
            z_window = torch.cat([z_window[:, 1:, :], z.unsqueeze(1)], dim=1)
            z_sim.append(z)

        z_path = torch.stack(z_sim, dim=1)
        level = None
        if use_levelscript:
            level = batch["y_hist"][:, -1, level_tenor_index : level_tenor_index + 1]

        decode_fn = make_full_curve_decoder(stage_a, level, use_levelscript)
        y_path = torch.stack([decode_fn(z_path[:, t, :]) for t in range(z_path.shape[1])], dim=1)

        z_paths.append(z_path.cpu())
        y_paths.append(y_path.cpu())
        z_true_paths.append(batch["z_fut"].cpu())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "z_paths": torch.cat(z_paths, dim=0),
            "y_paths": torch.cat(y_paths, dim=0),
            "z_true_paths": torch.cat(z_true_paths, dim=0),
        },
        output_path,
    )
    print(f"Saved forecast paths: {output_path.resolve()}")


def train_stage_b(config, ablation_override=None):
    """Train latent Neural SDE with optional constraint ablations."""
    project_cfg = config.get("project", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    set_seed(int(project_cfg.get("seed", 42)))
    device = get_device()
    print(f"Using device: {device}")

    ablation_flags = resolve_ablation_flags(config, ablation_override=ablation_override)
    ablation_name = ablation_flags["ablation"]
    checkpoint_metric = training_cfg.get("checkpoint_metric", "curve_rmse")
    history_steps = int(training_cfg.get("latent_history_steps", 5))
    latent_weight = float(training_cfg.get("latent_fit_weight", 0.05))
    curve_weight = float(training_cfg.get("curve_loss_weight", 1.0))
    train_horizons = [int(h) for h in training_cfg.get("train_horizons", [1])]
    val_horizon = int(training_cfg.get("val_horizon", 1))

    print(f"Stage B ablation: {ablation_name}")
    print(
        f"SDE history steps: {history_steps} | "
        f"latent_fit_weight: {latent_weight} | curve_loss_weight: {curve_weight}"
    )
    print(f"Train horizons: {train_horizons} | val horizon: {val_horizon}")
    print(f"Checkpoint metric: {checkpoint_metric}")

    latent_dir = training_cfg.get("latent_dir", "reports/latents/stage_a")
    processed_dir = training_cfg.get("processed_dir", "data/processed")
    lookback = int(training_cfg.get("lookback", 21))
    horizon = int(training_cfg.get("horizon", max(max(train_horizons), val_horizon)))
    dt = float(training_cfg.get("dt", 1.0))

    if lookback < history_steps:
        raise ValueError(
            f"lookback ({lookback}) must be >= latent_history_steps ({history_steps})."
        )
    if horizon < max(max(train_horizons), val_horizon):
        raise ValueError(
            f"horizon ({horizon}) must be >= max(train_horizons, val_horizon) "
            f"({max(max(train_horizons), val_horizon)})."
        )

    loaders = build_latent_window_dataloaders(
        latent_dir=latent_dir,
        processed_dir=processed_dir,
        batch_size=int(training_cfg.get("batch_size", 128)),
        lookback=lookback,
        horizon=horizon,
        scaled=True,
    )

    stage_a = load_stage_a_checkpoint(config, device)
    sde = build_latent_neural_sde(config).to(device)

    optimizer = torch.optim.Adam(
        sde.parameters(),
        lr=float(training_cfg.get("learning_rate_stage_b", training_cfg.get("learning_rate", 1e-3))),
    )

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir_stage_b", "reports/checkpoints/stage_b"))
    forecast_dir = Path(training_cfg.get("forecast_dir", "reports/forecasts/stage_b"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    forecast_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(training_cfg.get("epochs_stage_b", 150))
    best_metric_value = float("inf")
    best_state = None
    history = []

    print(f"Train batches: {len(loaders.train)} | Val batches: {len(loaders.val)}")

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            sde, stage_a, loaders.train, optimizer, device, config, ablation_flags, dt, train=True
        )
        val_metrics = run_epoch(
            sde, stage_a, loaders.val, optimizer, device, config, ablation_flags, dt, train=False
        )

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss {train_metrics['loss']:.4f} "
            f"(fit {train_metrics['fit']:.4f}, curve {train_metrics['curve_loss']:.4f}, "
            f"constraint {train_metrics['constraint']:.4f}) | "
            f"Val loss {val_metrics['loss']:.4f} | "
            f"Val accuracy: latent_rmse {val_metrics['latent_rmse']:.4f}, "
            f"curve_rmse {val_metrics['curve_rmse']:.4f}, "
            f"curve_mae {val_metrics['curve_mae']:.4f}"
        )

        metric_value = val_metrics.get(checkpoint_metric, val_metrics["curve_rmse"])
        if metric_value < best_metric_value:
            best_metric_value = metric_value
            best_state = copy.deepcopy(sde.state_dict())

    if best_state is not None:
        sde.load_state_dict(best_state)

    best_curve_rmse = min(row["val"]["curve_rmse"] for row in history) if history else float("inf")

    ckpt_path = checkpoint_dir / f"stage_b_{ablation_name}_best.pt"
    torch.save(
        {
            "sde_state_dict": sde.state_dict(),
            "config": config,
            "ablation": ablation_name,
            "ablation_flags": ablation_flags,
            "best_val_loss": min(row["val"]["loss"] for row in history) if history else float("inf"),
            "best_val_curve_rmse": best_curve_rmse,
            "checkpoint_metric": checkpoint_metric,
            "best_checkpoint_metric": best_metric_value,
            "history": history,
        },
        ckpt_path,
    )
    print(
        f"Saved best Stage B checkpoint: {ckpt_path.resolve()} "
        f"(best {checkpoint_metric}={best_metric_value:.4f}, best curve_rmse={best_curve_rmse:.4f})"
    )

    history_path = checkpoint_dir / f"stage_b_{ablation_name}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    save_forecast_paths(
        sde=sde,
        stage_a=stage_a,
        loader=loaders.val,
        device=device,
        config=config,
        output_path=forecast_dir / f"{ablation_name}_val_paths.pt",
        dt=dt,
    )

    print("Stage B training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train Stage B latent Neural SDE.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "default.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--ablation",
        choices=list(ABLATION_PRESETS.keys()),
        default=None,
        help="Constraint ablation preset (overrides config.constraints.ablation).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    train_stage_b(config, ablation_override=args.ablation)


if __name__ == "__main__":
    main()
