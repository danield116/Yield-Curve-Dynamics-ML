"""Stage A training entrypoint: VAE/CVAE/Student-t variants."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Allow running as: python training/train_stage_a.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataloaders import build_pointwise_dataloaders
from data.datasets import load_processed_splits_as_tensors
from models.cvae import CVAE
from models.student_t_vae import StudentTCVAE
from models.vae import VAE


def build_stage_a_model(config):
    """Instantiate Stage A model from config."""
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    variant = model_cfg.get("stage_a_variant", "student_t_cvae")
    latent_dim = int(model_cfg.get("latent_dim", 3))
    hidden_dim = int(model_cfg.get("hidden_dim", 64))

    tenors = data_cfg.get("tenors", ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"])
    n_tenors = len(tenors)

    use_levelscript = bool(data_cfg.get("use_levelscript", False))
    cond_dim = 1 if use_levelscript else 0

    if variant == "vae":
        if use_levelscript:
            raise ValueError("Standard VAE does not support LevelScript conditioning. Use cvae or student_t_cvae.")
        return VAE(n_tenors=n_tenors, latent_dim=latent_dim, hidden_dim=hidden_dim)

    if variant == "cvae":
        if cond_dim == 0:
            raise ValueError("CVAE requires a condition vector. Set data.use_levelscript=true or use vae.")
        return CVAE(n_tenors=n_tenors, cond_dim=cond_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)

    if variant == "student_t_cvae":
        if cond_dim == 0:
            raise ValueError("Student-t CVAE requires a condition vector. Set data.use_levelscript=true.")
        return StudentTCVAE(
            n_tenors=n_tenors,
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

    raise ValueError(
        f"Unknown stage_a_variant: {variant!r}. Expected one of: vae, cvae, student_t_cvae."
    )


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_batch(curves, use_levelscript, level_tenor_index=3):
    """Split curve batch into model input + optional LevelScript condition."""
    if not use_levelscript:
        return curves, None, None

    level = curves[:, [level_tenor_index]]
    shape = curves - level
    return shape, level, level


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def forward_model(model, x, cond, variant):
    if variant == "vae":
        return model(x)

    if variant == "student_t_cvae":
        return model(x, cond)

    return model(x, cond)


def compute_stage_a_loss(model, curves, config):
    """Return total ELBO-style loss and metric dict for one batch."""
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    variant = model_cfg.get("stage_a_variant", "student_t_cvae")
    use_levelscript = bool(data_cfg.get("use_levelscript", False))
    level_tenor_index = int(data_cfg.get("level_tenor_index", 3))

    x, cond, level = prepare_batch(curves, use_levelscript, level_tenor_index)
    outputs = forward_model(model, x, cond, variant)

    if variant == "student_t_cvae":
        x_hat, mu, logvar, log_scale = outputs
        recon = model.student_t_nll(x, x_hat, log_scale)
    else:
        x_hat, mu, logvar = outputs
        recon = F.mse_loss(x_hat, x)

    kl = kl_divergence(mu, logvar)
    loss = recon + kl

    with torch.no_grad():
        if use_levelscript and level is not None:
            full_true = x + level
            full_pred = x_hat + level
            curve_mse = F.mse_loss(full_pred, full_true).item()
        else:
            curve_mse = recon.item() if isinstance(recon, torch.Tensor) else float(recon)

    metrics = {
        "loss": loss.item(),
        "recon": recon.item() if isinstance(recon, torch.Tensor) else float(recon),
        "kl": kl.item(),
        "curve_mse": curve_mse,
    }
    return loss, metrics


def run_epoch(model, loader, optimizer, device, config, train=True):
    if train:
        model.train()
    else:
        model.eval()

    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "curve_mse": 0.0}
    n_batches = 0

    for curves in loader:
        curves = curves.to(device)

        if train:
            optimizer.zero_grad()

        loss, metrics = compute_stage_a_loss(model, curves, config)

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
def encode_latent_means(model, curves_tensor, config, device, batch_size=256):
    """Encode curves to latent means mu (deterministic embedding for Stage B)."""
    model.eval()
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    variant = model_cfg.get("stage_a_variant", "student_t_cvae")
    use_levelscript = bool(data_cfg.get("use_levelscript", False))
    level_tenor_index = int(data_cfg.get("level_tenor_index", 3))

    latents = []
    n = curves_tensor.shape[0]

    for start in range(0, n, batch_size):
        batch = curves_tensor[start : start + batch_size].to(device)
        x, cond, _ = prepare_batch(batch, use_levelscript, level_tenor_index)

        if variant == "vae":
            mu, _ = model.encode(x)
        else:
            mu, _ = model.encode(x, cond)

        latents.append(mu.cpu())

    return torch.cat(latents, dim=0)


def save_latent_embeddings(model, config, device, output_dir):
    """Save latent means for train/val/test splits."""
    training_cfg = config.get("training", {})
    processed_dir = training_cfg.get("processed_dir", "data/processed")

    splits = load_processed_splits_as_tensors(processed_dir=processed_dir, scaled=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, curves in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        z = encode_latent_means(model, curves, config, device)
        out_path = output_dir / f"{split_name}_latents.pt"
        torch.save(z, out_path)
        print(f"Saved {split_name} latents: {tuple(z.shape)} -> {out_path}")


def train_stage_a(config):
    """Train reconstruction manifold model and save checkpoint + latents."""
    project_cfg = config.get("project", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    set_seed(int(project_cfg.get("seed", 42)))
    device = get_device()
    print(f"Using device: {device}")

    processed_dir = training_cfg.get("processed_dir", "data/processed")
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "reports/checkpoints/stage_a"))
    latent_dir = Path(training_cfg.get("latent_dir", "reports/latents/stage_a"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_pointwise_dataloaders(
        processed_dir=processed_dir,
        batch_size=int(training_cfg.get("batch_size", 128)),
        scaled=True,
    )

    model = build_stage_a_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
    )

    epochs = int(training_cfg.get("epochs_stage_a", 100))
    best_val_loss = float("inf")
    best_state = None
    history = []

    print(f"Training Stage A variant: {model_cfg.get('stage_a_variant')}")
    print(f"Train batches: {len(loaders.train)} | Val batches: {len(loaders.val)}")

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, loaders.train, optimizer, device, config, train=True)
        val_metrics = run_epoch(model, loaders.val, optimizer, device, config, train=False)

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss {train_metrics['loss']:.4f} (recon {train_metrics['recon']:.4f}, kl {train_metrics['kl']:.4f}) | "
            f"Val loss {val_metrics['loss']:.4f} (curve_mse {val_metrics['curve_mse']:.4f})"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    variant = model_cfg.get("stage_a_variant", "student_t_cvae")
    ckpt_path = checkpoint_dir / f"stage_a_{variant}_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "best_val_loss": best_val_loss,
            "history": history,
        },
        ckpt_path,
    )
    print(f"Saved best checkpoint to: {ckpt_path.resolve()}")

    history_path = checkpoint_dir / f"stage_a_{variant}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    save_latent_embeddings(model, config, device, latent_dir)
    print("Stage A training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train Stage A manifold model.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "default.yaml"),
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    train_stage_a(config)


if __name__ == "__main__":
    main()
