"""Stage A training entrypoint: VAE/CVAE/Student-t variants."""

from pathlib import Path

import torch.nn as nn

from models.cvae import CVAE
from models.student_t_vae import StudentTCVAE
from models.vae import VAE


def build_stage_a_model(config: dict) -> nn.Module:
    """Instantiate Stage A model from config.

    Reads:
    - config["model"]["stage_a_variant"]: vae | cvae | student_t_cvae
    - config["model"]["latent_dim"]
    - config["data"]["tenors"] -> n_tenors
    - config["data"]["use_levelscript"] -> cond_dim for CVAE variants

    LevelScript note:
    - When use_levelscript is True, CVAE models use cond_dim=1 (rate level).
    - The training loop will pass shape + level; model construction only sets widths.
    """
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
        f"Unknown stage_a_variant: {variant!r}. "
        "Expected one of: vae, cvae, student_t_cvae."
    )


def train_stage_a(config: dict) -> None:
    """Train reconstruction manifold model.

    Pseudocode:
    1) load processed data
    2) create model/optimizer
    3) run epoch loop with ELBO (or Student-t ELBO)
    4) save checkpoints + latent embeddings
    """
    # TODO: implement torch training loop + logging.
    _ = config


def main() -> None:
    """CLI entrypoint for Stage A."""
    # TODO: read YAML config and dispatch training.
    config_path = Path("config/default.yaml")
    print(f"[TODO] Train Stage A using config: {config_path}")


if __name__ == "__main__":
    main()
