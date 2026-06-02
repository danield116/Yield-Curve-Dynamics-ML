"""Stage A training entrypoint: VAE/CVAE/Student-t variants."""

from pathlib import Path


def build_stage_a_model(config: dict):
    """Instantiate model variant based on config."""
    # TODO:
    # - if `vae` -> models.vae.VAE
    # - if `cvae` -> models.cvae.CVAE
    # - if `student_t_cvae` -> models.student_t_vae.StudentTCVAE
    return None


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
