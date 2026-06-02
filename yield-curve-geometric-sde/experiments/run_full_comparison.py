"""Main ablation runner for full model comparison."""


def run_ablation_suite(config: dict) -> None:
    """Run all requested variants.

    Variants:
    1) unconstrained Neural SDE
    2) Neural SDE + no-arbitrage PDE penalty
    3) Neural SDE + decoder-Jacobian projection
    4) Neural SDE + both PDE and Jacobian constraints
    """
    # TODO:
    # - train/load Stage A
    # - train each Stage B variant
    # - evaluate horizons [1, 5, 21, 63]
    # - aggregate metrics into report tables
    _ = config


def main() -> None:
    """CLI entrypoint for experiment orchestration."""
    # TODO: parse CLI args + read YAML config.
    print("[TODO] Run full comparison experiment.")


if __name__ == "__main__":
    main()
