"""Minimal smoke tests for scaffold integrity."""


def test_import_scaffold_modules() -> None:
    # TODO: expand with real unit tests once implementations are added.
    import models.vae  # noqa: F401
    import models.neural_sde  # noqa: F401
    import constraints.jacobian_projection  # noqa: F401
