"""Import smoke tests for core modules."""


def test_import_core_modules() -> None:
    import models.vae  # noqa: F401
    import models.neural_sde  # noqa: F401
    import constraints.jacobian_projection  # noqa: F401
