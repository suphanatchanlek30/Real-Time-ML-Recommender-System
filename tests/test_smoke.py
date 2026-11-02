def test_packages_importable():
    import src.etl  # noqa: F401
    import src.evaluation  # noqa: F401
    import src.features  # noqa: F401
    import src.serving  # noqa: F401
    import src.streaming  # noqa: F401
    import src.training  # noqa: F401
