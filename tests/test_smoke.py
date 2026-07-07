def test_import_and_version():
    import tcr_io

    assert isinstance(tcr_io.__version__, str) and tcr_io.__version__


def test_public_api_surface():
    import tcr_io

    for name in ["TcrDataset", "DatasetIngester", "ReaderFactory", "Filterer", "operations"]:
        assert hasattr(tcr_io, name), name
