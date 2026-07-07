from tcr_io._resources import resource_path


def test_resource_path_points_to_existing_parquet():
    p = resource_path("hla_model_weights.parquet")
    assert p.exists()
    assert p.name == "hla_model_weights.parquet"
