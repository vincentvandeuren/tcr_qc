import pytest


def test_hla_inference_loads_bundled_model():
    from tcr_io.operations import HlaInference

    op = HlaInference()  # no args -> bundled parquets
    tcr_cols = set(op.tcrs.collect_schema().names())
    wt_cols = set(op.wts.collect_schema().names())
    assert {"junction_aa", "v_gene", "j_gene", "allele", "gene", "score"} <= tcr_cols
    assert {"allele", "coef_1", "coef_2", "intercept"} <= wt_cols


def test_cmv_hits_loads_bundled_model():
    from tcr_io.operations import ECOClusterHits

    op = ECOClusterHits()  # no args -> bundled parquet
    cols = set(op.eco_df.collect_schema().names())
    assert {"v_gene", "j_gene", "junction_aa", "hla_cocluster", "eco_id"} <= cols
