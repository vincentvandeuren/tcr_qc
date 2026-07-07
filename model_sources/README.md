# model_sources/

Raw upstream model artifacts used to generate the bundled package resources in
`tcr_io/resources/`. **These data files are gitignored** (large / externally
sourced) — only this README is tracked. Obtain them separately and drop them
here, then run the converter:

```bash
python scripts/build_resources.py --build      # -> tcr_io/resources/*.parquet
python scripts/build_resources.py --inspect    # measure sizes only (no to_imgt)
```

## Expected files

| File | Used for | Required for `--build` |
|------|----------|------------------------|
| `tcr2hla_info.zip` | HLA inference (`hla_tcrs.parquet`, `hla_model_weights.parquet`) | yes |
| `CMV_ECOcluster_TCRs.tsv` | CMV EcoCluster hits (`cmv_ecocluster.parquet`) | yes |
| `mait_parsed.parquet` | MAIT hits — **deferred**, not bundled in the beta | no |

`--build` reads only the two TRB entries it needs from `tcr2hla_info.zip`
(`models/TRB_weights.pickle.gz`, `models/TRB_models.pickle`) and applies the
package's Rust `to_imgt` expression, so the compiled extension must be importable
(`maturin develop`).
