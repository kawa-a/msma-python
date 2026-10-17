# R 4.0 golden fixtures

Generate the fixtures from the frozen manuscript-derived R reference implementation:

```bash
Rscript tools/r/generate_s4pca_golden_fixtures.R \
  /path/to/msma/tests/testthat/reference/S3PCA_reference.R \
  tests/regression/fixtures/r_v4
```

The generator requires only base R. It writes decimal text using 17 significant digits, records the exact R normal vector and normalized initial score, stores both one-based and zero-based used-feature indices, writes an MD5 manifest, and captures `sessionInfo()`.

The Python regression test injects `initial_score.csv`; therefore it tests the numerical algorithm independently of differences between R and NumPy random-number streams. No sign alignment or component permutation is used. The required comparison is `rtol=0`, `atol=1e-12`.
