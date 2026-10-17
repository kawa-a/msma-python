# Numerical compatibility contract

The low-level implementation follows the frozen `S3PCA_reference.R` operation order. The next milestone is to generate R golden fixtures containing `X`, `Z`, the normalized initial score, `W`, `S`, RSS, AIC, BIC, EBIC, overlap, and used indices. Tests will compare without sign alignment at `rtol=0`, `atol=1e-12`.

Python `random_state` currently guarantees within-Python reproducibility. It does not reproduce R `set.seed()`/`rnorm()` streams. Golden regression tests will inject the R-generated initial score into the low-level engine.
