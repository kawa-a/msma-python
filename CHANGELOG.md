# Changelog

## 0.1.0rc1

### Added

- Public `msma()` dispatcher with the single-matrix S4PCA route.
- `S4PCA` estimator with soft and hard-exclusive structures.
- `sumabs`, `any`, and `count` loading-history rules.
- Fixed-iteration R-compatible reference engine.
- Training and new-data reconstruction.
- Overlap measures, diagnostics, and summaries.
- Repeated-split reconstruction-based conformal-style model selection.
- R golden-fixture regression workflow.

### Known numerical behavior

Unsupervised sparse paths can be platform sensitive after projection deflation.
The reference algorithm can produce a nearly zero next-component loading
candidate; normalization then amplifies BLAS-dependent differences. This
condition is recorded in `diagnostics_`.
