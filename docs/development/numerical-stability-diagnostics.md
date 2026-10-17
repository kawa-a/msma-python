# Numerical stability diagnostics

The R-compatible algorithm carries the final score from one component into the
next component. After projection-based deflation, this score is theoretically
orthogonal to the residual matrix. In an unsupervised path, the next initial
loading candidate can therefore be nearly zero. Normalizing that vector may
amplify platform-dependent floating-point differences.

For each extracted component, the implementation records

```text
initial_loading_norm_per_component
initial_direction_norm_per_component
initialization_tolerance_per_component
initial_loading_to_tolerance_ratio
near_zero_component_initialization
```

The scale-aware warning rule is

```text
||X_d.T @ direction||_2
    <= 100 * eps(float64) * ||X_d||_F * ||direction||_2
```

The diagnostics also provide one-based `platform_sensitive_components`, a
boolean `numerical_stability_warning`, a human-readable message, and the exact
rule string. These fields are observational only and do not change the frozen
reference computation.
