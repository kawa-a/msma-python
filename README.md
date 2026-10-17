# msma-python

The public entry point is `msma()`, matching the R package design. Version 0.0.2 connects the single-matrix S4PCA path; PCA, supervised PCA, PLS, and multiblock paths are declared by the dispatcher but intentionally raise `NotImplementedError` until implemented.

```python
from msma import msma

fit = msma(
    X,
    Z=Z,
    analysis="s4pca",
    comp=3,
    lambda_x=0.10,
    mu_x=0.80,
    structure_method="soft",
    gamma_x=0.10,
    history_method="sumabs",
    n_iter=30,
    scaling=False,
    random_state=4,
)

print(fit.summary(detailed=True))
```

Automatic structured dispatch is also supported: `gamma_x > 0` or `structure_method != "none"` selects S4PCA. Individual estimators remain available with `from msma import S4PCA` for scikit-learn-style use.

## Repeated-split model selection

```python
from msma import S4PCA
from msma.model_selection import select_msma

selection = select_msma(
    estimator=S4PCA(mu_x=0.8, max_iter=20),
    X=X,
    z=Z,
    param_grid={
        "lambda_x": [0.05, 0.10],
        "gamma_x": [0.0, 0.10],
        "n_components": [2, 3],
    },
    criterion="calibration_quantile",
    constraints={"max_overlap": 0.50},
    alpha=0.10,
    n_repeats=3,
    calibration_fraction=0.25,
    random_state=4,
)
```

This is reconstruction-based conformal-style selection. It does not claim a conformal prediction coverage guarantee.
