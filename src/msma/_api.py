"""Public dispatcher matching the R package's main `msma()` entry point."""

from __future__ import annotations

from .s4pca import S4PCA

_SUPPORTED = {
    "auto",
    "s4pca",
    "pca",
    "supervised_pca",
    "sparse_pca",
    "exclusive_pca",
    "pls",
    "multiblock_pca",
    "multiblock_pls",
}


def _is_multiblock(value):
    return isinstance(value, (list, tuple))


def msma(
    X,
    Y=None,
    Z=None,
    *,
    analysis="auto",
    comp=2,
    lambda_x=0.0,
    mu_x=0.0,
    structure_method="none",
    gamma_x=0.0,
    history_method="sumabs",
    n_iter=30,
    scaling=False,
    scale_z="none",
    random_state=None,
):
    """Fit an msma model selected from inputs and options.

    Version 0.0.2 connects the single-matrix S4PCA path. Other declared
    analysis paths are reserved and fail explicitly until their estimators are
    implemented.
    """
    if analysis not in _SUPPORTED:
        raise ValueError(f"analysis must be one of {sorted(_SUPPORTED)}.")
    x_multi = _is_multiblock(X)
    y_multi = _is_multiblock(Y)
    if x_multi:
        target = "multiblock_pls" if Y is not None else "multiblock_pca"
        if analysis not in {"auto", target}:
            raise ValueError(f"analysis='{analysis}' is incompatible with multiblock X.")
        raise NotImplementedError(
            f"{target.replace('_', ' ').title()} is planned but not implemented in this version."
        )
    if Y is not None:
        if y_multi:
            raise ValueError("A multiblock Y requires multiblock X.")
        if analysis not in {"auto", "pls"}:
            raise ValueError(f"analysis='{analysis}' is incompatible with Y.")
        raise NotImplementedError("PLS is planned but not implemented in this version.")
    structured = (structure_method != "none") or float(gamma_x) > 0
    if analysis == "auto":
        if structured:
            resolved = "s4pca"
        elif Z is not None:
            resolved = "supervised_pca"
        elif float(lambda_x) > 0:
            resolved = "sparse_pca"
        else:
            resolved = "pca"
    else:
        resolved = analysis
    if resolved == "exclusive_pca":
        structure_method = "exclusive"
        resolved = "s4pca"
    if resolved != "s4pca":
        raise NotImplementedError(
            f"Analysis '{resolved}' is planned but not implemented in this version. To use the implemented path, set analysis='s4pca' or request soft/exclusive structure."
        )
    if structure_method == "none" and float(gamma_x) > 0:
        structure_method = "soft"
    model = S4PCA(
        n_components=comp,
        lambda_x=lambda_x,
        mu_x=mu_x,
        structure_method=structure_method,
        gamma_x=gamma_x,
        history_method=history_method,
        max_iter=n_iter,
        scale=scaling,
        scale_z=scale_z,
        random_state=random_state,
    )
    return model.fit(X, z=Z)
