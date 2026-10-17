"""Repeated-split model selection for msma estimators."""

from __future__ import annotations

import numbers
import warnings
from dataclasses import dataclass
from itertools import product

import numpy as np
from sklearn.base import clone

from .._validation import validate_xy
from ..s4pca import S4PCA


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _unit_interval(value, name, *, closed=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    valid = 0.0 <= value <= 1.0 if closed else 0.0 < value < 1.0
    if not valid:
        bounds = "[0, 1]" if closed else "(0, 1)"
        raise ValueError(f"{name} must be in {bounds}.")
    return value


def _candidate_grid(param_grid):
    required = {"lambda_x", "gamma_x", "n_components"}
    if set(param_grid) != required:
        raise ValueError(f"param_grid must contain exactly {sorted(required)}.")
    lambdas = [float(x) for x in param_grid["lambda_x"]]
    gammas = [float(x) for x in param_grid["gamma_x"]]
    components = list(param_grid["n_components"])
    if not lambdas or any(not np.isfinite(x) or x < 0 for x in lambdas):
        raise ValueError("lambda_x candidates must be non-negative finite values.")
    if not gammas or any(not np.isfinite(x) or x < 0 for x in gammas):
        raise ValueError("gamma_x candidates must be non-negative finite values.")
    components = [_positive_int(x, "n_components candidate") for x in components]
    # Match R expand.grid(lambdaX, gammaX, comp): lambda changes fastest.
    rows = []
    seen = set()
    for comp, gamma, lam in product(components, gammas, lambdas):
        key = (lam, gamma, comp)
        if key not in seen:
            seen.add(key)
            rows.append({"lambda_x": lam, "gamma_x": gamma, "n_components": comp})
    return rows


def _median_unbiased_quantile(values, probability):
    """R quantile(type=8) via NumPy's corresponding estimator."""
    return float(np.quantile(values, probability, method="median_unbiased"))


@dataclass
class MSMASelectionResult:
    best_params_: dict[str, object]
    best_estimator_: S4PCA | None
    candidate_results_: list[dict[str, object]]
    split_results_: list[dict[str, object]]
    criterion_: str
    constraints_: dict[str, float]
    n_repeats_: int
    refit_: bool

    def summary(self, precision=6):
        if (
            isinstance(precision, bool)
            or not isinstance(precision, numbers.Integral)
            or precision < 1
        ):
            raise ValueError("precision must be a positive integer.")
        best = self.best_params_
        return "\n".join(
            [
                "MSMA Model Selection Summary",
                "============================",
                f"Criterion               {self.criterion_}",
                f"Repeated splits         {self.n_repeats_}",
                f"Candidate models        {len(self.candidate_results_)}",
                f"lambda_x                {best['lambda_x']:.{precision}f}",
                f"gamma_x                 {best['gamma_x']:.{precision}f}",
                f"n_components            {best['n_components']}",
                f"Median criterion        {best['score']:.{precision}f}",
                f"Median overlap selected {best['overlap_selected']:.{precision}f}",
                f"Admissible              {best['admissible']}",
                f"Refitted                {self.best_estimator_ is not None}",
            ]
        )


def select_msma(
    *,
    estimator,
    X,
    z=None,
    param_grid,
    criterion="calibration_quantile",
    constraints=None,
    alpha=0.10,
    n_repeats=5,
    calibration_fraction=0.25,
    random_state=1,
    refit=True,
):
    """Select hyperparameters by repeated calibration reconstruction error.

    This is reconstruction-based conformal-style model selection, not a
    conformal prediction procedure with a coverage guarantee.
    """
    if not isinstance(estimator, S4PCA):
        raise TypeError("Version 0.0.2 supports selection for S4PCA only.")
    if criterion != "calibration_quantile":
        raise ValueError("criterion must currently be 'calibration_quantile'.")
    alpha = _unit_interval(alpha, "alpha")
    calibration_fraction = _unit_interval(calibration_fraction, "calibration_fraction")
    n_repeats = _positive_int(n_repeats, "n_repeats")
    if not isinstance(refit, (bool, np.bool_)):
        raise ValueError("refit must be True or False.")
    constraints = {} if constraints is None else dict(constraints)
    unknown = set(constraints) - {"max_overlap"}
    if unknown:
        raise ValueError(f"Unknown constraints: {sorted(unknown)}.")
    max_overlap = _unit_interval(constraints.get("max_overlap", 1.0), "max_overlap", closed=True)
    X, z, _ = validate_xy(X, z)
    if X.shape[0] < 4:
        raise ValueError("At least four observations are required for calibration splitting.")
    grid = _candidate_grid(param_grid)
    base_seed = 1 if random_state is None else int(random_state)
    split_results = []

    for repeat in range(1, n_repeats + 1):
        split_seed = base_seed + 1000 * repeat
        rng = np.random.default_rng(split_seed)
        n_cal = max(2, min(X.shape[0] - 2, int(np.ceil(X.shape[0] * calibration_fraction))))
        calibration_indices = np.sort(rng.choice(X.shape[0], size=n_cal, replace=False))
        training_indices = np.setdiff1d(np.arange(X.shape[0]), calibration_indices)
        for candidate_id, params in enumerate(grid, start=1):
            model = clone(estimator).set_params(
                structure_method="soft",
                lambda_x=params["lambda_x"],
                gamma_x=params["gamma_x"],
                n_components=params["n_components"],
                random_state=split_seed + candidate_id,
            )
            model.fit(X[training_indices], z=None if z is None else z[training_indices])
            prediction = model.reconstruct(X[calibration_indices])
            errors = np.mean((X[calibration_indices] - prediction) ** 2, axis=1)
            q_hat = _median_unbiased_quantile(errors, 1.0 - alpha)
            split_results.append(
                {
                    **params,
                    "candidate_id": candidate_id,
                    "repeat": repeat,
                    "score": q_hat,
                    "overlap_all": float(model.overlap_all_),
                    "overlap_selected": float(model.overlap_selected_),
                    "admissible": bool(model.overlap_selected_ <= max_overlap),
                    "training_indices": training_indices.tolist(),
                    "calibration_indices": calibration_indices.tolist(),
                    "calibration_errors": errors.tolist(),
                    "fit_seed": split_seed + candidate_id,
                }
            )

    candidate_results = []
    for params in grid:
        rows = [
            row for row in split_results if all(row[key] == value for key, value in params.items())
        ]
        overlap_all = float(np.median([row["overlap_all"] for row in rows]))
        overlap_selected = float(np.median([row["overlap_selected"] for row in rows]))
        candidate_results.append(
            {
                **params,
                "score": float(np.median([row["score"] for row in rows])),
                "overlap_all": overlap_all,
                "overlap_selected": overlap_selected,
                "admissible": bool(overlap_selected <= max_overlap),
            }
        )

    eligible = [row for row in candidate_results if row["admissible"]]
    if not eligible:
        warnings.warn(
            "No candidate satisfied max_overlap; selecting from the full grid.",
            RuntimeWarning,
            stacklevel=2,
        )
        eligible = candidate_results
    best = min(
        eligible,
        key=lambda row: (
            row["score"],
            row["overlap_selected"],
            row["n_components"],
            row["lambda_x"],
            row["gamma_x"],
        ),
    )
    best_estimator = None
    if refit:
        best_estimator = (
            clone(estimator)
            .set_params(
                structure_method="soft",
                lambda_x=best["lambda_x"],
                gamma_x=best["gamma_x"],
                n_components=best["n_components"],
                random_state=base_seed,
            )
            .fit(X, z=z)
        )
    return MSMASelectionResult(
        best_params_=dict(best),
        best_estimator_=best_estimator,
        candidate_results_=candidate_results,
        split_results_=split_results,
        criterion_=criterion,
        constraints_={"max_overlap": max_overlap},
        n_repeats_=n_repeats,
        refit_=bool(refit),
    )
