from __future__ import annotations

import numbers

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._s4pca_reference import fit_reference, project
from ._validation import finite_scalar, positive_int, validate_xy


class S4PCA(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_components=2,
        *,
        lambda_x=0.0,
        mu_x=0.0,
        structure_method="none",
        gamma_x=0.0,
        history_method="sumabs",
        max_iter=30,
        scale=False,
        scale_z="none",
        random_state=None,
    ):
        self.n_components = n_components
        self.lambda_x = lambda_x
        self.mu_x = mu_x
        self.structure_method = structure_method
        self.gamma_x = gamma_x
        self.history_method = history_method
        self.max_iter = max_iter
        self.scale = scale
        self.scale_z = scale_z
        self.random_state = random_state

    def fit(self, X, y=None, *, z=None):
        if y is not None and z is not None:
            raise ValueError("Specify supervision using either y or z, not both.")
        z = y if z is None else z
        comp = positive_int(self.n_components, "n_components")
        nit = positive_int(self.max_iter, "max_iter")
        la = finite_scalar(self.lambda_x, "lambda_x", lower=0)
        ga = finite_scalar(self.gamma_x, "gamma_x", lower=0)
        mu = finite_scalar(self.mu_x, "mu_x", lower=0, upper=1)
        if self.structure_method not in {"none", "soft", "exclusive"}:
            raise ValueError("structure_method must be 'none', 'soft', or 'exclusive'.")
        if self.history_method not in {"sumabs", "any", "count"}:
            raise ValueError("history_method must be 'sumabs', 'any', or 'count'.")
        if self.scale_z != "none":
            raise ValueError("Only scale_z='none' is currently supported.")
        if not isinstance(self.scale, (bool, np.bool_)):
            raise ValueError("scale must be True or False.")
        X, z, names = validate_xy(X, z)
        self.n_features_in_ = X.shape[1]
        if names is not None:
            self.feature_names_in_ = names
        self.x_center_ = np.mean(X, axis=0) if self.scale else np.zeros(X.shape[1])
        self.x_scale_ = np.std(X, axis=0, ddof=1) if self.scale else np.ones(X.shape[1])
        self.x_scale_[~np.isfinite(self.x_scale_) | (self.x_scale_ == 0)] = 1
        Xp = (X - self.x_center_) / self.x_scale_
        method = "soft" if self.structure_method == "none" and ga > 0 else self.structure_method
        r = fit_reference(
            Xp,
            z,
            comp=comp,
            lambda_x=la,
            gamma_x=ga,
            history=self.history_method,
            mu_x=mu,
            exclusive=method == "exclusive",
            n_iter=nit,
            random_state=self.random_state,
        )
        self.loadings_ = r.W
        self.scores_ = r.S
        self.rss_ = r.rss
        self.aic_ = r.aic
        self.bic_ = r.bic
        self.ebic_ = r.ebic
        self.overlap_all_ = r.overlap_all
        self.overlap_selected_ = r.overlap_selected
        self.used_feature_indices_ = r.used
        self.diagnostics_ = r.diagnostics
        self.n_components_ = r.W.shape[1]
        self.n_iter_ = r.diagnostics["iterations_per_component"]
        self.structure_method_ = method
        self.analysis_ = "s4pca"
        self.input_structure_ = "single-matrix"
        self.engine_ = "reference-compatible"
        self.selected_mask_ = np.abs(r.W) > 1e-8
        idx = np.flatnonzero(np.any(self.selected_mask_, axis=1))
        self.selected_features_ = (
            self.feature_names_in_[idx] if hasattr(self, "feature_names_in_") else idx
        )
        self.reconstruction_loadings_ = project(Xp, r.S)
        self.fitted_reconstruction_scaled_ = r.S @ self.reconstruction_loadings_
        self.fitted_reconstruction_ = (
            self.fitted_reconstruction_scaled_ * self.x_scale_ + self.x_center_
        )
        self.reconstruction_error_ = float(np.mean((X - self.fitted_reconstruction_) ** 2))
        self.z_center_ = None
        self.z_scale_ = None
        return self

    def transform(self, X):
        check_is_fitted(self, "loadings_")
        X, _, _ = validate_xy(X, None)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must contain the fitted number of features.")
        return ((X - self.x_center_) / self.x_scale_) @ self.loadings_

    def inverse_transform(self, scores, *, scaled=False):
        check_is_fitted(self, "loadings_")
        scores = np.asarray(scores, dtype=float)
        X = scores @ self.reconstruction_loadings_
        return X if scaled else X * self.x_scale_ + self.x_center_

    def reconstruct(self, X=None, *, scaled=False):
        return self.inverse_transform(
            self.scores_ if X is None else self.transform(X), scaled=scaled
        )

    def get_summary(self):
        check_is_fitted(self, "loadings_")
        d = self.diagnostics_
        return {
            "estimator": {
                "analysis": self.analysis_,
                "engine": self.engine_,
                "input_structure": self.input_structure_,
                "structure_method": self.structure_method_,
                "history_method": self.history_method,
                "requested_components": int(d["requested_components"]),
                "extracted_components": int(d["extracted_components"]),
                "iterations_per_component": np.asarray(d["iterations_per_component"])
                .astype(int)
                .tolist(),
                "scale": bool(self.scale),
                "mu_x": float(self.mu_x),
                "lambda_x": float(self.lambda_x),
                "gamma_x": float(self.gamma_x),
                "random_state": self.random_state,
            },
            "fit": {
                "reconstruction_error": self.reconstruction_error_,
                "reference_rss": self.rss_,
                "aic": self.aic_,
                "bic": self.bic_,
                "ebic": self.ebic_,
            },
            "selection": {
                "n_features": self.n_features_in_,
                "selected_variables": int(d["selected_variables"]),
                "repeated_variables": int(d["repeated_variables"]),
                "selected_per_component": np.asarray(d["selected_per_component"])
                .astype(int)
                .tolist(),
                "overlap_all": self.overlap_all_,
                "overlap_selected": self.overlap_selected_,
            },
            "numerical_stability": {
                "warning": bool(d["numerical_stability_warning"]),
                "sensitive_components": np.asarray(d["platform_sensitive_components"])
                .astype(int)
                .tolist(),
                "initial_loading_norm": np.asarray(
                    d["initial_loading_norm_per_component"]
                ).tolist(),
                "tolerance": np.asarray(d["initialization_tolerance_per_component"]).tolist(),
                "ratio": np.asarray(d["initial_loading_to_tolerance_ratio"]).tolist(),
                "near_zero": np.asarray(d["near_zero_component_initialization"]).tolist(),
                "rule": d["numerical_stability_rule"],
                "message": d["numerical_stability_message"],
            },
        }

    def summary(self, *, detailed=False, precision=6):
        if not isinstance(detailed, (bool, np.bool_)):
            raise ValueError("detailed must be True or False.")
        if (
            isinstance(precision, (bool, np.bool_))
            or not isinstance(precision, numbers.Integral)
            or precision < 1
        ):
            raise ValueError("precision must be a positive integer.")
        q = self.get_summary()
        e = q["estimator"]
        f = q["fit"]
        s = q["selection"]
        n = q["numerical_stability"]

        def L(a, b):
            return f"{a:<24}{b}"

        def F(x):
            return f"{x:.{precision}f}"

        out = [
            "S4PCA Summary",
            "=============",
            "",
            "Estimator",
            "---------",
            L("Analysis", e["analysis"]),
            L("Engine", e["engine"]),
            L("Input structure", e["input_structure"]),
            L("Structure method", e["structure_method"]),
            L("Requested components", e["requested_components"]),
            L("Extracted components", e["extracted_components"]),
            "",
            "Model fit",
            "---------",
            L("Reconstruction MSE", F(f["reconstruction_error"])),
            L("Reference RSS", F(f["reference_rss"])),
            L("AIC", F(f["aic"])),
            L("BIC", F(f["bic"])),
            L("EBIC", F(f["ebic"])),
            "",
            "Variable selection",
            "------------------",
            L("Selected variables", f"{s['selected_variables']} / {s['n_features']}"),
            L("Repeated variables", s["repeated_variables"]),
            L("Overlap, all", F(s["overlap_all"])),
            L("Overlap, selected", F(s["overlap_selected"])),
            "",
            "Numerical stability",
            "-------------------",
            L(
                "Status",
                "WARNING" if n["warning"] else "No near-zero component initialization detected.",
            ),
        ]
        if n["warning"]:
            out += [
                L("Sensitive components", ", ".join(map(str, n["sensitive_components"]))),
                L("Message", n["message"]),
            ]
        if detailed:
            out += [
                "",
                "Component diagnostics",
                "---------------------",
                "Component  Initial norm   Tolerance      Ratio          Near zero",
            ]
            for i, v in enumerate(
                zip(n["initial_loading_norm"], n["tolerance"], n["ratio"], n["near_zero"]), 1
            ):
                out.append(
                    f"{i:<11}{v[0]:<15.{precision}e}{v[1]:<15.{precision}e}{v[2]:<15.{precision}e}{'Yes' if v[3] else 'No'}"
                )
            out += ["", "Diagnostic rule", "---------------", n["rule"]]
            if n["warning"]:
                out += [
                    "",
                    "Interpretation",
                    "--------------",
                    (
                        "Near-zero initial loading candidates were detected. "
                        "Normalization may amplify platform-dependent floating-point "
                        "differences in later sparse components. The fitted result "
                        "remains available, but cross-platform numerical identity "
                        "may not hold for the flagged components."
                    ),
                ]
        return "\n".join(out)
