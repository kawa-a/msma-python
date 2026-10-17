from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Result:
    W: np.ndarray
    S: np.ndarray
    rss: float
    aic: float
    bic: float
    ebic: float
    overlap_all: float
    overlap_selected: float
    used: np.ndarray
    diagnostics: dict

    # Backward-compatible descriptive attributes used by regression fixtures.
    @property
    def loadings(self):
        return self.W

    @property
    def scores(self):
        return self.S

    @property
    def used_feature_indices(self):
        return self.used


def norm(a):
    a = np.asarray(a, dtype=float).reshape(-1)
    return a.copy() if np.all(a == 0) else a / np.sqrt(a @ a)


def sparse(x, lam):
    x = np.asarray(x, dtype=float)
    la = np.asarray(lam, dtype=float)
    if np.all(x == 0):
        la = np.asarray(0.0)
    elif la.ndim == 0 and float(la) >= np.max(np.abs(x)):
        d = np.diff(np.sort(np.unique(np.abs(x))))
        d = d[d > 1e-8]
        la = np.asarray(np.max(np.abs(x)) - (d[-1] / 2 if d.size else 1e-6))
    lv = np.full(x.shape, float(la)) if la.ndim == 0 else np.broadcast_to(la, x.shape)
    return np.sign(x) * np.maximum(np.abs(x) - lv, 0)


def project(X, S):
    C = S.T @ S
    i = 0
    while True:
        try:
            return np.linalg.solve(C + 1e-18 * 10**i * np.eye(C.shape[0]), S.T @ X)
        except np.linalg.LinAlgError:
            i += 1
            if i > 30:
                raise


def fit_reference(
    X,
    z=None,
    *,
    comp=6,
    lambda_x=0.02,
    gamma_x=0,
    history="sumabs",
    mu_x=1,
    exclusive=False,
    n_iter=30,
    random_state=None,
    initial_score=None,
):
    X = np.asarray(X, dtype=float)
    z = None if z is None else np.asarray(z, dtype=float)
    n, p = X.shape
    rng = np.random.default_rng(random_state)
    s = norm(rng.normal(size=n) if initial_score is None else initial_score)
    Xd = X.copy()
    scores = []
    loads = []
    used = []
    eps = float(np.finfo(float).eps)
    factor = 100.0
    init_norm = []
    dir_norm = []
    tol = []
    ratio = []
    near = []
    for _ in range(comp):
        for it in range(n_iter):
            direction = (1 - mu_x) * s + mu_x * (0.0 if z is None else z)
            w = Xd.T @ direction
            if it == 0:
                wn = float(np.linalg.norm(w))
                dn = float(np.linalg.norm(direction))
                rn = float(np.linalg.norm(Xd))
                th = factor * eps * rn * dn
                init_norm.append(wn)
                dir_norm.append(dn)
                tol.append(th)
                ratio.append(wn / th if th > 0 else float("inf"))
                near.append(wn <= th)
            if exclusive and used:
                w[np.asarray(used, dtype=int)] = 0
            if gamma_x > 0 and loads:
                P = np.column_stack(loads)
                h = (
                    np.sum(np.abs(P), axis=1)
                    if history == "sumabs"
                    else (
                        (np.sum(np.abs(P) > 0, axis=1) > 0).astype(float)
                        if history == "any"
                        else np.sum(np.abs(P) > 0, axis=1).astype(float)
                    )
                )
                threshold = lambda_x + gamma_x * h
            else:
                threshold = lambda_x
            w = norm(sparse(norm(w), threshold))
            s = Xd @ w
        scores.append(s.copy())
        loads.append(w.copy())
        for j in np.flatnonzero(w != 0):
            if int(j) not in used:
                used.append(int(j))
        if len(used) == p:
            break
        S = np.column_stack(scores)
        Xd = X - S @ project(X, S)
    S = np.column_stack(scores)
    W = np.column_stack(loads)
    rss = float(np.sum(Xd**2))
    df = int(np.sum(W != 0))
    aic = float(n * np.log(rss / n) + 2 * df)
    bic = float(n * np.log(rss / n) + np.log(n) * df)
    ebic = float(bic + 2 * np.log(p))
    cnt = np.sum(np.abs(W) > 1e-8, axis=1)
    oa = float(np.mean(cnt >= 2))
    os = float(np.sum(cnt >= 2) / np.sum(cnt >= 1)) if np.any(cnt >= 1) else 0.0
    sensitive = (np.flatnonzero(near) + 1).astype(int)
    warning = bool(sensitive.size)
    msg = (
        f"Near-zero initial loading candidate detected for component(s) {', '.join(map(str, sensitive))}. Normalization may amplify platform-dependent floating-point differences in later sparse components."
        if warning
        else None
    )
    diagnostics = {
        "engine": "reference-compatible",
        "iterations_per_component": np.full(W.shape[1], n_iter, dtype=int),
        "extracted_components": W.shape[1],
        "requested_components": comp,
        "terminated_early": W.shape[1] < comp,
        "selected_per_component": np.sum(np.abs(W) > 1e-8, axis=0),
        "selected_variables": int(np.sum(cnt >= 1)),
        "repeated_variables": int(np.sum(cnt >= 2)),
        "convergence_rule": "fixed_iterations",
        "reference_compatible": True,
        "initial_loading_norm_per_component": np.asarray(init_norm),
        "initial_direction_norm_per_component": np.asarray(dir_norm),
        "initialization_tolerance_per_component": np.asarray(tol),
        "initial_loading_to_tolerance_ratio": np.asarray(ratio),
        "near_zero_component_initialization": np.asarray(near, dtype=bool),
        "platform_sensitive_components": sensitive,
        "numerical_stability_warning": warning,
        "numerical_stability_message": msg,
        "numerical_stability_rule": "||X_d.T @ direction||_2 <= 100 * eps(float64) * ||X_d||_F * ||direction||_2",
    }
    return Result(W, S, rss, aic, bic, ebic, oa, os, np.asarray(used, dtype=int), diagnostics)


# Stable low-level compatibility API. These aliases preserve the names used by
# the first golden-regression suite while the short internal names remain local.
_norm_reference = norm
_sparse_reference = sparse
_project_reference = project


def fit_s4pca_reference(
    X,
    z=None,
    *,
    n_components=6,
    lambda_x=0.02,
    gamma_x=0.0,
    history_method="sumabs",
    mu_x=1.0,
    hard_exclusion=False,
    n_iter=30,
    random_state=None,
    initial_score=None,
):
    return fit_reference(
        X,
        z,
        comp=n_components,
        lambda_x=lambda_x,
        gamma_x=gamma_x,
        history=history_method,
        mu_x=mu_x,
        exclusive=hard_exclusion,
        n_iter=n_iter,
        random_state=random_state,
        initial_score=initial_score,
    )
