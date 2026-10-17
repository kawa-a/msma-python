from __future__ import annotations

import numbers

import numpy as np


def positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def finite_scalar(value, name, *, lower=None, upper=None):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if lower is not None and value < lower:
        raise ValueError(f"{name} must be at least {lower}.")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must be at most {upper}.")
    return value


def validate_xy(X, z=None):
    names = np.asarray([str(v) for v in X.columns], dtype=object) if hasattr(X, "columns") else None
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must be a two-dimensional numeric array with at least two rows.")
    if not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X):
        raise TypeError("X must be a real numeric array.")
    X = np.asarray(X, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError("X must not contain NaN or infinite values.")
    if z is None:
        return X, None, names
    z = np.asarray(z)
    if z.ndim == 2 and z.shape[1] == 1:
        z = z[:, 0]
    if z.ndim != 1:
        raise ValueError("Z must be one-dimensional or a one-column array.")
    if len(z) != X.shape[0]:
        raise ValueError("len(Z) must equal the number of rows in X.")
    if not np.issubdtype(z.dtype, np.number) or np.iscomplexobj(z):
        raise TypeError("Z must be real numeric.")
    z = np.asarray(z, dtype=np.float64)
    if not np.all(np.isfinite(z)):
        raise ValueError("Z must not contain NaN or infinite values.")
    return X, z, names
