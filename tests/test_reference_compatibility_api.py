import numpy as np

from msma._s4pca_reference import (
    _norm_reference,
    _project_reference,
    _sparse_reference,
    fit_s4pca_reference,
)


def test_low_level_regression_api_is_preserved():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 8))
    z = X[:, 0] + rng.normal(size=30)
    result = fit_s4pca_reference(
        X,
        z,
        n_components=2,
        lambda_x=0.1,
        gamma_x=0.1,
        mu_x=0.8,
        n_iter=10,
        random_state=4,
    )
    assert result.loadings is result.W
    assert result.scores is result.S
    assert result.used_feature_indices is result.used
    assert _norm_reference(np.array([3.0, 4.0])).shape == (2,)
    assert _sparse_reference(np.array([0.5, -0.5]), 0.1).shape == (2,)
    assert _project_reference(X, result.S).shape == (2, X.shape[1])
