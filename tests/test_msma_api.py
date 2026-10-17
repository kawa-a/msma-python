import numpy as np
import pytest

from msma import S4PCA, msma


def data():
    r = np.random.default_rng(2)
    X = r.normal(size=(40, 12))
    Z = 0.7 * X[:, 0] - 0.4 * X[:, 4] + r.normal(size=40)
    return X, Z


def test_explicit_s4pca_dispatch():
    X, Z = data()
    fit = msma(
        X,
        Z=Z,
        analysis="s4pca",
        comp=3,
        lambda_x=0.1,
        mu_x=0.8,
        structure_method="soft",
        gamma_x=0.1,
        n_iter=20,
        random_state=4,
    )
    assert isinstance(fit, S4PCA)
    assert fit.analysis_ == "s4pca"
    assert fit.input_structure_ == "single-matrix"
    assert fit.get_summary()["estimator"]["analysis"] == "s4pca"


def test_auto_structured_dispatch():
    X, Z = data()
    fit = msma(X, Z=Z, comp=2, lambda_x=0.1, mu_x=0.8, gamma_x=0.1, random_state=4)
    assert fit.structure_method_ == "soft"


def test_exclusive_alias():
    X, Z = data()
    fit = msma(X, Z=Z, analysis="exclusive_pca", comp=2, lambda_x=0.05, mu_x=0.8, random_state=4)
    assert fit.structure_method_ == "exclusive"


def test_future_paths_fail_explicitly():
    X, Z = data()
    with pytest.raises(NotImplementedError, match="supervised_pca"):
        msma(X, Z=Z)
    with pytest.raises(NotImplementedError, match="Multiblock"):
        msma([X, X])
    with pytest.raises(NotImplementedError, match="PLS"):
        msma(X, Y=Z)


def test_invalid_analysis():
    X, _ = data()
    with pytest.raises(ValueError, match="analysis"):
        msma(X, analysis="unknown")
