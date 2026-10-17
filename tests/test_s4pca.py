import numpy as np
import pytest
from msma import S4PCA


def make_data(seed=7, n=40, p=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    z = 0.7 * X[:, 0] - 0.4 * X[:, 4] + rng.normal(size=n)
    return X, z


def test_fit_shapes_and_diagnostics():
    X, z = make_data()
    model = S4PCA(n_components=3, lambda_x=.1, mu_x=.8, structure_method="soft", gamma_x=.1, max_iter=20, random_state=4).fit(X, z=z)
    assert model.loadings_.shape[0] == 12
    assert 1 <= model.loadings_.shape[1] <= 3
    assert model.scores_.shape == (40, model.loadings_.shape[1])
    assert model.diagnostics_["engine"] == "reference-compatible"
    assert model.diagnostics_["convergence_rule"] == "fixed_iterations"
    selected = np.abs(model.loadings_) > 1e-8
    np.testing.assert_array_equal(model.diagnostics_["selected_per_component"], selected.sum(axis=0))


def test_reconstruction_shapes_and_scales():
    X, z = make_data()
    X = X * np.arange(1, 13) + np.arange(12)
    model = S4PCA(n_components=2, lambda_x=.1, mu_x=.8, structure_method="soft", gamma_x=.1, scale=True, random_state=4).fit(X, z=z)
    assert model.reconstruct().shape == X.shape
    assert model.reconstruct(X[:7]).shape == (7, 12)
    assert not np.allclose(model.reconstruct(), model.reconstruct(scaled=True))


def test_exclusive_prevents_reuse():
    X, z = make_data()
    model = S4PCA(n_components=3, lambda_x=.05, mu_x=.8, structure_method="exclusive", random_state=4).fit(X, z=z)
    reuse = (np.abs(model.loadings_) > 1e-8).sum(axis=1)
    assert np.all(reuse <= 1)


def test_reproducible_with_python_random_state():
    X, z = make_data()
    kwargs = dict(n_components=3, lambda_x=.1, mu_x=.8, structure_method="soft", gamma_x=.1, random_state=9)
    a = S4PCA(**kwargs).fit(X, z=z)
    b = S4PCA(**kwargs).fit(X, z=z)
    assert np.array_equal(a.loadings_, b.loadings_)
    assert np.array_equal(a.scores_, b.scores_)


def test_invalid_inputs():
    X, z = make_data()
    with pytest.raises(ValueError):
        S4PCA(n_components=0).fit(X, z=z)
    bad = X.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        S4PCA().fit(bad, z=z)
    with pytest.raises(ValueError):
        S4PCA().fit(X, z=np.c_[z, z])


def test_near_zero_initialization_diagnostics_detect_reference_instability():
    X, _ = make_data(seed=23, n=50, p=16)
    model = S4PCA(
        n_components=3,
        lambda_x=0.20,
        mu_x=0.0,
        structure_method="soft",
        gamma_x=0.10,
        max_iter=20,
        random_state=3,
    ).fit(X)

    d = model.diagnostics_
    n_extracted = model.n_components_
    assert d["initial_loading_norm_per_component"].shape == (n_extracted,)
    assert d["initial_direction_norm_per_component"].shape == (n_extracted,)
    assert d["initialization_tolerance_per_component"].shape == (n_extracted,)
    assert d["initial_loading_to_tolerance_ratio"].shape == (n_extracted,)
    assert d["near_zero_component_initialization"].shape == (n_extracted,)
    assert not d["near_zero_component_initialization"][0]
    if n_extracted > 1:
        assert d["near_zero_component_initialization"][1]
        assert 2 in d["platform_sensitive_components"]
        assert d["numerical_stability_warning"]
        assert "Near-zero initial loading candidate" in d["numerical_stability_message"]


def test_supervision_avoids_near_zero_second_component_initialization():
    X, z = make_data(seed=29, n=50, p=16)
    model = S4PCA(
        n_components=3,
        lambda_x=0.10,
        mu_x=0.8,
        structure_method="soft",
        gamma_x=0.10,
        max_iter=20,
        random_state=3,
    ).fit(X, z=z)

    d = model.diagnostics_
    if model.n_components_ > 1:
        assert not d["near_zero_component_initialization"][1]


def test_get_summary_is_json_compatible():
    import json

    X, z = make_data(seed=31)
    model = S4PCA(
        n_components=2,
        lambda_x=0.1,
        mu_x=0.8,
        structure_method="soft",
        gamma_x=0.1,
        random_state=3,
    ).fit(X, z=z)
    result = model.get_summary()
    assert result["estimator"]["engine"] == "reference-compatible"
    assert isinstance(result["selection"]["selected_per_component"], list)
    assert isinstance(result["numerical_stability"]["warning"], bool)
    json.dumps(result)


def test_summary_contains_core_sections():
    X, z = make_data(seed=32)
    model = S4PCA(n_components=2, mu_x=0.8, random_state=3).fit(X, z=z)
    text = model.summary()
    for heading in (
        "S4PCA Summary",
        "Estimator",
        "Model fit",
        "Variable selection",
        "Numerical stability",
    ):
        assert heading in text


def test_summary_reports_near_zero_initialization():
    X, _ = make_data(seed=33, n=50, p=16)
    model = S4PCA(
        n_components=3,
        lambda_x=0.2,
        mu_x=0.0,
        structure_method="soft",
        gamma_x=0.1,
        random_state=3,
    ).fit(X)
    text = model.summary(detailed=True)
    assert "WARNING" in text
    assert "Sensitive components" in text
    assert "Component diagnostics" in text
    assert "Interpretation" in text


def test_summary_rejects_unfitted_estimator():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        S4PCA().summary()


@pytest.mark.parametrize("precision", [0, -1, 1.5, True])
def test_summary_rejects_invalid_precision(precision):
    X, z = make_data(seed=34)
    model = S4PCA(n_components=1, mu_x=0.8, random_state=3).fit(X, z=z)
    with pytest.raises(ValueError, match="precision"):
        model.summary(precision=precision)


def test_summary_rejects_invalid_detailed():
    X, z = make_data(seed=35)
    model = S4PCA(n_components=1, mu_x=0.8, random_state=3).fit(X, z=z)
    with pytest.raises(ValueError, match="detailed"):
        model.summary(detailed="yes")
