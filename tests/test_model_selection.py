import json

import numpy as np
import pytest

from msma import S4PCA
from msma.model_selection import select_msma


def make_data(seed=7, n=36, p=20):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    z = 0.7 * X[:, 0] - 0.4 * X[:, 5] + rng.normal(size=n)
    return X, z


def choose(**overrides):
    X, z = make_data()
    kwargs = dict(
        estimator=S4PCA(mu_x=0.8, max_iter=10, scale=False),
        X=X,
        z=z,
        param_grid={"lambda_x": [0.05, 0.1], "gamma_x": [0, 0.1], "n_components": [2, 3]},
        alpha=0.1,
        n_repeats=2,
        calibration_fraction=0.25,
        random_state=9,
        refit=True,
    )
    kwargs.update(overrides)
    return select_msma(**kwargs)


def test_selection_grid_and_refit():
    r = choose()
    assert len(r.candidate_results_) == 8
    assert len(r.split_results_) == 16
    assert isinstance(r.best_estimator_, S4PCA)
    assert r.best_params_["n_components"] in [2, 3]


def test_selection_reproducibility():
    a = choose()
    b = choose()
    assert a.best_params_ == b.best_params_
    assert a.candidate_results_ == b.candidate_results_
    assert a.split_results_ == b.split_results_


def test_repeat_identifiers_and_no_refit():
    r = choose(refit=False)
    assert r.best_estimator_ is None
    assert sorted({x["repeat"] for x in r.split_results_}) == [1, 2]


def test_summary_and_json_compatible_results():
    r = choose()
    assert "MSMA Model Selection Summary" in r.summary()
    json.dumps(r.candidate_results_)
    json.dumps(r.split_results_)


@pytest.mark.parametrize(
    "name,value",
    [
        ("alpha", 0),
        ("alpha", 1),
        ("n_repeats", 0),
        ("calibration_fraction", 0),
        ("calibration_fraction", 1),
    ],
)
def test_invalid_controls(name, value):
    with pytest.raises(ValueError):
        choose(**{name: value})


def test_invalid_overlap():
    with pytest.raises(ValueError):
        choose(constraints={"max_overlap": 2})
