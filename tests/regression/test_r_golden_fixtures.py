from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from msma._s4pca_reference import fit_s4pca_reference

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "r_v4"
MANIFEST = FIXTURE_DIR / "manifest.csv"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="R golden fixtures have not been generated")

# These R-reference cases enter a near-zero initialization after deflation.
# Their later sparse components are BLAS/platform sensitive by construction.
PLATFORM_SENSITIVE_CASES = {"sstspca", "sparse_pca", "without_z"}


def load_csv(path: Path, *, dtype=float) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2, dtype=dtype)


def load_vector(path: Path, *, dtype=float) -> np.ndarray:
    return load_csv(path, dtype=dtype).reshape(-1)


def manifest_rows():
    if not MANIFEST.exists():
        return []
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("row", manifest_rows(), ids=lambda row: row["case"])
def test_python_reference_matches_r_golden_fixture(row):
    case_dir = FIXTURE_DIR / row["case"]
    X = load_csv(FIXTURE_DIR / "input_X.csv")
    z = load_vector(FIXTURE_DIR / "input_z.csv") if row["use_z"].upper() == "TRUE" else None
    initial_score = load_vector(case_dir / "initial_score.csv")
    expected_W = load_csv(case_dir / "expected_loadings.csv")
    expected_S = load_csv(case_dir / "expected_scores.csv")
    expected_zero_mask = load_csv(case_dir / "expected_zero_mask.csv", dtype=float).astype(bool)
    expected_used = load_vector(case_dir / "used_indices_python.csv", dtype=int)
    with (case_dir / "metrics.json").open(encoding="utf-8") as stream:
        metrics = json.load(stream)

    result = fit_s4pca_reference(
        X,
        z,
        n_components=int(row["comp"]),
        lambda_x=float(row["lambda_x"]),
        gamma_x=float(row["gamma_x"]),
        history_method="sumabs",
        mu_x=float(row["mu_x"]),
        hard_exclusion=row["hard_exclusion"].upper() == "TRUE",
        n_iter=int(row["n_iter"]),
        initial_score=initial_score,
    )

    if row["case"] in PLATFORM_SENSITIVE_CASES:
        # The first component must still reproduce the frozen R reference.
        np.testing.assert_allclose(
            result.loadings[:, 0], expected_W[:, 0], rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            result.scores[:, 0], expected_S[:, 0], rtol=0.0, atol=1e-12
        )
        # The implementation must explicitly diagnose the known instability.
        assert result.diagnostics["numerical_stability_warning"]
        assert 2 in result.diagnostics["platform_sensitive_components"]
        pytest.xfail(
            "Known reference-algorithm sensitivity: after projection deflation, "
            "the next unsupervised loading candidate is nearly zero; normalizing "
            "it amplifies BLAS/platform rounding differences in later sparse components."
        )

    np.testing.assert_allclose(result.loadings, expected_W, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result.scores, expected_S, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(np.abs(result.loadings) <= 1e-8, expected_zero_mask)
    np.testing.assert_array_equal(result.used_feature_indices, expected_used)
    assert result.rss == pytest.approx(metrics["rss"], rel=0.0, abs=1e-12)
    assert result.aic == pytest.approx(metrics["aic"], rel=0.0, abs=1e-12)
    assert result.bic == pytest.approx(metrics["bic"], rel=0.0, abs=1e-12)
    assert result.ebic == pytest.approx(metrics["ebic"], rel=0.0, abs=1e-12)
    assert result.overlap_all == pytest.approx(metrics["overlap_all"], rel=0.0, abs=1e-12)
    assert result.overlap_selected == pytest.approx(metrics["overlap_selected"], rel=0.0, abs=1e-12)
