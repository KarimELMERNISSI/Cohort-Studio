"""
Unit Tests for Clustering and Dimensionality Reduction Utilities (utils/clustering_utils.py).

Tests data standardization, PCA, t-SNE, K-Means, DBSCAN, Gaussian Mixture Models,
and optimal K elbow/silhouette metrics.
"""

import numpy as np
import pandas as pd
import pytest

from utils.clustering_utils import (
    fit_dbscan,
    fit_gaussian_mixture,
    fit_kmeans,
    optimal_k_analysis,
    prepare_data_for_clustering,
    run_pca,
    run_tsne,
    standardize_data,
)


@pytest.fixture
def synthetic_clustering_df() -> pd.DataFrame:
    """Create a synthetic dataset with distinct numerical clusters."""
    np.random.seed(42)
    # Cluster 1: centered around (10, 10)
    c1 = np.random.normal(loc=10, scale=1.0, size=(25, 3))
    # Cluster 2: centered around (50, 50)
    c2 = np.random.normal(loc=50, scale=1.0, size=(25, 3))
    # Cluster 3: centered around (100, 100)
    c3 = np.random.normal(loc=100, scale=1.0, size=(25, 3))

    data = np.vstack([c1, c2, c3])
    df = pd.DataFrame(data, columns=["biomarker_a", "biomarker_b", "biomarker_c"])
    return df


class TestClusteringUtils:
    """Test suite for utils/clustering_utils.py."""

    def test_standardize_data(self, synthetic_clustering_df: pd.DataFrame) -> None:
        """Standardizes numeric columns to zero mean and unit variance."""
        std_df = standardize_data(synthetic_clustering_df)
        for col in std_df.columns:
            assert std_df[col].mean() == pytest.approx(0.0, abs=1e-6)
            assert std_df[col].std() == pytest.approx(1.0, abs=1e-6)

    def test_prepare_data_for_clustering_missing_drop(self) -> None:
        """Drops missing values and standardizes numeric columns."""
        df = pd.DataFrame({
            "age": [20.0, 30.0, np.nan, 50.0],
            "bmi": [22.0, 25.0, 28.0, 31.0],
            "gender": ["M", "F", "F", "M"],
        })
        prep_df, valid_idx = prepare_data_for_clustering(
            df, numeric_cols=["age", "bmi"], categorical_cols=["gender"], handle_missing="drop"
        )
        assert len(prep_df) == 3
        assert len(valid_idx) == 3
        assert 2 not in valid_idx

    def test_prepare_data_for_clustering_missing_mean(self) -> None:
        """Fills missing numeric with mean and categorical with mode."""
        df = pd.DataFrame({
            "age": [20.0, 30.0, np.nan, 40.0],
            "bmi": [22.0, 25.0, 28.0, 31.0],
            "category": ["A", "A", None, "B"],
        })
        prep_df, valid_idx = prepare_data_for_clustering(
            df, numeric_cols=["age", "bmi"], categorical_cols=["category"], handle_missing="mean"
        )
        assert len(prep_df) == 4
        assert not prep_df["age"].isna().any()

    def test_run_pca(self, synthetic_clustering_df: pd.DataFrame) -> None:
        """PCA reduces dimensions and computes variance explained."""
        embeddings, info = run_pca(synthetic_clustering_df, n_components=2)
        assert embeddings.shape == (75, 2)
        assert "explained_variance_ratio" in info
        assert "total_variance_explained" in info
        assert info["total_variance_explained"] > 0.8
        assert len(info["feature_names"]) == 3

    def test_run_tsne(self, synthetic_clustering_df: pd.DataFrame) -> None:
        """t-SNE produces 2D non-linear embeddings."""
        embeddings, info = run_tsne(synthetic_clustering_df, n_components=2, perplexity=10)
        assert embeddings.shape == (75, 2)
        assert "perplexity_used" in info
        assert info["n_samples"] == 75

    def test_fit_kmeans(self, synthetic_clustering_df: pd.DataFrame) -> None:
        """K-Means identifies 3 clusters with valid inertia."""
        embeddings = synthetic_clustering_df.values
        labels, info = fit_kmeans(embeddings, n_clusters=3)
        assert len(labels) == 75
        assert set(labels) == {0, 1, 2}
        assert info["n_clusters"] == 3
        assert info["inertia"] > 0
        assert info["centers"].shape == (3, 3)

    def test_fit_dbscan(self, synthetic_clustering_df: pd.DataFrame) -> None:
        """DBSCAN identifies density-based clusters."""
        embeddings = synthetic_clustering_df.values
        labels, info = fit_dbscan(embeddings, eps=5.0, min_samples=3)
        assert len(labels) == 75
        assert "n_clusters" in info
        assert "noise_ratio" in info
        assert info["n_clusters"] >= 2

    def test_fit_gaussian_mixture(self, synthetic_clustering_df: pd.DataFrame) -> None:
        """Gaussian Mixture Model returns component labels and probabilities."""
        embeddings = synthetic_clustering_df.values
        labels, info = fit_gaussian_mixture(embeddings, n_components=3)
        assert len(labels) == 75
        assert set(labels) == {0, 1, 2}
        assert "bic" in info
        assert "aic" in info
        assert info["probabilities"].shape == (75, 3)
        # Probability rows sum to 1.0
        assert np.allclose(info["probabilities"].sum(axis=1), 1.0)

    def test_optimal_k_analysis(self, synthetic_clustering_df: pd.DataFrame) -> None:
        """Calculates elbow inertias and silhouette scores across k range."""
        embeddings = synthetic_clustering_df.values
        res = optimal_k_analysis(embeddings, k_range=range(2, 5))
        assert "inertias" in res
        assert "silhouette_scores" in res
        assert "optimal_k_silhouette" in res
        assert len(res["inertias"]) == 3
        assert len(res["silhouette_scores"]) == 3
        # Best K for this 3-cluster dataset should be 3
        assert res["optimal_k_silhouette"] == 3
