"""
Clustering and Dimensionality Reduction Utilities

This module provides functions for:
- Dimensionality reduction (PCA, t-SNE, UMAP, FAMD)
- Clustering (K-Means, DBSCAN, Gaussian Mixture)
- Optimal cluster analysis (Elbow, Silhouette)

All heavy imports are done lazily to avoid slowing app startup.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List


def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize numeric columns to mean=0, std=1."""
    return (df - df.mean()) / df.std()


def prepare_data_for_clustering(
    df: pd.DataFrame, 
    numeric_cols: List[str], 
    categorical_cols: Optional[List[str]] = None,
    handle_missing: str = "drop"
) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Prepare data for clustering: handle missing values, standardize.
    
    Returns:
        (prepared_df, valid_indices)
    """
    cols = numeric_cols.copy()
    if categorical_cols:
        cols.extend(categorical_cols)
    
    data = df[cols].copy()
    
    if handle_missing == "drop":
        data = data.dropna()
    elif handle_missing == "mean":
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())
        for col in categorical_cols or []:
            data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else "Unknown")
    
    valid_indices = data.index
    
    # Standardize numeric columns only
    for col in numeric_cols:
        if col in data.columns:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
    
    return data, valid_indices


# ============ DIMENSIONALITY REDUCTION ============

def run_pca(data: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run PCA on the data.
    
    Returns:
        (embeddings, info_dict with explained_variance)
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=n_components, random_state=42)
    embeddings = pca.fit_transform(data.select_dtypes(include=[np.number]))
    
    info = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_variance_explained": sum(pca.explained_variance_ratio_),
        "components": pca.components_,
        "feature_names": data.select_dtypes(include=[np.number]).columns.tolist()
    }
    
    return embeddings, info


def run_tsne(
    data: pd.DataFrame, 
    n_components: int = 2, 
    perplexity: float = 30.0,
    max_iter: int = 1000,
    max_samples: int = 5000
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run t-SNE on the data.
    Note: t-SNE is slow for large datasets, so we sample if needed.
    
    Returns:
        (embeddings, info_dict)
    """
    from sklearn.manifold import TSNE
    
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Sample if too large
    sampled = False
    if len(numeric_data) > max_samples:
        numeric_data = numeric_data.sample(max_samples, random_state=42)
        sampled = True
    
    # Adjust perplexity if needed
    effective_perplexity = min(perplexity, (len(numeric_data) - 1) / 3)
    
    tsne = TSNE(
        n_components=n_components, 
        perplexity=effective_perplexity,
        max_iter=max_iter,  # Changed from n_iter for newer sklearn
        random_state=42,
        init='pca'
    )
    embeddings = tsne.fit_transform(numeric_data)
    
    info = {
        "perplexity_used": effective_perplexity,
        "n_samples": len(numeric_data),
        "sampled": sampled,
        "kl_divergence": tsne.kl_divergence_ if hasattr(tsne, 'kl_divergence_') else None
    }
    
    return embeddings, info


def run_umap(
    data: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    max_samples: int = 10000
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run UMAP on the data.
    
    Returns:
        (embeddings, info_dict)
    """
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP not installed. Run: pip install umap-learn")
    
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Sample if too large
    sampled = False
    if len(numeric_data) > max_samples:
        numeric_data = numeric_data.sample(max_samples, random_state=42)
        sampled = True
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, len(numeric_data) - 1),
        min_dist=min_dist,
        random_state=42
    )
    embeddings = reducer.fit_transform(numeric_data)
    
    info = {
        "n_neighbors_used": min(n_neighbors, len(numeric_data) - 1),
        "min_dist": min_dist,
        "n_samples": len(numeric_data),
        "sampled": sampled
    }
    
    return embeddings, info


def run_famd(
    data: pd.DataFrame,
    n_components: int = 2
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run FAMD (Factor Analysis of Mixed Data) for mixed numeric/categorical data.
    
    Returns:
        (embeddings, info_dict)
    """
    try:
        import prince
    except ImportError:
        raise ImportError("Prince not installed. Run: pip install prince")
    
    famd = prince.FAMD(n_components=n_components, random_state=42)
    famd = famd.fit(data)
    embeddings = famd.row_coordinates(data).values
    
    info = {
        "explained_inertia": famd.explained_inertia_.tolist() if hasattr(famd, 'explained_inertia_') else None,
        "n_samples": len(data)
    }
    
    return embeddings, info


# ============ CLUSTERING ============

def fit_kmeans(embeddings: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit K-Means clustering.
    
    Returns:
        (cluster_labels, info_dict)
    """
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    info = {
        "n_clusters": n_clusters,
        "inertia": kmeans.inertia_,
        "centers": kmeans.cluster_centers_,
        "n_iter": kmeans.n_iter_
    }
    
    return labels, info


def fit_dbscan(embeddings: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit DBSCAN clustering.
    
    Returns:
        (cluster_labels, info_dict) - Note: -1 means noise
    """
    from sklearn.cluster import DBSCAN
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    info = {
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": n_noise / len(labels) if len(labels) > 0 else 0
    }
    
    return labels, info


def fit_gaussian_mixture(embeddings: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit Gaussian Mixture Model.
    
    Returns:
        (cluster_labels, info_dict with probabilities)
    """
    from sklearn.mixture import GaussianMixture
    
    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=3)
    labels = gmm.fit_predict(embeddings)
    probabilities = gmm.predict_proba(embeddings)
    
    info = {
        "n_components": n_components,
        "bic": gmm.bic(embeddings),
        "aic": gmm.aic(embeddings),
        "converged": gmm.converged_,
        "probabilities": probabilities
    }
    
    return labels, info


# ============ OPTIMAL K ANALYSIS ============

def optimal_k_analysis(
    embeddings: np.ndarray, 
    k_range: range = range(2, 11)
) -> Dict[str, Any]:
    """
    Compute elbow and silhouette metrics for different K values.
    
    Returns:
        dict with inertias, silhouette_scores, optimal_k_elbow, optimal_k_silhouette
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        if k >= len(embeddings):
            break
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        inertias.append(kmeans.inertia_)
        
        if k > 1 and k < len(embeddings):
            sil = silhouette_score(embeddings, labels)
            silhouette_scores.append(sil)
        else:
            silhouette_scores.append(0)
    
    # Find optimal K (highest silhouette)
    optimal_k_sil = list(k_range)[silhouette_scores.index(max(silhouette_scores))] if silhouette_scores else 2
    
    return {
        "k_values": list(k_range)[:len(inertias)],
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "optimal_k_silhouette": optimal_k_sil
    }


def compute_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute mean/mode of variables per cluster.
    
    Returns:
        DataFrame with cluster statistics
    """
    df_with_cluster = df.copy()
    df_with_cluster['Cluster'] = labels
    
    # Numeric: compute mean
    profiles = df_with_cluster.groupby('Cluster')[numeric_cols].mean()
    profiles['n'] = df_with_cluster.groupby('Cluster').size()
    
    # Categorical: compute mode (most frequent)
    if categorical_cols:
        for col in categorical_cols:
            modes = df_with_cluster.groupby('Cluster')[col].apply(lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
            profiles[f"{col}_mode"] = modes
    
    return profiles.round(3)
