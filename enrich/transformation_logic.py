"""
New variables derived from existing ones: statistics, scaling, dimensionality reduction, clustering and encoding.

``apply_variable_transformation`` is the only implementation: the Enrich page calls it, and the replay of a trace
calls it again with the parameters the page recorded, so both produce the same columns with the same names.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from core.expressions import safe_format

RANDOM_STATE = 42  # every stochastic method is seeded, so that a replay reproduces the recorded columns

TRANSFORMATIONS = {
    "Mean": lambda x: x.mean(),
    "Median": lambda x: x.median(),
    "Summation": lambda x: x.sum(),
    "Minimum Value": lambda x: x.min(),
    "Maximum Value": lambda x: x.max(),
    "Standard Deviation": lambda x: x.std(),
    "Natural Logarithm Transformation": lambda x: np.log(x + 1),
    "Min-Max Normalization": lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten(),
    "Z-Score Standardization": lambda x: (x - x.mean()) / x.std(),
}

REDUCTION = ("Dimensionality Reduction", "Dimensional Reduction-Based")
CLUSTERING = ("Clustering", "Cluster-Based")


def _unique_names(new_columns: dict, existing) -> dict:
    """A name already in the table, or already given to another new column, gets the first free suffix _1, _2..."""
    taken = set(existing)
    renamed = {}
    for name, values in new_columns.items():
        candidate, counter = name, 1
        while candidate in taken:
            candidate, counter = f"{name}_{counter}", counter + 1
        taken.add(candidate)
        renamed[candidate] = values
    return renamed


def apply_variable_transformation(dataframe, params):
    """
    The new columns of one transformation, on ``dataframe``'s index (None when the transformation makes none).

    Raises ValueError when the input cannot be transformed: missing values for a dimensionality reduction or a
    clustering (they need complete data), fewer than two columns to reduce, fewer than two rows to cluster.
    """
    transformation_type = params.get("transformation_type")
    transformation = params.get("transformation")
    columns = params.get("columns")
    naming_pattern = params.get("naming_pattern")
    n_components = params.get("n_components", 3)

    # t-SNE params
    perplexity = params.get("perplexity", 30)
    learning_rate = params.get("learning_rate", 200)

    # UMAP params
    n_neighbors = params.get("n_neighbors", 15)
    min_dist = params.get("min_dist", 0.1)

    # Ordinal params
    category_orders = params.get("category_orders", {})

    if transformation_type in REDUCTION + CLUSTERING and dataframe[columns].isnull().any().any():
        raise ValueError(
            "The input columns contain missing values: dimensionality reduction and clustering need complete "
            "data. Impute the missing values first (Handle missing data), or choose complete columns."
        )

    new_columns = {}

    if transformation_type in REDUCTION:
        if len(columns) < 2:
            raise ValueError("Choose at least two columns to reduce.")
        if transformation == "PCA":
            import prince

            model = prince.PCA(
                n_components=n_components,
                rescale_with_mean=True,
                rescale_with_std=True,
                copy=True,
                check_input=True,
                random_state=RANDOM_STATE,
                engine="sklearn",
            )
            transformed_data = model.fit_transform(dataframe[columns])

        elif transformation == "t-SNE":
            from sklearn.manifold import TSNE

            model = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                random_state=RANDOM_STATE,
            )
            transformed_data = model.fit_transform(StandardScaler().fit_transform(dataframe[columns]))
        elif transformation == "UMAP":
            import umap

            model = umap.UMAP(
                n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=RANDOM_STATE
            )
            transformed_data = model.fit_transform(StandardScaler().fit_transform(dataframe[columns]))
        else:  # FAMD
            import prince

            model = prince.FAMD(n_components=n_components, random_state=RANDOM_STATE)
            transformed_data = model.fit_transform(dataframe[columns])

        component_names = [
            safe_format(naming_pattern, method_applied=transformation.replace(" ", "_")) + f"_{i + 1}"
            for i in range(n_components)
        ]
        if transformation in ["t-SNE", "UMAP"]:
            result_df = pd.DataFrame(transformed_data, columns=component_names, index=dataframe.index)
        else:
            result_df = transformed_data.copy()
            result_df.columns = component_names
        new_columns.update(result_df.to_dict(orient="list"))

    elif transformation_type in CLUSTERING:
        from utils.clustering_utils import (
            fit_dbscan,
            fit_gaussian_mixture,
            fit_kmeans,
            prepare_data_for_clustering,
        )

        n_clusters = params.get("n_clusters", 3)
        eps = params.get("eps", 0.5)
        min_samples = params.get("min_samples", 5)

        # standardised by prepare_data_for_clustering
        numeric_cols = dataframe[columns].select_dtypes(include=np.number).columns.tolist()
        prep_data, valid_idx = prepare_data_for_clustering(dataframe[columns], numeric_cols)
        if len(prep_data) < 2:
            raise ValueError("Not enough valid data points for clustering.")
        if transformation == "K-Means":
            labels, _ = fit_kmeans(prep_data, n_clusters)
        elif transformation == "DBSCAN":
            labels, _ = fit_dbscan(prep_data, eps, min_samples)
        elif transformation == "Gaussian Mixture":
            labels, _ = fit_gaussian_mixture(prep_data, n_clusters)
        else:
            labels = []
        if len(labels) > 0:
            full_labels = pd.Series(index=dataframe.index, data=np.nan)
            full_labels.loc[valid_idx] = labels
            col_name = safe_format(naming_pattern, method_applied=transformation.replace(" ", "_"))
            new_columns[col_name] = full_labels.values

    elif transformation_type == "Encoding":
        if transformation == "One-Hot Encoding":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            for col in columns:
                encoded_data = encoder.fit_transform(dataframe[[col]])
                categories = encoder.categories_[0]
                for i, category in enumerate(categories):
                    new_col_name = safe_format(naming_pattern, initial_variable=col, category=category)
                    new_columns[new_col_name] = encoded_data[:, i]

        elif transformation == "Label Encoding":
            encoder = LabelEncoder()
            for col in columns:
                new_col_name = safe_format(naming_pattern, initial_variable=col, category="label_encoded")
                new_columns[new_col_name] = encoder.fit_transform(dataframe[col])

        elif transformation == "Ordinal Encoding":
            for col in columns:
                encoder = OrdinalEncoder(categories=[category_orders[col]])
                new_col_name = safe_format(naming_pattern, initial_variable=col, category="ordinal")
                new_columns[new_col_name] = encoder.fit_transform(dataframe[[col]]).flatten()

    else:
        # statistical and scaling transformations
        method_function = TRANSFORMATIONS.get(transformation, None)
        if method_function:
            for col in columns:
                new_col_name = safe_format(
                    naming_pattern, initial_variable=col, method_applied=transformation.replace(" ", "_")
                )
                result = method_function(dataframe[col])
                if np.isscalar(result):
                    result = [result] * len(dataframe)
                new_columns[new_col_name] = result

    if new_columns:
        return pd.DataFrame(_unique_names(new_columns, dataframe.columns), index=dataframe.index)
    return None
