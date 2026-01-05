import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.manifold import TSNE
import umap
import prince

def apply_variable_transformation(dataframe, params):
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

    new_columns = {}

    if transformation_type in ["Dimensionality Reduction", "Dimensional Reduction-Based"]:
        # Standardize data
        if transformation in ["t-SNE", "UMAP"]:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(dataframe[columns])

        if transformation == "PCA":
            model = prince.PCA(
                n_components=n_components,
                rescale_with_mean=True,
                rescale_with_std=True,
                copy=True,
                check_input=True,
                engine='sklearn'
            )
            transformed_data = model.fit_transform(dataframe[columns])

        elif transformation == "t-SNE":
            model = TSNE(n_components=n_components, perplexity=perplexity,
                        learning_rate=learning_rate, random_state=42)
            transformed_data = model.fit_transform(scaled_data)
        elif transformation == "UMAP":
            model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                            min_dist=min_dist, random_state=42)
            transformed_data = model.fit_transform(scaled_data)
        else:  # FAMD
            model = prince.FAMD(n_components=n_components, random_state=42)
            transformed_data = model.fit_transform(dataframe[columns])

        # Create column names and DataFrame
        component_names = [naming_pattern.format(method_applied=transformation.replace(" ", "_")) + f"_{i + 1}" for i in range(n_components)]

        if transformation in ["t-SNE", "UMAP"]:
            result_df = pd.DataFrame(transformed_data, columns=component_names, index=dataframe.index)
        else:
            result_df = transformed_data.copy()
            result_df.columns = component_names

        new_columns.update(result_df.to_dict(orient="list"))

    elif transformation_type in ["Clustering", "Cluster-Based"]:
        from utils.clustering_utils import prepare_data_for_clustering, fit_kmeans, fit_dbscan, fit_gaussian_mixture
        
        # Clustering params
        n_clusters = params.get("n_clusters", 3)
        eps = params.get("eps", 0.5)
        min_samples = params.get("min_samples", 5)

        # Prepare Data (handles standardization)
        numeric_cols = dataframe[columns].select_dtypes(include=np.number).columns.tolist()
        prep_data, valid_idx = prepare_data_for_clustering(dataframe[columns], numeric_cols)
        
        if len(prep_data) >= 2:
            # Run Clustering
            if transformation == "K-Means":
                labels, _ = fit_kmeans(prep_data, n_clusters)
            elif transformation == "DBSCAN":
                labels, _ = fit_dbscan(prep_data, eps, min_samples)
            elif transformation == "Gaussian Mixture":
                labels, _ = fit_gaussian_mixture(prep_data, n_clusters)
            else:
                 labels = []

            if len(labels) > 0:
                # Map results back to original index
                full_labels = pd.Series(index=dataframe.index, data=np.nan)
                full_labels.loc[valid_idx] = labels
                
                col_name = naming_pattern.format(method_applied=transformation.replace(" ", "_"))
                new_columns[col_name] = full_labels.values

    elif transformation_type == "Encoding":
        if transformation == "One-Hot Encoding":
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            for col in columns:
                encoded_data = encoder.fit_transform(dataframe[[col]])
                categories = encoder.categories_[0]
                for i, category in enumerate(categories):
                    new_col_name = naming_pattern.format(
                        initial_variable=col,
                        category=category
                    )
                    new_columns[new_col_name]  = encoded_data[:, i]
        
        elif transformation == "Label Encoding":
            encoder = LabelEncoder()
            for col in columns:
                new_col_name = naming_pattern.format(
                    initial_variable=col,
                    category="label_encoded"
                )
                new_columns[new_col_name] = encoder.fit_transform(dataframe[col])
        
        elif transformation == "Ordinal Encoding":
            for col in columns:
                encoder = OrdinalEncoder(categories=[category_orders[col]])
                new_col_name = naming_pattern.format(
                    initial_variable=col,
                    category="ordinal"
                )
                new_columns[new_col_name] = encoder.fit_transform(dataframe[[col]]).flatten()

    else:
        # Handle statistical and scaling transformations
        method_function = TRANSFORMATIONS.get(transformation, None)
        if method_function:
            for col in columns:
                new_col_name = naming_pattern.format(
                    initial_variable=col,
                    method_applied=transformation.replace(" ", "_")
                )
                result = method_function(dataframe[col])
                if np.isscalar(result):
                    result = [result] * len(dataframe)
                new_columns[new_col_name] = result

    if new_columns:
        return pd.DataFrame(new_columns, index=dataframe.index)
    return None
