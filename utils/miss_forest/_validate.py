from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from utils.miss_forest._errors import MultipleDataTypesError


def _is_estimator(estimator: Any | BaseEstimator) -> bool:
    """Checks if the argument `estimator` is an object that implements the
    scikit-learn estimator API.

    Parameters
    ----------
    estimator : estimator object
        This object is assumed to implement the scikit-learn estimator API.

    Returns
    -------
    bool
        Returns True if the argument `estimator` is None or has class
        methods `fit` and `predict`. Otherwise, returns False.
    """
    try:
        # Check if class methods `fit` and `predict` exist and callable.
        is_has_fit_method = callable(estimator.fit)
        is_has_predict_method = callable(estimator.predict)

        return is_has_fit_method and is_has_predict_method
    except AttributeError:
        return False


def _validate_single_datatype_features(x: pd.DataFrame) -> None:
    """Checks if all values in the features belong to the same datatype.

    Parameters
    ----------
    x : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needs to be checked.

    Raises
    ------
    MultipleDataTypesError
        Raised if not all values in the features belong to the same datatype.
    """
    vectorized_type = np.vectorize(type)
    for c in x.columns:
        all_type = vectorized_type(x[c].dropna())
        if len(pd.unique(all_type)) > 1:
            raise MultipleDataTypesError(
                f"Multiple data types found in feature {c}.")