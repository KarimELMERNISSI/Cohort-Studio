######################################## PACKAGES ########################################
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer  # split some of our processing to specific columns
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import (  # to use basic (median, mean, most_frequent, constant, etc.) and knn imputers
    IterativeImputer,  # importable only after the side-effect import of enable_iterative_imputer above
    KNNImputer,
    SimpleImputer,
)
from sklearn.pipeline import Pipeline  # to make our data pre-processing pipeline
from sklearn.preprocessing import (  # preprocess non numerical variables and scale numerical ones
    OneHotEncoder,
    StandardScaler,
)

from biostats._types import RANDOM_STATE
from utils.data_analyzer import DataAnalyzer

#from missforest import MissForest # our MissForest adaptation is based on this package
from utils.miss_forest.missforest import MissForest


######################################## MISSFOREST IMPUTER CLASS ######################################## 
class MissForestTransformer(TransformerMixin):
    def __init__(self, categorical_cols=None, debug=False, progress_bar=None, **kwargs):
        """
        A transformer wrapper for MissForest imputation for both numerical and categorical data.

        Parameters:
        ----------
        categorical_cols : list, default=None
            List of categorical column names for imputation. If None, assumes no categorical columns.
        debug : bool, default=False
            If True, enables debug information output.
        """
        self.categorical_cols = categorical_cols
        self.debug = debug
        
        self.current_progress = 0  # To track progress updates
        self.total_steps = 10
        try:  # LightGBM is in the optional 'ml' extra (follow-up programme M4)
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:
            raise ImportError(
                "MissForest needs LightGBM, in the optional 'ml' extra: install it with `uv sync --extra ml`."
            ) from exc
        self.imputer = MissForest(
            clf=LGBMClassifier(verbosity=-1, n_jobs=-1, random_state=RANDOM_STATE),
            rgr=LGBMRegressor(verbosity=-1, n_jobs=-1, random_state=RANDOM_STATE),
            initial_guess='median',
            max_iter=self.total_steps,
            early_stopping=True,
            progress_bar= progress_bar
        )
        self.progress_bar = self.imputer.progress_bar
        self._is_fitted = False
        self.missforest_params = kwargs  # Additional params for MissForest

    def debug_print(self, *args):
        """Helper function for controlled debug printing."""
        if self.debug:
            print(*args)
        
    def fit(self, X, y=None):
        """
        Fit the MissForest imputer on X.

        Parameters:
        ----------
        X : pd.DataFrame
            The input dataframe to fit the imputer.
        y : Ignored

        Returns:
        -------
        self : object
            Returns self.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        # Update categorical columns based on input DataFrame
        if self.categorical_cols is not None:
            self.categorical_cols = [col for col in self.categorical_cols if col in X.columns]
        if self.categorical_cols is not None:
            self.numerical_cols = [col for col in X.columns if col not in self.categorical_cols]
        else:
            self.numerical_cols = X.columns
        
        self.debug_print("Fitting MissForest Imputer")
        self.debug_print(f"Identified categorical columns: {self.categorical_cols}")
        self.debug_print(f"Identified numerical columns: {self.numerical_cols}")

        # Fitting the imputer
        self.imputer.fit(X, categorical=self.categorical_cols)
        self._is_fitted = True

        return self

    def transform(self, X):
        """
        Impute missing values in X based on the fitted MissForest imputer.

        Parameters:
        ----------
        X : pd.DataFrame
            The input dataframe to impute.

        Returns:
        -------
        X_imputed : pd.DataFrame
            Dataframe with imputed values.
        """
        if not self._is_fitted:
            raise RuntimeError("You must fit the transformer before transforming data!")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        self.debug_print("Transforming data with MissForest Imputer")
        
        # Transforming data
        X_imputed = self.imputer.transform(X)
        
        # Converting back to DataFrame with original column names
        X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        return X_imputed_df

    def fit_transform(self, X, y=None):
        """
        Fit the MissForest imputer and transform X.

        Parameters:
        ----------
        X : pd.DataFrame
            The input dataframe to fit and transform.
        progress_placeholder : st.empty, optional
            Streamlit placeholder for the progress bar. If None, a new progress bar is created.
        y : Ignored

        Returns:
        -------
        X_imputed : pd.DataFrame
            Dataframe with imputed values.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # Filter out categorical columns that are not in the DataFrame
        if self.categorical_cols is not None:
            self.categorical_cols = [col for col in self.categorical_cols if col in X.columns]
        if self.categorical_cols is not None:
            self.numerical_cols = [col for col in X.columns if col not in self.categorical_cols]
        else:
            self.numerical_cols = X.columns
        
        self.debug_print("Fitting and transforming data with MissForest Imputer")
        self.debug_print(f"Categorical columns for imputation: {self.categorical_cols}")
        self.debug_print(f"Numerical columns for imputation: {self.numerical_cols}")

        # Fit-transforming data
        X_imputed = self.imputer.fit_transform(X, categorical=self.categorical_cols)
        self.debug_print(f"X data imputation:\n {X_imputed}\n")
        # Converting back to DataFrame with original column names
        X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        self._is_fitted = True

        return X_imputed_df
    
######################################## MISSFOREST IMPUTER CLASS ######################################## 

def _is_text(series: pd.Series) -> bool:
    """Text-like column: object, pandas string ('str' in pandas 3) or categorical dtype."""
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def detect_remainder_columns(data, threshold=0.8, date_like_columns=None):
    """
    Columns that should not be imputed or encoded: dates (true date/time dtypes and text columns
    recognised as dates) and high-cardinality text columns such as identifiers.

    Parameters:
    - data: pd.DataFrame
        The input DataFrame.
    - threshold: float, default=0.8
        The proportion of distinct values to non-null values to qualify as a remainder column.
        If the value is greater than 1, it defines the exact number of distinct values used as the threshold.
    - date_like_columns: iterable of str, optional
        Text columns already recognised as dates (e.g. ``DataAnalyzer.date_cols``).

    Returns:
    - list: Names of columns to be included in the remainder, preserving the order in the input DataFrame.
    """
    date_like = set(date_like_columns or [])
    remainder_cols = []
    for col in data.columns:
        series = data[col]
        if (
            col in date_like
            or pd.api.types.is_datetime64_any_dtype(series)
            or pd.api.types.is_timedelta64_dtype(series)
        ):
            remainder_cols.append(col)
            continue
        # High-cardinality text (identifiers, free text). Pandas 3 stores text as 'str', not 'object'.
        if _is_text(series):
            non_null_count = series.notnull().sum()
            distinct_count = series.nunique()
            if (threshold > 1 and distinct_count >= threshold) or (
                threshold <= 1 and distinct_count >= threshold * non_null_count
            ):
                remainder_cols.append(col)
    return remainder_cols



######################################## "UNIVERSAL" IMPUTER FUNCTIONS DEFINITION ######################################## 

# Helper function for controlled debug printing
def debug_print(*args, debug=False):
    if debug:
        print(*args)


# Helper function for getting feature names after transformation
def get_feature_names(preprocessor, column_names, num_scaler=False, cat_encoder=False, debug=False):
    """
    Retrieve the new feature names after applying transformations in the preprocessor.

    Parameters:
    - preprocessor: The fitted ColumnTransformer or Pipeline that has been applied to the data.
    - column_names (list): List of original column names.
    - num_scaler (bool): Whether numerical columns are scaled.
    - cat_encoder (bool): Whether categorical columns are encoded.
    - debug (bool): Whether to print debug information.

    Returns:
    - new_cols (list): List of transformed column names.
    """
    new_cols = column_names.copy()  # Start with original names

    # Check if the preprocessor is a ColumnTransformer or Pipeline
    if isinstance(preprocessor, ColumnTransformer):
        debug_print("IN ColumnTransformer", debug=debug)
        # Check for scalers and encoders in the pipeline
        if 'num_scaler' in preprocessor.named_transformers_ and num_scaler:
            debug_print("IN 'num_scaler' ColumnTransformer", debug=debug)
            if hasattr(preprocessor.named_transformers_['num_scaler'], 'get_feature_names_out'):
                debug_print(f"Scaling numerical columns: {column_names}", debug=debug)
                new_cols = preprocessor.named_transformers_['num_scaler'].get_feature_names_out(column_names).tolist()
        elif 'cat_encoder' in preprocessor.named_transformers_ and cat_encoder:
            debug_print("IN 'cat_encoder' ColumnTransformer")
            if hasattr(preprocessor.named_transformers_['cat_encoder'], 'get_feature_names_out'):
                debug_print(f"Encoding categorical columns: {column_names}", debug=debug)
                new_cols = preprocessor.named_transformers_['cat_encoder'].get_feature_names_out(column_names).tolist()
        else:
            if ('num_preprocessor' in preprocessor.named_transformers_) and num_scaler:
                new_cols = get_feature_names(preprocessor.named_transformers_['num_preprocessor'], column_names, num_scaler=num_scaler, cat_encoder=cat_encoder, debug=debug)
            elif ('cat_preprocessor' in preprocessor.named_transformers_) and cat_encoder:
                new_cols = get_feature_names(preprocessor.named_transformers_['cat_preprocessor'], column_names, num_scaler=num_scaler, cat_encoder=cat_encoder, debug=debug)
            elif ('preprocess_selection' in preprocessor.named_transformers_):
                new_cols = get_feature_names(preprocessor.named_transformers_['preprocess_selection'], column_names, num_scaler=num_scaler, cat_encoder=cat_encoder, debug=debug)
            else:
                debug_print("IN ColumnTransformer, no change")
                
    elif isinstance(preprocessor, Pipeline):
        # Check for scalers and encoders in the pipeline
        debug_print("IN Pipeline", debug=debug)
        if 'num_scaler' in preprocessor.named_steps and num_scaler:
            debug_print("IN 'num_scaler' Pipeline", debug=debug)
            if hasattr(preprocessor.named_steps['num_scaler'], 'get_feature_names_out'):
                debug_print(f"Scaling numerical columns: {column_names}", debug=debug)
                new_cols = preprocessor.named_steps['num_scaler'].get_feature_names_out(column_names).tolist()
        elif 'cat_encoder' in preprocessor.named_steps and cat_encoder:
            debug_print("IN 'cat_encoder' Pipeline", debug=debug)
            if hasattr(preprocessor.named_steps['cat_encoder'], 'get_feature_names_out'):
                debug_print(f"Encoding categorical columns: {column_names}", debug=debug)
                new_cols = preprocessor.named_steps['cat_encoder'].get_feature_names_out(column_names).tolist()
        else:
            if ('scaling_x_encoding' in preprocessor.named_steps):
                new_cols = get_feature_names(preprocessor.named_steps['scaling_x_encoding'], column_names, num_scaler=num_scaler, cat_encoder=cat_encoder, debug=debug)
            else:
                debug_print("IN Pipeline, no change", debug=debug) 
                
    debug_print(f"New columns after transformations: {new_cols}", debug=debug)

    return new_cols


# Sort columns during our data pre-processing
def reorder_data_columns(data, numerical_columns, categorical_columns):
    """
    Numerical columns first, then categorical ones, then every other column in its original order.
    No column is ever dropped (true/false, date and text columns used to disappear here).
    """
    first = [c for c in list(numerical_columns) + list(categorical_columns) if c in data.columns]
    rest = [c for c in data.columns if c not in set(first)]
    return data[first + rest]


def partition_columns(data, remainder_columns, analyzer=None):
    """
    Assign every column to exactly one group: (numerical, categorical, passthrough).

    - categorical: text and categorical columns, true/false columns, binary and low-cardinality numeric columns;
    - numerical: the other numeric columns;
    - passthrough: everything else (requested remainder columns, dates, identifiers, any other dtype), kept in
      the input order, which is the order in which a ColumnTransformer appends its remainder.
    """
    analyzer = analyzer or DataAnalyzer(data)
    remainder = [c for c in remainder_columns if c in data.columns]
    excluded = set(remainder) | set(analyzer.date_cols)
    categorical_like = (
        set(analyzer.categorical_cols)
        | set(analyzer.binary_cols)
        | set(analyzer.non_binary_low_cardinality_numeric_cols)
        | {c for c in data.columns if pd.api.types.is_bool_dtype(data[c])}
    )
    categorical = pd.Index([c for c in data.columns if c in categorical_like and c not in excluded])
    numerical = pd.Index(
        [
            c
            for c in data.columns
            if pd.api.types.is_numeric_dtype(data[c])
            and not pd.api.types.is_bool_dtype(data[c])
            and c not in excluded
            and c not in set(categorical)
        ]
    )
    assigned = set(numerical) | set(categorical)
    passthrough = [c for c in data.columns if c not in assigned]
    return numerical, categorical, passthrough


def get_imputer(numerical_imputation_method='mean',
                categorical_imputation_method='most_frequent',
                cat_encoder=False,
                num_scaler=False,
                data=None,
                remainder_columns=None,
                remainder_strategy='passthrough',
                remainder_threshold=0.5,
                total_steps=10,
                progress_placeholder=None,
                debug=False):
    """
    Returns the appropriate imputer based on the specified imputation methods for numerical and categorical columns.

    Parameters:
    - numerical_imputation_method (str): Imputation method for numerical columns ('mean', 'median', 'knn', 'missforest').
    - categorical_imputation_method (str): Imputation method for categorical columns ('most_frequent', 'missforest').
    - cat_encoder (bool): Whether to apply one-hot encoding to categorical columns.
    - num_scaler (bool): Whether to scale numerical columns.
    - data (pandas.DataFrame): Input DataFrame used to identify categorical and numerical columns.
    - remainder_columns: list, default=None: Columns to pass through without transformation (e.g., ID columns), will be excluded from imputation, applying remainder strategy. 
        NB : If it is set to 'auto', it will automatically guess the remainder columns based on your input data.
    - remainder_threshold (float): Default (0.8), from 0 to 1, filter out non-numeric columns having a ratio distinct modalities / non empty rows higher than the fixed threshold if ramainder columns is set to 'auto'  
    - remainder_strategy (str): 'passthrough' (keep as it is) or 'drop' (remove from the final dataset)
    - progress_placeholder: Streamlit empty container for showing the progress bar, default=None.
    - debug (bool): Whether to print debug information.

    Returns:
    - A ColumnTransformer object or a MissForestTransformer if using MissForest for both.
    - A list of all output columns in the final transformed DataFrame.
    """
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("Invalid input data. Please provide a pandas DataFrame.")
    # Handle remainder columns
    if remainder_strategy != 'drop':
        remainder_strategy = 'passthrough'
    # inject analyzer for binary+cat low cardinality tbd
    analyzer = DataAnalyzer(data)

    # Remainder columns are kept out of imputation/encoding (dates, identifiers, user choice)
    if remainder_columns is None:
        remainder_columns = []
    elif isinstance(remainder_columns, str) and remainder_columns == 'auto':
        remainder_columns = detect_remainder_columns(
            data, threshold=remainder_threshold, date_like_columns=analyzer.date_cols
        )
    else:
        remainder_columns = [col for col in remainder_columns if col in data.columns]

    # Every column belongs to exactly one group, so the output names always match the output values.
    # Anything that is neither numerical nor categorical (dates stored as text, identifiers, other dtypes)
    # passes through unchanged (or is dropped with remainder_strategy='drop').
    numerical_cols, categorical_cols, remainder_columns = partition_columns(data, remainder_columns, analyzer)
    debug_print(
        f"Imputer columns - numerical: {numerical_cols.tolist()}; categorical: {categorical_cols.tolist()}; "
        f"pass-through/remainder: {remainder_columns}",
        debug=debug,
    )

    data = reorder_data_columns(data, numerical_cols, categorical_cols)
    debug_print(f"ordered data cols: {data.columns.tolist()}", debug=debug)

    # If MissForest is chosen for both numerical and categorical imputation
    if numerical_imputation_method == categorical_imputation_method == 'missforest':
        imputer = MissForestTransformer(categorical_cols=categorical_cols, debug=debug, progress_bar=progress_placeholder, total_steps=total_steps)

        # Handle scaling and encoding
        transformers = []
        if num_scaler:
            transformers.append(('num_scaler', StandardScaler(), numerical_cols))
        else:
            transformers.append(('num_variables', 'passthrough', numerical_cols))
        if cat_encoder:
            transformers.append(('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))
        else:
            transformers.append(('cat_variables', 'passthrough', categorical_cols))

        scaler_x_encoder = ColumnTransformer(transformers=transformers, remainder=remainder_strategy)

        preprocess_selection = Pipeline(steps=[
            ('missforest_imputation', imputer),
            ('scaling_x_encoding', scaler_x_encoder)
        ])
        
        preprocessor = ColumnTransformer(transformers=[
            ('preprocess_selection', preprocess_selection, numerical_cols.union(categorical_cols)),
            ('remainder_pass', remainder_strategy, remainder_columns)
        ])
        
        preprocessor.fit(data)

        new_num_cols = get_feature_names(preprocessor, numerical_cols, num_scaler=num_scaler, debug=debug)  
        new_cat_cols = get_feature_names(preprocessor, categorical_cols, cat_encoder=cat_encoder, debug=debug)

        if remainder_strategy == 'drop':
            all_output_columns = list(new_num_cols) + list(new_cat_cols)
        else:
            all_output_columns = list(new_num_cols) + list(new_cat_cols) + list(remainder_columns)

        debug_print("Final new_num_cols", new_num_cols, debug=debug)
        debug_print("Final new_cat_cols", new_cat_cols, debug=debug)
        debug_print("All output columns", all_output_columns, debug=debug)

        return preprocessor, all_output_columns

    if numerical_imputation_method == 'mean':
        num_imputer = SimpleImputer(strategy='mean')
    elif numerical_imputation_method == 'median':
        num_imputer = SimpleImputer(strategy='median')
    elif numerical_imputation_method == 'knn':
        num_imputer = KNNImputer()
    elif numerical_imputation_method == 'mice':
        # Single imputation by chained equations (one deterministic pass); not multiple imputation.
        num_imputer = IterativeImputer(random_state=RANDOM_STATE)
    elif numerical_imputation_method == 'missforest':
        num_imputer = MissForestTransformer(categorical_cols=None, debug=True, progress_bar=progress_placeholder, total_steps=total_steps)
    else:
        raise ValueError("Unsupported numerical imputation method. Choose 'mean', 'median', 'knn', 'mice', or 'missforest'.")

    if categorical_imputation_method == 'most_frequent':
        cat_imputer = SimpleImputer(strategy='most_frequent')
    elif categorical_imputation_method == 'missforest':
        cat_imputer = MissForestTransformer(categorical_cols=categorical_cols, debug=True, progress_bar=progress_placeholder, total_steps=total_steps)
    else:
        raise ValueError("Unsupported categorical imputation method. Choose 'most_frequent' or 'missforest'.")

    num_pipeline = Pipeline(steps=[('num_imputer', num_imputer)])
    cat_pipeline = Pipeline(steps=[('cat_imputer', cat_imputer)])

    if num_scaler:
        num_pipeline.steps.insert(0, ('num_scaler', StandardScaler()))
    if cat_encoder:
        cat_pipeline.steps.append(('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))

    preprocessor = ColumnTransformer(transformers=[
        ('num_preprocessor', num_pipeline, numerical_cols),
        ('cat_preprocessor', cat_pipeline, categorical_cols)
    ], remainder=remainder_strategy)

    preprocessor.fit(data)

    new_num_cols = get_feature_names(preprocessor, numerical_cols, num_scaler=num_scaler, debug=debug) #.named_transformers_['num_preprocessor']
    new_cat_cols = get_feature_names(preprocessor, categorical_cols, cat_encoder=cat_encoder, debug=debug) #.named_transformers_['cat_preprocessor']

    if remainder_strategy == 'drop':
        all_output_columns = list(new_num_cols) + list(new_cat_cols)
    else:
        all_output_columns = list(new_num_cols) + list(new_cat_cols) + list(remainder_columns)

    debug_print("Final new_num_cols", new_num_cols, debug=debug)
    debug_print("Final new_cat_cols", new_cat_cols, debug=debug)
    debug_print("All output columns", all_output_columns, debug=debug)

    return preprocessor, all_output_columns
