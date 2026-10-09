import logging
from typing import Optional, Dict, List, Any, Union
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
import enrich.custom_metrics_and_filters as ecm

logger = logging.getLogger(__name__)

class DataQualityAuditor:
    """
    Audits a pandas DataFrame to assess data quality across multiple dimensions:
    - Completeness: Presence of missing values.
    - Uniqueness: Presence of duplicate rows.
    - Validity: Presence of outliers in numerical data.
    - Consistency: Basic type consistency checks.
    - Clinical Validity: Compliance with declared anomaly criteria.
    """

    def __init__(self, df: pd.DataFrame, config: Optional[Any] = None) -> None:
        self.df: pd.DataFrame = df
        self.config = config
        self.metrics: Dict[str, float] = {}
        self.advice: List[Dict[str, str]] = []
        self.clinical_anomalies_df = None
        self.clinical_anomalies_booleans = None
        self.uniformity_details: List[Dict[str, str]] = []

    def compute_completeness(self) -> float:
        """Calculates the percentage of non-missing values."""
        total_cells = self.df.size
        if total_cells == 0:
            return 0.0
        missing_cells = self.df.isnull().sum().sum()
        score = (1 - (missing_cells / total_cells)) * 100
        return round(float(score), 2)

    def compute_uniqueness(self) -> float:
        """Calculates the percentage of unique rows."""
        total_rows = len(self.df)
        if total_rows == 0:
            return 0.0
        unique_rows = len(self.df.drop_duplicates())
        score = (unique_rows / total_rows) * 100
        return round(float(score), 2)

    def compute_validity(self, method: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculates a validity score based on the absence of outliers in numerical columns.
        
        Parameters:
        -----------
        method : str, optional
            Detection method: 'iqr' (default), 'zscore', 'quantile', 
            'Local Outlier Factor', 'Isolation Forest', 'DBSCAN'
        params : dict, optional
            Method-specific parameters (e.g., multiplier, threshold, etc.)
        """
        # Default to IQR if no method specified
        if method is None:
            method = "iqr"
        if params is None:
            params = {"multiplier": 1.5} if method == "iqr" else {}
        
        logger.debug("compute_validity method=%s, params=%s", method, params)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        logger.debug("compute_validity numeric_cols=%s", list(numeric_cols))
        
        if len(numeric_cols) == 0:
            return 100.0  # No numeric columns to be invalid

        total_outliers = 0
        total_numeric_values = 0

        # Univariate methods
        if method in ["iqr", "zscore", "quantile"]:
            for col in numeric_cols:
                data = self.df[col].dropna()
                if len(data) == 0:
                    continue
                
                outliers = 0
                
                if method == "iqr":
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    multiplier = params.get("multiplier", 1.5)
                    
                    # Handle edge case: IQR = 0 (constant data)
                    if IQR == 0:
                        # No outliers if all values are the same
                        outliers = 0
                    else:
                        lower_bound = Q1 - multiplier * IQR
                        upper_bound = Q3 + multiplier * IQR
                        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
                    
                elif method == "zscore":
                    threshold = params.get("threshold", 3.0)
                    std = data.std()
                    
                    # Handle edge case: std = 0 (constant data)
                    if std == 0:
                        outliers = 0
                    else:
                        z_scores = np.abs((data - data.mean()) / std)
                        outliers = (z_scores > threshold).sum()
                    
                elif method == "quantile":
                    lower = params.get("lower", 0.01)
                    upper = params.get("upper", 0.99)
                    lower_bound = data.quantile(lower)
                    upper_bound = data.quantile(upper)
                    outliers = ((data < lower_bound) | (data > upper_bound)).sum()
                
                total_outliers += outliers
                total_numeric_values += len(data)
                logger.debug("compute_validity col=%s, outliers=%s, total=%s", col, outliers, total_outliers)
        
        # Multivariate ML methods
        elif method in ["Local Outlier Factor", "Isolation Forest", "DBSCAN"]:
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.neighbors import LocalOutlierFactor
                from sklearn.cluster import DBSCAN
                from sklearn.preprocessing import StandardScaler
                
                ml_data = self.df[numeric_cols].dropna()
                logger.debug("compute_validity ML method, data shape=%s", ml_data.shape)
                
                if len(ml_data) > 10:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(ml_data)
                    
                    if method == "Isolation Forest":
                        model = IsolationForest(contamination='auto', random_state=42)
                        predictions = model.fit_predict(scaled_data)
                    elif method == "Local Outlier Factor":
                        model = LocalOutlierFactor(contamination='auto')
                        predictions = model.fit_predict(scaled_data)
                    elif method == "DBSCAN":
                        eps = params.get("eps", 0.5)
                        min_samples = params.get("min_samples", 5)
                        model = DBSCAN(eps=eps, min_samples=min_samples)
                        predictions = model.fit_predict(scaled_data)
                    
                    total_outliers = (predictions == -1).sum()
                    total_numeric_values = len(ml_data)
                    logger.debug("compute_validity ML outliers=%s", total_outliers)
                else:
                    logger.debug("compute_validity not enough data for ML: %s", len(ml_data))
                    return 100.0  # Not enough data
            except ImportError as e:
                logger.warning("compute_validity ImportError: %s, falling back to IQR", e)
                # Fall back to IQR if sklearn not available
                return self.compute_validity(method="iqr")

        if total_numeric_values == 0:
            logger.debug("compute_validity total_numeric_values=0, returning 100")
            return 100.0

        # Score penalizes outliers
        score = (1 - (total_outliers / total_numeric_values)) * 100
        logger.debug("compute_validity total_outliers=%s, total_values=%s, score=%s", total_outliers, total_numeric_values, score)
        return round(float(score), 2)

    def compute_consistency(self) -> float:
        """
        Approximates consistency by checking if text/object columns could be converted to numeric or datetime.
        If a column is text but contains mostly numbers, it might be inconsistent formatting.
        """
        object_cols = self.df.select_dtypes(include=['object', 'string']).columns
        if len(object_cols) == 0:
            return 100.0

        inconsistent_cols = 0
        for col in object_cols:
            # Check if it looks like a number
            try:
                pd.to_numeric(self.df[col].dropna())
                inconsistent_cols += 1
                continue
            except:
                pass
            
            # Check if it looks like a date
            try:
                pd.to_datetime(self.df[col].dropna())
                inconsistent_cols += 1
                continue
            except:
                pass

        # Score: 100 - (percentage of potentially mis-typed columns)
        score = (1 - (inconsistent_cols / len(object_cols))) * 100
        return round(score, 2)

    def compute_clinical_validity(self) -> Optional[float]:
        """
        Calculates a score based on declared anomaly criteria (if available).
        Score = 100 - (% of rows triggering at least one anomaly).
        Returns None if no anomaly rules are defined (shows as N/A in dashboard).
        """
        self.clinical_anomalies_df = None
        self.clinical_anomalies_booleans = None
        
        # If no config or no mask_families, return None (N/A - rules not configured)
        if not self.config or "mask_families" not in self.config:
            return None

        anomaly_config = self.config["mask_families"].get("clinical_anomalies", {})
        # If no anomaly rules defined, return None (N/A - no rules to evaluate)
        if not anomaly_config:
            return None

        try:
            # Generate masks for all defined anomalies
            masks_zip = list(ecm.zip_masks(self.df, anomaly_config))
            
            # Combine masks to find rows with ANY anomaly
            any_anomaly_mask = pd.Series(False, index=self.df.index)
            has_masks = False
            anomaly_data = {}
            
            for mask_name, mask_values in masks_zip:
                if mask_values is not None:
                    any_anomaly_mask |= mask_values
                    anomaly_data[mask_name] = mask_values
                    has_masks = True
            
            if not has_masks:
                # No valid masks generated - treat as N/A (rules exist but none matched columns)
                return None
            
            # Store details for UI
            if anomaly_data:
                # 1. Store the boolean mask for styling logic
                self.clinical_anomalies_booleans = pd.DataFrame(anomaly_data)
                self.clinical_anomalies_booleans = self.clinical_anomalies_booleans[any_anomaly_mask]
                
                # 2. Store the original values for display
                self.clinical_anomalies_df = self.df[any_anomaly_mask].copy()
                
                # 3. Add an "Anomalies" column listing the triggered masks
                def get_active_anomalies(row):
                    return [col for col in row.index if row[col]]
                
                anomalies_list = self.clinical_anomalies_booleans.apply(get_active_anomalies, axis=1)
                self.clinical_anomalies_df.insert(0, "Anomalies", anomalies_list)

            total_rows = len(self.df)
            if total_rows == 0:
                return 100.0
                
            anomaly_count = any_anomaly_mask.sum()
            score = (1 - (anomaly_count / total_rows)) * 100
            return round(score, 2)
            
        except Exception as e:
            logger.error("Error computing clinical validity: %s", e)
            return None

    def compute_uniformity(self) -> float:
        """
        Checks for string uniformity issues:
        - Leading/trailing whitespace.
        - Inconsistent capitalization (e.g., 'Male' vs 'male').
        """
        self.uniformity_details = []
        object_cols = self.df.select_dtypes(include=['object', 'string']).columns
        if len(object_cols) == 0:
            return 100.0
        
        issues_count = 0
        total_checks = 0
        
        for col in object_cols:
            series = self.df[col].dropna().astype(str)
            if len(series) == 0:
                continue
            
            total_checks += 1
            penalty = 0
            col_issues = []
            
            # Check 1: Whitespace
            whitespace_mask = series.str.strip() != series
            if whitespace_mask.any():
                penalty += 0.5
                examples = series[whitespace_mask].unique()[:3].tolist()
                col_issues.append(f"Whitespace issues (e.g., {examples})")
            
            # Check 2: Capitalization consistency
            n_unique = series.nunique()
            n_unique_lower = series.str.lower().nunique()
            
            if n_unique != n_unique_lower:
                penalty += 0.5
                # Find examples of duplicates when lowercased
                counts = series.groupby(series.str.lower()).nunique()
                problem_groups = counts[counts > 1].index.tolist()
                examples = []
                for pg in problem_groups[:3]:
                    original_values = series[series.str.lower() == pg].unique().tolist()
                    examples.append(f"{pg} -> {original_values}")
                col_issues.append(f"Capitalization inconsistency (e.g., {examples})")
                
            issues_count += min(penalty, 1.0) # Max 1 penalty per column
            
            if col_issues:
                self.uniformity_details.append({
                    "Column": col,
                    "Issues": "; ".join(col_issues)
                })

        if total_checks == 0:
            return 100.0
            
        score = (1 - (issues_count / total_checks)) * 100
        return round(score, 2)

    # --- Advanced Completeness Analysis ---

    def compute_nullity_correlation(self):
        """
        Calculates the correlation between the missingness of variables.
        Returns a DataFrame where 1 means variables tend to be missing together.
        """
        return self.df.isnull().corr()

    def check_mar_dependency(self, target_col):
        """
        Checks if missingness in `target_col` is dependent on other observed variables (MAR).
        Returns a list of dependencies found.
        """
        from scipy import stats
        
        dependencies = []
        if target_col not in self.df.columns:
            return dependencies
            
        missing_mask = self.df[target_col].isnull()
        if missing_mask.sum() == 0 or missing_mask.sum() == len(self.df):
            return dependencies # Cannot test if all or none are missing
            
        # Split data
        group_missing = self.df[missing_mask]
        group_observed = self.df[~missing_mask]
        
        # Test against other numerical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == target_col:
                continue
                
            # Get values for both groups, dropping NaNs in the predictor column
            vals_missing = group_missing[col].dropna()
            vals_observed = group_observed[col].dropna()
            
            if len(vals_missing) < 2 or len(vals_observed) < 2:
                continue
                
            # Perform T-test (ind)
            try:
                t_stat, p_val = stats.ttest_ind(vals_missing, vals_observed, equal_var=False)
                
                if p_val < 0.05:
                    dependencies.append({
                        "Predictor": col,
                        "p_value": p_val,
                        "Mean_Missing": vals_missing.mean(),
                        "Mean_Observed": vals_observed.mean(),
                        "Difference": vals_missing.mean() - vals_observed.mean()
                    })
            except Exception:
                pass
                
        return sorted(dependencies, key=lambda x: x['p_value'])

    def perform_mcar_test(self):
        """
        Performs a heuristic Little's MCAR test by aggregating pairwise MAR checks.
        If we find significant dependencies between missingness and observed values, 
        we reject the MCAR hypothesis.
        """
        from scipy import stats
        
        p_values = []
        cols_with_missing = [col for col in self.df.columns if self.df[col].isnull().any()]
        
        if not cols_with_missing:
            return {"p_value": 1.0, "interpretation": "No missing data (MCAR trivially true)", "is_mcar": True}
            
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for target_col in cols_with_missing:
            missing_mask = self.df[target_col].isnull()
            group_missing = self.df[missing_mask]
            group_observed = self.df[~missing_mask]
            
            for col in numeric_cols:
                if col == target_col: continue
                
                vals_missing = group_missing[col].dropna()
                vals_observed = group_observed[col].dropna()
                
                if len(vals_missing) >= 2 and len(vals_observed) >= 2:
                    try:
                        _, p_val = stats.ttest_ind(vals_missing, vals_observed, equal_var=False)
                        p_values.append(p_val)
                    except: pass
        
        if not p_values:
             return {"p_value": 1.0, "interpretation": "Insufficient data to test MCAR", "is_mcar": True}
             
        # Combine p-values (Fisher's method would be better, but simple min with Bonferroni is conservative)
        # Here we just return the minimum p-value as a signal of the strongest dependency found.
        # If min_p < 0.05/N, we reject MCAR.
        min_p = min(p_values)
        
        # Simple interpretation for the UI
        is_mcar = min_p > 0.05 
        
        return {
            "p_value": min_p,
            "interpretation": "Likely MCAR (Missing Completely At Random)" if is_mcar else "Likely Not MCAR (MAR or MNAR detected)",
            "is_mcar": is_mcar
        }

    def run_audit(
        self, 
        validity_method: Optional[str] = None, 
        validity_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Optional[float]]:
        """Runs all checks and populates metrics and advice.
        
        Parameters:
        -----------
        validity_method : str, optional
            Method to use for Statistical Validity calculation
        validity_params : dict, optional
            Parameters for the validity method
        """
        self.metrics = {
            "Completeness": self.compute_completeness(),
            "Uniqueness": self.compute_uniqueness(),
            "Statistical Validity": self.compute_validity(method=validity_method, params=validity_params),
            "Consistency": self.compute_consistency(),
            "Uniformity": self.compute_uniformity()
        }
        
        # Add Clinical Validity only if rules are defined (otherwise N/A)
        clinical_score = self.compute_clinical_validity()
        if clinical_score is not None:
            self.metrics["Clinical Validity"] = clinical_score
        
        self.generate_advice()
        return self.metrics

    def generate_advice(self):
        """Generates actionable advice based on metrics."""
        self.advice = []
        
        # 1. Completeness Advice
        if self.metrics["Completeness"] < 95:
            missing_series = self.df.isnull().sum()
            missing_cols = missing_series[missing_series > 0].sort_values(ascending=False)
            top_missing = missing_cols.head(3).index.tolist()
            
            msg = "Your dataset has missing values."
            if top_missing:
                msg += f" Top columns with missing data: **{', '.join(top_missing)}**."
            msg += " Consider using **Imputation** (Mean, Median, or MICE) in the 'Handle Missing Values' section."
            
            self.advice.append({
                "category": "Completeness",
                "message": msg,
                "severity": "high" if self.metrics["Completeness"] < 80 else "medium"
            })
        
        # 2. Uniqueness Advice
        if self.metrics["Uniqueness"] < 100:
            duplicates = len(self.df) - len(self.df.drop_duplicates())
            self.advice.append({
                "category": "Uniqueness",
                "message": f"Found **{duplicates}** duplicate rows. These can skew analysis results. Check if these are valid repetitions or data entry errors.",
                "severity": "medium"
            })

        # 3. Statistical Validity Advice (Outliers)
        if self.metrics["Statistical Validity"] < 90:
            # Quick check for top outlier columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            outlier_cols = []
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers > 0:
                    outlier_cols.append((col, outliers))
            
            outlier_cols.sort(key=lambda x: x[1], reverse=True)
            top_outliers = [x[0] for x in outlier_cols[:3]]
            
            msg = "Detected potential outliers in numerical columns."
            if top_outliers:
                msg += f" Top affected columns: **{', '.join(top_outliers)}**."
            msg += " Use the **Outlier Detection** module to inspect and handle them (Clip or Remove)."

            self.advice.append({
                "category": "Statistical Validity",
                "message": msg,
                "severity": "medium"
            })

        # 4. Consistency Advice
        if self.metrics["Consistency"] < 100:
            self.advice.append({
                "category": "Consistency",
                "message": "Some 'Object' (text) columns appear to contain numeric or date data. Converting them will enable better analysis and sorting.",
                "severity": "low"
            })

        # 5. Clinical Validity Advice
        if "Clinical Validity" in self.metrics and self.metrics["Clinical Validity"] < 100:
            msg = f"Detected rows violating declared anomaly criteria (Score: {self.metrics['Clinical Validity']}%)."
            
            # Try to find top violated rule
            if hasattr(self, 'clinical_anomalies_booleans') and self.clinical_anomalies_booleans is not None:
                rule_counts = self.clinical_anomalies_booleans.sum().sort_values(ascending=False)
                if not rule_counts.empty:
                    top_rule = rule_counts.index[0]
                    msg += f" Most frequent violation: **{top_rule}**."
            
            msg += " Review 'Data Validation & Monitoring' tab."

            self.advice.append({
                "category": "Clinical Validity",
                "message": msg,
                "severity": "high" if self.metrics["Clinical Validity"] < 90 else "medium"
            })

        # 6. Uniformity Advice
        if self.metrics.get("Uniformity", 100) < 100:
            msg = "String columns have inconsistent formatting."
            if hasattr(self, 'uniformity_details') and self.uniformity_details:
                top_cols = [d['Column'] for d in self.uniformity_details[:3]]
                msg += f" Check columns: **{', '.join(top_cols)}**."
            msg += " Consider normalizing text data (trim whitespace, standardize case)."
            
            self.advice.append({
                "category": "Uniformity",
                "message": msg,
                "severity": "low"
            })

        if not self.advice:
            self.advice.append({
                "category": "General",
                "message": "Great job! Your dataset looks clean and ready for analysis.",
                "severity": "success"
            })

        return self.advice
