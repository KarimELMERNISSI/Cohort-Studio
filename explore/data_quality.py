import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
import enrich.custom_metrics_and_filters as ecm

class DataQualityAuditor:
    """
    Audits a pandas DataFrame to assess data quality across multiple dimensions:
    - Completeness: Presence of missing values.
    - Uniqueness: Presence of duplicate rows.
    - Validity: Presence of outliers in numerical data.
    - Consistency: Basic type consistency checks.
    - Clinical Validity: Compliance with declared anomaly criteria.
    """

    def __init__(self, df, config=None):
        self.df = df
        self.config = config
        self.metrics = {}
        self.advice = []

    def compute_uniqueness(self):
        """Calculates the percentage of unique rows."""
        total_rows = len(self.df)
        if total_rows == 0:
            return 0.0
        unique_rows = len(self.df.drop_duplicates())
        score = (unique_rows / total_rows) * 100
        return round(score, 2)

    def compute_validity(self):
        """
        Calculates a validity score based on the absence of outliers in numerical columns.
        Uses the IQR method.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 100.0 # No numeric columns to be invalid

        total_outliers = 0
        total_numeric_values = 0

        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) == 0:
                continue
            
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            total_outliers += outliers
            total_numeric_values += len(data)

        if total_numeric_values == 0:
            return 100.0

        # Score penalizes outliers. 
        # If 10% of data are outliers, score is 90.
        score = (1 - (total_outliers / total_numeric_values)) * 100
        return round(score, 2)

    def compute_consistency(self):
        """
        Approximates consistency by checking if object columns could be converted to numeric or datetime.
        If a column is 'object' but contains mostly numbers, it might be inconsistent formatting.
        """
        object_cols = self.df.select_dtypes(include=['object']).columns
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

    def compute_clinical_validity(self):
        """
        Calculates a score based on declared anomaly criteria (if available).
        Score = 100 - (% of rows triggering at least one anomaly).
        """
        self.clinical_anomalies_df = None
        self.clinical_anomalies_booleans = None
        
        if not self.config or "mask_families" not in self.config:
            return None

        anomaly_config = self.config["mask_families"].get("clinical_anomalies", {})
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
                return 100.0
            
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
            print(f"Error computing clinical validity: {e}")
            return None

    def compute_uniformity(self):
        """
        Checks for string uniformity issues:
        - Leading/trailing whitespace.
        - Inconsistent capitalization (e.g., 'Male' vs 'male').
        """
        self.uniformity_details = []
        object_cols = self.df.select_dtypes(include=['object']).columns
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

    def run_audit(self):
        """Runs all checks and populates metrics and advice."""
        self.metrics = {
            "Completeness": self.compute_completeness(),
            "Uniqueness": self.compute_uniqueness(),
            "Statistical Validity": self.compute_validity(),
            "Consistency": self.compute_consistency(),
            "Uniformity": self.compute_uniformity()
        }
        
        # Add Clinical Validity if config is present
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
