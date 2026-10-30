import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from biostats.outliers import UNIVARIATE_METHODS, detect_outliers
from core.derived import from_config
from core.population import family_checks, family_masks
from core.registry import as_code
from enrich.derived import CANNOT, DISAGREES, MIN_ROWS, MOSTLY, TOO_FEW, check_derived

logger = logging.getLogger(__name__)

# What each dimension's details table holds: where the issue is, what it is, how many, and the share affected.
DETAIL_COLUMNS: tuple[str, ...] = ("Where", "Issue", "Count", "Share")
# the advice on the declared derived variables (DataQualityAuditor.derived_advice), which the dashboard links to their tab
DERIVED_CATEGORY = "Derived variables"


def _details(rows: list[tuple[str, str, float | None, float | None]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(DETAIL_COLUMNS))
    if not frame.empty:
        frame = frame.sort_values("Count", ascending=False, na_position="last", kind="stable").reset_index(drop=True)
    return frame


class DataQualityAuditor:
    """
    Audits a pandas DataFrame to assess data quality across multiple dimensions:
    - Completeness: Presence of missing values.
    - Uniqueness: Presence of duplicate rows.
    - Validity: Presence of outliers in numerical data.
    - Consistency: Basic type consistency checks.
    - Domain Validity: the share of the plausibility checks that pass (each rule checked on the rows holding the
      values it reads).
    """

    def __init__(
        self, df: pd.DataFrame, config: Any | None = None, imputed: Mapping[str, Any] | None = None
    ) -> None:
        self.df: pd.DataFrame = df
        self.config = config
        # the cells an imputation filled (row ids per column): a derived variable's check does not compare them
        self.imputed = imputed
        self.metrics: dict[str, float] = {}
        self.advice: list[dict[str, str]] = []
        self.clinical_anomalies_df = None
        self.clinical_anomalies_booleans = None
        # Domain Validity's (checks that fail, checks made, rows flagged); None while it is N/A
        self.domain_checks: tuple[int, int, int] | None = None
        self.uniformity_details: list[dict[str, str]] = []
        # Per dimension: the columns, rows or rules behind the points it loses (see DETAIL_COLUMNS).
        self.details: dict[str, pd.DataFrame] = {}

    def compute_completeness(self) -> float:
        """Calculates the percentage of non-missing values."""
        total_cells = self.df.size
        if total_cells == 0:
            return 0.0
        per_column = self.df.isnull().sum()
        n_rows = len(self.df)
        self.details["Completeness"] = _details(
            [(str(col), "empty cells", int(n), n / n_rows) for col, n in per_column.items() if n > 0]
        )
        missing_cells = per_column.sum()
        score = (1 - (missing_cells / total_cells)) * 100
        return round(float(score), 2)

    def compute_uniqueness(self) -> float:
        """Calculates the percentage of unique rows."""
        total_rows = len(self.df)
        if total_rows == 0:
            return 0.0
        unique_rows = len(self.df.drop_duplicates())
        duplicates = total_rows - unique_rows
        self.details["Uniqueness"] = _details(
            [("whole rows", "exact duplicate of an earlier row", duplicates, duplicates / total_rows)] if duplicates else []
        )
        score = (unique_rows / total_rows) * 100
        return round(float(score), 2)

    def compute_validity(self, method: str | None = None, params: dict[str, Any] | None = None) -> float:
        """
        Calculates a validity score based on the absence of outliers in numerical columns.

        The outliers are those of biostats.outliers.detect_outliers, the detector of the Validate page and of the
        dashboard's outlier tab, with the same parameters, so the score counts what they flag.
        - Univariate methods ('iqr', the default, 'zscore', 'quantile'): the share of the values of the continuous
          numeric columns that are outliers.
        - Multivariate methods ('Local Outlier Factor', 'Isolation Forest', 'DBSCAN'): the share of the rows
          complete on the numeric columns that are flagged (at least 11 rows are needed).

        Parameters:
        -----------
        method : str, optional
            Detection method: 'iqr' (default), 'zscore', 'quantile',
            'Local Outlier Factor', 'Isolation Forest', 'DBSCAN'
        params : dict, optional
            Method-specific parameters (e.g., multiplier, threshold, etc.); the defaults are those of
            biostats.outliers.DEFAULT_PARAMS
        """
        method = method or "iqr"
        logger.debug("compute_validity method=%s, params=%s", method, params)
        numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        self.details["Statistical Validity"] = _details([])
        if not numeric_cols:
            return 100.0  # No numeric columns to be invalid

        detection = detect_outliers(self.df, numeric_cols, method, params)
        detail_rows: list[tuple[str, str, float | None, float | None]] = []
        if method in UNIVARIATE_METHODS:
            counts = detection.flags.sum()
            observed = {col: int(self.df[col].notna().sum()) for col in detection.columns}
            total_outliers = int(counts.sum())
            total_numeric_values = sum(observed.values())
            detail_rows = [
                (str(col), f"outliers ({method})", int(n), int(n) / observed[col])
                for col, n in counts.items()
                if n
            ]
        else:
            total_numeric_values = len(self.df) - detection.not_assessed
            if total_numeric_values <= 10:
                logger.debug("compute_validity not enough data for %s: %s rows", method, total_numeric_values)
                return 100.0  # Not enough data
            total_outliers = int(detection.flags["overall"].sum())
            if total_outliers:
                detail_rows.append(
                    (
                        "numeric columns together",
                        f"rows flagged by {method}",
                        total_outliers,
                        total_outliers / total_numeric_values,
                    )
                )

        self.details["Statistical Validity"] = _details(detail_rows)
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
        self.details["Consistency"] = _details([])
        if len(object_cols) == 0:
            return 100.0

        inconsistent_cols = 0
        detail_rows: list[tuple[str, str, float | None, float | None]] = []
        for col in object_cols:
            values = self.df[col].dropna()
            # Check if it looks like a number
            try:
                pd.to_numeric(values)
                inconsistent_cols += 1
                detail_rows.append((str(col), "numbers stored as text", int(len(values)), None))
                continue
            except Exception:
                pass  # not numbers: the date check comes next

            # Check if it looks like a date
            try:
                pd.to_datetime(values)
                inconsistent_cols += 1
                detail_rows.append((str(col), "dates stored as text", int(len(values)), None))
                continue
            except Exception:
                pass  # not dates either: the column holds text, as its type says
        self.details["Consistency"] = _details(detail_rows)

        # Score: 100 - (percentage of potentially mis-typed columns)
        score = (1 - (inconsistent_cols / len(object_cols))) * 100
        return round(score, 2)

    def compute_domain_validity(self) -> float | None:
        """
        The share of the plausibility checks that pass, 100 x (1 - F/C) (core.population.family_checks): a check is
        one rule on one row that holds the values the rule reads, never a missing value, which Completeness counts; a
        bounds rule makes one check per column, and a family read with "all" or "at least k" one per row where its
        rules settle it. Before, every row counted, so a row missing a rule's value passed it. A rule that cannot be
        evaluated, or that no row can check, is listed in the details and not counted. The rows the family flags, as
        the analysis population reads it (core.population.family_masks), are listed apart. None (N/A) when no rule is
        declared or no check can be made.
        """
        self.clinical_anomalies_df = None
        self.clinical_anomalies_booleans = None
        self.domain_checks = None

        # If no config or no mask_families, return None (N/A - rules not configured)
        if not self.config or "mask_families" not in self.config:
            return None

        anomaly_config = self.config["mask_families"].get("clinical_anomalies", {})
        # If no plausibility rule is declared, return None (N/A - no rules to evaluate)
        if not anomaly_config:
            return None

        try:
            masks = family_masks(self.df, anomaly_config)
            broken = [(str(rule), f"cannot be evaluated: {message}", None, None) for rule, message in masks.errors.items()]
            if masks.errors and "operator" not in masks.errors:  # the family result over the rules that can be evaluated
                masks = family_masks(
                    self.df, {k: v for k, v in anomaly_config.items() if k == "operator" or k not in masks.errors}
                )
            if masks.flag is None:
                # no rule can be evaluated (or the operator is unusable): N/A, with the reasons
                self.details["Domain Validity"] = _details(broken)
                return None
            any_anomaly_mask = masks.flag
            anomaly_data = dict(masks.rules)

            # Store details for UI
            if anomaly_data:
                # 1. Store the boolean mask for styling logic
                self.clinical_anomalies_booleans = pd.DataFrame(anomaly_data)
                self.clinical_anomalies_booleans = self.clinical_anomalies_booleans[any_anomaly_mask]
                
                # 2. Store the original values for display
                self.clinical_anomalies_df = self.df[any_anomaly_mask].copy()
                
                # 3. Add a "Rules broken" column listing the rules that fire
                def get_active_anomalies(row):
                    return [col for col in row.index if row[col]]
                
                anomalies_list = self.clinical_anomalies_booleans.apply(get_active_anomalies, axis=1)
                self.clinical_anomalies_df.insert(0, "Rules broken", anomalies_list)

            found = family_checks(self.df, anomaly_config)  # a rule in error is left out, and listed above
            rows = []
            for unit, check in found.units.items():
                checked, fired = int(check.checked.sum()), int(check.fired.sum())
                if not checked:
                    rows.append((unit, "not checked: no row holds its values", None, None))
                elif fired:
                    rows.append((unit, f"flagged, of {checked:,} rows holding its values", fired, fired / checked))
            if found.operator != "any" and found.checks:
                words = "every rule" if found.operator == "all" else "enough rules"
                rows.append(
                    (
                        "the family",
                        f"rows where {words} fire, of {found.checks:,} rows where the rules settle it",
                        found.fails,
                        found.fails / found.checks,
                    )
                )
            self.details["Domain Validity"] = _details(rows + broken)
            if not found.checks:
                return None
            self.domain_checks = (found.fails, found.checks, int(any_anomaly_mask.sum()))
            return round((1 - found.fails / found.checks) * 100, 2)
            
        except Exception as e:
            logger.error("Error computing domain validity: %s", e)
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

        self.details["Uniformity"] = _details(
            [(d["Column"], d["Issues"], None, None) for d in self.uniformity_details]
        )
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
        if int(missing_mask.sum()) in (0, len(self.df)):
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
            except Exception as exc:
                logger.debug("MAR check: t-test of %s by missingness of %s skipped: %s", col, target_col, exc)
                
        return sorted(dependencies, key=lambda x: x['p_value'])

    def perform_mcar_test(self):
        """
        Performs a heuristic Little's MCAR test by aggregating pairwise MAR checks.
        If we find significant dependencies between missingness and observed values, 
        we reject the MCAR hypothesis.
        """
        from scipy import stats
        
        p_values = []
        cols_with_missing = [col for col in self.df.columns if bool(self.df[col].isnull().any())]
        
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
                    except Exception as exc:
                        logger.debug("MCAR check: t-test of %s by missingness of %s skipped: %s", col, target_col, exc)
        
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
        validity_method: str | None = None, 
        validity_params: dict[str, Any] | None = None
    ) -> dict[str, float | None]:
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
        
        # Add Domain Validity only if rules are declared (otherwise N/A)
        domain_score = self.compute_domain_validity()
        if domain_score is not None:
            self.metrics["Domain Validity"] = domain_score
        
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
                outliers = int(((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum())
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

        # 5. Domain Validity Advice
        if "Domain Validity" in self.metrics and self.metrics["Domain Validity"] < 100:
            msg = f"Some rows break the declared plausibility rules (Domain Validity: {self.metrics['Domain Validity']}%)."
            if self.domain_checks is not None:
                fails, checks, rows = self.domain_checks
                msg = (
                    f"{fails:,} of {checks:,} plausibility checks fail, on {rows:,} row{'s' if rows != 1 else ''} "
                    f"(Domain Validity: {self.metrics['Domain Validity']}%)."
                )
            
            # Try to find top violated rule
            if hasattr(self, 'clinical_anomalies_booleans') and self.clinical_anomalies_booleans is not None:
                rule_counts = self.clinical_anomalies_booleans.sum().sort_values(ascending=False)
                if not rule_counts.empty:
                    top_rule = rule_counts.index[0]
                    msg += f" Most frequent violation: {as_code(top_rule)}."
            
            msg += " Review them on the Handle Implausible Values tab."

            self.advice.append({
                "category": "Domain Validity",
                "message": msg,
                "severity": "high" if self.metrics["Domain Validity"] < 90 else "medium"
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

        # 7. Derived variables (advice only: the score does not change)
        derived = self.derived_advice()
        if derived is not None:
            self.advice.append(derived)

        if not self.advice:
            self.advice.append({
                "category": "General",
                "message": "Great job! Your dataset looks clean and ready for analysis.",
                "severity": "success"
            })

        return self.advice

    def derived_advice(self) -> dict[str, str] | None:
        """
        The declared derived variables (Enrich > Handle Missing Data > Derived variables) against the data, every row:
        the recorded values their formula does not give, and the missing values it can compute ("computational
        conformance", Kahn et al. 2016). Variable names (as code: a name can come from an uploaded file), counts and the
        check's tolerance, never a row's value nor a check's reason (it can quote one); none when nothing is declared
        or there is nothing to say. Shown on screen only: an export would need the study package's disclosure rule
        (k = 5 for each count and difference).
        """
        definitions = from_config(self.config) if isinstance(self.config, Mapping) else ()
        lines: list[str] = []
        severity = "low"
        for definition in definitions:
            check = check_derived(definition, self.df, imputed=self.imputed)
            name, evidence = as_code(definition.target), check.describe(as_code)
            if check.verdict == CANNOT:
                lines.append(f"{name}: its formula cannot be used on these data.")
                continue
            if check.verdict == TOO_FEW:
                lines.append(
                    f"{name}: too few rows have it and every variable of its formula recorded to check it "
                    f"({check.compared:,}; the check needs {MIN_ROWS})."
                    if check.compared < MIN_ROWS
                    else f"{name}: it takes one value in nearly every row, so the check cannot tell its formula "
                    "from chance."
                )
                continue
            other = check.compared - check.agreeing
            errors = "a probable entry error" if other == 1 else "probable entry errors"
            if check.verdict == DISAGREES:
                severity = "medium"
                if check.level_of:  # the level itself is a value of the data: never named
                    lines.append(
                        f"{name}: the data disagree with its formula for one of the values of "
                        f"{as_code(check.level_of)}, although it {evidence}."
                    )
                else:
                    hint = f" ({check.unit_hint})" if check.unit_hint else ""
                    lines.append(f"{name}: the data disagree with its formula, which {evidence}{hint}.")
            elif other:
                rows = "the other row" if other == 1 else f"the {other:,} other rows"
                if check.unit_hint:
                    why = check.unit_hint
                elif check.verdict == MOSTLY and check.tolerance is None:  # a text or a code
                    why = f"{errors}, or another definition (a cut-off)"
                else:
                    why = errors
                lines.append(f"{name}: its formula {evidence}; check {rows} ({why}).")
            if check.usable and check.fillable:
                lines.append(
                    f"{name}: its formula can compute {check.fillable:,} of its missing values (Fill exactly)."
                )
        if not lines:
            return None
        return {
            "category": DERIVED_CATEGORY,
            "message": "The declared derived variables, checked against their formulas:\n\n"
            + "\n".join(f"- {line}" for line in lines),
            "severity": severity,
        }
