"""
Unit tests for DataQualityAuditor (Multi-dimensional Data Quality & Anomaly Profiling).
"""

import numpy as np
import pandas as pd
import pytest

from explore.data_quality_auditor import DataQualityAuditor


@pytest.fixture
def clean_cohort_df() -> pd.DataFrame:
    """Fixture providing a perfectly clean clinical dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "patient_id": [f"ID_{i:04d}" for i in range(n)],
        "age": np.random.randint(30, 75, size=n).astype(float),
        "sbp": np.random.normal(120, 10, size=n),
        "gender": np.random.choice(["Male", "Female"], size=n),
    })


@pytest.fixture
def dirty_cohort_df() -> pd.DataFrame:
    """Fixture providing a dataset with missing values, duplicates, outliers, and formatting issues."""
    data = {
        "id": ["P1", "P2", "P3", "P4", "P4"],  # 1 duplicate row
        "age": [25.0, np.nan, 30.0, 45.0, 45.0],  # 1 missing value
        "glucose": [90.0, 95.0, 92.0, 999.0, 999.0],  # extreme outlier
        "country": ["France", " France", "italy", "Italy", "Italy"],  # whitespace & case inconsistency
    }
    return pd.DataFrame(data)


class TestDataQualityAuditor:
    """Test suite for DataQualityAuditor quality dimensions and audit engine."""

    def test_completeness_perfect_and_missing(self, clean_cohort_df: pd.DataFrame, dirty_cohort_df: pd.DataFrame) -> None:
        """Completeness score reflects ratio of non-null cells."""
        auditor_clean = DataQualityAuditor(clean_cohort_df)
        assert auditor_clean.compute_completeness() == 100.0

        auditor_dirty = DataQualityAuditor(dirty_cohort_df)
        # 20 total cells, 1 NaN -> (1 - 1/20) * 100 = 95.0%
        assert auditor_dirty.compute_completeness() == 95.0

        empty_df = pd.DataFrame()
        assert DataQualityAuditor(empty_df).compute_completeness() == 0.0

    def test_uniqueness_calculation(self, clean_cohort_df: pd.DataFrame, dirty_cohort_df: pd.DataFrame) -> None:
        """Uniqueness score calculates percentage of distinct rows."""
        auditor_clean = DataQualityAuditor(clean_cohort_df)
        assert auditor_clean.compute_uniqueness() == 100.0

        auditor_dirty = DataQualityAuditor(dirty_cohort_df)
        # 5 rows total, 4 unique -> 4/5 * 100 = 80.0%
        assert auditor_dirty.compute_uniqueness() == 80.0

    def test_validity_outlier_detection_methods(self) -> None:
        """Evaluates statistical outlier detection algorithms (IQR, Z-score, Quantile)."""
        # Create a sample with 30 normal observations and 1 extreme outlier
        np.random.seed(42)
        values = list(np.random.normal(100.0, 5.0, size=29)) + [9999.0]
        outlier_df = pd.DataFrame({"marker": values})
        auditor = DataQualityAuditor(outlier_df)

        # IQR method detects 9999.0 outlier
        iqr_score = auditor.compute_validity(method="iqr", params={"multiplier": 1.5})
        assert iqr_score < 100.0

        # Z-score method
        z_score = auditor.compute_validity(method="zscore", params={"threshold": 2.0})
        assert z_score < 100.0

        # Quantile method
        q_score = auditor.compute_validity(method="quantile", params={"lower": 0.05, "upper": 0.95})
        assert q_score < 100.0

    def test_validity_constant_data_edge_case(self) -> None:
        """Constant data has IQR=0 and std=0; should not falsely flag values as outliers."""
        constant_df = pd.DataFrame({"col_a": [10.0] * 50, "col_b": [25.0] * 50})
        auditor = DataQualityAuditor(constant_df)

        assert auditor.compute_validity(method="iqr") == 100.0
        assert auditor.compute_validity(method="zscore") == 100.0

    def test_validity_multivariate_methods(self, clean_cohort_df: pd.DataFrame) -> None:
        """Evaluates Isolation Forest and Local Outlier Factor without crashes."""
        auditor = DataQualityAuditor(clean_cohort_df)
        if_score = auditor.compute_validity(method="Isolation Forest")
        assert 0.0 <= if_score <= 100.0

        lof_score = auditor.compute_validity(method="Local Outlier Factor")
        assert 0.0 <= lof_score <= 100.0

    def test_consistency_check(self) -> None:
        """Consistency identifies text/object columns that could be numeric or datetime."""
        clean_df = pd.DataFrame({"names": ["Alice", "Bob", "Charlie"]})
        assert DataQualityAuditor(clean_df).compute_consistency() == 100.0

        inconsistent_df = pd.DataFrame({
            "numeric_as_text": ["10", "20", "30"],
            "legit_text": ["A", "B", "C"]
        })
        auditor = DataQualityAuditor(inconsistent_df)
        # 1 of 2 text columns is mis-typed -> 50.0%
        assert auditor.compute_consistency() == 50.0

    def test_uniformity_whitespace_and_capitalization(self, dirty_cohort_df: pd.DataFrame) -> None:
        """Uniformity flags trailing whitespace and casing mismatches."""
        auditor = DataQualityAuditor(dirty_cohort_df)
        uniformity = auditor.compute_uniformity()
        assert uniformity < 100.0
        assert len(auditor.uniformity_details) > 0
        issues = [d["Issues"] for d in auditor.uniformity_details]
        assert any("Whitespace" in i or "Capitalization" in i for i in issues)

    def test_clinical_validity_with_and_without_rules(self) -> None:
        """Clinical validity evaluates custom anomaly rules when provided in config."""
        df = pd.DataFrame({
            "sbp": [120, 130, 240, 115],  # 240 is clinically abnormal (> 200)
            "dbp": [80, 85, 130, 78],
        })

        # No config -> returns None (N/A)
        auditor_no_config = DataQualityAuditor(df)
        assert auditor_no_config.compute_clinical_validity() is None

        # Config with clinical anomaly rule formatted as expression
        config = {
            "mask_families": {
                "clinical_anomalies": {
                    "hypertensive_crisis": {
                        "expression": "df['sbp'] > 200"
                    }
                }
            }
        }
        auditor_with_rules = DataQualityAuditor(df, config=config)
        score = auditor_with_rules.compute_clinical_validity()
        # 1 out of 4 rows triggers anomaly -> 75.0%
        assert score == 75.0
        assert auditor_with_rules.clinical_anomalies_df is not None
        assert len(auditor_with_rules.clinical_anomalies_df) == 1

    def test_run_audit_and_generate_advice(self, dirty_cohort_df: pd.DataFrame) -> None:
        """Full audit run compiles all dimension scores and generates actionable advice."""
        auditor = DataQualityAuditor(dirty_cohort_df)
        metrics = auditor.run_audit()

        assert "Completeness" in metrics
        assert "Uniqueness" in metrics
        assert "Statistical Validity" in metrics
        assert "Consistency" in metrics
        assert "Uniformity" in metrics

        assert len(auditor.advice) > 0
        categories = [a["category"] for a in auditor.advice]
        assert "Completeness" in categories or "Uniqueness" in categories

    def test_mcar_missingness_test(self) -> None:
        """Evaluates heuristic Little's MCAR missingness analysis."""
        df_no_missing = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        res = DataQualityAuditor(df_no_missing).perform_mcar_test()
        assert res["is_mcar"] is True
        assert "No missing data" in res["interpretation"]


def _bmi_frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    weight = np.round(rng.normal(75, 12, n), 1)
    height = np.round(rng.normal(170, 9, n), 0)
    return pd.DataFrame(
        {"weight_kg": weight, "height_cm": height, "bmi": np.round(weight / (height / 100) ** 2, 1)}
    )


def _derived(*definitions) -> dict:
    from core.derived import with_definitions

    return with_definitions({}, definitions)


def test_the_declared_derived_variables_are_advised_on_without_changing_the_score() -> None:
    """Computational conformance (Kahn et al. 2016): the recorded values a formula does not give, and the missing
    values it can compute; names and counts only."""
    from core.derived import DerivedVariable

    frame = _bmi_frame()
    frame.loc[:4, "bmi"] = np.nan  # five BMIs the formula can compute
    frame.loc[10, "bmi"] = 77.7123  # an entry error
    bmi = DerivedVariable("bmi", "weight_kg / (height_cm / 100) ** 2")
    plain = DataQualityAuditor(frame.copy())
    plain.run_audit()
    audited = DataQualityAuditor(frame.copy(), config=_derived(bmi))
    assert audited.run_audit() == plain.metrics  # advice only: the score does not change
    assert not any(a["category"] == "Derived variables" for a in plain.advice)
    advice = next(a for a in audited.advice if a["category"] == "Derived variables")
    message = advice["message"]
    assert "gives the recorded value in 114 of 115 rows" in message
    assert "check the other row (a probable entry error)" in message
    assert "can compute 5 of its missing values" in message
    assert "77.7" not in message and advice["severity"] == "low"
    # an imputed cell is not compared: the error, filled by an imputation, is no longer counted
    imputed = DataQualityAuditor(frame.copy(), config=_derived(bmi), imputed={"bmi": [10]})
    imputed.run_audit()
    advice = next(a for a in imputed.advice if a["category"] == "Derived variables")
    assert "check the" not in advice["message"]


def test_a_formula_that_cannot_be_used_or_disagrees_is_named() -> None:
    from core.derived import DerivedVariable

    frame = _bmi_frame()
    audited = DataQualityAuditor(
        frame,
        config=_derived(
            DerivedVariable("bmi", "weight_kg / height_m ** 2"),  # no such column
            DerivedVariable("weight_kg", "bmi * (height_cm / 10) ** 2"),  # a factor of 100 out
        ),
    )
    audited.run_audit()
    advice = next(a for a in audited.advice if a["category"] == "Derived variables")
    assert "`bmi`: its formula cannot be used on these data." in advice["message"]
    assert "`weight_kg`: the data disagree with its formula" in advice["message"]
    assert advice["severity"] == "medium"
    # a formula the data follow everywhere, with nothing to compute: nothing to say
    agreeing = DataQualityAuditor(
        frame, config=_derived(DerivedVariable("bmi", "weight_kg / (height_cm / 100) ** 2"))
    )
    agreeing.run_audit()
    assert not any(a["category"] == "Derived variables" for a in agreeing.advice)


def _advice(frame: pd.DataFrame, *definitions) -> str:
    auditor = DataQualityAuditor(frame, config=_derived(*definitions))
    auditor.run_audit()
    return next(a for a in auditor.advice if a["category"] == "Derived variables")["message"]


def test_a_check_s_reason_never_reaches_the_advice() -> None:
    """A check's reason can quote a value of the data (a town, a level): the advice says what it is instead."""
    from core.derived import DerivedVariable
    from enrich.derived import CANNOT, DISAGREES, check_derived

    frame = _bmi_frame(200)
    frame["site"] = np.where(np.arange(200) % 2, "Esch-sur-Alzette", "Ettelbruck")
    reads_text = DerivedVariable("bmi", "weight_kg / site")
    assert check_derived(reads_text, frame).verdict == CANNOT
    assert "Esch-sur-Alzette" in (check_derived(reads_text, frame).reason or "")
    # a factor wrong for one sex only: it agrees overall, not for that level
    frame["sex"] = np.where(np.arange(200) < 16, 1, 2)
    frame["weight_adj"] = np.round(frame["weight_kg"] * np.where(frame["sex"] == 1, 0.9, 1.0), 1)
    by_sex = DerivedVariable("weight_adj", "np.where(sex == 1, 1.0, 1.0) * weight_kg")
    check = check_derived(by_sex, frame)
    assert check.verdict == DISAGREES and check.level_of == "sex" and "sex = 1" in (check.reason or "")
    message = _advice(frame, reads_text, by_sex)
    assert "`bmi`: its formula cannot be used on these data." in message
    assert "for one of the values of `sex`, although it gives the recorded value in 184 of 200 rows" in message
    assert "Esch" not in message and "Ettelbruck" not in message and "sex = 1" not in message


def test_too_few_rows_and_a_constant_variable_are_told_apart() -> None:
    from core.derived import DerivedVariable

    few = _advice(_bmi_frame(40), DerivedVariable("bmi", "weight_kg / (height_cm / 100) ** 2"))
    assert "too few rows have it and every variable of its formula recorded to check it (40; the check needs 60)" in few
    frame = _bmi_frame(120).assign(heavy=0)
    constant = _advice(frame, DerivedVariable("heavy", "np.where(weight_kg > 500, 1, 0)"))
    assert "`heavy`: it takes one value in nearly every row, so the check cannot tell its formula from chance." in constant


def test_a_name_is_quoted_as_code_and_cannot_become_a_link() -> None:
    """A variable named in an uploaded configuration is the user's text: on a page that renders markdown, as code."""
    from core.derived import DerivedVariable

    message = _advice(_bmi_frame(), DerivedVariable("[x](https://example.org)", "weight_kg * 2"))
    assert "`[x](https://example.org)`: its formula cannot be used on these data." in message
    assert message.count("](https://example.org)") == 1  # only inside the code span
