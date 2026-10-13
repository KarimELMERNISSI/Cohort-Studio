"""Power semantics (audit S2), reference Z-scores (S5) and the full-family missingness scan (S3)."""

import numpy as np
import pandas as pd
import pytest

from biostats.missing import scan_missingness_dependencies
from biostats.power import (
    ARE_NORMAL,
    anova_power,
    anova_sample_size,
    chi2_degrees_of_freedom,
    chi2_independence_power,
    chi2_independence_sample_size,
    kruskal_power,
    kruskal_sample_size,
)
from biostats.zscores import reference_zscores


class TestPower:
    def test_anova_total_versus_per_group(self) -> None:
        res = anova_sample_size(0.25, 3, alpha=0.05, power=0.8)
        assert res.n_total == pytest.approx(157.19, abs=0.1)  # statsmodels nobs is the TOTAL
        assert res.n_per_group == 53
        assert "TOTAL" in res.method

    def test_anova_power_per_group(self) -> None:
        res = anova_power(0.25, 30, 3, alpha=0.05)
        assert res.n_total == 90
        assert res.power == pytest.approx(
            0.5396, abs=2e-3
        )  # the page reported 0.195 before the fix

    def test_kruskal_are_scaling(self) -> None:
        res = kruskal_sample_size(0.25, 3, alpha=0.05, power=0.8, are=ARE_NORMAL)
        assert res.n_total == pytest.approx(157.19 / 0.955, abs=0.2)
        assert res.n_per_group == 55
        assert kruskal_power(0.25, 30, 3).power < anova_power(0.25, 30, 3).power

    def test_chi2_df_semantics(self) -> None:
        assert chi2_degrees_of_freedom(2, 2) == 1 and chi2_degrees_of_freedom(3, 2) == 2
        res = chi2_independence_sample_size(0.3, df=1, alpha=0.05, power=0.8)
        assert res.n_total == pytest.approx(87.2, abs=0.2)  # Cohen (1988) table: 88
        assert chi2_independence_power(0.3, 88, df=1).power == pytest.approx(0.8035, abs=2e-3)
        assert chi2_independence_power(0.3, 108, df=2).power == pytest.approx(0.8037, abs=2e-3)


class TestZScores:
    def test_control_group_coded_zero(self) -> None:
        df = pd.DataFrame({"grp": [0, 0, 0, 1, 1, 1], "x": [10, 12, 14, 20, 22, 24]})
        out, stats = reference_zscores(df, ["x"], "control", ref_col="grp", ref_value=0)
        assert stats["x"].mean == 12 and stats["x"].std == 2 and stats["x"].n == 3
        assert out.loc[3, "x_Z"] == pytest.approx(4.0)  # was 20.0 (raw value) before the fix

    def test_manual_reference(self) -> None:
        df = pd.DataFrame({"x": [130.0, 85.0]})
        out, stats = reference_zscores(df, ["x"], "manual", manual={"x": (100.0, 15.0)})
        assert out["x_Z"].tolist() == pytest.approx([2.0, -1.0])
        assert stats["x"].source == "manual reference values"

    def test_cohort_and_errors(self) -> None:
        df = pd.DataFrame({"grp": ["a", "b"], "x": [1.0, 3.0]})
        out, _ = reference_zscores(df, ["x"], "cohort")
        assert out["x_Z"].abs().max() == pytest.approx(np.sqrt(0.5))
        with pytest.raises(ValueError):
            reference_zscores(df, ["x"], "control", ref_col="grp", ref_value=None)
        with pytest.raises(ValueError, match="No rows"):
            reference_zscores(df, ["x"], "control", ref_col="grp", ref_value="zzz")
        with pytest.raises(ValueError, match="missing"):
            reference_zscores(df, ["x"], "manual", manual={})
        with pytest.raises(ValueError, match="> 0"):
            reference_zscores(df, ["x"], "manual", manual={"x": (0.0, 0.0)})


class TestMissingnessScan:
    @pytest.fixture
    def mar_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        n = 300
        df = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(1, 11)})
        lab = rng.normal(size=n)
        lab[df["x1"] > 0.5] = np.nan  # missing when x1 is high (MAR driver)
        df["lab"] = lab
        return df

    def test_every_test_is_returned_and_corrected_over_the_full_family(
        self, mar_df: pd.DataFrame
    ) -> None:
        result = scan_missingness_dependencies(mar_df, targets=["lab"], correction="fdr_bh")
        assert (
            result.n_tests == 10 and len(result.table) == 10
        )  # ten predictors, all complete: no co-occurrence tests
        assert set(result.table["Dependency Type"]) == {"Value association"}
        best = result.table.iloc[0]
        assert best["Predictor Variable"] == "x1" and best["Significant"]
        assert (result.table["P-Value (Adj)"] >= result.table["P-Value"]).all()
        assert result.n_significant >= 1 and "association" in best["Conclusion"]

    def test_no_correction_keeps_raw_pvalues(self, mar_df: pd.DataFrame) -> None:
        result = scan_missingness_dependencies(mar_df, targets=["lab"], correction="none")
        assert (result.table["P-Value (Adj)"] == result.table["P-Value"]).all()

    def test_legacy_shim_returns_all_tests(self, mar_df: pd.DataFrame) -> None:
        from utils.analysis_utils import scan_missingness_dependencies as legacy

        table = legacy(
            mar_df,
            target_cols=["lab"],
            predictor_cols=[f"x{i}" for i in range(1, 11)],
            test_preference="Auto-Detect",
        )
        assert (
            len(table) == 10 and "P-Value" in table.columns and "P-Value (Adj)" not in table.columns
        )
        assert table.attrs["n_tests"] == 10

    def test_small_groups_are_skipped_not_reported(self) -> None:
        df = pd.DataFrame({"x": range(30), "lab": [np.nan] * 2 + list(range(28))})
        result = scan_missingness_dependencies(df, targets=["lab"])
        assert result.n_tests == 0 and result.n_skipped >= 1 and result.table.empty
