"""Method references cited next to every result ("Method: ...; implementation: ...")."""

from __future__ import annotations

REFERENCES: dict[str, str] = {
    "welch1947": "Welch BL (1947). The generalization of 'Student's' problem when several different population variances are involved. Biometrika 34:28-35.",
    "delacre2017": "Delacre M, Lakens D, Leys C (2017). Why psychologists should by default use Welch's t-test instead of Student's t-test. Int Rev Soc Psychol 30:92-101.",
    "welch1951": "Welch BL (1951). On the comparison of several mean values: an alternative approach. Biometrika 38:330-336.",
    "games_howell1976": "Games PA, Howell JF (1976). Pairwise multiple comparison procedures with unequal n's and/or variances. J Educ Stat 1:113-125.",
    "tukey1949": "Tukey JW (1949). Comparing individual means in the analysis of variance. Biometrics 5:99-114.",
    "dunn1964": "Dunn OJ (1964). Multiple comparisons using rank sums. Technometrics 6:241-252.",
    "kruskal1952": "Kruskal WH, Wallis WA (1952). Use of ranks in one-criterion variance analysis. JASA 47:583-621.",
    "mann1947": "Mann HB, Whitney DR (1947). On a test of whether one of two random variables is stochastically larger than the other. Ann Math Stat 18:50-60.",
    "hedges1981": "Hedges LV (1981). Distribution theory for Glass's estimator of effect size and related estimators. J Educ Stat 6:107-128.",
    "steiger2004": "Steiger JH (2004). Beyond the F test: effect size confidence intervals and tests of close fit. Psychol Methods 9:164-182.",
    "vargha2000": "Vargha A, Delaney HD (2000). A critique and improvement of the CL common language effect size statistics. J Educ Behav Stat 25:101-132.",
    "kerby2014": "Kerby DS (2014). The simple difference formula: an approach to teaching nonparametric correlation. Compr Psychol 3:11.",
    "bergsma2013": "Bergsma W (2013). A bias-correction for Cramer's V and Tschuprow's T. J Korean Stat Soc 42:323-328.",
    "tomczak2014": "Tomczak M, Tomczak E (2014). The need to report effect size estimates revisited. Trends Sport Sci 1:19-25.",
    "lilliefors1967": "Lilliefors HW (1967). On the Kolmogorov-Smirnov test for normality with mean and variance unknown. JASA 62:399-402.",
    "shapiro1965": "Shapiro SS, Wilk MB (1965). An analysis of variance test for normality. Biometrika 52:591-611.",
    "levene1960": "Levene H (1960). Robust tests for equality of variances. In: Contributions to Probability and Statistics. Stanford UP, 278-292.",
    "little1988": "Little RJA (1988). A test of missing completely at random for multivariate data with missing values. JASA 83:1198-1202.",
    "dempster1977": "Dempster AP, Laird NM, Rubin DB (1977). Maximum likelihood from incomplete data via the EM algorithm. JRSS B 39:1-38.",
    "delong1988": "DeLong ER, DeLong DM, Clarke-Pearson DL (1988). Comparing the areas under two or more correlated ROC curves. Biometrics 44:837-845.",
    "sun2014": "Sun X, Xu W (2014). Fast implementation of DeLong's algorithm for comparing the areas under correlated ROC curves. IEEE Signal Process Lett 21:1389-1393.",
    "simel1991": "Simel DL, Samsa GP, Matchar DB (1991). Likelihood ratios with confidence: sample size estimation for diagnostic test studies. J Clin Epidemiol 44:763-770.",
    "wilson1927": "Wilson EB (1927). Probable inference, the law of succession, and statistical inference. JASA 22:209-212.",
    "youden1950": "Youden WJ (1950). Index for rating diagnostic tests. Cancer 3:32-35.",
    "bland1986": "Bland JM, Altman DG (1986). Statistical methods for assessing agreement between two methods of clinical measurement. Lancet 327:307-310.",
    "bland1999": "Bland JM, Altman DG (1999). Measuring agreement in method comparison studies. Stat Methods Med Res 8:135-160.",
    "bonett2000": "Bonett DG, Wright TA (2000). Sample size requirements for estimating Pearson, Kendall and Spearman correlations. Psychometrika 65:23-28.",
    "fisher1915": "Fisher RA (1915). Frequency distribution of the values of the correlation coefficient. Biometrika 10:507-521.",
    "holm1979": "Holm S (1979). A simple sequentially rejective multiple test procedure. Scand J Stat 6:65-70.",
    "benjamini1995": "Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate. JRSS B 57:289-300.",
    "iglewicz1993": "Iglewicz B, Hoaglin DC (1993). How to detect and handle outliers. ASQC Quality Press.",
    "schober2018": "Schober P, Boer C, Schwarte LA (2018). Correlation coefficients: appropriate use and interpretation. Anesth Analg 126:1763-1768.",
    "hodges1963": "Hodges JL, Lehmann EL (1963). Estimates of location based on rank tests. Ann Math Stat 34:598-611.",
    "wasserstein2016": "Wasserstein RL, Lazar NA (2016). The ASA statement on p-values: context, process, and purpose. Am Stat 70:129-133.",
}


def cite(*keys: str) -> list[str]:
    """Reference strings for the given keys (unknown keys are returned verbatim so nothing is silently lost)."""
    return [REFERENCES.get(k, k) for k in keys]


__all__ = ["REFERENCES", "cite"]
