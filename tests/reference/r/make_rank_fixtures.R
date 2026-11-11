# Reference values for biostats.rank_intervals, from R's stats::wilcox.test(conf.int = TRUE): the Hodges-Lehmann
# shift of two groups and the pseudo-median of one sample or of paired differences, each with its interval.
#
# Each case holds what R decided, not what this script derives again: the method R chose (exact, or a normal
# approximation with a continuity correction), the level its interval holds (R lowers it when the level asked for
# cannot be reached), its p-value and its warnings. It also holds the two middle pairs (pairwise differences, or
# Walsh averages) from R's own sort: in its normal branch R finds its estimate by a root search, which stops
# anywhere between them when the number of pairs is even. Numbers are written as text with 17 significant digits,
# so that both sides rank the same numbers; a paired case gives x and y, never their differences.
#
# Usage: Rscript tests/reference/r/make_rank_fixtures.R <output directory>
# Needs the jsonlite package. Run by the manual workflow .github/workflows/r-fixtures.yml; the values are compared
# by tests/biostats_layer/test_rank_intervals_r_parity.py.

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) >= 1) args[[1]] else "tests/reference/fixtures/rank"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

num <- function(v) {
  v <- as.numeric(v)
  ifelse(is.na(v), NA_character_, sprintf("%.17g", v))
}

# wilcox.test with its interval, and the warnings it gave (collected, not suppressed)
run <- function(...) {
  warned <- character(0)
  res <- withCallingHandlers(
    wilcox.test(..., conf.int = TRUE),
    warning = function(w) {
      warned <<- c(warned, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(res = res, warnings = warned)
}

# the one or two middle values, whose mean is the median
middle <- function(v) {
  v <- sort(v)
  n <- length(v)
  c(v[(n + 1) %/% 2], v[n %/% 2 + 1])
}

decided <- function(r, level) {
  res <- r$res
  list(
    conf_level = level,
    estimate = num(unname(res$estimate)),
    lower = num(res$conf.int[1]),
    upper = num(res$conf.int[2]),
    achieved_level = as.numeric(attr(res$conf.int, "conf.level")),
    method = res$method,
    statistic = num(unname(res$statistic)),
    p_value = num(res$p.value),
    warnings = as.list(r$warnings)
  )
}

two <- function(name, x, y, level = 0.95) {
  r <- run(x, y, conf.level = level)
  d <- as.vector(outer(x, y, "-"))
  c(
    list(name = name, x = num(x), y = num(y), pairs = length(d), middle = num(middle(d))),
    decided(r, level)
  )
}

# one sample against a reference (mu), or paired values (y given): the pairs are the Walsh averages of the
# differences that are not zero, as R takes them
one <- function(name, x, y = NULL, mu = 0, level = 0.95) {
  r <- if (is.null(y)) run(x, mu = mu, conf.level = level) else run(x, y, paired = TRUE, conf.level = level)
  d <- if (is.null(y)) x - mu else x - y
  nz <- d[d != 0]
  w <- outer(nz, nz, "+") / 2
  w <- w[!lower.tri(w)]
  base <- list(
    name = name, x = num(x), mu = num(mu), differences = length(d), nonzero = length(nz),
    pairs = length(w), middle = num(middle(w) + mu)
  )
  if (!is.null(y)) base$y <- num(y)
  c(base, decided(r, level))
}

set.seed(20261111)

# ---------------------------------------------------------------------------- two groups
permeability_x <- c(0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46)
permeability_y <- c(1.15, 0.88, 0.90, 0.74, 1.21)
small_x <- round(rlnorm(12), 4)
small_y <- round(rlnorm(9, 0.5), 4)
mid_x <- round(rnorm(49, 0, 1), 5)
mid_y <- round(rnorm(31, 0.4, 2), 5)
whole_x <- round(rnorm(25, 120, 15))
whole_odd <- round(rnorm(31, 128, 15))
whole_even <- round(rnorm(30, 128, 15))
large_x <- round(rlnorm(60), 4)
large_y <- round(rlnorm(80, 0.3), 4)
larger_x <- round(rnorm(400, 0, 1), 3)
larger_y <- round(rnorm(350, 0.1, 1.5), 3)
border_49 <- round(rnorm(49), 6)
border_49b <- round(rnorm(49, 0.3), 6)
border_50 <- round(rnorm(50, 0.3), 6)
five <- round(rnorm(5), 6)
sixty <- round(rnorm(60, 0.8), 6)
counted_x <- round(rnorm(1500), 3)
counted_y <- round(rnorm(1500, 0.05), 3)

oj <- ToothGrowth$len[ToothGrowth$supp == "OJ"]
vc <- ToothGrowth$len[ToothGrowth$supp == "VC"]
may <- airquality$Ozone[airquality$Month == 5]
may <- may[!is.na(may)]
august <- airquality$Ozone[airquality$Month == 8]
august <- august[!is.na(august)]

two_sample <- list(
  # Hollander & Wolfe: permeability constants (the example of R's help page)
  two("permeability constants (Hollander and Wolfe)", permeability_x, permeability_y),
  two("12 and 9, no ties", small_x, small_y),
  two("49 and 31, no ties", mid_x, mid_y),
  # ties (whole numbers): the normal approximation, with an odd and with an even number of pairs
  two("whole numbers, 25 and 31 (775 pairs)", whole_x, whole_odd),
  two("whole numbers, 25 and 30 (750 pairs)", whole_x, whole_even),
  # one group of 50 or more: the normal approximation
  two("60 and 80", large_x, large_y),
  two("400 and 350", larger_x, larger_y),
  # R's own datasets
  two("ToothGrowth, len, OJ against VC", oj, vc),
  two("mtcars, mpg, automatic against manual", mtcars$mpg[mtcars$am == 0], mtcars$mpg[mtcars$am == 1]),
  two("airquality, Ozone, May against August", may, august),
  # the edges of the exact rule: groups of 3 and 3 or 3 and 4 cannot hold 95%
  two("3 and 3", c(1.1, 2.5, 4.2), c(0.5, 3.3, 6.1)),
  two("3 and 4", c(1.1, 2.5, 4.2), c(0.5, 3.3, 6.1, 7.4)),
  two("3 and 5", c(1.1, 2.5, 4.2), c(0.5, 3.3, 6.1, 7.4, 9.8)),
  two("4 and 4", c(1.1, 2.5, 4.2, 5.7), c(0.5, 3.3, 6.1, 7.4)),
  two("3 and 4, tied", c(1, 1, 2), c(3, 3, 3, 4)),
  # the border of 50
  two("49 and 49, no ties", border_49, border_49b),
  two("49 and 50", border_49, border_50),
  two("5 and 60", five, sixty),
  # ties of each kind apart
  two("ties within one group only", c(1.5, 1.5, 2.5, 4.5, 6.5, 8.5, 9.5, 9.5), c(0.2, 3.1, 5.3, 7.7, 10.4, 12.6)),
  two("ties across the two groups only", c(1, 3, 5, 7, 9, 11, 13), c(2, 3, 6, 7, 10, 12, 14, 16)),
  # other levels
  two("12 and 9, no ties, at 90%", small_x, small_y, level = 0.90),
  two("12 and 9, no ties, at 99%", small_x, small_y, level = 0.99),
  two("whole numbers, 25 and 31, at 90%", whole_x, whole_odd, level = 0.90),
  two("whole numbers, 25 and 31, at 99%", whole_x, whole_odd, level = 0.99),
  # more than 2,000,000 pairs: found here by counting, not by writing them out
  two("1,500 and 1,500 (2,250,000 pairs)", counted_x, counted_y)
)

# ---------------------------------------------------------------------------- one sample, paired values
before <- c(1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30)
after <- c(0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29)
t15 <- round(rt(15, 3) + 0.5, 4)
skewed40 <- round(rlnorm(40) - 1, 5)
whole30 <- round(rnorm(30, -4, 8))
half120 <- round(rnorm(120, -5.5, 12) * 2) / 2
untied49 <- round(rnorm(49, 0.2), 6)
untied50 <- round(rnorm(50, 0.2), 6)
counted_d <- round(rnorm(2100, 0.02), 3)
tied <- c(2, 2, 3, 3, 5, 6, 7, 8, 9, 11)

one_sample <- c(
  list(
    # Hollander & Wolfe: depression scores before and after (the example of R's help page)
    one("depression scores, paired (Hollander and Wolfe)", before, after),
    one("15 values, no ties", t15),
    one("40 values, no ties", skewed40),
    # ties and zeros (whole numbers): the normal approximation, on the differences that are not zero
    one("whole numbers, 30 values", whole30),
    one("half numbers, 120 values", half120),
    # R's own dataset, paired: one pair without a difference
    one("ToothGrowth, len, OJ against VC, paired", oj, vc),
    # the edges of the exact rule: 5 differences cannot hold 95%
    one("5 values, no ties", c(0.7, -1.3, 2.1, 3.4, 4.6)),
    one("6 values, no ties", c(0.7, -1.3, 2.1, 3.4, 4.6, 5.9)),
    # the border of 50
    one("49 values, no ties", untied49),
    one("50 values, no ties", untied50),
    # ties in the absolute values only: R leaves the exact branch, and the variance has no tie term
    one("ties in the absolute values only", c(-2, 2, 3, 5, 7, 11, 13))
  ),
  # 5 to 10 tied values: R lowers its level for the smallest
  lapply(5:10, function(n) one(sprintf("%d tied values", n), tied[1:n])),
  list(
    # other levels, another reference
    one("15 values, no ties, at 90%", t15, level = 0.90),
    one("15 values, no ties, at 99%", t15, level = 0.99),
    one("whole numbers, 30 values, at 90%", whole30, level = 0.90),
    one("whole numbers, 30 values, at 99%", whole30, level = 0.99),
    one("15 values, no ties, against 0.5", t15, mu = 0.5),
    one("whole numbers, 30 values, against -4", whole30, mu = -4),
    # more than 2,000,000 Walsh averages: found here by counting
    one("2,100 values (2,206,050 pairs at most)", counted_d)
  )
)

jsonlite::write_json(
  list(generated_by = R.version.string, two_sample = two_sample, one_sample = one_sample),
  file.path(out, "wilcox_reference.json"),
  auto_unbox = TRUE, digits = NA, na = "null", pretty = TRUE
)
cat("Wrote", file.path(out, "wilcox_reference.json"), "\n")
