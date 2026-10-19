# Reference values for biostats.rates, from R's stats::poisson.test: the exact interval of one rate and the exact
# conditional interval of a rate ratio. poisson.test reports a minimum-likelihood p-value, so the central one
# (twice the smaller binomial tail, the p-value biostats.rates reports) is computed here as well.
#
# Usage: Rscript tests/reference/r/make_rates_fixtures.R <output directory>
# Needs the jsonlite package. Run by the manual workflow .github/workflows/r-fixtures.yml; the values are compared
# by tests/biostats_layer/test_rates_r_parity.py.

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) >= 1) args[[1]] else "tests/reference/fixtures/rates"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

# events and person-time: exposed, then reference
ratio_cases <- list(
  c(15, 1200, 4, 1000),
  c(11, 800, 21, 3011),
  c(0, 500, 7, 900),
  c(9, 250, 0, 400),
  c(120, 10500.5, 95, 11200),
  c(3, 100, 3, 100)
)
rate_ratio <- lapply(ratio_cases, function(k) {
  res <- poisson.test(c(k[1], k[3]), c(k[2], k[4]))
  n <- k[1] + k[3]
  null <- k[2] / (k[2] + k[4])
  central <- min(1, 2 * min(pbinom(k[1], n, null), pbinom(k[1] - 1, n, null, lower.tail = FALSE)))
  list(
    d1 = k[1], t1 = k[2], d0 = k[3], t0 = k[4],
    ratio = unname(res$estimate), lower = res$conf.int[1], upper = res$conf.int[2],
    p_minlike = res$p.value, p_central = central
  )
})

rate_cases <- list(c(15, 1200), c(0, 300), c(4, 1000), c(250, 18000.25))
rate <- lapply(rate_cases, function(k) {
  res <- poisson.test(k[1], k[2])
  list(events = k[1], time = k[2], lower = res$conf.int[1], upper = res$conf.int[2])
})

jsonlite::write_json(
  list(generated_by = R.version.string, rate_ratio = rate_ratio, rate = rate),
  file.path(out, "poisson_reference.json"),
  auto_unbox = TRUE, digits = NA, na = "string", pretty = TRUE
)
cat("Wrote", file.path(out, "poisson_reference.json"), "\n")
