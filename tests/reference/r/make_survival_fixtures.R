# Reference values for biostats.survival.km_quantile and the median's interval, from R's survival::survfit with a
# log-log band (conf.type = "log-log", as biostats.survival.kaplan_meier's) and quantile.survfit: the quartiles and
# their limits, on curves that fall through 0.5, stay at exactly 0.5 for a stretch (to the last time, or not), never
# reach it, and the 6-MP arm of the Gehan leukaemia data.
#
# Usage: Rscript tests/reference/r/make_survival_fixtures.R <output directory>
# Needs survival and jsonlite. Run by the manual workflow .github/workflows/r-fixtures.yml; the values are compared
# by tests/biostats_layer/test_survival_r_parity.py.

library(survival)

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) >= 1) args[[1]] else "tests/reference/fixtures/survival"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

cases <- list(
  every_event = list(time = 1:10, event = rep(1, 10)),
  flat_to_the_end = list(time = c(1, 2, 3, 4, 5, 5, 5, 5), event = c(1, 1, 1, 1, 0, 0, 0, 0)),
  flat_then_falls = list(time = c(1, 2, 3, 4, 6, 7, 8, 9), event = c(1, 1, 1, 1, 0, 1, 1, 1)),
  not_reached = list(time = c(1, 2, 3, 4), event = c(1, 0, 0, 0)),
  ties_and_censoring = list(
    time = c(2, 3, 3, 5, 5, 5, 8, 9, 9, 12, 14, 14),
    event = c(1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0)
  ),
  gehan_6mp = list(
    time = c(6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32, 32, 34, 35),
    event = c(1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0)
  )
)

result <- lapply(cases, function(k) {
  fit <- survfit(Surv(k$time, k$event) ~ 1, conf.type = "log-log")
  q <- quantile(fit, probs = c(0.25, 0.5, 0.75))
  list(
    time = k$time, event = k$event, probs = c(0.25, 0.5, 0.75),
    quantile = unname(q$quantile), lower = unname(q$lower), upper = unname(q$upper)
  )
})

jsonlite::write_json(
  list(generated_by = R.version.string, cases = result),
  file.path(out, "km_quantiles.json"),
  auto_unbox = TRUE, digits = NA, na = "null", pretty = TRUE
)
cat("Wrote", file.path(out, "km_quantiles.json"), "\n")
