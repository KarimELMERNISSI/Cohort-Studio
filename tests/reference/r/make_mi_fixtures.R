#!/usr/bin/env Rscript
# Reference values from R mice / mitml for the multiple-imputation layer (biostats/mi).
#
# Run by .github/workflows/r-fixtures.yml (manual dispatch); no local R is needed.
# Writes, in the output folder (default tests/reference/fixtures/mi):
#   mice_reference.json      pool.scalar cases, pooled model tables, D1 and D2 tests, pooled R2, versions
#   mice_completed_long.csv  the m datasets completed by mice (complete(imp, "long")) that the models use
# The Python tests (tests/biostats_layer/test_mi_r_parity.py) fit the same models on the same completed
# datasets and compare the pooled results.

suppressPackageStartupMessages({
  library(mice)
  library(mitml)
  library(survival)
  library(sandwich)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) >= 1) args[[1]] else "tests/reference/fixtures/mi"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

num <- function(x) ifelse(is.infinite(x), ifelse(x > 0, "Inf", "-Inf"), x)

# 1. pool.scalar on fixed inputs -------------------------------------------------------------------
scalar_cases <- list(
  list(name = "classic", Q = c(1.0, 1.2, 0.9, 1.1, 1.05), U = c(0.04, 0.05, 0.045, 0.04, 0.05), n = Inf, k = 1),
  list(name = "finite_n", Q = c(1.0, 1.2, 0.9, 1.1, 1.05), U = c(0.04, 0.05, 0.045, 0.04, 0.05), n = 51, k = 1),
  list(
    name = "regression_df",
    Q = c(-0.31, -0.27, -0.35, -0.29, -0.33, -0.30, -0.28, -0.32, -0.34, -0.26),
    U = rep(c(0.0042, 0.0047), 5), n = 200, k = 5
  ),
  list(name = "no_between", Q = c(2, 2, 2), U = c(0.25, 0.25, 0.25), n = Inf, k = 1),
  list(name = "high_fmi", Q = c(0.1, 0.9, -0.4, 1.3, 0.2, -0.8), U = c(0.05, 0.04, 0.06, 0.05, 0.04, 0.05), n = 40, k = 3)
)
scalar <- lapply(scalar_cases, function(cs) {
  p <- pool.scalar(cs$Q, cs$U, n = cs$n, k = cs$k)
  se <- sqrt(p$t)
  stat <- p$qbar / se
  list(
    name = cs$name, Q = cs$Q, U = cs$U, dfcom = num(cs$n - cs$k),
    qbar = p$qbar, ubar = p$ubar, b = p$b, t = p$t, df = p$df, riv = p$r, fmi = p$fmi,
    statistic = stat, p_value = 2 * pt(abs(stat), p$df, lower.tail = FALSE),
    ci_low = p$qbar - qt(0.975, p$df) * se, ci_high = p$qbar + qt(0.975, p$df) * se
  )
})

# 2. A synthetic cohort with MAR missingness, imputed by mice ---------------------------------------
set.seed(20261014)
n <- 300
age <- round(rnorm(n, 60, 10), 1)
group <- factor(sample(c("a", "b", "c"), n, replace = TRUE))
x <- round(0.5 * age + rnorm(n, 0, 5), 2)
y <- 2 + 0.05 * age + 0.8 * (group == "b") - 0.4 * (group == "c") + 0.03 * x + rnorm(n)
bin <- rbinom(n, 1, plogis(-3 + 0.04 * age + 0.5 * (group == "b")))
count <- rpois(n, exp(0.2 + 0.01 * age))
time <- rexp(n, rate = exp(-4 + 0.03 * age + 0.4 * (group == "b")))
status <- rbinom(n, 1, 0.7)
dat <- data.frame(y, age, group, x, bin, count, time, status)
dat$x[runif(n) < plogis(-2 + 0.05 * (age - 60))] <- NA
dat$y[runif(n) < 0.15] <- NA
imp <- mice(dat, m = 5, seed = 20261014, printFlag = FALSE)
write.csv(complete(imp, "long"), file.path(out, "mice_completed_long.csv"), row.names = FALSE)

# 3. Pooled models, D1 and D2 tests of the group terms ----------------------------------------------
pooled_table <- function(fit) {
  p <- pool(fit)
  s <- summary(p, conf.int = TRUE)
  pp <- p$pooled
  data.frame(
    term = as.character(s$term), estimate = s$estimate, std_error = s$std.error, statistic = s$statistic,
    df = s$df, p_value = s$p.value, ci_low = s[["2.5 %"]], ci_high = s[["97.5 %"]],
    riv = pp$riv, lambda = pp$lambda, fmi = pp$fmi, ubar = pp$ubar, b = pp$b, t = pp$t, dfcom = num(pp$dfcom)
  )
}
test_row <- function(test) {
  r <- as.data.frame(test$result)
  as.list(r[1, , drop = FALSE])
}
fits <- list(
  lm = with(imp, lm(y ~ age + group + x)),
  logistic = with(imp, glm(bin ~ age + group + x, family = binomial)),
  poisson = with(imp, glm(count ~ age + group + x, family = poisson)),
  cox = with(imp, coxph(Surv(time, status) ~ age + group + x))
)
reduced <- list(
  lm = with(imp, lm(y ~ age + x)),
  logistic = with(imp, glm(bin ~ age + x, family = binomial)),
  poisson = with(imp, glm(count ~ age + x, family = poisson)),
  cox = with(imp, coxph(Surv(time, status) ~ age + x))
)
models <- lapply(names(fits), function(k) {
  d1 <- D1(fits[[k]], reduced[[k]])
  d2 <- D2(fits[[k]], reduced[[k]], use = "wald")
  list(
    model = k,
    table = pooled_table(fits[[k]]),
    d1 = test_row(d1), d1_dfcom = num(d1$dfcom),
    d2 = test_row(d2)
  )
})

# 4. Pooled R2 (Harel) -------------------------------------------------------------------------------
r2 <- pool.r.squared(fits$lm)
r2_adj <- pool.r.squared(fits$lm, adjusted = TRUE)

# 5. The groups compared with robust (HC3) variances: lm per dataset, sandwich::vcovHC, pooled by mitml ----------
robust_fit <- function(d) {
  fit <- lm(y ~ group, data = d)
  b <- coef(fit)
  v <- vcovHC(fit, type = "HC3")
  terms <- sub("(Intercept)", "Intercept", names(b), fixed = TRUE)  # syntactic names for testConstraints
  names(b) <- terms
  dimnames(v) <- list(terms, terms)
  list(coef = b, vcov = v, df = df.residual(fit))
}
robust_fits <- lapply(complete(imp, "all"), robust_fit)
qhat <- lapply(robust_fits, `[[`, "coef")
uhat <- lapply(robust_fits, `[[`, "vcov")
robust_dfcom <- robust_fits[[1]]$df
estimates <- as.data.frame(testEstimates(qhat = qhat, uhat = uhat, df.com = robust_dfcom)$estimates)
robust_d1 <- as.data.frame(
  testConstraints(qhat = qhat, uhat = uhat, constraints = c("groupb", "groupc"), method = "D1", df.com = robust_dfcom)$test
)
robust <- list(
  formula = "y ~ group",
  dfcom = robust_dfcom,
  first_vcov = list(terms = names(qhat[[1]]), matrix = unname(uhat[[1]])),
  table = data.frame(
    term = rownames(estimates), estimate = estimates[["Estimate"]], std_error = estimates[["Std.Error"]],
    statistic = estimates[["t.value"]], df = estimates[["df"]], p_value = estimates[["P(>|t|)"]],
    riv = estimates[["RIV"]], fmi = estimates[["FMI"]]
  ),
  d1 = list(
    statistic = robust_d1[["F.value"]][1], df1 = robust_d1[["df1"]][1], df2 = robust_d1[["df2"]][1],
    p_value = robust_d1[["P(>F)"]][1], riv = robust_d1[["RIV"]][1]
  )
)

write_json(
  list(
    meta = list(
      r = R.version.string,
      mice = as.character(packageVersion("mice")),
      mitml = as.character(packageVersion("mitml")),
      survival = as.character(packageVersion("survival")),
      sandwich = as.character(packageVersion("sandwich")),
      seed = 20261014, m = 5, n = n
    ),
    scalar = scalar,
    models = models,
    robust = robust,
    r_squared = list(
      estimate = r2[1, 1], low = r2[1, 2], high = r2[1, 3], fmi = r2[1, 4],
      adjusted_estimate = r2_adj[1, 1], adjusted_low = r2_adj[1, 2], adjusted_high = r2_adj[1, 3]
    )
  ),
  file.path(out, "mice_reference.json"),
  digits = NA, auto_unbox = TRUE, pretty = TRUE
)
cat("Fixtures written to", out, "\n")
print(sessionInfo())
