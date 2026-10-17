#!/usr/bin/env Rscript

# Generate cross-language golden fixtures from the frozen manuscript-derived
# S3PCA reference implementation. No non-base R package is required.

args <- commandArgs(trailingOnly = TRUE)
reference_file <- if (length(args) >= 1L) args[[1L]] else "S3PCA_reference.R"
out_dir <- if (length(args) >= 2L) args[[2L]] else file.path("tests", "regression", "fixtures", "r_v4")

if (!file.exists(reference_file)) {
  stop("Reference file not found: ", reference_file)
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
source(reference_file, local = TRUE)
options(digits = 17)

write_numeric_csv <- function(x, path, column_names = NULL) {
  x <- as.matrix(x)
  if (!is.null(column_names)) colnames(x) <- column_names
  utils::write.table(
    x, file = path, sep = ",", row.names = FALSE,
    col.names = !is.null(colnames(x)), quote = FALSE, na = "NA",
    qmethod = "double", fileEncoding = "UTF-8"
  )
}

write_vector_csv <- function(x, path, name = "value") {
  write_numeric_csv(matrix(as.numeric(x), ncol = 1L, dimnames = list(NULL, name)), path)
}

json_escape <- function(x) {
  x <- gsub("\\\\", "\\\\\\\\", x)
  x <- gsub('"', '\\"', x, fixed = TRUE)
  x
}
json_value <- function(x) {
  if (is.null(x)) return("null")
  if (is.logical(x)) return(ifelse(x, "true", "false"))
  if (is.numeric(x)) return(format(x, digits = 17, scientific = TRUE, trim = TRUE))
  paste0('"', json_escape(as.character(x)), '"')
}
write_json_object <- function(x, path) {
  keys <- names(x)
  lines <- vapply(seq_along(x), function(i) {
    comma <- if (i < length(x)) "," else ""
    paste0("  ", json_value(keys[[i]]), ": ", json_value(x[[i]]), comma)
  }, character(1L))
  writeLines(c("{", lines, "}"), path, useBytes = TRUE)
}

make_s4pca_fixture <- function(seed = 20260806L, n = 75L, p = 40L) {
  set.seed(seed)
  score <- matrix(stats::rnorm(n * 4L), nrow = n, ncol = 4L)
  loading <- matrix(0, nrow = p, ncol = 4L)
  support_template <- list(1:10, 8:17, 18:27, 25:34)
  support <- lapply(support_template, function(idx) idx[idx <= p])
  for (k in seq_along(support)) {
    idx <- support[[k]]
    if (!length(idx)) next
    loading[idx, k] <- stats::runif(length(idx), 0.5, 1.0) *
      sample(c(-1, 1), length(idx), replace = TRUE)
  }
  X <- score %*% t(loading) + matrix(stats::rnorm(n * p, sd = 0.5), n, p)
  X <- scale(X)
  Z <- as.numeric(scale(score[, 1] + 0.8 * score[, 2] + stats::rnorm(n)))
  colnames(X) <- paste0("x", seq_len(p))
  list(X = X, Z = Z, score = score, loading = loading)
}

run_reference <- function(X, Z = NULL, comp = 6L, lambda = 0.20,
                          gamma = 0.10, mu = 1.0, str = FALSE,
                          niter = 40L, seed = 3L) {
  set.seed(seed)
  S3PCA_reference(
    Xs = X, Zs = Z, comp = comp, lambda = lambda, gamma = gamma,
    excl_mode = "sumabs", mu = mu, str = str, niter = niter,
    verbose = FALSE
  )
}

cases <- list(
  s4pca = list(comp = 6L, lambda = 0.20, gamma = 0.10, mu = 1.0, str = FALSE, niter = 40L, seed = 3L, use_z = TRUE),
  s2pca = list(comp = 6L, lambda = 0.20, gamma = 0.00, mu = 0.8, str = FALSE, niter = 40L, seed = 3L, use_z = TRUE),
  sstspca = list(comp = 6L, lambda = 0.20, gamma = 0.10, mu = 0.0, str = FALSE, niter = 40L, seed = 3L, use_z = TRUE),
  sparse_pca = list(comp = 6L, lambda = 0.20, gamma = 0.00, mu = 0.0, str = FALSE, niter = 40L, seed = 3L, use_z = TRUE),
  exclusive = list(comp = 6L, lambda = 0.05, gamma = 0.00, mu = 1.0, str = TRUE, niter = 40L, seed = 3L, use_z = TRUE),
  no_sparsity = list(comp = 6L, lambda = 0.00, gamma = 0.00, mu = 0.0, str = FALSE, niter = 40L, seed = 3L, use_z = TRUE),
  without_z = list(comp = 4L, lambda = 0.20, gamma = 0.10, mu = 0.0, str = FALSE, niter = 40L, seed = 3L, use_z = FALSE),
  components_1 = list(comp = 1L, lambda = 0.20, gamma = 0.10, mu = 1.0, str = FALSE, niter = 40L, seed = 3L, use_z = TRUE),
  components_2 = list(comp = 2L, lambda = 0.20, gamma = 0.10, mu = 1.0, str = FALSE, niter = 40L, seed = 3L, use_z = TRUE),
  components_4 = list(comp = 4L, lambda = 0.20, gamma = 0.10, mu = 1.0, str = FALSE, niter = 40L, seed = 3L, use_z = TRUE)
)

dat <- make_s4pca_fixture()
write_numeric_csv(dat$X, file.path(out_dir, "input_X.csv"), colnames(dat$X))
write_vector_csv(dat$Z, file.path(out_dir, "input_z.csv"), "z")
write_numeric_csv(dat$score, file.path(out_dir, "generating_scores.csv"), paste0("latent", 1:4))
write_numeric_csv(dat$loading, file.path(out_dir, "generating_loadings.csv"), paste0("latent", 1:4))

manifest <- data.frame(
  case = names(cases), comp = integer(length(cases)), lambda_x = numeric(length(cases)),
  gamma_x = numeric(length(cases)), mu_x = numeric(length(cases)),
  hard_exclusion = logical(length(cases)), n_iter = integer(length(cases)),
  seed = integer(length(cases)), use_z = logical(length(cases)),
  stringsAsFactors = FALSE
)

for (i in seq_along(cases)) {
  case_name <- names(cases)[[i]]
  cfg <- cases[[i]]
  case_dir <- file.path(out_dir, case_name)
  dir.create(case_dir, recursive = TRUE, showWarnings = FALSE)
  Z <- if (isTRUE(cfg$use_z)) dat$Z else NULL

  # Capture exactly the initial normal vector consumed by S3PCA_reference,
  # then reset the seed before executing the reference fit.
  set.seed(cfg$seed)
  initial_normal <- stats::rnorm(nrow(dat$X))
  initial_score <- .reference_normvec(initial_normal)
  fit <- run_reference(
    dat$X, Z, comp = cfg$comp, lambda = cfg$lambda,
    gamma = cfg$gamma, mu = cfg$mu, str = cfg$str,
    niter = cfg$niter, seed = cfg$seed
  )

  selected_count <- rowSums(abs(fit$W) > 1e-8)
  overlap_all <- mean(selected_count >= 2L)
  overlap_selected <- if (any(selected_count >= 1L)) {
    sum(selected_count >= 2L) / sum(selected_count >= 1L)
  } else 0

  write_vector_csv(initial_normal, file.path(case_dir, "initial_normal.csv"), "initial_normal")
  write_vector_csv(initial_score, file.path(case_dir, "initial_score.csv"), "initial_score")
  write_numeric_csv(fit$W, file.path(case_dir, "expected_loadings.csv"), colnames(fit$W))
  write_numeric_csv(fit$S, file.path(case_dir, "expected_scores.csv"), colnames(fit$S))
  write_numeric_csv(1L * (abs(fit$W) <= 1e-8), file.path(case_dir, "expected_zero_mask.csv"), colnames(fit$W))
  write_vector_csv(fit$used_idx, file.path(case_dir, "used_indices_r.csv"), "index")
  write_vector_csv(fit$used_idx - 1L, file.path(case_dir, "used_indices_python.csv"), "index")
  write_json_object(list(
    case = case_name,
    rss = fit$rss,
    aic = fit$AIC,
    bic = fit$BIC,
    ebic = fit$EBIC,
    overlap_all = overlap_all,
    overlap_selected = overlap_selected,
    extracted_components = ncol(fit$W),
    requested_components = cfg$comp,
    tolerance = 1e-12
  ), file.path(case_dir, "metrics.json"))

  manifest$comp[[i]] <- cfg$comp
  manifest$lambda_x[[i]] <- cfg$lambda
  manifest$gamma_x[[i]] <- cfg$gamma
  manifest$mu_x[[i]] <- cfg$mu
  manifest$hard_exclusion[[i]] <- cfg$str
  manifest$n_iter[[i]] <- cfg$niter
  manifest$seed[[i]] <- cfg$seed
  manifest$use_z[[i]] <- cfg$use_z
}

utils::write.csv(manifest, file.path(out_dir, "manifest.csv"), row.names = FALSE, quote = TRUE)
write_json_object(list(
  fixture_schema_version = 1L,
  fixture_seed = 20260806L,
  n_samples = nrow(dat$X),
  n_features = ncol(dat$X),
  reference_function = "S3PCA_reference",
  reference_file = normalizePath(reference_file, winslash = "/", mustWork = TRUE),
  r_version = R.version.string,
  platform = R.version$platform,
  tolerance = 1e-12,
  zero_pattern_threshold = 1e-8
), file.path(out_dir, "metadata.json"))

capture.output(sessionInfo(), file = file.path(out_dir, "sessionInfo.txt"))
checksum_files <- list.files(out_dir, recursive = TRUE, full.names = FALSE)
checksums <- tools::md5sum(file.path(out_dir, checksum_files))
utils::write.csv(
  data.frame(file = checksum_files, md5 = unname(checksums), row.names = NULL),
  file.path(out_dir, "checksums.csv"), row.names = FALSE, quote = TRUE
)
cat("Generated S4PCA golden fixtures in:", normalizePath(out_dir, winslash = "/"), "\n")
