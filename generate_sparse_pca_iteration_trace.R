#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3L) stop("Usage: Rscript generate_sparse_pca_iteration_trace.R S3PCA_reference.R input_X.csv output_dir")
source(args[[1L]], local = TRUE)
X <- as.matrix(utils::read.csv(args[[2L]], check.names = FALSE))
out <- args[[3L]]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
options(digits = 17)
set.seed(3L)
s <- .reference_normvec(stats::rnorm(nrow(X)))
Xd <- X
S <- W <- NULL
write_mat <- function(x, path) utils::write.table(as.matrix(x), path, sep=",", row.names=FALSE, col.names=FALSE, quote=FALSE)
for (component in 1:6) {
  for (iteration in 1:40) {
    raw_w <- c(t(Xd) %*% s)
    norm_w <- .reference_normvec(raw_w)
    sparse_w <- .reference_sparse(norm_w, lam=0.20, type="lasso")
    w <- .reference_normvec(sparse_w)
    s <- c(Xd %*% w)
    prefix <- sprintf("c%02d_i%02d", component, iteration)
    write_mat(raw_w, file.path(out, paste0(prefix, "_raw_w.csv")))
    write_mat(norm_w, file.path(out, paste0(prefix, "_norm_w.csv")))
    write_mat(sparse_w, file.path(out, paste0(prefix, "_sparse_w.csv")))
    write_mat(w, file.path(out, paste0(prefix, "_w.csv")))
    write_mat(s, file.path(out, paste0(prefix, "_s.csv")))
  }
  S <- cbind(S, s); W <- cbind(W, w)
  write_mat(Xd, file.path(out, sprintf("c%02d_residual_before_deflation.csv", component)))
  if (component < 6) {
    B <- .reference_project(X, S)
    write_mat(B, file.path(out, sprintf("c%02d_projection_coefficients.csv", component)))
    Xd <- X - S %*% B
    write_mat(Xd, file.path(out, sprintf("c%02d_residual_after_deflation.csv", component)))
  }
}
write_mat(W, file.path(out, "final_W.csv")); write_mat(S, file.path(out, "final_S.csv"))
capture.output(sessionInfo(), file=file.path(out, "sessionInfo.txt"))
cat("Trace written to", normalizePath(out, winslash="/"), "\n")
