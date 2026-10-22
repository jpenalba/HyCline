# bgchm (Gompert, DeRaad & Buerkle 2024): parental allele frequencies,
# hybrid indexes, then a hierarchical genomic cline model. Run from this
# folder: Rscript run_bgchm.R [genotype|glik]
suppressMessages(library(bgchm))
args <- commandArgs(trailingOnly = TRUE)
model <- if (length(args)) args[1] else "genotype"
rd <- function(f) as.matrix(read.delim(f, row.names = 1, check.names = FALSE))
G0 <- rd("G0.tsv"); G1 <- rd("G1.tsv"); Gx <- rd("Gx.tsv")
set.seed(1)
if (model == "glik") {
    gl <- function(set) lapply(0:2, function(k) rd(sprintf("GL%d_%s.tsv", k, set)))
    GL0 <- gl("G0"); GL1 <- gl("G1"); GLx <- gl("Gx")
    # HMC = TRUE is bgchm's default for genotype likelihoods; its analytical
    # shortcut (HMC = FALSE) shrinks parental frequencies toward 0.5 at low depth.
    p <- est_p(G0 = GL0, G1 = GL1, model = "glik", ploidy = "diploid", HMC = TRUE,
               n_iters = 2000, full = FALSE)
    h <- est_hi(Gx = GLx, p0 = p$p0[, 1], p1 = p$p1[, 1], model = "glik",
                ploidy = "diploid", n_iters = 2000, full = FALSE)
    cl <- est_genocl(Gx = GLx, p0 = p$p0[, 1], p1 = p$p1[, 1], H = h$hi[, 1],
                     model = "glik", ploidy = "diploid", hier = TRUE,
                     n_iters = 2000, full = TRUE)
} else if (anyNA(G0) || anyNA(G1) || anyNA(Gx)) {
    # Missing genotypes: bgchm encodes them with ploidy 0 under ploidy = "mixed".
    pl <- function(G) { m <- ifelse(is.na(G), 0, 2); G[is.na(G)] <- 0; list(G = G, pl = m) }
    a0 <- pl(G0); a1 <- pl(G1); ax <- pl(Gx)
    p <- est_p(G0 = a0$G, G1 = a1$G, model = "genotype", ploidy = "mixed",
               pldat = list(ax$pl, a0$pl, a1$pl))
    h <- est_hi(Gx = ax$G, p0 = p$p0[, 1], p1 = p$p1[, 1], model = "genotype",
                ploidy = "mixed", pldat = ax$pl, n_iters = 2000, full = FALSE)
    cl <- est_genocl(Gx = ax$G, p0 = p$p0[, 1], p1 = p$p1[, 1], H = h$hi[, 1],
                     model = "genotype", ploidy = "mixed", pldat = ax$pl, hier = TRUE,
                     n_iters = 2000, full = TRUE)
} else {
    p <- est_p(G0 = G0, G1 = G1, model = "genotype", ploidy = "diploid")
    h <- est_hi(Gx = Gx, p0 = p$p0[, 1], p1 = p$p1[, 1], model = "genotype",
                ploidy = "diploid", n_iters = 2000, full = FALSE)
    cl <- est_genocl(Gx = Gx, p0 = p$p0[, 1], p1 = p$p1[, 1], H = h$hi[, 1],
                     model = "genotype", ploidy = "diploid", hier = TRUE,
                     n_iters = 2000, full = TRUE)
}
hi <- data.frame(individual = rownames(Gx), h$hi, check.names = FALSE)
names(hi)[-1] <- c("hi_median", "hi_q2.5", "hi_q5", "hi_q95", "hi_q97.5")
write.table(hi, "hybrid_index.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
q <- function(par) t(apply(rstan::extract(cl$gencline_hmc, par)[[1]], 2, quantile,
                           probs = c(.5, .025, .975)))
cc <- q("center"); vv <- q("v")
cl_out <- data.frame(locus = colnames(Gx), center = cc[, 1], center_q2.5 = cc[, 2],
                     center_q97.5 = cc[, 3], v = vv[, 1], v_q2.5 = vv[, 2], v_q97.5 = vv[, 3])
write.table(cl_out, "clines.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
cat("SDc", cl$SDc, "\nSDv", cl$SDv, "\n")
