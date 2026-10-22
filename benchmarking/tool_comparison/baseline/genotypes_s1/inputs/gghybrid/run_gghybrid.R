# gghybrid (Bailey 2023): hybrid index (esth) then per-locus genomic clines
# (ggcline). Run from this folder: Rscript run_gghybrid.R
suppressMessages({library(gghybrid); library(data.table)})
set.seed(1)
dat <- read.data("gghybrid_input.csv", nprecol = 2, MISSINGVAL = NA, NUMINDS = 140)
prep <- data.prep(data = dat$data, loci = dat$loci, alleles = dat$alleles,
                  S0 = "pop1", S1 = "pop2", precols = dat$precols)
pdf(NULL)
hi <- esth(data.prep.object = prep$data.prep, read.data.precols = dat$precols,
           include.Source = TRUE, nitt = 3000, burnin = 1000)
fwrite(hi$hi, "hybrid_index.tsv", sep = "\t")
gc <- ggcline(data.prep.object = prep$data.prep, esth.object = hi,
              read.data.precols = dat$precols, nitt = 5000, burnin = 2000)
fwrite(gc$gc, "clines.tsv", sep = "\t")
