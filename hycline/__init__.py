"""HyCline: a Python package for geographic and genomic cline analysis.

Phase 1: sigmoid + exponential-tail cline models, binomial likelihood for
allele-frequency data, MLE fitting, and AICc-based model comparison --
functional parity with HZAR's core geographic-cline workflow.

Phase 2: a Bayesian/MCMC fitting backend (NumPyro primary, PyMC secondary),
reachable via `fit_cline(..., method="bayesian")` or directly through
`hycline.inference.mcmc_numpyro`/`hycline.inference.mcmc_pymc`.

Phase 3: the genomic cline model (Fitzpatrick 2013 logit-logistic form) --
locus ancestry as a function of genome-wide hybrid index rather than
geographic distance. Reachable via `fit_genomic_cline()` or directly
through `hycline.inference.mle_genomic`. See BACKEND_PHASE3.md for why
this is a sibling model to the geographic cline, not a variant of it, and
why HZAR (otherwise this project's main comparison point) has no
equivalent to validate against here.

Phase 4: hierarchical Bayesian partial pooling across loci -- Component A
(geographic, `hycline.inference.mcmc_hierarchical`) and Component B
(genomic, `hycline.inference.mcmc_genomic_hierarchical`), both
Python-API-only for now, plus Component C (coupling/concordance analysis,
`hycline.analysis.coupling`), which quantifies how concordant fitted
clines are across loci and estimates Barton's (1983) theoretical coupling
coefficient following Firneno et al. (2023), plus Component D
(`hycline.inference.mcmc_hierarchical_gl`), which combines Component A's
partial pooling with Phase 5's genotype-likelihood marginalization for
geographic clines -- answering a real open question (does genotype-
likelihood-awareness still matter once you're already pooling across
loci?) with a clear yes, and for a more consequential reason than
expected: a hard-called sparse locus's distorted estimate contaminates
the *shared* population-level hyperparameters that every other locus in
the batch gets pooled toward, not just its own estimate -- plus Component
E (`hycline.inference.mcmc_genomic_hierarchical_gl`), the genomic-cline
analog of Component D, which found a genuinely *different* result rather
than repeating Component D's: at a moderate depth contrast,
genotype-likelihood-awareness makes no detectable difference for genomic
clines under pooling, and only at a more extreme contrast does a modest
(and differently-mechanismed) advantage appear. See BACKEND_PHASE4.md for
the full design rationale, including the biological motivation for
hierarchical pooling in the first place (also covered in the project
README's "Why hierarchical clines?" section).

Phase 5: genotype-likelihood-aware fitting -- marginalizing over
genotype uncertainty (as reported by variant callers, e.g. GATK/ANGSD/
bcftools) rather than requiring a hard genotype call first, for both
geographic (`hycline.inference.mle_genotype_likelihood.
fit_mle_geographic_gl`) and genomic
(`fit_mle_genomic_gl`) clines. See BACKEND_PHASE5.md for the full
derivation and a real, validated finding: this doesn't primarily fix
point-estimate bias (which didn't show up in idealized simulation without
mapping bias) -- it fixes *uncertainty calibration*, giving properly-sized
confidence intervals at low sequencing depth where naive hard-calling is
measurably overconfident.

`fit_cline()` is the recommended entry point when you want the fitting
*procedure* -- classic MLE vs. full Bayesian posterior -- to be a keyword
argument rather than a decision about which module to import. See
BACKEND_PHASE1.md and BACKEND_PHASE2.md for the design rationale behind
each, and BACKEND_PHASE1.md section 8 for guidance on which to use when.
"""

__version__ = "0.1.0-dev"

from hycline.api import Method, fit_cline, fit_genomic_cline
from hycline.models.cline import ClineParams, TailType, cline_function
from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function
from hycline.data.schema import AlleleFreqTransect, HybridIndexLocusData
from hycline.inference.mle import fit_mle, fit_mle_fixed_extremes, FitResult
from hycline.inference.mle_genomic import GenomicFitResult
from hycline.selection.aicc import compare_models
from hycline.analysis.coupling import (
    geographic_cline_variance, genomic_cline_variance, estimate_coupling_coefficient,
    GeographicClineVariance, GenomicClineVariance,
)
from hycline.data.schema import GenotypeLikelihoodTransect, GenotypeLikelihoodLocusData
from hycline.data.genotype_likelihood import (
    phred_to_log_likelihood, log_likelihood_to_phred, simulate_genotype_likelihoods,
)
from hycline.inference.mle_genotype_likelihood import fit_mle_geographic_gl, fit_mle_genomic_gl

__all__ = [
    "ClineParams", "TailType", "cline_function",
    "AlleleFreqTransect", "fit_mle", "fit_mle_fixed_extremes", "FitResult", "compare_models",
    "fit_cline", "Method",
    "GenomicClineParams", "genomic_cline_function", "HybridIndexLocusData",
    "fit_genomic_cline", "GenomicFitResult",
    "geographic_cline_variance", "genomic_cline_variance", "estimate_coupling_coefficient",
    "GeographicClineVariance", "GenomicClineVariance",
    "GenotypeLikelihoodTransect", "GenotypeLikelihoodLocusData",
    "phred_to_log_likelihood", "log_likelihood_to_phred", "simulate_genotype_likelihoods",
    "fit_mle_geographic_gl", "fit_mle_genomic_gl",
]
