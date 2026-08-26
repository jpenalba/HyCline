"""
Phase 4, Component C demo: coupling/concordance analysis across loci.

Demonstrates the full pipeline: fit many loci hierarchically (Components
A/B), then quantify how concordant the fitted clines are and estimate
Barton's coupling coefficient (theta) -- following Firneno et al. (2023).

Two scenarios: a "high coupling" set of loci with concordant clines
(similar centers/widths -- expect low variance, high estimated theta) and
a "low coupling" set with discordant clines (expect high variance, low
estimated theta), for both geographic and genomic clines.

Run with:
    python examples/demo_coupling.py
"""

import numpy as np

from hycline.analysis.coupling import geographic_cline_variance, genomic_cline_variance
from hycline.inference.mcmc_genomic_hierarchical import fit_hierarchical_genomic_batch
from hycline.inference.mcmc_hierarchical import fit_hierarchical_batch
from hycline.models.cline import ClineParams, TailType, cline_function
from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function


def geographic_scenario(concordant: bool, seed: int):
    rng = np.random.default_rng(seed)
    distance = np.sort(rng.uniform(0, 150, size=30))
    n_loci = 6
    if concordant:
        true_widths = rng.normal(20.0, 1.5, n_loci)
    else:
        true_widths = np.exp(rng.normal(np.log(20.0), 0.7, n_loci))

    n_alleles = np.full((n_loci, 30), 20.0)
    n_focal = np.zeros((n_loci, 30))
    for j in range(n_loci):
        params = ClineParams(center=75.0, width=max(true_widths[j], 2.0), pmin=0.05, pmax=0.95)
        p = cline_function(distance, params, TailType.NONE)
        n_focal[j, :] = rng.binomial(20, p)
    return distance, n_alleles, n_focal, true_widths


def genomic_scenario(concordant: bool, seed: int):
    rng = np.random.default_rng(seed)
    hybrid_index = rng.uniform(0.02, 0.98, 150)
    n_loci = 6
    if concordant:
        true_u = rng.normal(0.0, 0.15, n_loci)
        true_v = np.exp(rng.normal(0.0, 0.1, n_loci))
    else:
        true_u = rng.normal(0.0, 1.2, n_loci)
        true_v = np.exp(rng.normal(0.0, 0.6, n_loci))

    n_alleles = np.full((n_loci, 150), 2.0)
    n_focal = np.zeros((n_loci, 150))
    for j in range(n_loci):
        params = GenomicClineParams(u=true_u[j], v=true_v[j])
        phi = genomic_cline_function(hybrid_index, params)
        n_focal[j, :] = rng.binomial(2, phi)
    return hybrid_index, n_alleles, n_focal, true_u, true_v


def main():
    print("=" * 74)
    print("HyCline Phase 4 Component C -- coupling/concordance analysis demo")
    print("=" * 74)

    print("\n### Geographic clines ###\n")
    for label, concordant, seed in [("Concordant (high coupling)", True, 10),
                                     ("Discordant (low coupling)", False, 11)]:
        distance, n_alleles, n_focal, true_widths = geographic_scenario(concordant, seed)
        fit = fit_hierarchical_batch(distance, n_alleles, n_focal,
                                      num_warmup=800, num_samples=800, num_chains=4, seed=0)
        locus_df = fit.locus_summary()
        result = geographic_cline_variance(locus_df["width_mean"])
        print(f"--- {label} ---")
        print(f"True widths: {np.round(true_widths, 1)}")
        print(result.summary())
        print()

    print("\n### Genomic clines ###\n")
    for label, concordant, seed in [("Concordant (high coupling)", True, 20),
                                     ("Discordant (low coupling)", False, 21)]:
        hybrid_index, n_alleles, n_focal, true_u, true_v = genomic_scenario(concordant, seed)
        fit = fit_hierarchical_genomic_batch(hybrid_index, n_alleles, n_focal,
                                              num_warmup=800, num_samples=800,
                                              num_chains=4, seed=0)
        locus_df = fit.locus_summary()
        result = genomic_cline_variance(locus_df["u_mean"], locus_df["v_mean"])
        print(f"--- {label} ---")
        print(f"True u: {np.round(true_u, 2)}, True v: {np.round(true_v, 2)}")
        print(result.summary())
        print()

    print("Expected pattern: the 'concordant' scenarios should show lower "
          "SD/CV and (for genomic clines) a higher estimated theta than the "
          "'discordant' scenarios -- see BACKEND_PHASE4.md section 9 for "
          "the full theta calibration methodology and its caveats.")


if __name__ == "__main__":
    main()
