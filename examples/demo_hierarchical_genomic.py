"""
Phase 4, Component B demo: hierarchical Bayesian genomic cline fitting
(partial pooling of u, v across many loci sharing the same set of
individuals/hybrid indices).

The genomic-cline analog of Component A
(examples/demo_hierarchical_geographic.py). There's no HZAR-equivalent
gold standard here either -- HZAR doesn't do genomic clines at all, let
alone hierarchical ones (BACKEND_PHASE4.md section 1) -- so validation is
against synthetic data with a known population-level structure,
demonstrating the same shrinkage signature Component A demonstrated: a
sparse locus's estimate gets pulled toward the population mean more than
a data-rich locus's does.

Run with:
    python examples/demo_hierarchical_genomic.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hycline.data.schema import HybridIndexLocusData
from hycline.inference.mcmc_genomic_hierarchical import fit_hierarchical_genomic_batch
from hycline.inference.mle_genomic import fit_genomic_cline_mle
from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function

HERE = Path(__file__).parent


def simulate(seed=21):
    rng = np.random.default_rng(seed)

    mu_u_true = 0.0
    sigma_u_true = 1.0
    mu_log_v_true = 0.0  # mu_v = 1: neutral genome-wide average
    sigma_log_v_true = 0.5

    n_loci = 8
    n_ind = 120
    hybrid_index = rng.uniform(0.02, 0.98, n_ind)

    true_u = rng.normal(mu_u_true, sigma_u_true, n_loci)
    true_v = np.exp(rng.normal(mu_log_v_true, sigma_log_v_true, n_loci))

    n_alleles = np.zeros((n_loci, n_ind))
    n_focal = np.zeros((n_loci, n_ind))

    # locus_0: a single low-confidence SNP (n=1) -- e.g. a low-coverage
    # sequencing call. locus_1: a resolved multi-SNP ancestry block (n=20)
    # -- pooled evidence from several tightly linked markers treated as
    # one "locus." Both are realistic shapes for real genomic cline data.
    sample_sizes = [1, 20, 2, 2, 2, 2, 2, 2]
    for j in range(n_loci):
        params = GenomicClineParams(u=true_u[j], v=true_v[j])
        phi = genomic_cline_function(hybrid_index, params)
        n = sample_sizes[j]
        n_alleles[j, :] = n
        n_focal[j, :] = rng.binomial(n, phi)

    truth = dict(mu_u=mu_u_true, sigma_u=sigma_u_true, mu_v=np.exp(mu_log_v_true),
                 true_u=true_u, true_v=true_v, sample_sizes=sample_sizes)
    return hybrid_index, n_alleles, n_focal, truth


def main():
    hybrid_index, n_alleles, n_focal, truth = simulate()
    n_loci = n_alleles.shape[0]
    locus_names = [f"locus_{i}" for i in range(n_loci)]

    print("=" * 74)
    print("HyCline Phase 4 Component B -- hierarchical genomic cline demo")
    print("=" * 74)
    print(f"\nTrue population-level: mu_u={truth['mu_u']}, sigma_u={truth['sigma_u']}, "
          f"mu_v={truth['mu_v']:.1f}")
    print(f"True per-locus u: {np.round(truth['true_u'], 2)}")
    print(f"True per-locus v: {np.round(truth['true_v'], 2)}")
    print(f"Per-individual sample sizes: {truth['sample_sizes']}  "
          f"(locus_0 = sparse SNP, locus_1 = resolved ancestry block)")

    result = fit_hierarchical_genomic_batch(hybrid_index, n_alleles, n_focal,
                                             locus_names=locus_names,
                                             num_warmup=1200, num_samples=1200,
                                             num_chains=4, seed=0)
    print("\n--- Hierarchical fit ---")
    print(result.summary())

    # --- Independent (unpooled) single-locus fits, for shrinkage comparison ---
    independent = {}
    for j in [0, 1]:
        d = HybridIndexLocusData(hybrid_index=hybrid_index, n_alleles=n_alleles[j],
                                  n_focal=n_focal[j])
        fit = fit_genomic_cline_mle(d)
        independent[j] = (fit.params.u, fit.params.v)

    locus_df = result.locus_summary()
    pop_u = float(result.idata.posterior["mu_u"].mean())
    pop_v = float(np.exp(result.idata.posterior["mu_log_v"].mean()))

    print("\n--- Shrinkage comparison: sparse SNP (locus_0) vs. ancestry block (locus_1) ---")
    print(f"Population mean (hierarchical posterior): u={pop_u:.3f}, v={pop_v:.3f}\n")
    print(f"{'':<26}{'independent':>14}{'hierarchical':>14}{'shrunk toward pop mean by':>28}")
    for j, label in [(0, "locus_0 (sparse, n=1)"), (1, "locus_1 (rich, n=20)")]:
        indep_u, indep_v = independent[j]
        hier_u = locus_df.loc[j, "u_mean"]
        hier_v = locus_df.loc[j, "v_mean"]
        gap_u = pop_u - indep_u
        shrink_u = (hier_u - indep_u) / gap_u * 100 if abs(gap_u) > 1e-6 else 0.0
        gap_v = pop_v - indep_v
        shrink_v = (hier_v - indep_v) / gap_v * 100 if abs(gap_v) > 1e-6 else 0.0
        print(f"{label:<26}")
        print(f"  u   {indep_u:>18.3f}{hier_u:>14.3f}{shrink_u:>27.1f}%")
        print(f"  v   {indep_v:>18.3f}{hier_v:>14.3f}{shrink_v:>27.1f}%")

    print("\nExpected signature: locus_0 (sparse) should show larger shrinkage than "
          "locus_1 (data-rich) for both parameters. This is the same behavior "
          "demonstrated for geographic clines in Component A -- see "
          "examples/demo_hierarchical_geographic.py -- now shown for genomic clines.")

    # --- Plot: all loci, hierarchical fit vs. true generating curve ---
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    h_smooth = np.linspace(0.001, 0.999, 200)
    for j, ax in enumerate(axes.flatten()):
        obs_freq = n_focal[j] / n_alleles[j]
        ax.scatter(hybrid_index, obs_freq, s=15 + 3 * n_alleles[j], alpha=0.4, color="#2b6cb0")
        fitted_params = GenomicClineParams(u=locus_df.loc[j, "u_mean"], v=locus_df.loc[j, "v_mean"])
        phi_fitted = genomic_cline_function(h_smooth, fitted_params)
        ax.plot(h_smooth, phi_fitted, color="#c53030", linewidth=2)
        true_params = GenomicClineParams(u=truth["true_u"][j], v=truth["true_v"][j])
        phi_true = genomic_cline_function(h_smooth, true_params)
        ax.plot(h_smooth, phi_true, color="#38a169", linewidth=1.5, linestyle="--")
        tag = " (sparse)" if j == 0 else (" (rich)" if j == 1 else "")
        ax.set_title(f"locus_{j}{tag}", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle("Hierarchical fit (red, solid) vs. true generating genomic cline (green, dashed)")
    fig.tight_layout()
    out_path = HERE / "hierarchical_genomic_demo.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
