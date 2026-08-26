"""
Phase 4, Component A demo: hierarchical Bayesian geographic cline fitting
(partial pooling of center/width across many loci sharing a transect).

There's no HZAR-equivalent gold standard for hierarchical pooling (HZAR
doesn't do this at all -- see BACKEND_PHASE4.md section 1), so validation
here is against synthetic data with a known population-level structure,
demonstrating the actual signature of hierarchical models: a sparse,
noisy locus gets pulled ("shrunk") toward the population mean more than a
data-rich locus does, while a data-rich locus's hierarchical estimate
should closely track its own independent (unpooled) estimate.

Run with:
    python examples/demo_hierarchical_geographic.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hycline.data.schema import AlleleFreqTransect
from hycline.inference.mcmc_hierarchical import fit_hierarchical_batch, _derive_pmin_pmax
from hycline.inference.mle import fit_mle_fixed_extremes
from hycline.models.cline import ClineParams, TailType, cline_function

HERE = Path(__file__).parent


def simulate(seed=7):
    rng = np.random.default_rng(seed)

    mu_center_true = 75.0
    sigma_center_true = 10.0
    mu_log_width_true = np.log(25.0)
    sigma_log_width_true = 0.25

    n_loci = 8
    distance = np.sort(rng.uniform(0, 150, size=40))

    true_centers = rng.normal(mu_center_true, sigma_center_true, n_loci)
    true_widths = np.exp(rng.normal(mu_log_width_true, sigma_log_width_true, n_loci))

    n_alleles = np.zeros((n_loci, len(distance)))
    n_focal = np.zeros((n_loci, len(distance)))

    sample_sizes = [4, 60, 20, 20, 20, 20, 20, 20]  # locus 0 = sparse, locus 1 = data-rich
    for j in range(n_loci):
        params = ClineParams(center=true_centers[j], width=true_widths[j], pmin=0.05, pmax=0.95)
        p = cline_function(distance, params, TailType.NONE)
        n = sample_sizes[j]
        n_alleles[j, :] = n
        n_focal[j, :] = rng.binomial(n, p)

    truth = dict(mu_center=mu_center_true, sigma_center=sigma_center_true,
                 mu_width=np.exp(mu_log_width_true), true_centers=true_centers,
                 true_widths=true_widths, sample_sizes=sample_sizes)
    return distance, n_alleles, n_focal, truth


def main():
    distance, n_alleles, n_focal, truth = simulate()
    n_loci = n_alleles.shape[0]
    locus_names = [f"locus_{i}" for i in range(n_loci)]

    print("=" * 74)
    print("HyCline Phase 4 Component A -- hierarchical geographic cline demo")
    print("=" * 74)
    print(f"\nTrue population-level: mu_center={truth['mu_center']}, "
          f"sigma_center={truth['sigma_center']}, mu_width={truth['mu_width']:.1f}")
    print(f"True per-locus centers: {np.round(truth['true_centers'], 1)}")
    print(f"True per-locus widths:  {np.round(truth['true_widths'], 1)}")
    print(f"Per-site sample sizes:  {truth['sample_sizes']}  "
          f"(locus_0 = sparse, locus_1 = data-rich)")

    result = fit_hierarchical_batch(distance, n_alleles, n_focal, locus_names=locus_names,
                                     num_warmup=1200, num_samples=1200, num_chains=4, seed=0)
    print("\n--- Hierarchical fit ---")
    print(result.summary())

    # --- Independent (unpooled) single-locus fits, for the shrinkage comparison ---
    pmin_all, pmax_all = _derive_pmin_pmax(distance, n_alleles, n_focal, n_edge_sites=3)
    independent = {}
    for j in [0, 1]:  # sparse and data-rich only, to keep the demo fast
        transect = AlleleFreqTransect(distance=distance, n_alleles=n_alleles[j], n_focal=n_focal[j])
        fit = fit_mle_fixed_extremes(transect, tail_type=TailType.NONE,
                                      pmin=pmin_all[j], pmax=pmax_all[j])
        independent[j] = (fit.params.center, fit.params.width)

    locus_df = result.locus_summary()
    pop_center = float(result.idata.posterior["mu_center"].mean())
    pop_width = float(np.exp(result.idata.posterior["mu_log_width"].mean()))

    print("\n--- Shrinkage comparison: sparse (locus_0) vs. data-rich (locus_1) ---")
    print(f"Population mean (hierarchical posterior): center={pop_center:.2f}, "
          f"width={pop_width:.2f}\n")
    print(f"{'':<12}{'independent':>14}{'hierarchical':>14}{'shrunk toward pop mean by':>28}")
    for j, label in [(0, "locus_0 (sparse, n=4)"), (1, "locus_1 (rich, n=60)")]:
        indep_c, indep_w = independent[j]
        hier_c = locus_df.loc[j, "center_mean"]
        hier_w = locus_df.loc[j, "width_mean"]
        gap_c = pop_center - indep_c
        shrink_c = (hier_c - indep_c) / gap_c * 100 if abs(gap_c) > 1e-6 else 0.0
        gap_w = pop_width - indep_w
        shrink_w = (hier_w - indep_w) / gap_w * 100 if abs(gap_w) > 1e-6 else 0.0
        print(f"{label:<12}")
        print(f"  center    {indep_c:>14.2f}{hier_c:>14.2f}{shrink_c:>27.1f}%")
        print(f"  width     {indep_w:>14.2f}{hier_w:>14.2f}{shrink_w:>27.1f}%")

    print("\nExpected signature: locus_0 (sparse) should show a larger shrinkage "
          "percentage than locus_1 (data-rich) for both parameters -- the sparse "
          "locus has less of its own information to resist the pull of the "
          "population-level prior, while the data-rich locus's own likelihood "
          "dominates and its hierarchical estimate should closely track its "
          "independent one.")

    # --- Plot: all loci, hierarchical fit vs. observed data ---
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    x_smooth = np.linspace(distance.min(), distance.max(), 200)
    for j, ax in enumerate(axes.flatten()):
        obs_freq = n_focal[j] / n_alleles[j]
        ax.scatter(distance, obs_freq, s=15 + 2 * n_alleles[j], alpha=0.5, color="#2b6cb0")
        params = ClineParams(center=locus_df.loc[j, "center_mean"],
                              width=locus_df.loc[j, "width_mean"],
                              pmin=pmin_all[j], pmax=pmax_all[j])
        y_smooth = cline_function(x_smooth, params, TailType.NONE)
        ax.plot(x_smooth, y_smooth, color="#c53030", linewidth=2)
        true_params = ClineParams(center=truth["true_centers"][j], width=truth["true_widths"][j],
                                   pmin=0.05, pmax=0.95)
        y_true = cline_function(x_smooth, true_params, TailType.NONE)
        ax.plot(x_smooth, y_true, color="#38a169", linewidth=1.5, linestyle="--")
        tag = " (sparse)" if j == 0 else (" (rich)" if j == 1 else "")
        ax.set_title(f"locus_{j}{tag}", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle("Hierarchical fit (red, solid) vs. true generating cline (green, dashed)")
    fig.tight_layout()
    out_path = HERE / "hierarchical_geographic_demo.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
