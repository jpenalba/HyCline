"""Plotting utilities: observed data + fitted cline curve."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from hycline.data.schema import AlleleFreqTransect
from hycline.inference.mle import FitResult
from hycline.models.cline import ClineParams, TailType, cline_function, hzar_cline_function


def plot_fit(transect: AlleleFreqTransect, fit: FitResult, ax=None,
             title: str | None = None):
    """Plot observed allele frequencies against a fitted cline curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    # observed data, sized by sample depth
    sizes = 20 + 60 * (transect.n_alleles / transect.n_alleles.max())
    ax.scatter(transect.distance, transect.observed_freq, s=sizes,
               alpha=0.7, color="#2b6cb0", edgecolor="white", linewidth=0.5,
               label="Observed", zorder=3)

    # smooth fitted curve
    x_smooth = np.linspace(transect.distance.min(), transect.distance.max(), 400)
    curve_fn = hzar_cline_function if getattr(fit, "hzar_tail", False) else cline_function
    y_smooth = curve_fn(x_smooth, fit.params, fit.tail_type)
    ax.plot(x_smooth, y_smooth, color="#c53030", linewidth=2,
             label=f"Fitted cline ({fit.tail_type.value})", zorder=2)

    ax.set_xlabel("Geographic distance")
    ax.set_ylabel("Allele frequency")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title is None:
        title = (f"center={fit.params.center:.2f}, width={fit.params.width:.2f}, "
                  f"AICc={fit.aicc:.2f}")
    ax.set_title(title)

    return ax


def plot_bayesian_fit(transect: AlleleFreqTransect, fit, ax=None,
                       title: str | None = None, n_posterior_lines: int = 200,
                       credible_prob: float = 0.95):
    """Plot observed data against a Bayesian cline fit: posterior mean curve
    plus a shaded credible band computed by evaluating the cline function
    over posterior draws.

    Parameters
    ----------
    fit : a `hycline.inference.mcmc_numpyro.BayesianFitResult`
    n_posterior_lines : how many posterior draws to use for the credible
        band (subsampled from the full posterior for speed -- the full
        posterior is often tens of thousands of draws, far more than
        needed for a smooth band).
    credible_prob : width of the shaded credible band (e.g. 0.95 for 95%)
    """
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    sizes = 20 + 60 * (transect.n_alleles / transect.n_alleles.max())
    ax.scatter(transect.distance, transect.observed_freq, s=sizes,
               alpha=0.7, color="#2b6cb0", edgecolor="white", linewidth=0.5,
               label="Observed", zorder=3)

    x_smooth = np.linspace(transect.distance.min(), transect.distance.max(), 200)

    post = fit.idata.posterior
    center_draws = post["center"].values.flatten()
    width_draws = post["width"].values.flatten()
    pmin_draws = (post["pmin"].values.flatten() if "pmin" in post
                  else np.full_like(center_draws, fit._fixed_pmin))
    pmax_draws = (post["pmax"].values.flatten() if "pmax" in post
                  else np.full_like(center_draws, fit._fixed_pmax))

    delta_l_draws = post["delta_l"].values.flatten() if "delta_l" in post else None
    tau_l_draws = post["tau_l"].values.flatten() if "tau_l" in post else None
    delta_r_draws = post["delta_r"].values.flatten() if "delta_r" in post else None
    tau_r_draws = post["tau_r"].values.flatten() if "tau_r" in post else None

    rng = np.random.default_rng(0)
    idx = rng.choice(len(center_draws), size=min(n_posterior_lines, len(center_draws)),
                      replace=False)

    curves = np.empty((len(idx), len(x_smooth)))
    curve_fn = hzar_cline_function if getattr(fit, "hzar_tail", False) else cline_function
    for row, i in enumerate(idx):
        p = ClineParams(
            center=float(center_draws[i]), width=float(width_draws[i]),
            pmin=float(pmin_draws[i]), pmax=float(pmax_draws[i]),
            delta_l=float(delta_l_draws[i]) if delta_l_draws is not None else None,
            tau_l=float(tau_l_draws[i]) if tau_l_draws is not None else None,
            delta_r=float(delta_r_draws[i]) if delta_r_draws is not None else None,
            tau_r=float(tau_r_draws[i]) if tau_r_draws is not None else None,
        )
        curves[row] = curve_fn(x_smooth, p, fit.tail_type)

    lo = (100 - credible_prob * 100) / 2
    hi = 100 - lo
    band_lo = np.percentile(curves, lo, axis=0)
    band_hi = np.percentile(curves, hi, axis=0)
    band_mid = np.percentile(curves, 50, axis=0)

    ax.fill_between(x_smooth, band_lo, band_hi, color="#c53030", alpha=0.2,
                     label=f"{int(credible_prob*100)}% credible band", zorder=1)
    ax.plot(x_smooth, band_mid, color="#c53030", linewidth=2,
             label="Posterior median cline", zorder=2)

    ax.set_xlabel("Geographic distance")
    ax.set_ylabel("Allele frequency")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title is None:
        title = (f"center={float(np.mean(center_draws)):.2f} "
                  f"(posterior mean), width={float(np.mean(width_draws)):.2f}")
    ax.set_title(title)

    return ax


def plot_multi_locus_overlay(curves: dict, observed: dict | None = None,
                              max_legend_entries: int = 15, xlabel: str = "Geographic distance",
                              ylabel: str = "Allele frequency"):
    """Single-axes overlay: one fitted-curve line per locus, all sharing
    one set of axes, rather than a grid of separate subplots
    (`plot_multi_locus_grid`, kept for other callers, but no longer used
    by the geographic-cline CLI commands).

    Takes pre-computed curves directly -- `{locus_name: (x_smooth,
    y_smooth)}` -- rather than raw transect/fit objects, so the same
    function serves hard-count and genotype-likelihood-aware fits,
    classic and Bayesian, without needing to know which kind of fit
    object produced the curve. Callers already compute these arrays for
    the fitted-curve CSV export; this reuses them rather than
    recomputing.

    `observed`, if given, is `{locus_name: (x_obs, y_obs)}` -- plotted as
    small, semi-transparent points in each locus's own color. Omit it
    (the default) to show fitted curves only, which is usually the more
    readable choice once there are more than a handful of loci, since
    scatter from many loci sharing similar sites overlaps heavily.

    Scales far better than the small-multiples grid at scale (hundreds
    to thousands of loci) -- see BACKEND_SCALE.md section 3 for the
    concrete problem this replaces (a 300-locus grid plot already large
    enough to trip Pillow's own decompression-bomb heuristic). A
    discrete per-locus legend is only shown when there are few enough
    loci for it to stay readable (default cutoff: 15); beyond that, a
    colorbar conveys locus identity via a continuous colormap position
    instead of a legend entry per locus.
    """
    n = len(curves)
    fig, ax = plt.subplots(figsize=(8, 5.5))

    cmap = plt.get_cmap("viridis")
    locus_names = list(curves.keys())
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)] if n > 1 else [cmap(0.5)]

    for color, locus_name in zip(colors, locus_names):
        x_smooth, y_smooth = curves[locus_name]
        ax.plot(x_smooth, y_smooth, color=color, linewidth=1.6, alpha=0.85,
                 label=locus_name, zorder=2)
        if observed is not None and locus_name in observed:
            x_obs, y_obs = observed[locus_name]
            ax.scatter(x_obs, y_obs, s=18, color=color, alpha=0.35,
                       edgecolor="none", zorder=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{n} loci")

    if n <= max_legend_entries:
        ax.legend(loc="best", frameon=False, fontsize=8, ncol=1 if n <= 8 else 2)
    else:
        import matplotlib as mpl
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=n - 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("locus index (too many loci for a legend)")

    fig.tight_layout()
    return fig


def plot_multi_locus_grid(results: dict, ncols: int = 3, method: str = "classic"):
    """Small-multiples plot: one subplot per locus, from a dict of
    {locus_name: (transect, fit)} pairs. `method` selects plot_fit
    (classic) or plot_bayesian_fit (bayesian) per subplot. Loci where
    fitting failed should simply be excluded from `results` by the caller
    (see hycline/cli.py::cmd_fit_multi for how per-locus failures are
    tracked and reported separately)."""
    import math
    n = len(results)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, (locus_name, (transect, fit)) in enumerate(results.items()):
        ax = axes_flat[i]
        if method == "classic":
            plot_fit(transect, fit, ax=ax, title=locus_name)
        else:
            plot_bayesian_fit(transect, fit, ax=ax, title=locus_name)
        legend = ax.get_legend()
        if legend is not None and i > 0:
            legend.remove()

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    return fig


def plot_trait_fit(transect, fit, ax=None, title: str | None = None):
    """Plot observed per-site trait means (with error bars showing each
    site's own observed SD/sqrt(n)) against a fitted trait cline curve
    -- the trait analog of `plot_fit`. Unlike allele-frequency plots,
    the y-axis is not fixed to [-0.05, 1.05] -- trait values are
    unbounded and can be negative or decreasing across the transect (see
    `hycline.data.schema.TraitTransect`'s docstring).

    `fit` : a `hycline.inference.mle_trait.TraitFitResult` (full HZAR
    parity -- `transect.trait_sd` is always required and used directly
    for the error bars here, since the model itself uses it as data,
    not merely a display convenience).
    """
    from hycline.likelihoods.gaussian import mean_cline

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    se = transect.trait_sd / np.sqrt(transect.n)
    ax.errorbar(transect.distance, transect.trait_mean, yerr=se, fmt="o",
                 markersize=6, alpha=0.7, color="#2b6cb0", ecolor="#2b6cb0",
                 elinewidth=1, capsize=3, label="Observed (mean +/- SE)", zorder=3)

    x_smooth = np.linspace(transect.distance.min(), transect.distance.max(), 400)
    mp = fit.mean_params
    y_smooth = mean_cline(x_smooth, mp.center, mp.width, mp.pmin, mp.pmax, fit.tail_type,
                           mp.delta_l, mp.tau_l, mp.delta_r, mp.tau_r)
    ax.plot(x_smooth, y_smooth, color="#c53030", linewidth=2,
             label=f"Fitted cline ({fit.tail_type.value})", zorder=2)

    ax.set_xlabel("Geographic distance")
    ax.set_ylabel("Trait value")
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title is None:
        title = (f"center={fit.mean_params.center:.2f}, width={fit.mean_params.width:.2f}, "
                  f"AICc={fit.aicc:.2f}")
    ax.set_title(title)

    return ax


def plot_trait_multi_grid(results: dict, ncols: int = 3):
    """Small-multiples plot for multiple traits: one subplot per trait,
    from a dict of {trait_name: (transect, fit)} pairs -- the trait
    analog of `plot_multi_locus_grid`."""
    import math
    n = len(results)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, (trait_name, (transect, fit)) in enumerate(results.items()):
        ax = axes_flat[i]
        plot_trait_fit(transect, fit, ax=ax, title=trait_name)
        legend = ax.get_legend()
        if legend is not None and i > 0:
            legend.remove()

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    return fig


def plot_genomic_cline_fit(data, fit, ax=None, title: str | None = None):
    """Plot observed per-individual genotype data against a fitted
    genomic cline curve (hybrid index on the x-axis, rather than
    geographic distance -- see hycline.models.genomic_cline)."""
    from hycline.models.genomic_cline import genomic_cline_function

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    sizes = 20 + 60 * (data.n_alleles / data.n_alleles.max())
    ax.scatter(data.hybrid_index, data.observed_freq, s=sizes,
               alpha=0.6, color="#2b6cb0", edgecolor="white", linewidth=0.5,
               label="Observed", zorder=3)

    h_smooth = np.linspace(0.001, 0.999, 300)
    phi_smooth = genomic_cline_function(h_smooth, fit.params)
    ax.plot(h_smooth, phi_smooth, color="#c53030", linewidth=2,
             label="Fitted genomic cline", zorder=2)
    # neutral null expectation phi(h) = h, for visual reference
    ax.plot([0, 1], [0, 1], color="#a0aec0", linewidth=1, linestyle="--",
             label="Neutral (v=1, u=0)", zorder=1)

    ax.set_xlabel("Hybrid index")
    ax.set_ylabel("Locus ancestry (focal-allele frequency)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title is None:
        title = f"u={fit.params.u:.2f}, v={fit.params.v:.2f}, AICc={fit.aicc:.2f}"
    ax.set_title(title)

    return ax


def plot_genomic_multi_locus_grid(results: dict, ncols: int = 3):
    """Small-multiples plot: one subplot per locus, from a dict of
    {locus_name: (data, fit)} pairs -- the genomic-cline analog of
    plot_multi_locus_grid."""
    import math
    n = len(results)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, (locus_name, (data, fit)) in enumerate(results.items()):
        ax = axes_flat[i]
        plot_genomic_cline_fit(data, fit, ax=ax, title=locus_name)
        legend = ax.get_legend()
        if legend is not None and i > 0:
            legend.remove()

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    return fig


def plot_hierarchical_geographic_grid(distance, n_alleles, n_focal, locus_names,
                                       locus_summary_df, pmin_per_locus, pmax_per_locus,
                                       ncols: int = 3):
    """Small-multiples plot for a hierarchical geographic batch fit: one
    panel per locus, observed data plus the fitted curve built from that
    locus's posterior-mean center/width (with pmin/pmax at the fixed
    values the hierarchical fit used -- see
    hycline.inference.mcmc_hierarchical._derive_pmin_pmax). Point
    estimates only (posterior means), not full posterior uncertainty --
    for the full posterior, use the Python API's `HierarchicalFitResult.
    idata` directly."""
    import math
    n = len(locus_names)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    x_smooth = np.linspace(distance.min(), distance.max(), 200)
    for i, name in enumerate(locus_names):
        ax = axes_flat[i]
        obs_freq = n_focal[i] / n_alleles[i]
        sizes = 15 + 40 * (n_alleles[i] / n_alleles[i].max())
        ax.scatter(distance, obs_freq, s=sizes, alpha=0.6, color="#2b6cb0",
                   edgecolor="white", linewidth=0.5, zorder=2)

        row = locus_summary_df[locus_summary_df["locus"] == name].iloc[0]
        params = ClineParams(center=row["center_mean"], width=row["width_mean"],
                              pmin=pmin_per_locus[i], pmax=pmax_per_locus[i])
        y_smooth = cline_function(x_smooth, params, TailType.NONE)
        ax.plot(x_smooth, y_smooth, color="#c53030", linewidth=2, zorder=1)

        ax.set_title(name, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    return fig


def plot_hierarchical_genomic_grid(hybrid_index, n_alleles, n_focal, locus_names,
                                    locus_summary_df, ncols: int = 3):
    """Small-multiples plot for a hierarchical genomic batch fit -- the
    genomic-cline analog of `plot_hierarchical_geographic_grid`. No
    pmin/pmax needed (genomic clines have no free asymptotes)."""
    import math
    from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function

    n = len(locus_names)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    h_smooth = np.linspace(0.001, 0.999, 200)
    for i, name in enumerate(locus_names):
        ax = axes_flat[i]
        obs_freq = n_focal[i] / n_alleles[i]
        sizes = 15 + 30 * (n_alleles[i] / n_alleles[i].max())
        ax.scatter(hybrid_index, obs_freq, s=sizes, alpha=0.5, color="#2b6cb0",
                   edgecolor="white", linewidth=0.3, zorder=2)

        row = locus_summary_df[locus_summary_df["locus"] == name].iloc[0]
        params = GenomicClineParams(u=row["u_mean"], v=row["v_mean"])
        phi_smooth = genomic_cline_function(h_smooth, params)
        ax.plot(h_smooth, phi_smooth, color="#c53030", linewidth=2, zorder=1)
        ax.plot([0, 1], [0, 1], color="#a0aec0", linewidth=1, linestyle="--", zorder=0)

        ax.set_title(name, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.02, 1.02)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    return fig


def plot_gl_multi_locus_grid(results: dict, ncols: int = 3):
    """Small-multiples plot for independent (non-hierarchical) genotype-
    likelihood-aware geographic cline fits: one panel per locus, each
    with its own transect/fit -- unlike `plot_hierarchical_geographic_grid`,
    loci here don't need to share a common distance vector, since
    independent fitting (unlike Components A/D's hierarchical batch
    models) doesn't require dense, aligned data across loci.

    `results` is a dict of {locus_name: (GenotypeLikelihoodTransect, FitResult)}.
    Marker positions use each individual's most-likely genotype (argmax of
    the GL triple) for display only -- the fit itself used the full
    genotype-likelihood marginalization."""
    import math
    from hycline.models.cline import TailType, cline_function

    n = len(results)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, (locus_name, (transect, fit)) in enumerate(results.items()):
        ax = axes_flat[i]
        gl_stack = np.stack([transect.log_gl0, transect.log_gl1, transect.log_gl2])
        hard_genotype_proxy = np.argmax(gl_stack, axis=0)
        ax.scatter(transect.distance, hard_genotype_proxy / 2.0, s=20, alpha=0.5,
                   color="#2b6cb0", edgecolor="white", linewidth=0.4, zorder=2)

        x_smooth = np.linspace(transect.distance.min(), transect.distance.max(), 200)
        y_smooth = cline_function(x_smooth, fit.params, fit.tail_type)
        ax.plot(x_smooth, y_smooth, color="#c53030", linewidth=2, zorder=1)

        ax.set_title(locus_name, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    return fig


def plot_genomic_gl_multi_locus_grid(results: dict, ncols: int = 3):
    """Small-multiples plot for independent genotype-likelihood-aware
    genomic cline fits -- the genomic-cline analog of
    `plot_gl_multi_locus_grid`. `results` is a dict of
    {locus_name: (GenotypeLikelihoodLocusData, GenomicFitResult)}."""
    import math
    from hycline.models.genomic_cline import genomic_cline_function

    n = len(results)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    h_smooth = np.linspace(0.001, 0.999, 200)
    for i, (locus_name, (data, fit)) in enumerate(results.items()):
        ax = axes_flat[i]
        gl_stack = np.stack([data.log_gl0, data.log_gl1, data.log_gl2])
        hard_genotype_proxy = np.argmax(gl_stack, axis=0)
        ax.scatter(data.hybrid_index, hard_genotype_proxy / 2.0, s=20, alpha=0.5,
                   color="#2b6cb0", edgecolor="white", linewidth=0.3, zorder=2)

        phi_smooth = genomic_cline_function(h_smooth, fit.params)
        ax.plot(h_smooth, phi_smooth, color="#c53030", linewidth=2, zorder=1)
        ax.plot([0, 1], [0, 1], color="#a0aec0", linewidth=1, linestyle="--", zorder=0)

        ax.set_title(locus_name, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.02, 1.02)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    return fig
