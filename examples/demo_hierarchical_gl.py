"""
Phase 4, Component D demo: hierarchical geographic cline fitting with
genotype-likelihood marginalization, compared against the "hard-call then
hierarchically pool" alternative.

The question this answers (see BACKEND_PHASE4.md section 11 for the full
investigation): once you're already pooling information across loci
(Component A), does genotype-likelihood-awareness (Phase 5) still add
value for a sparse/low-depth locus, or does pooling alone already fix
what Phase 5 fixes for independent per-locus fits? Answer: pooling alone
is not enough -- and the mechanism is more consequential than expected: a
hard-called sparse locus's inflated estimate contaminates the *shared*
population-level hyperparameters, affecting every other locus in the
batch, not just the corrupted one.

Run with:
    python examples/demo_hierarchical_gl.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hycline.data.genotype_likelihood import simulate_genotype_likelihoods
from hycline.inference.mcmc_hierarchical import fit_hierarchical_batch
from hycline.inference.mcmc_hierarchical_gl import fit_hierarchical_batch_gl
from hycline.models.cline import ClineParams, TailType, cline_function

HERE = Path(__file__).parent


def simulate_multi_locus_gl(seed, n_loci=6, n_ind=100, sparse_depth=2, rich_depth=8):
    rng = np.random.default_rng(seed)
    mu_center_true, sigma_center_true = 75.0, 10.0
    mu_log_width_true, sigma_log_width_true = np.log(25.0), 0.25
    distance = np.sort(rng.uniform(0, 150, size=n_ind))
    true_centers = rng.normal(mu_center_true, sigma_center_true, n_loci)
    true_widths = np.exp(rng.normal(mu_log_width_true, sigma_log_width_true, n_loci))
    depths = [sparse_depth] + [rich_depth] * (n_loci - 1)

    log_gl0 = np.zeros((n_loci, n_ind))
    log_gl1 = np.zeros((n_loci, n_ind))
    log_gl2 = np.zeros((n_loci, n_ind))
    n_alleles_hard = np.zeros((n_loci, n_ind))
    n_focal_hard = np.zeros((n_loci, n_ind))

    for j in range(n_loci):
        params = ClineParams(center=true_centers[j], width=true_widths[j], pmin=0.05, pmax=0.95)
        p = cline_function(distance, params, TailType.NONE)
        u = rng.uniform(size=n_ind)
        true_geno = np.where(u < (1 - p) ** 2, 0,
                              np.where(u < (1 - p) ** 2 + 2 * p * (1 - p), 1, 2))
        gls = simulate_genotype_likelihoods(true_geno, depth=depths[j], error_rate=0.01,
                                             rng=np.random.default_rng(seed * 100 + j))
        log_gl0[j], log_gl1[j], log_gl2[j] = gls["log_gl0"], gls["log_gl1"], gls["log_gl2"]
        gl_stack = np.stack([gls["log_gl0"], gls["log_gl1"], gls["log_gl2"]])
        hard_geno = np.argmax(gl_stack, axis=0)
        n_alleles_hard[j] = 2.0
        n_focal_hard[j] = hard_geno

    return (distance, log_gl0, log_gl1, log_gl2, n_alleles_hard, n_focal_hard,
            true_centers, true_widths, depths)


def hdi(draws, prob=0.95):
    return np.percentile(draws, [(100 - prob * 100) / 2, 100 - (100 - prob * 100) / 2])


def main():
    n_reps = 12
    results = []

    print("=" * 74)
    print("HyCline Phase 4 Component D demo -- GL-aware vs. hard-call hierarchical fitting")
    print("6 loci, 100 individuals, locus_0 at depth=2 (sparse), rest at depth=8")
    print(f"Running {n_reps} independent replicate simulations ...")
    print("=" * 74)

    for rep in range(n_reps):
        (distance, log_gl0, log_gl1, log_gl2, n_alleles_hard, n_focal_hard,
         true_centers, true_widths, depths) = simulate_multi_locus_gl(seed=rep)

        fit_gl = fit_hierarchical_batch_gl(distance, log_gl0, log_gl1, log_gl2,
                                            num_warmup=500, num_samples=500,
                                            num_chains=2, seed=0)
        fit_hard = fit_hierarchical_batch(distance, n_alleles_hard, n_focal_hard,
                                           num_warmup=500, num_samples=500,
                                           num_chains=2, seed=0)

        width_gl = np.exp(fit_gl.idata.posterior["log_width"].isel(locus=0).values.flatten())
        width_hard = np.exp(fit_hard.idata.posterior["log_width"].isel(locus=0).values.flatten())
        center_gl = fit_gl.idata.posterior["center"].isel(locus=0).values.flatten()
        center_hard = fit_hard.idata.posterior["center"].isel(locus=0).values.flatten()

        lo_wg, hi_wg = hdi(width_gl)
        lo_wh, hi_wh = hdi(width_hard)
        lo_cg, hi_cg = hdi(center_gl)
        lo_ch, hi_ch = hdi(center_hard)

        mu_width_gl = float(np.exp(fit_gl.idata.posterior["mu_log_width"]).mean())
        mu_width_hard = float(np.exp(fit_hard.idata.posterior["mu_log_width"]).mean())

        results.append(dict(
            rep=rep, true_width=float(true_widths[0]), true_center=float(true_centers[0]),
            width_cov_gl=bool(lo_wg <= true_widths[0] <= hi_wg),
            width_cov_hard=bool(lo_wh <= true_widths[0] <= hi_wh),
            center_cov_gl=bool(lo_cg <= true_centers[0] <= hi_cg),
            center_cov_hard=bool(lo_ch <= true_centers[0] <= hi_ch),
            mu_width_gl=mu_width_gl, mu_width_hard=mu_width_hard,
        ))
        print(f"  rep {rep:2d}: width_cov  GL={results[-1]['width_cov_gl']!s:5}  "
              f"hard={results[-1]['width_cov_hard']!s:5}   "
              f"population mu_width  GL={mu_width_gl:5.1f}  hard={mu_width_hard:5.1f}")

    width_cov_gl = np.mean([r["width_cov_gl"] for r in results])
    width_cov_hard = np.mean([r["width_cov_hard"] for r in results])
    center_cov_gl = np.mean([r["center_cov_gl"] for r in results])
    center_cov_hard = np.mean([r["center_cov_hard"] for r in results])

    print(f"\n{'':25}{'GL-aware':>12}{'Hard-call':>12}")
    print(f"{'sparse-locus center coverage':25}{center_cov_gl:>11.1%}{center_cov_hard:>11.1%}")
    print(f"{'sparse-locus width coverage':25}{width_cov_gl:>11.1%}{width_cov_hard:>11.1%}")

    print("\nKey finding: hard-calling the sparse locus doesn't just distort its own "
          "estimate -- it visibly pulls the *shared* population-level mu_width upward "
          "(see the per-replicate table above), because the hierarchical model treats "
          "the corrupted sparse-locus estimate as real signal to pool. This means the "
          "cost of skipping genotype-likelihood-awareness isn't contained to the sparse "
          "locus alone once you're pooling across loci.")

    (HERE / "hierarchical_gl_results.json").write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    labels = ["center\ncoverage", "width\ncoverage"]
    gl_vals = [center_cov_gl, width_cov_gl]
    hard_vals = [center_cov_hard, width_cov_hard]
    x = np.arange(2)
    ax.bar(x - 0.18, gl_vals, width=0.35, label="GL-aware", color="#2b6cb0")
    ax.bar(x + 0.18, hard_vals, width=0.35, label="Hard-call", color="#c53030")
    ax.axhline(0.95, color="#a0aec0", linestyle="--", linewidth=1, label="Nominal 95%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("95% CI coverage (sparse locus)")
    ax.set_title(f"Coverage across {n_reps} replicates")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    mu_gl = [r["mu_width_gl"] for r in results]
    mu_hard = [r["mu_width_hard"] for r in results]
    ax.scatter(range(n_reps), mu_gl, color="#2b6cb0", label="GL-aware")
    ax.scatter(range(n_reps), mu_hard, color="#c53030", label="Hard-call")
    ax.axhline(25.0, color="#a0aec0", linestyle="--", linewidth=1, label="True mu_width")
    ax.set_xlabel("Replicate")
    ax.set_ylabel("Population-level mu_width (posterior mean)")
    ax.set_title("Population-level contamination from one sparse locus")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Component D: GL-awareness still matters even with hierarchical pooling")
    fig.tight_layout()
    out_path = HERE / "hierarchical_gl_demo.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
