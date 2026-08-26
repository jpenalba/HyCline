"""
Genotype-likelihood-aware fitting demo: comparing GL-aware fitting against
naive hard-genotype-calling across a range of sequencing depths.

The key finding this demonstrates (see BACKEND_PHASE5.md section 3 for the
full investigation): at low depth, both approaches give similar *point*
estimates, but GL-aware fitting's confidence intervals are properly
calibrated while hard-calling's are overconfident (too narrow) -- because
hard-calling discards real genotype uncertainty and then treats the
(possibly wrong) hard call as if it were certain data. At high depth,
where genotype calls are effectively certain, the two approaches converge.

Run with:
    python examples/demo_genotype_likelihood.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hycline.data.genotype_likelihood import simulate_genotype_likelihoods
from hycline.data.schema import GenotypeLikelihoodTransect, AlleleFreqTransect
from hycline.inference.mle import fit_mle
from hycline.inference.mle_genotype_likelihood import fit_mle_geographic_gl
from hycline.models.cline import ClineParams, TailType, cline_function

HERE = Path(__file__).parent


def simulate_true_genotypes(true_params, n_ind, seed):
    rng = np.random.default_rng(seed)
    distance = np.sort(rng.uniform(0, 100, n_ind))
    true_p = cline_function(distance, true_params, TailType.NONE)
    u = rng.uniform(size=n_ind)
    true_genotype = np.where(
        u < (1 - true_p) ** 2, 0,
        np.where(u < (1 - true_p) ** 2 + 2 * true_p * (1 - true_p), 1, 2))
    return distance, true_genotype


def run_at_depth(distance, true_genotype, depth, n_reps, true_params, error_rate=0.01):
    n_ind = len(distance)
    gl_covers = gl_ses = 0
    hard_covers = hard_ses = 0
    gl_n = hard_n = 0

    for rep in range(n_reps):
        gls = simulate_genotype_likelihoods(true_genotype, depth=depth, error_rate=error_rate,
                                             rng=np.random.default_rng(rep))
        transect_gl = GenotypeLikelihoodTransect(distance=distance, log_gl0=gls["log_gl0"],
                                                   log_gl1=gls["log_gl1"], log_gl2=gls["log_gl2"])
        fit_gl = fit_mle_geographic_gl(transect_gl, tail_type=TailType.NONE)

        gl_stack = np.stack([gls["log_gl0"], gls["log_gl1"], gls["log_gl2"]])
        hard_genotype = np.argmax(gl_stack, axis=0)
        transect_hard = AlleleFreqTransect(distance=distance, n_alleles=np.full(n_ind, 2.0),
                                            n_focal=hard_genotype.astype(float))
        fit_hard = fit_mle(transect_hard, tail_type=TailType.NONE)

        if fit_gl.std_errors is not None and np.isfinite(fit_gl.std_errors[1]):
            se = fit_gl.std_errors[1]
            lo, hi = fit_gl.params.width - 1.96 * se, fit_gl.params.width + 1.96 * se
            gl_covers += (lo <= true_params.width <= hi)
            gl_ses += se
            gl_n += 1
        if fit_hard.std_errors is not None and np.isfinite(fit_hard.std_errors[1]):
            se = fit_hard.std_errors[1]
            lo, hi = fit_hard.params.width - 1.96 * se, fit_hard.params.width + 1.96 * se
            hard_covers += (lo <= true_params.width <= hi)
            hard_ses += se
            hard_n += 1

    return dict(gl_coverage=gl_covers / gl_n, gl_mean_se=gl_ses / gl_n,
                hard_coverage=hard_covers / hard_n, hard_mean_se=hard_ses / hard_n)


def main():
    true_params = ClineParams(center=50.0, width=15.0, pmin=0.05, pmax=0.95)
    distance, true_genotype = simulate_true_genotypes(true_params, n_ind=200, seed=42)

    depths = [1, 2, 3, 5, 8, 15, 30]
    n_reps = 40

    print("=" * 74)
    print("HyCline genotype-likelihood-aware fitting demo")
    print(f"True cline: center={true_params.center}, width={true_params.width}")
    print(f"n_individuals=200, n_reps per depth={n_reps}")
    print("=" * 74)
    print(f"\n{'depth':>6}{'GL-aware coverage':>20}{'GL-aware mean SE':>18}"
          f"{'hard-call coverage':>20}{'hard-call mean SE':>18}")

    results = []
    for depth in depths:
        r = run_at_depth(distance, true_genotype, depth, n_reps, true_params)
        results.append(r)
        print(f"{depth:>6}{r['gl_coverage']:>19.1%}{r['gl_mean_se']:>18.2f}"
              f"{r['hard_coverage']:>19.1%}{r['hard_mean_se']:>18.2f}")

    print("\nExpected pattern: at low depth, GL-aware coverage should stay close to the "
          "nominal 95%, while hard-call coverage is noticeably lower (overconfident, too-"
          "narrow CIs). Both should converge to similar, well-calibrated coverage at high "
          "depth, where genotype calls become effectively certain either way.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(depths, [r["gl_coverage"] for r in results], "o-", color="#2b6cb0",
            label="GL-aware")
    ax.plot(depths, [r["hard_coverage"] for r in results], "o-", color="#c53030",
            label="Hard-call")
    ax.axhline(0.95, color="#a0aec0", linestyle="--", linewidth=1, label="Nominal 95%")
    ax.set_xlabel("Sequencing depth")
    ax.set_ylabel("95% CI coverage of true width")
    ax.set_title("Uncertainty calibration vs. depth")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(depths, [r["gl_mean_se"] for r in results], "o-", color="#2b6cb0",
            label="GL-aware")
    ax.plot(depths, [r["hard_mean_se"] for r in results], "o-", color="#c53030",
            label="Hard-call")
    ax.set_xlabel("Sequencing depth")
    ax.set_ylabel("Mean reported SE (cline width)")
    ax.set_title("Reported uncertainty vs. depth")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Genotype-likelihood-aware vs. hard-call fitting: calibration, not bias")
    fig.tight_layout()
    out_path = HERE / "genotype_likelihood_calibration.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
