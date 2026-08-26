"""
Phase 4, Component E demo: hierarchical genomic cline fitting with
genotype-likelihood marginalization, compared against "hard-call then
hierarchically pool" -- the genomic-cline analog of Component D.

Unlike Component D's dramatic finding for geographic clines, this shows a
genuinely different, more nuanced result (see BACKEND_PHASE4.md section
12 for the full investigation): at a moderate sparse/rich depth contrast,
GL-aware and hard-call hierarchical fitting perform *identically* for
genomic clines. Only at a more extreme contrast does a real (but modest)
difference emerge, and it looks like an overconfidence pattern rather
than Component D's "population-level contamination" mechanism.

Run with:
    python examples/demo_genomic_hierarchical_gl.py
"""

import json
from pathlib import Path

import numpy as np

from hycline.data.genotype_likelihood import simulate_genotype_likelihoods
from hycline.inference.mcmc_genomic_hierarchical import fit_hierarchical_genomic_batch
from hycline.inference.mcmc_genomic_hierarchical_gl import fit_hierarchical_genomic_batch_gl
from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function

HERE = Path(__file__).parent


def simulate_multi_locus_genomic_gl(seed, n_loci=6, n_ind=100, sparse_depth=2, rich_depth=8):
    rng = np.random.default_rng(seed)
    mu_u_true, sigma_u_true = 0.0, 1.0
    mu_log_v_true, sigma_log_v_true = 0.0, 0.5
    hybrid_index = rng.uniform(0.02, 0.98, n_ind)
    true_u = rng.normal(mu_u_true, sigma_u_true, n_loci)
    true_v = np.exp(rng.normal(mu_log_v_true, sigma_log_v_true, n_loci))
    depths = [sparse_depth] + [rich_depth] * (n_loci - 1)

    log_gl0 = np.zeros((n_loci, n_ind))
    log_gl1 = np.zeros((n_loci, n_ind))
    log_gl2 = np.zeros((n_loci, n_ind))
    n_alleles_hard = np.zeros((n_loci, n_ind))
    n_focal_hard = np.zeros((n_loci, n_ind))

    for j in range(n_loci):
        params = GenomicClineParams(u=true_u[j], v=true_v[j])
        phi = genomic_cline_function(hybrid_index, params)
        u_draw = rng.uniform(size=n_ind)
        true_geno = np.where(u_draw < (1 - phi) ** 2, 0,
                              np.where(u_draw < (1 - phi) ** 2 + 2 * phi * (1 - phi), 1, 2))
        gls = simulate_genotype_likelihoods(true_geno, depth=depths[j], error_rate=0.01,
                                             rng=np.random.default_rng(seed * 100 + j))
        log_gl0[j], log_gl1[j], log_gl2[j] = gls["log_gl0"], gls["log_gl1"], gls["log_gl2"]
        gl_stack = np.stack([gls["log_gl0"], gls["log_gl1"], gls["log_gl2"]])
        hard_geno = np.argmax(gl_stack, axis=0)
        n_alleles_hard[j] = 2.0
        n_focal_hard[j] = hard_geno

    return (hybrid_index, log_gl0, log_gl1, log_gl2, n_alleles_hard, n_focal_hard,
            true_u, true_v)


def hdi(draws, prob=0.95):
    return np.percentile(draws, [(100 - prob * 100) / 2, 100 - (100 - prob * 100) / 2])


def run_study(label, n_reps, n_ind, sparse_depth, rich_depth):
    print(f"\n--- {label} (sparse_depth={sparse_depth}, rich_depth={rich_depth}, "
          f"n_ind={n_ind}) ---")
    results = []
    for rep in range(n_reps):
        (hybrid_index, log_gl0, log_gl1, log_gl2, n_alleles_hard, n_focal_hard,
         true_u, true_v) = simulate_multi_locus_genomic_gl(
            seed=rep, n_ind=n_ind, sparse_depth=sparse_depth, rich_depth=rich_depth)

        fit_gl = fit_hierarchical_genomic_batch_gl(hybrid_index, log_gl0, log_gl1, log_gl2,
                                                    num_warmup=500, num_samples=500,
                                                    num_chains=2, seed=0)
        fit_hard = fit_hierarchical_genomic_batch(hybrid_index, n_alleles_hard, n_focal_hard,
                                                   num_warmup=500, num_samples=500,
                                                   num_chains=2, seed=0)

        v_gl = np.exp(fit_gl.idata.posterior["log_v"].isel(locus=0).values.flatten())
        v_hard = np.exp(fit_hard.idata.posterior["log_v"].isel(locus=0).values.flatten())
        lo_gl, hi_gl = hdi(v_gl)
        lo_hard, hi_hard = hdi(v_hard)

        results.append(dict(
            rep=rep, true_v=float(true_v[0]),
            v_cov_gl=bool(lo_gl <= true_v[0] <= hi_gl),
            v_cov_hard=bool(lo_hard <= true_v[0] <= hi_hard),
            v_point_gl=float(np.mean(v_gl)), v_point_hard=float(np.mean(v_hard)),
        ))
        print(f"  rep {rep}: true_v={true_v[0]:.2f}  "
              f"point_est GL={results[-1]['v_point_gl']:.2f} hard={results[-1]['v_point_hard']:.2f}  "
              f"cov GL={results[-1]['v_cov_gl']} hard={results[-1]['v_cov_hard']}")

    cov_gl = np.mean([r["v_cov_gl"] for r in results])
    cov_hard = np.mean([r["v_cov_hard"] for r in results])
    err_gl = np.mean([abs(r["v_point_gl"] - r["true_v"]) for r in results])
    err_hard = np.mean([abs(r["v_point_hard"] - r["true_v"]) for r in results])
    print(f"  Coverage: GL-aware={cov_gl:.1%}  hard-call={cov_hard:.1%}")
    print(f"  Mean |point estimate - true v|: GL-aware={err_gl:.3f}  hard-call={err_hard:.3f}")
    return results


def main():
    print("=" * 74)
    print("HyCline Phase 4 Component E demo -- genomic-cline analog of Component D")
    print("=" * 74)

    moderate = run_study("Moderate depth contrast (matches Component D's own setup)",
                          n_reps=8, n_ind=100, sparse_depth=2, rich_depth=8)
    extreme = run_study("Extreme depth contrast", n_reps=8, n_ind=60,
                         sparse_depth=1, rich_depth=20)

    print("\n" + "=" * 74)
    print("Summary: unlike Component D's dramatic geographic-cline finding, genotype-")
    print("likelihood-awareness makes little to no difference for genomic clines at a")
    print("moderate depth contrast under hierarchical pooling -- only at a more extreme")
    print("contrast does a modest difference emerge, and it looks like an overconfidence")
    print("pattern (narrower intervals relative to error) rather than Component D's")
    print("population-level-hyperparameter-contamination mechanism. See BACKEND_PHASE4.md")
    print("section 12 for the full investigation and why the two components' findings")
    print("genuinely differ, tied to real structural differences between the two models.")

    (HERE / "genomic_hierarchical_gl_results.json").write_text(
        json.dumps({"moderate": moderate, "extreme": extreme}, indent=2))


if __name__ == "__main__":
    main()
