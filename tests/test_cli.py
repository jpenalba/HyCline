import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
MANAKIN_CSV = REPO_ROOT / "examples" / "manakin_ada.csv"


def _run_cli(args, cwd=None):
    """Run the CLI as a subprocess (closest to how a real user/cluster job
    would invoke it) and return the completed process."""
    cmd = [sys.executable, str(REPO_ROOT / "run_hycline.py")] + args
    return subprocess.run(cmd, cwd=cwd or REPO_ROOT, capture_output=True, text=True)


def test_cli_help_runs_without_error():
    result = _run_cli(["--help"])
    assert result.returncode == 0
    assert "fit-geographic" in result.stdout
    assert "compare-tails" in result.stdout


def test_cli_fit_help_runs_without_error():
    result = _run_cli(["fit-geographic", "--help"])
    assert result.returncode == 0
    assert "--method" in result.stdout
    assert "--scaling" in result.stdout


def test_cli_fit_hzar_tail_recovers_known_cline_simulated_from_hzar_formula(tmp_path):
    import numpy as np
    from hycline.models.cline import ClineParams, TailType, hzar_cline_function

    rng = np.random.default_rng(1003)
    n_sites = 40
    distance = np.sort(rng.uniform(0, 200, n_sites))
    n_alleles = rng.integers(20, 50, n_sites)
    true_params = ClineParams(center=100, width=30, pmin=0.05, pmax=0.95,
                               delta_l=15.0, tau_l=0.5)
    p_true = hzar_cline_function(distance, true_params, TailType.MIRROR)
    n_focal = rng.binomial(n_alleles, np.clip(p_true, 0, 1))

    input_path = tmp_path / "genotypes.csv"
    pd.DataFrame({"distance": distance, "locus0.nSamples": n_alleles,
                  "locus0.A": n_focal / n_alleles}).to_csv(input_path, index=False)
    out_dir = tmp_path / "out"

    result = _run_cli([
        "fit-geographic", "--input", str(input_path), "--tail", "mirror", "--hzar-tail",
        "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr
    assert "HZAR-parameterized" in result.stdout

    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    row = summary_df.iloc[0]
    assert row["center"] == pytest.approx(100, abs=10)

    # Regression check for a real bug found while extending hzar_tail to
    # the Bayesian path: the fitted-curve CSV export unconditionally used
    # cline_function (the original, non-HZAR tail model) regardless of
    # fit.hzar_tail, present since --hzar-tail was first built for
    # classic MLE. Confirm the exported curve actually matches
    # hzar_cline_function at the fitted parameters, not cline_function
    # (which differ meaningfully here, since delta_l means a raw
    # distance under HZAR's parameterization but a [0,1] fraction under
    # cline_function's).
    from hycline.models.cline import cline_function
    curve_df = pd.read_csv(out_dir / "hycline_fitted_curves.csv")
    fitted_params = ClineParams(
        center=row["center"], width=row["width"], pmin=row["pmin"], pmax=row["pmax"],
        delta_l=row["delta_l"], tau_l=row["tau_l"],
    )
    expected_hzar = hzar_cline_function(curve_df["distance"].to_numpy(), fitted_params, TailType.MIRROR)
    expected_original = cline_function(curve_df["distance"].to_numpy(), fitted_params, TailType.MIRROR)
    assert not np.allclose(expected_hzar, expected_original, atol=1e-3)  # confirm they'd actually differ
    np.testing.assert_allclose(curve_df["fitted_freq"].to_numpy(), expected_hzar, atol=1e-6)


def test_cli_fit_geographic_t_cline_recovers_known_cline(tmp_path):
    """End-to-end CLI check for the T-cline model (Baird & Daley 2025):
    real subprocess invocation, real CSV I/O, real fitted-curve export --
    not just the direct-API check in test_mle.py."""
    import numpy as np
    from hycline.models.cline import ClineParams, TailType, t_cline_function

    rng = np.random.default_rng(2024)
    n_sites = 150
    distance = np.linspace(0.0, 100.0, n_sites)
    true_params = ClineParams(center=50.0, width=20.0, pmin=0.02, pmax=0.98, t_shape=1.5)
    p_true = t_cline_function(distance, true_params)
    n_alleles = np.full(n_sites, 200)
    n_focal = rng.binomial(n_alleles, p_true)

    input_path = tmp_path / "genotypes.csv"
    pd.DataFrame({"distance": distance, "locus0.nSamples": n_alleles,
                  "locus0.A": n_focal / n_alleles}).to_csv(input_path, index=False)
    out_dir = tmp_path / "out"

    result = _run_cli([
        "fit-geographic", "--input", str(input_path), "--tail", "t-cline",
        "--method", "classic", "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr
    assert "t_shape" in result.stdout

    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    row = summary_df.iloc[0]
    assert "t_shape" in summary_df.columns
    assert row["center"] == pytest.approx(50.0, abs=2.0)
    assert row["t_shape"] == pytest.approx(1.5, abs=0.6)

    # the exported fitted-curve CSV should match t_cline_function at the
    # fitted parameters (a real bug elsewhere in this project, caught only
    # by checking curve export against the fitting function directly
    # rather than assuming they agree -- see the hzar_tail test above)
    from hycline.models.cline import ClineParams as _CP
    curve_df = pd.read_csv(out_dir / "hycline_fitted_curves.csv")
    fitted_params = _CP(center=row["center"], width=row["width"],
                         pmin=row["pmin"], pmax=row["pmax"], t_shape=row["t_shape"])
    expected = t_cline_function(curve_df["distance"].to_numpy(), fitted_params)
    np.testing.assert_allclose(curve_df["fitted_freq"].to_numpy(), expected, atol=1e-6)


def test_cli_fit_geographic_t_cline_bayesian_gives_actionable_error(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(REPO_ROOT / "examples" / "manakin_multilocus.csv"),
        "--tail", "t-cline", "--method", "bayesian", "--output-dir", str(out_dir),
    ])
    assert result.returncode != 0
    assert "classic MLE only" in result.stderr


def test_cli_fit_geographic_t_cline_asymmetric_recovers_known_cline(tmp_path):
    """End-to-end CLI check for the asymmetric T(v,alpha) T-cline: real
    subprocess invocation, real CSV I/O, real fitted-curve export."""
    import numpy as np
    from hycline.models.cline import ClineParams, TailType, t_cline_function

    rng = np.random.default_rng(42)
    n_sites = 200
    distance = np.linspace(0.0, 150.0, n_sites)
    true_params = ClineParams(center=50.0, width=20.0, pmin=0.02, pmax=0.98,
                               t_shape=3.0, t_skew=2.5)
    p_true = t_cline_function(distance, true_params)
    n_alleles = np.full(n_sites, 150)
    n_focal = rng.binomial(n_alleles, p_true)

    input_path = tmp_path / "genotypes.csv"
    pd.DataFrame({"distance": distance, "locus0.nSamples": n_alleles,
                  "locus0.A": n_focal / n_alleles}).to_csv(input_path, index=False)
    out_dir = tmp_path / "out"

    result = _run_cli([
        "fit-geographic", "--input", str(input_path), "--tail", "t-cline",
        "--t-cline-asymmetric", "--method", "classic", "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr
    assert "t_skew" in result.stdout

    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    row = summary_df.iloc[0]
    assert "t_skew" in summary_df.columns
    assert row["center"] == pytest.approx(50.0, abs=3.0)
    assert row["width"] == pytest.approx(20.0, abs=5.0)

    curve_df = pd.read_csv(out_dir / "hycline_fitted_curves.csv")
    fitted_params = ClineParams(center=row["center"], width=row["width"],
                                 pmin=row["pmin"], pmax=row["pmax"],
                                 t_shape=row["t_shape"], t_skew=row["t_skew"])
    expected = t_cline_function(curve_df["distance"].to_numpy(), fitted_params)
    np.testing.assert_allclose(curve_df["fitted_freq"].to_numpy(), expected, atol=1e-6)


def test_cli_fit_geographic_t_cline_asymmetric_requires_t_cline_tail(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(REPO_ROOT / "examples" / "manakin_multilocus.csv"),
        "--tail", "none", "--t-cline-asymmetric", "--output-dir", str(out_dir),
    ])
    assert result.returncode != 0
    assert "only meaningful with --tail t-cline" in result.stderr


def test_cli_fit_geographic_t_cline_rejected_for_gl_input(tmp_path):
    """--tail t-cline is only wired into the hard-count classic-MLE path
    (hycline.inference.mle.fit_mle) -- genotype-likelihood input should
    get a clear, immediate error, not a confusing crash deep inside
    hycline.inference.mle_genotype_likelihood, which never learned about
    TailType.T_CLINE at all."""
    input_path = tmp_path / "gl_input.csv"
    _write_single_gl_geographic_csv(input_path)
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(input_path), "--tail", "t-cline",
        "--output-dir", str(out_dir),
    ])
    assert result.returncode != 0
    assert "does not support --tail t-cline" in result.stderr


def test_cli_compare_tails_includes_t_cline_candidate(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli([
        "compare-tails", "--input", str(MANAKIN_CSV),
        "--freq-col", "freq", "--site-col", "site",
        "--candidates", "none,mirror,t-cline",
        "--output-dir", str(out_dir), "--prefix", "cmp",
    ])
    assert result.returncode == 0, result.stderr

    csv_text = (out_dir / "cmp_model_comparison.csv").read_text()
    assert "t-cline" in csv_text


def test_cli_fit_bayesian_hzar_tail_recovers_known_cline_and_exports_correct_curve(tmp_path):
    """The Bayesian analog of the classic-MLE test above -- also checks
    the exported curve CSV directly, since this is exactly the kind of
    check that would have caught the curve-export bug (which affected
    both the classic and Bayesian paths, found and fixed together)."""
    import numpy as np
    from hycline.models.cline import ClineParams, TailType, hzar_cline_function, cline_function

    rng = np.random.default_rng(1007)
    n_sites = 30
    distance = np.sort(rng.uniform(0, 150, n_sites))
    n_alleles = rng.integers(20, 40, n_sites)
    true_params = ClineParams(center=70, width=25, pmin=0.05, pmax=0.95,
                               delta_l=10.0, tau_l=0.5)
    p_true = hzar_cline_function(distance, true_params, TailType.MIRROR)
    n_focal = rng.binomial(n_alleles, np.clip(p_true, 0, 1))

    input_path = tmp_path / "genotypes.csv"
    pd.DataFrame({"distance": distance, "locus0.nSamples": n_alleles,
                  "locus0.A": n_focal / n_alleles}).to_csv(input_path, index=False)
    out_dir = tmp_path / "out"

    result = _run_cli([
        "fit-geographic", "--input", str(input_path), "--method", "bayesian", "--tail", "mirror",
        "--hzar-tail", "--width-upper-bound", "150",
        "--num-warmup", "500", "--num-samples", "500", "--num-chains", "2",
        "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr
    run_config = json.loads((out_dir / "hycline_run_config.json").read_text())
    assert run_config["hzar_tail"] is True

    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    row = summary_df.iloc[0]
    assert row["center_mean"] == pytest.approx(70, abs=15)

    def _midpoint(name):
        return (row[f"{name}_credible_lower"] + row[f"{name}_credible_upper"]) / 2

    curve_df = pd.read_csv(out_dir / "hycline_fitted_curves.csv")
    fitted_params = ClineParams(
        center=row["center_mean"], width=row["width_mean"],
        pmin=_midpoint("pmin"), pmax=_midpoint("pmax"),
        # Bayesian posterior always samples delta_l/tau_l internally, even
        # for MIRROR (reused as the shared value) -- unlike classic MLE's
        # summary CSV, which labels these delta_m/tau_m for MIRROR via
        # _cline_param_names(). A real, pre-existing inconsistency between
        # the two paths' column naming, not introduced by this change
        # and not fixed here (out of scope for this pass).
        delta_l=_midpoint("delta_l"), tau_l=_midpoint("tau_l"),
    )
    # loose tolerance: the CSV's median curve isn't exactly the posterior-
    # mean-parameter curve (it's a per-x median across many posterior
    # draws), so this checks the two parameterizations are in the right
    # ballpark relative to each other, not exact equality.
    x = curve_df["distance"].to_numpy()
    expected_hzar = hzar_cline_function(x, fitted_params, TailType.MIRROR)
    expected_original = cline_function(x, fitted_params, TailType.MIRROR)
    hzar_err = np.mean(np.abs(curve_df["fitted_freq_median"].to_numpy() - expected_hzar))
    original_err = np.mean(np.abs(curve_df["fitted_freq_median"].to_numpy() - expected_original))
    assert hzar_err < original_err  # exported curve is much closer to the HZAR formula


def test_cli_fit_without_hzar_tail_flag_shows_no_hzar_label(tmp_path):
    import numpy as np
    from hycline.models.cline import ClineParams, TailType, cline_function

    rng = np.random.default_rng(1005)
    n_sites = 30
    distance = np.sort(rng.uniform(0, 100, n_sites))
    n_alleles = rng.integers(20, 40, n_sites)
    true_params = ClineParams(center=50, width=20, pmin=0.1, pmax=0.9, delta_l=0.15, tau_l=8.0)
    p_true = cline_function(distance, true_params, TailType.MIRROR)
    n_focal = rng.binomial(n_alleles, np.clip(p_true, 0, 1))

    input_path = tmp_path / "genotypes.csv"
    pd.DataFrame({"distance": distance, "locus0.nSamples": n_alleles,
                  "locus0.A": n_focal / n_alleles}).to_csv(input_path, index=False)
    out_dir = tmp_path / "out"

    result = _run_cli([
        "fit-geographic", "--input", str(input_path), "--tail", "mirror", "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr
    run_config = json.loads((out_dir / "hycline_run_config.json").read_text())
    assert run_config["hzar_tail"] is False


def test_cli_fit_classic_on_real_manakin_data(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(REPO_ROOT / "examples" / "manakin_multilocus.csv"),
        "--loci", "ada",
        "--method", "classic", "--scaling", "fixed",
        "--output-dir", str(out_dir), "--prefix", "test",
    ])
    assert result.returncode == 0, result.stderr

    summary_df = pd.read_csv(out_dir / "test_summary.csv")
    assert summary_df.loc[0, "converged"] == True  # noqa: E712
    assert summary_df.loc[0, "center"] == pytest.approx(170.9, abs=2.0)
    assert summary_df.loc[0, "width"] == pytest.approx(263.1, abs=3.0)

    assert (out_dir / "test_fitted_curves.csv").exists()
    assert (out_dir / "test_observed_data.csv").exists()
    assert (out_dir / "test_overlay.png").exists()
    assert (out_dir / "test_run_config.json").exists()


def test_cli_fit_bayesian_on_real_manakin_data(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(REPO_ROOT / "examples" / "manakin_multilocus.csv"),
        "--loci", "ada",
        "--method", "bayesian", "--scaling", "fixed",
        "--width-upper-bound", "569.5",
        "--num-warmup", "500", "--num-samples", "500", "--num-chains", "2",
        "--output-dir", str(out_dir), "--prefix", "test_bayes",
    ])
    assert result.returncode == 0, result.stderr

    summary_df = pd.read_csv(out_dir / "test_bayes_summary.csv")
    assert "pmin" not in summary_df.columns  # fixed, should not appear as a fitted param
    assert "pmax" not in summary_df.columns
    assert summary_df.loc[0, "center_mean"] == pytest.approx(170.9, abs=15.0)

    assert (out_dir / "test_bayes_fitted_curves.csv").exists()
    curves_csv = (out_dir / "test_bayes_fitted_curves.csv").read_text()
    assert "fitted_freq_median" in curves_csv


def _write_tail_hard_count_csv(path, seed=41, n_sites=50):
    import numpy as np
    import pandas as pd
    from hycline.models.cline import ClineParams, TailType, cline_function

    rng = np.random.default_rng(seed)
    distance = np.sort(rng.uniform(0, 100, n_sites))
    params = ClineParams(center=50, width=15, pmin=0.05, pmax=0.95, delta_l=0.15, tau_l=10.0)
    p = cline_function(distance, params, TailType.LEFT)
    n_alleles = np.full(n_sites, 20.0)
    n_focal = rng.binomial(20, p).astype(float)
    pd.DataFrame({"distance": distance, "locus0.nSamples": n_alleles,
                  "locus0.A": n_focal / n_alleles}).to_csv(path, index=False)


def test_cli_fit_bayesian_tail_model_does_not_crash_and_includes_tail_params(tmp_path):
    """Regression test for a real, previously-shipped bug found while
    wiring up Bayesian tail-model support (BACKEND_PHASE2.md section
    12.5): the fitted-curve/posterior-samples construction hardcoded a
    ClineParams with delta_l=None etc, which would break for any
    non-NONE tail type once Bayesian tail fitting was actually enabled.
    Confirms the fix: the command completes without error, and the
    summary CSV includes the tail parameters' credible intervals."""
    csv_path = tmp_path / "tail_hard.csv"
    _write_tail_hard_count_csv(csv_path)

    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(csv_path), "--method", "bayesian", "--tail", "mirror",
        "--num-warmup", "150", "--num-samples", "150", "--num-chains", "2",
        "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr
    assert "known, real convergence difficulty" in result.stderr  # proactive warning fired

    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    assert any(col.startswith("delta_l") for col in summary_df.columns)
    assert any(col.startswith("tau_l") for col in summary_df.columns)

    curves_csv = (out_dir / "hycline_fitted_curves.csv").read_text()
    assert "fitted_freq_median" in curves_csv


def test_cli_fit_multi_bayesian_tail_model_does_not_crash(tmp_path):
    """Same regression as above, for the multi-locus path
    (cmd_fit_multi's bayesian branch had the identical bug)."""
    import numpy as np
    import pandas as pd
    from hycline.models.cline import ClineParams, TailType, cline_function

    rng = np.random.default_rng(43)
    distance = np.sort(rng.uniform(0, 100, 40))
    n_alleles = np.full(40, 20.0)
    df = {"distance": distance}
    params = ClineParams(center=45, width=14, pmin=0.05, pmax=0.95, delta_l=0.1, tau_l=8.0)
    p = cline_function(distance, params, TailType.LEFT)
    df["locusA.nSamples"] = n_alleles
    df["locusA.A"] = rng.binomial(20, p).astype(float) / 20.0
    csv_path = tmp_path / "tail_multi.csv"
    pd.DataFrame(df).to_csv(csv_path, index=False)

    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(csv_path), "--method", "bayesian", "--tail", "mirror",
        "--num-warmup", "150", "--num-samples", "150", "--num-chains", "2",
        "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr

    summary_csv = (out_dir / "hycline_summary.csv").read_text()
    assert "delta_l_credible_lower" in summary_csv or "delta_m" in summary_csv.lower() \
        or "delta_l" in summary_csv


def test_cli_fit_multi_classic_tail_model_includes_tail_values_in_csv(tmp_path):
    """Regression test for a real, previously-shipped bug (not related to
    this turn's Bayesian work): fit-geographic-multi's classic-method summary CSV
    never included tail parameter *values* (only, inconsistently, their
    standard errors) -- meaning `fit-geographic-multi --tail mirror` had silently
    dropped delta_l/tau_l from its output CSV since Phase 1 shipped."""
    import numpy as np
    import pandas as pd
    from hycline.models.cline import ClineParams, TailType, cline_function

    rng = np.random.default_rng(45)
    distance = np.sort(rng.uniform(0, 100, 40))
    n_alleles = np.full(40, 20.0)
    df = {"distance": distance}
    params = ClineParams(center=45, width=14, pmin=0.05, pmax=0.95, delta_l=0.1, tau_l=8.0)
    p = cline_function(distance, params, TailType.LEFT)
    df["locusA.nSamples"] = n_alleles
    df["locusA.A"] = rng.binomial(20, p).astype(float) / 20.0
    csv_path = tmp_path / "tail_multi.csv"
    pd.DataFrame(df).to_csv(csv_path, index=False)

    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(csv_path), "--method", "classic", "--tail", "mirror",
        "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr

    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    assert "delta_l" in summary_df.columns
    assert "tau_l" in summary_df.columns
    assert not summary_df["delta_l"].isna().all()


def test_cli_fit_bayesian_scaling_fixed_actually_fixes_pmin_pmax(tmp_path):
    """Regression test for a real bug caught during CLI testing: --scaling
    fixed without explicit --pmin/--pmax must derive fixed values from the
    data (matching the classic path's own default), not silently fall back
    to freely estimating pmin/pmax. See hycline/cli.py's cmd_fit for the
    fix (deriving observed extremes before calling fit_mcmc)."""
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(REPO_ROOT / "examples" / "manakin_multilocus.csv"),
        "--loci", "ada",
        "--method", "bayesian", "--scaling", "fixed",
        "--width-upper-bound", "569.5",
        "--num-warmup", "500", "--num-samples", "500", "--num-chains", "2",
        "--output-dir", str(out_dir), "--prefix", "regress",
    ])
    assert result.returncode == 0, result.stderr

    summary_df = pd.read_csv(out_dir / "regress_summary.csv")
    assert "pmin" not in summary_df.columns and "pmax" not in summary_df.columns
    assert "center_mean" in summary_df.columns and "width_mean" in summary_df.columns


def test_cli_compare_tails_on_real_manakin_data(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli([
        "compare-tails", "--input", str(MANAKIN_CSV),
        "--freq-col", "freq", "--site-col", "site",
        "--candidates", "none,mirror",
        "--output-dir", str(out_dir), "--prefix", "cmp",
    ])
    assert result.returncode == 0, result.stderr

    csv_text = (out_dir / "cmp_model_comparison.csv").read_text()
    assert "none" in csv_text and "mirror" in csv_text
    assert "AICc" in csv_text


def test_cli_gives_actionable_error_on_missing_column(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(REPO_ROOT / "examples" / "manakin_multilocus.csv"),
        "--loci", "this_locus_does_not_exist",
        "--output-dir", str(out_dir),
    ])
    assert result.returncode != 0
    assert "this_locus_does_not_exist" in result.stderr
    assert "Loci detected in" in result.stderr  # helpful, not just a traceback


def test_cli_gives_actionable_error_on_missing_file(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(tmp_path / "nonexistent.csv"),
        "--output-dir", str(out_dir),
    ])
    assert result.returncode != 0
    assert "not found" in result.stderr


def test_cli_fit_with_coordinates_and_n_ind(tmp_path):
    csv_path = tmp_path / "coord_data.csv"
    csv_path.write_text(
        "site,lat,lon,locusA.n_ind,locusA.n_focal\n"
        "A,40.00,-74.00,10,1\n"
        "B,40.25,-74.00,20,9\n"
        "C,40.50,-74.00,15,14\n"
    )
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(csv_path),
        "--lat-col", "lat", "--lon-col", "lon", "--site-col", "site",
        "--n-ind-suffix", ".n_ind", "--n-focal-suffix", ".n_focal", "--ploidy", "2",
        "--method", "classic",
        "--output-dir", str(out_dir), "--prefix", "coordtest",
    ])
    assert result.returncode == 0, result.stderr

    observed = (out_dir / "coordtest_observed_data.csv").read_text()
    assert "distance" in observed
    # n_alleles should be double n_ind (ploidy=2): 20, 40, 30
    assert "20.0" in observed and "40.0" in observed and "30.0" in observed


def test_cli_fit_multi_classic_on_real_hzar_multilocus_data(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(REPO_ROOT / "examples" / "manakin_multilocus.csv"),
        "--site-col", "locationID",
        "--method", "classic", "--scaling", "fixed",
        "--output-dir", str(out_dir), "--prefix", "multi",
    ])
    assert result.returncode == 0, result.stderr

    summary_csv = (out_dir / "multi_summary.csv").read_text()
    for locus in ["ada", "ak2", "gsr", "pgm2", "l5", "pscn3", "mtDNA"]:
        assert locus in summary_csv

    import csv as csv_module
    with open(out_dir / "multi_summary.csv") as f:
        rows = {r["locus"]: r for r in csv_module.DictReader(f)}
    assert float(rows["ada"]["center"]) == pytest.approx(170.9, abs=2.0)
    assert float(rows["ada"]["width"]) == pytest.approx(263.1, abs=3.0)

    assert (out_dir / "multi_fitted_curves.csv").exists()
    assert (out_dir / "multi_observed_data.csv").exists()
    assert (out_dir / "multi_overlay.png").exists()


def test_cli_fit_multi_restricts_to_requested_loci(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(REPO_ROOT / "examples" / "manakin_multilocus.csv"),
        "--site-col", "locationID", "--loci", "ada,ak2",
        "--method", "classic",
        "--output-dir", str(out_dir), "--prefix", "subset",
    ])
    assert result.returncode == 0, result.stderr

    summary_csv = (out_dir / "subset_summary.csv").read_text()
    assert "ada" in summary_csv and "ak2" in summary_csv
    assert "gsr" not in summary_csv


def test_cli_fit_multi_gives_actionable_error_when_no_loci_detected(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("distance,foo,bar\n0,1,2\n")
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(csv_path),
        "--output-dir", str(out_dir),
    ])
    assert result.returncode != 0
    assert "could not detect the input type" in result.stderr


# ---------------------------------------------------------------------------
# fit-genomic / fit-genomic-multi
# ---------------------------------------------------------------------------

SYNTHETIC_GENOMIC_CSV = REPO_ROOT / "examples" / "synthetic_genomic_clines.csv"


def test_cli_fit_genomic_help_runs_without_error():
    result = _run_cli(["fit-genomic", "--help"])
    assert result.returncode == 0
    assert "--hybrid-index-col" in result.stdout


def test_cli_fit_genomic_recovers_known_barrier_locus(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-genomic", "--input", str(SYNTHETIC_GENOMIC_CSV),
        "--hybrid-index-col", "hybrid_index", "--individual-col", "individual",
        "--loci", "barrier1",
        "--output-dir", str(out_dir), "--prefix", "barrier",
    ])
    assert result.returncode == 0, result.stderr

    summary_df = pd.read_csv(out_dir / "barrier_summary.csv")
    assert summary_df.loc[0, "v"] > 2.0  # true v=5.0

    assert (out_dir / "barrier_fitted_curves.csv").exists()
    assert (out_dir / "barrier_overlay.png").exists()
    assert (out_dir / "barrier_run_config.json").exists()


def test_cli_fit_genomic_multi_fits_all_synthetic_loci(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-genomic", "--input", str(SYNTHETIC_GENOMIC_CSV),
        "--hybrid-index-col", "hybrid_index", "--individual-col", "individual",
        "--output-dir", str(out_dir), "--prefix", "all",
    ])
    assert result.returncode == 0, result.stderr

    summary_csv = (out_dir / "all_summary.csv").read_text()
    for locus in ["neutral1", "neutral2", "barrier1", "permeable1", "biased1"]:
        assert locus in summary_csv

    import csv as csv_module
    with open(out_dir / "all_summary.csv") as f:
        rows = {r["locus"]: r for r in csv_module.DictReader(f)}
    assert float(rows["barrier1"]["v"]) > 2.0
    assert float(rows["permeable1"]["v"]) < 0.7
    assert float(rows["biased1"]["u"]) > 1.0

    assert (out_dir / "all_fitted_curves.csv").exists()
    assert (out_dir / "all_overlay.png").exists()


def test_cli_fit_genomic_multi_restricts_to_requested_loci(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-genomic", "--input", str(SYNTHETIC_GENOMIC_CSV),
        "--individual-col", "individual", "--loci", "barrier1,permeable1",
        "--output-dir", str(out_dir), "--prefix", "subset",
    ])
    assert result.returncode == 0, result.stderr

    summary_csv = (out_dir / "subset_summary.csv").read_text()
    assert "barrier1" in summary_csv and "permeable1" in summary_csv
    assert "neutral1" not in summary_csv


def test_cli_fit_genomic_gives_actionable_error_on_bad_hybrid_index(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("hybrid_index,locus0.n,locus0.freq\n1.5,2,0.5\n")
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-genomic", "--input", str(csv_path),
        "--output-dir", str(out_dir),
    ])
    assert result.returncode != 0
    assert "outside [0, 1]" in result.stderr


# ---------------------------------------------------------------------------
# fit-hierarchical / fit-hierarchical-genomic (Phase 4 Components A/B CLI wiring)
# ---------------------------------------------------------------------------

MULTILOCUS_CSV = REPO_ROOT / "examples" / "manakin_multilocus.csv"


def test_cli_fit_hierarchical_help_runs_without_error():
    result = _run_cli(["fit-hierarchical-geographic", "--help"])
    assert result.returncode == 0
    assert "--sigma-center-scale" in result.stdout
    # should NOT show flags that don't apply to hierarchical fitting
    assert "--backend" not in result.stdout
    assert "--width-upper-bound" not in result.stdout


def test_cli_fit_hierarchical_genomic_help_runs_without_error():
    result = _run_cli(["fit-hierarchical-genomic", "--help"])
    assert result.returncode == 0
    assert "--sigma-log-v-scale" in result.stdout
    assert "--backend" not in result.stdout


def test_cli_fit_hierarchical_runs_on_real_manakin_multilocus_data(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-hierarchical-geographic", "--input", str(MULTILOCUS_CSV),
        "--site-col", "locationID",
        "--num-warmup", "300", "--num-samples", "300", "--num-chains", "2",
        "--output-dir", str(out_dir), "--prefix", "manakin",
    ])
    assert result.returncode == 0, result.stderr

    locus_summary = out_dir / "manakin_hierarchical_locus_summary.csv"
    assert locus_summary.exists()
    content = locus_summary.read_text()
    for locus in ["ada", "ak2", "gsr", "pgm2", "l5", "pscn3", "mtDNA"]:
        assert locus in content

    assert (out_dir / "manakin_hierarchical_population_summary.csv").exists()
    assert (out_dir / "manakin_hierarchical_fitted_curves.csv").exists()
    assert (out_dir / "manakin_hierarchical_overlay.png").exists()
    assert (out_dir / "manakin_run_config.json").exists()

    summary_json = json.loads((out_dir / "manakin_hierarchical_summary.json").read_text())
    assert summary_json["n_loci"] == 7
    assert "convergence" in summary_json


def test_cli_fit_hierarchical_genomic_recovers_known_barrier_locus(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-hierarchical-genomic", "--input", str(SYNTHETIC_GENOMIC_CSV),
        "--individual-col", "individual",
        "--num-warmup", "400", "--num-samples", "400", "--num-chains", "2",
        "--output-dir", str(out_dir), "--prefix", "synth",
    ])
    assert result.returncode == 0, result.stderr

    import csv as csv_module
    with open(out_dir / "synth_hierarchical_locus_summary.csv") as f:
        rows = {r["locus"]: r for r in csv_module.DictReader(f)}
    assert float(rows["barrier1"]["v_mean"]) > 2.0  # true v=5.0
    assert float(rows["permeable1"]["v_mean"]) < 0.7  # true v=0.3
    assert float(rows["biased1"]["u_mean"]) > 1.0  # true u=2.5

    assert (out_dir / "synth_hierarchical_population_summary.csv").exists()
    assert (out_dir / "synth_hierarchical_overlay.png").exists()


def test_cli_fit_hierarchical_genomic_pool_center_help_shows_new_flags():
    result = _run_cli(["fit-hierarchical-genomic", "--help"])
    assert result.returncode == 0
    assert "--pool-center" in result.stdout
    assert "--mu-center-scale" in result.stdout
    assert "--sigma-center-scale" in result.stdout


def test_cli_fit_hierarchical_genomic_pool_center_runs_and_reports_sigma_center(tmp_path):
    """Confirms the --pool-center CLI plumbing works end to end: the
    population summary CSV reports sigma_center (not sigma_u), and the
    per-locus summary still includes u_mean/v_mean for downstream
    compatibility regardless of which parameterization was used."""
    import numpy as np
    import pandas as pd
    from hycline.models.genomic_cline import genomic_cline_function, GenomicClineParams

    rng = np.random.default_rng(211)
    n_loci, n_ind = 6, 60
    hybrid_index = rng.uniform(0.02, 0.98, n_ind)
    sample_names = [f"ind{i}" for i in range(n_ind)]
    logit_centers_true = rng.normal(0, 0.9, n_loci)
    log_v_true = rng.normal(0, 0.3, n_loci)
    v_true = np.exp(log_v_true)
    u_true = v_true * logit_centers_true

    df = {"individual": sample_names, "hybrid_index": hybrid_index}
    for i in range(n_loci):
        phi = genomic_cline_function(hybrid_index, GenomicClineParams(u=u_true[i], v=v_true[i]))
        n_focal = rng.binomial(2, np.clip(phi, 0, 1))
        df[f"locus{i}.n"] = 2
        df[f"locus{i}.freq"] = n_focal / 2.0
    input_path = tmp_path / "genomic.csv"
    pd.DataFrame(df).to_csv(input_path, index=False)

    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-hierarchical-genomic", "--input", str(input_path), "--individual-col", "individual",
        "--pool-center", "--num-warmup", "400", "--num-samples", "400", "--num-chains", "2",
        "--no-plot-grid", "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr

    pop_df = pd.read_csv(out_dir / "hycline_hierarchical_population_summary.csv",
                          index_col=0)
    assert "sigma_center" in pop_df.index
    assert "sigma_u" not in pop_df.index

    locus_df = pd.read_csv(out_dir / "hycline_hierarchical_locus_summary.csv")
    assert "center_mean" in locus_df.columns
    assert "u_mean" in locus_df.columns  # still present regardless of parameterization


def test_cli_fit_hierarchical_genomic_default_still_reports_sigma_u(tmp_path):
    """Confirms --pool-center is opt-in: without it, the population
    summary still reports sigma_u (the original, unchanged default)."""
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-hierarchical-genomic", "--input", str(SYNTHETIC_GENOMIC_CSV),
        "--individual-col", "individual",
        "--num-warmup", "300", "--num-samples", "300", "--num-chains", "2",
        "--no-plot-grid", "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr
    pop_df = pd.read_csv(out_dir / "hycline_hierarchical_population_summary.csv",
                          index_col=0)
    assert "sigma_u" in pop_df.index
    assert "sigma_center" not in pop_df.index


def test_cli_fit_hierarchical_gives_actionable_error_on_missing_data(tmp_path):
    import pandas as pd
    import numpy as np

    df = pd.read_csv(MULTILOCUS_CSV)
    df.loc[0, "ada.A"] = np.nan
    csv_path = tmp_path / "manakin_missing.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-hierarchical-geographic", "--input", str(csv_path), "--site-col", "locationID",
        "--output-dir", str(out_dir),
    ])
    assert result.returncode != 0
    assert "hierarchical batch fitting requires every locus" in result.stderr
    assert "fit-geographic" in result.stderr


def test_cli_fit_hierarchical_genomic_missing_data_needs_individual_names(tmp_path):
    """Loci missing at different individuals are lined up by individual
    name; without an individual column there is nothing to line them up by,
    so the command stops with an actionable error."""
    import pandas as pd
    import numpy as np

    df = pd.read_csv(SYNTHETIC_GENOMIC_CSV)
    df.loc[0, "barrier1.freq"] = np.nan
    df.loc[0, "barrier1.n"] = 0
    with_names = tmp_path / "synth_missing.csv"
    df.to_csv(with_names, index=False)
    without_names = tmp_path / "synth_missing_no_names.csv"
    df.drop(columns="individual").to_csv(without_names, index=False)

    result = _run_cli([
        "fit-hierarchical-genomic", "--input", str(without_names),
        "--output-dir", str(tmp_path / "out1"),
    ])
    assert result.returncode != 0
    assert "hierarchical batch fitting requires every locus" in result.stderr
    assert "--individual-col" in result.stderr and "fit-genomic" in result.stderr

    result = _run_cli([
        "fit-hierarchical-genomic", "--input", str(with_names),
        "--individual-col", "individual", "--num-warmup", "150", "--num-samples", "150",
        "--num-chains", "1", "--no-plot", "--output-dir", str(tmp_path / "out2"),
    ])
    assert result.returncode == 0, result.stderr
    assert "1 missing genotype(s) enter with 0 sampled alleles" in result.stdout


# ---------------------------------------------------------------------------
# fit-hierarchical-gl / fit-hierarchical-genomic-gl (Phase 4 Components D/E)
# coupling-geographic / coupling-genomic (Phase 4 Component C)
# ---------------------------------------------------------------------------

def _write_gl_geographic_csv(path, seed=21, n_ind=60):
    import numpy as np
    import pandas as pd
    from hycline.models.cline import ClineParams, TailType, cline_function
    from hycline.data.genotype_likelihood import simulate_genotype_likelihoods

    rng = np.random.default_rng(seed)
    distance = np.sort(rng.uniform(0, 100, n_ind))
    df = {"individual": [f"ind{i}" for i in range(n_ind)], "distance": distance}
    for locus, (center, width, depth) in [("locusA", (50, 15, 10)), ("locusB", (45, 12, 10))]:
        params = ClineParams(center=center, width=width, pmin=0.05, pmax=0.95)
        p = cline_function(distance, params, TailType.NONE)
        u = rng.uniform(size=n_ind)
        true_geno = np.where(u < (1 - p) ** 2, 0,
                              np.where(u < (1 - p) ** 2 + 2 * p * (1 - p), 1, 2))
        gls = simulate_genotype_likelihoods(true_geno, depth=depth, error_rate=0.01, rng=rng)
        df[f"{locus}.gl0"] = gls["log_gl0"]
        df[f"{locus}.gl1"] = gls["log_gl1"]
        df[f"{locus}.gl2"] = gls["log_gl2"]
    pd.DataFrame(df).to_csv(path, index=False)


def _write_gl_genomic_csv(path, seed=23, n_ind=80):
    import numpy as np
    import pandas as pd
    from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function
    from hycline.data.genotype_likelihood import simulate_genotype_likelihoods

    rng = np.random.default_rng(seed)
    hybrid_index = rng.uniform(0.02, 0.98, n_ind)
    df = {"individual": [f"ind{i}" for i in range(n_ind)], "hybrid_index": hybrid_index}
    for locus, (u_true, v_true, depth) in [("barrierX", (0.0, 4.0, 10)), ("neutralX", (0.0, 1.0, 10))]:
        params = GenomicClineParams(u=u_true, v=v_true)
        phi = genomic_cline_function(hybrid_index, params)
        uu = rng.uniform(size=n_ind)
        true_geno = np.where(uu < (1 - phi) ** 2, 0,
                              np.where(uu < (1 - phi) ** 2 + 2 * phi * (1 - phi), 1, 2))
        gls = simulate_genotype_likelihoods(true_geno, depth=depth, error_rate=0.01, rng=rng)
        df[f"{locus}.gl0"] = gls["log_gl0"]
        df[f"{locus}.gl1"] = gls["log_gl1"]
        df[f"{locus}.gl2"] = gls["log_gl2"]
    pd.DataFrame(df).to_csv(path, index=False)


def test_cli_fit_hierarchical_geographic_help_mentions_gl_suffix_flags():
    result = _run_cli(["fit-hierarchical-geographic", "--help"])
    assert result.returncode == 0
    assert "--gl0-suffix" in result.stdout
    assert "--phred" in result.stdout


def test_cli_fit_hierarchical_genomic_help_mentions_gl_suffix_flags():
    result = _run_cli(["fit-hierarchical-genomic", "--help"])
    assert result.returncode == 0
    assert "--gl0-suffix" in result.stdout


def test_cli_coupling_geographic_help_runs_without_error():
    result = _run_cli(["coupling-geographic", "--help"])
    assert result.returncode == 0
    assert "--width-col" in result.stdout


def test_cli_coupling_genomic_help_runs_without_error():
    result = _run_cli(["coupling-genomic", "--help"])
    assert result.returncode == 0
    assert "--u-col" in result.stdout
    assert "--no-theta" in result.stdout


def test_cli_fit_hierarchical_geographic_gl_runs_end_to_end(tmp_path):
    csv_path = tmp_path / "gl_geo.csv"
    _write_gl_geographic_csv(csv_path)

    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-hierarchical-geographic", "--input", str(csv_path), "--individual-col", "individual",
        "--num-warmup", "300", "--num-samples", "300", "--num-chains", "2",
        "--output-dir", str(out_dir), "--prefix", "test",
    ])
    assert result.returncode == 0, result.stderr

    locus_summary = out_dir / "test_hierarchical_locus_summary.csv"
    assert locus_summary.exists()
    content = locus_summary.read_text()
    assert "locusA" in content and "locusB" in content
    assert (out_dir / "test_hierarchical_population_summary.csv").exists()
    assert (out_dir / "test_hierarchical_overlay.png").exists()


def test_cli_fit_hierarchical_genomic_gl_runs_end_to_end(tmp_path):
    csv_path = tmp_path / "gl_genomic.csv"
    _write_gl_genomic_csv(csv_path)

    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-hierarchical-genomic", "--input", str(csv_path),
        "--individual-col", "individual",
        "--num-warmup", "300", "--num-samples", "300", "--num-chains", "2",
        "--output-dir", str(out_dir), "--prefix", "test",
    ])
    assert result.returncode == 0, result.stderr

    import csv as csv_module
    with open(out_dir / "test_hierarchical_locus_summary.csv") as f:
        rows = {r["locus"]: r for r in csv_module.DictReader(f)}
    assert float(rows["barrierX"]["v_mean"]) > 1.5  # true v=4.0
    assert (out_dir / "test_hierarchical_overlay.png").exists()


def test_cli_fit_hierarchical_geographic_gl_actionable_error_on_missing_data(tmp_path):
    import pandas as pd
    import numpy as np

    csv_path = tmp_path / "gl_geo.csv"
    _write_gl_geographic_csv(csv_path)
    df = pd.read_csv(csv_path)
    df.loc[0, "locusA.gl0"] = np.nan
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-hierarchical-geographic", "--input", str(csv_path), "--individual-col", "individual",
        "--output-dir", str(out_dir),
    ])
    assert result.returncode != 0
    assert "hierarchical batch fitting requires every locus" in result.stderr


def test_cli_coupling_geographic_end_to_end(tmp_path):
    summary_csv = tmp_path / "locus_summary.csv"
    summary_csv.write_text(
        "locus,width_mean,center_mean\n"
        "locusA,15.0,50.0\n"
        "locusB,16.5,48.0\n"
        "locusC,14.0,52.0\n"
    )
    out_dir = tmp_path / "out"
    result = _run_cli([
        "coupling-geographic", "--input", str(summary_csv),
        "--output-dir", str(out_dir), "--prefix", "cov",
    ])
    assert result.returncode == 0, result.stderr
    assert "width_mean" in result.stdout

    summary_json = json.loads((out_dir / "cov_coupling_geographic_summary.json").read_text())
    assert summary_json["n_loci"] == 3
    assert summary_json["width_col_used"] == "width_mean"
    assert "sd_slope" in summary_json


def test_cli_coupling_genomic_end_to_end(tmp_path):
    summary_csv = tmp_path / "locus_summary.csv"
    summary_csv.write_text(
        "locus,u_mean,v_mean\n"
        "locusA,0.1,1.0\n"
        "locusB,-0.1,1.1\n"
        "locusC,0.05,0.95\n"
    )
    out_dir = tmp_path / "out"
    result = _run_cli([
        "coupling-genomic", "--input", str(summary_csv),
        "--output-dir", str(out_dir), "--prefix", "cov",
    ])
    assert result.returncode == 0, result.stderr

    summary_json = json.loads((out_dir / "cov_coupling_genomic_summary.json").read_text())
    assert summary_json["n_loci"] == 3
    assert summary_json["u_col_used"] == "u_mean"
    assert summary_json["theta_estimate"] is not None


def test_cli_coupling_genomic_no_theta_flag(tmp_path):
    summary_csv = tmp_path / "locus_summary.csv"
    summary_csv.write_text(
        "locus,u,v\n"
        "locusA,0.1,1.0\n"
        "locusB,-0.1,1.1\n"
    )
    out_dir = tmp_path / "out"
    result = _run_cli([
        "coupling-genomic", "--input", str(summary_csv), "--no-theta",
        "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr
    summary_json = json.loads((out_dir / "hycline_coupling_genomic_summary.json").read_text())
    assert summary_json["theta_estimate"] is None
    assert summary_json["u_col_used"] == "u"  # fell back from u_mean to u


def test_cli_coupling_geographic_missing_column_actionable_error(tmp_path):
    summary_csv = tmp_path / "bad_summary.csv"
    summary_csv.write_text("locus,foo\nlocusA,1\nlocusB,2\n")
    out_dir = tmp_path / "out"
    result = _run_cli([
        "coupling-geographic", "--input", str(summary_csv), "--output-dir", str(out_dir),
    ])
    assert result.returncode != 0
    assert "could not auto-detect" in result.stderr


# ---------------------------------------------------------------------------
# fit-gl / fit-genomic-gl / fit-gl-multi / fit-genomic-gl-multi (Phase 5 CLI)
# ---------------------------------------------------------------------------

def _write_single_gl_geographic_csv(path, seed=31, n_ind=60):
    import numpy as np
    import pandas as pd
    from hycline.models.cline import ClineParams, TailType, cline_function
    from hycline.data.genotype_likelihood import simulate_genotype_likelihoods

    rng = np.random.default_rng(seed)
    distance = np.sort(rng.uniform(0, 100, n_ind))
    params = ClineParams(center=50, width=15, pmin=0.05, pmax=0.95)
    p = cline_function(distance, params, TailType.NONE)
    u = rng.uniform(size=n_ind)
    true_geno = np.where(u < (1 - p) ** 2, 0, np.where(u < (1 - p) ** 2 + 2 * p * (1 - p), 1, 2))
    gls = simulate_genotype_likelihoods(true_geno, depth=10, error_rate=0.01, rng=rng)
    pd.DataFrame({"distance": distance, "locus0.gl0": gls["log_gl0"],
                  "locus0.gl1": gls["log_gl1"], "locus0.gl2": gls["log_gl2"]}).to_csv(path, index=False)


def _write_single_gl_genomic_csv(path, seed=33, n_ind=80):
    import numpy as np
    import pandas as pd
    from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function
    from hycline.data.genotype_likelihood import simulate_genotype_likelihoods

    rng = np.random.default_rng(seed)
    hybrid_index = rng.uniform(0.02, 0.98, n_ind)
    params = GenomicClineParams(u=0.0, v=4.0)
    phi = genomic_cline_function(hybrid_index, params)
    u = rng.uniform(size=n_ind)
    true_geno = np.where(u < (1 - phi) ** 2, 0,
                          np.where(u < (1 - phi) ** 2 + 2 * phi * (1 - phi), 1, 2))
    gls = simulate_genotype_likelihoods(true_geno, depth=10, error_rate=0.01, rng=rng)
    pd.DataFrame({"hybrid_index": hybrid_index, "locus0.gl0": gls["log_gl0"],
                  "locus0.gl1": gls["log_gl1"], "locus0.gl2": gls["log_gl2"]}).to_csv(path, index=False)


def test_cli_fit_genomic_help_mentions_gl_suffix_flags():
    result = _run_cli(["fit-genomic", "--help"])
    assert result.returncode == 0
    assert "--gl0-suffix" in result.stdout


def test_cli_fit_geographic_help_mentions_gl_suffix_flags():
    result = _run_cli(["fit-geographic", "--help"])
    assert result.returncode == 0
    assert "--gl0-suffix" in result.stdout
    assert "--phred" in result.stdout


def test_cli_fit_gl_recovers_known_cline(tmp_path):
    csv_path = tmp_path / "gl_single.csv"
    _write_single_gl_geographic_csv(csv_path)

    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(csv_path), "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr

    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    center = summary_df.loc[0, "center"]
    assert 30 < center < 70  # true center=50, generous tolerance for a single small fit
    assert (out_dir / "hycline_overlay.png").exists()
    assert (out_dir / "hycline_fitted_curves.csv").exists()


def test_cli_fit_genomic_gl_recovers_known_barrier_locus(tmp_path):
    csv_path = tmp_path / "genomic_gl_single.csv"
    _write_single_gl_genomic_csv(csv_path)

    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-genomic", "--input", str(csv_path), "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr

    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    assert summary_df.loc[0, "v"] > 2.0  # true v=4.0
    assert (out_dir / "hycline_overlay.png").exists()


def test_cli_fit_gl_multi_runs_end_to_end(tmp_path):
    csv_path = tmp_path / "gl_geo.csv"
    _write_gl_geographic_csv(csv_path)

    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(csv_path), "--individual-col", "individual",
        "--output-dir", str(out_dir), "--prefix", "test",
    ])
    assert result.returncode == 0, result.stderr

    content = (out_dir / "test_summary.csv").read_text()
    assert "locusA" in content and "locusB" in content
    assert (out_dir / "test_overlay.png").exists()


def test_cli_fit_genomic_gl_multi_runs_end_to_end(tmp_path):
    csv_path = tmp_path / "gl_genomic.csv"
    _write_gl_genomic_csv(csv_path)

    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-genomic", "--input", str(csv_path), "--individual-col", "individual",
        "--output-dir", str(out_dir), "--prefix", "test",
    ])
    assert result.returncode == 0, result.stderr

    import csv as csv_module
    with open(out_dir / "test_summary.csv") as f:
        rows = {r["locus"]: r for r in csv_module.DictReader(f)}
    assert float(rows["barrierX"]["v"]) > 1.5  # true v=4.0
    assert (out_dir / "test_overlay.png").exists()


def test_cli_fit_gl_phred_matches_non_phred(tmp_path):
    import pandas as pd
    from hycline.data.genotype_likelihood import log_likelihood_to_phred

    csv_path = tmp_path / "gl_single.csv"
    _write_single_gl_geographic_csv(csv_path)
    df = pd.read_csv(csv_path)
    phred_path = tmp_path / "gl_single_phred.csv"
    df_phred = df.copy()
    for col in ["locus0.gl0", "locus0.gl1", "locus0.gl2"]:
        df_phred[col] = log_likelihood_to_phred(df[col].to_numpy())
    df_phred.to_csv(phred_path, index=False)

    out_dir1 = tmp_path / "out1"
    out_dir2 = tmp_path / "out2"
    r1 = _run_cli(["fit-geographic", "--input", str(csv_path), "--output-dir", str(out_dir1)])
    r2 = _run_cli(["fit-geographic", "--input", str(phred_path), "--phred",
                   "--output-dir", str(out_dir2)])
    assert r1.returncode == 0 and r2.returncode == 0

    s1 = pd.read_csv(out_dir1 / "hycline_summary.csv")
    s2 = pd.read_csv(out_dir2 / "hycline_summary.csv")
    assert s1.loc[0, "center"] == pytest.approx(s2.loc[0, "center"], abs=1e-4)


def test_cli_fit_gl_actionable_error_on_missing_column(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("distance,foo\n0,1\n")
    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(csv_path), "--output-dir", str(out_dir),
    ])
    assert result.returncode != 0
    assert "could not detect the input type" in result.stderr


# ---------------------------------------------------------------------------
# genotype-to-gl (the hard-genotype convenience path)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# estimate-hybrid-index
# ---------------------------------------------------------------------------

def _write_hybrid_index_input_vcf(vcf_path, metadata_path, seed=71, n_loci=20,
                                    n_ref_each=15, n_hybrid=30, flip_every_other=False):
    """Writes a VCF (parental reference individuals plus hybrid-zone test
    individuals, all in one file) and a separate metadata CSV giving
    population labels for just the two reference groups -- matching how
    estimate-hybrid-index's Mode 1 actually reads population labels now
    (a VCF has no room for its own population column, the same reason
    filter-aims takes a separate --metadata file)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    true_freq_pop0 = rng.uniform(0.0, 0.1, n_loci)
    true_freq_pop1 = rng.uniform(0.9, 1.0, n_loci)

    def sample_genotypes(freqs, n_ind):
        return rng.binomial(2, freqs, size=(n_ind, len(freqs)))

    pop0_geno = sample_genotypes(true_freq_pop0, n_ref_each)
    pop1_geno = sample_genotypes(true_freq_pop1, n_ref_each)
    true_h = rng.uniform(0.0, 1.0, n_hybrid)
    hybrid_geno = np.array([
        rng.binomial(2, true_freq_pop0 * (1 - h) + true_freq_pop1 * h) for h in true_h
    ])

    individuals = ([f"pop0_ref{i}" for i in range(n_ref_each)] +
                   [f"pop1_ref{i}" for i in range(n_ref_each)] +
                   [f"hybrid{i}" for i in range(n_hybrid)])
    all_geno = np.vstack([pop0_geno, pop1_geno, hybrid_geno])  # (n_ind, n_loci)
    if flip_every_other:
        # ALT is population 0's allele at odd loci, as at ~half of real AIMs.
        all_geno[:, 1::2] = 2 - all_geno[:, 1::2]

    header = ("##fileformat=VCFv4.2\n"
              "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
              + "\t".join(individuals) + "\n")
    lines = [header]
    gt_lookup = {0: "0/0", 1: "0/1", 2: "1/1"}
    for j in range(n_loci):
        gts = [gt_lookup[int(all_geno[i, j])] for i in range(len(individuals))]
        lines.append(f"chr1\t{(j + 1) * 100}\tlocus{j}\tA\tT\t99\tPASS\t.\tGT\t"
                     + "\t".join(gts) + "\n")
    with open(vcf_path, "w") as f:
        f.writelines(lines)

    meta_rows = (
        [{"individual": f"pop0_ref{i}", "population": "house"} for i in range(n_ref_each)]
        + [{"individual": f"pop1_ref{i}", "population": "spanish"} for i in range(n_ref_each)]
    )
    pd.DataFrame(meta_rows).to_csv(metadata_path, sep="\t", index=False)

    return true_h


def test_cli_estimate_hybrid_index_help_runs_without_error():
    result = _run_cli(["estimate-hybrid-index", "--help"])
    assert result.returncode == 0
    assert "--pop1-label" in result.stdout
    assert "--pop2-label" in result.stdout
    for removed in ("--pop0-label", "--parental-freq-file", "--ploidy", "--loci",
                    "--method", "--weight-by-differentiation"):
        assert removed not in result.stdout


def test_cli_estimate_hybrid_index_recovers_known_true_hybrid_index(tmp_path):
    """The core validation: not just 'does the command run,' but 'does it
    actually recover known ground truth well' -- checked via correlation
    with a known true hybrid index for a set of synthetic hybrid-zone
    individuals, plus a sanity check that parental reference individuals
    land near their expected extremes."""
    import numpy as np
    import pandas as pd

    vcf_path = tmp_path / "genotypes.vcf"
    metadata_path = tmp_path / "metadata.tsv"
    true_h = _write_hybrid_index_input_vcf(vcf_path, metadata_path)

    output_path = tmp_path / "hybrid_index.tsv"
    result = _run_cli([
        "estimate-hybrid-index", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--population-col", "population", "--pop1-label", "house",
        "--pop2-label", "spanish", "--output", str(output_path),
    ])
    assert result.returncode == 0, result.stderr

    out_df = pd.read_csv(output_path, sep="\t")
    assert len(out_df) == 15 + 15 + 30
    assert set(out_df.columns) == {"individual", "hybrid_index", "hybrid_index_lower",
                                   "hybrid_index_upper", "n_loci"}
    assert (out_df["hybrid_index_lower"] <= out_df["hybrid_index"]).all()
    assert (out_df["hybrid_index"] <= out_df["hybrid_index_upper"]).all()

    pop0_h = out_df[out_df["individual"].str.startswith("pop0")]["hybrid_index"]
    pop1_h = out_df[out_df["individual"].str.startswith("pop1")]["hybrid_index"]
    hybrid_h = out_df[out_df["individual"].str.startswith("hybrid")]["hybrid_index"].to_numpy()

    assert pop0_h.mean() < 0.1
    assert pop1_h.mean() > 0.9

    correlation = np.corrcoef(hybrid_h, true_h)[0, 1]
    assert correlation > 0.9
    assert np.mean(np.abs(hybrid_h - true_h)) < 0.1


def test_cli_estimate_hybrid_index_orients_alleles_by_parental_frequency(tmp_path):
    """Regression test: loci were averaged as ALT dosage without orienting
    them, so when ALT is population 0's allele at some loci (as in any real
    VCF) the loci cancelled out. On benchmarking/simulate_vcf.py output,
    where the ALT allele's population is random per AIM, every individual
    -- parentals included -- came out at 0.487-0.518, with correlation
    -0.14 to the true hybrid index."""
    import numpy as np

    vcf_path = tmp_path / "genotypes.vcf"
    metadata_path = tmp_path / "metadata.tsv"
    true_h = _write_hybrid_index_input_vcf(vcf_path, metadata_path,
                                           flip_every_other=True)
    output_path = tmp_path / "hybrid_index.tsv"
    result = _run_cli([
        "estimate-hybrid-index", "--input", str(vcf_path), "--metadata",
        str(metadata_path), "--population-col", "population", "--pop1-label", "house",
        "--pop2-label", "spanish", "--output", str(output_path),
    ])
    assert result.returncode == 0, result.stderr
    out_df = pd.read_csv(output_path, sep="\t")
    h = out_df.set_index("individual")["hybrid_index"]
    assert h[h.index.str.startswith("pop0")].mean() < 0.1
    assert h[h.index.str.startswith("pop1")].mean() > 0.9
    hybrid_h = h[h.index.str.startswith("hybrid")].to_numpy()
    assert np.corrcoef(hybrid_h, true_h)[0, 1] > 0.9
    assert np.mean(np.abs(hybrid_h - true_h)) < 0.1


def test_cli_estimate_hybrid_index_beagle_missing_genotypes_not_counted(tmp_path):
    """ANGSD's 0.333333 x3 no-data rows used to enter as expected dosage 1
    (a heterozygote), pulling hybrid indices toward 0.5."""
    hom_ref, hom_alt, miss = ("1.0\t0.0\t0.0", "0.0\t0.0\t1.0",
                              "0.333333\t0.333333\t0.333333")
    names = ["r0", "r1", "s0", "s1", "x"]
    header = "marker\tallele1\tallele2\t" + "\t".join(f"{n}\t{n}\t{n}" for n in names)
    rows = [f"m{j}\t0\t1\t" + "\t".join([hom_ref, hom_ref, hom_alt, hom_alt,
                                            miss if j < 3 else hom_ref])
            for j in range(4)]
    beagle_path = tmp_path / "in.beagle"
    beagle_path.write_text(header + "\n" + "\n".join(rows) + "\n")
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": names,
                  "population": ["house", "house", "spanish", "spanish", "hybrid"]}).to_csv(metadata_path, sep="\t", index=False)
    output_path = tmp_path / "hi.csv"
    result = _run_cli([
        "estimate-hybrid-index", "--input", str(beagle_path), "--metadata",
        str(metadata_path), "--population-col", "population", "--pop1-label", "house",
        "--pop2-label", "spanish", "--output", str(output_path),
    ])
    assert result.returncode == 0, result.stderr
    h = pd.read_csv(output_path, sep="\t").set_index("individual")["hybrid_index"]
    # x is missing at 3 of 4 loci and pure house at the fourth: 0, not 0.375
    assert h["x"] == pytest.approx(0.0, abs=1e-9)


def test_cli_estimate_hybrid_index_ml_recovers_hybrid_index_at_low_depth(tmp_path):
    """Regression test built with benchmarking/simulate_vcf.py, the data that
    surfaced the problem: at depth 2 a mean-dosage estimate pulled every
    hybrid index toward 0.5 (parental references at ~0.19/0.81); the
    maximum-likelihood estimate uses the genotype likelihoods directly."""
    import subprocess
    import sys
    from pathlib import Path

    import numpy as np

    repo = Path(__file__).resolve().parents[1]
    prefix = tmp_path / "zone.vcf"
    subprocess.run([sys.executable, str(repo / "benchmarking" / "simulate_vcf.py"),
                    "--n-pop1", "15", "--n-pop2", "15", "--n-popH", "40",
                    "--n-snps", "1000", "--aim-percent", "30", "--missing-global", "0.1",
                    "--format-beagle", "--mean-depth", "2", "--seed", "21",
                    "--output", str(prefix)], check=True, capture_output=True)
    metadata = tmp_path / "zone_metadata.tsv"
    truth = pd.read_csv(metadata, sep="\t").set_index("individual")

    out = tmp_path / "hi.tsv"
    result = _run_cli([
        "estimate-hybrid-index", "--input", str(tmp_path / "zone.beagle"),
        "--metadata", str(metadata), "--population-col", "population",
        "--pop1-label", "pop1", "--pop2-label", "pop2", "--output", str(out),
    ])
    assert result.returncode == 0, result.stderr
    est = pd.read_csv(out, sep="\t").set_index("individual").join(truth, rsuffix="_true")
    hybrids = est[est["population"] == "popH"]
    error = np.mean(np.abs(hybrids["hybrid_index"] - hybrids["hybrid_index_true"]))
    covered = ((hybrids["hybrid_index_lower"] <= hybrids["hybrid_index_true"])
               & (hybrids["hybrid_index_true"] <= hybrids["hybrid_index_upper"]))
    assert covered.mean() >= 0.8
    assert error < 0.03


def _simulated_zone(tmp_path, *args):
    """Run benchmarking/simulate_vcf.py into tmp_path; returns the output prefix."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(repo / "benchmarking" / "simulate_vcf.py"),
                    *args, "--output", str(tmp_path / "zone.vcf")],
                   check=True, capture_output=True)
    return tmp_path / "zone"


def test_cli_filter_aims_comma_separated_metadata_gives_clear_error(tmp_path):
    header = ("##fileformat=VCFv4.2\n"
              "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tind0\tind1\n")
    vcf_path = tmp_path / "g.vcf"
    vcf_path.write_text(header + "chr1\t100\tlocusA\tA\tT\t99\tPASS\t.\tGT\t0/0\t1/1\n")
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"individual": ["ind0", "ind1"],
                  "population": ["parent1", "parent2"]}).to_csv(metadata_path, index=False)
    result = _run_cli(["filter-aims", "--input", str(vcf_path), "--metadata",
                       str(metadata_path), "--output", str(tmp_path / "o.vcf")])
    assert result.returncode == 2
    assert "looks comma-separated" in result.stderr and "tab-delimited" in result.stderr


def test_simulator_writes_tsv_outputs_and_reproducible_params(tmp_path):
    """benchmarking/simulate_vcf.py: tab-delimited metadata/truth, a params
    record, and a recorded seed that reproduces a run made without --seed."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    script = str(repo / "benchmarking" / "simulate_vcf.py")
    base = ["--n-pop1", "6", "--n-pop2", "6", "--n-popH", "8", "--n-snps", "40",
            "--aim-percent", "50", "--barrier-percent", "20", "--format-beagle"]
    subprocess.run([sys.executable, script, *base, "--output", str(tmp_path / "a.vcf")],
                   check=True, capture_output=True)
    params = pd.read_csv(tmp_path / "a_params.tsv", sep="\t", keep_default_na=False)
    p = dict(zip(params["parameter"], params["value"]))
    assert p["option.n_snps"] == "40" and p["option.barrier_percent"] == "20.0"
    assert p["realized.n_aims"] == "20" and p["realized.n_cline_barrier"] == "4"
    assert p["output.output_truth"].endswith("a_truth.tsv")
    meta = pd.read_csv(tmp_path / "a_metadata.tsv", sep="\t")
    truth = pd.read_csv(tmp_path / "a_truth.tsv", sep="\t")
    assert list(meta.columns[:3]) == ["individual", "population", "hybrid_index"]
    assert {"cline_class", "u", "v", "source_freq_pop1"} <= set(truth.columns)
    assert "##simulate_vcf_command=" in (tmp_path / "a.vcf").read_text()

    subprocess.run([sys.executable, script, *base, "--seed", p["option.seed"],
                    "--output", str(tmp_path / "b.vcf")], check=True, capture_output=True)
    strip = lambda f: [l for l in (tmp_path / f).read_text().splitlines()
                       if not l.startswith("##simulate_vcf")]
    assert strip("a.vcf") == strip("b.vcf")
    assert (tmp_path / "a.beagle").read_text() == (tmp_path / "b.beagle").read_text()
    assert (tmp_path / "a_truth.tsv").read_text() == (tmp_path / "b_truth.tsv").read_text()


def test_cli_default_hybrid_index_gives_unflattened_genomic_clines(tmp_path):
    """Regression test: the ml hybrid index puts parental references at
    exactly 0 or 1. Fitted with them under the original cline model
    (phi(0) = 0, no parental frequencies), a parental at h = 0 carrying the
    minority allele was near-impossible and fit-genomic flattened the cline
    (v ~ 0.17-0.28 instead of ~1 on simulated all-AIM data, where every
    locus is neutral). The documented pipeline gives generate-genomic-input
    the parental labels, so the references are left out of the cline fit
    and each locus carries its parental frequencies."""
    prefix = _simulated_zone(tmp_path, "--n-pop1", "15", "--n-pop2", "15", "--n-popH", "40",
                             "--n-snps", "200", "--aim-percent", "100",
                             "--aim-cutoff", "0.05", "--seed", "4")
    vcf, meta = f"{prefix}.vcf", f"{prefix}_metadata.tsv"
    hi = tmp_path / "hi.csv"
    result = _run_cli(["estimate-hybrid-index", "--input", vcf, "--metadata", meta,
                       "--population-col", "population", "--pop1-label", "pop1",
                       "--pop2-label", "pop2", "--output", str(hi)])
    assert result.returncode == 0, result.stderr
    genomic = tmp_path / "genomic.csv"
    result = _run_cli(["generate-genomic-input", "--input", vcf, "--metadata", str(hi),
                       "--parental-metadata", meta, "--pop1-label", "pop1",
                       "--pop2-label", "pop2",
                       "--output", str(genomic)])
    assert result.returncode == 0, result.stderr

    truth = pd.read_csv(f"{prefix}_truth.tsv", sep="\t").set_index("id")
    # ALT is parental population 1's allele where pop1 (population 0) is low,
    # and at least one parental carries the minority allele.
    forward = [l for l in truth.index
               if truth.loc[l, "low_pop"] == "pop1" and truth.loc[l, "freq_pop1"] > 0][:4]
    assert len(forward) == 4
    data = pd.read_csv(genomic)
    subset = tmp_path / "subset.csv"
    data[["individual", "hybrid_index"]
         + [f"{l}.{c}" for l in forward for c in ("n", "freq", "p0", "p1")]].to_csv(subset, index=False)
    result = _run_cli(["fit-genomic", "--input", str(subset), "--individual-col", "individual",
                       "--output-dir", str(tmp_path / "fit")])
    assert result.returncode == 0, result.stderr
    summary = pd.read_csv(tmp_path / "fit" / "hycline_summary.csv")
    assert (summary["v"] > 0.5).all(), summary[["locus", "u", "v"]]


def test_cli_estimate_hybrid_index_actionable_error_missing_population_label(tmp_path):
    vcf_path = tmp_path / "genotypes.vcf"
    metadata_path = tmp_path / "metadata.tsv"
    _write_hybrid_index_input_vcf(vcf_path, metadata_path)
    output_path = tmp_path / "hybrid_index.tsv"
    result = _run_cli([
        "estimate-hybrid-index", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--population-col", "population", "--pop1-label", "nonexistent_label",
        "--pop2-label", "spanish", "--output", str(output_path),
    ])
    assert result.returncode != 0
    assert "no samples labeled" in result.stderr.lower()


def test_cli_estimate_hybrid_index_actionable_error_missing_column(tmp_path):
    vcf_path = tmp_path / "genotypes.vcf"
    metadata_path = tmp_path / "metadata.tsv"
    _write_hybrid_index_input_vcf(vcf_path, metadata_path)
    output_path = tmp_path / "hybrid_index.tsv"
    result = _run_cli([
        "estimate-hybrid-index", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--population-col", "nonexistent_column", "--pop1-label", "house",
        "--pop2-label", "spanish", "--output", str(output_path),
    ])
    assert result.returncode != 0
    assert "not found" in result.stderr


def test_cli_estimate_hybrid_index_output_chains_into_genomic_fitting(tmp_path):
    """End-to-end: the hybrid_index output actually merges correctly with
    genotype-likelihood data and produces a fittable genomic cline
    dataset -- not just a standalone CSV that looks right on paper."""
    import numpy as np
    import pandas as pd
    from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function

    vcf_path = tmp_path / "genotypes.vcf"
    metadata_path = tmp_path / "metadata.tsv"
    _write_hybrid_index_input_vcf(vcf_path, metadata_path, n_hybrid=40)
    hi_path = tmp_path / "hybrid_index.tsv"
    r1 = _run_cli([
        "estimate-hybrid-index", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--population-col", "population", "--pop1-label", "house",
        "--pop2-label", "spanish", "--output", str(hi_path),
    ])
    assert r1.returncode == 0, r1.stderr

    hi_df = pd.read_csv(hi_path, sep="\t")
    hybrid_rows = hi_df[hi_df["individual"].str.startswith("hybrid")].reset_index(drop=True)

    # simulate a real genomic-cline locus using the estimated hybrid index
    rng = np.random.default_rng(3)
    params = GenomicClineParams(u=0.0, v=3.0)
    phi = genomic_cline_function(hybrid_rows["hybrid_index"].to_numpy(), params)
    n_focal = rng.binomial(2, phi)
    fit_input = pd.DataFrame({
        "individual": hybrid_rows["individual"],
        "hybrid_index": hybrid_rows["hybrid_index"],
        "locus0.n": 2, "locus0.freq": n_focal / 2.0,
    })
    fit_input_path = tmp_path / "genomic_fit_input.csv"
    fit_input.to_csv(fit_input_path, index=False)

    out_dir = tmp_path / "out"
    r2 = _run_cli([
        "fit-genomic", "--input", str(fit_input_path),
        "--hybrid-index-col", "hybrid_index", "--output-dir", str(out_dir),
    ])
    assert r2.returncode == 0, r2.stderr
    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    assert summary_df.loc[0, "v"] > 1.5  # true v=3.0


def test_cli_estimate_hybrid_index_requires_metadata(tmp_path):
    """Parental frequencies always come from reference individuals labeled
    in --metadata, so it is required."""
    vcf_path = tmp_path / "genotypes.vcf"
    metadata_path = tmp_path / "metadata.tsv"
    _write_hybrid_index_input_vcf(vcf_path, metadata_path)
    result = _run_cli([
        "estimate-hybrid-index", "--input", str(vcf_path),
        "--pop1-label", "house", "--pop2-label", "spanish",
        "--output", str(tmp_path / "out.csv"),
    ])
    assert result.returncode != 0
    assert "required: --metadata" in result.stderr

def test_cli_estimate_hybrid_index_beagle_input_works(tmp_path):
    """Confirms estimate-hybrid-index accepts beagle input directly,
    closing what was previously a real gap: filter-aims produces VCF/
    beagle, but estimate-hybrid-index required a pre-converted genotype-
    dosage CSV with no path from one to the other."""
    import numpy as np
    from scipy.stats import binom
    import pandas as pd

    rng = np.random.default_rng(42)
    n_ref = 5
    individuals = [f"pop0_ref{i}" for i in range(n_ref)] + [f"pop1_ref{i}" for i in range(n_ref)]

    def gl_row(dosage, depth=10, error=0.01):
        p_alt = {0: error, 1: 0.5, 2: 1 - error}[dosage]
        n_alt = rng.binomial(depth, p_alt)
        return [binom.pmf(n_alt, depth, {0: error, 1: 0.5, 2: 1 - error}[g]) for g in [0, 1, 2]]

    header = "marker\tallele1\tallele2\t" + "\t".join(f"{ind}\t{ind}\t{ind}" for ind in individuals) + "\n"
    lines = [header]
    dosage = [0] * n_ref + [2] * n_ref  # pop0 refs pure ref, pop1 refs pure alt
    row = ["locus0", "0", "1"]
    for d in dosage:
        row.extend(f"{v:.6f}" for v in gl_row(d))
    lines.append("\t".join(row) + "\n")
    beagle_path = tmp_path / "test.beagle"
    beagle_path.write_text("".join(lines))

    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": individuals,
                  "population": ["house"] * n_ref + ["spanish"] * n_ref}).to_csv(metadata_path, sep="\t", index=False)

    output_path = tmp_path / "hybrid_index.tsv"
    result = _run_cli([
        "estimate-hybrid-index", "--input", str(beagle_path), "--metadata", str(metadata_path),
        "--population-col", "population", "--pop1-label", "house", "--pop2-label", "spanish",
        "--output", str(output_path),
    ])
    assert result.returncode == 0, result.stderr
    out_df = pd.read_csv(output_path, sep="\t")
    assert len(out_df) == 2 * n_ref
    house_h = out_df[out_df["individual"].str.startswith("pop0")]["hybrid_index"]
    spanish_h = out_df[out_df["individual"].str.startswith("pop1")]["hybrid_index"]
    assert house_h.mean() < 0.1
    assert spanish_h.mean() > 0.9


# ---------------------------------------------------------------------------
# generate-geographic-input (VCF) / convert-vcf-genomic
# ---------------------------------------------------------------------------

def _write_test_vcf(path, sample_names, dosages_by_locus, locus_names, with_pl=False):
    """dosages_by_locus: list of arrays (one per locus), each length len(sample_names)."""
    lines = ["##fileformat=VCFv4.2", '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">']
    if with_pl:
        lines.append('##FORMAT=<ID=PL,Number=G,Type=Integer,Description="PL">')
    header = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(sample_names)
    lines.append(header)
    fmt = "GT:PL" if with_pl else "GT"
    for locus_i, (locus_name, dosages) in enumerate(zip(locus_names, dosages_by_locus)):
        fields = ["chr1", str(1000 + locus_i * 100), locus_name, "A", "G", ".", "PASS", ".", fmt]
        for d in dosages:
            gt = {0: "0/0", 1: "0/1", 2: "1/1"}[int(d)]
            if with_pl:
                pl = {0: "0,50,100", 1: "50,0,50", 2: "100,50,0"}[int(d)]
                fields.append(f"{gt}:{pl}")
            else:
                fields.append(gt)
        lines.append("\t".join(fields))
    Path(path).write_text("\n".join(lines) + "\n")


def test_cli_generate_geographic_input_vcf_help_runs_without_error():
    result = _run_cli(["generate-geographic-input", "--help"])
    assert result.returncode == 0
    assert "--site-col" in result.stdout
    assert "--genotype-likelihoods" not in result.stdout


def test_cli_generate_genomic_input_vcf_help_runs_without_error():
    result = _run_cli(["generate-genomic-input", "--help"])
    assert result.returncode == 0
    assert "--hybrid-index-col" in result.stdout
    assert "--genotype-likelihoods" not in result.stdout


def test_cli_generate_geographic_input_vcf_gives_actionable_guidance_on_skipped_sites(tmp_path):
    """A real usability improvement, not just a correctness fix: when a
    VCF has multiallelic/indel sites this converter deliberately doesn't
    handle (BACKEND_PHASE1.md section 9.4), the CLI now surfaces
    actionable bcftools guidance inline, at the moment it's relevant,
    rather than requiring the user to go find it in the backend docs."""
    vcf_content = (
        "##fileformat=VCFv4.2\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"GT\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsampleA\tsampleB\n"
        "chr1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\n"
        "chr1\t200\t.\tA\tAT\t.\tPASS\t.\tGT\t0/1\t0/0\n"
        "chr1\t300\t.\tG\tA,C\t.\tPASS\t.\tGT\t0/1\t0/2\n"
    )
    vcf_path = tmp_path / "test.vcf"
    vcf_path.write_text(vcf_content)
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": ["sampleA", "sampleB"], "distance": [0, 10],
                  "site": ["X", "Y"]}).to_csv(metadata_path, sep="\t", index=False)

    result = _run_cli([
        "generate-geographic-input", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert result.returncode == 0, result.stderr
    assert "bcftools view -m2 -M2 -v snps" in result.stdout
    assert "2 sites" in result.stdout  # 1 indel + 1 multiallelic


def test_cli_generate_geographic_input_vcf_no_guidance_when_nothing_skipped(tmp_path):
    """Confirms the guidance from the test above doesn't appear when
    there's nothing to skip -- it should only show up when relevant."""
    vcf_content = (
        "##fileformat=VCFv4.2\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"GT\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsampleA\tsampleB\n"
        "chr1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\n"
    )
    vcf_path = tmp_path / "test.vcf"
    vcf_path.write_text(vcf_content)
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": ["sampleA", "sampleB"], "distance": [0, 10],
                  "site": ["X", "Y"]}).to_csv(metadata_path, sep="\t", index=False)

    result = _run_cli([
        "generate-geographic-input", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert result.returncode == 0, result.stderr
    assert "bcftools" not in result.stdout


def test_cli_generate_geographic_input_vcf_lat_lon_recovers_known_cline(tmp_path):
    """Full end-to-end validation: VCF + lat/lon metadata -> converter
    (computing distance in km) -> fit-geographic-multi, checking the
    recovered cline against known true generating parameters and
    confirming the computed distances match a direct hand computation."""
    import numpy as np
    from hycline.models.cline import ClineParams, TailType, cline_function
    from hycline.data.geo import distance_from_reference

    rng = np.random.default_rng(211)
    n_sites, n_per_site = 8, 6
    # sites spread along a simple north-south transect
    site_lats = np.linspace(40.0, 41.4, n_sites)
    site_lons = np.full(n_sites, -74.0)
    site_distances = distance_from_reference(site_lats, site_lons)
    individual_distance = np.repeat(site_distances, n_per_site)
    individual_lat = np.repeat(site_lats, n_per_site)
    individual_lon = np.repeat(site_lons, n_per_site)
    sample_names = [f"ind{i}" for i in range(n_sites * n_per_site)]

    true_params = ClineParams(center=70, width=25, pmin=0.05, pmax=0.95)
    p = cline_function(individual_distance, true_params, TailType.NONE)
    u = rng.uniform(size=len(sample_names))
    dosage = np.where(u < (1 - p) ** 2, 0, np.where(u < (1 - p) ** 2 + 2 * p * (1 - p), 1, 2))

    vcf_path = tmp_path / "test.vcf"
    _write_test_vcf(vcf_path, sample_names, [dosage], ["locus0"])
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({
        "individual": sample_names, "lat": individual_lat, "lon": individual_lon,
        "site": [f"site{i}" for i in np.repeat(np.arange(n_sites), n_per_site)],
    }).to_csv(metadata_path, sep="\t", index=False)

    converted_path = tmp_path / "converted.csv"
    r1 = _run_cli([
        "generate-geographic-input", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--lat-col", "lat", "--lon-col", "lon", "--output", str(converted_path),
    ])
    assert r1.returncode == 0, r1.stderr

    converted_df = pd.read_csv(converted_path)
    np.testing.assert_allclose(sorted(converted_df["distance"]), sorted(site_distances), atol=1e-6)

    out_dir = tmp_path / "out"
    r2 = _run_cli(["fit-geographic", "--input", str(converted_path), "--output-dir", str(out_dir)])
    assert r2.returncode == 0, r2.stderr

    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    assert summary_df.loc[0, "center"] == pytest.approx(70, abs=25)


def test_cli_generate_geographic_input_vcf_inconsistent_coordinates_actionable_error(tmp_path):
    import numpy as np
    sample_names = ["ind0", "ind1"]
    vcf_path = tmp_path / "test.vcf"
    _write_test_vcf(vcf_path, sample_names, [np.array([0, 2])], ["locus0"])
    metadata_path = tmp_path / "metadata.tsv"
    # both individuals assigned to site 'S0' but with different coordinates
    pd.DataFrame({"individual": sample_names, "site": ["S0", "S0"],
                  "lat": [40.0, 41.0], "lon": [-74.0, -74.0]}).to_csv(metadata_path, sep="\t", index=False)

    result = _run_cli([
        "generate-geographic-input", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--lat-col", "lat", "--lon-col", "lon", "--output", str(tmp_path / "out.csv"),
    ])
    assert result.returncode != 0
    assert "inconsistent coordinates" in result.stderr
    assert "S0" in result.stderr


def test_cli_generate_geographic_input_beagle_lat_lon_computes_correct_distance(tmp_path):
    import numpy as np
    from hycline.data.geo import distance_from_reference

    beagle_content = (
        "marker\tallele1\tallele2\tind0\tind0\tind0\tind1\tind1\tind1\n"
        "chr1_100\t0\t1\t0.98\t0.02\t0.00\t0.00\t0.02\t0.98\n"
    )
    beagle_path = tmp_path / "test.beagle"
    beagle_path.write_text(beagle_content)
    metadata_path = tmp_path / "metadata.tsv"
    lat, lon = [40.0, 41.0], [-74.0, -73.0]
    pd.DataFrame({"individual": ["ind0", "ind1"], "site": ["S0", "S1"],
                  "lat": lat, "lon": lon}).to_csv(metadata_path, sep="\t", index=False)

    out_path = tmp_path / "out.csv"
    result = _run_cli([
        "generate-geographic-input", "--input", str(beagle_path), "--metadata", str(metadata_path),
        "--lat-col", "lat", "--lon-col", "lon", "--output", str(out_path),
    ])
    assert result.returncode == 0, result.stderr

    out_df = pd.read_csv(out_path)
    expected = distance_from_reference(np.array(lat), np.array(lon))
    np.testing.assert_allclose(sorted(out_df["distance"]), sorted(expected), atol=1e-6)


def test_cli_generate_geographic_input_vcf_output_format_identical_for_single_and_multi_locus(tmp_path):
    """Confirms a real design property, not just describes it: the
    output CSV shape (site, distance, then <locus>.nSamples/<locus>.A
    column pairs) is exactly the same whether the VCF has one locus or
    many -- no special-cased "single-locus" format exists. The same
    file is directly usable by both fit-geographic (naming one locus's
    columns explicitly) and fit-geographic-multi (reading every locus
    automatically)."""
    import numpy as np
    sample_names = [f"ind{i}" for i in range(10)]
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": sample_names, "site": [f"S{i}" for i in range(10)],
                  "distance": range(10)}).to_csv(metadata_path, sep="\t", index=False)

    rng = np.random.default_rng(5)
    single_vcf = tmp_path / "single.vcf"
    _write_test_vcf(single_vcf, sample_names, [rng.integers(0, 3, 10)], ["locus0"])
    multi_vcf = tmp_path / "multi.vcf"
    _write_test_vcf(multi_vcf, sample_names,
                     [rng.integers(0, 3, 10) for _ in range(3)], ["locus0", "locus1", "locus2"])

    single_out = tmp_path / "single_out.csv"
    r1 = _run_cli(["generate-geographic-input", "--input", str(single_vcf), "--metadata", str(metadata_path),
                   "--output", str(single_out)])
    assert r1.returncode == 0, r1.stderr

    multi_out = tmp_path / "multi_out.csv"
    r2 = _run_cli(["generate-geographic-input", "--input", str(multi_vcf), "--metadata", str(metadata_path),
                   "--output", str(multi_out)])
    assert r2.returncode == 0, r2.stderr

    single_cols = list(pd.read_csv(single_out).columns)
    multi_cols = list(pd.read_csv(multi_out).columns)
    assert single_cols == ["site", "distance", "locus0.nSamples", "locus0.A"]
    assert multi_cols[:4] == single_cols  # multi-locus is the exact same shape, just more columns
    assert multi_cols == ["site", "distance", "locus0.nSamples", "locus0.A",
                           "locus1.nSamples", "locus1.A", "locus2.nSamples", "locus2.A"]

    # the single-locus file is directly usable by fit-geographic, auto-detecting
    # its one locus from the wide-format suffix convention with zero extra flags
    out_dir = tmp_path / "fit_out"
    r3 = _run_cli(["fit-geographic", "--input", str(single_out), "--output-dir", str(out_dir)])
    assert r3.returncode == 0, r3.stderr


def test_cli_generate_geographic_input_vcf_recovers_known_cline(tmp_path):
    """Full end-to-end validation: VCF -> converter -> fit-geographic-multi, checking
    the recovered cline against the known true generating parameters."""
    import numpy as np
    from hycline.models.cline import ClineParams, TailType, cline_function

    rng = np.random.default_rng(201)
    n_sites, n_per_site = 8, 6
    site_distances = np.linspace(0, 140, n_sites)
    individual_distance = np.repeat(site_distances, n_per_site)
    sample_names = [f"ind{i}" for i in range(n_sites * n_per_site)]

    true_params = ClineParams(center=70, width=25, pmin=0.05, pmax=0.95)
    p = cline_function(individual_distance, true_params, TailType.NONE)
    u = rng.uniform(size=len(sample_names))
    dosage = np.where(u < (1 - p) ** 2, 0, np.where(u < (1 - p) ** 2 + 2 * p * (1 - p), 1, 2))

    vcf_path = tmp_path / "test.vcf"
    _write_test_vcf(vcf_path, sample_names, [dosage], ["locus0"])
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({
        "individual": sample_names, "distance": individual_distance,
        "site": [f"site{i}" for i in np.repeat(np.arange(n_sites), n_per_site)],
    }).to_csv(metadata_path, sep="\t", index=False)

    converted_path = tmp_path / "converted.csv"
    r1 = _run_cli([
        "generate-geographic-input", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--output", str(converted_path),
    ])
    assert r1.returncode == 0, r1.stderr

    out_dir = tmp_path / "out"
    r2 = _run_cli(["fit-geographic", "--input", str(converted_path), "--output-dir", str(out_dir)])
    assert r2.returncode == 0, r2.stderr

    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    assert summary_df.loc[0, "center"] == pytest.approx(70, abs=20)


def test_cli_generate_genomic_input_vcf_recovers_known_barrier_locus(tmp_path):
    import numpy as np
    from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function

    rng = np.random.default_rng(203)
    n_ind = 60
    hybrid_index = rng.uniform(0.02, 0.98, n_ind)
    sample_names = [f"ind{i}" for i in range(n_ind)]

    true_params = GenomicClineParams(u=0.0, v=4.0)
    phi = genomic_cline_function(hybrid_index, true_params)
    u = rng.uniform(size=n_ind)
    dosage = np.where(u < (1 - phi) ** 2, 0, np.where(u < (1 - phi) ** 2 + 2 * phi * (1 - phi), 1, 2))

    vcf_path = tmp_path / "test.vcf"
    _write_test_vcf(vcf_path, sample_names, [dosage], ["locus0"])
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": sample_names, "hybrid_index": hybrid_index}).to_csv(metadata_path, sep="\t", index=False)

    converted_path = tmp_path / "converted.csv"
    r1 = _run_cli([
        "generate-genomic-input", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--output", str(converted_path),
    ])
    assert r1.returncode == 0, r1.stderr

    out_dir = tmp_path / "out"
    r2 = _run_cli([
        "fit-genomic", "--input", str(converted_path), "--individual-col", "individual",
        "--output-dir", str(out_dir),
    ])
    assert r2.returncode == 0, r2.stderr

    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    assert summary_df.loc[0, "v"] > 1.5  # true v=4.0


def test_cli_generate_geographic_input_missing_input_file_error(tmp_path):
    metadata_path = tmp_path / "metadata.tsv"
    metadata_path.write_text("individual,distance,site\nind0,0,X\n")
    result = _run_cli([
        "generate-geographic-input", "--input", str(tmp_path / "nonexistent.vcf"),
        "--metadata", str(metadata_path), "--output", str(tmp_path / "out.csv"),
    ])
    assert result.returncode != 0
    assert "not found" in result.stderr


def _neutral_zone_pipeline(tmp_path, with_parentals=True, extra_ggi=()):
    """benchmarking/simulate_vcf.py data in which every locus is neutral,
    run through filter-aims -> estimate-hybrid-index -> generate-genomic-input
    -> fit-genomic; returns the fit summary and the truth table."""
    import numpy as np  # noqa: F401
    prefix = _simulated_zone(tmp_path, "--n-pop1", "15", "--n-pop2", "15", "--n-popH", "50",
                             "--n-snps", "600", "--aim-percent", "30",
                             "--aim-freq-diff", "0.3", "--seed", "12")
    vcf, meta = f"{prefix}.vcf", f"{prefix}_metadata.tsv"
    labels = ["--pop1-label", "pop1", "--pop2-label", "pop2"]
    aims = tmp_path / "aims.vcf"
    assert _run_cli(["filter-aims", "--input", vcf, "--metadata", meta, "--pop1-label", "pop1",
                     "--pop2-label", "pop2", "--allele-freq-diff", "0.3",
                     "--output", str(aims)]).returncode == 0
    hi = tmp_path / "hi.csv"
    result = _run_cli(["estimate-hybrid-index", "--input", str(aims), "--metadata", meta,
                       *labels, "--output", str(hi)])
    assert result.returncode == 0, result.stderr
    genomic = tmp_path / "genomic.csv"
    ggi = ["generate-genomic-input", "--input", str(aims), "--metadata", str(hi),
           "--output", str(genomic), *extra_ggi]
    if with_parentals:
        ggi += ["--parental-metadata", meta, *labels]
    ggi_result = _run_cli(ggi)
    assert ggi_result.returncode == 0, ggi_result.stderr
    result = _run_cli(["fit-genomic", "--input", str(genomic), "--individual-col", "individual",
                       "--no-plot", "--output-dir", str(tmp_path / "fit")])
    assert result.returncode == 0, result.stderr
    summary = pd.read_csv(tmp_path / "fit" / "hycline_summary.csv")
    truth = pd.read_csv(f"{prefix}_truth.tsv", sep="\t").set_index("id")
    return summary, truth, ggi_result, genomic


def test_cli_genomic_pipeline_with_parental_frequencies_recovers_neutral_clines(tmp_path):
    """Regression test, built with the simulator data that surfaced it: every
    simulated locus is neutral (u = 0, v = 1). Without parental frequencies,
    loci whose ALT allele is parental population 0's all fitted at the v
    lower bound (0.001) and partially differentiated loci fitted as shallow,
    shifted clines. With them, loci of either polarity and any degree of
    differentiation should fit near neutral."""
    import numpy as np
    summary, truth, ggi, genomic = _neutral_zone_pipeline(tmp_path)
    assert "50 individuals" in ggi.stdout  # the 30 reference individuals are left out
    assert "30 parental reference individual(s) left out" in ggi.stdout
    reversed_ = summary["parental_freq_pop1"] < summary["parental_freq_pop0"]
    assert reversed_.sum() > 20 and (~reversed_).sum() > 20
    # A chance extreme can still fit at the bound (1 of 180 loci here, a locus
    # whose sampled genotypes really are near-flat); without parental
    # frequencies it was every reversed locus.
    assert (summary["v"] <= 0.0011).mean() < 0.02
    for group in (summary[reversed_], summary[~reversed_]):
        assert 0.8 < group["v"].median() < 1.3
        assert abs(group["u"].median()) < 0.2


def test_cli_genomic_pipeline_without_parental_frequencies_pins_reversed_loci(tmp_path):
    """The failure the parental frequencies fix, kept visible: without them
    generate-genomic-input warns, and reversed loci hit the v lower bound."""
    summary, truth, ggi, _ = _neutral_zone_pipeline(tmp_path, with_parentals=False)
    assert "no parental reference individuals given" in ggi.stderr
    reversed_ = summary["locus"].map(truth["low_pop"]) == "pop2"
    assert (summary.loc[reversed_, "v"] <= 0.0011).mean() > 0.9


def test_cli_generate_genomic_input_parental_options(tmp_path):
    import numpy as np
    prefix = _simulated_zone(tmp_path, "--n-pop1", "6", "--n-pop2", "6", "--n-popH", "10",
                             "--n-snps", "20", "--aim-percent", "100", "--seed", "3")
    vcf, meta = f"{prefix}.vcf", f"{prefix}_metadata.tsv"
    labels = ["--pop1-label", "pop1", "--pop2-label", "pop2"]
    out = tmp_path / "g.csv"

    result = _run_cli(["generate-genomic-input", "--input", vcf, "--metadata", meta, *labels,
                       "--include-parentals", "--output", str(out)])
    assert result.returncode == 0, result.stderr
    g = pd.read_csv(out)
    assert len(g) == 22
    truth = pd.read_csv(f"{prefix}_truth.tsv", sep="\t").set_index("id")
    # parental frequencies are the reference panels' ALT frequencies, clipped
    np.testing.assert_allclose(g["snp1.p0"].iloc[0], np.clip(truth.loc["snp1", "freq_pop1"], 1e-3, 1 - 1e-3))
    np.testing.assert_allclose(g["snp1.p1"].iloc[0], np.clip(truth.loc["snp1", "freq_pop2"], 1e-3, 1 - 1e-3))

    result = _run_cli(["generate-genomic-input", "--input", vcf, "--metadata", meta,
                       "--pop1-label", "pop1", "--output", str(out)])
    assert result.returncode == 2 and "must be given together" in result.stderr


def test_cli_generate_genomic_input_drops_loci_with_equal_parental_frequencies(tmp_path):
    import numpy as np
    names = ["a0", "a1", "b0", "b1", "h0", "h1"]
    vcf = tmp_path / "in.vcf"
    _write_test_vcf(vcf, names, [np.array([0, 1, 0, 1, 1, 1]),   # 0.25 in both panels
                                 np.array([0, 0, 2, 2, 1, 2])],  # 0 vs 1
                    ["equal", "aim"])
    meta = tmp_path / "meta.tsv"
    pd.DataFrame({"individual": names, "population": ["p1", "p1", "p2", "p2", "h", "h"],
                  "hybrid_index": [0, 0, 1, 1, 0.5, 0.7]}).to_csv(meta, sep="\t", index=False)
    out = tmp_path / "g.csv"
    result = _run_cli(["generate-genomic-input", "--input", str(vcf), "--metadata", str(meta),
                       "--pop1-label", "p1", "--pop2-label", "p2", "--output", str(out)])
    assert result.returncode == 0, result.stderr
    assert "1 with equal parental frequencies" in result.stdout
    g = pd.read_csv(out)
    assert "equal.p0" not in g.columns
    assert g["aim.p0"].iloc[0] == 0.001 and g["aim.p1"].iloc[0] == 0.999
    assert list(g["individual"]) == ["h0", "h1"]


def test_cli_generate_genomic_input_no_genotype_likelihoods_flag_exists(tmp_path):
    """Regression test mirroring the same design decision made for
    generate-geographic-input: --genotype-likelihoods was removed
    entirely, not just made a no-op. VCF input always means hard counts
    (GT is a hard call, regardless of caller); beagle input always means
    per-individual genotype-likelihood output."""
    result = _run_cli(["generate-genomic-input", "--help"])
    assert result.returncode == 0
    assert "--genotype-likelihoods" not in result.stdout
    for removed in ("--pop0-label", "--parental-freq-file", "--ploidy"):
        assert removed not in result.stdout


def test_cli_generate_geographic_input_vcf_site_aggregation_matches_hand_computation(tmp_path):
    import numpy as np

    sample_names = ["ind0", "ind1", "ind2", "ind3"]
    dosage = np.array([0, 2, 1, 1])  # ind0,ind1 at site X; ind2,ind3 at site Y
    vcf_path = tmp_path / "test.vcf"
    _write_test_vcf(vcf_path, sample_names, [dosage], ["locus0"])
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({
        "individual": sample_names, "distance": [0, 0, 10, 10], "site": ["X", "X", "Y", "Y"],
    }).to_csv(metadata_path, sep="\t", index=False)

    out_path = tmp_path / "out.csv"
    result = _run_cli([
        "generate-geographic-input", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--output", str(out_path),
    ])
    assert result.returncode == 0, result.stderr

    out_df = pd.read_csv(out_path)
    row_x = out_df[out_df["site"] == "X"].iloc[0]
    # site X: dosage=[0,2], n_alleles=2*2=4, n_focal=0+2=2, freq=0.5
    assert row_x["locus0.nSamples"] == 4
    assert row_x["locus0.A"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# genotype-to-hard-counts (the reverse of genotype-to-gl)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# filter-aims
# ---------------------------------------------------------------------------

def test_cli_filter_aims_help_runs_without_error():
    result = _run_cli(["filter-aims", "--help"])
    assert result.returncode == 0
    assert "--minor-allele-freq" in result.stdout
    assert "--allele-freq-diff" in result.stdout
    assert "--low-threshold" not in result.stdout
    assert "--pop1-label" in result.stdout


def test_cli_filter_aims_next_step_guidance_references_live_commands(tmp_path):
    """Regression test for a real bug: when convert-vcf-geographic/
    convert-beagle-geographic were consolidated into
    generate-geographic-input, filter-aims's own "Ready for: ..."
    next-step guidance (both the VCF and beagle branches) kept
    referencing the two removed command names -- a real, live bug in
    actual CLI output, not just stale documentation, caught by manually
    re-running filter-aims after the consolidation rather than assuming
    the earlier test suite's passing status meant nothing needed
    updating (none of those tests asserted on this exact message)."""
    header = ("##fileformat=VCFv4.2\n"
              "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tind0\tind1\n")
    line = "chr1\t100\tlocusA\tA\tT\t99\tPASS\t.\tGT\t0/0\t1/1\n"
    vcf_path = tmp_path / "genotypes.vcf"
    vcf_path.write_text(header + line)
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": ["ind0", "ind1"],
                  "population": ["parent1", "parent2"]}).to_csv(metadata_path, sep="\t", index=False)

    vcf_result = _run_cli([
        "filter-aims", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--output", str(tmp_path / "filtered.vcf"),
    ])
    assert vcf_result.returncode == 0
    assert "generate-geographic-input" in vcf_result.stdout
    assert "convert-vcf-geographic" not in vcf_result.stdout

    beagle_content = (
        "marker\tallele1\tallele2\tind0\tind0\tind0\tind1\tind1\tind1\n"
        "chr1_100\t0\t1\t0.98\t0.02\t0.00\t0.00\t0.02\t0.98\n"
    )
    beagle_path = tmp_path / "genotypes.beagle"
    beagle_path.write_text(beagle_content)
    beagle_result = _run_cli([
        "filter-aims", "--input", str(beagle_path), "--metadata", str(metadata_path),
        "--output", str(tmp_path / "filtered.beagle"),
    ])
    assert beagle_result.returncode == 0
    assert "generate-geographic-input" in beagle_result.stdout
    assert "convert-beagle-geographic" not in beagle_result.stdout


def test_cli_filter_aims_writes_summary_and_warnings(tmp_path):
    header = ("##fileformat=VCFv4.2\n"
              "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tind0\tind1\n")
    lines = ("chr1\t100\tlocusA\tA\tT\t99\tPASS\t.\tGT\t0/0\t1/1\n"
             "chr1\t200\tlocusB\tA\tT\t99\tPASS\t.\tGT\t./.\t1/1\n")
    vcf_path = tmp_path / "genotypes.vcf.gz"
    import gzip
    with gzip.open(vcf_path, "wt") as fh:
        fh.write(header + lines)
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": ["ind0", "ind1"],
                  "population": ["parent1", "parent2"]}).to_csv(metadata_path, sep="\t", index=False)

    result = _run_cli([
        "filter-aims", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--output", str(tmp_path / "filtered.vcf.gz"),
    ])
    assert result.returncode == 0
    summary = (tmp_path / "filtered_summary.txt").read_text()
    assert "AIMs retained:  1 (50.00% of loci checked)" in summary
    assert "ind0\tparent1\t50.00%\t0.00%\t" in summary
    assert "'parent1' has only 1 individual(s)" in result.stderr
    assert "'parent2' has only 1 individual(s)" in result.stderr
    assert "'ind0' is missing" not in result.stderr  # 50% is not > 50%

    custom = tmp_path / "sub" / "report.txt"
    result = _run_cli([
        "filter-aims", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--output", str(tmp_path / "filtered.vcf.gz"), "--summary", str(custom),
        "--min-parental-n", "1", "--max-sample-missing", "0.4",
    ])
    assert result.returncode == 0
    text = custom.read_text()
    assert "'ind0' is missing data at 50.00%" in text
    assert "has only" not in text

    result = _run_cli([
        "filter-aims", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--output", str(tmp_path / "x.vcf"), "--summary", str(tmp_path / "x.vcf"),
    ])
    assert result.returncode == 2
    assert "--summary must be a different path" in result.stderr


def test_cli_filter_aims_beagle_auto_detect_and_filter(tmp_path):
    beagle_content = (
        "marker\tallele1\tallele2\tind0\tind0\tind0\tind1\tind1\tind1\tind2\tind2\tind2\n"
        "chr1_100\t0\t1\t0.98\t0.02\t0.00\t0.00\t0.02\t0.98\t0.49\t0.02\t0.49\n"
        "chr1_200\t0\t1\t0.10\t0.80\t0.10\t0.10\t0.80\t0.10\t0.33\t0.34\t0.33\n"
    )
    beagle_path = tmp_path / "test.beagle"
    beagle_path.write_text(beagle_content)
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": ["ind0", "ind1", "ind2"],
                  "population": ["parent1", "parent2", "hybrid"]}).to_csv(metadata_path, sep="\t", index=False)

    out_path = tmp_path / "filtered.beagle"
    result = _run_cli([
        "filter-aims", "--input", str(beagle_path), "--metadata", str(metadata_path),
        "--output", str(out_path),
    ])
    assert result.returncode == 0, result.stderr
    assert "Detected input format: beagle" in result.stdout
    assert "1 retained" in result.stdout

    from hycline.data.beagle import read_beagle
    filtered = read_beagle(str(out_path))
    assert filtered.marker_names == ["chr1_100"]


def test_cli_filter_aims_vcf_end_to_end_recovers_known_cline(tmp_path):
    import numpy as np
    from hycline.models.cline import ClineParams, TailType, cline_function

    rng = np.random.default_rng(73)
    n_parent_each = 8
    n_hybrid = 30
    n_ind = 2 * n_parent_each + n_hybrid
    distance = np.concatenate([
        np.full(n_parent_each, -50.0), np.full(n_parent_each, 150.0),
        np.sort(rng.uniform(0, 100, n_hybrid)),
    ])
    population = ["parent1"] * n_parent_each + ["parent2"] * n_parent_each + ["hybrid"] * n_hybrid
    true_params = ClineParams(center=55, width=18, pmin=0.05, pmax=0.95)
    p_true = cline_function(distance, true_params, TailType.NONE)

    freqs = np.where(np.array(population) == "parent1", 0.02,
                      np.where(np.array(population) == "parent2", 0.98, p_true))
    u = rng.uniform(size=n_ind)
    dosage_informative = np.where(u < (1 - freqs) ** 2, 0,
                                   np.where(u < (1 - freqs) ** 2 + 2 * freqs * (1 - freqs), 1, 2))
    dosage_uninformative = rng.integers(0, 3, size=n_ind)

    def dosage_to_gt(d):
        return {0: "0/0", 1: "0/1", 2: "1/1"}[d]

    individuals = [f"ind{i}" for i in range(n_ind)]
    header = ("##fileformat=VCFv4.2\n"
               "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
               + "\t".join(individuals) + "\n")
    line_informative = ("chr1\t100\tinformative\tA\tT\t99\tPASS\t.\tGT\t"
                         + "\t".join(dosage_to_gt(d) for d in dosage_informative) + "\n")
    line_uninformative = ("chr1\t200\tuninformative\tA\tT\t99\tPASS\t.\tGT\t"
                           + "\t".join(dosage_to_gt(d) for d in dosage_uninformative) + "\n")
    vcf_path = tmp_path / "genotypes.vcf"
    vcf_path.write_text(header + line_informative + line_uninformative)

    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": individuals, "population": population,
                  "distance": distance, "site": individuals}).to_csv(metadata_path, sep="\t", index=False)

    filtered_path = tmp_path / "filtered.vcf"
    r1 = _run_cli([
        "filter-aims", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--output", str(filtered_path),
    ])
    assert r1.returncode == 0, r1.stderr

    filtered_text = filtered_path.read_text()
    assert "informative" in filtered_text
    assert "uninformative" not in filtered_text.replace("informative", "", 1)

    converted_path = tmp_path / "converted.csv"
    r2 = _run_cli([
        "generate-geographic-input", "--input", str(filtered_path), "--metadata", str(metadata_path),
        "--output", str(converted_path),
    ])
    assert r2.returncode == 0, r2.stderr

    out_dir = tmp_path / "out"
    r3 = _run_cli([
        "fit-geographic", "--input", str(converted_path), "--output-dir", str(out_dir),
    ])
    assert r3.returncode == 0, r3.stderr

    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    assert summary_df.loc[0, "center"] == pytest.approx(55, abs=15)


def test_cli_filter_aims_no_loci_pass_gives_warning_not_crash(tmp_path):
    header = ("##fileformat=VCFv4.2\n"
               "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tind0\tind1\n")
    line = "chr1\t100\tlocusA\tA\tT\t99\tPASS\t.\tGT\t0/1\t0/1\n"  # both intermediate, uninformative
    input_path = tmp_path / "genotypes.vcf"
    input_path.write_text(header + line)
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": ["ind0", "ind1"],
                  "population": ["parent1", "parent2"]}).to_csv(metadata_path, sep="\t", index=False)

    out_path = tmp_path / "filtered.vcf"
    result = _run_cli([
        "filter-aims", "--input", str(input_path), "--metadata", str(metadata_path),
        "--output", str(out_path),
    ])
    assert result.returncode == 0
    assert "no loci passed the filter" in result.stderr
    assert out_path.exists()


def test_cli_filter_aims_missing_metadata_file_actionable_error(tmp_path):
    input_path = tmp_path / "genotypes.vcf"
    input_path.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\n")
    result = _run_cli([
        "filter-aims", "--input", str(input_path),
        "--metadata", str(tmp_path / "nonexistent.csv"), "--output", str(tmp_path / "out.vcf"),
    ])
    assert result.returncode != 0
    assert "not found" in result.stderr


def test_cli_filter_aims_custom_thresholds(tmp_path):
    header = ("##fileformat=VCFv4.2\n"
               "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tind0\tind1\n")
    line = "chr1\t100\tlocusA\tA\tT\t99\tPASS\t.\tGT\t0/0\t1/1\n"
    input_path = tmp_path / "genotypes.vcf"
    input_path.write_text(header + line)
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": ["ind0", "ind1"],
                  "population": ["parent1", "parent2"]}).to_csv(metadata_path, sep="\t", index=False)

    out_path = tmp_path / "filtered.vcf"
    result = _run_cli([
        "filter-aims", "--input", str(input_path), "--metadata", str(metadata_path),
        "--minor-allele-freq", "0.0", "--output", str(out_path),
    ])
    assert result.returncode == 0, result.stderr
    assert "1 retained" in result.stdout
    assert "<= 0 in one parental population and >= 1 in the other" in result.stdout


def _write_freq_diff_vcf(tmp_path):
    """Five loci, five individuals per parental population; locus IDs give
    the parent1/parent2 ALT frequencies (f010_080 = 0.1 vs 0.8)."""
    rows = {
        "f010_080": ["0/1"] + ["0/0"] * 4 + ["1/1"] * 3 + ["0/1"] * 2,
        "f000_075": ["0/0"] * 5 + ["1/1"] * 3 + ["0/1", "0/0"],
        "f010_070": ["0/1"] + ["0/0"] * 4 + ["1/1"] * 2 + ["0/1"] * 3,
        "f000_100": ["0/0"] * 5 + ["1/1"] * 5,
        "f100_020": ["1/1"] * 5 + ["0/1"] * 2 + ["0/0"] * 3,
    }
    names = [f"a{i}" for i in range(5)] + [f"b{i}" for i in range(5)]
    header = ("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\t"
              "FORMAT\t" + "\t".join(names) + "\n")
    lines = [f"chr1\t{i + 1}\t{snp}\tA\tT\t.\tPASS\t.\tGT\t" + "\t".join(gts) + "\n"
             for i, (snp, gts) in enumerate(rows.items())]
    vcf_path = tmp_path / "freqs.vcf"
    vcf_path.write_text(header + "".join(lines))
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": names,
                  "population": ["parent1"] * 5 + ["parent2"] * 5}).to_csv(metadata_path, sep="\t", index=False)
    return vcf_path, metadata_path


def _kept_ids(vcf_path):
    return [l.split("\t")[2] for l in vcf_path.read_text().splitlines()
            if not l.startswith("#")]


def test_cli_filter_aims_minor_allele_freq_sets_both_thresholds(tmp_path):
    vcf_path, metadata_path = _write_freq_diff_vcf(tmp_path)
    out_path = tmp_path / "out.vcf"
    result = _run_cli([
        "filter-aims", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--minor-allele-freq", "0.2", "--min-parental-n", "5", "--output", str(out_path),
    ])
    assert result.returncode == 0, result.stderr
    # <= 0.2 and >= 0.8: 0.1/0.8, 0/1 and 1/0.2 pass; 0/0.75 and 0.1/0.7 do not
    assert _kept_ids(out_path) == ["f010_080", "f000_100", "f100_020"]


def test_cli_filter_aims_allele_freq_diff(tmp_path):
    vcf_path, metadata_path = _write_freq_diff_vcf(tmp_path)
    out_path = tmp_path / "out.vcf"
    result = _run_cli([
        "filter-aims", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--allele-freq-diff", "0.7", "--min-parental-n", "5", "--output", str(out_path),
    ])
    assert result.returncode == 0, result.stderr
    assert "allele-frequency difference >= 0.7" in result.stdout
    # 0.8 - 0.1 is 0.7000000000000001 in floating point and must still count
    assert _kept_ids(out_path) == ["f010_080", "f000_075", "f000_100", "f100_020"]
    summary = (tmp_path / "out_summary.txt").read_text()
    assert "AIM rule:   allele-frequency difference >= 0.7" in summary

    result = _run_cli([
        "filter-aims", "--input", str(vcf_path), "--metadata", str(metadata_path),
        "--allele-freq-diff", "0.7", "--minor-allele-freq", "0.1",
        "--output", str(out_path),
    ])
    assert result.returncode != 0
    assert "not allowed with" in result.stderr

    for flag, value in (("--allele-freq-diff", "0"), ("--minor-allele-freq", "0.5")):
        result = _run_cli([
            "filter-aims", "--input", str(vcf_path), "--metadata", str(metadata_path),
            flag, value, "--output", str(out_path),
        ])
        assert result.returncode == 2
        assert flag in result.stderr


# ---------------------------------------------------------------------------
# generate-geographic-input (beagle) / convert-beagle-genomic
# ---------------------------------------------------------------------------

def test_cli_generate_geographic_input_beagle_help_runs_without_error():
    result = _run_cli(["generate-geographic-input", "--help"])
    assert result.returncode == 0
    assert "--genotype-likelihoods" not in result.stdout
    assert "--site-col" in result.stdout


def test_cli_generate_genomic_input_beagle_help_runs_without_error():
    result = _run_cli(["generate-genomic-input", "--help"])
    assert result.returncode == 0
    assert "--hybrid-index-col" in result.stdout
    assert "--genotype-likelihoods" not in result.stdout


def _write_beagle_geographic_recovery_files(tmp_path, true_center=70, true_width=25, seed=91):
    import numpy as np
    from hycline.models.cline import ClineParams, TailType, cline_function

    rng = np.random.default_rng(seed)
    n_hybrid = 80
    distance = np.sort(rng.uniform(0, 150, n_hybrid))
    sample_names = [f"ind{i}" for i in range(n_hybrid)]
    true_params = ClineParams(center=true_center, width=true_width, pmin=0.05, pmax=0.95)
    p_true = cline_function(distance, true_params, TailType.NONE)
    u = rng.uniform(size=n_hybrid)
    true_geno = np.where(u < (1 - p_true) ** 2, 0,
                          np.where(u < (1 - p_true) ** 2 + 2 * p_true * (1 - p_true), 1, 2))

    def make_gl(true_g, noise=0.05):
        gl = np.full(3, noise)
        gl[true_g] = 1.0 - 2 * noise
        return gl / gl.sum()

    header = ["marker", "allele1", "allele2"] + sum([[s, s, s] for s in sample_names], [])
    row = ["locusA", "0", "1"]
    for g in true_geno:
        row += [f"{v:.4f}" for v in make_gl(g)]
    beagle_path = tmp_path / "test.beagle"
    beagle_path.write_text("\t".join(header) + "\n" + "\t".join(row) + "\n")

    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": sample_names, "distance": distance}).to_csv(metadata_path, sep="\t", index=False)
    return beagle_path, metadata_path


def test_cli_generate_geographic_input_no_genotype_likelihoods_flag_exists(tmp_path):
    """Regression test for a real design decision: --genotype-likelihoods
    was removed entirely, not just made a no-op. Input format alone now
    determines the output path -- VCF always means hard counts (GT is a
    hard call, not a genotype likelihood, regardless of caller), beagle
    always means per-individual genotype-likelihood output (its entire
    purpose is preserving that uncertainty). Anyone wanting GL-aware
    fitting from a VCF is expected to generate genotype likelihoods with
    ANGSD (-doGlf 2) and provide that beagle file instead."""
    result = _run_cli(["generate-geographic-input", "--help"])
    assert result.returncode == 0
    assert "--genotype-likelihoods" not in result.stdout


def test_cli_generate_geographic_input_vcf_always_hard_count_beagle_always_gl(tmp_path):
    """Direct confirmation of the format-determines-path rule: the exact
    same conceptual data (a single informative locus, 2 individuals), fed
    in as a VCF vs. as a beagle file, produces structurally different
    output -- site-aggregated .nSamples/.A columns for VCF, per-individual
    .gl0/.gl1/.gl2 columns for beagle -- with no flag involved either way."""
    header = ("##fileformat=VCFv4.2\n"
              "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tind0\tind1\n")
    line = "chr1\t100\tsnp1\tA\tT\t99\tPASS\t.\tGT\t0/0\t1/1\n"
    vcf_path = tmp_path / "test.vcf"
    vcf_path.write_text(header + line)
    vcf_metadata = tmp_path / "vcf_meta.csv"
    pd.DataFrame({"individual": ["ind0", "ind1"], "site": ["S0", "S1"],
                  "distance": [0, 10]}).to_csv(vcf_metadata, sep="\t", index=False)

    vcf_out = tmp_path / "vcf_out.csv"
    r1 = _run_cli(["generate-geographic-input", "--input", str(vcf_path),
                   "--metadata", str(vcf_metadata), "--output", str(vcf_out)])
    assert r1.returncode == 0, r1.stderr
    vcf_cols = list(pd.read_csv(vcf_out).columns)
    assert vcf_cols == ["site", "distance", "snp1.nSamples", "snp1.A"]

    beagle_content = (
        "marker\tallele1\tallele2\tind0\tind0\tind0\tind1\tind1\tind1\n"
        "chr1_100\t0\t1\t0.98\t0.02\t0.00\t0.00\t0.02\t0.98\n"
    )
    beagle_path = tmp_path / "test.beagle"
    beagle_path.write_text(beagle_content)
    beagle_metadata = tmp_path / "beagle_meta.csv"
    pd.DataFrame({"individual": ["ind0", "ind1"],
                  "distance": [0, 10]}).to_csv(beagle_metadata, sep="\t", index=False)

    beagle_out = tmp_path / "beagle_out.csv"
    r2 = _run_cli(["generate-geographic-input", "--input", str(beagle_path),
                   "--metadata", str(beagle_metadata), "--output", str(beagle_out)])
    assert r2.returncode == 0, r2.stderr
    beagle_cols = list(pd.read_csv(beagle_out).columns)
    assert beagle_cols == ["individual", "distance",
                            "chr1_100.gl0", "chr1_100.gl1", "chr1_100.gl2"]


def test_cli_generate_geographic_input_beagle_recovers_known_cline(tmp_path):
    beagle_path, metadata_path = _write_beagle_geographic_recovery_files(tmp_path)

    converted_path = tmp_path / "converted.csv"
    r1 = _run_cli([
        "generate-geographic-input", "--input", str(beagle_path), "--metadata", str(metadata_path),
        "--output", str(converted_path),
    ])
    assert r1.returncode == 0, r1.stderr

    out_dir = tmp_path / "out"
    r2 = _run_cli([
        "fit-geographic", "--input", str(converted_path), "--individual-col", "individual",
        "--output-dir", str(out_dir),
    ])
    assert r2.returncode == 0, r2.stderr

    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    assert summary_df.loc[0, "center"] == pytest.approx(70, abs=20)


def test_cli_generate_genomic_input_beagle_recovers_known_barrier_locus(tmp_path):
    import numpy as np
    from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function

    rng = np.random.default_rng(95)
    n_ind = 80
    hybrid_index = rng.uniform(0.02, 0.98, n_ind)
    sample_names = [f"ind{i}" for i in range(n_ind)]
    true_params = GenomicClineParams(u=0.0, v=4.0)
    phi = genomic_cline_function(hybrid_index, true_params)
    u = rng.uniform(size=n_ind)
    true_geno = np.where(u < (1 - phi) ** 2, 0,
                          np.where(u < (1 - phi) ** 2 + 2 * phi * (1 - phi), 1, 2))

    def make_gl(true_g, noise=0.05):
        gl = np.full(3, noise)
        gl[true_g] = 1.0 - 2 * noise
        return gl / gl.sum()

    header = ["marker", "allele1", "allele2"] + sum([[s, s, s] for s in sample_names], [])
    row = ["locusA", "0", "1"]
    for g in true_geno:
        row += [f"{v:.4f}" for v in make_gl(g)]
    beagle_path = tmp_path / "test.beagle"
    beagle_path.write_text("\t".join(header) + "\n" + "\t".join(row) + "\n")
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": sample_names, "hybrid_index": hybrid_index}).to_csv(metadata_path, sep="\t", index=False)

    converted_path = tmp_path / "converted.csv"
    r1 = _run_cli([
        "generate-genomic-input", "--input", str(beagle_path), "--metadata", str(metadata_path),
        "--output", str(converted_path),
    ])
    assert r1.returncode == 0, r1.stderr

    out_dir = tmp_path / "out"
    r2 = _run_cli([
        "fit-genomic", "--input", str(converted_path), "--individual-col", "individual",
        "--output-dir", str(out_dir),
    ])
    assert r2.returncode == 0, r2.stderr
    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    assert summary_df.loc[0, "v"] > 1.5  # true v=4.0


def test_cli_filter_aims_then_generate_geographic_input_beagle_full_chain(tmp_path):
    """The complete originally-requested workflow: filter genome-wide
    beagle SNPs to AIMs, then convert the filtered output directly into
    a HyCline-ready CSV, with no manual reshaping in between."""
    beagle_content = (
        "marker\tallele1\tallele2\tind0\tind0\tind0\tind1\tind1\tind1\tind2\tind2\tind2\n"
        "informative\t0\t1\t0.98\t0.02\t0.00\t0.00\t0.02\t0.98\t0.49\t0.02\t0.49\n"
        "uninformative\t0\t1\t0.33\t0.34\t0.33\t0.33\t0.34\t0.33\t0.33\t0.34\t0.33\n"
    )
    beagle_path = tmp_path / "test.beagle"
    beagle_path.write_text(beagle_content)
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": ["ind0", "ind1", "ind2"],
                  "population": ["parent1", "parent2", "hybrid"],
                  "distance": [0, 100, 50]}).to_csv(metadata_path, sep="\t", index=False)

    filtered_path = tmp_path / "filtered.beagle"
    r1 = _run_cli([
        "filter-aims", "--input", str(beagle_path), "--metadata", str(metadata_path),
        "--output", str(filtered_path),
    ])
    assert r1.returncode == 0, r1.stderr

    converted_path = tmp_path / "converted.csv"
    r2 = _run_cli([
        "generate-geographic-input", "--input", str(filtered_path), "--metadata", str(metadata_path),
        "--output", str(converted_path),
    ])
    assert r2.returncode == 0, r2.stderr

    converted_df = pd.read_csv(converted_path)
    assert "informative.gl0" in converted_df.columns
    assert "uninformative.gl0" not in converted_df.columns


def test_cli_generate_geographic_input_beagle_missing_file_actionable_error(tmp_path):
    metadata_path = tmp_path / "metadata.tsv"
    metadata_path.write_text("individual,distance,site\nind0,0,X\n")
    result = _run_cli([
        "generate-geographic-input", "--input", str(tmp_path / "nonexistent.beagle"),
        "--metadata", str(metadata_path), "--output", str(tmp_path / "out.csv"),
    ])
    assert result.returncode != 0
    assert "not found" in result.stderr


# ---------------------------------------------------------------------------
# Scale-related fixes found during a real "thousands of loci" stress test
# (BACKEND_SCALE.md)
# ---------------------------------------------------------------------------

def test_cli_fit_geographic_multi_truncates_long_locus_list_in_console_output(tmp_path):
    """Regression test for a real usability issue found at scale: with
    hundreds/thousands of loci, printing the full locus-name list to the
    console produced an unreadable wall of text. Confirms the list is now
    truncated with a clear '... and N more' suffix. Previously tested via
    genotype-to-gl (removed once the pipeline moved to VCF/beagle-only
    input) -- fit-geographic-multi exercises the exact same shared
    _format_loci_list helper."""
    import numpy as np
    import pandas as pd

    n_loci = 50
    n_sites = 5
    row = {"site": [f"S{i}" for i in range(n_sites)], "distance": list(range(n_sites))}
    rng = np.random.default_rng(1)
    for i in range(n_loci):
        row[f"locus{i}.nSamples"] = [10] * n_sites
        row[f"locus{i}.A"] = rng.uniform(0, 1, n_sites)
    input_path = tmp_path / "multilocus.csv"
    pd.DataFrame(row).to_csv(input_path, index=False)

    result = _run_cli([
        "fit-geographic", "--input", str(input_path), "--output-dir", str(tmp_path / "out"),
        "--no-plot", "--no-plot-grid",
    ])
    assert result.returncode == 0, result.stderr
    detected_line = next(line for line in result.stdout.splitlines() if "loci detected" in line)
    assert "and 40 more" in detected_line
    assert "locus49" not in detected_line  # truncated in the summary line specifically


# ---------------------------------------------------------------------------
# fit-geographic-multi's plate-vectorized batched Bayesian path (BACKEND_PHASE2.md
# section 13)
# ---------------------------------------------------------------------------

def _write_dense_bayesian_multilocus_csv(path, seed=401):
    import numpy as np
    from hycline.models.cline import ClineParams, TailType, cline_function

    rng = np.random.default_rng(seed)
    n_sites = 15
    distance = np.sort(rng.uniform(0, 100, n_sites))
    n_alleles = np.full(n_sites, 25.0)
    true_params = [ClineParams(center=c, width=w, pmin=0.05, pmax=0.95)
                   for c, w in [(30, 10), (50, 15), (70, 12)]]

    df = {"site": [f"site{i}" for i in range(n_sites)], "distance": distance}
    for i, tp in enumerate(true_params):
        p = cline_function(distance, tp, TailType.NONE)
        n_focal = rng.binomial(25, p)
        df[f"locus{i}.nSamples"] = n_alleles
        df[f"locus{i}.A"] = n_focal / n_alleles
    pd.DataFrame(df).to_csv(path, index=False)
    return {"locus0": (30, 10), "locus1": (50, 15), "locus2": (70, 12)}


def test_cli_fit_multi_bayesian_uses_batched_path_on_dense_data_and_recovers_truth(tmp_path):
    input_path = tmp_path / "dense.csv"
    truth = _write_dense_bayesian_multilocus_csv(input_path)
    out_dir = tmp_path / "out"

    result = _run_cli([
        "fit-geographic", "--input", str(input_path), "--method", "bayesian",
        "--num-warmup", "600", "--num-samples", "600", "--num-chains", "2",
        "--no-plot-grid", "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr
    assert "plate-vectorized batch fitting path" in result.stdout
    assert "3/3 loci fit successfully" in result.stdout

    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    assert len(summary_df) == 3
    for _, row in summary_df.iterrows():
        true_center, true_width = truth[row["locus"]]
        assert row["center_mean"] == pytest.approx(true_center, abs=15)
        assert row["max_r_hat"] < 1.05  # clean convergence


def test_cli_fit_multi_bayesian_falls_back_on_ragged_data(tmp_path):
    """The core backward-compatibility guarantee: data that isn't dense
    (different loci have different site sets, e.g. from independent
    per-locus missing-value handling) must still work via the original
    per-locus loop, not fail or silently misalign."""
    input_path = tmp_path / "dense.csv"
    _write_dense_bayesian_multilocus_csv(input_path)
    df = pd.read_csv(input_path)
    df.loc[3, "locus1.A"] = float("nan")
    df.loc[3, "locus1.nSamples"] = float("nan")
    ragged_path = tmp_path / "ragged.csv"
    df.to_csv(ragged_path, index=False)

    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-geographic", "--input", str(ragged_path), "--method", "bayesian",
        "--num-warmup", "150", "--num-samples", "150", "--num-chains", "2",
        "--no-plot-grid", "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr
    assert "plate-vectorized" not in result.stdout
    assert "--- Locus" in result.stdout

    summary_df = pd.read_csv(out_dir / "hycline_summary.csv")
    assert len(summary_df) == 3  # all 3 loci still fit, just independently


def test_cli_fit_multi_bayesian_scaling_fixed_falls_back_even_on_dense_data(tmp_path):
    """fit_mcmc_batch has no fixed-extremes equivalent, so --scaling fixed
    must force the fallback path even when the data would otherwise
    qualify for batching."""
    input_path = tmp_path / "dense.csv"
    _write_dense_bayesian_multilocus_csv(input_path)
    out_dir = tmp_path / "out"

    result = _run_cli([
        "fit-geographic", "--input", str(input_path), "--method", "bayesian",
        "--scaling", "fixed",
        "--num-warmup", "150", "--num-samples", "150", "--num-chains", "2",
        "--no-plot-grid", "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr
    assert "plate-vectorized" not in result.stdout
    assert "--- Locus" in result.stdout


def test_cli_fit_multi_classic_method_never_uses_batched_path(tmp_path):
    """The batched path only applies to --method bayesian --backend
    numpyro -- classic MLE must always use the original per-locus loop
    regardless of data density."""
    input_path = tmp_path / "dense.csv"
    _write_dense_bayesian_multilocus_csv(input_path)
    out_dir = tmp_path / "out"

    result = _run_cli([
        "fit-geographic", "--input", str(input_path), "--method", "classic",
        "--no-plot-grid", "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr
    assert "plate-vectorized" not in result.stdout


# ---------------------------------------------------------------------------
# fit-trait / fit-trait-multi (quantitative trait clines, HZAR-parity --
# BACKEND_TRAIT.md)
# ---------------------------------------------------------------------------

def test_cli_fit_trait_help_runs_without_error():
    result = _run_cli(["fit-trait", "--help"])
    assert result.returncode == 0
    assert "--sd-col" in result.stdout
    assert "var_l" in result.stdout or "HZAR" in result.stdout


def test_cli_fit_trait_multi_help_runs_without_error():
    result = _run_cli(["fit-trait-multi", "--help"])
    assert result.returncode == 0
    assert "--mean-suffix" in result.stdout


def test_cli_fit_trait_raw_input_recovers_known_cline(tmp_path):
    import numpy as np
    from hycline.models.cline import TailType
    from hycline.likelihoods.gaussian import mean_cline, variance_cline

    rng = np.random.default_rng(1101)
    n_sites = 25
    site_distance = np.sort(rng.uniform(0, 150, n_sites))
    n_per_site = rng.integers(8, 20, n_sites)
    true_center, true_width = 70.0, 25.0
    mu = mean_cline(site_distance, true_center, true_width, 10.0, 25.0, TailType.NONE)
    var = variance_cline(site_distance, true_center, true_width, 1.0, 2.0, 1.5, TailType.NONE)

    rows = []
    for i in range(n_sites):
        obs = rng.normal(mu[i], np.sqrt(var[i]), n_per_site[i])
        for v in obs:
            rows.append({"site": f"site{i}", "distance": site_distance[i], "trait_value": v})
    input_path = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(input_path, sep="\t", index=False)
    out_dir = tmp_path / "out"

    result = _run_cli([
        "fit-trait", "--input", str(input_path), "--raw", "--trait-col", "trait_value",
        "--site-col", "site", "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr

    summary = json.loads((out_dir / "hycline_summary.json").read_text())
    assert summary["parameters"]["center"]["estimate"] == pytest.approx(true_center, abs=10)


def test_cli_fit_trait_missing_sd_col_without_raw_actionable_error(tmp_path):
    input_path = tmp_path / "agg.csv"
    pd.DataFrame({"distance": [0, 50, 100], "trait_mean": [1.0, 5.0, 9.0],
                  "n": [10, 10, 10]}).to_csv(input_path, sep="\t", index=False)
    result = _run_cli([
        "fit-trait", "--input", str(input_path), "--output-dir", str(tmp_path / "out"),
    ])
    assert result.returncode != 0
    assert "--sd-col" in result.stderr
    assert "--raw" in result.stderr


def test_cli_fit_trait_raw_and_aggregated_give_consistent_results(tmp_path):
    """The two input modes should produce the same fit when the raw data
    aggregates to exactly the same site summaries an aggregated-input
    file would have contained."""
    import numpy as np
    from hycline.models.cline import TailType
    from hycline.likelihoods.gaussian import mean_cline, variance_cline

    rng = np.random.default_rng(1103)
    n_sites = 20
    site_distance = np.sort(rng.uniform(0, 100, n_sites))
    n_per_site = rng.integers(10, 15, n_sites)
    mu = mean_cline(site_distance, 50.0, 15.0, 2.0, 8.0, TailType.NONE)
    var = variance_cline(site_distance, 50.0, 15.0, 0.5, 1.0, 0.5, TailType.NONE)

    raw_rows = []
    site_means, site_sds, site_ns = [], [], []
    for i in range(n_sites):
        obs = rng.normal(mu[i], np.sqrt(var[i]), n_per_site[i])
        for v in obs:
            raw_rows.append({"site": f"site{i}", "distance": site_distance[i], "trait_value": v})
        site_means.append(obs.mean())
        site_sds.append(obs.std())
        site_ns.append(len(obs))

    raw_path = tmp_path / "raw.csv"
    pd.DataFrame(raw_rows).to_csv(raw_path, sep="\t", index=False)
    agg_path = tmp_path / "agg.csv"
    pd.DataFrame({"distance": site_distance, "trait_mean": site_means, "n": site_ns,
                  "trait_sd": site_sds}).to_csv(agg_path, sep="\t", index=False)

    out_raw = tmp_path / "out_raw"
    r1 = _run_cli([
        "fit-trait", "--input", str(raw_path), "--raw", "--trait-col", "trait_value",
        "--site-col", "site", "--output-dir", str(out_raw),
    ])
    assert r1.returncode == 0, r1.stderr

    out_agg = tmp_path / "out_agg"
    r2 = _run_cli([
        "fit-trait", "--input", str(agg_path), "--sd-col", "trait_sd",
        "--output-dir", str(out_agg),
    ])
    assert r2.returncode == 0, r2.stderr

    summary_raw = json.loads((out_raw / "hycline_summary.json").read_text())
    summary_agg = json.loads((out_agg / "hycline_summary.json").read_text())
    # loose tolerance: the aggregated path's CSV round-trip (writing then
    # re-reading site_means/site_sds) truncates float precision slightly
    # compared to the raw path's in-memory aggregation -- this checks for
    # logical consistency between the two paths, not exact CSV round-trip
    # precision, which isn't the point of this test.
    assert summary_raw["parameters"]["center"]["estimate"] == pytest.approx(
        summary_agg["parameters"]["center"]["estimate"], abs=1e-2)


def _write_hzar_trait_csv(path, seed=901, tail_type_str="none", true_center=70.0, true_width=25.0):
    import numpy as np
    from hycline.models.cline import TailType
    from hycline.likelihoods.gaussian import mean_cline, variance_cline

    rng = np.random.default_rng(seed)
    n_sites = 25
    distance = np.sort(rng.uniform(0, 150, n_sites))
    n_per_site = rng.integers(10, 25, n_sites)
    tt = {"none": TailType.NONE, "left": TailType.LEFT}[tail_type_str]
    kwargs = {"delta_l": 15.0, "tau_l": 0.5} if tt == TailType.LEFT else {}

    mu = mean_cline(distance, true_center, true_width, 10.0, 25.0, tt, **kwargs)
    var = variance_cline(distance, true_center, true_width, 1.0, 2.0, 1.5, tt, **kwargs)

    site_means, site_sds = [], []
    for i in range(n_sites):
        obs = rng.normal(mu[i], np.sqrt(var[i]), n_per_site[i])
        site_means.append(obs.mean())
        site_sds.append(obs.std())

    pd.DataFrame({"site": [f"s{i}" for i in range(n_sites)], "distance": distance,
                  "trait_mean": site_means, "n": n_per_site, "trait_sd": site_sds}).to_csv(
        path, sep="\t", index=False)
    return {"center": true_center, "width": true_width}


def test_cli_fit_trait_recovers_known_cline_hzar_parity(tmp_path):
    input_path = tmp_path / "trait.csv"
    truth = _write_hzar_trait_csv(input_path)
    out_dir = tmp_path / "out"

    result = _run_cli([
        "fit-trait", "--input", str(input_path), "--site-col", "site", "--sd-col", "trait_sd",
        "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr

    summary = json.loads((out_dir / "hycline_summary.json").read_text())
    assert set(summary["parameters"].keys()) == {"center", "width", "pmin", "pmax",
                                                   "var_l", "var_r", "var_h"}
    assert summary["parameters"]["center"]["estimate"] == pytest.approx(truth["center"], abs=10)
    assert summary["parameters"]["var_l"]["estimate"] > 0
    assert summary["parameters"]["var_h"]["estimate"] >= 0

    assert (out_dir / "hycline_trait_cline_plot.png").exists()
    assert (out_dir / "hycline_fitted_curve.csv").exists()


def test_cli_fit_trait_missing_sd_col_actionable_error(tmp_path):
    input_path = tmp_path / "trait.csv"
    _write_hzar_trait_csv(input_path)
    result = _run_cli([
        "fit-trait", "--input", str(input_path), "--output-dir", str(tmp_path / "out"),
    ])
    assert result.returncode != 0
    assert "--sd-col" in result.stderr or "required" in result.stderr.lower()


def test_cli_fit_trait_left_tail_recovers_known_cline(tmp_path):
    input_path = tmp_path / "trait.csv"
    truth = _write_hzar_trait_csv(input_path, seed=903, tail_type_str="left", true_center=100.0,
                                   true_width=30.0)
    out_dir = tmp_path / "out"

    result = _run_cli([
        "fit-trait", "--input", str(input_path), "--site-col", "site", "--sd-col", "trait_sd",
        "--tail", "left", "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr

    summary = json.loads((out_dir / "hycline_summary.json").read_text())
    assert "delta_l" in summary["parameters"]
    assert "tau_l" in summary["parameters"]
    assert summary["parameters"]["center"]["estimate"] == pytest.approx(truth["center"], abs=10)


def test_cli_fit_trait_multi_fits_multiple_traits_and_recovers_known_cline(tmp_path):
    import numpy as np
    from hycline.models.cline import TailType
    from hycline.likelihoods.gaussian import mean_cline, variance_cline

    rng = np.random.default_rng(905)
    n_sites = 25
    distance = np.sort(rng.uniform(0, 150, n_sites))
    n_per_site = rng.integers(10, 25, n_sites)
    mu = mean_cline(distance, 60.0, 20.0, 5.0, 15.0, TailType.NONE)
    var = variance_cline(distance, 60.0, 20.0, 0.5, 1.0, 0.7, TailType.NONE)

    data = {"distance": distance}
    for trait in ["wing", "tail"]:
        means, sds = [], []
        for i in range(n_sites):
            obs = rng.normal(mu[i], np.sqrt(var[i]), n_per_site[i])
            means.append(obs.mean())
            sds.append(obs.std())
        data[f"{trait}.mean"] = means
        data[f"{trait}.n"] = n_per_site
        data[f"{trait}.sd"] = sds
    input_path = tmp_path / "multi_trait.csv"
    pd.DataFrame(data).to_csv(input_path, sep="\t", index=False)

    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-trait-multi", "--input", str(input_path), "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr
    assert "2/2 traits fit successfully" in result.stdout

    summary_df = pd.read_csv(out_dir / "hycline_trait_multi_summary.csv")
    assert len(summary_df) == 2
    assert "var_l" in summary_df.columns
    assert "var_h" in summary_df.columns
    for _, row in summary_df.iterrows():
        assert row["center"] == pytest.approx(60.0, abs=10)

    assert (out_dir / "hycline_trait_multi_grid.png").exists()


def test_cli_fit_trait_multi_skips_trait_missing_sd_column(tmp_path):
    import numpy as np
    from hycline.models.cline import TailType
    from hycline.likelihoods.gaussian import mean_cline, variance_cline

    rng = np.random.default_rng(907)
    n_sites = 20
    distance = np.sort(rng.uniform(0, 100, n_sites))
    n_per_site = rng.integers(8, 15, n_sites)
    mu = mean_cline(distance, 50.0, 15.0, 1.0, 5.0, TailType.NONE)
    var = variance_cline(distance, 50.0, 15.0, 0.5, 0.5, 0.3, TailType.NONE)

    data = {"distance": distance}
    means_a, sds_a, means_b = [], [], []
    for i in range(n_sites):
        obs = rng.normal(mu[i], np.sqrt(var[i]), n_per_site[i])
        means_a.append(obs.mean())
        sds_a.append(obs.std())
        means_b.append(rng.normal(mu[i], np.sqrt(var[i]), n_per_site[i]).mean())
    data["has_sd.mean"] = means_a
    data["has_sd.n"] = n_per_site
    data["has_sd.sd"] = sds_a
    data["no_sd.mean"] = means_b
    data["no_sd.n"] = n_per_site
    input_path = tmp_path / "multi_trait.csv"
    pd.DataFrame(data).to_csv(input_path, sep="\t", index=False)

    out_dir = tmp_path / "out"
    result = _run_cli([
        "fit-trait-multi", "--input", str(input_path), "--output-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr
    assert "SKIPPED" in result.stdout
    assert "no_sd" in result.stdout

    summary_df = pd.read_csv(out_dir / "hycline_trait_multi_summary.csv")
    assert list(summary_df["trait"]) == ["has_sd"]


def test_cli_fit_hzar_tail_with_fixed_scaling_gives_actionable_error(tmp_path):
    """Regression test for a real gap found while extending hzar_tail to
    the Bayesian path: fit_mle_fixed_extremes has no HZAR-parameterized
    tail option (unlike fit_mle), so combining --hzar-tail with
    --scaling fixed previously would have silently ignored --hzar-tail
    rather than erroring or working correctly. Now gives a clear,
    actionable error instead."""
    input_path = tmp_path / "genotypes.csv"
    pd.DataFrame({"distance": [0, 50, 100], "locus0.nSamples": [30, 30, 30],
                  "locus0.A": [0.1, 0.5, 0.9]}).to_csv(input_path, index=False)

    result = _run_cli([
        "fit-geographic", "--input", str(input_path), "--tail", "mirror", "--hzar-tail",
        "--scaling", "fixed", "--pmin", "0.05", "--pmax", "0.95",
        "--output-dir", str(tmp_path / "out"),
    ])
    # With the single/multi-locus collapse, an unsupported per-locus
    # configuration is now a per-locus skip-and-continue (matching how
    # fit-geographic-multi always handled bad loci), not a hard top-level
    # error -- since there's only one locus here, that means an overall
    # "no loci fit successfully" failure (exit code 1), with the actual
    # validation message shown as a per-locus SKIPPED line first.
    assert result.returncode == 1
    assert "--hzar-tail is not supported together with --scaling fixed" in result.stdout
    assert "no loci fit successfully" in result.stderr
    assert "Traceback" not in result.stdout and "Traceback" not in result.stderr  # handled, not a crash


def test_cli_fit_hierarchical_genomic_handles_missing_genotypes(tmp_path):
    """Regression test from the tool-comparison benchmark
    (benchmarking/compare_tools.py --missing-global 0.2): with genotypes
    missing at different individuals per locus, fit-hierarchical-genomic
    refused the data ("requires every locus to have data at exactly the
    same individuals"), although gghybrid and bgchm fit it. Loci are now
    aligned by individual and a missing genotype enters with 0 sampled
    alleles."""
    import numpy as np

    prefix = _simulated_zone(tmp_path, "--n-pop1", "8", "--n-pop2", "8", "--n-popH", "30",
                             "--n-snps", "12", "--aim-percent", "100",
                             "--aim-freq-diff", "0.6", "--missing-global", "0.2",
                             "--seed", "1")
    vcf, meta = f"{prefix}.vcf", f"{prefix}_metadata.tsv"
    hi, genomic = tmp_path / "hi.tsv", tmp_path / "g.csv"
    assert _run_cli(["estimate-hybrid-index", "--input", vcf, "--metadata", meta,
                     "--pop1-label", "pop1", "--pop2-label", "pop2",
                     "--output", str(hi)]).returncode == 0
    assert _run_cli(["generate-genomic-input", "--input", vcf, "--metadata", str(hi),
                     "--parental-metadata", meta, "--pop1-label", "pop1",
                     "--pop2-label", "pop2", "--output", str(genomic)]).returncode == 0
    g = pd.read_csv(genomic)
    n_cols = [c for c in g.columns if c.endswith(".n")]
    n_missing = int((g[n_cols] == 0).sum().sum())  # missing: n = 0, freq empty
    assert n_missing > 0
    result = _run_cli(["fit-hierarchical-genomic", "--input", str(genomic),
                       "--individual-col", "individual", "--num-warmup", "150",
                       "--num-samples", "150", "--num-chains", "1", "--no-plot",
                       "--output-dir", str(tmp_path / "fit")])
    assert result.returncode == 0, result.stderr
    assert f"{n_missing} missing genotype(s) enter with 0 sampled alleles" in result.stdout
    loci = pd.read_csv(tmp_path / "fit" / "hycline_hierarchical_locus_summary.csv")
    assert len(loci) == 12 and np.isfinite(loci[["u_mean", "v_mean"]]).all().all()


def test_stack_genomic_by_individual_aligns_loci_and_zeroes_missing():
    import numpy as np
    from hycline.cli import _stack_genomic_by_individual
    from hycline.data.schema import HybridIndexLocusData

    data = {
        "A": HybridIndexLocusData(hybrid_index=np.array([0.1, 0.5, 0.9]),
                                  n_alleles=np.array([2, 2, 2]), n_focal=np.array([0, 1, 2]),
                                  individual_names=["x", "y", "z"]),
        "B": HybridIndexLocusData(hybrid_index=np.array([0.1, 0.9]),
                                  n_alleles=np.array([2, 2]), n_focal=np.array([1, 2]),
                                  individual_names=["x", "z"]),
    }
    names, h, n, k, n_missing = _stack_genomic_by_individual(data)
    assert names == ["A", "B"] and list(h) == [0.1, 0.5, 0.9] and n_missing == 1
    np.testing.assert_array_equal(n, [[2, 2, 2], [2, 0, 2]])
    np.testing.assert_array_equal(k, [[0, 1, 2], [1, 0, 2]])


def test_cli_genomic_pipeline_carries_reference_panel_uncertainty(tmp_path):
    """Regression test, built from the tool-comparison data that surfaced it
    (benchmarking/tool_comparison/small_panels/genotypes_s1: 5 + 5 parental
    reference individuals sampled from their populations, 100 hybrids drawn
    from the population frequencies). Taking the panels' frequencies as
    exact flagged 33% of neutral loci as centre outliers and gave hybrid-index
    intervals covering the true value 82% of the time; with the panels'
    sampling error carried through (the Beta posteriors generate-genomic-input
    writes as .p0_a/.p0_b/.p1_a/.p1_b), 1% and 95%."""
    sys.path.insert(0, str(REPO_ROOT / "benchmarking"))
    from compare_tools import collect, score_clines, score_hi
    run = REPO_ROOT / "benchmarking" / "tool_comparison" / "small_panels" / "genotypes_s1"
    src = run / "inputs" / "hycline"
    hy = tmp_path / "inputs" / "hycline"
    hy.mkdir(parents=True)
    labels = ["--pop1-label", "pop1", "--pop2-label", "pop2"]
    result = _run_cli(["estimate-hybrid-index", "--input", str(src / "input.vcf"),
                       "--metadata", str(src / "metadata.tsv"), *labels,
                       "--output", str(hy / "hybrid_index.tsv")])
    assert result.returncode == 0, result.stderr
    genomic = tmp_path / "genomic_input.csv"
    result = _run_cli(["generate-genomic-input", "--input", str(src / "input.vcf"),
                       "--metadata", str(hy / "hybrid_index.tsv"),
                       "--parental-metadata", str(src / "metadata.tsv"), *labels,
                       "--output", str(genomic)])
    assert result.returncode == 0, result.stderr
    g = pd.read_csv(genomic)
    beta_cols = [c for c in g.columns if c.endswith((".p0_a", ".p0_b", ".p1_a", ".p1_b"))]
    assert len(beta_cols) == 4 * len([c for c in g.columns if c.endswith(".p0")])
    result = _run_cli(["fit-genomic", "--input", str(genomic), "--individual-col", "individual",
                       "--no-plot", "--output-dir", str(hy / "fit_mle")])
    assert result.returncode == 0, result.stderr

    truth = pd.read_csv(run / "sim" / "sim_truth.tsv", sep="\t")
    meta = pd.read_csv(run / "sim" / "sim_metadata.tsv", sep="\t")
    his, cls = collect(tmp_path / "inputs", truth, False)
    hi = score_hi(his["HyCline"], meta)
    clines = score_clines(cls["HyCline MLE"], truth)
    assert hi["coverage95"] >= 0.9
    assert clines["c_false_pos"] < 0.1
    assert clines["v_false_pos"] < 0.2
    assert pd.read_csv(hy / "fit_mle" / "hycline_summary.csv")["parental_freqs_estimated"].all()

    # The same input without the panel sizes: frequencies taken as exact.
    g.drop(columns=beta_cols).to_csv(genomic, index=False)
    result = _run_cli(["fit-genomic", "--input", str(genomic), "--individual-col", "individual",
                       "--no-plot", "--output-dir", str(hy / "fit_mle")])
    assert result.returncode == 0, result.stderr
    _, cls = collect(tmp_path / "inputs", truth, False)
    assert score_clines(cls["HyCline MLE"], truth)["c_false_pos"] > 0.25
