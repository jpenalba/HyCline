import numpy as np
import pandas as pd
import pytest

from hycline.data.filter_aims import (
    compute_aim_mask, compute_aim_mask_by_diff, _resolve_aim_mask,
    match_samples_to_populations,
    filter_aims_beagle, filter_aims_csv, filter_aims_vcf,
)


# ---------------------------------------------------------------------------
# compute_aim_mask
# ---------------------------------------------------------------------------

def test_compute_aim_mask_basic():
    freq_pop1 = np.array([0.02, 0.5, 0.95, 0.5])
    freq_pop2 = np.array([0.98, 0.5, 0.03, 0.6])
    mask = compute_aim_mask(freq_pop1, freq_pop2)
    np.testing.assert_array_equal(mask, [True, False, True, False])


def test_compute_aim_mask_direction_agnostic():
    # locus where pop1 is high and pop2 is low should also pass
    freq_pop1 = np.array([0.95])
    freq_pop2 = np.array([0.02])
    mask = compute_aim_mask(freq_pop1, freq_pop2)
    assert mask[0]


def test_compute_aim_mask_custom_threshold():
    # Regression test for the --low-threshold/--high-threshold collapse:
    # the low side is always 1 - high_threshold now, a single knob
    # instead of two independently-settable ones.
    freq_pop1 = np.array([0.15])
    freq_pop2 = np.array([0.80])
    assert not compute_aim_mask(freq_pop1, freq_pop2, high_threshold=0.9)[0]
    assert compute_aim_mask(freq_pop1, freq_pop2, high_threshold=0.8)[0]


def test_compute_aim_mask_invalid_threshold_raises():
    with pytest.raises(ValueError, match="must be greater than 0.5"):
        compute_aim_mask([0.1], [0.9], high_threshold=0.4)
    with pytest.raises(ValueError, match="must be greater than 0.5"):
        compute_aim_mask([0.1], [0.9], high_threshold=1.5)


# ---------------------------------------------------------------------------
# compute_aim_mask_by_diff
# ---------------------------------------------------------------------------

def test_compute_aim_mask_by_diff_basic():
    # freq_pop1=0.7/freq_pop2=0.1 has a 0.6 difference -- passes
    # --min-freq-diff 0.5 but would fail compute_aim_mask's default
    # --high-threshold 0.9 test outright (0.7 < 0.9), the exact
    # real-world case this mode exists for.
    freq_pop1 = np.array([0.7, 0.5, 0.95])
    freq_pop2 = np.array([0.1, 0.5, 0.90])
    mask = compute_aim_mask_by_diff(freq_pop1, freq_pop2, min_freq_diff=0.5)
    np.testing.assert_array_equal(mask, [True, False, False])
    assert not compute_aim_mask(freq_pop1, freq_pop2)[0]  # confirms the threshold mode misses it


def test_compute_aim_mask_by_diff_direction_agnostic():
    assert compute_aim_mask_by_diff(np.array([0.1]), np.array([0.7]), min_freq_diff=0.5)[0]
    assert compute_aim_mask_by_diff(np.array([0.7]), np.array([0.1]), min_freq_diff=0.5)[0]


def test_compute_aim_mask_by_diff_invalid_threshold_raises():
    with pytest.raises(ValueError, match="must be greater than 0.0"):
        compute_aim_mask_by_diff([0.1], [0.9], min_freq_diff=0.0)
    with pytest.raises(ValueError, match="must be greater than 0.0"):
        compute_aim_mask_by_diff([0.1], [0.9], min_freq_diff=1.5)


# ---------------------------------------------------------------------------
# _resolve_aim_mask (the two-mode dispatch)
# ---------------------------------------------------------------------------

def test_resolve_aim_mask_defaults_to_threshold_mode():
    freq_pop1, freq_pop2 = np.array([0.95]), np.array([0.02])
    result = _resolve_aim_mask(freq_pop1, freq_pop2, None, None)
    np.testing.assert_array_equal(result, compute_aim_mask(freq_pop1, freq_pop2, 0.9))


def test_resolve_aim_mask_uses_diff_mode_when_given():
    freq_pop1, freq_pop2 = np.array([0.7]), np.array([0.1])
    result = _resolve_aim_mask(freq_pop1, freq_pop2, None, 0.5)
    np.testing.assert_array_equal(
        result, compute_aim_mask_by_diff(freq_pop1, freq_pop2, 0.5))


def test_resolve_aim_mask_both_given_is_mutually_exclusive_error():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve_aim_mask(np.array([0.9]), np.array([0.1]), 0.9, 0.5)


# ---------------------------------------------------------------------------
# match_samples_to_populations
# ---------------------------------------------------------------------------

def test_match_samples_to_populations_basic():
    metadata = pd.DataFrame({
        "individual": ["a", "b", "c", "d"],
        "population": ["parent1", "parent1", "parent2", "hybrid"],
    })
    pop1_mask, pop2_mask = match_samples_to_populations(
        ["a", "b", "c", "d"], metadata, "individual", "population", "parent1", "parent2")
    np.testing.assert_array_equal(pop1_mask, [True, True, False, False])
    np.testing.assert_array_equal(pop2_mask, [False, False, True, False])


def test_match_samples_to_populations_unlisted_sample_not_an_error():
    metadata = pd.DataFrame({
        "individual": ["a", "b"], "population": ["parent1", "parent2"],
    })
    # "c" has no metadata row at all -- should just be excluded, not raise
    pop1_mask, pop2_mask = match_samples_to_populations(
        ["a", "b", "c"], metadata, "individual", "population", "parent1", "parent2")
    np.testing.assert_array_equal(pop1_mask, [True, False, False])
    np.testing.assert_array_equal(pop2_mask, [False, True, False])


def test_match_samples_to_populations_empty_population_raises():
    metadata = pd.DataFrame({"individual": ["a"], "population": ["parent1"]})
    with pytest.raises(ValueError, match="No samples labeled 'parent2'"):
        match_samples_to_populations(["a"], metadata, "individual", "population",
                                      "parent1", "parent2")


def test_match_samples_to_populations_missing_column_raises():
    metadata = pd.DataFrame({"sample": ["a"], "pop": ["parent1"]})
    with pytest.raises(ValueError, match="--sample-col"):
        match_samples_to_populations(["a"], metadata, "individual", "population",
                                      "parent1", "parent2")


def test_match_samples_to_populations_duplicate_sample_row_raises_clear_error():
    # Regression test: a metadata CSV with more than one row for the same
    # sample name used to break with an opaque numpy error ("setting an
    # array element with a sequence... inhomogeneous shape"), because
    # pandas Series.get() on a non-unique index returns a Series instead
    # of a scalar -- instead of a clear, actionable duplicate-row error.
    metadata = pd.DataFrame({
        "individual": ["ind1", "ind1", "ind2", "ind3"],
        "population": ["parent1", "parent1", "parent1", "parent2"],
    })
    with pytest.raises(ValueError, match="more than one row"):
        match_samples_to_populations(["ind1", "ind2", "ind3"], metadata, "individual",
                                      "population", "parent1", "parent2")


# ---------------------------------------------------------------------------
# filter_aims_beagle
# ---------------------------------------------------------------------------

BEAGLE_CONTENT = (
    "marker\tallele1\tallele2\tind0\tind0\tind0\tind1\tind1\tind1\tind2\tind2\tind2\n"
    "chr1_100\t0\t1\t0.98\t0.02\t0.00\t0.00\t0.02\t0.98\t0.49\t0.02\t0.49\n"
    "chr1_200\t0\t1\t0.60\t0.30\t0.10\t0.10\t0.30\t0.60\t0.33\t0.34\t0.33\n"
    "chr1_300\t0\t1\t1.00\t0.00\t0.00\t0.00\t0.00\t1.00\t1.00\t0.00\t0.00\n"
)


def test_filter_aims_beagle_matches_hand_computation(tmp_path):
    beagle_path = tmp_path / "test.beagle"
    beagle_path.write_text(BEAGLE_CONTENT)
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"individual": ["ind0", "ind1", "ind2"],
                  "population": ["parent1", "parent2", "hybrid"]}).to_csv(
        metadata_path, index=False)

    out_path = tmp_path / "filtered.beagle"
    result = filter_aims_beagle(str(beagle_path), str(metadata_path), str(out_path))

    assert result["n_total"] == 3
    assert result["n_retained"] == 2
    np.testing.assert_allclose(result["freq_pop1"], [0.01, 0.25, 0.0], atol=1e-6)
    np.testing.assert_allclose(result["freq_pop2"], [0.99, 0.75, 1.0], atol=1e-6)
    np.testing.assert_array_equal(result["mask"], [True, False, True])

    from hycline.data.beagle import read_beagle
    filtered = read_beagle(str(out_path))
    assert filtered.marker_names == ["chr1_100", "chr1_300"]


# ---------------------------------------------------------------------------
# filter_aims_csv
# ---------------------------------------------------------------------------

def test_filter_aims_csv_matches_hand_computation(tmp_path):
    input_path = tmp_path / "genotypes.csv"
    pd.DataFrame({
        "individual": ["ind0", "ind1", "ind2", "ind3"],
        "locusA": [0, 2, 1, 0],
        "locusB": [1, 1, 0, 2],
    }).to_csv(input_path, index=False)
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"individual": ["ind0", "ind1", "ind2", "ind3"],
                  "population": ["parent1", "parent2", "hybrid", "hybrid"]}).to_csv(
        metadata_path, index=False)

    out_path = tmp_path / "filtered.csv"
    result = filter_aims_csv(str(input_path), str(metadata_path), str(out_path))

    assert result["n_total"] == 2
    assert result["n_retained"] == 1
    np.testing.assert_allclose(result["freq_pop1"], [0.0, 0.5])
    np.testing.assert_allclose(result["freq_pop2"], [1.0, 0.5])

    out_df = pd.read_csv(out_path)
    assert list(out_df.columns) == ["individual", "locusA"]
    assert len(out_df) == 4  # all individuals retained, including hybrids


def test_filter_aims_csv_no_loci_detected_raises(tmp_path):
    input_path = tmp_path / "genotypes.csv"
    pd.DataFrame({"individual": ["a"], "distance": [0]}).to_csv(input_path, index=False)
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"individual": ["a"], "population": ["parent1"]}).to_csv(
        metadata_path, index=False)
    with pytest.raises(ValueError, match="No locus columns detected"):
        filter_aims_csv(str(input_path), str(metadata_path), str(tmp_path / "out.csv"))


def test_filter_aims_csv_explicit_loci(tmp_path):
    input_path = tmp_path / "genotypes.csv"
    pd.DataFrame({
        "individual": ["ind0", "ind1"], "locusA": [0, 2], "locusB": [0, 2],
    }).to_csv(input_path, index=False)
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"individual": ["ind0", "ind1"],
                  "population": ["parent1", "parent2"]}).to_csv(metadata_path, index=False)

    out_path = tmp_path / "filtered.csv"
    result = filter_aims_csv(str(input_path), str(metadata_path), str(out_path),
                              loci=["locusA"])
    assert result["n_total"] == 1
    assert result["marker_names"] == ["locusA"]


def test_filter_aims_csv_output_chains_into_genotype_to_gl(tmp_path):
    """End-to-end: the filtered CSV must be directly usable by the
    existing genotype-to-gl converter with zero extra configuration,
    matching every other converter's own validation standard."""
    import numpy as np
    from hycline.models.cline import ClineParams, TailType, cline_function
    from hycline.data.converter import convert_genotype_csv_to_gl_csv
    from hycline.data.io import read_gl_multi_locus_wide_csv
    from hycline.inference.mle_genotype_likelihood import fit_mle_geographic_gl

    rng = np.random.default_rng(71)
    n_parent_each = 8
    n_hybrid = 30
    n_ind = 2 * n_parent_each + n_hybrid
    distance = np.concatenate([
        np.full(n_parent_each, -50.0), np.full(n_parent_each, 150.0),
        np.sort(rng.uniform(0, 100, n_hybrid)),
    ])
    population = ["parent1"] * n_parent_each + ["parent2"] * n_parent_each + ["hybrid"] * n_hybrid
    true_params = ClineParams(center=50, width=20, pmin=0.05, pmax=0.95)
    p_true = cline_function(distance, true_params, TailType.NONE)

    freqs = np.where(np.array(population) == "parent1", 0.02,
                      np.where(np.array(population) == "parent2", 0.98, p_true))
    u = rng.uniform(size=n_ind)
    dosage = np.where(u < (1 - freqs) ** 2, 0,
                       np.where(u < (1 - freqs) ** 2 + 2 * freqs * (1 - freqs), 1, 2))

    input_path = tmp_path / "genotypes.csv"
    pd.DataFrame({
        "individual": [f"ind{i}" for i in range(n_ind)], "distance": distance,
        "locusA": dosage,
    }).to_csv(input_path, index=False)
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"individual": [f"ind{i}" for i in range(n_ind)],
                  "population": population}).to_csv(metadata_path, index=False)

    filtered_path = tmp_path / "filtered.csv"
    result = filter_aims_csv(str(input_path), str(metadata_path), str(filtered_path))
    assert result["n_retained"] == 1  # the one locus is genuinely informative

    gl_path = tmp_path / "converted.csv"
    convert_genotype_csv_to_gl_csv(str(filtered_path), str(gl_path))
    transects = read_gl_multi_locus_wide_csv(str(gl_path), individual_col="individual")
    fit = fit_mle_geographic_gl(transects["locusA"])
    assert fit.params.center == pytest.approx(50, abs=15)


# ---------------------------------------------------------------------------
# filter_aims_vcf
# ---------------------------------------------------------------------------

_VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    '##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tind0\tind1\tind2\n"
)


def _vcf_line(pos, snp_id, gts_with_dp):
    return (f"chr1\t{pos}\t{snp_id}\tA\tT\t99\tPASS\tDP=100\tGT:DP\t"
            + "\t".join(gts_with_dp) + "\n")


def test_filter_aims_vcf_matches_hand_computation(tmp_path):
    # ind0=parent1, ind1=parent2, ind2=hybrid (excluded from the frequency calc)
    # site1: ind0=0/0 (freq 0), ind1=1/1 (freq 1) -- informative
    # site2: ind0=0/1 (freq 0.5), ind1=0/1 (freq 0.5) -- not informative
    # site3: ind0=1/1 (freq 1), ind1=0/0 (freq 0) -- informative, opposite direction
    lines = [
        _VCF_HEADER,
        _vcf_line(100, "snp1", ["0/0:10", "1/1:12", "0/1:8"]),
        _vcf_line(200, "snp2", ["0/1:11", "0/1:9", "0/0:14"]),
        _vcf_line(300, "snp3", ["1/1:13", "0/0:15", "1/1:7"]),
    ]
    vcf_path = tmp_path / "test.vcf"
    vcf_path.write_text("".join(lines))
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"individual": ["ind0", "ind1", "ind2"],
                  "population": ["parent1", "parent2", "hybrid"]}).to_csv(
        metadata_path, index=False)

    out_path = tmp_path / "filtered.vcf"
    result = filter_aims_vcf(str(vcf_path), str(metadata_path), str(out_path))

    assert result["n_total"] == 3
    assert result["n_retained"] == 2
    np.testing.assert_allclose(result["freq_pop1"], [0.0, 0.5, 1.0], atol=1e-6)
    np.testing.assert_allclose(result["freq_pop2"], [1.0, 0.5, 0.0], atol=1e-6)
    np.testing.assert_array_equal(result["mask"], [True, False, True])
    assert result["marker_names"] == ["snp1", "snp2", "snp3"]
    assert result["n_skipped_multiallelic"] == 0
    assert result["n_skipped_indel"] == 0
    assert result["n_skipped_no_gt"] == 0


def test_filter_aims_vcf_output_preserves_lines_verbatim(tmp_path):
    """The core correctness guarantee: retained lines must be copied
    byte-for-byte, including FORMAT fields (here, DP) beyond GT that
    this project's own VCF parser doesn't otherwise model."""
    lines = [
        _VCF_HEADER,
        _vcf_line(100, "snp1", ["0/0:10", "1/1:12", "0/1:8"]),
        _vcf_line(200, "snp2", ["0/1:11", "0/1:9", "0/0:14"]),
    ]
    vcf_path = tmp_path / "test.vcf"
    vcf_path.write_text("".join(lines))
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"individual": ["ind0", "ind1", "ind2"],
                  "population": ["parent1", "parent2", "hybrid"]}).to_csv(
        metadata_path, index=False)

    out_path = tmp_path / "filtered.vcf"
    filter_aims_vcf(str(vcf_path), str(metadata_path), str(out_path))

    out_text = out_path.read_text()
    assert _VCF_HEADER in out_text
    assert lines[1] in out_text  # snp1 retained, byte-for-byte including :DP values
    assert lines[2] not in out_text  # snp2 filtered out
    assert "0/0:10" in out_text  # the extra FORMAT field specifically survived


def test_filter_aims_vcf_skips_and_counts_multiallelic_indel_no_gt(tmp_path):
    lines = [
        _VCF_HEADER,
        _vcf_line(100, "snp1", ["0/0:10", "1/1:12", "0/1:8"]),  # informative, kept
        "chr1\t150\tmulti\tA\tT,C\t99\tPASS\tDP=100\tGT:DP\t0/0:1\t1/1:1\t0/1:1\n",  # multiallelic
        "chr1\t175\tindel\tA\tAT\t99\tPASS\tDP=100\tGT:DP\t0/0:1\t1/1:1\t0/1:1\n",  # indel
        "chr1\t190\tnogt\tA\tT\t99\tPASS\tDP=100\tDP\t10\t12\t8\n",  # no GT format key
    ]
    vcf_path = tmp_path / "test.vcf"
    vcf_path.write_text("".join(lines))
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"individual": ["ind0", "ind1", "ind2"],
                  "population": ["parent1", "parent2", "hybrid"]}).to_csv(
        metadata_path, index=False)

    out_path = tmp_path / "filtered.vcf"
    result = filter_aims_vcf(str(vcf_path), str(metadata_path), str(out_path))

    assert result["n_total"] == 1  # only the valid biallelic SNP is counted
    assert result["n_retained"] == 1
    assert result["n_skipped_multiallelic"] == 1
    assert result["n_skipped_indel"] == 1
    assert result["n_skipped_no_gt"] == 1


def test_filter_aims_vcf_output_chains_into_convert_vcf_geographic(tmp_path):
    lines = [_VCF_HEADER]
    rng_gts = [
        ["0/0:10", "1/1:12", "0/1:8"],
        ["0/1:11", "0/1:9", "0/0:14"],
        ["1/1:13", "0/0:15", "1/1:7"],
    ]
    for i, gts in enumerate(rng_gts):
        lines.append(_vcf_line(100 + i * 100, f"snp{i+1}", gts))
    vcf_path = tmp_path / "test.vcf"
    vcf_path.write_text("".join(lines))
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"individual": ["ind0", "ind1", "ind2"],
                  "population": ["parent1", "parent2", "hybrid"]}).to_csv(
        metadata_path, index=False)

    filtered_path = tmp_path / "filtered.vcf"
    filter_aims_vcf(str(vcf_path), str(metadata_path), str(filtered_path))

    from hycline.data.vcf import read_vcf
    vcf_data = read_vcf(str(filtered_path))
    assert len(vcf_data) == 2  # snp1 and snp3, snp2 filtered out
    assert vcf_data.locus_names == ["snp1", "snp3"]


def test_filter_aims_vcf_no_chrom_header_raises(tmp_path):
    vcf_path = tmp_path / "bad.vcf"
    vcf_path.write_text("chr1\t100\tsnp1\tA\tT\t99\tPASS\tDP=100\tGT\t0/0\n")
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"individual": ["ind0"], "population": ["parent1"]}).to_csv(
        metadata_path, index=False)
    with pytest.raises(ValueError, match="not a valid VCF"):
        filter_aims_vcf(str(vcf_path), str(metadata_path), str(tmp_path / "out.vcf"))
