import numpy as np
import pandas as pd
import pytest

from hycline.data.filter_aims import (
    compute_aim_mask, match_samples_to_populations,
    filter_aims_beagle, filter_aims_csv, filter_aims_vcf,
    beagle_missing_mask, build_filter_summary,
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


def test_compute_aim_mask_custom_thresholds():
    freq_pop1 = np.array([0.15])
    freq_pop2 = np.array([0.80])
    assert not compute_aim_mask(freq_pop1, freq_pop2, low_threshold=0.1, high_threshold=0.9)[0]
    assert compute_aim_mask(freq_pop1, freq_pop2, low_threshold=0.2, high_threshold=0.8)[0]


def test_compute_aim_mask_invalid_thresholds_raises():
    with pytest.raises(ValueError, match="must be less than"):
        compute_aim_mask([0.1], [0.9], low_threshold=0.9, high_threshold=0.1)


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


# ---------------------------------------------------------------------------
# missing-data bookkeeping and build_filter_summary
# ---------------------------------------------------------------------------

def _summary_vcf(tmp_path, rows, samples, populations):
    header = ("##fileformat=VCFv4.2\n"
              "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
              + "\t".join(samples) + "\n")
    lines = [f"chr1\t{100 * (i + 1)}\tsnp{i + 1}\tA\tT\t.\tPASS\t.\tGT\t"
             + "\t".join(gts) + "\n" for i, gts in enumerate(rows)]
    vcf_path = tmp_path / "in.vcf"
    vcf_path.write_text(header + "".join(lines))
    meta_path = tmp_path / "meta.csv"
    pd.DataFrame({"individual": samples, "population": populations}).to_csv(
        meta_path, index=False)
    return vcf_path, meta_path


def test_filter_aims_vcf_missing_data_counts_match_hand_computation(tmp_path):
    # p1a/p1b = parent1, p2a = parent2, h = hybrid.
    # snp1: AIM, p1b missing          snp2: not an AIM, p2a and h missing
    # snp3: AIM, nothing missing
    rows = [["0/0", "./.", "1/1", "0/1"],
            ["0/1", "0/1", "./.", "./."],
            ["1/1", "1/1", "0/0", "0/0"]]
    vcf_path, meta_path = _summary_vcf(
        tmp_path, rows, ["p1a", "p1b", "p2a", "h"],
        ["parent1", "parent1", "parent2", "hybrid"])
    result = filter_aims_vcf(str(vcf_path), str(meta_path), str(tmp_path / "out.vcf"))

    np.testing.assert_array_equal(result["mask"], [True, False, True])
    assert result["sample_names"] == ["p1a", "p1b", "p2a", "h"]
    np.testing.assert_array_equal(result["missing_per_sample"], [0, 1, 1, 1])
    np.testing.assert_array_equal(result["missing_per_sample_aims"], [0, 1, 0, 0])
    np.testing.assert_array_equal(result["n_called_pop1"], [1, 2, 2])
    np.testing.assert_array_equal(result["n_called_pop2"], [1, 0, 1])


def test_beagle_missing_mask_treats_equal_and_absent_likelihoods_as_missing():
    gl0 = np.array([[1/3, 0.98, 0.0, np.nan]])
    gl1 = np.array([[1/3, 0.02, 0.0, 0.5]])
    gl2 = np.array([[1/3, 0.00, 0.0, 0.5]])
    np.testing.assert_array_equal(beagle_missing_mask(gl0, gl1, gl2),
                                  [[True, False, True, True]])


def test_filter_aims_beagle_reports_missing_per_sample(tmp_path):
    content = (
        "marker\tallele1\tallele2\tind0\tind0\tind0\tind1\tind1\tind1\n"
        "m1\t0\t1\t0.98\t0.02\t0.00\t0.333333\t0.333333\t0.333333\n"
        "m2\t0\t1\t0.98\t0.02\t0.00\t0.00\t0.02\t0.98\n"
    )
    beagle_path = tmp_path / "in.beagle"
    beagle_path.write_text(content)
    meta_path = tmp_path / "meta.csv"
    pd.DataFrame({"individual": ["ind0", "ind1"],
                  "population": ["parent1", "parent2"]}).to_csv(meta_path, index=False)
    result = filter_aims_beagle(str(beagle_path), str(meta_path),
                                str(tmp_path / "out.beagle"))
    np.testing.assert_array_equal(result["missing_per_sample"], [0, 1])
    np.testing.assert_array_equal(result["n_called_pop2"], [0, 1])


def _summary(result, **kw):
    return build_filter_summary(
        result, input_path="in.vcf", output_path="out.vcf", fmt="vcf",
        pop1_label="parent1", pop2_label="parent2",
        low_threshold=0.1, high_threshold=0.9, **kw)


def test_build_filter_summary_reports_counts_and_no_warnings(tmp_path):
    # 6 + 6 parentals, one hybrid; 4 loci, 2 AIMs; 2 missing calls total
    samples = [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(6)] + ["h"]
    pops = ["parent1"] * 6 + ["parent2"] * 6 + ["hybrid"]
    aim = ["0/0"] * 6 + ["1/1"] * 6 + ["0/1"]
    non = ["0/1"] * 13
    non_missing = ["./."] + ["0/1"] * 11 + ["./."]
    vcf_path, meta_path = _summary_vcf(tmp_path, [aim, non, aim, non_missing],
                                       samples, pops)
    result = filter_aims_vcf(str(vcf_path), str(meta_path), str(tmp_path / "out.vcf"))
    text, warnings = _summary(result, min_parental_n=6)

    assert warnings == []
    assert "AIMs retained:  2 (50.00% of loci checked)" in text
    # 2 of 13*4 = 52 genotypes missing overall; 1 of 12*4 = 48 parental
    assert "3.85%" in text and "2.08%" in text
    assert "a0\tparent1\t25.00%\t0.00%\t" in text
    assert "\nh\t" not in text  # hybrids are not listed per-sample
    assert text.rstrip().endswith("None.")


def test_build_filter_summary_warns_small_panel_and_high_missing(tmp_path):
    samples = ["a0", "a1", "b0", "b1", "b2", "b3", "b4", "b5"]
    pops = ["parent1"] * 2 + ["parent2"] * 6
    rows = [["0/0", "./."] + ["1/1"] * 6,
            ["0/0", "./."] + ["1/1"] * 6,
            ["0/0", "0/0"] + ["1/1"] * 6]
    vcf_path, meta_path = _summary_vcf(tmp_path, rows, samples, pops)
    result = filter_aims_vcf(str(vcf_path), str(meta_path), str(tmp_path / "out.vcf"))
    text, warnings = _summary(result, min_parental_n=6)

    assert len(warnings) == 2
    assert "'parent1' has only 2 individual(s)" in warnings[0]
    assert "'a1' is missing data at 66.67%" in warnings[1]
    assert "a1\tparent1\t66.67%\t66.67%\tHIGH_MISSING" in text
    # parent1's per-locus shortfall is already covered by the panel-size warning
    assert not any("called individuals" in w for w in warnings)

    _, warnings = _summary(result, min_parental_n=2, max_sample_missing=0.7)
    assert len(warnings) == 1
    assert "2 of 3 retained AIMs had fewer than 2 called individuals" in warnings[0]


def test_filter_aims_beagle_excludes_missing_genotypes_from_frequencies(tmp_path):
    """Regression test: ANGSD writes 0.333333/0.333333/0.333333 for an
    individual with no reads. Those rows used to enter the population mean
    as a frequency of 0.5, so one missing parental individual out of three
    pulled a fixed difference (0 vs 1) to 0.167 vs 1 and the AIM failed the
    default 0.1 cutoff. Missing rows must be excluded instead."""
    miss = "0.333333\t0.333333\t0.333333"
    hom_ref, hom_alt = "1.00\t0.00\t0.00", "0.00\t0.00\t1.00"
    header = "marker\tallele1\tallele2\t" + "\t".join(
        f"{s}\t{s}\t{s}" for s in ["a0", "a1", "a2", "b0", "b1", "b2"]) + "\n"
    content = (header
               + "\t".join(["m1", "0", "1", hom_ref, hom_ref, miss,
                            hom_alt, hom_alt, hom_alt]) + "\n"
               + "\t".join(["m2", "0", "1", miss, miss, miss,
                            hom_alt, hom_alt, hom_alt]) + "\n")
    beagle_path = tmp_path / "in.beagle"
    beagle_path.write_text(content)
    meta_path = tmp_path / "meta.csv"
    pd.DataFrame({"individual": ["a0", "a1", "a2", "b0", "b1", "b2"],
                  "population": ["parent1"] * 3 + ["parent2"] * 3}).to_csv(
        meta_path, index=False)

    result = filter_aims_beagle(str(beagle_path), str(meta_path),
                                str(tmp_path / "out.beagle"))
    np.testing.assert_allclose(result["freq_pop1"][:1], [0.0], atol=1e-9)
    assert np.isnan(result["freq_pop1"][1])  # no calls at all in parent1
    np.testing.assert_array_equal(result["mask"], [True, False])
    np.testing.assert_array_equal(result["n_called_pop1"], [2, 0])
