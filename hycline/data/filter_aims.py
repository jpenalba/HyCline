"""Filter whole-genome SNP data down to ancestry-informative markers
(AIMs) -- loci with strongly divergent allele frequencies between two
parental reference populations -- before conversion into HyCline's own
input formats.

A genome-wide SNP dataset is mostly uninformative for cline analysis:
most loci don't differ much between the two parental populations, and
including them dilutes real signal without adding power (this is exactly
the same reason `hycline.data.hybrid_index.estimate_hybrid_index`
supports `weight_by_differentiation` -- see BACKEND_PHASE3.md section 5).
Pre-filtering to loci that are strongly differentiated -- by default, one
population's frequency in `[0.9, 1.0]` and the other's in `[0.0, 0.1]`
(`compute_aim_mask`, direction-agnostic), or alternatively any locus with
at least a given absolute frequency difference between the two
populations regardless of where each one falls (`compute_aim_mask_by_diff`)
-- is standard practice in this field before any downstream cline or
hybrid-index analysis. This module does that
filtering directly on the two real upstream formats this project's own
workflow is built around -- a VCF file or ANGSD's own beagle-format
genotype-likelihood output -- producing a filtered file in the *same*
format, ready for this project's existing converters
(`generate-geographic-input`/`generate-genomic-input`) or, for direct
comparison against any other tool that also reads VCF/beagle.

`filter_aims_csv` (a wide genotype-dosage CSV, one row per individual)
is kept for backward compatibility but is no longer part of this
project's documented CLI workflow -- VCF and beagle are the two
directly-supported genome-wide input formats going forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hycline.data.converter import DEFAULT_RESERVED_COLUMNS, detect_genotype_loci


def compute_aim_mask(freq_pop1: np.ndarray, freq_pop2: np.ndarray,
                      high_threshold: float = 0.9) -> np.ndarray:
    """Boolean mask, one entry per locus: True where the locus is a
    strong ancestry-informative marker -- one population's frequency at
    or above `high_threshold` and the other's at or below
    `1 - high_threshold`, in *either* direction (it doesn't matter which
    parental population happens to carry the higher-frequency allele,
    only that the two populations are strongly differentiated).

    Only one threshold, not two: the low side is always `1 -
    high_threshold`, never independently set -- an asymmetric pair
    (e.g. `>= 0.9` in one population but `<= 0.2`, not `0.1`, in the
    other) was never actually used, and carrying two numbers that must
    always satisfy `low == 1 - high` is redundant, not extra
    flexibility. See `compute_aim_mask_by_diff` for a genuinely
    different (not just re-parameterized) criterion: a locus can pass
    that test without either population being anywhere near the 0/1
    extremes this function requires."""
    freq_pop1 = np.asarray(freq_pop1, dtype=float)
    freq_pop2 = np.asarray(freq_pop2, dtype=float)
    if not (0.5 < high_threshold <= 1.0):
        raise ValueError(
            f"--high-threshold ({high_threshold}) must be greater than 0.5 and at "
            f"most 1.0 (the low side is always 1 - --high-threshold, so it must stay "
            f"below --high-threshold itself)."
        )
    low_threshold = 1.0 - high_threshold
    direction_a = (freq_pop1 >= high_threshold) & (freq_pop2 <= low_threshold)
    direction_b = (freq_pop1 <= low_threshold) & (freq_pop2 >= high_threshold)
    return direction_a | direction_b


def compute_aim_mask_by_diff(freq_pop1: np.ndarray, freq_pop2: np.ndarray,
                              min_freq_diff: float = 0.5) -> np.ndarray:
    """Boolean mask, one entry per locus: True where the absolute
    allele-frequency difference between the two parental populations is
    at least `min_freq_diff`.

    A genuinely different criterion from `compute_aim_mask`'s two-sided
    extremes test, not just a different parameterization of the same
    one: a locus with freq_pop1=0.7, freq_pop2=0.1 has a 0.6 frequency
    difference (would pass `--min-freq-diff 0.5`) but fails
    `compute_aim_mask`'s default `--high-threshold 0.9` test outright,
    since neither population reaches 0.9 or drops to 0.1. Which
    criterion is more appropriate depends on the downstream use --
    `compute_aim_mask`'s extremes test favors loci closest to fixed
    differences (the strongest possible ancestry signal per locus,
    fewer retained loci); this favors loci with a real, usable
    difference wherever it falls (more retained loci, weaker signal
    each)."""
    freq_pop1 = np.asarray(freq_pop1, dtype=float)
    freq_pop2 = np.asarray(freq_pop2, dtype=float)
    if not (0.0 < min_freq_diff <= 1.0):
        raise ValueError(
            f"--min-freq-diff ({min_freq_diff}) must be greater than 0.0 and at "
            f"most 1.0."
        )
    return np.abs(freq_pop1 - freq_pop2) >= min_freq_diff


def _resolve_aim_mask(freq_pop1: np.ndarray, freq_pop2: np.ndarray,
                       high_threshold: float | None, min_freq_diff: float | None) -> np.ndarray:
    """Shared dispatch between `compute_aim_mask`'s two-sided-threshold
    mode and `compute_aim_mask_by_diff`'s absolute-difference mode --
    mutually exclusive, matching every other "give exactly one of X or
    Y" convention in this project (e.g. `estimate-hybrid-index`'s two
    modes). `high_threshold` defaults to 0.9 (the pre-existing default
    threshold-mode behavior) when neither is given explicitly."""
    if high_threshold is not None and min_freq_diff is not None:
        raise ValueError(
            "--high-threshold and --min-freq-diff are mutually exclusive -- give "
            "at most one (default: --high-threshold 0.9 if neither is given)."
        )
    if min_freq_diff is not None:
        return compute_aim_mask_by_diff(freq_pop1, freq_pop2, min_freq_diff)
    return compute_aim_mask(freq_pop1, freq_pop2,
                             high_threshold if high_threshold is not None else 0.9)


def match_samples_to_populations(sample_names: list, metadata_df: pd.DataFrame,
                                  sample_col: str, population_col: str,
                                  pop1_label: str, pop2_label: str) -> tuple:
    """Match a fixed-order list of sample names against a metadata
    DataFrame's population labels, returning boolean masks (aligned to
    `sample_names`' order) for each population.

    Individuals with no metadata entry, or a label other than
    `pop1_label`/`pop2_label` (e.g. hybrid-zone individuals also present
    in the same file as the parental references), are simply excluded
    from both masks -- not an error. Only an empty population (zero
    matching samples) or a missing required column raises.
    """
    if sample_col not in metadata_df.columns:
        raise ValueError(f"--sample-col '{sample_col}' not found in the metadata file.")
    if population_col not in metadata_df.columns:
        raise ValueError(f"--population-col '{population_col}' not found in the metadata file.")

    dup_samples = metadata_df[sample_col][metadata_df[sample_col].duplicated()].unique().tolist()
    if dup_samples:
        raise ValueError(
            f"The metadata file has more than one row for sample(s) {dup_samples} "
            f"(--sample-col '{sample_col}') -- each sample must appear at most once. "
            f"Remove or merge the duplicate rows before running this command."
        )

    meta_by_sample = metadata_df.set_index(sample_col)[population_col]
    labels = [meta_by_sample.get(s) for s in sample_names]
    pop1_mask = np.array([l == pop1_label for l in labels])
    pop2_mask = np.array([l == pop2_label for l in labels])

    if not pop1_mask.any():
        raise ValueError(
            f"No samples labeled '{pop1_label}' (--pop1-label) found in the metadata "
            f"file, matched against {len(sample_names)} samples in the input file."
        )
    if not pop2_mask.any():
        raise ValueError(
            f"No samples labeled '{pop2_label}' (--pop2-label) found in the metadata "
            f"file, matched against {len(sample_names)} samples in the input file."
        )
    return pop1_mask, pop2_mask


def _expected_freq_from_gl(gl0: np.ndarray, gl1: np.ndarray, gl2: np.ndarray) -> np.ndarray:
    """Per-individual expected focal-allele frequency from genotype
    likelihoods, treating them as a flat-prior posterior (normalized per
    individual to guard against beagle values that aren't already
    normalized to sum to 1): E[dosage]/2 = (0*p0 + 1*p1 + 2*p2)/2. A
    simplified but standard and defensible estimator for filtering
    purposes -- not the full per-site MLE-via-EM allele-frequency
    estimator ANGSD's own `-doMaf` implements (see this module's
    docstring / BACKEND_PHASE1.md section 9.6 for the honest scope note).
    """
    total = gl0 + gl1 + gl2
    with np.errstate(invalid="ignore", divide="ignore"):
        dosage = (gl1 + 2 * gl2) / total
    return dosage / 2.0


def filter_aims_beagle(input_path: str, metadata_path: str, output_path: str,
                        sample_col: str = "individual", population_col: str = "population",
                        pop1_label: str = "parent1", pop2_label: str = "parent2",
                        high_threshold: float | None = None,
                        min_freq_diff: float | None = None) -> dict:
    """Filter an ANGSD beagle-format file down to ancestry-informative
    markers, writing a beagle file with the same column structure and
    unmodified per-individual genotype-likelihood values -- only rows
    (markers) are removed, nothing is recomputed.

    `high_threshold`/`min_freq_diff` select between `compute_aim_mask`'s
    two-sided-extremes test and `compute_aim_mask_by_diff`'s absolute-
    difference test -- mutually exclusive, see `_resolve_aim_mask`.
    Default (both `None`): `compute_aim_mask` with `high_threshold=0.9`.

    Returns a dict with `n_total`, `n_retained`, `freq_pop1`, `freq_pop2`
    (per-marker, for every marker -- not just the retained ones, so a
    caller can inspect what was filtered out and why), `marker_names`,
    and `mask`.
    """
    from hycline.data.beagle import read_beagle, write_beagle

    beagle_data = read_beagle(input_path)
    metadata_df = pd.read_csv(metadata_path)
    pop1_mask, pop2_mask = match_samples_to_populations(
        beagle_data.sample_names, metadata_df, sample_col, population_col,
        pop1_label, pop2_label)

    ind_freq = _expected_freq_from_gl(beagle_data.gl0, beagle_data.gl1, beagle_data.gl2)
    freq_pop1 = np.nanmean(ind_freq[:, pop1_mask], axis=1)
    freq_pop2 = np.nanmean(ind_freq[:, pop2_mask], axis=1)

    mask = _resolve_aim_mask(freq_pop1, freq_pop2, high_threshold, min_freq_diff)
    write_beagle(beagle_data, output_path, mask)

    return {
        "n_total": len(beagle_data), "n_retained": int(mask.sum()),
        "n_pop1_samples": int(pop1_mask.sum()), "n_pop2_samples": int(pop2_mask.sum()),
        "freq_pop1": freq_pop1, "freq_pop2": freq_pop2,
        "marker_names": beagle_data.marker_names, "mask": mask,
    }


def filter_aims_csv(input_path: str, metadata_path: str, output_path: str,
                     sample_col: str = "individual", population_col: str = "population",
                     pop1_label: str = "parent1", pop2_label: str = "parent2",
                     high_threshold: float | None = None, min_freq_diff: float | None = None,
                     ploidy: int = 2, loci: list | None = None,
                     reserved_columns=DEFAULT_RESERVED_COLUMNS) -> dict:
    """Filter a wide genotype-dosage CSV (the same shape
    `estimate-hybrid-index` uses -- one row per individual, one column
    per locus holding 0/1/2 focal-allele copies) down to ancestry-
    informative markers, writing out the same rows with only the
    retained locus columns kept
    (every reserved/covariate column is passed through unchanged).

    Returns the same dict shape as `filter_aims_beagle`.

    Kept for backward compatibility -- VCF and beagle are the two
    directly-supported genome-wide input formats going forward (see
    `filter_aims_vcf`); this CSV path is no longer exercised by the
    CLI's own documented workflow, but the function itself is unchanged
    and still works for any caller still using it directly.
    """
    geno_df = pd.read_csv(input_path)
    metadata_df = pd.read_csv(metadata_path)

    if sample_col not in geno_df.columns:
        raise ValueError(f"--sample-col '{sample_col}' not found in '{input_path}'.")

    if loci is None:
        loci = detect_genotype_loci(geno_df, reserved_columns={*reserved_columns,
                                                                 sample_col.lower()})
        if not loci:
            raise ValueError(
                f"No locus columns detected in '{input_path}'. Pass --loci explicitly."
            )
    else:
        missing = [l for l in loci if l not in geno_df.columns]
        if missing:
            raise ValueError(f"Requested loci {missing} not found in '{input_path}'.")

    sample_names = geno_df[sample_col].tolist()
    pop1_mask, pop2_mask = match_samples_to_populations(
        sample_names, metadata_df, sample_col, population_col, pop1_label, pop2_label)

    dosage = geno_df[loci].to_numpy(dtype=float)
    freq_pop1 = np.nanmean(dosage[pop1_mask], axis=0) / ploidy
    freq_pop2 = np.nanmean(dosage[pop2_mask], axis=0) / ploidy

    mask = _resolve_aim_mask(freq_pop1, freq_pop2, high_threshold, min_freq_diff)
    retained_loci = [l for l, keep in zip(loci, mask) if keep]

    reserved_present = [c for c in geno_df.columns if c not in loci]
    out_df = geno_df[reserved_present + retained_loci]
    out_df.to_csv(output_path, index=False)

    return {
        "n_total": len(loci), "n_retained": len(retained_loci),
        "n_pop1_samples": int(pop1_mask.sum()), "n_pop2_samples": int(pop2_mask.sum()),
        "freq_pop1": freq_pop1, "freq_pop2": freq_pop2,
        "marker_names": loci, "mask": mask,
    }


def filter_aims_vcf(input_path: str, metadata_path: str, output_path: str,
                     sample_col: str = "individual", population_col: str = "population",
                     pop1_label: str = "parent1", pop2_label: str = "parent2",
                     high_threshold: float | None = None,
                     min_freq_diff: float | None = None) -> dict:
    """Filter a VCF file down to ancestry-informative markers, writing a
    VCF file with the same header and unmodified data lines -- only
    lines (variants) are removed; every retained line is copied
    byte-for-byte from the input, not re-derived from parsed fields, so
    INFO annotations, QUAL/FILTER values, and any FORMAT field beyond
    GT survive filtering exactly as ANGSD/GATK/bcftools wrote them.

    Uses the same biallelic-SNP-with-GT scope as
    `hycline.data.vcf.read_vcf` (multiallelic sites, indels, and sites
    with no GT field are skipped and counted, not guessed at -- see
    that function's own docstring for why), but is a self-contained,
    single-pass implementation rather than calling `read_vcf` directly:
    `read_vcf` discards the original line text once it has parsed the
    fields it needs, and re-serializing a VCF line from parsed values
    risks silently dropping information a full VCF can carry that this
    project's own parser doesn't model (multi-sample FORMAT fields
    beyond GT/PL, INFO annotations, etc.) -- copying the original line
    verbatim avoids that risk entirely.

    Returns the same dict shape as `filter_aims_beagle`, plus
    `n_skipped_multiallelic`/`n_skipped_indel`/`n_skipped_no_gt` (using
    the same field names `hycline.data.vcf.VCFData` uses, for a
    familiar reporting story alongside `generate-geographic-input`/
    `generate-genomic-input`).
    """
    from hycline.data.beagle import _open_maybe_gzip as _open_write_maybe_gzip
    from hycline.data.vcf import _open_maybe_gzip, _parse_gt

    metadata_df = pd.read_csv(metadata_path)

    locus_names = []
    freq_pop1_list = []
    freq_pop2_list = []
    keep_flags = []
    n_skipped_multiallelic = 0
    n_skipped_indel = 0
    n_skipped_no_gt = 0

    header_samples = None
    pop1_mask = pop2_mask = None
    header_lines = []

    with _open_maybe_gzip(input_path) as fin, _open_write_maybe_gzip(output_path, mode="wt") as fout:
        for line in fin:
            raw_line = line
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("##"):
                fout.write(raw_line)
                continue
            if line.startswith("#CHROM"):
                fields = line.split("\t")
                header_samples = fields[9:]
                pop1_mask, pop2_mask = match_samples_to_populations(
                    header_samples, metadata_df, sample_col, population_col,
                    pop1_label, pop2_label)
                fout.write(raw_line)
                continue

            if header_samples is None:
                raise ValueError(f"'{input_path}' has no #CHROM header line -- not a valid VCF.")

            fields = line.split("\t")
            chrom, pos, variant_id, ref, alt = (fields[0], fields[1], fields[2],
                                                 fields[3], fields[4])

            alt_alleles = alt.split(",")
            if len(alt_alleles) != 1:
                n_skipped_multiallelic += 1
                continue
            if len(ref) != 1 or len(alt_alleles[0]) != 1:
                n_skipped_indel += 1
                continue

            format_keys = fields[8].split(":")
            try:
                gt_idx = format_keys.index("GT")
            except ValueError:
                n_skipped_no_gt += 1
                continue

            sample_fields = fields[9:]
            dosage_row = np.full(len(header_samples), np.nan)
            for i, sample_str in enumerate(sample_fields):
                sub_fields = sample_str.split(":")
                gt_str = sub_fields[gt_idx] if gt_idx < len(sub_fields) else "."
                dosage_row[i] = _parse_gt(gt_str)

            freq_pop1 = np.nanmean(dosage_row[pop1_mask]) / 2.0
            freq_pop2 = np.nanmean(dosage_row[pop2_mask]) / 2.0
            keep = bool(_resolve_aim_mask(
                np.array([freq_pop1]), np.array([freq_pop2]),
                high_threshold, min_freq_diff)[0])

            locus_name = variant_id if variant_id != "." else f"{chrom}:{pos}"
            locus_names.append(locus_name)
            freq_pop1_list.append(freq_pop1)
            freq_pop2_list.append(freq_pop2)
            keep_flags.append(keep)

            if keep:
                fout.write(raw_line)

    if not locus_names:
        raise ValueError(
            f"No biallelic SNPs found in '{input_path}' (after skipping "
            f"{n_skipped_multiallelic} multiallelic and {n_skipped_indel} indel sites). "
            f"Check the file is a valid VCF with biallelic SNP records."
        )

    mask = np.array(keep_flags, dtype=bool)
    return {
        "n_total": len(locus_names), "n_retained": int(mask.sum()),
        "n_pop1_samples": int(pop1_mask.sum()), "n_pop2_samples": int(pop2_mask.sum()),
        "freq_pop1": np.array(freq_pop1_list), "freq_pop2": np.array(freq_pop2_list),
        "marker_names": locus_names, "mask": mask,
        "n_skipped_multiallelic": n_skipped_multiallelic,
        "n_skipped_indel": n_skipped_indel,
        "n_skipped_no_gt": n_skipped_no_gt,
    }
