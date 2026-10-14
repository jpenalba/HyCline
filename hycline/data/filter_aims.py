"""Filter whole-genome SNP data down to ancestry-informative markers
(AIMs) -- loci with strongly divergent allele frequencies between two
parental reference populations -- before conversion into HyCline's own
input formats.

A genome-wide SNP dataset is mostly uninformative for cline analysis:
most loci don't differ much between the two parental populations, and
including them dilutes real signal without adding power (this is exactly
the same reason `hycline.data.hybrid_index.estimate_hybrid_index`
supports `weight_by_differentiation` -- see BACKEND_PHASE3.md section 5).
Pre-filtering to loci that are strongly differentiated (by default, one
population's frequency in `[0.9, 1.0]` and the other's in `[0.0, 0.1]`,
direction-agnostic) is standard practice in this field before any
downstream cline or hybrid-index analysis. This module does that
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
from hycline.data.io import read_user_table


# Frequencies are ratios of allele counts, so a difference that is exactly
# the requested value on paper (0.82 - 0.12) can come out a hair below it
# in floating point; comparisons allow this much slack.
_FREQ_TOL = 1e-9


def compute_aim_mask(freq_pop1: np.ndarray, freq_pop2: np.ndarray,
                      low_threshold: float = 0.1, high_threshold: float = 0.9,
                      min_freq_diff: float | None = None) -> np.ndarray:
    """Boolean mask, one entry per locus: True where the locus is a
    strong ancestry-informative marker, in *either* direction (it doesn't
    matter which parental population carries the higher-frequency
    allele, only that the two populations are strongly differentiated).

    By default a locus qualifies when one population's frequency is at or
    above `high_threshold` and the other's at or below `low_threshold`
    (the CLI's --minor-allele-freq m sets these to m and 1 - m). If
    `min_freq_diff` is given, the thresholds are ignored and a locus
    qualifies when |freq_pop1 - freq_pop2| >= min_freq_diff (the CLI's
    --allele-freq-diff). NaN frequencies never qualify."""
    freq_pop1 = np.asarray(freq_pop1, dtype=float)
    freq_pop2 = np.asarray(freq_pop2, dtype=float)
    if min_freq_diff is not None:
        if not 0.0 < min_freq_diff <= 1.0:
            raise ValueError(f"min_freq_diff ({min_freq_diff}) must be in (0, 1].")
        with np.errstate(invalid="ignore"):
            return np.abs(freq_pop1 - freq_pop2) >= min_freq_diff - _FREQ_TOL
    if low_threshold >= high_threshold:
        raise ValueError(
            f"low_threshold ({low_threshold}) must be less than high_threshold "
            f"({high_threshold})."
        )
    low_threshold += _FREQ_TOL
    high_threshold -= _FREQ_TOL
    direction_a = (freq_pop1 >= high_threshold) & (freq_pop2 <= low_threshold)
    direction_b = (freq_pop1 <= low_threshold) & (freq_pop2 >= high_threshold)
    return direction_a | direction_b


def describe_aim_criterion(low_threshold: float = 0.1, high_threshold: float = 0.9,
                           min_freq_diff: float | None = None) -> str:
    """One-line description of the AIM criterion, for CLI output and summaries."""
    if min_freq_diff is not None:
        return (f"allele-frequency difference >= {min_freq_diff:g} between the "
                f"parental populations")
    return (f"<= {low_threshold:g} in one parental population and >= "
            f"{high_threshold:g} in the other")


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


def em_allele_freq(gl0: np.ndarray, gl1: np.ndarray, gl2: np.ndarray,
                    include: np.ndarray, max_iter: int = 1000,
                    tol: float = 1e-10) -> np.ndarray:
    """Per-locus maximum-likelihood focal-allele frequency from genotype
    likelihoods, pooling the individuals flagged in `include` under
    Hardy-Weinberg proportions, fitted by EM. All arrays are
    (n_loci, n_samples); returns (n_loci,), NaN where no individual is
    included.

    The likelihood of frequency f at a locus is
    prod_i [L0_i (1-f)^2 + L1_i 2f(1-f) + L2_i f^2]. Each EM step computes
    every individual's genotype posterior given the current f and sets f to
    the mean posterior dosage / 2. This is the same model as ANGSD's
    `-doMaf` EM estimator. Unlike averaging each individual's flat-prior
    expected dosage, it does not pull low-depth frequencies toward 0.5: a
    population of true homozygotes read at depth 2 estimates ~0, where the
    flat-prior average gives ~0.11 and fails a 0.05 AIM cutoff.

    The likelihood can have more than one local maximum (e.g. individuals
    favouring both homozygotes with few heterozygotes), so EM starts from
    the best point of a coarse grid over f rather than from a single guess.
    """
    include = np.asarray(include, dtype=bool)
    n_inc = include.sum(axis=1)
    freq = np.full(gl0.shape[0], np.nan)
    active = np.flatnonzero(n_inc > 0)
    if active.size == 0:
        return freq

    # Work only on loci and samples that contribute at least once.
    cols = np.flatnonzero(include.any(axis=0))
    rows = active[:, None]
    L = np.stack([gl0[rows, cols], gl1[rows, cols], gl2[rows, cols]],
                 axis=-1).astype(float)
    inc = include[rows, cols]
    L[~inc] = 1.0 / 3.0
    L /= L.sum(axis=-1, keepdims=True)
    w = inc.astype(float)
    n = n_inc[active].astype(float)

    # Start EM from the best point on a coarse grid, kept off the exact 0/1
    # boundaries (EM cannot move away from a boundary). Chunked over loci
    # to bound memory on large files.
    grid = np.linspace(1e-3, 1 - 1e-3, 51)
    grid_prior = np.stack([(1 - grid) ** 2, 2 * grid * (1 - grid), grid ** 2])
    f = np.empty(active.size)
    chunk = max(1, 2_000_000 // (L.shape[1] * grid.size))
    for a in range(0, active.size, chunk):
        b = min(a + chunk, active.size)
        with np.errstate(divide="ignore"):
            ll = (np.log(L[a:b] @ grid_prior) * w[a:b, :, None]).sum(axis=1)
        f[a:b] = grid[np.argmax(ll, axis=1)]
    todo = np.arange(active.size)
    for _ in range(max_iter):
        ft = f[todo][:, None]
        prior = np.stack([(1 - ft) ** 2, 2 * ft * (1 - ft), ft ** 2], axis=-1)
        post = L[todo] * prior
        post /= post.sum(axis=-1, keepdims=True)
        f_new = ((post[..., 1] + 2 * post[..., 2]) * w[todo]).sum(axis=1) / (2 * n[todo])
        moved = np.abs(f_new - f[todo]) > tol
        f[todo] = f_new
        todo = todo[moved]
        if todo.size == 0:
            break
    freq[active] = f
    return freq


def filter_aims_beagle(input_path: str, metadata_path: str, output_path: str,
                        sample_col: str = "individual", population_col: str = "population",
                        pop1_label: str = "parent1", pop2_label: str = "parent2",
                        low_threshold: float = 0.1, high_threshold: float = 0.9,
                        min_freq_diff: float | None = None) -> dict:
    """Filter an ANGSD beagle-format file down to ancestry-informative
    markers, writing a beagle file with the same column structure and
    unmodified per-individual genotype-likelihood values -- only rows
    (markers) are removed, nothing is recomputed.

    Returns a dict with `n_total`, `n_retained`, `freq_pop1`, `freq_pop2`
    (per-marker, for every marker -- not just the retained ones, so a
    caller can inspect what was filtered out and why), `marker_names`,
    and `mask`.
    """
    from hycline.data.beagle import read_beagle, write_beagle

    beagle_data = read_beagle(input_path)
    metadata_df = read_user_table(metadata_path, "metadata file")
    pop1_mask, pop2_mask = match_samples_to_populations(
        beagle_data.sample_names, metadata_df, sample_col, population_col,
        pop1_label, pop2_label)

    # Missing genotypes (ANGSD's equal-likelihood "no data" rows) carry no
    # information about the individual's alleles and are left out. A locus
    # with no calls in a population gets NaN, which never passes the filter.
    gl0, gl1, gl2 = beagle_data.gl0, beagle_data.gl1, beagle_data.gl2
    missing = beagle_missing_mask(gl0, gl1, gl2)
    freq_pop1 = em_allele_freq(gl0, gl1, gl2, ~missing & pop1_mask[None, :])
    freq_pop2 = em_allele_freq(gl0, gl1, gl2, ~missing & pop2_mask[None, :])

    mask = compute_aim_mask(freq_pop1, freq_pop2, low_threshold, high_threshold,
                            min_freq_diff)
    write_beagle(beagle_data, output_path, mask)

    return {
        "n_total": len(beagle_data), "n_retained": int(mask.sum()),
        "n_pop1_samples": int(pop1_mask.sum()), "n_pop2_samples": int(pop2_mask.sum()),
        "freq_pop1": freq_pop1, "freq_pop2": freq_pop2,
        "marker_names": beagle_data.marker_names, "mask": mask,
        **_missingness_fields(missing, mask, beagle_data.sample_names,
                              pop1_mask, pop2_mask),
    }


def beagle_missing_mask(gl0: np.ndarray, gl1: np.ndarray, gl2: np.ndarray) -> np.ndarray:
    """(n_loci, n_samples) boolean mask of missing beagle genotypes: NaN or
    all-zero likelihoods, or three equal likelihoods -- ANGSD writes
    0.333333/0.333333/0.333333 for an individual with no reads at a site,
    so equal likelihoods are its encoding of "no data"."""
    total = gl0 + gl1 + gl2
    return (np.isnan(total) | (total <= 0)
            | (np.isclose(gl0, gl1) & np.isclose(gl1, gl2)))


def _missingness_fields(missing: np.ndarray, aim_mask: np.ndarray, sample_names: list,
                         pop1_mask: np.ndarray, pop2_mask: np.ndarray) -> dict:
    """Missing-data bookkeeping shared by the VCF and beagle filters, from an
    (n_loci, n_samples) boolean missing mask."""
    missing = np.asarray(missing, dtype=bool)
    called = ~missing
    return {
        "sample_names": list(sample_names),
        "pop1_mask": pop1_mask, "pop2_mask": pop2_mask,
        "missing_per_sample": missing.sum(axis=0).astype(np.int64),
        "missing_per_sample_aims": missing[aim_mask].sum(axis=0).astype(np.int64),
        "n_called_pop1": called[:, pop1_mask].sum(axis=1).astype(np.int64),
        "n_called_pop2": called[:, pop2_mask].sum(axis=1).astype(np.int64),
    }


def filter_aims_csv(input_path: str, metadata_path: str, output_path: str,
                     sample_col: str = "individual", population_col: str = "population",
                     pop1_label: str = "parent1", pop2_label: str = "parent2",
                     low_threshold: float = 0.1, high_threshold: float = 0.9,
                     min_freq_diff: float | None = None, ploidy: int = 2, loci: list | None = None,
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
    metadata_df = read_user_table(metadata_path, "metadata file")

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

    mask = compute_aim_mask(freq_pop1, freq_pop2, low_threshold, high_threshold,
                            min_freq_diff)
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


def _open_write_maybe_gzip(path: str):
    import gzip
    if path.endswith(".gz"):
        return gzip.open(path, "wt")
    return open(path, "w")


def filter_aims_vcf(input_path: str, metadata_path: str, output_path: str,
                     sample_col: str = "individual", population_col: str = "population",
                     pop1_label: str = "parent1", pop2_label: str = "parent2",
                     low_threshold: float = 0.1, high_threshold: float = 0.9,
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
    from hycline.data.vcf import _open_maybe_gzip, _parse_gt

    metadata_df = read_user_table(metadata_path, "metadata file")

    locus_names = []
    freq_pop1_list = []
    freq_pop2_list = []
    keep_flags = []
    missing_rows = []
    n_skipped_multiallelic = 0
    n_skipped_indel = 0
    n_skipped_no_gt = 0

    header_samples = None
    pop1_mask = pop2_mask = None
    header_lines = []

    with _open_maybe_gzip(input_path) as fin, _open_write_maybe_gzip(output_path) as fout:
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
            keep = bool(compute_aim_mask(
                np.array([freq_pop1]), np.array([freq_pop2]),
                low_threshold, high_threshold, min_freq_diff)[0])

            locus_name = variant_id if variant_id != "." else f"{chrom}:{pos}"
            locus_names.append(locus_name)
            freq_pop1_list.append(freq_pop1)
            freq_pop2_list.append(freq_pop2)
            keep_flags.append(keep)
            missing_rows.append(np.isnan(dosage_row))

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
        **_missingness_fields(np.vstack(missing_rows), mask, header_samples,
                              pop1_mask, pop2_mask),
    }


def _pct(num: float, den: float) -> str:
    return f"{100.0 * num / den:.2f}%" if den else "n/a"


def build_filter_summary(result: dict, *, input_path: str, output_path: str, fmt: str,
                          pop1_label: str, pop2_label: str,
                          low_threshold: float = 0.1, high_threshold: float = 0.9,
                          min_freq_diff: float | None = None,
                          min_parental_n: int = 10,
                          max_sample_missing: float = 0.5) -> tuple:
    """Plain-text summary of a `filter_aims_vcf`/`filter_aims_beagle` run.

    Returns `(text, warnings)`. Warnings are raised for:
      - a parental population with fewer than `min_parental_n` individuals;
      - a parental individual missing at more than `max_sample_missing` of
        the loci checked;
      - retained AIMs where fewer than `min_parental_n` individuals of a
        parental population had a genotype call -- missing data shrinks the
        per-locus sample the frequency was actually computed from, so these
        AIMs rest on fewer individuals than the panel size suggests. Only
        populations whose panel meets `min_parental_n` are checked here (a
        smaller panel already gets the population-size warning).
    Missing data is counted over the biallelic SNPs checked (VCF: `./.`;
    beagle: equal or absent genotype likelihoods).
    """
    names = result["sample_names"]
    pop1_mask, pop2_mask = result["pop1_mask"], result["pop2_mask"]
    n_loci, n_aims = result["n_total"], result["n_retained"]
    miss_all, miss_aims = result["missing_per_sample"], result["missing_per_sample_aims"]
    parental = pop1_mask | pop2_mask
    n_samples, n_par = len(names), int(parental.sum())

    issues = []
    for label, n in ((pop1_label, result["n_pop1_samples"]),
                     (pop2_label, result["n_pop2_samples"])):
        if n < min_parental_n:
            issues.append(
                f"Parental population '{label}' has only {n} individual(s) (fewer than "
                f"{min_parental_n}); allele-frequency estimates and AIM calls from so "
                f"few individuals are unreliable.")
    flagged = []
    for i in np.flatnonzero(parental):
        frac = miss_all[i] / n_loci if n_loci else 0.0
        if frac > max_sample_missing:
            flagged.append(names[i])
            issues.append(
                f"Parental individual '{names[i]}' is missing data at {_pct(miss_all[i], n_loci)} "
                f"of loci checked (more than {100 * max_sample_missing:g}%).")
    aim_mask = result["mask"]
    thin = np.zeros_like(aim_mask)
    for key, n in (("n_called_pop1", result["n_pop1_samples"]),
                   ("n_called_pop2", result["n_pop2_samples"])):
        if n >= min_parental_n:
            thin |= aim_mask & (result[key] < min_parental_n)
    n_thin = int(thin.sum())
    if n_thin:
        issues.append(
            f"{n_thin} of {n_aims} retained AIMs had fewer than {min_parental_n} called "
            f"individuals in at least one parental population at that locus, because of "
            f"missing data.")

    lines = [
        "HyCline filter-aims summary",
        "===========================",
        f"Input:      {input_path} ({fmt})",
        f"Output:     {output_path}",
        f"Parentals:  {pop1_label} = {result['n_pop1_samples']} individuals, "
        f"{pop2_label} = {result['n_pop2_samples']} individuals",
        f"AIM rule:   "
        f"{describe_aim_criterion(low_threshold, high_threshold, min_freq_diff)}",
        "",
        "AIMs",
        "----",
        f"Loci checked:   {n_loci}",
        f"AIMs retained:  {n_aims} ({_pct(n_aims, n_loci)} of loci checked)",
    ]
    if fmt == "vcf":
        lines.append(f"Sites skipped:  {result['n_skipped_multiallelic']} multiallelic, "
                     f"{result['n_skipped_indel']} indel, "
                     f"{result['n_skipped_no_gt']} with no GT field")
    lines += [
        "",
        "Missing data (over loci checked)",
        "--------------------------------",
        f"{'Global, all samples (' + str(n_samples) + '):':<38}"
        f"{_pct(miss_all.sum(), n_loci * n_samples)}",
        f"{'Global, parental samples (' + str(n_par) + '):':<38}"
        f"{_pct(miss_all[parental].sum(), n_loci * n_par)}",
        f"{'Parental samples, retained AIMs only:':<38}"
        f"{_pct(miss_aims[parental].sum(), n_aims * n_par)}",
        "",
        "Per-sample missing data (parental samples)",
        "------------------------------------------",
        "sample\tpopulation\tmissing_all_loci\tmissing_aims\tflag",
    ]
    for i in np.flatnonzero(parental):
        label = pop1_label if pop1_mask[i] else pop2_label
        flag = "HIGH_MISSING" if names[i] in flagged else ""
        lines.append(f"{names[i]}\t{label}\t{_pct(miss_all[i], n_loci)}\t"
                     f"{_pct(miss_aims[i], n_aims)}\t{flag}")
    lines += ["", "Warnings", "--------"]
    lines += [f"WARNING: {w}" for w in issues] or ["None."]
    return "\n".join(lines) + "\n", issues
