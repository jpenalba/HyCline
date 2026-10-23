"""A pure-Python VCF parser, and the VCF + metadata -> HyCline CSV
converter built on top of it.

No `pysam`/`cyvcf2` dependency, deliberately: both are C-extension
packages needing compilation against `htslib`, a real installation risk
in a sandboxed environment with no guarantee those will build cleanly.
VCF is a well-specified, line-oriented tab-delimited text format (VCFv4.x,
https://samtools.github.io/hts-specs/VCFv4.2.pdf) -- parseable with the
standard library alone (`gzip` + string splitting) for the subset of the
format this project actually needs (biallelic SNP genotype calls and,
optionally, Phred-scaled genotype likelihoods).

WHAT'S SUPPORTED: biallelic SNPs (single-base REF, single single-base
ALT) with a `GT` FORMAT field, and optionally a `PL` FORMAT field
(Phred-scaled genotype likelihoods, the modern GATK/bcftools/ANGSD
convention -- converted via `hycline.data.genotype_likelihood.
phred_to_log_likelihood`, the same converter every other GL-consuming
part of this project already uses). Handles both phased (`|`) and
unphased (`/`) genotype separators, and gzip-compressed (`.vcf.gz`)
files transparently.

WHAT'S NOT SUPPORTED, on purpose rather than by oversight: indels,
multiallelic sites (more than one ALT allele), and the older `GL`
(log10-scaled, not Phred-scaled) genotype-likelihood FORMAT field.
Multiallelic sites and indels are common in real VCFs and would need a
real modeling decision (which ALT is "focal"? how does a 3+-genotype-
class site fit this project's inherently biallelic HWE marginalization?)
rather than an arbitrary one -- skipped and counted, not guessed at. See
`read_vcf`'s docstring for the exact skip-counting behavior.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hycline.data.genotype_likelihood import phred_to_log_likelihood


@dataclass
class VCFData:
    locus_names: list
    sample_names: list
    dosage: np.ndarray  # (n_loci, n_samples), 0/1/2/NaN
    log_gl0: np.ndarray | None = None  # (n_loci, n_samples) or None if PL wasn't requested/found
    log_gl1: np.ndarray | None = None
    log_gl2: np.ndarray | None = None
    n_skipped_multiallelic: int = 0
    n_skipped_indel: int = 0
    n_skipped_no_gt: int = 0
    n_loci_without_pl: int = 0  # loci where PL was requested but missing for every sample

    def __len__(self) -> int:
        return len(self.locus_names)


def _open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _parse_gt(gt_str: str) -> float:
    """Parse a VCF GT field ('0/0', '0|1', '1/1', './.', '.') into a
    0/1/2 focal-allele dosage, NaN for missing. Assumes biallelic (only
    '0' and '1' as valid allele indices) -- callers filter to biallelic
    sites before this is used."""
    if gt_str in (".", "./.", ".|."):
        return np.nan
    sep = "|" if "|" in gt_str else "/"
    alleles = gt_str.split(sep)
    if len(alleles) != 2 or any(a == "." for a in alleles):
        return np.nan
    try:
        return float(int(alleles[0]) + int(alleles[1]))
    except ValueError:
        return np.nan


def read_vcf(path: str, samples: list | None = None, extract_pl: bool = False,
             max_variants: int | None = None) -> VCFData:
    """Parse a VCF file into per-sample, per-locus genotype dosages (and,
    optionally, genotype likelihoods derived from the `PL` FORMAT field).

    Parameters
    ----------
    path : path to a `.vcf` or `.vcf.gz` file
    samples : optional list of sample names to restrict output to (in the
        given order). If not given, every sample in the VCF header is
        included, in the VCF's own column order.
    extract_pl : if True, also extract and convert the `PL` FORMAT field
        (Phred-scaled genotype likelihoods) for every site that has one.
        Sites without a `PL` field still contribute their `GT`-based
        dosage; their GL entries are left as `NaN` in the output arrays
        (not silently zero-filled, so a downstream check for "does this
        locus actually have GL data" is possible) and a warning-worthy
        count is tracked in `VCFData.n_loci_without_pl`.
    max_variants : optional cap on the number of biallelic SNPs to parse
        (a safety valve for a quick look at a large VCF, or for testing)
        -- variants are read in file order and parsing simply stops once
        the cap is reached, not a random sample.

    Returns
    -------
    VCFData
    """
    locus_names = []
    dosage_rows = []
    log_gl0_rows = [] if extract_pl else None
    log_gl1_rows = [] if extract_pl else None
    log_gl2_rows = [] if extract_pl else None
    n_skipped_multiallelic = 0
    n_skipped_indel = 0
    n_skipped_no_gt = 0
    n_loci_without_pl = 0

    header_samples = None
    sample_indices = None

    with _open_maybe_gzip(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                fields = line.split("\t")
                header_samples = fields[9:]
                if samples is not None:
                    missing = [s for s in samples if s not in header_samples]
                    if missing:
                        raise ValueError(
                            f"Requested samples {missing} not found in '{path}'. "
                            f"Samples present: {header_samples}"
                        )
                    sample_indices = [header_samples.index(s) for s in samples]
                else:
                    samples = header_samples
                    sample_indices = list(range(len(header_samples)))
                continue

            if header_samples is None:
                raise ValueError(f"'{path}' has no #CHROM header line -- not a valid VCF.")

            if max_variants is not None and len(locus_names) >= max_variants:
                break

            fields = line.split("\t")
            chrom, pos, variant_id, ref, alt = fields[0], fields[1], fields[2], fields[3], fields[4]

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
            pl_idx = format_keys.index("PL") if (extract_pl and "PL" in format_keys) else None

            locus_name = variant_id if variant_id != "." else f"{chrom}:{pos}"
            locus_names.append(locus_name)

            sample_fields = fields[9:]
            dosage_row = np.full(len(sample_indices), np.nan)
            if extract_pl:
                gl0_row = np.full(len(sample_indices), np.nan)
                gl1_row = np.full(len(sample_indices), np.nan)
                gl2_row = np.full(len(sample_indices), np.nan)

            any_pl_this_locus = False
            for out_i, sample_i in enumerate(sample_indices):
                sample_str = sample_fields[sample_i]
                sub_fields = sample_str.split(":")
                gt_str = sub_fields[gt_idx] if gt_idx < len(sub_fields) else "."
                dosage_row[out_i] = _parse_gt(gt_str)

                if extract_pl and pl_idx is not None and pl_idx < len(sub_fields):
                    pl_str = sub_fields[pl_idx]
                    if pl_str not in (".", ""):
                        try:
                            pl_vals = [float(x) for x in pl_str.split(",")]
                            if len(pl_vals) == 3:
                                log_gls = phred_to_log_likelihood(np.array(pl_vals))
                                gl0_row[out_i], gl1_row[out_i], gl2_row[out_i] = log_gls
                                any_pl_this_locus = True
                        except ValueError:
                            pass

            dosage_rows.append(dosage_row)
            if extract_pl:
                if not any_pl_this_locus:
                    n_loci_without_pl += 1
                log_gl0_rows.append(gl0_row)
                log_gl1_rows.append(gl1_row)
                log_gl2_rows.append(gl2_row)

    if not locus_names:
        raise ValueError(
            f"No biallelic SNPs found in '{path}' (after skipping "
            f"{n_skipped_multiallelic} multiallelic and {n_skipped_indel} indel sites). "
            f"Check the file is a valid VCF with biallelic SNP records."
        )

    return VCFData(
        locus_names=locus_names, sample_names=samples,
        dosage=np.array(dosage_rows),
        log_gl0=np.array(log_gl0_rows) if extract_pl else None,
        log_gl1=np.array(log_gl1_rows) if extract_pl else None,
        log_gl2=np.array(log_gl2_rows) if extract_pl else None,
        n_skipped_multiallelic=n_skipped_multiallelic,
        n_skipped_indel=n_skipped_indel,
        n_skipped_no_gt=n_skipped_no_gt,
        n_loci_without_pl=n_loci_without_pl,
    )


# ---------------------------------------------------------------------------
# VCF + metadata -> HyCline CSV writers
#
# All four functions below share the same basic shape: join VCFData (from
# read_vcf) against a metadata DataFrame by sample name, then write one of
# the wide-format CSVs the rest of this project's readers already expect
# -- using each reader's own DEFAULT column-suffix convention, so the
# output works with any downstream fitting command with zero extra flags.
# See BACKEND_PHASE1.md section 9.4 for why this was the highest-priority
# remaining gap, and the module docstring above for what VCF features are
# (and aren't) supported going in.
# ---------------------------------------------------------------------------

def _join_metadata(vcf_data: VCFData, metadata_df, sample_col: str,
                    covariate_cols: list) -> tuple:
    """Restrict to samples present in both the VCF and the metadata (in
    metadata row order, so the metadata's own ordering -- e.g. sorted by
    site -- is preserved in the output), and pull out the requested
    covariate columns. Returns (common_samples, sample_indices_into_vcf,
    metadata subset DataFrame)."""
    if sample_col not in metadata_df.columns:
        raise ValueError(f"--sample-col '{sample_col}' not found in the metadata file.")
    for col in covariate_cols:
        if col not in metadata_df.columns:
            raise ValueError(f"Column '{col}' not found in the metadata file.")

    vcf_sample_set = set(vcf_data.sample_names)
    meta_subset = metadata_df[metadata_df[sample_col].isin(vcf_sample_set)].reset_index(drop=True)
    if len(meta_subset) == 0:
        raise ValueError(
            "No sample names in common between the VCF and the metadata file -- check "
            "--sample-col matches the metadata column holding VCF sample names exactly."
        )

    common_samples = meta_subset[sample_col].tolist()
    sample_to_vcf_idx = {s: i for i, s in enumerate(vcf_data.sample_names)}
    vcf_indices = [sample_to_vcf_idx[s] for s in common_samples]

    n_dropped = len(vcf_data.sample_names) - len(set(common_samples) & vcf_sample_set)
    return common_samples, vcf_indices, meta_subset, n_dropped


def write_geographic_gl_csv(vcf_data: VCFData, metadata_df, sample_col: str,
                             distance_col: str, output_path: str) -> dict:
    """Write a per-individual genotype-likelihood CSV for geographic
    clines -- the format `read_gl_multi_locus_wide_csv` (Phase 4
    Component D's own reader) expects by default (`.gl0`/`.gl1`/`.gl2`
    suffixes). Requires `vcf_data` to have been read with `extract_pl=True`.

    Loci with no `PL` data at all (every individual missing it) are
    excluded from the output, not written as unusable all-missing
    columns -- `read_gl_multi_locus_wide_csv` would reject such a column
    outright anyway (it requires at least one complete row per locus), so
    excluding it here up front gives a clearer, earlier explanation of
    why that locus is absent, and lets the rest of the file convert
    successfully instead of failing entirely over one bad locus.
    """
    if vcf_data.log_gl0 is None:
        raise ValueError(
            "vcf_data has no genotype likelihoods -- re-read the VCF with extract_pl=True "
            "(--genotype-likelihoods on the CLI) to use this output format."
        )
    common_samples, vcf_idx, meta, n_dropped = _join_metadata(
        vcf_data, metadata_df, sample_col, [distance_col])

    base = {sample_col: common_samples, "distance": meta[distance_col].to_numpy()}
    new_columns = {}
    excluded_loci = []
    for j, locus in enumerate(vcf_data.locus_names):
        gl0, gl1, gl2 = vcf_data.log_gl0[j, vcf_idx], vcf_data.log_gl1[j, vcf_idx], vcf_data.log_gl2[j, vcf_idx]
        if np.all(np.isnan(gl0)):
            excluded_loci.append(locus)
            continue
        new_columns[f"{locus}.gl0"] = gl0
        new_columns[f"{locus}.gl1"] = gl1
        new_columns[f"{locus}.gl2"] = gl2

    # built as one concat, not looped column-by-column assignment, to
    # avoid pandas' DataFrame-fragmentation performance warning at scale
    # (found during a real "thousands of loci" stress test -- see
    # BACKEND_SCALE.md)
    out = pd.concat([pd.DataFrame(base), pd.DataFrame(new_columns)], axis=1)

    out.to_csv(output_path, index=False)
    return {"n_individuals": len(common_samples),
            "n_loci": len(vcf_data.locus_names) - len(excluded_loci),
            "n_samples_dropped": n_dropped, "excluded_loci_no_pl": excluded_loci}


def write_genomic_gl_csv(vcf_data: VCFData, metadata_df, sample_col: str,
                          hybrid_index_col: str, output_path: str) -> dict:
    """Write a per-individual genotype-likelihood CSV for genomic clines
    -- the format `read_genomic_gl_multi_locus_wide_csv` (Phase 4
    Component E's own reader) expects by default. See
    `write_geographic_gl_csv`'s docstring for why loci with no PL data at
    all are excluded rather than written as unusable columns."""
    if vcf_data.log_gl0 is None:
        raise ValueError(
            "vcf_data has no genotype likelihoods -- re-read the VCF with extract_pl=True "
            "(--genotype-likelihoods on the CLI) to use this output format."
        )
    common_samples, vcf_idx, meta, n_dropped = _join_metadata(
        vcf_data, metadata_df, sample_col, [hybrid_index_col])

    hybrid_index = meta[hybrid_index_col].to_numpy(dtype=float)
    if np.any((hybrid_index < 0) | (hybrid_index > 1)):
        raise ValueError(f"Column '{hybrid_index_col}' has values outside [0, 1].")

    base = {sample_col: common_samples, "hybrid_index": hybrid_index}
    new_columns = {}
    excluded_loci = []
    for j, locus in enumerate(vcf_data.locus_names):
        gl0, gl1, gl2 = vcf_data.log_gl0[j, vcf_idx], vcf_data.log_gl1[j, vcf_idx], vcf_data.log_gl2[j, vcf_idx]
        if np.all(np.isnan(gl0)):
            excluded_loci.append(locus)
            continue
        new_columns[f"{locus}.gl0"] = gl0
        new_columns[f"{locus}.gl1"] = gl1
        new_columns[f"{locus}.gl2"] = gl2

    out = pd.concat([pd.DataFrame(base), pd.DataFrame(new_columns)], axis=1)

    out.to_csv(output_path, index=False)
    return {"n_individuals": len(common_samples),
            "n_loci": len(vcf_data.locus_names) - len(excluded_loci),
            "n_samples_dropped": n_dropped, "excluded_loci_no_pl": excluded_loci}


def write_genomic_hard_count_csv(vcf_data: VCFData, metadata_df, sample_col: str,
                                  hybrid_index_col: str, output_path: str,
                                  ploidy: int = 2,
                                  parental_freqs: tuple | None = None,
                                  exclude_samples=None) -> dict:
    """Write a per-individual hard-count CSV for genomic clines -- the
    format `read_genomic_cline_multi_locus_wide_csv` expects by default
    (`.n`/`.freq` suffixes).

    `parental_freqs`, if given, is a pair of (n_loci,) arrays: each
    locus's ALT-allele frequency in parental populations 0 and 1, written
    as constant '{locus}.p0'/'{locus}.p1' columns; loci where either is NaN
    are left out. `exclude_samples` names individuals to leave out (e.g.
    the parental reference individuals)."""
    common_samples, vcf_idx, meta, n_dropped = _join_metadata(
        vcf_data, metadata_df, sample_col, [hybrid_index_col])
    common_samples, vcf_idx, meta = _drop_samples(common_samples, vcf_idx, meta,
                                                  exclude_samples)

    hybrid_index = meta[hybrid_index_col].to_numpy(dtype=float)
    if np.any((hybrid_index < 0) | (hybrid_index > 1)):
        raise ValueError(f"Column '{hybrid_index_col}' has values outside [0, 1].")

    base = {sample_col: common_samples, "hybrid_index": hybrid_index}
    new_columns = {}
    n_written = 0
    for j, locus in enumerate(vcf_data.locus_names):
        if parental_freqs is not None and not (np.isfinite(parental_freqs[0][j])
                                               and np.isfinite(parental_freqs[1][j])):
            continue
        dosage = vcf_data.dosage[j, vcf_idx]
        valid = ~np.isnan(dosage)
        new_columns[f"{locus}.n"] = np.where(valid, ploidy, 0)
        new_columns[f"{locus}.freq"] = np.where(valid, dosage / ploidy, np.nan)
        if parental_freqs is not None:
            new_columns[f"{locus}.p0"] = np.full(len(common_samples), parental_freqs[0][j])
            new_columns[f"{locus}.p1"] = np.full(len(common_samples), parental_freqs[1][j])
        n_written += 1

    out = pd.concat([pd.DataFrame(base), pd.DataFrame(new_columns)], axis=1)

    out.to_csv(output_path, index=False)
    return {"n_individuals": len(common_samples), "n_loci": n_written,
            "n_samples_dropped": n_dropped}


def _drop_samples(common_samples: list, indices, meta, exclude_samples) -> tuple:
    """Remove `exclude_samples` from a joined (samples, indices, metadata)
    triple, keeping the three aligned."""
    if not exclude_samples:
        return common_samples, indices, meta
    exclude = set(exclude_samples)
    keep = [s not in exclude for s in common_samples]
    common_samples = [s for s, k in zip(common_samples, keep) if k]
    indices = [i for i, k in zip(indices, keep) if k]
    meta = meta[np.array(keep)].reset_index(drop=True)
    if not common_samples:
        raise ValueError("No individuals left after excluding the parental reference "
                         "individuals -- the metadata has hybrid index only for them.")
    return common_samples, indices, meta


def write_geographic_hard_count_csv(vcf_data: VCFData, metadata_df, sample_col: str,
                                     site_col: str, distance_col: str, output_path: str,
                                     ploidy: int = 2) -> dict:
    """Write a site-aggregated hard-count CSV for geographic clines --
    the format `read_multi_locus_wide_csv` expects by default
    (`.nSamples`/`.A` suffixes, matching HZAR's own bundled dataset
    convention exactly -- see BACKEND_PHASE1.md section 9.3). Unlike the
    other three writers, this one genuinely aggregates: individuals
    sharing a site are pooled into one row of allele counts, since that's
    the shape HZAR-style geographic-cline data has always taken in this
    project. Every individual at a given site must have the *same*
    distance value in the metadata -- raises a clear error naming the
    site otherwise, rather than silently picking one.
    """
    common_samples, vcf_idx, meta, n_dropped = _join_metadata(
        vcf_data, metadata_df, sample_col, [site_col, distance_col])

    sites = meta[site_col].tolist()
    distances = meta[distance_col].tolist()

    site_distance = {}
    site_order = []
    site_sample_positions = {}  # site -> list of positions into common_samples/vcf_idx
    for pos, (site, dist) in enumerate(zip(sites, distances)):
        if site not in site_distance:
            site_distance[site] = dist
            site_order.append(site)
            site_sample_positions[site] = []
        elif site_distance[site] != dist:
            raise ValueError(
                f"Site '{site}' has inconsistent {distance_col} values across its "
                f"individuals in the metadata ({site_distance[site]} vs {dist}) -- every "
                f"individual at the same site must share the same distance value."
            )
        site_sample_positions[site].append(pos)

    out_rows = {site_col: site_order, "distance": [site_distance[s] for s in site_order]}
    vcf_idx_arr = np.asarray(vcf_idx)

    for j, locus in enumerate(vcf_data.locus_names):
        n_alleles_col = []
        freq_col = []
        for site in site_order:
            positions = site_sample_positions[site]
            idxs = vcf_idx_arr[positions]
            dosage = vcf_data.dosage[j, idxs]
            valid = ~np.isnan(dosage)
            n_alleles = int(valid.sum()) * ploidy
            n_focal = float(np.nansum(dosage))
            n_alleles_col.append(n_alleles)
            freq_col.append(n_focal / n_alleles if n_alleles > 0 else np.nan)
        out_rows[f"{locus}.nSamples"] = n_alleles_col
        out_rows[f"{locus}.A"] = freq_col

    out = pd.DataFrame(out_rows)
    out.to_csv(output_path, index=False)
    return {"n_individuals": len(common_samples), "n_sites": len(site_order),
            "n_loci": len(vcf_data.locus_names), "n_samples_dropped": n_dropped}
