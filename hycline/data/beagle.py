"""Reader/writer for ANGSD's beagle-format genotype-likelihood output
(`angsd -doGlf 2`), the standard input format for downstream tools like
NGSadmix/PCAngsd -- and, via `hycline.data.filter_aims`, a natural raw
input for ancestry-informative-marker filtering ahead of HyCline's own
converters.

Format (tab-separated, optionally gzip-compressed): a header row
`marker allele1 allele2 <ind> <ind> <ind> ...` (three columns per
individual, ANGSD's own convention of repeating that individual's label
once per column), then one row per site: marker ID, the two allele
codes, then three genotype-likelihood/posterior values per individual
(major/major, major/minor, minor/minor, in that order).

This project's own dosage/focal-allele convention (0/1/2 = copies of the
*focal* allele) treats `allele2` (ANGSD's "minor" slot) as focal, the
same way this project's VCF parser treats the ALT allele as focal --
consistent with every other reader in this project, not a beagle-specific
choice.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hycline.data.schema import values_consistent


@dataclass
class BeagleData:
    marker_names: list
    sample_names: list
    allele1: list  # major allele code per marker, as given in the file (not interpreted)
    allele2: list  # minor/focal allele code per marker
    gl0: np.ndarray  # (n_markers, n_samples) -- P(major/major), as given in the file
    gl1: np.ndarray  # (n_markers, n_samples) -- P(major/minor)
    gl2: np.ndarray  # (n_markers, n_samples) -- P(minor/minor)

    def __len__(self) -> int:
        return len(self.marker_names)


def _open_maybe_gzip(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def read_beagle(path: str) -> BeagleData:
    """Parse an ANGSD beagle-format file (`.beagle` or `.beagle.gz`).

    Raises `ValueError` with an actionable message if the file doesn't
    have the expected `marker/allele1/allele2` header or a column count
    that's a clean multiple of 3 per individual.
    """
    with _open_maybe_gzip(path) as f:
        header_line = f.readline().rstrip("\n")
        if not header_line:
            raise ValueError(f"'{path}' is empty or has no header row.")
        header = header_line.split("\t")
        if len(header) < 3 or header[0].lower() != "marker":
            raise ValueError(
                f"'{path}' doesn't look like an ANGSD beagle file -- expected a header "
                f"starting with 'marker\\tallele1\\tallele2\\t...', got: "
                f"{header[:3]}{'...' if len(header) > 3 else ''}"
            )
        sample_cols = header[3:]
        if len(sample_cols) == 0 or len(sample_cols) % 3 != 0:
            raise ValueError(
                f"'{path}' has {len(sample_cols)} genotype-likelihood columns after "
                f"marker/allele1/allele2 -- expected a multiple of 3 (one triple per "
                f"individual). The file may be truncated or not a real beagle file."
            )
        n_ind = len(sample_cols) // 3
        sample_names = [sample_cols[3 * i] for i in range(n_ind)]

        marker_names, allele1, allele2 = [], [], []
        gl0_rows, gl1_rows, gl2_rows = [], [], []
        for line_no, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 3 + 3 * n_ind:
                raise ValueError(
                    f"'{path}' line {line_no}: expected {3 + 3 * n_ind} columns "
                    f"(matching the header), found {len(fields)}."
                )
            marker_names.append(fields[0])
            allele1.append(fields[1])
            allele2.append(fields[2])
            try:
                vals = np.array(fields[3:], dtype=float)
            except ValueError:
                raise ValueError(
                    f"'{path}' line {line_no}: genotype-likelihood columns must be "
                    f"numeric."
                )
            gl0_rows.append(vals[0::3])
            gl1_rows.append(vals[1::3])
            gl2_rows.append(vals[2::3])

    if not marker_names:
        raise ValueError(f"'{path}' has a header but no data rows.")

    return BeagleData(
        marker_names=marker_names, sample_names=sample_names,
        allele1=allele1, allele2=allele2,
        gl0=np.array(gl0_rows), gl1=np.array(gl1_rows), gl2=np.array(gl2_rows),
    )


def write_beagle(beagle_data: BeagleData, output_path: str, marker_mask: np.ndarray) -> None:
    """Write a beagle-format file containing only the markers where
    `marker_mask` is True, preserving the original column structure and
    per-individual GL values exactly (a row filter, not a re-derivation)
    -- so a downstream tool reading the output sees the same numbers
    ANGSD originally produced for every marker that survives filtering."""
    open_fn = gzip.open if output_path.endswith(".gz") else open
    header = ["marker", "allele1", "allele2"]
    for s in beagle_data.sample_names:
        header += [s, s, s]

    with open_fn(output_path, "wt") as f:
        f.write("\t".join(header) + "\n")
        keep_idx = np.where(marker_mask)[0]
        for i in keep_idx:
            row = [beagle_data.marker_names[i], beagle_data.allele1[i], beagle_data.allele2[i]]
            for j in range(len(beagle_data.sample_names)):
                row += [f"{beagle_data.gl0[i, j]:g}", f"{beagle_data.gl1[i, j]:g}",
                        f"{beagle_data.gl2[i, j]:g}"]
            f.write("\t".join(row) + "\n")


# ---------------------------------------------------------------------------
# Beagle + metadata -> HyCline CSV writers
#
# Mirrors hycline.data.vcf's four VCF writer functions closely (same
# join-then-write shape, same output column conventions), for beagle
# input instead of a VCF. See BACKEND_PHASE1.md section 9.7 for why this
# didn't exist initially (the beagle reader/filter came first) and why
# it was a comparatively small addition once that reader existed.
# ---------------------------------------------------------------------------

def _join_beagle_metadata(beagle_data: BeagleData, metadata_df, sample_col: str,
                           covariate_cols: list) -> tuple:
    """Same shape as `hycline.data.vcf._join_metadata`, for `BeagleData`
    instead of `VCFData`. Returns (common_samples, sample_indices,
    metadata subset DataFrame, n_dropped)."""
    if sample_col not in metadata_df.columns:
        raise ValueError(f"--sample-col '{sample_col}' not found in the metadata file.")
    for col in covariate_cols:
        if col not in metadata_df.columns:
            raise ValueError(f"Column '{col}' not found in the metadata file.")

    beagle_sample_set = set(beagle_data.sample_names)
    meta_subset = metadata_df[metadata_df[sample_col].isin(beagle_sample_set)].reset_index(drop=True)
    if len(meta_subset) == 0:
        raise ValueError(
            "No sample names in common between the beagle file and the metadata file -- "
            "check --sample-col matches the metadata column holding the beagle file's own "
            "sample labels exactly (as ANGSD wrote them into the header row)."
        )

    common_samples = meta_subset[sample_col].tolist()
    sample_to_idx = {s: i for i, s in enumerate(beagle_data.sample_names)}
    indices = [sample_to_idx[s] for s in common_samples]

    n_dropped = len(beagle_data.sample_names) - len(set(common_samples))
    return common_samples, indices, meta_subset, n_dropped


def _beagle_log_gl(gl0: np.ndarray, gl1: np.ndarray, gl2: np.ndarray) -> tuple:
    """Convert beagle's plain-probability genotype likelihoods to
    natural-log space (this project's internal convention throughout --
    see BACKEND_PHASE5.md section 2). A large-but-finite floor (-1e10,
    the same convention `hycline.data.genotype_likelihood.
    hard_genotype_to_log_likelihood` uses) is used in place of literal
    `-inf` for a true zero probability, to keep CSV round-tripping
    well-behaved -- numerically indistinguishable from `-inf` for any
    likelihood computation in double precision."""
    floor = -1e10
    with np.errstate(divide="ignore"):
        log_gl0 = np.where(gl0 > 0, np.log(gl0), floor)
        log_gl1 = np.where(gl1 > 0, np.log(gl1), floor)
        log_gl2 = np.where(gl2 > 0, np.log(gl2), floor)
    return log_gl0, log_gl1, log_gl2


def _beagle_most_likely_dosage(gl0: np.ndarray, gl1: np.ndarray, gl2: np.ndarray) -> np.ndarray:
    """Argmax-based point-estimate dosage (0/1/2) per individual, for
    beagle-to-hard-count conversion. A real simplification, stated
    plainly: this discards the genotype uncertainty beagle data was
    built to carry, the same trade-off `hycline.data.vcf`'s hard-count
    writers make implicitly whenever GT (not PL) is the only information
    available. Superseded by `_beagle_expected_dosage` for geographic
    hard-count output (see that function's docstring) -- kept here
    because `write_beagle_genomic_hard_count_csv` still uses it, and
    because argmax dosage may still be a reasonable, explicit choice for
    a caller who wants it directly.

    Individuals with a missing/uninformative genotype likelihood --
    ANGSD's own convention for this is a flat/tied triple, e.g.
    (0.333333, 0.333333, 0.333333) rather than NaN -- get NaN dosage
    here instead of an arbitrary argmax pick. Without this, `np.argmax`
    on an exact or near-exact tie always resolves to index 0 (dosage 0),
    silently reporting a missing individual as a confident, fully
    -genotyped reference homozygote."""
    stacked = np.stack([gl0, gl1, gl2], axis=0)
    dosage = np.argmax(stacked, axis=0).astype(float)
    spread = stacked.max(axis=0) - stacked.min(axis=0)
    missing = spread < 1e-4
    dosage[missing] = np.nan
    return dosage


def _beagle_expected_dosage(gl0: np.ndarray, gl1: np.ndarray, gl2: np.ndarray) -> np.ndarray:
    """Expected dosage E[dosage] = 0*p0 + 1*p1 + 2*p2 per individual,
    treating the three genotype likelihoods as a flat-prior posterior
    (normalized per individual to guard against beagle values that
    aren't already normalized to sum to 1) -- the same standard,
    defensible estimator `hycline.data.filter_aims._expected_freq_from_gl`
    already uses for AIM filtering, applied here to site-aggregated
    allele-frequency output instead.

    Real, not just theoretical, difference from
    `_beagle_most_likely_dosage`: for an individual with GLs
    (0.49, 0.02, 0.49) -- genuinely ambiguous between homozygous
    reference and homozygous alternate, with almost no support for the
    heterozygote -- argmax arbitrarily picks whichever allele comes
    first (dosage 0, discarding that the two homozygous classes are
    almost equally likely), while the expected dosage correctly comes
    out to 1.0 (reflecting the genuine 50/50 uncertainty). Verified
    directly: `tests/test_beagle.py::
    test_write_beagle_geographic_hard_count_csv_uses_expected_dosage_not_argmax`.
    """
    total = gl0 + gl1 + gl2
    with np.errstate(invalid="ignore", divide="ignore"):
        return (gl1 + 2.0 * gl2) / total


def write_beagle_geographic_gl_csv(beagle_data: BeagleData, metadata_df, sample_col: str,
                                    distance_col: str, output_path: str) -> dict:
    """Write a per-individual genotype-likelihood CSV for geographic
    clines from beagle data -- the format `read_gl_multi_locus_wide_csv`
    (Phase 4 Component D's own reader) expects by default."""
    common_samples, idx, meta, n_dropped = _join_beagle_metadata(
        beagle_data, metadata_df, sample_col, [distance_col])

    base = {sample_col: common_samples, "distance": meta[distance_col].to_numpy()}
    new_columns = {}
    for j, marker in enumerate(beagle_data.marker_names):
        log_gl0, log_gl1, log_gl2 = _beagle_log_gl(
            beagle_data.gl0[j, idx], beagle_data.gl1[j, idx], beagle_data.gl2[j, idx])
        new_columns[f"{marker}.gl0"] = log_gl0
        new_columns[f"{marker}.gl1"] = log_gl1
        new_columns[f"{marker}.gl2"] = log_gl2

    out = pd.concat([pd.DataFrame(base), pd.DataFrame(new_columns)], axis=1)

    out.to_csv(output_path, index=False)
    return {"n_individuals": len(common_samples), "n_loci": len(beagle_data),
            "n_samples_dropped": n_dropped}


def write_beagle_genomic_gl_csv(beagle_data: BeagleData, metadata_df, sample_col: str,
                                 hybrid_index_col: str, output_path: str) -> dict:
    """Write a per-individual genotype-likelihood CSV for genomic clines
    from beagle data -- the format `read_genomic_gl_multi_locus_wide_csv`
    (Phase 4 Component E's own reader) expects by default."""
    common_samples, idx, meta, n_dropped = _join_beagle_metadata(
        beagle_data, metadata_df, sample_col, [hybrid_index_col])

    hybrid_index = meta[hybrid_index_col].to_numpy(dtype=float)
    if np.any((hybrid_index < 0) | (hybrid_index > 1)):
        raise ValueError(f"Column '{hybrid_index_col}' has values outside [0, 1].")

    base = {sample_col: common_samples, "hybrid_index": hybrid_index}
    new_columns = {}
    for j, marker in enumerate(beagle_data.marker_names):
        log_gl0, log_gl1, log_gl2 = _beagle_log_gl(
            beagle_data.gl0[j, idx], beagle_data.gl1[j, idx], beagle_data.gl2[j, idx])
        new_columns[f"{marker}.gl0"] = log_gl0
        new_columns[f"{marker}.gl1"] = log_gl1
        new_columns[f"{marker}.gl2"] = log_gl2

    out = pd.concat([pd.DataFrame(base), pd.DataFrame(new_columns)], axis=1)

    out.to_csv(output_path, index=False)
    return {"n_individuals": len(common_samples), "n_loci": len(beagle_data),
            "n_samples_dropped": n_dropped}


def write_beagle_geographic_hard_count_csv(beagle_data: BeagleData, metadata_df, sample_col: str,
                                            site_col: str, distance_col: str, output_path: str,
                                            ploidy: int = 2) -> dict:
    """Write a site-aggregated allele-frequency CSV for geographic
    clines from beagle data -- the format `read_multi_locus_wide_csv`
    expects by default (`.nSamples`/`.A` suffixes). Uses each
    individual's *expected* dosage (posterior mean under a flat prior
    over the three genotype likelihoods, `_beagle_expected_dosage`), not
    an argmax point estimate -- the resulting `.A` frequency genuinely
    incorporates genotype uncertainty from the GLs, rather than
    discarding it the way collapsing to a single most-likely genotype
    per individual would. Every individual at a site must share the
    same distance value, the same requirement
    `hycline.data.vcf.write_geographic_hard_count_csv` has."""
    common_samples, idx, meta, n_dropped = _join_beagle_metadata(
        beagle_data, metadata_df, sample_col, [site_col, distance_col])

    sites = meta[site_col].tolist()
    distances = meta[distance_col].tolist()

    site_distance, site_order, site_positions = {}, [], {}
    for pos, (site, dist) in enumerate(zip(sites, distances)):
        if site not in site_distance:
            site_distance[site] = dist
            site_order.append(site)
            site_positions[site] = []
        elif not values_consistent(site_distance[site], dist):
            raise ValueError(
                f"Site '{site}' has inconsistent {distance_col} values across its "
                f"individuals in the metadata ({site_distance[site]} vs {dist})."
            )
        site_positions[site].append(pos)

    out_rows = {site_col: site_order, "distance": [site_distance[s] for s in site_order]}
    idx_arr = np.asarray(idx)

    for j, marker in enumerate(beagle_data.marker_names):
        dosage = _beagle_expected_dosage(
            beagle_data.gl0[j, idx_arr], beagle_data.gl1[j, idx_arr], beagle_data.gl2[j, idx_arr])
        n_alleles_col, freq_col = [], []
        for site in site_order:
            positions = site_positions[site]
            d = dosage[positions]
            valid = ~np.isnan(d)
            n_alleles = int(valid.sum()) * ploidy
            n_focal = float(np.nansum(d))
            n_alleles_col.append(n_alleles)
            freq_col.append(n_focal / n_alleles if n_alleles > 0 else np.nan)
        out_rows[f"{marker}.nSamples"] = n_alleles_col
        out_rows[f"{marker}.A"] = freq_col

    out = pd.DataFrame(out_rows)
    out.to_csv(output_path, index=False)
    return {"n_individuals": len(common_samples), "n_sites": len(site_order),
            "n_loci": len(beagle_data), "n_samples_dropped": n_dropped}


def write_beagle_genomic_hard_count_csv(beagle_data: BeagleData, metadata_df, sample_col: str,
                                         hybrid_index_col: str, output_path: str,
                                         ploidy: int = 2) -> dict:
    """Write a per-individual hard-count CSV for genomic clines from
    beagle data -- the format `read_genomic_cline_multi_locus_wide_csv`
    expects by default (`.n`/`.freq` suffixes). Uses each individual's
    most-likely genotype; see `write_beagle_geographic_hard_count_csv`'s
    docstring for the same caveat."""
    common_samples, idx, meta, n_dropped = _join_beagle_metadata(
        beagle_data, metadata_df, sample_col, [hybrid_index_col])

    hybrid_index = meta[hybrid_index_col].to_numpy(dtype=float)
    if np.any((hybrid_index < 0) | (hybrid_index > 1)):
        raise ValueError(f"Column '{hybrid_index_col}' has values outside [0, 1].")

    base = {sample_col: common_samples, "hybrid_index": hybrid_index}
    new_columns = {}
    idx_arr = np.asarray(idx)
    for j, marker in enumerate(beagle_data.marker_names):
        dosage = _beagle_most_likely_dosage(
            beagle_data.gl0[j, idx_arr], beagle_data.gl1[j, idx_arr], beagle_data.gl2[j, idx_arr])
        valid = ~np.isnan(dosage)
        new_columns[f"{marker}.n"] = np.where(valid, ploidy, 0)
        new_columns[f"{marker}.freq"] = np.where(valid, dosage / ploidy, np.nan)

    out = pd.concat([pd.DataFrame(base), pd.DataFrame(new_columns)], axis=1)

    out.to_csv(output_path, index=False)
    return {"n_individuals": len(common_samples), "n_loci": len(beagle_data),
            "n_samples_dropped": n_dropped}
