"""Reading transect data from plain CSV files.

Designed for the CLI (`hycline/cli.py`), where the person running it may
not be comfortable debugging a Python traceback -- error messages here are
written to say exactly what's wrong and what columns were actually found.

Three independent axes of flexibility, each handled by a small resolver
function shared between the single-locus and multi-locus readers:

- geographic distance: a pre-computed `distance` column, OR lat/lon
  coordinates auto-collapsed to 1D (see `hycline.data.geo`)
- sample size: total sampled alleles directly, OR individual count + ploidy
- focal-allele data: raw counts, OR observed frequency
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hycline.data.schema import (
    AlleleFreqTransect, HybridIndexLocusData,
    GenotypeLikelihoodTransect, GenotypeLikelihoodLocusData, TraitTransect,
)


class TransectCSVError(ValueError):
    """Raised for any problem reading a transect CSV, with a message meant
    to be read directly by the person running the CLI, not just a developer."""


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise TransectCSVError(f"Input file not found: '{path}'. Check the --input path.")
    except Exception as e:
        raise TransectCSVError(f"Could not read '{path}' as a CSV file: {e}")
    if df.shape[0] == 0:
        raise TransectCSVError(f"'{path}' has a header but no data rows.")
    return df


def _numeric_col(df: pd.DataFrame, col_name: str, path: str,
                  allow_missing: bool = False) -> np.ndarray:
    """Coerce a column to float, raising an actionable error on bad values.
    If allow_missing, missing values are returned as NaN rather than
    raising -- used by the multi-locus reader, where different loci can
    legitimately have data missing at different sites."""
    if col_name not in df.columns:
        available = ", ".join(f"'{c}'" for c in df.columns)
        raise TransectCSVError(
            f"Column '{col_name}' not found in '{path}'. "
            f"Columns present in the file: {available}."
        )
    col = pd.to_numeric(df[col_name], errors="coerce")
    bad_rows = df.index[col.isna() & df[col_name].notna()]
    if len(bad_rows) > 0:
        examples = df.loc[bad_rows[:5], col_name].tolist()
        raise TransectCSVError(
            f"Column '{col_name}' has non-numeric values, e.g. {examples}. "
            f"Every row must be a plain number in this column."
        )
    if not allow_missing and col.isna().any():
        n_missing = int(col.isna().sum())
        raise TransectCSVError(
            f"Column '{col_name}' has {n_missing} missing/blank value(s). "
            f"Remove those rows or fill them in before running HyCline "
            f"(HZAR itself drops rows with missing data for a locus -- "
            f"do the same here rather than guessing a value)."
        )
    return col.to_numpy(dtype=float)


def _resolve_distance(df: pd.DataFrame, path: str,
                       distance_col: str | None = "distance",
                       lat_col: str | None = None, lon_col: str | None = None,
                       transect_method: str = "reference",
                       reference_lat: float | None = None,
                       reference_lon: float | None = None) -> np.ndarray:
    """Resolve geographic distance from either a direct column or lat/lon
    coordinates. If both are given, coordinates take priority (documented
    in the CLI's --help text) -- a direct --distance-col is assumed to be
    the fallback/default, not something the user deliberately chose over
    coordinates they also happened to specify.
    """
    if lat_col is not None and lon_col is not None:
        from hycline.data.geo import distance_from_reference, project_onto_transect_axis

        lat = _numeric_col(df, lat_col, path)
        lon = _numeric_col(df, lon_col, path)
        if transect_method == "reference":
            return distance_from_reference(lat, lon, reference_lat, reference_lon)
        elif transect_method == "pca":
            return project_onto_transect_axis(lat, lon)
        else:
            raise TransectCSVError(
                f"Unknown --transect-method '{transect_method}'. Use 'reference' or 'pca'."
            )
    elif lat_col is not None or lon_col is not None:
        raise TransectCSVError(
            "Specify both --lat-col and --lon-col together, or neither."
        )
    elif distance_col is not None:
        return _numeric_col(df, distance_col, path)
    else:
        raise TransectCSVError(
            "No distance source specified: give either --distance-col, or "
            "both --lat-col and --lon-col."
        )


def _resolve_n_alleles(df: pd.DataFrame, path: str,
                        n_alleles_col: str | None = "n_alleles",
                        n_ind_col: str | None = None, ploidy: int = 2,
                        allow_missing: bool = False) -> np.ndarray:
    """Resolve total sampled alleles from either a direct count column or
    individual count + ploidy."""
    if n_alleles_col is not None and n_ind_col is not None:
        raise TransectCSVError(
            "Specify either --n-alleles-col or --n-ind-col, not both."
        )
    if n_ind_col is not None:
        if ploidy < 1:
            raise TransectCSVError(f"--ploidy must be a positive integer, got {ploidy}.")
        n_ind = _numeric_col(df, n_ind_col, path, allow_missing=allow_missing)
        return n_ind * ploidy
    elif n_alleles_col is not None:
        return _numeric_col(df, n_alleles_col, path, allow_missing=allow_missing)
    else:
        raise TransectCSVError(
            "No sample-size source specified: give either --n-alleles-col "
            "(total sampled alleles) or --n-ind-col (individual count; "
            "combined with --ploidy, default 2 for diploid autosomal markers)."
        )


def _resolve_n_focal(df: pd.DataFrame, path: str, n_alleles: np.ndarray,
                      n_focal_col: str | None = None, freq_col: str | None = None,
                      allow_missing: bool = False) -> np.ndarray:
    if n_focal_col is not None and freq_col is not None:
        raise TransectCSVError("Specify either --n-focal-col or --freq-col, not both.")
    if n_focal_col is None and freq_col is None:
        raise TransectCSVError(
            "You must specify either --n-focal-col (a column of focal-allele "
            "counts) or --freq-col (a column of observed frequencies, 0-1)."
        )
    if n_focal_col is not None:
        return _numeric_col(df, n_focal_col, path, allow_missing=allow_missing)
    freq = _numeric_col(df, freq_col, path, allow_missing=allow_missing)
    out_of_range = (freq < 0) | (freq > 1)
    out_of_range &= ~np.isnan(freq)
    if np.any(out_of_range):
        raise TransectCSVError(
            f"Column '{freq_col}' has values outside [0, 1] -- it should be "
            f"an observed frequency, not a count. If you have counts, use "
            f"--n-focal-col instead of --freq-col."
        )
    return np.round(freq * n_alleles)


# ---------------------------------------------------------------------------
# Single-locus reader
# ---------------------------------------------------------------------------

def read_allele_freq_csv(path: str,
                          distance_col: str | None = "distance",
                          n_alleles_col: str | None = "n_alleles",
                          n_focal_col: str | None = "n_focal",
                          freq_col: str | None = None,
                          n_ind_col: str | None = None, ploidy: int = 2,
                          lat_col: str | None = None, lon_col: str | None = None,
                          transect_method: str = "reference",
                          reference_lat: float | None = None,
                          reference_lon: float | None = None,
                          site_col: str | None = None) -> AlleleFreqTransect:
    """Read allele-frequency transect data from a CSV file (one row per
    sampling site).

    Distance: either `distance_col` (a pre-computed column) or both
    `lat_col`/`lon_col` (coordinates, auto-collapsed to 1D -- see
    `hycline.data.geo` and `transect_method`). Coordinates take priority if
    both are given.

    Sample size: either `n_alleles_col` (total sampled alleles) or
    `n_ind_col` + `ploidy` (individual count; alleles = individuals x ploidy).

    Focal-allele data: either `n_focal_col` (counts) or `freq_col`
    (observed frequency 0-1; counts are recovered as round(freq * n_alleles)).

    Raises
    ------
    TransectCSVError : with a message describing exactly what went wrong,
        meant to be actionable without reading a Python traceback.
    """
    df = _read_csv(path)

    distance = _resolve_distance(df, path, distance_col, lat_col, lon_col,
                                  transect_method, reference_lat, reference_lon)
    n_alleles = _resolve_n_alleles(df, path, n_alleles_col, n_ind_col, ploidy)
    n_focal = _resolve_n_focal(df, path, n_alleles, n_focal_col, freq_col)

    site_names = None
    if site_col is not None:
        if site_col not in df.columns:
            raise TransectCSVError(f"--site-col '{site_col}' not found in '{path}'.")
        site_names = df[site_col].astype(str).tolist()

    try:
        return AlleleFreqTransect(distance=distance, n_alleles=n_alleles,
                                   n_focal=n_focal, site_names=site_names)
    except ValueError as e:
        raise TransectCSVError(f"Problem with the data in '{path}': {e}")


# ---------------------------------------------------------------------------
# Multi-locus wide-format reader
# ---------------------------------------------------------------------------

def detect_loci(df: pd.DataFrame, n_alleles_suffix: str = ".nSamples",
                 freq_suffix: str | None = ".A",
                 n_focal_suffix: str | None = None) -> list[str]:
    """Auto-detect locus names from wide-format column naming, e.g. for
    columns 'ada.nSamples' and 'ada.A', the detected locus is 'ada'.
    Returns locus names in the order their n_alleles column appears."""
    if freq_suffix is not None and n_focal_suffix is not None:
        raise TransectCSVError(
            "Specify either freq_suffix or n_focal_suffix, not both."
        )
    focal_suffix = freq_suffix if freq_suffix is not None else n_focal_suffix
    if focal_suffix is None:
        raise TransectCSVError("Must specify either freq_suffix or n_focal_suffix.")

    loci = []
    for col in df.columns:
        if col.endswith(n_alleles_suffix):
            locus = col[: -len(n_alleles_suffix)]
            if f"{locus}{focal_suffix}" in df.columns:
                loci.append(locus)
    return loci


def read_multi_locus_wide_csv(path: str,
                               distance_col: str | None = "distance",
                               n_alleles_suffix: str = ".nSamples",
                               freq_suffix: str | None = ".A",
                               n_focal_suffix: str | None = None,
                               n_ind_suffix: str | None = None, ploidy: int = 2,
                               lat_col: str | None = None, lon_col: str | None = None,
                               transect_method: str = "reference",
                               reference_lat: float | None = None,
                               reference_lon: float | None = None,
                               site_col: str | None = None,
                               loci: list[str] | None = None,
                               ) -> dict[str, AlleleFreqTransect]:
    """Read a wide-format multi-locus CSV: one row per sampling site, and
    for each locus L, a pair of columns '{L}{n_alleles_suffix}' (or
    '{L}{n_ind_suffix}') and '{L}{freq_suffix}' (or '{L}{n_focal_suffix}').

    Default suffixes ('.nSamples', '.A') match HZAR's own bundled example
    datasets' naming convention exactly (e.g. `manakinMolecular.csv`) -- an
    HZAR wide-format CSV should be readable with this function's defaults,
    unmodified.

    Rows with missing/blank data for a *particular* locus are dropped from
    that locus's transect only, not from the whole file -- different loci
    commonly have data missing at different sites (e.g. a marker that
    failed to amplify at one locality), and requiring every locus to have
    complete data at every site would throw away usable data unnecessarily.

    Parameters
    ----------
    loci : restrict to this subset of locus names (as auto-detected from
        column names) rather than fitting every detected locus.

    Returns
    -------
    dict mapping locus name -> AlleleFreqTransect, in the order loci were
    detected in the file (or the order given in `loci`, if provided).
    """
    df = _read_csv(path)

    distance = _resolve_distance(df, path, distance_col, lat_col, lon_col,
                                  transect_method, reference_lat, reference_lon)

    detected = detect_loci(df, (n_ind_suffix if n_ind_suffix is not None else n_alleles_suffix),
                            freq_suffix, n_focal_suffix)
    if not detected:
        focal_suffix = freq_suffix if freq_suffix is not None else n_focal_suffix
        effective_n_suffix = n_ind_suffix if n_ind_suffix is not None else n_alleles_suffix
        raise TransectCSVError(
            f"No loci detected in '{path}' using "
            f"{'n_ind_suffix' if n_ind_suffix is not None else 'n_alleles_suffix'}="
            f"'{effective_n_suffix}' and focal-allele suffix '{focal_suffix}'. "
            f"Expected paired columns like '<locus>{effective_n_suffix}' and "
            f"'<locus>{focal_suffix}'. Columns present in the file: "
            f"{', '.join(df.columns)}."
        )

    if loci is not None:
        unknown = [l for l in loci if l not in detected]
        if unknown:
            raise TransectCSVError(
                f"Requested loci {unknown} not found. Loci detected in "
                f"'{path}': {detected}."
            )
        selected = loci
    else:
        selected = detected

    site_names_full = None
    if site_col is not None:
        if site_col not in df.columns:
            raise TransectCSVError(f"--site-col '{site_col}' not found in '{path}'.")
        site_names_full = df[site_col].astype(str).tolist()

    transects: dict[str, AlleleFreqTransect] = {}
    for locus in selected:
        n_alleles_col = (f"{locus}{n_ind_suffix}" if n_ind_suffix is not None
                          else f"{locus}{n_alleles_suffix}")
        n_alleles = _resolve_n_alleles(
            df, path,
            n_alleles_col=(None if n_ind_suffix is not None else n_alleles_col),
            n_ind_col=(n_alleles_col if n_ind_suffix is not None else None),
            ploidy=ploidy, allow_missing=True,
        )
        n_focal = _resolve_n_focal(
            df, path, n_alleles,
            n_focal_col=(f"{locus}{n_focal_suffix}" if n_focal_suffix is not None else None),
            freq_col=(f"{locus}{freq_suffix}" if freq_suffix is not None else None),
            allow_missing=True,
        )

        mask = ~(np.isnan(n_alleles) | np.isnan(n_focal) | np.isnan(distance))
        if not np.any(mask):
            raise TransectCSVError(
                f"Locus '{locus}' has no complete (non-missing) rows in '{path}'."
            )

        site_names = ([s for s, m in zip(site_names_full, mask) if m]
                       if site_names_full is not None else None)

        try:
            transects[locus] = AlleleFreqTransect(
                distance=distance[mask], n_alleles=n_alleles[mask],
                n_focal=n_focal[mask], site_names=site_names,
            )
        except ValueError as e:
            raise TransectCSVError(f"Problem with locus '{locus}' data in '{path}': {e}")

    return transects


# ---------------------------------------------------------------------------
# Genomic cline readers: per-individual data against hybrid index
# ---------------------------------------------------------------------------

def read_genomic_cline_csv(path: str,
                            hybrid_index_col: str = "hybrid_index",
                            n_alleles_col: str | None = None,
                            n_focal_col: str | None = "n_focal",
                            freq_col: str | None = None,
                            ploidy: int = 2,
                            individual_col: str | None = None) -> HybridIndexLocusData:
    """Read single-locus genomic cline data: one row per individual, a
    hybrid-index column, and that individual's genotype at one locus.

    Sample size defaults to `ploidy` for every individual (the normal case
    for a genotyped, non-missing diploid call) unless `n_alleles_col` is
    given explicitly (e.g. for genotype-likelihood-derived counts that
    vary per individual).

    Focal-allele data: either `n_focal_col` (counts) or `freq_col`
    (frequency 0-1, e.g. 0/0.5/1 for a diploid genotype expressed as a
    proportion; counts recovered as round(freq * n_alleles)).
    """
    df = _read_csv(path)

    hybrid_index = _numeric_col(df, hybrid_index_col, path)
    if np.any((hybrid_index < 0) | (hybrid_index > 1)):
        raise TransectCSVError(
            f"Column '{hybrid_index_col}' has values outside [0, 1] -- hybrid "
            f"index must be a proportion between 0 and 1."
        )

    if n_alleles_col is not None:
        n_alleles = _numeric_col(df, n_alleles_col, path)
    else:
        n_alleles = np.full(len(df), float(ploidy))

    n_focal = _resolve_n_focal(df, path, n_alleles, n_focal_col, freq_col)

    individual_names = None
    if individual_col is not None:
        if individual_col not in df.columns:
            raise TransectCSVError(f"--individual-col '{individual_col}' not found in '{path}'.")
        individual_names = df[individual_col].astype(str).tolist()

    try:
        return HybridIndexLocusData(hybrid_index=hybrid_index, n_alleles=n_alleles,
                                     n_focal=n_focal, individual_names=individual_names)
    except ValueError as e:
        raise TransectCSVError(f"Problem with the data in '{path}': {e}")


def detect_genomic_loci(df: pd.DataFrame, n_alleles_suffix: str = ".n",
                         freq_suffix: str | None = ".freq",
                         n_focal_suffix: str | None = None) -> list[str]:
    """Auto-detect locus names from wide-format column naming, analogous
    to `detect_loci` for geographic multi-locus files but with genomic
    cline's own (different) default suffix convention -- there is no
    HZAR-equivalent naming convention to match here since HZAR has no
    genomic cline support (BACKEND_PHASE3.md section 1), so these defaults
    are HyCline's own, not borrowed."""
    if freq_suffix is not None and n_focal_suffix is not None:
        raise TransectCSVError("Specify either freq_suffix or n_focal_suffix, not both.")
    focal_suffix = freq_suffix if freq_suffix is not None else n_focal_suffix
    if focal_suffix is None:
        raise TransectCSVError("Must specify either freq_suffix or n_focal_suffix.")

    loci = []
    for col in df.columns:
        if col.endswith(n_alleles_suffix):
            locus = col[: -len(n_alleles_suffix)]
            if f"{locus}{focal_suffix}" in df.columns:
                loci.append(locus)
    return loci


def read_genomic_cline_multi_locus_wide_csv(
        path: str,
        hybrid_index_col: str = "hybrid_index",
        n_alleles_suffix: str = ".n",
        freq_suffix: str | None = ".freq",
        n_focal_suffix: str | None = None,
        ploidy: int = 2,
        individual_col: str | None = None,
        loci: list[str] | None = None,
) -> dict[str, HybridIndexLocusData]:
    """Read a wide-format multi-locus genomic cline CSV: one row per
    individual, a hybrid-index column, and for each locus L, columns
    '{L}{n_alleles_suffix}' and '{L}{freq_suffix}' (or '{L}{n_focal_suffix}').

    If a locus has no explicit '{L}{n_alleles_suffix}' column pattern
    matched (i.e. you're relying on `ploidy` instead of per-locus sample
    size columns), pass `n_alleles_suffix=None` and every individual is
    assumed to contribute `ploidy` alleles at every locus except where the
    focal-allele column itself is missing.

    Missing data is dropped per locus, not per file -- same rationale as
    `read_multi_locus_wide_csv` for geographic data (BACKEND_PHASE1.md
    section 9.3): different loci commonly have different missing-genotype
    patterns, and requiring complete data across all loci would discard
    usable information unnecessarily.
    """
    df = _read_csv(path)

    hybrid_index = _numeric_col(df, hybrid_index_col, path)
    if np.any((hybrid_index < 0) | (hybrid_index > 1)):
        raise TransectCSVError(
            f"Column '{hybrid_index_col}' has values outside [0, 1] -- hybrid "
            f"index must be a proportion between 0 and 1."
        )

    if n_alleles_suffix is not None:
        detected = detect_genomic_loci(df, n_alleles_suffix, freq_suffix, n_focal_suffix)
    else:
        focal_suffix = freq_suffix if freq_suffix is not None else n_focal_suffix
        if focal_suffix is None:
            raise TransectCSVError("Must specify either freq_suffix or n_focal_suffix.")
        detected = [c[: -len(focal_suffix)] for c in df.columns if c.endswith(focal_suffix)]

    if not detected:
        focal_suffix = freq_suffix if freq_suffix is not None else n_focal_suffix
        raise TransectCSVError(
            f"No loci detected in '{path}' using n_alleles_suffix="
            f"'{n_alleles_suffix}' and focal-allele suffix '{focal_suffix}'. "
            f"Columns present in the file: {', '.join(df.columns)}."
        )

    if loci is not None:
        unknown = [l for l in loci if l not in detected]
        if unknown:
            raise TransectCSVError(
                f"Requested loci {unknown} not found. Loci detected in '{path}': {detected}."
            )
        selected = loci
    else:
        selected = detected

    individual_names_full = None
    if individual_col is not None:
        if individual_col not in df.columns:
            raise TransectCSVError(f"--individual-col '{individual_col}' not found in '{path}'.")
        individual_names_full = df[individual_col].astype(str).tolist()

    results: dict[str, HybridIndexLocusData] = {}
    for locus in selected:
        if n_alleles_suffix is not None:
            n_alleles = _numeric_col(df, f"{locus}{n_alleles_suffix}", path, allow_missing=True)
        else:
            n_alleles = np.full(len(df), float(ploidy))

        n_focal = _resolve_n_focal(
            df, path, n_alleles,
            n_focal_col=(f"{locus}{n_focal_suffix}" if n_focal_suffix is not None else None),
            freq_col=(f"{locus}{freq_suffix}" if freq_suffix is not None else None),
            allow_missing=True,
        )

        mask = ~(np.isnan(n_alleles) | np.isnan(n_focal) | np.isnan(hybrid_index))
        if not np.any(mask):
            raise TransectCSVError(
                f"Locus '{locus}' has no complete (non-missing) rows in '{path}'."
            )

        individual_names = ([s for s, m in zip(individual_names_full, mask) if m]
                             if individual_names_full is not None else None)

        try:
            results[locus] = HybridIndexLocusData(
                hybrid_index=hybrid_index[mask], n_alleles=n_alleles[mask],
                n_focal=n_focal[mask], individual_names=individual_names,
            )
        except ValueError as e:
            raise TransectCSVError(f"Problem with locus '{locus}' data in '{path}': {e}")

    return results


# ---------------------------------------------------------------------------
# Genotype-likelihood readers: multi-locus wide-format CSVs for Phase 4
# Components D/E (hierarchical fitting with genotype-likelihood
# marginalization). One row per *individual* -- see
# `hycline.data.schema.GenotypeLikelihoodTransect`'s docstring for why
# this differs from the hard-count multi-locus readers' one-row-per-site
# convention (genotype likelihoods can't be losslessly aggregated the way
# hard counts can). Missing data is still dropped per locus, same
# rationale as the other multi-locus readers -- the dense-data
# requirement Components D/E actually need is enforced separately, at
# fitting time (see `hycline.cli._stack_for_hierarchical`), not here.
# ---------------------------------------------------------------------------

def detect_gl_loci(df: pd.DataFrame, gl0_suffix: str = ".gl0",
                    gl1_suffix: str = ".gl1", gl2_suffix: str = ".gl2") -> list[str]:
    """Auto-detect locus names from genotype-likelihood wide-format column
    naming: a locus L is detected if columns '{L}{gl0_suffix}',
    '{L}{gl1_suffix}', and '{L}{gl2_suffix}' all exist."""
    loci = []
    for col in df.columns:
        if col.endswith(gl0_suffix):
            locus = col[: -len(gl0_suffix)]
            if f"{locus}{gl1_suffix}" in df.columns and f"{locus}{gl2_suffix}" in df.columns:
                loci.append(locus)
    return loci


def _resolve_gl_triple(df: pd.DataFrame, path: str, locus: str,
                        gl0_suffix: str, gl1_suffix: str, gl2_suffix: str,
                        phred: bool) -> tuple:
    gl0 = _numeric_col(df, f"{locus}{gl0_suffix}", path, allow_missing=True)
    gl1 = _numeric_col(df, f"{locus}{gl1_suffix}", path, allow_missing=True)
    gl2 = _numeric_col(df, f"{locus}{gl2_suffix}", path, allow_missing=True)
    if phred:
        from hycline.data.genotype_likelihood import phred_to_log_likelihood
        gl0, gl1, gl2 = (phred_to_log_likelihood(gl0), phred_to_log_likelihood(gl1),
                         phred_to_log_likelihood(gl2))
    return gl0, gl1, gl2


def read_gl_csv(path: str,
                 distance_col: str | None = "distance",
                 lat_col: str | None = None, lon_col: str | None = None,
                 transect_method: str = "reference",
                 reference_lat: float | None = None, reference_lon: float | None = None,
                 gl0_col: str = "log_gl0", gl1_col: str = "log_gl1", gl2_col: str = "log_gl2",
                 phred: bool = False,
                 individual_col: str | None = None) -> GenotypeLikelihoodTransect:
    """Read single-locus genotype-likelihood data for a geographic cline:
    one row per individual (individuals at the same site share the same
    distance value), and three columns holding natural-log genotype
    likelihoods P(reads | genotype=0/1/2) -- the single-locus analog of
    `read_gl_multi_locus_wide_csv` (Phase 4 Component D's own reader),
    feeding `hycline.inference.mle_genotype_likelihood.
    fit_mle_geographic_gl` (Phase 5) instead.

    Pass `phred=True` if your columns hold Phred-scaled PL values instead
    (as GATK/bcftools/ANGSD commonly output).
    """
    df = _read_csv(path)

    distance = _resolve_distance(df, path, distance_col, lat_col, lon_col,
                                  transect_method, reference_lat, reference_lon)

    gl0 = _numeric_col(df, gl0_col, path)
    gl1 = _numeric_col(df, gl1_col, path)
    gl2 = _numeric_col(df, gl2_col, path)
    if phred:
        from hycline.data.genotype_likelihood import phred_to_log_likelihood
        gl0, gl1, gl2 = (phred_to_log_likelihood(gl0), phred_to_log_likelihood(gl1),
                         phred_to_log_likelihood(gl2))

    individual_names = None
    if individual_col is not None:
        if individual_col not in df.columns:
            raise TransectCSVError(f"--individual-col '{individual_col}' not found in '{path}'.")
        individual_names = df[individual_col].astype(str).tolist()

    try:
        return GenotypeLikelihoodTransect(distance=distance, log_gl0=gl0, log_gl1=gl1,
                                           log_gl2=gl2, individual_names=individual_names)
    except ValueError as e:
        raise TransectCSVError(f"Problem with the data in '{path}': {e}")


def read_genomic_gl_csv(path: str,
                         hybrid_index_col: str = "hybrid_index",
                         gl0_col: str = "log_gl0", gl1_col: str = "log_gl1",
                         gl2_col: str = "log_gl2",
                         phred: bool = False,
                         individual_col: str | None = None) -> GenotypeLikelihoodLocusData:
    """Read single-locus genotype-likelihood data for a genomic cline: one
    row per individual, a hybrid-index column, and three columns holding
    natural-log genotype likelihoods -- the single-locus analog of
    `read_genomic_gl_multi_locus_wide_csv` (Phase 4 Component E's own
    reader), feeding `hycline.inference.mle_genotype_likelihood.
    fit_mle_genomic_gl` (Phase 5) instead.
    """
    df = _read_csv(path)

    hybrid_index = _numeric_col(df, hybrid_index_col, path)
    if np.any((hybrid_index < 0) | (hybrid_index > 1)):
        raise TransectCSVError(
            f"Column '{hybrid_index_col}' has values outside [0, 1] -- hybrid "
            f"index must be a proportion between 0 and 1."
        )

    gl0 = _numeric_col(df, gl0_col, path)
    gl1 = _numeric_col(df, gl1_col, path)
    gl2 = _numeric_col(df, gl2_col, path)
    if phred:
        from hycline.data.genotype_likelihood import phred_to_log_likelihood
        gl0, gl1, gl2 = (phred_to_log_likelihood(gl0), phred_to_log_likelihood(gl1),
                         phred_to_log_likelihood(gl2))

    individual_names = None
    if individual_col is not None:
        if individual_col not in df.columns:
            raise TransectCSVError(f"--individual-col '{individual_col}' not found in '{path}'.")
        individual_names = df[individual_col].astype(str).tolist()

    try:
        return GenotypeLikelihoodLocusData(hybrid_index=hybrid_index, log_gl0=gl0,
                                            log_gl1=gl1, log_gl2=gl2,
                                            individual_names=individual_names)
    except ValueError as e:
        raise TransectCSVError(f"Problem with the data in '{path}': {e}")


def read_gl_multi_locus_wide_csv(
        path: str,
        distance_col: str | None = "distance",
        lat_col: str | None = None, lon_col: str | None = None,
        transect_method: str = "reference",
        reference_lat: float | None = None, reference_lon: float | None = None,
        gl0_suffix: str = ".gl0", gl1_suffix: str = ".gl1", gl2_suffix: str = ".gl2",
        phred: bool = False,
        individual_col: str | None = None,
        loci: list[str] | None = None,
) -> dict[str, GenotypeLikelihoodTransect]:
    """Read a wide-format multi-locus genotype-likelihood CSV for
    geographic clines: one row per individual (individuals at the same
    site simply share the same distance value), and for each locus L,
    three columns '{L}{gl0_suffix}', '{L}{gl1_suffix}', '{L}{gl2_suffix}'
    holding natural-log genotype likelihoods P(reads | genotype=0/1/2).

    Pass `phred=True` if your columns hold Phred-scaled PL values instead
    (as GATK/bcftools/ANGSD commonly output) -- they're converted via
    `hycline.data.genotype_likelihood.phred_to_log_likelihood` before use.

    Feeds `hycline.inference.mcmc_hierarchical_gl.fit_hierarchical_batch_gl`
    (Phase 4 Component D) once stacked into dense arrays -- see
    `hycline.cli._stack_for_hierarchical` for that step and the dense-data
    requirement it enforces (not handled by this reader, which -- like the
    other multi-locus readers -- drops missing rows per locus rather than
    requiring complete data up front).
    """
    df = _read_csv(path)

    distance = _resolve_distance(df, path, distance_col, lat_col, lon_col,
                                  transect_method, reference_lat, reference_lon)

    detected = detect_gl_loci(df, gl0_suffix, gl1_suffix, gl2_suffix)
    if not detected:
        raise TransectCSVError(
            f"No loci detected in '{path}' using genotype-likelihood suffixes "
            f"'{gl0_suffix}'/'{gl1_suffix}'/'{gl2_suffix}'. Expected column triples "
            f"like '<locus>{gl0_suffix}', '<locus>{gl1_suffix}', '<locus>{gl2_suffix}'. "
            f"Columns present in the file: {', '.join(df.columns)}."
        )

    if loci is not None:
        unknown = [l for l in loci if l not in detected]
        if unknown:
            raise TransectCSVError(
                f"Requested loci {unknown} not found. Loci detected in '{path}': {detected}."
            )
        selected = loci
    else:
        selected = detected

    individual_names_full = None
    if individual_col is not None:
        if individual_col not in df.columns:
            raise TransectCSVError(f"--individual-col '{individual_col}' not found in '{path}'.")
        individual_names_full = df[individual_col].astype(str).tolist()

    results: dict[str, GenotypeLikelihoodTransect] = {}
    for locus in selected:
        gl0, gl1, gl2 = _resolve_gl_triple(df, path, locus, gl0_suffix, gl1_suffix,
                                            gl2_suffix, phred)

        mask = ~(np.isnan(gl0) | np.isnan(gl1) | np.isnan(gl2) | np.isnan(distance))
        if not np.any(mask):
            raise TransectCSVError(
                f"Locus '{locus}' has no complete (non-missing) rows in '{path}'."
            )

        individual_names = ([s for s, m in zip(individual_names_full, mask) if m]
                             if individual_names_full is not None else None)

        try:
            results[locus] = GenotypeLikelihoodTransect(
                distance=distance[mask], log_gl0=gl0[mask], log_gl1=gl1[mask],
                log_gl2=gl2[mask], individual_names=individual_names,
            )
        except ValueError as e:
            raise TransectCSVError(f"Problem with locus '{locus}' data in '{path}': {e}")

    return results


def read_genomic_gl_multi_locus_wide_csv(
        path: str,
        hybrid_index_col: str = "hybrid_index",
        gl0_suffix: str = ".gl0", gl1_suffix: str = ".gl1", gl2_suffix: str = ".gl2",
        phred: bool = False,
        individual_col: str | None = None,
        loci: list[str] | None = None,
) -> dict[str, GenotypeLikelihoodLocusData]:
    """Read a wide-format multi-locus genotype-likelihood CSV for genomic
    clines: one row per individual, a hybrid-index column, and for each
    locus L, three columns holding natural-log genotype likelihoods
    (or Phred-scaled PL values, with `phred=True`) -- the genomic-cline
    analog of `read_gl_multi_locus_wide_csv`.

    Feeds `hycline.inference.mcmc_genomic_hierarchical_gl.
    fit_hierarchical_genomic_batch_gl` (Phase 4 Component E).
    """
    df = _read_csv(path)

    hybrid_index = _numeric_col(df, hybrid_index_col, path)
    if np.any((hybrid_index < 0) | (hybrid_index > 1)):
        raise TransectCSVError(
            f"Column '{hybrid_index_col}' has values outside [0, 1] -- hybrid "
            f"index must be a proportion between 0 and 1."
        )

    detected = detect_gl_loci(df, gl0_suffix, gl1_suffix, gl2_suffix)
    if not detected:
        raise TransectCSVError(
            f"No loci detected in '{path}' using genotype-likelihood suffixes "
            f"'{gl0_suffix}'/'{gl1_suffix}'/'{gl2_suffix}'. Expected column triples "
            f"like '<locus>{gl0_suffix}', '<locus>{gl1_suffix}', '<locus>{gl2_suffix}'. "
            f"Columns present in the file: {', '.join(df.columns)}."
        )

    if loci is not None:
        unknown = [l for l in loci if l not in detected]
        if unknown:
            raise TransectCSVError(
                f"Requested loci {unknown} not found. Loci detected in '{path}': {detected}."
            )
        selected = loci
    else:
        selected = detected

    individual_names_full = None
    if individual_col is not None:
        if individual_col not in df.columns:
            raise TransectCSVError(f"--individual-col '{individual_col}' not found in '{path}'.")
        individual_names_full = df[individual_col].astype(str).tolist()

    results: dict[str, GenotypeLikelihoodLocusData] = {}
    for locus in selected:
        gl0, gl1, gl2 = _resolve_gl_triple(df, path, locus, gl0_suffix, gl1_suffix,
                                            gl2_suffix, phred)

        mask = ~(np.isnan(gl0) | np.isnan(gl1) | np.isnan(gl2) | np.isnan(hybrid_index))
        if not np.any(mask):
            raise TransectCSVError(
                f"Locus '{locus}' has no complete (non-missing) rows in '{path}'."
            )

        individual_names = ([s for s, m in zip(individual_names_full, mask) if m]
                             if individual_names_full is not None else None)

        try:
            results[locus] = GenotypeLikelihoodLocusData(
                hybrid_index=hybrid_index[mask], log_gl0=gl0[mask], log_gl1=gl1[mask],
                log_gl2=gl2[mask], individual_names=individual_names,
            )
        except ValueError as e:
            raise TransectCSVError(f"Problem with locus '{locus}' data in '{path}': {e}")

    return results


# ---------------------------------------------------------------------------
# Quantitative trait readers (HZAR's other core data type -- see
# hycline.data.schema.TraitTransect and BACKEND_TRAIT.md)
# ---------------------------------------------------------------------------

def read_trait_csv(path: str,
                    distance_col: str | None = "distance",
                    mean_col: str = "trait_mean",
                    n_col: str = "n",
                    sd_col: str | None = None,
                    lat_col: str | None = None, lon_col: str | None = None,
                    transect_method: str = "reference",
                    reference_lat: float | None = None,
                    reference_lon: float | None = None,
                    site_col: str | None = None) -> TraitTransect:
    """Read site-aggregated quantitative trait transect data from a CSV
    file (one row per sampling site) -- the trait analog of
    `read_allele_freq_csv`.

    Distance: either `distance_col` (a pre-computed column) or both
    `lat_col`/`lon_col` (coordinates, auto-collapsed to 1D), same as
    `read_allele_freq_csv`.

    `sd_col` is optional *at the reader level* (so this function can
    still be used to inspect/preprocess data that doesn't have it yet),
    but `hycline.inference.mle_trait.fit_mle_trait` requires it and will
    raise a clear error if it's missing -- HZAR's own model always uses
    each site's observed sample variance as data, not just its mean
    (BACKEND_TRAIT.md section 1).

    Raises
    ------
    TransectCSVError : with a message describing exactly what went wrong.
    """
    df = _read_csv(path)

    distance = _resolve_distance(df, path, distance_col, lat_col, lon_col,
                                  transect_method, reference_lat, reference_lon)

    if mean_col not in df.columns:
        raise TransectCSVError(f"--mean-col '{mean_col}' not found in '{path}'.")
    trait_mean = df[mean_col].to_numpy(dtype=float)

    if n_col not in df.columns:
        raise TransectCSVError(f"--n-col '{n_col}' not found in '{path}'.")
    n = df[n_col].to_numpy(dtype=float)

    trait_sd = None
    if sd_col is not None:
        if sd_col not in df.columns:
            raise TransectCSVError(f"--sd-col '{sd_col}' not found in '{path}'.")
        trait_sd = df[sd_col].to_numpy(dtype=float)

    site_names = None
    if site_col is not None:
        if site_col not in df.columns:
            raise TransectCSVError(f"--site-col '{site_col}' not found in '{path}'.")
        site_names = df[site_col].astype(str).tolist()

    try:
        return TraitTransect(distance=distance, trait_mean=trait_mean, n=n,
                              trait_sd=trait_sd, site_names=site_names)
    except ValueError as e:
        raise TransectCSVError(f"Problem with the data in '{path}': {e}")


def detect_traits(df: pd.DataFrame, mean_suffix: str = ".mean",
                   n_suffix: str = ".n", sd_suffix: str = ".sd") -> list[str]:
    """Auto-detect trait names from wide-format column naming, e.g. for
    columns 'mass.mean' and 'mass.n', the detected trait is 'mass'.
    `sd_suffix` columns are optional per trait -- a trait is detected as
    long as its `.mean`/`.n` pair is present, regardless of whether a
    `.sd` column also exists for it."""
    traits = []
    for col in df.columns:
        if col.endswith(mean_suffix):
            trait = col[: -len(mean_suffix)]
            if f"{trait}{n_suffix}" in df.columns:
                traits.append(trait)
    return traits


def read_raw_trait_csv(path: str, trait_col: str = "trait_value",
                        distance_col: str | None = "distance",
                        site_col: str | None = None,
                        lat_col: str | None = None, lon_col: str | None = None,
                        transect_method: str = "reference",
                        reference_lat: float | None = None,
                        reference_lon: float | None = None,
                        ddof: int = 0) -> TraitTransect:
    """Read RAW, per-individual quantitative trait measurements (one row
    per individual) and aggregate to the site-level (mean, n, SD) shape
    `TraitTransect`/`fit_mle_trait` need -- an alternative to
    `read_trait_csv` for users who have individual-level data rather
    than pre-aggregated site summaries. HZAR itself supports both input
    styles; this is the raw-input analog, aggregating internally to feed
    the same sufficient-statistics likelihood
    (`hycline.likelihoods.gaussian`) -- see BACKEND_TRAIT.md section 4.

    Individuals are grouped into sites by `site_col` if given, or by
    exact matching `distance_col`/coordinate value otherwise. Every
    individual grouped into the same site must share the same distance
    value -- checked directly (not assumed), the same requirement
    `hycline.data.vcf`'s hard-count writers already have for site
    aggregation elsewhere in this project.

    `ddof` (delta degrees of freedom for the per-site SD, default 0 --
    numpy's own default, population/biased SD) is exposed explicitly
    rather than silently fixed: whether HZAR's own raw-data aggregation
    uses `ddof=0` or `ddof=1` (R's `var()` default) wasn't independently
    confirmed while building this reader (a real, stated scope limit,
    not a claim of exact parity the way the core likelihood formula
    itself was verified -- see BACKEND_TRAIT.md section 2). The
    difference matters most at small per-site sample sizes and shrinks
    as n grows; pass `ddof=1` if you specifically need to match a
    `var()`-based computation elsewhere.

    Rows with a missing (`NaN`) trait value are dropped before
    aggregation, not counted toward that site's `n`.

    Raises
    ------
    TransectCSVError : with a message describing exactly what went wrong.
    """
    df = _read_csv(path)
    distance = _resolve_distance(df, path, distance_col, lat_col, lon_col,
                                  transect_method, reference_lat, reference_lon)

    if trait_col not in df.columns:
        raise TransectCSVError(f"--trait-col '{trait_col}' not found in '{path}'.")
    trait_value = df[trait_col].to_numpy(dtype=float)

    if site_col is not None:
        if site_col not in df.columns:
            raise TransectCSVError(f"--site-col '{site_col}' not found in '{path}'.")
        site_labels = df[site_col].astype(str).tolist()
    else:
        site_labels = [repr(float(d)) for d in distance]

    site_distance: dict = {}
    site_order: list = []
    site_values: dict = {}
    for i, site in enumerate(site_labels):
        if site not in site_distance:
            site_distance[site] = distance[i]
            site_order.append(site)
            site_values[site] = []
        elif site_distance[site] != distance[i]:
            raise TransectCSVError(
                f"Site '{site}' has inconsistent {distance_col or 'distance'} values "
                f"({site_distance[site]} vs {distance[i]}) across its individuals in "
                f"'{path}' -- every individual at the same site must share the same "
                f"distance value."
            )
        site_values[site].append(trait_value[i])

    out_distance, site_means, site_sds, site_ns = [], [], [], []
    for site in site_order:
        vals = np.asarray(site_values[site], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        out_distance.append(site_distance[site])
        site_means.append(float(vals.mean()))
        site_sds.append(float(vals.std(ddof=ddof)) if len(vals) > ddof else 0.0)
        site_ns.append(len(vals))

    if not site_means:
        raise TransectCSVError(
            f"No sites with at least one valid (non-missing) '{trait_col}' measurement "
            f"found in '{path}'."
        )

    try:
        return TraitTransect(distance=np.array(out_distance), trait_mean=np.array(site_means),
                              n=np.array(site_ns, dtype=float), trait_sd=np.array(site_sds))
    except ValueError as e:
        raise TransectCSVError(f"Problem aggregating the data in '{path}': {e}")


def read_multi_trait_wide_csv(path: str,
                               distance_col: str | None = "distance",
                               mean_suffix: str = ".mean",
                               n_suffix: str = ".n",
                               sd_suffix: str = ".sd",
                               lat_col: str | None = None, lon_col: str | None = None,
                               transect_method: str = "reference",
                               reference_lat: float | None = None,
                               reference_lon: float | None = None,
                               site_col: str | None = None,
                               traits: list[str] | None = None
                               ) -> dict[str, TraitTransect]:
    """Read a wide-format CSV with multiple traits (one row per site,
    `<trait>.mean`/`<trait>.n`/optionally `<trait>.sd` columns per trait)
    -- the trait analog of `read_multi_locus_wide_csv`. Returns a dict
    mapping trait name -> `TraitTransect`.

    Missing values are dropped independently per trait (a row missing
    only one trait's mean doesn't exclude that site from other traits'
    transects) -- the same convention `read_multi_locus_wide_csv` uses.
    """
    df = _read_csv(path)

    distance = _resolve_distance(df, path, distance_col, lat_col, lon_col,
                                  transect_method, reference_lat, reference_lon)

    if traits is None:
        traits = detect_traits(df, mean_suffix, n_suffix, sd_suffix)
        if not traits:
            raise TransectCSVError(
                f"No traits detected in '{path}' (looked for columns ending in "
                f"'{mean_suffix}' with a matching '{n_suffix}' column). Specify "
                f"--traits explicitly if your columns use different names."
            )

    site_names_full = None
    if site_col is not None:
        if site_col not in df.columns:
            raise TransectCSVError(f"--site-col '{site_col}' not found in '{path}'.")
        site_names_full = df[site_col].astype(str).tolist()

    results = {}
    for trait in traits:
        mean_col = f"{trait}{mean_suffix}"
        n_col = f"{trait}{n_suffix}"
        sd_col = f"{trait}{sd_suffix}"
        if mean_col not in df.columns or n_col not in df.columns:
            raise TransectCSVError(
                f"Trait '{trait}' is missing its '{mean_suffix}' or '{n_suffix}' column in "
                f"'{path}'."
            )
        trait_mean = df[mean_col].to_numpy(dtype=float)
        n = df[n_col].to_numpy(dtype=float)
        trait_sd = df[sd_col].to_numpy(dtype=float) if sd_col in df.columns else None

        mask = ~(np.isnan(trait_mean) | np.isnan(n))
        if trait_sd is not None:
            mask &= ~np.isnan(trait_sd)
        if not np.any(mask):
            raise TransectCSVError(f"Trait '{trait}' has no complete (non-missing) rows in '{path}'.")

        site_names = ([s for s, m in zip(site_names_full, mask) if m]
                       if site_names_full is not None else None)

        try:
            results[trait] = TraitTransect(
                distance=distance[mask], trait_mean=trait_mean[mask], n=n[mask],
                trait_sd=trait_sd[mask] if trait_sd is not None else None,
                site_names=site_names,
            )
        except ValueError as e:
            raise TransectCSVError(f"Problem with trait '{trait}' data in '{path}': {e}")

    return results
