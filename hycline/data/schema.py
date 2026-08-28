"""Typed data containers for cline-fitting input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def values_consistent(a, b) -> bool:
    """True if `a` and `b` should be treated as the same value for a
    per-site/per-individual consistency check -- ordinary equality, plus
    NaN treated as consistent with NaN (a site/individual where every
    record genuinely has the same missing value is not an inconsistency
    to report). Two different NaN-producing inputs (e.g. two truly
    different missing values) are indistinguishable from each other once
    they're NaN, which is the same limitation any NaN-as-missing
    convention has -- this function doesn't try to work around that,
    only avoids the opposite, more common failure of a spurious
    "inconsistent" error firing on `nan != nan` itself.

    Shared by every reader/writer that aggregates individuals into sites
    (or checks per-individual consistency) and needs to tolerate this --
    `hycline.data.vcf`, `.beagle`, `.converter`, `.io`, `.geo` all have
    their own copy of the same site-aggregation loop with this exact
    comparison at its core."""
    if a == b:
        return True
    try:
        return bool(np.isnan(a)) and bool(np.isnan(b))
    except TypeError:
        return False


@dataclass
class AlleleFreqTransect:
    """Allele-count data along a geographic transect (the classical HZAR
    "molecular marker" data type).

    Attributes
    ----------
    distance : geographic distance of each sampling site along the transect
    n_alleles : total number of alleles sampled at each site (2 * n_individuals
        for a diploid autosomal marker)
    n_focal : count of the focal allele at each site (0 <= n_focal <= n_alleles)
    site_names : optional human-readable labels for each site
    """

    distance: np.ndarray
    n_alleles: np.ndarray
    n_focal: np.ndarray
    site_names: list[str] | None = None

    def __post_init__(self):
        self.distance = np.asarray(self.distance, dtype=float)
        self.n_alleles = np.asarray(self.n_alleles, dtype=float)
        self.n_focal = np.asarray(self.n_focal, dtype=float)

        if not (self.distance.shape == self.n_alleles.shape == self.n_focal.shape):
            raise ValueError("distance, n_alleles, and n_focal must have the same shape")
        if len(self.distance) == 0:
            raise ValueError("Need at least one site")
        if np.any(self.n_focal < 0):
            raise ValueError("n_focal cannot be negative at any site")
        if np.any(self.n_focal > self.n_alleles):
            raise ValueError("n_focal cannot exceed n_alleles at any site")
        if np.any(self.n_alleles <= 0):
            raise ValueError("n_alleles must be positive at every site")

    @property
    def observed_freq(self) -> np.ndarray:
        """Observed (raw) allele frequency at each site -- useful for plotting."""
        return self.n_focal / self.n_alleles

    def __len__(self) -> int:
        return len(self.distance)


def polarize_allele_freq_transect(transect: "AlleleFreqTransect",
                                   n_edge_sites: int = 1) -> tuple:
    """Re-orient a geographic-cline locus so its observed focal-allele
    frequency is lower at the near (low-distance) end of the transect
    than at the far (high-distance) end, flipping `n_focal ->
    n_alleles - n_focal` if the raw data has it the other way round.

    Which physical allele happens to be "focal" (REF vs. ALT in a VCF,
    or whichever label an upstream reader assigned) is arbitrary with
    respect to geography -- it doesn't change the underlying biology,
    or the cline's fit (center/width/log-likelihood are all invariant
    under this flip; only `pmin`/`pmax` swap). But it matters visually
    the moment more than one locus is plotted on shared axes
    (`plot_multi_locus_overlay`, `plot_hierarchical_geographic_grid`,
    the candidate-overlay plots): with no consistent orientation, some
    loci's curves rise left-to-right and others fall, which is exactly
    the "arbitrary REF/ALT direction" bug already fixed for hybrid-index
    estimation (`hycline.data.hybrid_index.polarize_genotypes`) showing
    up again in a different pipeline. This is the geographic-cline
    analog of that fix.

    Uses the same edge-site-pooling convention as
    `hycline.inference.mle._derive_pmin_pmax_single_locus`/
    `hycline.inference.mcmc_hierarchical._derive_pmin_pmax` to decide
    what "the extremes" means (the `n_edge_sites` sites nearest each end
    of the transect, pooled by summed counts) -- the same statistic
    already used to derive `pmin`/`pmax` for `--scaling fixed`, reused
    here rather than inventing a second, inconsistent definition of
    "near the edge." Unlike that function, this needs no zero-coverage
    guard: `AlleleFreqTransect`'s own constructor already requires
    `n_alleles > 0` at every site, so every pooled sum is guaranteed
    positive.

    Returns `(new_transect, flipped)` -- `new_transect` is a fresh
    `AlleleFreqTransect` (the input is never mutated in place);
    `flipped` is `True` iff a flip was actually applied, for the caller
    to report to the user.
    """
    order = np.argsort(transect.distance)
    n_alleles_sorted = transect.n_alleles[order]
    n_focal_sorted = transect.n_focal[order]
    k = min(n_edge_sites, len(n_alleles_sorted))
    # AlleleFreqTransect's own constructor already requires n_alleles > 0
    # at every site, so the pooled sums below are always > 0 -- no
    # zero-coverage guard needed here the way _derive_pmin_pmax's own
    # (operating on raw, not-yet-validated arrays) needs one.
    freq_low = n_focal_sorted[:k].sum() / n_alleles_sorted[:k].sum()
    freq_high = n_focal_sorted[-k:].sum() / n_alleles_sorted[-k:].sum()
    if freq_low <= freq_high:
        return transect, False

    flipped = AlleleFreqTransect(
        distance=transect.distance, n_alleles=transect.n_alleles,
        n_focal=transect.n_alleles - transect.n_focal,
        site_names=transect.site_names,
    )
    return flipped, True


@dataclass
class TraitTransect:
    """Site-aggregated quantitative trait data along a geographic
    transect -- HZAR's other core data type, alongside
    `AlleleFreqTransect`'s molecular-marker counts. Any continuous,
    measured phenotype (body size, a color/plumage score, a song
    frequency, a morphological index) that varies smoothly across a
    hybrid zone can be fit with the same sigmoid cline shape
    (`hycline.models.cline.cline_function`/`ClineParams`) used for allele
    frequencies -- only the likelihood differs (Gaussian here, binomial
    there). See BACKEND_TRAIT.md for the full design and how this differs
    from HZAR's own internal choices where they aren't independently
    verifiable from published HZAR output alone.

    Attributes
    ----------
    distance : geographic distance of each sampling site along the transect
    trait_mean : mean trait value of the individuals sampled at each site
    trait_sd : sample SD of the *individual* trait measurements at each
        site (not the SD/SE of the site mean itself -- see
        `hycline.likelihoods.gaussian` for exactly how this feeds into
        the likelihood). Required for fitting
        (`hycline.inference.mle_trait.fit_mle_trait`) -- HZAR's own
        quantitative-trait model always uses each site's observed sample
        variance as data, not just its mean (BACKEND_TRAIT.md section 1).
        `None` is only accepted here at construction time for callers
        that don't yet have it (e.g. reading a CSV without an SD column);
        fitting itself will raise a clear error if it's still `None`.
    n : number of individuals measured at each site.
    site_names : optional human-readable labels for each site.
    """

    distance: np.ndarray
    trait_mean: np.ndarray
    n: np.ndarray
    trait_sd: np.ndarray | None = None
    site_names: list[str] | None = None

    def __post_init__(self):
        self.distance = np.asarray(self.distance, dtype=float)
        self.trait_mean = np.asarray(self.trait_mean, dtype=float)
        self.n = np.asarray(self.n, dtype=float)
        if self.trait_sd is not None:
            self.trait_sd = np.asarray(self.trait_sd, dtype=float)

        shapes = [self.distance.shape, self.trait_mean.shape, self.n.shape]
        if self.trait_sd is not None:
            shapes.append(self.trait_sd.shape)
        if len(set(shapes)) != 1:
            raise ValueError(
                "distance, trait_mean, n, and trait_sd (if given) must have the same shape")
        if len(self.distance) == 0:
            raise ValueError("Need at least one site")
        if np.any(self.n <= 0):
            raise ValueError("n must be positive at every site")
        if self.trait_sd is not None and np.any(self.trait_sd < 0):
            raise ValueError("trait_sd cannot be negative")
        if self.trait_sd is not None and np.any((self.trait_sd == 0) & (self.n > 1)):
            raise ValueError(
                "trait_sd is 0 at a site with n > 1 -- if this is genuinely zero variance "
                "(e.g. a single repeated measurement recorded as n>1 by mistake), fix the "
                "input; a literal 0 SD would give that site infinite weight in the "
                "known-variance likelihood."
            )

    def __len__(self) -> int:
        return len(self.distance)


@dataclass
class HybridIndexLocusData:
    """Per-individual genotype data for one locus, against genome-wide
    hybrid index (the genomic-cline analog of `AlleleFreqTransect`).

    Where `AlleleFreqTransect` holds per-*site* allele counts against
    geographic distance, this holds per-*individual* allele counts against
    hybrid index -- the two data types the geographic and genomic cline
    models respectively expect. See BACKEND_PHASE3.md section 2.

    Attributes
    ----------
    hybrid_index : each individual's genome-wide admixture proportion
        (0 = pure parental population 0, 1 = pure parental population 1),
        typically estimated across many loci -- see `hycline.data.hybrid_index`.
    n_alleles : total alleles scored at this locus for that individual
        (ploidy, if genotyped without missing data; may be less if some
        allele copies are missing/uncalled).
    n_focal : count of the "parental population 1" allele at this locus
        for that individual (0 <= n_focal <= n_alleles).
    individual_names : optional sample labels.
    """

    hybrid_index: np.ndarray
    n_alleles: np.ndarray
    n_focal: np.ndarray
    individual_names: list[str] | None = None

    def __post_init__(self):
        self.hybrid_index = np.asarray(self.hybrid_index, dtype=float)
        self.n_alleles = np.asarray(self.n_alleles, dtype=float)
        self.n_focal = np.asarray(self.n_focal, dtype=float)

        if not (self.hybrid_index.shape == self.n_alleles.shape == self.n_focal.shape):
            raise ValueError("hybrid_index, n_alleles, and n_focal must have the same shape")
        if len(self.hybrid_index) == 0:
            raise ValueError("Need at least one individual")
        if np.any(self.n_focal < 0):
            raise ValueError("n_focal cannot be negative for any individual")
        if np.any(self.n_focal > self.n_alleles):
            raise ValueError("n_focal cannot exceed n_alleles for any individual")
        if np.any(self.n_alleles <= 0):
            raise ValueError("n_alleles must be positive for every individual")
        if np.any((self.hybrid_index < 0) | (self.hybrid_index > 1)):
            raise ValueError("hybrid_index must be in [0, 1] for every individual")

    @property
    def observed_freq(self) -> np.ndarray:
        return self.n_focal / self.n_alleles

    def __len__(self) -> int:
        return len(self.hybrid_index)


@dataclass
class GenotypeLikelihoodTransect:
    """Per-individual genotype-likelihood data for one locus, against
    geographic distance -- the genotype-likelihood analog of
    `AlleleFreqTransect`.

    Unlike `AlleleFreqTransect` (which holds site-level aggregated allele
    counts, implicitly treating genotypes as known with certainty), this
    holds per-*individual* genotype likelihoods, since GL data cannot be
    losslessly aggregated to a site-level count the way hard genotype
    calls can. `distance` therefore has one entry per *individual*, not
    per site -- individuals sampled at the same locality simply share the
    same `distance` value. See BACKEND_PHASE5.md section 2 for the full
    rationale and the marginal-likelihood math this feeds into.

    Attributes
    ----------
    distance : each individual's geographic distance along the transect
        (repeated for individuals sharing a site).
    log_gl0, log_gl1, log_gl2 : natural-log genotype likelihoods
        P(reads | genotype = 0, 1, 2 copies of the focal allele) for a
        diploid locus, one entry per individual. These do not need to be
        normalized to sum to 1 across genotypes (genotype likelihoods are
        not genotype *probabilities*) -- see
        `hycline.likelihoods.genotype_likelihood` for why an arbitrary
        per-individual additive constant in log-space doesn't affect
        inference.
    individual_names : optional sample labels.
    """

    distance: np.ndarray
    log_gl0: np.ndarray
    log_gl1: np.ndarray
    log_gl2: np.ndarray
    individual_names: list[str] | None = None

    def __post_init__(self):
        self.distance = np.asarray(self.distance, dtype=float)
        self.log_gl0 = np.asarray(self.log_gl0, dtype=float)
        self.log_gl1 = np.asarray(self.log_gl1, dtype=float)
        self.log_gl2 = np.asarray(self.log_gl2, dtype=float)

        shapes = {self.distance.shape, self.log_gl0.shape, self.log_gl1.shape,
                  self.log_gl2.shape}
        if len(shapes) != 1:
            raise ValueError("distance, log_gl0, log_gl1, and log_gl2 must have the same shape")
        if len(self.distance) == 0:
            raise ValueError("Need at least one individual")

    def __len__(self) -> int:
        return len(self.distance)


def polarize_genotype_likelihood_transect(transect: "GenotypeLikelihoodTransect",
                                           edge_fraction: float = 0.2) -> tuple:
    """Genotype-likelihood analog of `polarize_allele_freq_transect`:
    re-orients a locus so its (argmax-genotype-proxy) focal-allele
    frequency is lower at the near end of the transect than at the far
    end, flipping `log_gl0 <-> log_gl2` (dosage 0 and dosage 2 swap;
    `log_gl1`, the heterozygote, is unaffected by which homozygote is
    "focal") if the raw data has it the other way round. See
    `polarize_allele_freq_transect`'s docstring for why this matters and
    why it doesn't change the fit itself, only which direction it plots.

    GL data is per-individual, not per-site (see
    `GenotypeLikelihoodTransect`'s own docstring), so there's no
    `n_edge_sites`-style site count to pool -- instead, this uses the
    same "nearest `edge_fraction` of individuals by distance" convention
    `hycline.inference.mle_genotype_likelihood._initial_guess_geographic`
    already uses for its own pmin/pmax starting-value guess (default
    20%, i.e. the same "nearest fifth" that helper uses), rather than
    inventing a second, inconsistent notion of "near the edge" for GL
    data specifically. The argmax-genotype proxy this (and that
    function) uses is a rough stand-in only, appropriate for an
    orientation *decision* -- the fit itself properly marginalizes over
    the full genotype-likelihood uncertainty regardless of this flip.

    Returns `(new_transect, flipped)`, same convention as
    `polarize_allele_freq_transect`.
    """
    gl_stack = np.stack([transect.log_gl0, transect.log_gl1, transect.log_gl2], axis=0)
    hard_genotype = np.argmax(gl_stack, axis=0)
    freq_proxy = hard_genotype / 2.0

    order = np.argsort(transect.distance)
    freq_sorted = freq_proxy[order]
    k = max(1, int(len(freq_sorted) * edge_fraction))
    freq_low = float(np.mean(freq_sorted[:k]))
    freq_high = float(np.mean(freq_sorted[-k:]))
    if freq_low <= freq_high:
        return transect, False

    flipped = GenotypeLikelihoodTransect(
        distance=transect.distance, log_gl0=transect.log_gl2,
        log_gl1=transect.log_gl1, log_gl2=transect.log_gl0,
        individual_names=transect.individual_names,
    )
    return flipped, True


@dataclass
class GenotypeLikelihoodLocusData:
    """Per-individual genotype-likelihood data for one locus, against
    genome-wide hybrid index -- the genotype-likelihood analog of
    `HybridIndexLocusData`.

    See `GenotypeLikelihoodTransect`'s docstring for the general rationale
    (per-individual, not aggregated) and BACKEND_PHASE5.md section 2 for
    the marginal-likelihood math.

    Attributes
    ----------
    hybrid_index : each individual's genome-wide hybrid index (0-1).
    log_gl0, log_gl1, log_gl2 : natural-log genotype likelihoods
        P(reads | genotype = 0, 1, 2 copies of the "parental population 1"
        allele) for a diploid locus, one entry per individual.
    individual_names : optional sample labels.
    """

    hybrid_index: np.ndarray
    log_gl0: np.ndarray
    log_gl1: np.ndarray
    log_gl2: np.ndarray
    individual_names: list[str] | None = None

    def __post_init__(self):
        self.hybrid_index = np.asarray(self.hybrid_index, dtype=float)
        self.log_gl0 = np.asarray(self.log_gl0, dtype=float)
        self.log_gl1 = np.asarray(self.log_gl1, dtype=float)
        self.log_gl2 = np.asarray(self.log_gl2, dtype=float)

        shapes = {self.hybrid_index.shape, self.log_gl0.shape, self.log_gl1.shape,
                  self.log_gl2.shape}
        if len(shapes) != 1:
            raise ValueError("hybrid_index, log_gl0, log_gl1, and log_gl2 must have the same shape")
        if len(self.hybrid_index) == 0:
            raise ValueError("Need at least one individual")
        if np.any((self.hybrid_index < 0) | (self.hybrid_index > 1)):
            raise ValueError("hybrid_index must be in [0, 1] for every individual")

    def __len__(self) -> int:
        return len(self.hybrid_index)
