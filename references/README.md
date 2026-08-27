# References

Primary-source PDFs for papers cited in this project's design decisions,
docstrings, and `README.md`'s roadmap. Drop PDFs here as they're read
directly, so a future session (or a future you) can go to the actual source
rather than a secondhand summary of it.

## Read directly, in full, and cited elsewhere in this project

- Baird, S. J. E. & Daley, N. (2025). "The Shapes of Clines and Wavefronts."
  *Molecular Ecology* 34:e70109. doi:10.1111/mec.70109
- Gompert, Z., Parchman, T. L. & Buerkle, C. A. (2012). "Genomics of
  isolation in hybrids." *Phil Trans R Soc B* 367:439-450.
  doi:10.1098/rstb.2011.0196
- Bailey, R. I. (2024). "Bayesian hybrid index and genomic cline estimation
  with the R package gghybrid." *Mol Ecol Resour* 24:e13910.
  doi:10.1111/1755-0998.13910
- Jofre, G. I. & Rosenthal, G. G. (2021). "A narrow window for geographic
  cline analysis using genomic data." *Mol Ecol Resour* 21:2278-2287.
  doi:10.1111/1755-0998.13428
- Gompert, Z. & Buerkle, C. A. (2011). "Bayesian estimation of genomic
  clines." *Mol Ecol* 20:2111-2127. doi:10.1111/j.1365-294X.2011.05074.x
- Fitzpatrick, B. M. (2013). "Alternative forms for genomic clines."
  *Ecology and Evolution* 3(7):1951-1966. doi:10.1002/ece3.609. The
  original derivation of the logit-logistic model HyCline's genomic-cline
  code implements (`hycline/models/genomic_cline.py`): `logit(phi(h)) = u +
  v*logit(h)`, derived from Bazykin's (1969) diffusion-model logistic cline
  applied to genome-wide ancestry S rather than geographic position.
  Confirms HyCline's `u`/`v` sign convention matches Firneno et al.
  (2023)'s reparameterization (both non-negated), a sign flip from
  Fitzpatrick's own original `u_i = b_i(m_i - m)` -- a deliberate,
  already-consistent convention choice, not a divergence to fix. Also
  directly relevant to the adaptive-introgression flagging feature (see
  `README.md` Tier 1): Fitzpatrick's own simulations found the logit-
  logistic model has the best precision/sensitivity balance for outlier
  detection among the models compared (precision 0.73, sensitivity 0.19,
  vs. the Barton/bgc cline's 0.55/0.08 -- lowest sensitivity of all models
  tested) but is not usually the best *goodness-of-fit* model, and that
  naive null-hypothesis testing (p_i = S, ignoring drift) is badly
  miscalibrated (flags 99-100% of neutral loci as "significant" purely
  from drift) -- direct support for ranking candidates by delta-AIC/
  delta-WAIC against a fixed neutral null rather than by raw p-value, as
  already planned. Also independently reinforces the documented epistatic-
  incompatibility blind spot: Dobzhansky-Muller incompatibilities were the
  one selection type in Fitzpatrick's simulations that left neutral
  background loci undistorted (no hitchhiking effect), unlike
  environmental/heterozygote selection.
- Firneno, T. J. Jr, Semenov, G., Dopman, E. B., Taylor, S. A., Larson,
  E. L. & Gompert, Z. (2023). "Quantitative Analyses of Coupling in Hybrid
  Zones." *Cold Spring Harb Perspect Biol*, doi:10.1101/cshperspect.a041434
  -- journal/year now confirmed directly from the paper's own header
  (previously only this project's own past sessions had worked from
  `Supplemental_Material.pdf`, not the main text). Underlies HyCline's
  `--pool-center` reparameterization (`logit(center) = u/v`, confirmed
  here as *exactly* the paper's own "following Bailey (2022)"
  reparameterization) and `coupling-genomic`'s theta calibration
  (`hycline/analysis/coupling.py::FIRNENO_2023_TABLE1`, confirmed against
  the paper's own Table 1 digit-for-digit during this read). One caveat
  from the main text, since folded into HyCline's own coupling-module
  docs (`hycline/analysis/coupling.py`'s theta-calibration docstring and
  `HyCline_UserGuide_Coupling.md`'s `coupling-genomic` section): the
  paper's own robustness checks found the sigma-to-theta relationship is
  sensitive to the migration rate and genetic map size assumed in the
  calibration simulations -- a model fit at one migration rate had ~0%
  out-of-bag explanatory power when applied to data simulated at a lower
  migration rate, and the linear model's R^2 dropped from 0.523 to 0.415
  under a lower-gene-flow regime; genetic map size showed a smaller
  version of the same sensitivity (37% vs. 52.3% variance explained).

## Read directly, newly added, not yet reflected in this project's design docs

- Payseur, B. A. (2010). "Using differential introgression in hybrid zones
  to identify genomic regions involved in speciation." *Mol Ecol Resour*
  10:806-820. doi:10.1111/j.1755-0998.2010.02883.x. A review, not a new
  method -- but directly relevant to the adaptive-introgression flagging
  feature (`README.md` Tier 1), and raises confounds not yet listed in that
  feature's spec:
  - **Incomplete lineage sorting (ILS)** produces the same signature as
    recent gene flow at individual loci, and is a bigger confound than
    drift alone for species with recent divergence times or large
    effective population sizes -- worth naming explicitly alongside drift
    in the feature's caveat text, not left implicit under "drift."
  - **Marker ascertainment bias**: using fixed-difference (diagnostic)
    markers -- exactly what `filter-aims` selects for -- systematically
    biases toward *underestimating* introgression, and, paradoxically,
    genomic regions that actually carry reproductive-isolation genes are
    *more* likely to have fixed differences, making it specifically harder
    to detect reduced introgression there by comparison to other loci.
    Worth a caveat on `filter-aims`/the flagging feature's own docs.
  - **Recombination-rate heterogeneity** across the genome means loci with
    identical selective effects will distort their genomic neighbors to
    different degrees depending on local recombination rate -- ameliorated
    by denser marker coverage but not eliminated at realistic densities.
  - **General outlier-detection caveat from the wider genome-scan
    literature** (Kelley et al. 2006; Teshima et al. 2006, cited therein):
    misspecified demographic assumptions produce false positives, and
    truly selected loci need not lie in the most extreme tail of the
    empirical distribution -- i.e. this class of method has a real false-
    negative risk too, not just the false-positive risk already documented.
  - Independently reinforces the epistatic-incompatibility blind spot
    already documented (Gompert & Buerkle 2011's 0.81-0.95 FDR): "epistatic
    selection does not always cause strong distortions in genotypic
    frequencies... existing analytical approaches are not specifically
    designed to find it" (citing Gompert & Buerkle 2009), and empirically,
    initial genomic-cline studies found little evidence of epistatic
    selection compared to plain underdominance -- consistent with epistasis
    being systematically under-detected rather than genuinely rare.
  - Also documents the full HZAR-style tripartite geographic cline model
    (its equations 4-6, matching `hycline.models.cline.hzar_cline_function`)
    and a reproductive-barrier-strength formula not currently implemented
    in HyCline, B = delta-p_center / (dp/dx)_edge (eq. 7, Nagylaki 1976) --
    a candidate for a future Tier 2/3 addition, not scoped as part of
    either current Tier 1 item.
