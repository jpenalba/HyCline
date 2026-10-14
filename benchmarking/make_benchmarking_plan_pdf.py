"""Build benchmarking/HyCline_Benchmarking_Plan.pdf (requires reportlab).

Usage: python benchmarking/make_benchmarking_plan_pdf.py benchmarking/HyCline_Benchmarking_Plan.pdf
"""
import sys
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (KeepTogether, ListFlowable, ListItem, PageBreak,
                                Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

OUT = sys.argv[1]
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=14, spaceAfter=6,
                    textColor=colors.HexColor("#1a365d"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4,
                    textColor=colors.HexColor("#2c5282"))
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10.5, spaceBefore=6, spaceAfter=2)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.5, leading=12.5, alignment=TA_LEFT,
                      spaceAfter=5)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.3, leading=10.3, spaceAfter=0)
SMALLB = ParagraphStyle("SmallB", parent=SMALL, fontName="Helvetica-Bold")
NOTE = ParagraphStyle("Note", parent=BODY, fontSize=9, leading=12, backColor=colors.HexColor("#f0f4f8"),
                      borderPadding=6, spaceBefore=4, spaceAfter=8)

story = []


def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def h3(t): story.append(Paragraph(t, H3))
def p(t): story.append(Paragraph(t, BODY))
def note(t): story.append(Paragraph(t, NOTE))


def bullets(items, style=BODY):
    story.append(ListFlowable([ListItem(Paragraph(i, style), leftIndent=12) for i in items],
                              bulletType="bullet", start="•", leftIndent=12,
                              bulletFontSize=8, spaceAfter=4))


def table(rows, widths, header=True):
    data = [[Paragraph(c, SMALLB if (header and r == 0) else SMALL) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#a0aec0")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")))
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 6))


def step(title, what, checks):
    """One pipeline step: a short description, then a table of benchmarks."""
    h2(title)
    p(what)
    table([["Benchmark type", "What to do", "Pass criterion / what it tells you"]] + checks,
          [3.0, 7.6, 6.4])


# ---------------------------------------------------------------------------
story.append(Paragraph("HyCline Benchmarking Plan", ss["Title"]))
story.append(Paragraph("Independent ways to check that every step of the pipeline works", ParagraphStyle(
    "Sub", parent=BODY, fontSize=11, alignment=1, textColor=colors.HexColor("#4a5568"))))
story.append(Spacer(1, 10))

h1("1. Purpose and principles")
p("HyCline's unit tests (469 at the time of writing) check that the code does what it was written to do. "
  "Benchmarks answer a different question: <b>are the results right?</b> A result is trustworthy when "
  "several benchmarks that could fail for <i>different</i> reasons all agree. This plan lists, for each "
  "step of HyCline, the independent kinds of evidence available and what counts as passing.")
p("Six kinds of benchmark are used throughout. They are deliberately different, because each one has a "
  "blind spot the others cover:")
table([
    ["Type", "What it is", "What it catches", "Blind spot"],
    ["A. Exact / hand-computed", "Tiny inputs whose answer can be worked out by hand or by brute force "
     "(e.g. a 20,001-point likelihood grid).", "Arithmetic, indexing, sign and off-by-one errors.",
     "Says nothing about realistic data or statistical behaviour."],
    ["B. Invariance / consistency", "Transformations that must not change the answer: allele polarity "
     "flips, reordering samples or loci, VCF vs beagle at high depth, hard calls vs one-hot genotype "
     "likelihoods, missing data vs dropping it, MLE vs flat-prior Bayesian.", "Hidden assumptions "
     "(the allele-orientation bugs found so far were exactly this kind).", "A consistently wrong method "
     "passes."],
    ["C. Simulation with known truth", "Generate data from known parameters and check recovery: "
     "bias, error, interval coverage, false-positive and detection rates over many replicates.",
     "Statistical bias, mis-calibrated intervals, sensitivity to depth, missing data and sample size.",
     "Circular if the simulator makes the same assumptions as the model."],
    ["D. Other software", "Run established tools on identical input and compare (HZAR, gghybrid, "
     "bgc/bgchm, ANGSD, bcftools/vcftools/PLINK, ADMIXTURE/NGSadmix, introgress, Clinalyse).",
     "Model or implementation differences; anchors HyCline to accepted practice.",
     "Tools can share the same simplification; differences need explaining, not just measuring."],
    ["E. Published empirical results", "Reproduce published analyses on their real data (Manacus "
     "manakin data bundled with HZAR; the Italian sparrow data bundled with gghybrid; Firneno et al. "
     "2023 Table 1).", "Behaviour on real, messy data.", "Truth unknown; agreement only shows "
     "consistency with the original analysis."],
    ["F. Robustness, scale and performance", "Edge cases (tiny panels, all-missing loci, monomorphic "
     "loci, extreme depth), large inputs, runtime and memory.", "Crashes, silent failures, "
     "quadratic slow-downs (one was found in estimate-hybrid-index).", "Says nothing about accuracy."],
], [2.8, 5.2, 4.6, 4.4])
note("<b>Independence rule.</b> A benchmark is only independent if the truth it checks against was not "
     "produced by the code under test or by a simulator built on exactly the same model. "
     "<font face='Courier'>benchmarking/simulate_vcf.py</font> is valuable but shares HyCline's "
     "assumptions (independent loci, Hardy-Weinberg proportions, hybrids drawn from the reference panel's "
     "exact sample frequencies), so every simulation result should be paired with a type B, D or E "
     "benchmark, and with an independent forward simulator (Section 4) where possible.")

h1("2. HyCline's steps and functions")
p("The pipeline, and the parts that sit beside it:")
table([
    ["Step / function", "Commands", "Main output"],
    ["1. AIM filtering", "filter-aims", "Filtered VCF/beagle + summary (AIM count, missing data, warnings)"],
    ["2. Hybrid index", "estimate-hybrid-index (--method ml / mean)", "Per-individual h with 95% interval"],
    ["3. Input conversion", "generate-geographic-input, generate-genomic-input",
     "Site-level counts or per-individual genotypes/likelihoods (+ parental frequencies for genomic)"],
    ["4. Geographic clines", "fit-geographic (MLE and Bayesian), compare-tails",
     "Centre, width, pmin/pmax, tail parameters, AICc"],
    ["5. Trait clines", "fit-trait, fit-trait-multi", "Trait cline parameters (HZAR trait model)"],
    ["6. Genomic clines", "fit-genomic", "Per-locus u, v (allele frequency p0 + (p1 - p0) phi(h))"],
    ["7. Hierarchical clines", "fit-hierarchical-geographic, fit-hierarchical-genomic",
     "Pooled per-locus estimates + across-locus means and SDs"],
    ["8. Coupling / concordance", "coupling-geographic, coupling-genomic", "Cline variance, theta"],
    ["9. Genotype-likelihood machinery", "beagle input throughout; GL-aware likelihoods",
     "Same outputs, marginalizing genotype uncertainty"],
    ["10. Whole pipeline", "all of the above chained", "End-to-end results"],
], [3.6, 6.6, 6.8])

h1("3. Benchmarks for each step")

step("Step 1. AIM filtering (filter-aims)",
     "Selects loci whose parental frequencies meet --minor-allele-freq or --allele-freq-diff, and writes a "
     "summary with missing-data statistics and warnings. VCF frequencies are sample frequencies; beagle "
     "frequencies are pooled maximum-likelihood (EM) estimates.",
     [["A. Exact", "Small hand-written VCF/beagle files with known frequencies at, just inside and just "
       "outside each threshold (including floating-point boundary cases such as 0.57 vs 0.07 for D = 0.5).",
       "Exactly the expected loci kept (already partly in tests/test_filter_aims.py)."],
      ["B. Invariance", "Swap REF/ALT on half the loci; swap --pop1-label/--pop2-label; shuffle sample "
       "order.", "Identical retained set."],
      ["C. Simulation", "simulate_vcf.py with --aim-cutoff or --aim-freq-diff, varying panel size "
       "(5-50), missing data (0-40%, global and per-individual), and depth for beagle (1-30x). Many seeds "
       "per setting.", "VCF: retained set equals the simulated AIM set exactly. Beagle: report recall and "
       "precision by depth (current: 180-298 of 300 AIMs recovered from 1x to 30x, no false AIMs). "
       "Summary missing-data figures equal the simulator's metadata exactly."],
      ["D. Other software", "Compute per-population allele frequencies with bcftools (+fill-tags with "
       "population groups), vcftools --freq --keep, or PLINK --freq --within, and apply the same rule. "
       "For beagle, run ANGSD -doMaf per population. Compare per-sample missingness with vcftools "
       "--missing-indv or bcftools stats.", "VCF: identical AIM set and identical frequencies. Beagle: "
       "frequency correlation > 0.99 with ANGSD -doMaf at moderate depth; explain any differences."],
      ["E. Empirical", "A real dataset with known diagnostic markers (e.g. a published AIM panel) run "
       "through filter-aims.", "Published diagnostic markers are retained."],
      ["F. Robustness", "Monomorphic loci, loci missing in a whole population, multiallelic sites/indels, "
       "100k-1M SNPs.", "Correct skips and counts; runtime and memory recorded."]])

step("Step 2. Hybrid index (estimate-hybrid-index)",
     "Default --method ml maximizes, per individual, the likelihood of its genotypes given "
     "p(h) = (1 - h) p0 + h p1 and Hardy-Weinberg proportions (the gghybrid likelihood without its prior), "
     "with a 95% likelihood-profile interval. --method mean averages oriented dosages.",
     [["A. Exact", "Brute-force likelihood grid on small inputs; the worked three-locus example in the "
       "Genomic guide (h = 0.576, interval [0.089, 0.964]).", "Estimate and interval ends within 1e-4 of "
       "the grid (already in tests/test_hybrid_index.py)."],
      ["B. Invariance", "Flip allele polarity at random loci (with p0/p1 flipped); reorder loci; VCF vs "
       "beagle at 30x; missing genotypes vs dropping them.", "Identical h (to ~1e-8)."],
      ["C. Simulation", "simulate_vcf.py across depth (1-30x), AIM fraction (5-100%), AIM strength, "
       "missing data, panel size, and hybrid-index distributions (uniform, beta, fixed). Use the true h "
       "in the metadata CSV.", "Mean absolute error and bias by setting; 95% interval coverage 93-97% "
       "over enough replicates; no compression toward 0.5. Current: error 0.005-0.016 (ml) vs 0.010-0.106 "
       "(mean); coverage 0.90-0.97 on 40 hybrids per setting."],
      ["C. Simulation (structure)", "Pedigree-based hybrids: F1, F2 and backcrosses built from parental "
       "genotypes (not independent draws per locus).", "F1s at h ~ 0.5, backcrosses at ~0.25/0.75; checks "
       "the independence assumption the simulator currently shares with the model."],
      ["D. Other software", "gghybrid esth with a uniform Beta(1,1) prior on the same loci and parental "
       "samples; bgc / bgchm hybrid index; ADMIXTURE (or NGSadmix for genotype likelihoods) with K = 2; "
       "introgress.", "gghybrid posterior mode should match closely (same likelihood); ADMIXTURE/NGSadmix "
       "Q should correlate > 0.98 (different model: admixture proportion, not hybrid index)."],
      ["E. Empirical", "Italian sparrow data bundled with gghybrid (Hermansen et al. 2014); compare with "
       "gghybrid's published hybrid indices.", "Close agreement; parental samples near 0/1."],
      ["F. Robustness / scale", "Individuals with very few called loci; 100k-1M loci; low depth.",
       "Clear errors for unusable individuals; record runtime (currently ~49 s for 100k SNPs x 160 "
       "samples). Note: with many weakly differentiated loci intervals become overconfident (coverage 57% "
       "on 100k SNPs, 94% on its AIMs) - keep this as a standing benchmark."]])

step("Step 3. Input conversion (generate-geographic-input, generate-genomic-input)",
     "Converts VCF/beagle + metadata into HyCline CSVs. The genomic converter also writes each locus's "
     "parental frequencies (p0/p1, clipped to [0.001, 0.999]) and leaves reference individuals out by default.",
     [["A. Exact", "Small files where site aggregates, genotype frequencies, log genotype likelihoods and "
       "p0/p1 can be computed by hand.", "Exact equality."],
      ["B. Invariance", "Reorder samples in the VCF vs the metadata; gzip vs plain; extra FORMAT fields.",
       "Identical CSV (up to row order)."],
      ["D. Other software", "bcftools query -f '[%GT\\t]' and a short script to aggregate per site; "
       "compare p0/p1 with bcftools/PLINK per-population frequencies (VCF) or ANGSD -doMaf (beagle).",
       "Exact agreement for VCF; close agreement for beagle EM frequencies."],
      ["F. Robustness", "Samples missing from metadata, loci with no data, equal parental frequencies, "
       "Mode 1 vs Mode 2 parental frequencies.", "Correct drops and counts, clear messages."]])

step("Step 4. Geographic clines (fit-geographic, compare-tails)",
     "Fits sigmoid clines (centre, width, pmin/pmax), with optional HZAR tail models, by MLE or Bayesian "
     "NUTS, from hard counts or genotype likelihoods.",
     [["A. Exact", "Likelihood values at fixed parameters vs hand computation; HZAR's cline equations "
       "evaluated directly.", "Equality to rounding."],
      ["B. Consistency", "MLE vs NUTS with flat priors on well-informed data; hard counts vs one-hot "
       "genotype likelihoods; site order.", "Agreement within uncertainty (already documented: MLE matches "
       "HZAR on the manakin Ada locus)."],
      ["C. Simulation", "Needs a transect simulator (Section 4): sites along a line, per-site hybrid index "
       "from a known geographic cline, genotypes per locus with known centre/width, optional tails. Vary "
       "number of sites, individuals per site, depth.", "Bias and RMSE of centre and width; 95% interval "
       "coverage; compare-tails picks the generating tail model at a known rate."],
      ["D. Other software", "HZAR on identical data (all model variants: fixed/free ends, tails); "
       "Clinalyse for T-clines once implemented.", "Parameters and log-likelihoods match HZAR's MLE; "
       "AICc ranks agree."],
      ["E. Empirical", "Manacus manakin data bundled with HZAR: all 7 molecular loci, reproducing "
       "Derryberry et al. (2014) Tables 3-4.", "Already done for Ada (centre 171.0 vs 170.9 km); extend "
       "to all loci and tail models."],
      ["F. Robustness", "Few sites, sites only on one side, saturated frequencies, tail models under "
       "Bayesian fitting (known convergence difficulty).", "Clear warnings; convergence diagnostics "
       "flag problems."]])

step("Step 5. Trait clines (fit-trait, fit-trait-multi)",
     "HZAR's quantitative-trait cline model, including tail models, verified against HZAR's R source.",
     [["A. Exact", "Gaussian likelihood at fixed parameters vs HZAR's R functions evaluated on the same "
       "input.", "Equality to rounding."],
      ["C. Simulation", "Simulate trait values per individual from known cline mean/variance profiles "
       "(including unequal parental variances and tails).", "Recovery of centre, width and variances; "
       "interval coverage."],
      ["D. Other software", "HZAR trait models on identical data.", "Matching estimates and "
       "log-likelihoods."],
      ["E. Empirical", "Trait data analysed in Derryberry et al. (2014) or other published trait-cline "
       "studies.", "Reproduce the published parameters."]])

step("Step 6. Genomic clines (fit-genomic)",
     "Per-locus logit-logistic cline: allele frequency p0 + (p1 - p0) phi(h), "
     "logit(phi) = u + v logit(h), by MLE from hard counts or genotype likelihoods.",
     [["A. Exact", "Likelihood at fixed (u, v) vs hand computation; allele-frequency form with p0 = 0, "
       "p1 = 1 equals phi.", "Equality (in tests)."],
      ["B. Invariance", "Flip polarity at a locus (swap counts and p0/p1); include vs exclude parental "
       "references.", "Same u, v (reference exclusion should change little on large samples)."],
      ["C. Simulation, neutral", "simulate_vcf.py (every locus neutral, u = 0, v = 1) through the full "
       "pipeline.", "Median u ~ 0, v ~ 1 for every locus class; false-positive rate ~5% per parameter. "
       "Current: median u 0.02, v 1.07; u 1.5%, v 6.5% (VCF); v 9.2% at 5x beagle - investigate."],
      ["C. Simulation, selection", "Needs a simulator extension: per-locus (u, v) for hybrids "
       "(barrier loci v > 1, permeable v < 1, biased u != 0).", "Recovery of u, v; detection power vs "
       "false-positive rate; ranking of outliers."],
      ["D. Other software", "gghybrid ggcline (same model; note gghybrid's u has the opposite sign, and "
       "it reports centre with logit(centre) = u/v); bgc (different parameterisation - compare outlier "
       "sets and ranks); introgress.", "Close agreement with gghybrid; overlapping outlier sets with bgc."],
      ["E. Empirical", "Italian sparrow data with gghybrid and BGC output (77-locus comparison already "
       "done for BGC); rerun with parental frequencies.", "Agreement at least as good as before."]])

step("Step 7. Hierarchical clines (fit-hierarchical-geographic / -genomic)",
     "Partial pooling of per-locus parameters with NUTS; hard counts or genotype likelihoods.",
     [["C. Simulation", "Simulate per-locus parameters from known across-locus distributions (known "
       "mu and sigma for u and log v, or centre and log width), including all-neutral (sigma = 0).",
       "Recovery of mu and sigma; shrinkage strongest for data-poor loci; sigma near 0 when there is no "
       "real variation (current short runs gave 0.2-0.3 - check with longer chains)."],
      ["B. Consistency", "With weak pooling (large sigma priors), per-locus posteriors approach "
       "independent MLE fits.", "Agreement within uncertainty."],
      ["D. Other software", "bgchm (Gompert, DeRaad & Buerkle 2024) hierarchical genomic clines on "
       "identical data.", "Similar across-locus SDs and per-locus estimates."],
      ["F. Robustness", "Convergence diagnostics (R-hat, ESS, divergences) across seeds; runtime vs "
       "number of loci.", "Diagnostics clean on standard settings; record time."]])

step("Step 8. Coupling / concordance (coupling-geographic, coupling-genomic)",
     "Across-locus cline variance and Firneno et al. (2023)'s coupling coefficient theta.",
     [["E. Published", "Reproduce Firneno et al. (2023) Table 1 (already documented).",
       "Values match the published table."],
      ["C. Simulation", "Simulate cline sets with known across-locus variance (and, where possible, "
       "from forward simulations with known selection and dispersal - Section 4).",
       "Recovered variance and theta track the simulated values; record bias."]])

step("Step 9. Genotype-likelihood machinery (beagle input throughout)",
     "Marginalizes genotype uncertainty instead of using hard calls.",
     [["B. Consistency", "One-hot likelihoods from hard calls give exactly the hard-count likelihood; "
       "high-depth beagle gives the same results as true-genotype VCF.", "Equality / near-equality."],
      ["C. Simulation", "simulate_vcf.py --format-beagle with --vcf-genotypes called gives matched hard-call "
       "and likelihood versions of one dataset across depth, dispersion and per-individual depth.",
       "Likelihood path stays accurate as depth falls; hard calls degrade (heterozygotes called "
       "correctly 25% at 1x, 100% at 30x)."],
      ["D. Realistic upstream", "Simulate reads (wgsim or ART) from genotypes, align (bwa), and produce "
       "beagle files with ANGSD -doGlf 2 instead of HyCline's own likelihood simulator.",
       "Checks HyCline against real genotype-likelihood output rather than its own idealised model."]])

step("Step 10. Whole pipeline",
     "filter-aims -> estimate-hybrid-index -> generate-*-input -> fitting -> coupling.",
     [["C. Simulation", "End-to-end runs on simulated zones (neutral, and with barrier/permeable loci), "
       "both VCF and beagle, at several depths and panel sizes.", "Final per-locus parameters and outlier "
       "calls match the truth at the rates established step by step."],
      ["C. Forward simulation", "SLiM or msprime hybrid zones with selection, linkage and geography "
       "(Section 4).", "Qualitative truth: barrier loci steeper, linked loci concordant, neutral loci "
       "near the genome-wide average."],
      ["E. Empirical", "Manakin (geographic) and Italian sparrow (genomic) end to end.",
       "Reproduces the published conclusions."]])

h1("4. Simulators: what exists and what to add")
p("<font face='Courier'>benchmarking/simulate_vcf.py</font> currently provides: pop1/pop2/popH samples; "
  "AIMs by minor-allele-frequency cutoff or frequency difference; exact global and per-individual missing "
  "data; hybrid-index distributions; beagle genotype likelihoods from simulated reads (depth, "
  "dispersion, per-individual depth, sequencing error, called-genotype VCFs); population-frequency "
  "hybrids and reference panels (<font face='Courier'>--hybrid-freqs population</font>); and "
  "non-neutral genomic clines (barrier, permeable and biased classes among the AIMs, plus random "
  "across-locus variation in u and log v). Its truth files give AIM status, panel and source "
  "frequencies, cline class, u and v, and true hybrid indices.")
table([
    ["Extension", "Why it is needed", "Benchmarks it enables"],
    ["<b>Done:</b> hybrids drawn from population frequencies, with the reference panel a sample of them "
     "(--hybrid-freqs population)", "In the default panel mode hybrids use the panel's exact sample "
     "frequencies, so VCF benchmarks give HyCline error-free parental frequencies.",
     "Realistic accuracy for filter-aims, hybrid index and clines; the cost of small panels. First "
     "results in Section 6."],
    ["<b>Done:</b> non-neutral genomic clines (--barrier/--permeable/--biased-percent, "
     "--cline-u-sd, --cline-log-v-sd)", "Every simulated locus used to be neutral.",
     "Power, outlier detection, recovery of u and v (Step 6), hierarchical sigma (Step 7)."],
    ["Geographic transects (sites, distances, clines with tails)", "No geography at present.",
     "Steps 4 and 7 (geographic), compare-tails, coupling-geographic."],
    ["Quantitative traits", "No trait simulation.", "Step 5."],
    ["Pedigree hybrid classes (F1, F2, backcrosses)", "Independent per-locus draws can't produce F1-like "
     "genotype structure.", "Hybrid-index accuracy for real hybrid classes; comparison with NewHybrids."],
    ["Linkage between loci", "Loci are independent now.", "Effect of LD on intervals and on the "
     "many-loci overconfidence seen for hybrid index."],
    ["Read-level pipeline (wgsim/ART + bwa + ANGSD)", "HyCline's likelihood simulator matches HyCline's "
     "model by construction.", "An independent test of every beagle path (Step 9)."],
    ["Forward simulations (SLiM; msprime for neutral backgrounds)", "Parametric simulators share the "
     "fitted model's assumptions.", "Independent truth from evolutionary processes (selection against "
     "hybrids, dispersal, tension zones) - the strongest test of whether the clines mean what the "
     "documentation says they mean."],
], [4.4, 6.2, 6.4])

h1("5. Metrics and acceptance criteria")
table([
    ["Metric", "Use for", "Target / how to read it"],
    ["Bias (mean estimate - truth) and RMSE", "All parameters with simulated truth", "Bias within Monte "
     "Carlo error; RMSE shrinking as data increase."],
    ["95% interval coverage", "Hybrid-index intervals, cline-parameter CIs, Bayesian credible intervals",
     "93-97% given replicates; with 100 replicates the Monte Carlo error is about +/-2.2 points, so use "
     "several hundred where possible."],
    ["False-positive rate", "Neutral loci flagged as non-neutral; non-AIMs kept as AIMs", "Close to the "
     "nominal rate per test (5% per parameter); remember u-or-v flags two tests (~10%)."],
    ["Power / recall and precision", "Detecting AIMs, barrier/permeable loci", "Reported as curves vs "
     "depth, sample size and effect size."],
    ["Agreement with other software", "Every type D comparison", "Exact where the model is identical; "
     "otherwise correlation, Bland-Altman limits, and an explanation of systematic differences."],
    ["Convergence diagnostics", "All NUTS fits", "R-hat < 1.01, bulk/tail ESS > 400, no divergences."],
    ["Runtime and peak memory", "Every command at several sizes", "Recorded per release; no unexplained "
     "regressions."],
], [4.3, 5.9, 6.8])

h1("6. Open issues these benchmarks should track")
bullets([
    "Wald intervals for genomic-cline v are narrower than nominal at low depth (9.2% vs 5% at 5x with 60 "
    "hybrids; 7.7% even with true frequencies and hybrid index). Compare with profile-likelihood intervals.",
    "Parental frequencies are treated as exact everywhere (as in gghybrid). With many weakly differentiated "
    "loci this makes hybrid-index intervals overconfident (57% coverage on 100k SNPs). Benchmark with the "
    "population-frequency simulator extension.",
    "Hierarchical across-locus SDs were 0.2-0.3 on all-neutral data in short runs; check with longer chains.",
    "First population-frequency results (15 + 15 references, 100 hybrids, 761 AIMs): with every locus "
    "neutral, hybrid-index error rose from 0.011 (panel mode) to 0.020 and 95% interval coverage fell from "
    "0.93 to 0.68 -- the parental frequencies' sampling error, which the estimator treats as exact. "
    "filter-aims missed 13-20% of true AIMs (those near the threshold) and, with 5 references per "
    "population, 6% of what it kept were not AIMs.",
    "With 45% of AIMs non-neutral (15% each barrier v = 4, permeable v = 0.25, biased |u| = 2), the "
    "hybrid index -- whose model assumes neutral loci -- degraded (error 0.026-0.037, coverage 0.54-0.58), "
    "and fitted clines were pulled toward neutral (barrier v 3.3-3.9, permeable 0.33-0.37). Given the "
    "true hybrid index and source frequencies the same fits were accurate (barrier 4.2-4.3, permeable "
    "0.25-0.26, neutral 1.02-1.04), so the cline fitting itself is sound and the attenuation comes from "
    "the estimated inputs. Benchmark realistic outlier fractions next, and consider excluding loci "
    "flagged as outliers from the hybrid-index set in a second pass.",
    "The genomic pipeline without parental frequencies (legacy mode) still pins reversed loci at the v "
    "bound; keep the regression benchmark that documents it.",
    "tests/test_mcmc_numpyro.py::test_diagnostics_check_runs_and_flags_nothing_on_a_clean_fit fails on "
    "main in the current environment (R-hat 1.0136 vs 1.01); unrelated to the benchmarks but should be "
    "resolved.",
])

h1("7. How to organise and run the benchmarks")
bullets([
    "Keep benchmarks separate from unit tests: <font face='Courier'>benchmarking/</font> holds one script "
    "per benchmark, each writing a results table (CSV) and a short summary; unit tests keep checking code "
    "behaviour quickly.",
    "Fix and record seeds, HyCline version (git commit), and versions of every external tool (HZAR, gghybrid, "
    "bgc/bgchm, ANGSD, bcftools, PLINK, ADMIXTURE, SLiM).",
    "Run replicates (100+ per setting for coverage and false-positive rates) and report Monte Carlo error.",
    "Re-run the full benchmark set before releases and after changes to models, likelihoods or "
    "converters; compare against the previous results table.",
    "When a benchmark fails, follow the project's rule: find the root cause, add a regression test built "
    "from the data that exposed it, and record the finding in BACKEND_*.md.",
])

h1("8. Suggested order")
table([
    ["Priority", "Work", "Reason"],
    ["1", "<b>Done:</b> population-frequency hybrids and non-neutral genomic clines in simulate_vcf.py "
     "(first results in Section 6)", "Removes the main circularity and enables power/outlier benchmarks "
     "for the genomic pipeline."],
    ["2", "gghybrid comparison (esth + ggcline) on simulated and Italian sparrow data", "Same model, so "
     "differences point straight at bugs."],
    ["3", "bcftools/PLINK and ANGSD -doMaf comparisons for filter-aims and parental frequencies", "Cheap "
     "and exact for VCF."],
    ["4", "Transect simulator + HZAR comparison on all manakin loci and tail models", "Covers the "
     "geographic and trait half of HyCline."],
    ["5", "Read-level beagle pipeline (wgsim/ART + bwa + ANGSD)", "Independent check of every "
     "genotype-likelihood path."],
    ["6", "Pedigree hybrid classes, linkage, SLiM forward simulations", "Tests assumptions shared by the "
     "simulator and the models."],
    ["7", "bgc/bgchm and ADMIXTURE/NGSadmix comparisons; scale and performance suite",
     "Broader cross-software and practical limits."],
], [1.6, 8.0, 7.4])

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="HyCline Benchmarking Plan",
                        author="HyCline project")


def footer(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#718096"))
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"HyCline Benchmarking Plan - page {doc_.page}")
    canvas.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
