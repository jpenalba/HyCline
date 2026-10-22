"""Build benchmarking/HyCline_Tool_Comparison.pdf: HyCline against gghybrid
and bgchm on empirical data (Italian sparrows) and on simulated hybrid
zones under several designs. Reads the archived scores under
benchmarking/tool_comparison/ (requires reportlab and matplotlib).

Usage: python benchmarking/make_comparison_report_pdf.py benchmarking/HyCline_Tool_Comparison.pdf
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem,  # noqa: E402
                                PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE / "tool_comparison"
OUT = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "HyCline_Tool_Comparison.pdf")
FIG = Path(tempfile.mkdtemp())

# Categorical palette (dataviz reference instance, first four slots, fixed
# order; validated: CVD and normal-vision separation pass, contrast relief
# given by the tables beside every chart).
TOOL_COLOR = {"HyCline": "#2a78d6", "HyCline hierarchical": "#2a78d6",
              "bgchm": "#eb6834", "gghybrid": "#1baf7a", "HyCline MLE": "#eda100"}
HI_TOOLS = ["HyCline", "bgchm", "gghybrid"]
CL_TOOLS = ["HyCline hierarchical", "bgchm", "gghybrid", "HyCline MLE"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

DESIGNS = [  # (folder, scenario, label, what changes)
    ("baseline", "genotypes", "Baseline", "20 + 20 parentals, 100 hybrids (uniform h), 100 loci "
     "differing by >= 0.6, true genotypes"),
    ("baseline", "lowdepth", "Low depth", "as baseline, reads at mean depth 3 (HyCline and bgchm "
     "on genotype likelihoods, gghybrid on called genotypes)"),
    ("weak_differentiation", "genotypes", "Weak differentiation",
     "parental frequencies differ by >= 0.3"),
    ("population_panels", "genotypes", "Sampled panels",
     "as baseline, but panels and hybrids are drawn from population frequencies, so the "
     "20 + 20 reference panels only estimate them (reference for Small panels)"),
    ("small_panels", "genotypes", "Small panels",
     "as Sampled panels, with 5 + 5 parental individuals"),
    ("missing_data", "genotypes", "Missing data", "20% of genotypes missing at random"),
    ("bimodal_hybrids", "genotypes", "Bimodal hybrids",
     "hybrid indices ~ Beta(0.5, 0.5): mostly near the parentals"),
    ("skewed_hybrids", "genotypes", "Skewed hybrids",
     "hybrid indices ~ Beta(1.5, 6): mostly backcrosses toward pop1"),
]

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=14, spaceAfter=6,
                    textColor=colors.HexColor("#1a365d"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4,
                    textColor=colors.HexColor("#2c5282"))
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.5, leading=12.5,
                      alignment=TA_LEFT, spaceAfter=5)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.2, leading=10.2, spaceAfter=0)
SMALLB = ParagraphStyle("SmallB", parent=SMALL, fontName="Helvetica-Bold")
CAP = ParagraphStyle("Cap", parent=BODY, fontSize=8.5, leading=11,
                     textColor=colors.HexColor("#52514e"), spaceAfter=8)
NOTE = ParagraphStyle("Note", parent=BODY, fontSize=9, leading=12,
                      backColor=colors.HexColor("#f0f4f8"), borderPadding=6, spaceBefore=4,
                      spaceAfter=8)
story = []


def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def p(t): story.append(Paragraph(t, BODY))
def note(t): story.append(Paragraph(t, NOTE))


def bullets(items):
    story.append(ListFlowable([ListItem(Paragraph(i, BODY), leftIndent=12) for i in items],
                              bulletType="bullet", start="•", leftIndent=12,
                              bulletFontSize=8, spaceAfter=4))


def table(rows, widths, bold_first_col=False):
    data = [[Paragraph(str(c), SMALLB if (r == 0 or (bold_first_col and k == 0)) else SMALL)
             for k, c in enumerate(row)] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#a0aec0")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story.append(t)
    story.append(Spacer(1, 6))


def figure(path, caption, width=16.5):
    img = Image(str(path))
    img.drawWidth, img.drawHeight = width * cm, width * cm * img.imageHeight / img.imageWidth
    story.append(KeepTogether([img, Paragraph(caption, CAP)]))


def f(x, d=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def style_axes(ax, ylabel):
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8.5)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b9b8b3")


def grouped_bars(df, metric, tools, ylabel, path, ref=None, ylim=None):
    """Bars grouped by design, one colour per tool (fixed order), 2 px gaps."""
    labels = [lab for _, _, lab, _ in DESIGNS if lab in df.index.get_level_values(0)]
    n = len(tools)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(8.6, 3.0), dpi=200)
    x = np.arange(len(labels))
    for i, tool in enumerate(tools):
        vals = [df.loc[(lab, tool), metric] if (lab, tool) in df.index else np.nan
                for lab in labels]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width * 0.9, color=TOOL_COLOR[tool],
               label=tool, edgecolor="white", linewidth=1.0)
    if ref is not None:
        ax.axhline(ref, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, color=INK)
    if ylim:
        ax.set_ylim(*ylim)
    style_axes(ax, ylabel)
    ax.legend(frameon=False, fontsize=8, ncol=n, loc="upper left", bbox_to_anchor=(0, 1.16),
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Load archived scores
hi_parts, cl_parts, settings = [], [], {}
for folder in sorted({d[0] for d in DESIGNS}):
    d = ROOT / folder
    if not (d / "hybrid_index_scores.tsv").exists():
        continue
    settings[folder] = json.loads((d / "design.json").read_text())
    hi_parts.append(pd.read_csv(d / "hybrid_index_scores.tsv", sep="\t").assign(folder=folder))
    cl_parts.append(pd.read_csv(d / "cline_scores.tsv", sep="\t").assign(folder=folder))
hi = pd.concat(hi_parts)
cl = pd.concat(cl_parts)
label_of = {(fo, sc): lab for fo, sc, lab, _ in DESIGNS}
for df in (hi, cl):
    df["label"] = [label_of.get((fo, sc)) for fo, sc in zip(df["folder"], df["scenario"])]
hi = hi.dropna(subset=["label"])
cl = cl.dropna(subset=["label"])
hi_m = hi.groupby(["label", "tool"]).mean(numeric_only=True)
cl_m = cl.groupby(["label", "tool"]).mean(numeric_only=True)
n_seeds = hi.groupby("label")["seed"].nunique()
present = [d for d in DESIGNS if d[2] in hi_m.index.get_level_values(0)]

italy = ROOT / "italy_sparrows"
it_hi = pd.read_csv(italy / "italy_hybrid_index_scores.tsv", sep="\t")
it_cl = pd.read_csv(italy / "italy_cline_scores.tsv", sep="\t")
it_t = json.loads((italy / "times.json").read_text())

# Figures
grouped_bars(hi_m, "mae", HI_TOOLS, "Mean absolute error of h", FIG / "hi_mae.png")
grouped_bars(hi_m, "coverage95", HI_TOOLS, "95% interval coverage", FIG / "hi_cov.png",
             ref=0.95, ylim=(0, 1.0))
grouped_bars(cl_m, "mdae_log_v", CL_TOOLS, "Median |error| of log v", FIG / "cl_logv.png")
grouped_bars(cl_m, "mdae_logit_c", CL_TOOLS, "Median |error| of logit c", FIG / "cl_c.png")
grouped_bars(cl_m, "v_power", CL_TOOLS, "v outliers detected (power)", FIG / "cl_vpow.png",
             ylim=(0, 1))
grouped_bars(cl_m, "v_false_pos", CL_TOOLS, "Neutral loci flagged for v", FIG / "cl_vfp.png",
             ref=0.05)
grouped_bars(cl_m, "c_false_pos", CL_TOOLS, "Neutral loci flagged for centre",
             FIG / "cl_cfp.png", ref=0.05)

# ---------------------------------------------------------------------------
story.append(Paragraph("HyCline, bgchm and gghybrid compared", ss["Title"]))
story.append(Paragraph(
    "Hybrid index and genomic cline estimates on empirical and simulated hybrid zones",
    ParagraphStyle("Sub", parent=BODY, fontSize=11, alignment=1,
                   textColor=colors.HexColor("#4a5568"))))
story.append(Spacer(1, 8))

h1("Summary")
base_hi = hi_m.loc["Baseline"]
bullets([
    "<b>Hybrid index.</b> All three tools give practically the same estimates when they see the "
    "same information. On the Italian sparrows they agree with each other at r &ge; 0.9998 and with "
    "BGC at r = 0.992. In the simulations their errors are within a few thousandths of each other "
    "in every design. Where they differ is uncertainty: with low-depth reads gghybrid, which can only "
    "use called genotypes, gives intervals that are too narrow.",
    "<b>Genomic clines.</b> HyCline's hierarchical model and bgchm (also hierarchical) agree closely "
    "with each other (r = 0.99 on the sparrows) and are the most accurate in the simulations, with "
    "few false outliers. HyCline's per-locus maximum-likelihood fit is less accurate and flags more "
    "neutral loci; gghybrid's per-locus Bayesian fit has the largest errors and the lowest power to "
    "detect cline-rate outliers.",
    "<b>Speed.</b> HyCline's whole pipeline ran 3-6 times faster than bgchm and gghybrid.",
    "<b>Found and fixed along the way.</b> HyCline's hierarchical genomic cline fit refused genotype "
    "data missing at different individuals per locus; it now handles it (Section 4). One bgchm run "
    "setting was wrong in the first attempt and was corrected (Section 5).",
])

h1("1. Tools and settings")
table([["Tool", "Hybrid index", "Genomic clines", "Settings used"],
       ["HyCline", "maximum likelihood (estimate-hybrid-index), 95% likelihood-profile interval",
        "per-locus MLE (fit-genomic) and hierarchical NumPyro model (fit-hierarchical-genomic)",
        "defaults; genotype likelihoods for beagle input"],
       ["bgchm (Gompert, DeRaad &amp; Buerkle 2024)", "HMC posterior (est_hi), median and 95% "
        "quantiles", "hierarchical HMC model (est_genocl, hier = TRUE)",
        "2000 iterations; est_p analytical for genotypes, HMC for genotype likelihoods"],
       ["gghybrid (Bailey 2023)", "MCMC posterior mode (esth), Jeffreys prior",
        "per-locus MCMC (ggcline)", "esth 3000/1000 burn-in; ggcline 5000/2000"]],
      [3.2, 4.6, 4.8, 4.4])
p("The original bgc 1.03 could not be run (it is distributed only from its authors' site, which "
  "this environment could not reach); bgchm is its successor from the same authors. Input files "
  "for bgc 1.03 are saved alongside the others.")
p("All three tools model a locus's allele frequency in an individual with hybrid index h as "
  "p<sub>0</sub> + (p<sub>1</sub> - p<sub>0</sub>) &phi;(h). HyCline writes the cline as logit &phi; = "
  "u + v logit h; bgchm and gghybrid write &phi; = h<super>v</super> / (h<super>v</super> + "
  "(1 - h)<super>v</super> e<super>u'</super>) with u' = -u. Clines are compared on their centre "
  "c = invlogit(-u / v) (the h at which &phi; = 0.5) and slope v.")
p("<b>Metrics.</b> Hybrid index (test individuals only): mean absolute error against the true value, "
  "and the share of 95% intervals containing it. Clines: median absolute error of logit c and log v "
  "(medians, because a nearly flat cline has no identifiable centre); a locus is a v outlier when its "
  "95% interval for v excludes 1 and a centre outlier when its interval for c excludes 0.5. Power "
  "is measured on the 20 barrier/permeable loci (v) or 10 biased loci (centre), false positives on "
  "the 70 neutral loci. HyCline's hierarchical intervals are normal approximations from its per-locus "
  "posterior mean and SD.")

h1("2. Empirical data: Italian sparrows")
p("Hermansen et al. (2014): 77 loci, 84 house sparrow and 52 Spanish sparrow references, 433 Italian "
  "sparrows; set up as in Bailey (2023, section 2.3.1). Reference: BGC's published hybrid indices "
  "and cline parameters for the same birds and loci.")
rows = [["Comparison", "n", "Pearson r", "Mean |diff|", "Mean diff", "Inside BGC 95% CI"]]
for _, r in it_hi.iterrows():
    rows.append([r["comparison"], int(r["n"]), f(r["pearson_r"], 4), f(r["mean_abs_diff"], 4),
                 f(r["mean_diff"], 4), f(r.get("inside_bgc_ci95"), 3)])
table(rows, [5.2, 1.2, 2.2, 2.4, 2.4, 3.2])
rows = [["Comparison", "Loci", "log v (r)", "logit c (r)", "log v vs BGC beta (rho)",
         "logit c vs BGC alpha (rho)"]]
for _, r in it_cl.iterrows():
    rows.append([r["comparison"], int(r["n_loci"]), f(r.get("pearson_logv")),
                 f(r.get("pearson_logitc")), f(r.get("spearman_logv_beta")),
                 f(r.get("spearman_logitc_alpha"))])
table(rows, [5.6, 1.2, 2.0, 2.0, 3.2, 3.2])
p("BGC's cline function differs (alpha shifts the centre, beta the rate, with opposite sign "
  "conventions), so only rank agreement is meaningful there. Run times on this data set: "
  f"HyCline {it_t['HyCline']:.0f} s, gghybrid {it_t['gghybrid']:.0f} s, bgchm {it_t['bgchm']:.0f} s "
  "(the machine was shared with other runs, so these are rough).")

story.append(PageBreak())
h1("3. Simulated hybrid zones")
p("Each design changes one thing from the baseline; three replicate data sets each (seeds 1-3). "
  "Clines: 10 barrier loci (v = 3), 10 permeable (v = 0.33), 10 with a shifted centre (u = +/-1.5) "
  "and 70 neutral.")
rows = [["Design", "What changes", "Seeds"]]
for fo, sc, lab, what in present:
    rows.append([lab, what, int(n_seeds.get(lab, 0))])
table(rows, [3.4, 12.4, 1.4], bold_first_col=True)

h2("3.1 Hybrid index")
figure(FIG / "hi_mae.png", "Mean absolute error of the test individuals' hybrid indices "
       "(lower is better), mean over seeds.")
figure(FIG / "hi_cov.png", "Share of 95% intervals containing the true hybrid index; dashed line "
       "at the nominal 0.95.")
rows = [["Design", "Tool", "Mean |error|", "Bias", "r", "95% coverage", "Interval width", "Time (s)"]]
for fo, sc, lab, _ in present:
    for tool in HI_TOOLS:
        if (lab, tool) in hi_m.index:
            r = hi_m.loc[(lab, tool)]
            rows.append([lab, tool, f(r["mae"]), f(r["bias"]), f(r["r"]), f(r["coverage95"], 2),
                         f(r["width95"]), f(r["seconds"], 0)])
table(rows, [3.2, 2.2, 2.0, 1.6, 1.6, 2.2, 2.2, 1.6])

story.append(PageBreak())
h2("3.2 Genomic clines")
figure(FIG / "cl_logv.png", "Median absolute error of log v (cline slope), mean over seeds.")
figure(FIG / "cl_c.png", "Median absolute error of logit c (cline centre).")
figure(FIG / "cl_vpow.png", "Share of the 20 barrier and permeable loci detected as v outliers.")
figure(FIG / "cl_vfp.png", "Share of the 70 neutral loci wrongly flagged as v outliers; dashed "
       "line at 0.05.")
figure(FIG / "cl_cfp.png", "Share of neutral loci wrongly flagged as centre outliers.")
rows = [["Design", "Tool", "logit c |err|", "log v |err|", "r log v", "v power", "v false pos.",
         "c power", "c false pos."]]
for fo, sc, lab, _ in present:
    for tool in CL_TOOLS:
        if (lab, tool) in cl_m.index:
            r = cl_m.loc[(lab, tool)]
            rows.append([lab, tool, f(r["mdae_logit_c"]), f(r["mdae_log_v"]), f(r["r_log_v"]),
                         f(r["v_power"], 2), f(r["v_false_pos"]), f(r["c_power"], 2),
                         f(r["c_false_pos"])])
table(rows, [2.7, 3.0, 1.7, 1.6, 1.4, 1.5, 1.7, 1.5, 1.7])

h1("4. HyCline limitation found and fixed")
p("With 20% of genotypes missing, <i>fit-hierarchical-genomic</i> stopped with an error: it required "
  "every locus to be observed at the same individuals, while bgchm and gghybrid fit the data. "
  "Loci are now aligned by individual and a missing genotype enters the likelihood with zero sampled "
  "alleles, which leaves it out exactly. A regression test built from the simulated data covers it; "
  "the missing-data results in this report use the fixed version.")

h1("5. Corrections and caveats")
bullets([
    "<b>bgchm setting, corrected.</b> The first low-depth run estimated bgchm's parental allele "
    "frequencies from genotype likelihoods with its analytical shortcut (est_p, HMC = FALSE). That "
    "shrank them toward 0.5 (mean |p - 0.5| 0.29 against 0.36 in the panels) and inflated bgchm's "
    "hybrid-index error to 0.064. The runs reported here use HMC, bgchm's default for genotype "
    "likelihoods.",
    "<b>gghybrid at low depth</b> takes genotypes only, so it was given genotypes called from the reads; "
    "its narrower intervals there reflect treating those calls as certain.",
    "<b>HyCline's hierarchical intervals</b> are normal approximations from the per-locus posterior mean "
    "and SD, which is all its output gives; bgchm's and gghybrid's are posterior quantiles.",
    "<b>Run times</b> are wall-clock on 4 cores for each tool's whole pipeline. Several runs shared the "
    "machine, so compare times within a design, not across designs.",
    "<b>Scope.</b> 100 loci and 100 hybrids per data set, three seeds per design; small differences "
    "(a few thousandths in hybrid-index error, a few percentage points in power) are within "
    "replicate variation.",
])

h1("6. Files")
p("Everything needed to rerun or rescore is in <i>benchmarking/tool_comparison/</i>: one folder per "
  "design with, for each data set, the simulation truth (<i>sim/</i>), the inputs and run scripts for "
  "HyCline, gghybrid, bgchm and bgc 1.03 (<i>inputs/</i>), the scored outputs and the timings. "
  "<i>python benchmarking/compare_tools.py --score-only --reps 3 --outdir "
  "benchmarking/tool_comparison/&lt;design&gt;</i> reproduces the scores; "
  "<i>benchmarking/italy_sparrow_tools.py</i> reruns the empirical comparison from a gghybrid "
  "checkout.")

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                        title="HyCline, bgchm and gghybrid compared", author="HyCline")
doc.build(story)
print(f"wrote {OUT}")
