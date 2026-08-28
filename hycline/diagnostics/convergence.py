"""Convergence diagnostics for MCMC fits, wrapping ArviZ.

See BACKEND_PHASE2.md section 3 for what these diagnostics mean specifically for
cline fits (as opposed to the generic definitions).
"""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np

# Thresholds used to flag problems in `check()`. These are the commonly
# cited defaults (Vehtari et al. 2021 for R-hat; a few hundred draws as a
# rough ESS floor for a model this simple), not hard theoretical cutoffs --
# treat `check()`'s output as "worth a second look", not gospel.
RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400


@dataclass
class DiagnosticsReport:
    rhat_max: float
    ess_bulk_min: float
    ess_tail_min: float
    n_divergences: int
    n_chains: int
    n_draws_per_chain: int
    ok: bool
    warnings: list

    def summary(self) -> str:
        lines = [
            f"Convergence diagnostics ({self.n_chains} chains x "
            f"{self.n_draws_per_chain} draws):",
            f"  max R-hat        = {self.rhat_max:.4f}  (want < {RHAT_THRESHOLD})",
            f"  min ESS (bulk)   = {self.ess_bulk_min:.0f}  (want > {ESS_THRESHOLD})",
            f"  min ESS (tail)   = {self.ess_tail_min:.0f}  (want > {ESS_THRESHOLD})",
            f"  divergences      = {self.n_divergences}",
            f"  overall: {'OK' if self.ok else 'ISSUES FLAGGED'}",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def check(idata: az.InferenceData) -> DiagnosticsReport:
    """Run standard convergence checks on a fitted InferenceData object."""
    summary_df = az.summary(idata)
    rhat_max = float(summary_df["r_hat"].max())
    ess_bulk_min = float(summary_df["ess_bulk"].min())
    ess_tail_min = float(summary_df["ess_tail"].min())

    n_divergences = 0
    if "diverging" in idata.get("sample_stats", {}):
        n_divergences = int(np.sum(idata.sample_stats["diverging"].values))

    n_chains = idata.posterior.sizes.get("chain", 1)
    n_draws = idata.posterior.sizes.get("draw", 0)

    warnings = []
    if np.isnan(rhat_max):
        warnings.append(
            "R-hat is NaN -- most commonly because num_chains=1, which makes the "
            "cross-chain comparison R-hat measures literally uncomputable (not "
            "evidence of good convergence). Use num_chains>=2 (the CLI default is "
            "4) to get a real R-hat check."
        )
    elif rhat_max >= RHAT_THRESHOLD:
        warnings.append(
            f"R-hat = {rhat_max:.4f} >= {RHAT_THRESHOLD}. See BACKEND_PHASE2.md section 3: "
            "usually means either the data poorly constrains the cline, or the "
            "width prior's upper bound is too loose relative to the data."
        )
    if ess_bulk_min < ESS_THRESHOLD or ess_tail_min < ESS_THRESHOLD:
        warnings.append(
            f"Low effective sample size (bulk={ess_bulk_min:.0f}, "
            f"tail={ess_tail_min:.0f}). Often the `width` parameter specifically "
            "when sampling near the transition zone is sparse -- consider more "
            "samples or a tighter/better-informed width prior."
        )
    if n_divergences > 0:
        warnings.append(
            f"{n_divergences} divergent transitions. A handful on this model "
            "usually indicates the width prior's boundary is being pushed against; "
            "many divergences suggest a real geometry problem worth investigating."
        )

    ok = len(warnings) == 0
    return DiagnosticsReport(
        rhat_max=rhat_max, ess_bulk_min=ess_bulk_min, ess_tail_min=ess_tail_min,
        n_divergences=n_divergences, n_chains=n_chains, n_draws_per_chain=n_draws,
        ok=ok, warnings=warnings,
    )
