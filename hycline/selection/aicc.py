"""Model comparison across nested cline models via AICc.

Mirrors HZAR's `autofit` behaviour: fit a set of candidate models (e.g.
varying tail configuration) and rank them by small-sample-corrected AIC,
since hybrid-zone transects often have few sampling sites relative to the
number of parameters in the more flexible tail models.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hycline.data.schema import AlleleFreqTransect, TraitTransect
from hycline.inference.mle import FitResult, fit_mle
from hycline.inference.mle_trait import TraitFitResult, fit_mle_trait
from hycline.models.cline import TailType


@dataclass
class ModelComparisonResult:
    results: dict[TailType, FitResult]
    table: pd.DataFrame
    best_model: TailType

    def summary(self) -> str:
        lines = [self.table.to_string(index=False),
                 f"\nBest model by AICc: {self.best_model.value}"]
        return "\n".join(lines)


def compare_models(transect: AlleleFreqTransect,
                    candidates: list[TailType] | None = None) -> ModelComparisonResult:
    """Fit each candidate tail model and rank by AICc.

    Parameters
    ----------
    transect : the data to fit
    candidates : which TailType variants to compare; defaults to the
        full standard set (none, mirror, both).
    """
    if candidates is None:
        candidates = [TailType.NONE, TailType.MIRROR, TailType.BOTH]

    results: dict[TailType, FitResult] = {}
    for tail_type in candidates:
        results[tail_type] = fit_mle(transect, tail_type)

    rows = []
    aiccs = {tt: r.aicc for tt, r in results.items()}
    min_aicc = min(aiccs.values())
    for tt, r in results.items():
        rows.append({
            "model": tt.value,
            "k": r.n_params,
            "logL": r.log_likelihood,
            "AICc": r.aicc,
            "delta_AICc": r.aicc - min_aicc,
            "converged": r.success,
        })
    table = pd.DataFrame(rows).sort_values("AICc").reset_index(drop=True)
    best_model = TailType(table.iloc[0]["model"])

    return ModelComparisonResult(results=results, table=table, best_model=best_model)


def compare_trait_models(transect: TraitTransect,
                          candidates: list[TailType] | None = None) -> ModelComparisonResult:
    """The trait-cline analog of `compare_models`: fit each candidate
    tail model to quantitative trait data (`fit_mle_trait`, full HZAR
    parity) and rank by AICc.
    """
    if candidates is None:
        candidates = [TailType.NONE, TailType.MIRROR, TailType.BOTH]

    results: dict[TailType, TraitFitResult] = {}
    for tail_type in candidates:
        results[tail_type] = fit_mle_trait(transect, tail_type)

    rows = []
    aiccs = {tt: r.aicc for tt, r in results.items()}
    min_aicc = min(aiccs.values())
    for tt, r in results.items():
        rows.append({
            "model": tt.value,
            "k": r.n_params,
            "logL": r.log_likelihood,
            "AICc": r.aicc,
            "delta_AICc": r.aicc - min_aicc,
            "converged": r.success,
        })
    table = pd.DataFrame(rows).sort_values("AICc").reset_index(drop=True)
    best_model = TailType(table.iloc[0]["model"])

    return ModelComparisonResult(results=results, table=table, best_model=best_model)
