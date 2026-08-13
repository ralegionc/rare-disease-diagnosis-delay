"""Does any of this recover the truth?

The pipeline produces numbers whether or not it works. This module is what
distinguishes the two cases, by running the whole analysis against simulated
data where every latent quantity is known and scoring it.

Four things are checked, in increasing order of how much they can embarrass the
rest of the package:

:func:`phenotype_accuracy`
    Does the case definition find the right people? PPV and sensitivity against
    known disease status, with the rule-out coding noise in play.
:func:`onset_proxy_accuracy`
    How late is the claims-based origin, in years? This is a *measurement*
    error with a known sign, and it propagates one-for-one into delay.
:func:`delay_recovery`
    Estimated versus true median delay across the panel: rank correlation,
    absolute bias, and the decomposition of that bias into selection and
    measurement components.
:func:`multi_seed_recovery`
    All of the above across independent replicates, so a good result cannot be
    one lucky seed.

The thresholds in :data:`QUALITY_GATES` are asserted by the test suite. They
exist so that a future change which quietly breaks recovery fails CI instead of
producing a slightly different plot nobody checks.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats

from . import profiles as prof_mod
from .bias import design_comparison_panel
from .cohort import apply_phenotype, build_cohort
from .config import DAYS_PER_YEAR, DEFAULT_SIM, DEFAULT_STUDY, SimConfig, StudyConfig
from .decompose import observed_vs_true_origin
from .schema import ClaimsDataset

#: A fourth bias, found while building the quality gates rather than looked
#: for. Agreement between the estimate and the truth is strongly sample-size
#: dependent, and the sizes at which it degrades are the sizes real
#: rare-disease claims studies actually operate at.
SAMPLE_SIZE_NOTE = """
Most simulated cases never reach the analysable cohort -- they are diagnosed
outside their coverage window, or not at all within it -- so 1,000 simulated
cases per disease yields only 50-90 usable odysseys each. On the same seed,
correlation between estimated and true median delay rises from r=0.81 at 400
cases per disease to r=0.98 at 1,000. Below roughly 60 analysable patients the
per-disease median is dominated by sampling noise rather than by any property
of the method.

The practical reading: a published delay estimate for a single rare disease,
drawn from a cohort of a few dozen patients, is not precise enough to compare
against another disease or another era, however carefully it was computed.
Ranking a panel of diseases is a far more robust use of the same data than
quoting any one of their delays.
""".strip()

#: Recovery levels the test suite enforces. Deliberately modest: the point is
#: to catch regressions, not to claim the estimators are unbiased. Two of these
#: bounds describe how *badly* the retrospective design does, and tightening
#: them would mean the simulator had stopped reproducing a real problem.
QUALITY_GATES = {
    "min_phenotype_ppv": 0.90,
    # Pearson is the primary agreement gate. With six diseases, Spearman takes
    # only seven distinct values, and a single adjacent swap between two
    # diseases whose true delays sit within a few percent of each other drops
    # it by 0.06 -- which says nothing about the estimator. The rank gate is
    # kept as a coarse sanity floor rather than as the measure of record.
    "min_delay_correlation": 0.85,
    "min_delay_rank_correlation": 0.75,
    "max_median_proxy_error_years": 6.0,
    "min_retrospective_understatement": 0.15,  # fraction of true delay lost
}


def phenotype_accuracy(ds: ClaimsDataset, study: StudyConfig = DEFAULT_STUDY) -> pd.DataFrame:
    """PPV / sensitivity of each case definition against known disease status."""
    if not ds.has_truth:
        raise ValueError("phenotype_accuracy needs a simulated dataset")
    truth = ds.truth
    rows = []
    for p in prof_mod.PANEL:
        found = set(apply_phenotype(ds, p, study).ids())
        actual = set(truth.loc[truth["disease"] == p.key, "patient_id"])
        tp = len(found & actual)
        rows.append(
            {
                "disease": p.key,
                "n_flagged": len(found),
                "n_true": len(actual),
                "ppv": tp / len(found) if found else np.nan,
                "sensitivity": tp / len(actual) if actual else np.nan,
            }
        )
    return pd.DataFrame(rows)


def onset_proxy_accuracy(
    ds: ClaimsDataset, study: StudyConfig = DEFAULT_STUDY, design: str = "diagnosed"
) -> pd.DataFrame:
    frames = []
    for p in prof_mod.PANEL:
        co = build_cohort(ds, p, study, design=design)
        if co.empty:
            continue
        frames.append(observed_vs_true_origin(ds.truth, co))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def delay_recovery(ds: ClaimsDataset, study: StudyConfig = DEFAULT_STUDY) -> pd.DataFrame:
    """Estimated vs true median delay, with the bias split into its parts.

    Total error decomposes as::

        retrospective - true_overall
            = (analysed_sample_true - true_overall)   <- selection
            + (retrospective - analysed_sample_true)  <- measurement (late origin)

    Separating them matters for what you would do about it. Selection bias is
    fixable by design (follow a suspicion cohort forward). Measurement bias is
    not fixable by design at all -- it is a property of the data source, and the
    only remedies are a longer look-back or an external onset date.
    """
    panel = design_comparison_panel(ds, prof_mod.PANEL, study)
    if panel.empty:
        return panel
    panel["selection_component"] = (
        panel["true_median_in_analysed_sample"] - panel["true_median_from_onset"]
    )
    panel["measurement_component"] = (
        panel["retrospective_median"] - panel["true_median_in_analysed_sample"]
    )
    panel["understatement_fraction"] = 1.0 - (
        panel["retrospective_median"] / panel["true_median_from_onset"]
    )
    return panel


def recovery_scores(panel: pd.DataFrame) -> dict:
    ok = panel.dropna(subset=["retrospective_median", "true_median_from_onset"])
    if len(ok) < 3:
        return {"n": len(ok)}
    rho, p = stats.spearmanr(ok["retrospective_median"], ok["true_median_from_onset"])
    pear = float(np.corrcoef(ok["retrospective_median"], ok["true_median_from_onset"])[0, 1])
    return {
        "n": int(len(ok)),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "pearson_r": pear,
        "mean_abs_bias_years": float(np.mean(np.abs(ok["retrospective_bias"]))),
        "mean_understatement_fraction": float(ok["understatement_fraction"].mean()),
        "median_selection_component": float(ok["selection_component"].median()),
        "median_measurement_component": float(ok["measurement_component"].median()),
    }


def multi_seed_recovery(
    seeds=(1, 2, 3),
    sim: SimConfig = DEFAULT_SIM,
    study: StudyConfig = DEFAULT_STUDY,
    calibration: dict | None = None,
    n_cases: int = 400,
    n_controls: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replicate the whole pipeline across independent simulator seeds."""
    from .simulate import simulate_population

    panels, scores = [], []
    for s in seeds:
        cfg = replace(sim, seed=int(s), n_cases_per_disease=n_cases, n_controls=n_controls)
        ds = simulate_population(cfg, study, calibration or {})
        panel = delay_recovery(ds, study)
        panel.insert(0, "seed", s)
        panels.append(panel)
        sc = recovery_scores(panel)
        sc["seed"] = s
        scores.append(sc)
    return pd.concat(panels, ignore_index=True), pd.DataFrame(scores)


def check_gates(scores: dict, phen: pd.DataFrame, proxy: pd.DataFrame) -> list[str]:
    """Return a list of gate failures; empty means everything passed."""
    fails = []
    ppv = float(phen["ppv"].min()) if len(phen) else np.nan
    if not (ppv >= QUALITY_GATES["min_phenotype_ppv"]):
        fails.append(f"phenotype PPV {ppv:.3f} < {QUALITY_GATES['min_phenotype_ppv']}")

    r = scores.get("pearson_r", np.nan)
    if not (r >= QUALITY_GATES["min_delay_correlation"]):
        fails.append(
            f"delay correlation r={r:.3f} < {QUALITY_GATES['min_delay_correlation']}"
        )

    rho = scores.get("spearman_rho", np.nan)
    if not (rho >= QUALITY_GATES["min_delay_rank_correlation"]):
        fails.append(
            f"delay rank correlation {rho:.3f} < "
            f"{QUALITY_GATES['min_delay_rank_correlation']}"
        )

    if len(proxy):
        err = float(proxy["median_proxy_error_years"].median())
        if not (err <= QUALITY_GATES["max_median_proxy_error_years"]):
            fails.append(
                f"median onset-proxy error {err:.2f}y > "
                f"{QUALITY_GATES['max_median_proxy_error_years']}y"
            )

    und = scores.get("mean_understatement_fraction", np.nan)
    if not (und >= QUALITY_GATES["min_retrospective_understatement"]):
        fails.append(
            f"retrospective understatement {und:.3f} < "
            f"{QUALITY_GATES['min_retrospective_understatement']} -- the "
            "simulator has stopped reproducing the bias this study is about"
        )
    return fails
